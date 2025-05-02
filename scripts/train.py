import os
import pathlib
import re
import shutil
import sys

root_dir = pathlib.Path(__file__).parent.parent.resolve()
os.environ["PYTHONPATH"] = str(root_dir)
sys.path.insert(0, str(root_dir))

import click
import yaml

from lib.config.formatter import ModelFormatter
from lib.config.io import load_raw_config
from lib.config.schema import (
    ConfigurationScope, RootConfig,
    PeriodicCheckpointConfig, ExpressionCheckpointConfig,
)

__all__ = [
    "find_latest_checkpoints",
    "train_model",
]


def _load_config(config_path: pathlib.Path, scope: int, overrides: list[str] = None) -> RootConfig:
    config = load_raw_config(config_path, overrides)
    config = RootConfig.model_validate(config, scope=scope)
    config.resolve(scope_mask=scope)
    config.check(scope_mask=scope)
    return config


def find_latest_checkpoints(
        ckpt_dir: pathlib.Path,
        candidate_tags: list[str] = None
) -> list[pathlib.Path]:
    candidates = []
    max_step = -1
    for ckpt in ckpt_dir.glob("model-*-steps=*-epochs=*.ckpt"):
        step = int(re.search(r"steps=(\d+)", ckpt.name).group(1))
        if step > max_step:
            max_step = step
            candidates = [ckpt]
        elif step == max_step:
            candidates.append(ckpt)
    for tag in candidate_tags or []:
        filtered_candidates = []
        for ckpt in candidates:
            ckpt_tag = re.search(r"model-(.*?)-steps=", ckpt.name).group(1)
            if tag == ckpt_tag:
                filtered_candidates.append(ckpt)
        if filtered_candidates:
            return filtered_candidates
    return candidates


def train_model(
        config: RootConfig, pl_module_cls,
        ckpt_save_dir: pathlib.Path,
        log_save_dir: pathlib.Path,
        resume_from: pathlib.Path = None
):
    import lightning.pytorch
    import lightning.pytorch.loggers
    from lightning_utilities.core.rank_zero import rank_zero_only, rank_zero_info
    from training.pl_module_base import BaseLightningModule

    from training.callbacks import PeriodicModelCheckpoint, ExpressionModelCheckpoint, FriendlyTQDMProgressBar
    from training.strategy import get_strategy

    if not issubclass(pl_module_cls, BaseLightningModule):
        raise ValueError("pl_module_cls must be a subclass of BaseLightningModule")

    @rank_zero_only
    def _payload_copy(from_dir: pathlib.Path, to_dir: pathlib.Path):
        shutil.copy(from_dir / "spk_map.json", to_dir)
        shutil.copy(from_dir / "lang_map.json", to_dir)
        shutil.copy(from_dir / "ph_map.json", to_dir)

    @rank_zero_only
    def _config_dump(cfg: RootConfig, to_dir: pathlib.Path):
        with open(to_dir / "config.yaml", "w", encoding="utf8") as f:
            yaml.safe_dump(cfg.model_dump(include={"model", "inference"}), f, allow_unicode=True, sort_keys=False)
        with open(to_dir / "hparams.yaml", "w", encoding="utf8") as f:
            yaml.safe_dump(cfg.model_dump(), f, allow_unicode=True, sort_keys=False)

    binary_data_dir = config.binarizer.binary_data_dir_resolved
    ckpt_save_dir.mkdir(parents=True, exist_ok=True)
    _payload_copy(binary_data_dir, ckpt_save_dir)
    _config_dump(config, ckpt_save_dir)
    model_config = config.model
    training_config = config.training

    pl_module = pl_module_cls(binary_data_dir, model_config, training_config)
    if resume_from is None and training_config.finetuning.pretraining_enabled:
        pl_module.load_from_pretrained_model(training_config.finetuning.pretraining_from)
    if resume_from is None:
        rank_zero_info(f"No checkpoint found or specified to resume from. Starting new training.")
    else:
        rank_zero_info(f"Resuming training from checkpoint: {resume_from}")

    if training_config.trainer.unit == "step":
        val_check_interval = (
                training_config.trainer.val_every_n_units * training_config.trainer.accumulate_grad_batches
        )
        check_val_every_n_epoch = None
    elif training_config.trainer.unit == "epoch":
        val_check_interval = None
        check_val_every_n_epoch = training_config.trainer.val_every_n_units
    else:
        raise ValueError(f"Unit must be 'step' or 'epoch', got '{training_config.trainer.unit}'.")
    callbacks = [
        FriendlyTQDMProgressBar()
    ]
    for ckpt_config in training_config.trainer.checkpoints:
        if ckpt_config.type == "periodic":
            ckpt_config: PeriodicCheckpointConfig
            checkpoint = PeriodicModelCheckpoint(
                dirpath=ckpt_save_dir,
                tag=ckpt_config.tag,
                unit=ckpt_config.unit,
                every_n_units=ckpt_config.every_n_units,
                since_m_units=ckpt_config.since_m_units,
                save_last_k=ckpt_config.save_last_k,
                save_weights_only=ckpt_config.weights_only,
            )
        elif ckpt_config.type == "expression":
            ckpt_config: ExpressionCheckpointConfig
            checkpoint = ExpressionModelCheckpoint(
                dirpath=ckpt_save_dir,
                tag=ckpt_config.tag,
                expression=ckpt_config.expression,
                mode=ckpt_config.mode,
                save_top_k=ckpt_config.save_top_k,
                save_weights_only=ckpt_config.weights_only,
            )
        else:
            raise ValueError(f"Invalid checkpoint monitor type: {ckpt_config.type}")
        callbacks.append(checkpoint)
    trainer = lightning.pytorch.Trainer(
        accelerator=training_config.trainer.accelerator,
        # TODO: strategy
        strategy=get_strategy(
            devices=training_config.trainer.devices,
            num_nodes=training_config.trainer.num_nodes,
            accelerator=training_config.trainer.accelerator,
            strategy={
                "name": training_config.trainer.strategy.name,
                **training_config.trainer.strategy.kwargs,
            },
            precision=training_config.trainer.precision,
        ),
        devices=training_config.trainer.devices,
        num_nodes=training_config.trainer.num_nodes,
        precision=training_config.trainer.precision,
        logger=lightning.pytorch.loggers.TensorBoardLogger(
            save_dir=log_save_dir,
            name="lightning_logs",
            version="latest",
        ),
        callbacks=callbacks,
        min_steps=training_config.trainer.min_steps,
        max_steps=training_config.trainer.max_steps,
        min_epochs=training_config.trainer.min_epochs,
        max_epochs=training_config.trainer.max_epochs,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=check_val_every_n_epoch,
        num_sanity_val_steps=training_config.trainer.num_sanity_val_steps,
        log_every_n_steps=1,
        accumulate_grad_batches=training_config.trainer.accumulate_grad_batches,
        gradient_clip_val=training_config.trainer.gradient_clip_val,
        use_distributed_sampler=False,
    )
    trainer.fit(model=pl_module, ckpt_path=resume_from)


def _get_config_scope(key: str) -> int:
    if key == "acoustic":
        return ConfigurationScope.ACOUSTIC
    elif key == "variance":
        return ConfigurationScope.VARIANCE
    elif key == "duration":
        return ConfigurationScope.DURATION
    else:
        raise ValueError(f"Invalid config scope key: {key}")


def _get_lightning_module_cls(key: str):
    if key == "acoustic":
        from training.acoustic_module import AcousticLightningModule
        return AcousticLightningModule
    elif key == "variance":
        from training.variance_module import VarianceLightningModule
        return VarianceLightningModule
    else:
        raise ValueError(f"Invalid lightning module key: {key}")


def _exec_training(
        recipe_key: str,
        config: pathlib.Path, override: list[str],
        exp_name: str, work_dir: pathlib.Path,
        log_dir: pathlib.Path,
        restart: bool,
        resume_from: pathlib.Path
):
    scope = _get_config_scope(recipe_key)
    config = _load_config(config, scope=scope, overrides=override)
    ckpt_save_dir = work_dir / exp_name
    if log_dir is None:
        log_save_dir = ckpt_save_dir
    else:
        log_save_dir = log_dir / exp_name
    if not restart and resume_from is None:
        latest_checkpoints = find_latest_checkpoints(ckpt_save_dir, candidate_tags=[
            ckpt_config.tag
            for ckpt_config in config.training.trainer.checkpoints
            if not ckpt_config.weights_only  # weights_only checkpoints cannot be resumed from
        ])
        if len(latest_checkpoints) > 1:
            raise ValueError(
                f"Cannot perform auto resuming because multiple latest checkpoints were found:\n"
                + "\n".join(f"  {ckpt}" for ckpt in latest_checkpoints)
                + "\nPlease manually choose a specific checkpoint using --resume-from."
            )
        elif len(latest_checkpoints) == 1:
            resume_from = latest_checkpoints[0]

    from lightning_utilities.core.rank_zero import rank_zero_only

    @rank_zero_only
    def log_config(cfg: RootConfig):
        formatter = ModelFormatter()
        print(formatter.format(cfg.model))
        print(formatter.format(cfg.training))

    pl_module_cls = _get_lightning_module_cls(recipe_key)
    log_config(config)
    train_model(
        config=config, pl_module_cls=pl_module_cls,
        ckpt_save_dir=ckpt_save_dir, log_save_dir=log_save_dir,
        resume_from=resume_from
    )


@click.group()
def main():
    pass


def shared_options(func):
    func = click.option(
        "--config", type=click.Path(
            exists=True, dir_okay=False, file_okay=True, readable=True, path_type=pathlib.Path
        ),
        required=True,
        help="Path to the configuration file."
    )(func)
    func = click.option(
        "--override", multiple=True,
        type=click.STRING, required=False,
        help="Override configuration values in dotlist format."
    )(func)
    func = click.option(
        "--work-dir", type=click.Path(
            dir_okay=True, file_okay=False, path_type=pathlib.Path
        ),
        required=False, default=pathlib.Path(__file__).parent.parent / "experiments",
        show_default=True,
        help="Path to the working directory. The experiment subdirectory will be created here."
    )(func)
    func = click.option(
        "--exp-name", type=click.STRING,
        required=True,
        help="Experiment name. Checkpoints will be saved in subdirectory with this name."
    )(func)
    func = click.option(
        "--log-dir", type=click.Path(
            dir_okay=True, file_okay=False, path_type=pathlib.Path
        ),
        required=False,
        help="Directory to save logs. If not specified, logs will be saved in the checkpoints directory."
    )(func)
    func = click.option(
        "--restart", is_flag=True, default=False,
        help="Ignore existing checkpoints and start new training."
    )(func)
    func = click.option(
        "--resume-from", type=click.Path(
            exists=True, dir_okay=False, file_okay=True, readable=True, path_type=pathlib.Path
        ),
        required=False,
        help="Resume training from this specific checkpoint."
    )(func)
    return func


@main.command(name="acoustic", help="Train acoustic model.")
@shared_options
def _train_acoustic_model_cli(
        config: pathlib.Path, override: list[str],
        exp_name: str, work_dir: pathlib.Path,
        log_dir: pathlib.Path,
        restart: bool,
        resume_from: pathlib.Path,
):
    _exec_training(
        recipe_key="acoustic",
        config=config, override=override,
        exp_name=exp_name, work_dir=work_dir,
        log_dir=log_dir,
        restart=restart, resume_from=resume_from,
    )


@main.command(name="variance", help="Train variance model.")
@shared_options
def _train_variance_model_cli(
        config: pathlib.Path, override: list[str],
        exp_name: str, work_dir: pathlib.Path,
        log_dir: pathlib.Path,
        restart: bool,
        resume_from: pathlib.Path,
):
    _exec_training(
        recipe_key="variance",
        config=config, override=override,
        exp_name=exp_name, work_dir=work_dir,
        log_dir=log_dir,
        restart=restart, resume_from=resume_from,
    )


if __name__ == '__main__':
    main()
