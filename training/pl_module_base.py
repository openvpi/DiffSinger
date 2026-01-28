import abc
import pathlib
from fnmatch import fnmatch
from typing import Any

import lightning.pytorch
import matplotlib
import torch
import tqdm
from lightning_utilities.core.rank_zero import rank_zero_info
from torch import nn
from torchmetrics import Metric, MeanMetric

from lib import logging
from lib.config.schema import ModelConfig, TrainingConfig
from lib.reflection import build_optimizer_from_config, build_lr_scheduler_from_config
from .data import BaseDataset, DynamicBatchSampler
from .weight_averaging import ExponentialMovingAverage

__all__ = [
    "BaseLightningModule",
]
matplotlib.use("Agg")  # fix Tcl_AsyncDelete: async handler deleted by the wrong thread


class BaseLightningModule(lightning.pytorch.LightningModule, abc.ABC):
    def __init__(
            self,
            binary_data_dir: pathlib.Path,
            model_config: ModelConfig,
            training_config: TrainingConfig,
            load_pretrained: bool = False,
    ):
        super().__init__()

        self.binary_data_dir = binary_data_dir
        self.model_config = model_config
        self.training_config = training_config

        self.model: nn.Module = self.build_model()
        self.losses = nn.ModuleDict()
        self.metrics = nn.ModuleDict()
        self.val_losses: dict[str, Metric] = {  # use built-in dict to not be printed in the model summary
            "total_loss": MeanMetric()
        }
        self.register_losses_and_metrics()
        if len(self.losses) == 0:
            raise ValueError("No losses defined.")
        self.freeze_parameters()  # caution: this can break when resuming training
        self.print_arch()

        self.use_ema = self.training_config.weight_averaging.ema_enabled
        if self.use_ema:
            self.ema: ExponentialMovingAverage = self.build_ema()
        if load_pretrained and self.training_config.finetuning.pretraining_enabled:
            self.load_from_pretrained_model(training_config.finetuning.pretraining_from)

        self.train_dataset: BaseDataset = None
        self.valid_dataset: BaseDataset = None
        self.train_sampler: DynamicBatchSampler = None
        self.valid_sampler: DynamicBatchSampler = None

        self.logger_step = -1  # when accumulate_grad_batches > 1, this helps to avoid redundant logging

        self.post_init()

    @abc.abstractmethod
    def build_model(self) -> nn.Module:
        """Build and return the model."""
        pass

    @abc.abstractmethod
    def register_losses_and_metrics(self) -> None:
        """
        Register the losses and metrics.
        Use `self.register_loss(name, loss)` to register a loss,
        and `self.register_metric(name, metric)` to register a metric.
        Registered losses and metrics can be accessed via `self.losses` and `self.metrics`.
        """
        pass

    def post_init(self) -> None:
        """This method is called after the model is initialized, useful for custom initialization logic."""
        pass

    @abc.abstractmethod
    def forward_model(self, sample: dict[str, torch.Tensor], infer: bool) -> dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        :param sample: the training or validation batch.
        :param infer: whether in inference mode.
        :return: if `infer` is True, update all registered metrics and return the model outputs;
            otherwise, return a dictionary containing values of all registered losses.
        """
        pass

    @abc.abstractmethod
    def plot_validation_results(self, sample: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]) -> None:
        """
        Plot the validation results on external logger like TensorBoard.
        :param sample: the validation batch.
        :param outputs: the model outputs returned by `self.forward_model(sample, infer=True)` directly.
        """
        pass

    def print_arch(self):
        rank_zero_info(f"Model: {self.model}")
        rank_zero_info(f"Losses: {self.losses}")
        rank_zero_info(f"Metrics: {self.metrics}")

    def freeze_parameters(self):
        if not self.training_config.finetuning.freezing_enabled:
            return
        param_dict = dict(self.named_parameters())
        size = len(param_dict)
        _apply_include_exclude(
            param_dict,
            includes=self.training_config.finetuning.freezing_include_params,
            excludes=self.training_config.finetuning.freezing_exclude_params
        )
        if len(param_dict) == size:
            raise ValueError(
                "Freezing all parameters is not allowed."
            )
        for param in param_dict.values():
            param.requires_grad = False
        logging.info(f"Freezing {len(param_dict)} parameter(s).", callback=rank_zero_info)

    def build_ema(self) -> ExponentialMovingAverage:
        parameters = dict(self.named_parameters())
        _apply_include_exclude(
            parameters,
            includes=self.training_config.weight_averaging.ema_include_params,
            excludes=self.training_config.weight_averaging.ema_exclude_params
        )
        ema = ExponentialMovingAverage(
            parameters=parameters,
            decay=self.training_config.weight_averaging.ema_decay
        )
        logging.info(f"EMA: {ema.size()} parameter(s) registered.", callback=rank_zero_info)
        return ema

    def load_from_pretrained_model(self, pretrained_model_path: pathlib.Path) -> None:
        ckpt = torch.load(
            pretrained_model_path, map_location=self.device, weights_only=True
        )
        source_state_dict = ckpt["state_dict"]
        _apply_include_exclude(
            source_state_dict,
            includes=self.training_config.finetuning.pretraining_include_params,
            excludes=self.training_config.finetuning.pretraining_exclude_params
        )
        target_state_dict = self.state_dict()
        for name in list(source_state_dict.keys()):
            if name not in target_state_dict:
                del source_state_dict[name]
        _check_shape_consistency(
            source_state_dict, target_state_dict,
            error_message=f"Pretrained model '{pretrained_model_path}' has mismatched parameter(s)"
        )
        self.load_state_dict(source_state_dict, strict=False)
        logging.info(
            f"Loaded {len(source_state_dict)} parameter(s) from '{pretrained_model_path}'",
            callback=rank_zero_info
        )
        if self.use_ema:
            self.ema.register()  # copy to shadow again after updating referenced parameters
            source_ema_state_dict = ckpt.get("ema_state_dict", {})
            target_ema_state_dict = self.ema.state_dict()
            for name in list(source_ema_state_dict.keys()):
                if name not in source_state_dict or name not in target_ema_state_dict:
                    del source_ema_state_dict[name]
            _check_shape_consistency(
                source_ema_state_dict, target_ema_state_dict,
                error_message=f"Pretrained model '{pretrained_model_path}' has mismatched EMA parameter(s)"
            )
            self.ema.load_state_dict(source_ema_state_dict, strict=False)
            logging.info(
                f"Loaded {len(source_ema_state_dict)} EMA parameter(s) from '{pretrained_model_path}'",
                callback=rank_zero_info
            )

    def register_loss(self, name: str, loss: nn.Module) -> None:
        """
        Register a loss module that can be accessed via `self.losses`.
        """
        if name in self.losses:
            raise ValueError(f"Loss '{name}' already registered.")
        if name in self.metrics:
            raise ValueError(f"Loss name '{name}' is already used by a metric.")
        self.losses[name] = loss
        self.val_losses[name] = MeanMetric()  # for validation logging

    def register_metric(self, name: str, metric: Metric) -> None:
        """
        Register a metric that can be accessed via `self.metrics`.
        """
        if name in self.metrics:
            raise ValueError(f"Metric '{name}' already registered.")
        if name in self.losses:
            raise ValueError(f"Metric name {name} is already used by a loss.")
        self.metrics[name] = metric

    def build_train_dataset(self) -> BaseDataset:
        """
        Build the training dataset.
        """
        return BaseDataset(self.binary_data_dir, "train")

    def build_valid_dataset(self) -> BaseDataset:
        """
        Build the validation dataset.
        """
        return BaseDataset(self.binary_data_dir, "valid")

    def setup(self, stage: str) -> None:
        if stage != "fit":
            raise ValueError("This module only supports the 'fit' stage.")
        self.train_dataset = self.build_train_dataset()
        self.valid_dataset = self.build_valid_dataset()

    def train_dataloader(self):
        dataloader_config = self.training_config.dataloader
        self.train_sampler = DynamicBatchSampler(
            self.train_dataset,
            max_batch_size=dataloader_config.max_batch_size,
            max_batch_frames=dataloader_config.max_batch_frames,
            sort_by_len=True,
            frame_count_grid=dataloader_config.frame_count_grid,
            batch_count_multiple_of=self.training_config.trainer.accumulate_grad_batches,
            reassign_batches=True,
            shuffle_batches=False,
            seed=42,
        )
        return torch.utils.data.DataLoader(
            self.train_dataset,
            collate_fn=self.train_dataset.collate,
            batch_sampler=self.train_sampler,
            num_workers=dataloader_config.num_workers,
            prefetch_factor=dataloader_config.prefetch_factor if dataloader_config.num_workers > 0 else None,
            pin_memory=True,
            persistent_workers=dataloader_config.num_workers > 0
        )

    def val_dataloader(self):
        dataloader_config = self.training_config.dataloader
        return torch.utils.data.DataLoader(
            self.valid_dataset,
            collate_fn=self.valid_dataset.collate,
            batch_sampler=DynamicBatchSampler(
                self.valid_dataset,
                max_batch_size=dataloader_config.max_val_batch_size,
                max_batch_frames=dataloader_config.max_val_batch_frames,
                sort_by_len=False,
                reassign_batches=False,
                shuffle_batches=False,
            ),
            num_workers=dataloader_config.num_workers,
            prefetch_factor=dataloader_config.prefetch_factor if dataloader_config.num_workers > 0 else None,
            persistent_workers=dataloader_config.num_workers > 0
        )

    def configure_optimizers(self):
        optimizer = build_optimizer_from_config(self, self.training_config.optimizer)
        scheduler = build_lr_scheduler_from_config(optimizer, self.training_config.lr_scheduler)
        interval = self.training_config.lr_scheduler.unit
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if interval != self.training_config.trainer.unit:
                # ReduceLROnPlateau requires the scheduler to synchronize with the validation period
                raise ValueError(
                    f"ReduceLROnPlateau scheduler requires training.lr_scheduler.unit and training.trainer.unit "
                    f"to be the same, got '{interval}' and '{self.training_config.trainer.unit}'."
                )
            # Call scheduler.step() after each validation
            frequency = self.training_config.trainer.val_every_n_units
            monitor = self.training_config.lr_scheduler.monitor
            if monitor not in self.val_losses and monitor not in self.metrics:
                raise ValueError(
                    f"Invalid monitor '{monitor}' for ReduceLROnPlateau scheduler. Should be one of "
                    f"losses {list(self.val_losses.keys())} or metrics {list(self.metrics.keys())}."
                )
        else:
            frequency = 1
            monitor = None
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": interval,
                "frequency": frequency,
                "monitor": monitor,
                "strict": False,  # in case the candidates are empty after resuming
            }
        }

    def on_train_epoch_start(self):
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self.current_epoch)

    def training_step(self, sample: dict[str, torch.Tensor], batch_index: int):
        losses = self.forward_model(sample, infer=False)
        total_loss = sum(losses.values())
        log_outputs = {
            **losses,
            "batch_size": sample["size"],
        }
        # logs to progress bar
        self.log_dict(log_outputs, prog_bar=True, logger=False, on_step=True, on_epoch=False)
        self.log("lr", self.lr_schedulers().get_last_lr()[0], prog_bar=True, logger=False, on_step=True, on_epoch=False)
        # logs to tensorboard
        if (
                self.global_step > self.logger_step and
                self.global_step % self.training_config.trainer.log_every_n_steps == 0
        ):
            tb_log = {f"training/{k}": v for k, v in log_outputs.items()}
            tb_log["training/lr"] = self.lr_schedulers().get_last_lr()[0]
            tb_log["training/epoch"] = self.current_epoch
            self.logger.log_metrics(tb_log, step=self.global_step)
            self.logger_step = self.global_step
        return total_loss

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.use_ema:
            self.ema.step()  # update EMA parameters after optimizer step

    def on_validation_epoch_start(self):
        for metric in self.val_losses.values():
            # self.val_losses is a built-in dict, so we need to move them to device manually
            metric.to(self.device)
            metric.reset()
        for metric in self.metrics.values():
            metric.reset()
        if self.use_ema:
            self.ema.apply()  # switch to EMA parameters for validation

    def validation_step(self, sample: dict[str, torch.Tensor], batch_index: int):
        if sample["size"] == 0:
            return
        save_obj = {}
        with torch.autocast(self.device.type, enabled=False):
            losses = self.forward_model(sample, infer=False)
            if min(sample["indices"]) < self.training_config.validation.max_plots:
                outputs = self.forward_model(sample, infer=True)
                save_obj["sample"] = sample
                save_obj["outputs"] = outputs
                filename = f"validation_step{self.global_step}_rank{self.global_rank}_batch{batch_index}.pt"
                torch.save(
                    obj=save_obj,
                    f=pathlib.Path(self.logger.log_dir) / filename,
                )
            losses = {
                "total_loss": sum(losses.values()),
                **losses,
            }
            for k, v in losses.items():
                self.val_losses[k].update(v, weight=sample["size"])

    def on_validation_epoch_end(self):
        loss_vals = {k: v.compute() for k, v in self.val_losses.items()}
        metric_vals = {}
        for k, v in self.metrics.items():
            m = v.compute()
            if isinstance(m, dict):
                metric_vals.update(m)
            else:
                metric_vals[k] = m
        self.log_dict(
            {**loss_vals, **metric_vals},
            on_epoch=True, prog_bar=False, logger=False, sync_dist=True
        )
        if self.global_rank != 0:
            return
        self.logger.log_metrics({f"validation/{k}": v for k, v in loss_vals.items()}, step=self.global_step)
        self.logger.log_metrics({f"validation/{k}": v for k, v in metric_vals.items()}, step=self.global_step)
        filelist = list(pathlib.Path(self.logger.log_dir).glob(f"validation_step{self.global_step}_rank*_batch*.pt"))
        with torch.autocast(self.device.type, enabled=False):
            for file in tqdm.tqdm(filelist, desc="Plotting", leave=False):
                obj = torch.load(file, map_location=self.device, weights_only=True)
                sample = obj["sample"]
                outputs = obj["outputs"]
                self.plot_validation_results(sample, outputs)
                file.unlink()
        if self.use_ema:
            self.ema.restore()  # restore original parameters after validation

    def on_save_checkpoint(self, checkpoint: dict[str, torch.Tensor]):
        if self.use_ema:
            checkpoint["ema_state_dict"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint: dict[str, torch.Tensor]):
        if self.use_ema:
            self.ema.load_state_dict(checkpoint.pop("ema_state_dict"), strict=True)


def _apply_include_exclude(dict_to_filter: dict[str, Any], includes: list[str] = None, excludes: list[str] = None):
    for key in list(dict_to_filter.keys()):
        if includes and not any(fnmatch(key, pattern) for pattern in includes):
            del dict_to_filter[key]
        elif excludes and any(fnmatch(key, pattern) for pattern in excludes):
            del dict_to_filter[key]


def _check_shape_consistency(
        source_state_dict: dict[str, torch.Tensor],
        target_state_dict: dict[str, torch.Tensor],
        error_message: str
):
    errors = []
    for name in list(source_state_dict.keys()):
        source_param = source_state_dict[name]
        target_param = target_state_dict[name]
        if source_param.shape != target_param.shape:
            errors.append((name, tuple(source_param.shape), tuple(target_param.shape)))
    if errors:
        raise RuntimeError(
            f"{error_message}:\n"
            + "\n".join(
                f"  {name}: source {source_shape}, target {target_shape}"
                for name, source_shape, target_shape in errors
            )
        )
