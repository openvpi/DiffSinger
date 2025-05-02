import os
import pathlib
import sys

root_dir = pathlib.Path(__file__).parent.parent.resolve()
os.environ["PYTHONPATH"] = str(root_dir)
sys.path.insert(0, str(root_dir))

import click
import dask

from lib.config.formatter import ModelFormatter
from lib.config.io import load_raw_config
from lib.config.schema import RootConfig, DataConfig, BinarizerConfig, ConfigurationScope

__all__ = [
    "binarize_datasets",
]

dask.config.set(scheduler="synchronous")


def _load_and_log_config(config_path: pathlib.Path, scope: int, overrides: list[str] = None) -> RootConfig:
    config = load_raw_config(config_path, overrides)
    config = RootConfig.model_validate(config, scope=scope)
    config.resolve(scope_mask=scope)
    config.check(scope_mask=scope)
    formatter = ModelFormatter()
    print(formatter.format(config.data))
    print(formatter.format(config.binarizer))
    return config


def binarize_datasets(
        binarizer_cls, data_config: DataConfig, binarizer_config: BinarizerConfig,
        coverage_check_option: str = "strict"
):
    binarizer = binarizer_cls(data_config, binarizer_config, coverage_check_option=coverage_check_option)
    print("| Binarizer: ", binarizer.__class__)
    binarizer.process()


@click.group(help="Binarize raw datasets.")
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
        "--coverage-check-option",
        type=click.Choice(["strict", "bypass", "compat"], case_sensitive=False),
        default="strict", required=False
    )(func)
    return func


@main.command(name="acoustic", help="Binarize raw acoustic datasets.")
@shared_options
def _binarize_acoustic_datasets_cli(config: pathlib.Path, override: list[str], coverage_check_option: str):
    config = _load_and_log_config(config, scope=ConfigurationScope.ACOUSTIC, overrides=override)
    from preprocessing.acoustic_binarizer import AcousticBinarizer
    binarize_datasets(
        AcousticBinarizer, config.data, config.binarizer,
        coverage_check_option=coverage_check_option
    )


@main.command(name="variance", help="Binarize raw variance datasets.")
@shared_options
def _binarize_variance_datasets_cli(config: pathlib.Path, override: list[str], coverage_check_option: str):
    config = _load_and_log_config(config, scope=ConfigurationScope.VARIANCE, overrides=override)
    from preprocessing.variance_binarizer import VarianceBinarizer
    binarize_datasets(
        VarianceBinarizer, config.data, config.binarizer,
        coverage_check_option=coverage_check_option
    )


@main.command(name="duration", help="Binarize raw duration datasets.")
@shared_options
def _binarize_duration_datasets_cli(config: pathlib.Path, override: list[str], coverage_check_option: str):
    config = _load_and_log_config(config, scope=ConfigurationScope.DURATION, overrides=override)
    from preprocessing.duration_binarizer import DurationBinarizer
    binarize_datasets(
        DurationBinarizer, config.data, config.binarizer,
        coverage_check_option=coverage_check_option
    )


if __name__ == "__main__":
    main()
