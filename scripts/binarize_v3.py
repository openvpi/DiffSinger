import os
import pathlib
import sys

root_dir = pathlib.Path(__file__).parent.parent.resolve()
os.environ["PYTHONPATH"] = str(root_dir)
sys.path.insert(0, str(root_dir))

import click
import dask

from lib.conf.formatter import ModelFormatter
from lib.conf.io import load_raw_config
from lib.conf.schema import RootConfig, DataConfig, BinarizerConfig, ConfigurationScope

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


def binarize_acoustic_datasets(data_config: DataConfig, binarizer_config: BinarizerConfig):
    from preprocessing.acoustic_binarizer import AcousticBinarizer
    binarizer = AcousticBinarizer(data_config, binarizer_config)
    print("| Binarizer: ", binarizer.__class__)
    binarizer.process()


def binarize_variance_datasets(data_config: DataConfig, binarizer_config: BinarizerConfig):
    from preprocessing.variance_binarizer import VarianceBinarizer
    binarizer = VarianceBinarizer(data_config, binarizer_config)
    print("| Binarizer: ", binarizer.__class__)
    binarizer.process()


def binarize_duration_datasets(data_config: DataConfig, binarizer_config: BinarizerConfig):
    from preprocessing.duration_binarizer import DurationBinarizer
    binarizer = DurationBinarizer(data_config, binarizer_config)
    print("| Binarizer: ", binarizer.__class__)
    binarizer.process()


@click.group(help="Binarize raw datasets.")
def main():
    pass


@main.command(name="acoustic", help="Binarize raw acoustic datasets.")
@click.option(
    "--config", type=click.Path(
        exists=True, dir_okay=False, file_okay=True, readable=True, path_type=pathlib.Path
    ),
    required=True,
    help="Path to the configuration file."
)
@click.option(
    "--override", multiple=True,
    type=str, required=False,
    help="Override configuration values in dotlist format."
)
def _binarize_acoustic_datasets_cli(config: pathlib.Path, override: list[str]):
    config = _load_and_log_config(config, scope=ConfigurationScope.ACOUSTIC, overrides=override)
    binarize_acoustic_datasets(config.data, config.binarizer)


@main.command(name="variance", help="Binarize raw variance datasets.")
@click.option(
    "--config", type=click.Path(
        exists=True, dir_okay=False, file_okay=True, readable=True, path_type=pathlib.Path
    ),
    required=True,
    help="Path to the configuration file."
)
@click.option(
    "--override", multiple=True,
    type=str, required=False,
    help="Override configuration values in dotlist format."
)
def _binarize_variance_datasets_cli(config: pathlib.Path, override: list[str]):
    config = _load_and_log_config(config, scope=ConfigurationScope.VARIANCE, overrides=override)
    binarize_variance_datasets(config.data, config.binarizer)


@main.command(name="duration", help="Binarize raw duration datasets.")
@click.option(
    "--config", type=click.Path(
        exists=True, dir_okay=False, file_okay=True, readable=True, path_type=pathlib.Path
    ),
    required=True,
    help="Path to the configuration file."
)
@click.option(
    "--override", multiple=True,
    type=str, required=False,
    help="Override configuration values in dotlist format."
)
def _binarize_duration_datasets_cli(config: pathlib.Path, override: list[str]):
    config = _load_and_log_config(config, scope=ConfigurationScope.DURATION, overrides=override)
    binarize_duration_datasets(config.data, config.binarizer)


if __name__ == "__main__":
    main()
