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
from lib.conf.schema import ConfigurationScope, RootConfig, DataConfig, BinarizerConfig


dask.config.set(scheduler="synchronous")


def _load_and_log_config(config_path: pathlib.Path, overrides: list[str] = None) -> RootConfig:
    config = load_raw_config(config_path, overrides)
    config = RootConfig.model_validate(config, scope=ConfigurationScope.ACOUSTIC)
    config.resolve(ConfigurationScope.ACOUSTIC, "data")
    config.resolve(ConfigurationScope.ACOUSTIC, "binarizer")
    config.check(ConfigurationScope.ACOUSTIC, "data")
    config.check(ConfigurationScope.ACOUSTIC, "binarizer")
    formatter = ModelFormatter()
    print(formatter.format(config.data))
    print(formatter.format(config.binarizer))
    return config


def binarize_acoustic_datasets(data_config: DataConfig, binarizer_config: BinarizerConfig):
    from preprocessing.acoustic_binarizer_v3 import AcousticBinarizer
    binarizer = AcousticBinarizer(data_config, binarizer_config)
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
    config = _load_and_log_config(config, overrides=override)
    binarize_acoustic_datasets(config.data, config.binarizer)


if __name__ == "__main__":
    main()
