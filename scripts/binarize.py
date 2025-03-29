import importlib
import os
import sys
from pathlib import Path

import click

root_dir = Path(__file__).parent.parent.resolve()
os.environ['PYTHONPATH'] = str(root_dir)
sys.path.insert(0, str(root_dir))

from utils.config_utils import read_full_config, print_config


@click.command(help="Process the raw dataset into binary format")
@click.option(
    "--config", 'config_path',
    type=click.Path(file_okay=True, dir_okay=False, resolve_path=True, path_type=Path),
    metavar='FILE',
    help='Path to the configuration file'
)
def binarize(config_path: Path):
    config, config_chain = read_full_config(config_path=config_path)
    print_config(config, config_chain)
    binarizer_cls = config["binarizer_cls"]
    pkg = ".".join(binarizer_cls.split(".")[:-1])
    cls_name = binarizer_cls.split(".")[-1]
    binarizer_cls = getattr(importlib.import_module(pkg), cls_name)
    print("| Binarizer: ", binarizer_cls)
    binarizer_cls(config).process()


if __name__ == '__main__':
    binarize()
