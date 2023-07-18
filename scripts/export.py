import os
import re
import sys
from pathlib import Path
from typing import List

import click
import torch

root_dir = Path(__file__).parent.parent.resolve()
os.environ['PYTHONPATH'] = str(root_dir)
sys.path.insert(0, str(root_dir))

from utils.hparams import set_hparams, hparams


def find_exp(exp):
    if not (root_dir / 'checkpoints' / exp).exists():
        for subdir in (root_dir / 'checkpoints').iterdir():
            if not subdir.is_dir():
                continue
            if subdir.name.startswith(exp):
                print(f'| match ckpt by prefix: {subdir.name}')
                exp = subdir.name
                break
        else:
            assert False, \
                f'There are no matching exp starting with \'{exp}\' in \'checkpoints\' folder. ' \
                'Please specify \'--exp\' as the folder name or prefix.'
    else:
        print(f'| found ckpt by name: {exp}')
    return exp


def parse_spk_settings(export_spk, freeze_spk):
    if export_spk is None:
        export_spk = []
    else:
        export_spk = list(export_spk)
    from utils.infer_utils import parse_commandline_spk_mix
    spk_name_pattern = r'[0-9A-Za-z_-]+'
    export_spk_mix = []
    for spk in export_spk:
        assert '=' in spk or '|' not in spk, \
            'You must specify an alias with \'NAME=\' for each speaker mix.'
        if '=' in spk:
            alias, mix = spk.split('=', maxsplit=1)
            assert re.fullmatch(spk_name_pattern, alias) is not None, f'Invalid alias \'{alias}\' for speaker mix.'
            export_spk_mix.append((alias, parse_commandline_spk_mix(mix)))
        else:
            export_spk_mix.append((spk, {spk: 1.0}))
    freeze_spk_mix = None
    if freeze_spk is not None:
        assert '=' in freeze_spk or '|' not in freeze_spk, \
            'You must specify an alias with \'NAME=\' for each speaker mix.'
        if '=' in freeze_spk:
            alias, mix = freeze_spk.split('=', maxsplit=1)
            assert re.fullmatch(spk_name_pattern, alias) is not None, f'Invalid alias \'{alias}\' for speaker mix.'
            freeze_spk_mix = (alias, parse_commandline_spk_mix(mix))
        else:
            freeze_spk_mix = (freeze_spk, {freeze_spk: 1.0})
    return export_spk_mix, freeze_spk_mix


@click.group()
def main():
    pass


@main.command(help='Export DiffSinger acoustic model to ONNX format.')
@click.option('--exp', type=str, required=True, metavar='<exp>', help='Choose an experiment to export.')
@click.option('--ckpt', type=int, required=False, metavar='<steps>', help='Checkpoint training steps.')
@click.option('--out', type=str, required=False, metavar='<dir>', help='Output directory for the artifacts.')
@click.option('--expose_gender', is_flag=True, show_default=True,
              help='(for random pitch shifting) Expose gender control functionality.')
@click.option('--freeze_gender', type=float, default=0., show_default=True, metavar='<value>',
              help='(for random pitch shifting) Freeze gender value into the model.')
@click.option('--expose_velocity', is_flag=True, show_default=True,
              help='(for random time stretching) Expose velocity control functionality.')
@click.option('--export_spk', type=str, required=False, multiple=True, metavar='<mix>',
              help='(for multi-speaker models) Export one or more speaker or speaker mix keys.')
@click.option('--freeze_spk', type=str, required=False, metavar='<mix>',
              help='(for multi-speaker models) Freeze one speaker or speaker mix into the model.')
def acoustic(
        exp: str,
        ckpt: int = None,
        out: str = None,
        expose_gender: bool = False,
        freeze_gender: float = 0.,
        expose_velocity: bool = False,
        export_spk: List[str] = None,
        freeze_spk: str = None
):
    # Validate arguments
    if expose_gender and freeze_gender:
        print('--expose_gender is exclusive to --freeze_gender.')
        exit(-1)
    if export_spk and freeze_spk:
        print('--export_spk is exclusive to --freeze_spk.')
        exit(-1)
    if freeze_gender is not None:
        assert -1. <= freeze_gender <= 1., 'Frozen gender must be in [-1, 1].'
    exp = find_exp(exp)
    if out is None:
        out = root_dir / 'artifacts' / exp
    else:
        out = Path(out)
    out = out.resolve()
    export_spk_mix, freeze_spk_mix = parse_spk_settings(export_spk, freeze_spk)

    # Load configurations
    sys.argv = [
        sys.argv[0],
        '--exp_name',
        exp,
        '--infer'
    ]
    set_hparams()

    # Export artifacts
    from deployment.exporters import DiffSingerAcousticExporter
    print(f'| Exporter: {DiffSingerAcousticExporter}')
    exporter = DiffSingerAcousticExporter(
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        cache_dir=root_dir / 'deployment' / 'cache',
        ckpt_steps=ckpt,
        expose_gender=expose_gender,
        freeze_gender=freeze_gender,
        expose_velocity=expose_velocity,
        export_spk=export_spk_mix,
        freeze_spk=freeze_spk_mix
    )
    try:
        exporter.export(out)
    except KeyboardInterrupt:
        exit(-1)


@main.command(help='Export DiffSinger variance model to ONNX format.')
@click.option('--exp', type=str, required=True, metavar='<exp>', help='Choose an experiment to export.')
@click.option('--ckpt', type=int, required=False, metavar='<steps>', help='Checkpoint training steps.')
@click.option('--out', type=str, required=False, metavar='<dir>', help='Output directory for the artifacts.')
@click.option('--export_spk', type=str, required=False, multiple=True, metavar='<mix>',
              help='(for multi-speaker models) Export one or more speaker or speaker mix keys.')
@click.option('--freeze_spk', type=str, required=False, metavar='<mix>',
              help='(for multi-speaker models) Freeze one speaker or speaker mix into the model.')
def variance(
        exp: str,
        ckpt: int = None,
        out: str = None,
        export_spk: List[str] = None,
        freeze_spk: str = None
):
    # Validate arguments
    if export_spk and freeze_spk:
        print('--export_spk is exclusive to --freeze_spk.')
        exit(-1)
    exp = find_exp(exp)
    if out is None:
        out = root_dir / 'artifacts' / exp
    else:
        out = Path(out)
    out = out.resolve()
    export_spk_mix, freeze_spk_mix = parse_spk_settings(export_spk, freeze_spk)

    # Load configurations
    sys.argv = [
        sys.argv[0],
        '--exp_name',
        exp,
        '--infer'
    ]
    set_hparams()
    from deployment.exporters import DiffSingerVarianceExporter
    print(f'| Exporter: {DiffSingerVarianceExporter}')
    exporter = DiffSingerVarianceExporter(
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        cache_dir=root_dir / 'deployment' / 'cache',
        ckpt_steps=ckpt,
        export_spk=export_spk_mix,
        freeze_spk=freeze_spk_mix
    )
    try:
        exporter.export(out)
    except KeyboardInterrupt:
        exit(-1)


@main.command(help='Export NSF-HiFiGAN vocoder model to ONNX format.')
@click.option('--config', type=str, required=True, metavar='<path>', help='Specify a config path of the vocoder.')
@click.option('--out', type=str, required=False, metavar='<dir>', help='Output directory for the artifacts.')
@click.option('--name', type=str, required=False, metavar='<name>', default='nsf_hifigan', show_default=False,
              help='Specify filename (without suffix) of the target model file.')
def nsf_hifigan(
        config: str,
        out: str = None,
        name: str = None
):
    # Check arguments
    if not Path(config).resolve().exists():
        raise FileNotFoundError(f'{config} is not a valid config path.')
    if out is None:
        out = root_dir / 'artifacts' / 'nsf_hifigan'
    else:
        out = Path(out)
    out = out.resolve()

    # Load configurations
    set_hparams(config)

    # Export artifacts
    from deployment.exporters import NSFHiFiGANExporter
    print(f'| Exporter: {NSFHiFiGANExporter}')
    exporter = NSFHiFiGANExporter(
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        cache_dir=root_dir / 'deployment' / 'cache',
        model_path=Path(hparams['vocoder_ckpt']).resolve(),
        model_name=name
    )
    try:
        exporter.export(out)
    except KeyboardInterrupt:
        exit(-1)


if __name__ == '__main__':
    main()
