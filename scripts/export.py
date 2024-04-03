import os
import pathlib
import re
import sys
from typing import List

import click
import torch

root_dir = pathlib.Path(__file__).resolve().parent.parent
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
            raise click.BadParameter(
                f'There are no matching exp starting with \'{exp}\' in \'checkpoints\' folder. '
                'Please specify \'--exp\' as the folder name or prefix.'
            )
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
@click.option(
    '--exp', type=click.STRING,
    required=True, metavar='EXP', callback=lambda ctx, param, value: find_exp(value),
    help='Choose an experiment to export.'
)
@click.option(
    '--ckpt', type=click.IntRange(min=0),
    required=False, metavar='STEPS',
    help='Checkpoint training steps.'
)
@click.option(
    '--out', type=click.Path(
        dir_okay=True, file_okay=False,
        path_type=pathlib.Path, resolve_path=True
    ),
    required=False,
    help='Output directory for the artifacts.'
)
@click.option(
    '--freeze_gender', type=click.FloatRange(min=-1, max=1),
    help='(for random pitch shifting) Freeze gender value into the model.'
)
@click.option(
    '--freeze_velocity', is_flag=True,
    help='(for random time stretching) Freeze default velocity value into the model.'
)
@click.option(
    '--export_spk', type=click.STRING,
    required=False, multiple=True,
    help='(for multi-speaker models) Export one or more speaker or speaker mixture keys.'
)
@click.option(
    '--freeze_spk', type=click.STRING,
    required=False,
    help='(for multi-speaker models) Freeze one speaker or speaker mixture into the model.'
)
def acoustic(
        exp: str,
        ckpt: int = None,
        out: pathlib.Path = None,
        freeze_gender: float = 0.,
        freeze_velocity: bool = False,
        export_spk: List[str] = None,
        freeze_spk: str = None
):
    # Validate arguments
    if export_spk and freeze_spk:
        print('--export_spk is exclusive to --freeze_spk.')
        exit(-1)
    if out is None:
        out = root_dir / 'artifacts' / exp
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
        freeze_gender=freeze_gender,
        freeze_velocity=freeze_velocity,
        export_spk=export_spk_mix,
        freeze_spk=freeze_spk_mix
    )
    try:
        exporter.export(out)
    except KeyboardInterrupt:
        exit(-1)


@main.command(help='Export DiffSinger variance model to ONNX format.')
@click.option(
    '--exp', type=click.STRING,
    required=True, metavar='EXP', callback=lambda ctx, param, value: find_exp(value),
    help='Choose an experiment to export.'
)
@click.option(
    '--ckpt', type=click.IntRange(min=0),
    required=False, metavar='STEPS',
    help='Checkpoint training steps.'
)
@click.option(
    '--out', type=click.Path(
        dir_okay=True, file_okay=False,
        path_type=pathlib.Path, resolve_path=True
    ),
    required=False,
    help='Output directory for the artifacts.'
)
@click.option(
    '--freeze_glide', is_flag=True,
    help='Freeze default glide embedding into the model.'
)
@click.option(
    '--freeze_expr', is_flag=True,
    help='Freeze default pitch expressiveness factor into the model.'
)
@click.option(
    '--export_spk', type=click.STRING,
    required=False, multiple=True,
    help='(for multi-speaker models) Export one or more speaker or speaker mixture keys.'
)
@click.option(
    '--freeze_spk', type=click.STRING,
    required=False,
    help='(for multi-speaker models) Freeze one speaker or speaker mixture into the model.'
)
def variance(
        exp: str,
        ckpt: int = None,
        out: str = None,
        freeze_glide: bool = False,
        freeze_expr: bool = False,
        export_spk: List[str] = None,
        freeze_spk: str = None
):
    # Validate arguments
    if export_spk and freeze_spk:
        print('--export_spk is exclusive to --freeze_spk.')
        exit(-1)
    if out is None:
        out = root_dir / 'artifacts' / exp
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
        freeze_glide=freeze_glide,
        freeze_expr=freeze_expr,
        export_spk=export_spk_mix,
        freeze_spk=freeze_spk_mix
    )
    try:
        exporter.export(out)
    except KeyboardInterrupt:
        exit(-1)


@main.command(help='Export NSF-HiFiGAN vocoder model to ONNX format.')
@click.option(
    '--config', type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True
    ),
    required=True,
    help='Specify a configuration file for the vocoder.'
)
@click.option(
    '--ckpt', type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True
    ),
    required=False,
    help='Specify a model path of the vocoder checkpoint.'
)
@click.option(
    '--out', type=click.Path(
        dir_okay=True, file_okay=False,
        path_type=pathlib.Path, resolve_path=True
    ),
    required=False,
    help='Output directory for the artifacts.'
)
@click.option(
    '--name', type=click.STRING,
    required=False, default='nsf_hifigan', show_default=False,
    help='Specify filename (without suffix) of the target model file.'
)
def nsf_hifigan(
        config: pathlib.Path,
        ckpt: pathlib.Path = None,
        out: pathlib.Path = None,
        name: str = None
):
    # Check arguments
    if out is None:
        out = root_dir / 'artifacts' / 'nsf_hifigan'

    # Load configurations
    set_hparams(config)
    if ckpt is None:
        model_path = pathlib.Path(hparams['vocoder_ckpt']).resolve()
    else:
        model_path = ckpt

    # Export artifacts
    from deployment.exporters import NSFHiFiGANExporter
    print(f'| Exporter: {NSFHiFiGANExporter}')
    exporter = NSFHiFiGANExporter(
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        cache_dir=root_dir / 'deployment' / 'cache',
        model_path=model_path,
        model_name=name
    )
    try:
        exporter.export(out)
    except KeyboardInterrupt:
        exit(-1)


if __name__ == '__main__':
    main()
