import json
import os
import pathlib
import sys
from collections import OrderedDict
from pathlib import Path

import click
from typing import Tuple

root_dir = Path(__file__).resolve().parent.parent
os.environ['PYTHONPATH'] = str(root_dir)
sys.path.insert(0, str(root_dir))


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


@click.group()
def main():
    pass


@main.command(help='Run DiffSinger acoustic model inference')
@click.argument(
    'proj', type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True,
        path_type=pathlib.Path, resolve_path=True
    ),
    metavar='DS_FILE'
)
@click.option(
    '--exp', type=str,
    required=True, metavar='EXP',
    callback=lambda ctx, param, value: find_exp(value),
    help='Selection of model'
)
@click.option(
    '--ckpt', type=click.IntRange(min=0),
    required=False, metavar='STEPS',
    help='Selection of checkpoint training steps'
)
@click.option(
    '--spk', type=click.STRING,
    required=False,
    help='Speaker name or mixture of speakers'
)
@click.option(
    '--out', type=click.Path(
        file_okay=False, dir_okay=True, path_type=pathlib.Path
    ),
    required=False,
    help='Path of the output folder'
)
@click.option(
    '--title', type=click.STRING,
    required=False,
    help='Title of output file'
)
@click.option(
    '--num', type=click.IntRange(min=1),
    required=False, default=1,
    help='Number of runs'
)
@click.option(
    '--key', type=click.INT,
    required=False, default=0,
    help='Key transition of pitch'
)
@click.option(
    '--gender', type=click.FloatRange(min=-1, max=1),
    required=False,
    help='Formant shifting (gender control)'
)
@click.option(
    '--seed', type=click.INT,
    required=False, default=-1,
    help='Random seed of the inference'
)
@click.option(
    '--depth', type=click.FloatRange(min=0, max=1),
    required=False,
    help='Shallow diffusion depth'
)
@click.option(
    '--steps', type=click.IntRange(min=1),
    required=False,
    help='Diffusion sampling steps'
)
@click.option(
    '--mel', is_flag=True,
    help='Save intermediate mel format instead of waveform'
)
def acoustic(
        proj: pathlib.Path,
        exp: str,
        ckpt: int,
        spk: str,
        out: pathlib.Path,
        title: str,
        num: int,
        key: int,
        gender: float,
        seed: int,
        depth: float,
        steps: int,
        mel: bool
):
    name = proj.stem if not title else title
    if out is None:
        out = proj.parent

    with open(proj, 'r', encoding='utf-8') as f:
        params = json.load(f)

    if not isinstance(params, list):
        params = [params]

    if len(params) == 0:
        print('The input file is empty.')
        exit()

    from utils.infer_utils import trans_key, parse_commandline_spk_mix

    if key != 0:
        params = trans_key(params, key)
        key_suffix = '%+dkey' % key
        if not title:
            name += key_suffix
        print(f'| key transition: {key:+d}')

    sys.argv = [
        sys.argv[0],
        '--exp_name',
        exp,
        '--infer'
    ]
    from utils.hparams import set_hparams, hparams
    set_hparams()

    # Check for vocoder path
    assert mel or (root_dir / hparams['vocoder_ckpt']).exists(), \
        f'Vocoder ckpt \'{hparams["vocoder_ckpt"]}\' not found. ' \
        f'Please put it to the checkpoints directory to run inference.'

    # For compatibility:
    # migrate timesteps, K_step, K_step_infer, diff_speedup to time_scale_factor, T_start, T_start_infer, sampling_steps
    if 'diff_speedup' not in hparams and 'pndm_speedup' in hparams:
        hparams['diff_speedup'] = hparams['pndm_speedup']
    if 'T_start' not in hparams:
        hparams['T_start'] = 1 - hparams['K_step'] / hparams['timesteps']
    if 'T_start_infer' not in hparams:
        hparams['T_start_infer'] = 1 - hparams['K_step_infer'] / hparams['timesteps']
    if 'sampling_steps' not in hparams:
        if hparams['use_shallow_diffusion']:
            hparams['sampling_steps'] = hparams['K_step_infer'] // hparams['diff_speedup']
        else:
            hparams['sampling_steps'] = hparams['timesteps'] // hparams['diff_speedup']
    if 'time_scale_factor' not in hparams:
        hparams['time_scale_factor'] = hparams['timesteps']

    if depth is not None:
        assert depth <= 1 - hparams['T_start'], (
            f"Depth should not be larger than 1 - T_start ({1 - hparams['T_start']})"
        )
        hparams['K_step_infer'] = round(hparams['timesteps'] * depth)
        hparams['T_start_infer'] = 1 - depth
    if steps is not None:
        if hparams['use_shallow_diffusion']:
            step_size = (1 - hparams['T_start_infer']) / steps
            if 'K_step_infer' in hparams:
                hparams['diff_speedup'] = round(step_size * hparams['K_step_infer'])
        else:
            if 'timesteps' in hparams:
                hparams['diff_speedup'] = round(hparams['timesteps'] / steps)
        hparams['sampling_steps'] = steps
    # log all the six migrated parameters
    print('timesteps:', hparams.get('timesteps'))
    print('K_step:', hparams.get('K_step'))
    print('K_step_infer:', hparams.get('K_step_infer'))
    print('diff_speedup:', hparams.get('diff_speedup'))
    print('T_start:', hparams['T_start'])
    print('T_start_infer:', hparams.get('T_start_infer'))
    print('sampling_steps:', hparams.get('sampling_steps'))
    print('time_scale_factor:', hparams.get('time_scale_factor'))

    spk_mix = parse_commandline_spk_mix(spk) if hparams['use_spk_id'] and spk is not None else None
    for param in params:
        if gender is not None and hparams['use_key_shift_embed']:
            param['gender'] = gender

        if spk_mix is not None:
            param['spk_mix'] = spk_mix

    from inference.ds_acoustic import DiffSingerAcousticInfer
    infer_ins = DiffSingerAcousticInfer(load_vocoder=not mel, ckpt_steps=ckpt)
    print(f'| Model: {type(infer_ins.model)}')

    try:
        infer_ins.run_inference(
            params, out_dir=out, title=name, num_runs=num,
            spk_mix=spk_mix, seed=seed, save_mel=mel
        )
    except KeyboardInterrupt:
        exit(-1)


@main.command(help='Run DiffSinger variance model inference')
@click.argument(
    'proj', type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True,
        path_type=pathlib.Path, resolve_path=True
    ),
    metavar='DS_FILE'
)
@click.option(
    '--exp', type=str,
    required=True, metavar='EXP',
    callback=lambda ctx, param, value: find_exp(value),
    help='Selection of model'
)
@click.option(
    '--ckpt', type=click.IntRange(min=0),
    required=False, metavar='STEPS',
    help='Selection of checkpoint training steps'
)
@click.option(
    '--predict', type=click.STRING,
    multiple=True, metavar='TAGS',
    help='Parameters to predict'
)
@click.option(
    '--spk', type=click.STRING,
    required=False,
    help='Speaker name or mixture of speakers'
)
@click.option(
    '--out', type=click.Path(
        file_okay=False, dir_okay=True, path_type=pathlib.Path
    ),
    required=False,
    help='Path of the output folder'
)
@click.option(
    '--title', type=click.STRING,
    required=False,
    help='Title of output file'
)
@click.option(
    '--num', type=click.IntRange(min=1),
    required=False, default=1,
    help='Number of runs'
)
@click.option(
    '--key', type=click.INT,
    required=False, default=0,
    help='Key transition of pitch'
)
@click.option(
    '--expr', type=click.FloatRange(min=0, max=1),
    required=False, help='Static expressiveness control'
)
@click.option(
    '--seed', type=click.INT,
    required=False, default=-1,
    help='Random seed of the inference'
)
@click.option(
    '--steps', type=click.IntRange(min=1),
    required=False,
    help='Diffusion sampling steps'
)
def variance(
        proj: pathlib.Path,
        exp: str,
        ckpt: int,
        spk: str,
        predict: Tuple[str],
        out: pathlib.Path,
        title: str,
        num: int,
        key: int,
        expr: float,
        seed: int,
        steps: int
):
    name = proj.stem if not title else title
    if out is None:
        out = proj.parent
    if (not out or out.resolve() == proj.parent.resolve()) and not title:
        name += '_variance'

    with open(proj, 'r', encoding='utf-8') as f:
        params = json.load(f)

    if not isinstance(params, list):
        params = [params]
    params = [OrderedDict(p) for p in params]

    if len(params) == 0:
        print('The input file is empty.')
        exit()

    from utils.infer_utils import trans_key, parse_commandline_spk_mix

    if key != 0:
        params = trans_key(params, key)
        key_suffix = '%+dkey' % key
        if not title:
            name += key_suffix
        print(f'| key transition: {key:+d}')

    sys.argv = [
        sys.argv[0],
        '--exp_name',
        exp,
        '--infer'
    ]
    from utils.hparams import set_hparams, hparams
    set_hparams()

    # For compatibility:
    # migrate timesteps, K_step, K_step_infer, diff_speedup to time_scale_factor, T_start, T_start_infer, sampling_steps
    if 'diff_speedup' not in hparams and 'pndm_speedup' in hparams:
        hparams['diff_speedup'] = hparams['pndm_speedup']
    if 'sampling_steps' not in hparams:
        hparams['sampling_steps'] = hparams['timesteps'] // hparams['diff_speedup']
    if 'time_scale_factor' not in hparams:
        hparams['time_scale_factor'] = hparams['timesteps']

    if steps is not None:
        if 'timesteps' in hparams:
            hparams['diff_speedup'] = round(hparams['timesteps'] / steps)
        hparams['sampling_steps'] = steps
    # log all the six migrated parameters
    print('timesteps:', hparams.get('timesteps'))
    print('diff_speedup:', hparams.get('diff_speedup'))
    print('sampling_steps:', hparams.get('sampling_steps'))
    print('time_scale_factor:', hparams.get('time_scale_factor'))

    spk_mix = parse_commandline_spk_mix(spk) if hparams['use_spk_id'] and spk is not None else None
    for param in params:
        if expr is not None:
            param['expr'] = expr

        if spk_mix is not None:
            param['ph_spk_mix_backup'] = param.get('ph_spk_mix')
            param['spk_mix_backup'] = param.get('spk_mix')
            param['ph_spk_mix'] = param['spk_mix'] = spk_mix

    from inference.ds_variance import DiffSingerVarianceInfer
    infer_ins = DiffSingerVarianceInfer(ckpt_steps=ckpt, predictions=set(predict))
    print(f'| Model: {type(infer_ins.model)}')

    try:
        infer_ins.run_inference(
            params, out_dir=out, title=name,
            num_runs=num, seed=seed
        )
    except KeyboardInterrupt:
        exit(-1)


if __name__ == '__main__':
    main()
