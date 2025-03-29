import os
from pathlib import Path
import yaml

from lightning.pytorch.utilities.rank_zero import rank_zero_only


def override_config(old_config: dict, new_config: dict):
    for k, v in new_config.items():
        if isinstance(v, dict) and k in old_config:
            override_config(old_config[k], new_config[k])
        else:
            old_config[k] = v

def _load_config(config_fn, config_chain, loaded_config):  # deep first
    with open(config_fn, encoding='utf-8') as f:
        hparams_ = yaml.safe_load(f)
    loaded_config.add(config_fn)
    if 'base_config' in hparams_:
        ret_hparams = {}
        if not isinstance(hparams_['base_config'], list):
            hparams_['base_config'] = [hparams_['base_config']]
        for c in hparams_['base_config']:
            if c not in loaded_config:
                if c.startswith('.'):
                    c = f'{os.path.dirname(config_fn)}/{c}'
                    c = os.path.normpath(c)
                override_config(ret_hparams, _load_config(c, config_chain, loaded_config))
        override_config(ret_hparams, hparams_)
    else:
        ret_hparams = hparams_
    config_chain.append(config_fn)
    return ret_hparams

def read_full_config(config_path: Path=None, exp_name: str='', ckpt_folder: Path=None, infer: bool=False, reset: bool=False):
    assert config_path is not None or exp_name != '', "Must set either config_path or exp_name"
    assert config_path is None or config_path.exists(), f'config_path {config_path} does not exist'
    if ckpt_folder is None:
        ckpt_folder = Path("checkpoints")
    work_dir = None
    config = {}

    config_chain = []
    loaded_config = set()
    if config_path is not None and config_path.exists():
        config.update(_load_config(config_path, config_chain, loaded_config))

    if exp_name != '':
        work_dir = (ckpt_folder / exp_name).resolve().as_posix()
        ckpt_config_path = ckpt_folder / exp_name / 'config.yaml'
        if not reset and ckpt_config_path.exists():
            with open(ckpt_config_path, encoding='utf-8') as f:
                config.update(yaml.safe_load(f))

    config['work_dir'] = work_dir
    config['infer'] = infer
    config['reset'] = reset
    if config.get('exp_name') is None:
        config['exp_name'] = exp_name

    return config, config_chain

@rank_zero_only
def dump_config(config: dict):
    if config['work_dir'] is not None:
        ckpt_config_path = Path(config['work_dir']) / 'config.yaml'
        if (not os.path.exists(ckpt_config_path) or config['reset']) and not config['infer']:
            ckpt_config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(ckpt_config_path, 'w', encoding='utf-8') as f:
                config_non_recursive = config.copy()
                config_non_recursive['base_config'] = []
                yaml.safe_dump(config_non_recursive, f, allow_unicode=True, encoding='utf-8')

@rank_zero_only
def print_config(config: dict, config_chain=None):
    if config_chain is not None:
        print('| Hparams chains: ', list(map(str, config_chain)))
    print('| Hparams: ')
    for i, (k, v) in enumerate(sorted(config.items())):
        print(f"\033[0;33m{k}\033[0m: {v}, ", end="\n" if i % 5 == 4 else "")
    print("")