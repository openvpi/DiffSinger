import argparse
import os
import yaml

try:
    from lightning.pytorch.utilities.rank_zero import rank_zero_only
except ModuleNotFoundError:
    def rank_zero_only(f):
        return f

from utils.multiprocess_utils import is_main_process as mp_is_main_process
global_print_hparams = True
hparams = {}


class Args:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__setattr__(k, v)


def override_config(old_config: dict, new_config: dict):
    for k, v in new_config.items():
        if isinstance(v, dict) and k in old_config:
            override_config(old_config[k], new_config[k])
        else:
            old_config[k] = v


def set_hparams(config='', exp_name='', hparams_str='', print_hparams=True, global_hparams=True):
    """
        Load hparams from multiple sources:
        1. config chain (i.e. first load base_config, then load config);
        2. if reset == False, load from the (auto-saved) complete config file ('config.yaml')
           which contains all settings and do not rely on base_config;
        3. load from argument --hparams or hparams_str, as temporary modification.
    """
    if config == '':
        parser = argparse.ArgumentParser(description='neural music')
        parser.add_argument('--config', type=str, default='',
                            help='path to the YAML configuration file')
        parser.add_argument('--exp_name', type=str, default='', help='exp_name')
        parser.add_argument('--hparams', type=str, default='',
                            help='comma-separated configuration overrides: key=value,key=value')
        parser.add_argument('--infer', action='store_true', help='infer')
        parser.add_argument('--reset', action='store_true', help='reset hparams')
        args, unknown = parser.parse_known_args()
        
        tmp_args_hparams = args.hparams.split(',') if args.hparams.strip() != '' else []
        tmp_args_hparams.extend(hparams_str.split(',') if hparams_str.strip() != '' else [])
        args.hparams = ','.join(tmp_args_hparams)
    else:
        args = Args(config=config, exp_name=exp_name, hparams=hparams_str,
                    infer=False, reset=False)

    args_work_dir = ''
    if args.exp_name != '':
        args.work_dir = args.exp_name
        args_work_dir = os.path.join('checkpoints', args.work_dir)

    config_chains = []
    loaded_config = set()

    def load_config(config_fn):  # deep first
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
                    override_config(ret_hparams, load_config(c))
            override_config(ret_hparams, hparams_)
        else:
            ret_hparams = hparams_
        config_chains.append(config_fn)
        return ret_hparams

    global hparams
    assert args.config != '' or args_work_dir != '', 'Either config or exp name should be specified.'
    saved_hparams = {}
    ckpt_config_path = os.path.join(args_work_dir, 'config.yaml')
    if args_work_dir != '' and os.path.exists(ckpt_config_path):
        with open(ckpt_config_path, encoding='utf-8') as f:
            saved_hparams.update(yaml.safe_load(f))

    hparams_ = {}
    if args.config != '':
        hparams_.update(load_config(args.config))

    if not args.reset:
        hparams_.update(saved_hparams)
    hparams_['work_dir'] = args_work_dir

    if args.hparams != "":
        for new_hparam in args.hparams.split(","):
            if new_hparam.strip() == "":
                continue
            k, v = new_hparam.split("=")
            if k not in hparams_:
                hparams_[k] = eval(v)
            if v in ['True', 'False'] or type(hparams_[k]) == bool:
                hparams_[k] = eval(v)
            else:
                hparams_[k] = type(hparams_[k])(v)

    @rank_zero_only
    def dump_hparams():
        if args_work_dir != '' and (not os.path.exists(ckpt_config_path) or args.reset) and not args.infer:
            os.makedirs(hparams_['work_dir'], exist_ok=True)
            if mp_is_main_process:
                # Only the main process will save the config file
                with open(ckpt_config_path, 'w', encoding='utf-8') as f:
                    hparams_non_recursive = hparams_.copy()
                    hparams_non_recursive['base_config'] = []
                    yaml.safe_dump(hparams_non_recursive, f, allow_unicode=True, encoding='utf-8')
    dump_hparams()

    hparams_['infer'] = args.infer
    if global_hparams:
        hparams.clear()
        hparams.update(hparams_)
    
    if hparams.get('exp_name') is None:
        hparams['exp_name'] = args.exp_name
    if hparams_.get('exp_name') is None:
        hparams_['exp_name'] = args.exp_name
    
    @rank_zero_only
    def print_out_hparams():
        global global_print_hparams
        if mp_is_main_process and print_hparams and global_print_hparams and global_hparams:
            print('| Hparams chains: ', config_chains)
            print('| Hparams: ')
            for i, (k, v) in enumerate(sorted(hparams_.items())):
                print(f"\033[0;33m{k}\033[0m: {v}, ", end="\n" if i % 5 == 4 else "")
            print("")
            global_print_hparams = False
    print_out_hparams()
    
    return hparams_
