import pathlib

import torch
import yaml

from .nets import CascadedNet


class DotDict(dict):
    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_sep_model(model_path, device='cpu'):
    model_path = pathlib.Path(model_path)
    config_file = model_path.with_name('config.yaml')
    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    model = CascadedNet(
        args.n_fft,
        args.hop_length,
        args.n_out,
        args.n_out_lstm,
        True,
        is_mono=args.is_mono
    )
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model
