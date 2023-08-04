from __future__ import annotations

import pathlib
import re
import time
import types
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

from basics.base_module import CategorizedModule
from utils.hparams import hparams
from utils.training_utils import get_latest_checkpoint_path


def tensors_to_scalars(metrics):
    new_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            v = v.item()
        if type(v) is dict:
            v = tensors_to_scalars(v)
        new_metrics[k] = v
    return new_metrics


def collate_nd(values, pad_value=0, max_len=None):
    """
    Pad a list of Nd tensors on their first dimension and stack them into a (N+1)d tensor.
    """
    size = ((max(v.size(0) for v in values) if max_len is None else max_len), *values[0].shape[1:])
    res = torch.full((len(values), *size), fill_value=pad_value, dtype=values[0].dtype, device=values[0].device)

    for i, v in enumerate(values):
        res[i, :len(v), ...] = v
    return res


def random_continuous_masks(*shape: int, dim: int, device: str | torch.device = 'cpu'):
    start, end = torch.sort(
        torch.randint(
            low=0, high=shape[dim] + 1, size=(*shape[:dim], 2, *((1,) * (len(shape) - dim - 1))), device=device
        ).expand(*((-1,) * (dim + 1)), *shape[dim + 1:]), dim=dim
    )[0].split(1, dim=dim)
    idx = torch.arange(
        0, shape[dim], dtype=torch.long, device=device
    ).reshape(*((1,) * dim), shape[dim], *((1,) * (len(shape) - dim - 1)))
    masks = (idx >= start) & (idx < end)
    return masks


def _is_batch_full(batch, num_frames, max_batch_frames, max_batch_size):
    if len(batch) == 0:
        return 0
    if len(batch) == max_batch_size:
        return 1
    if num_frames > max_batch_frames:
        return 1
    return 0


def batch_by_size(
        indices, num_frames_fn, max_batch_frames=80000, max_batch_size=48,
        required_batch_size_multiple=1
):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.

    Args:
        indices (List[int]): ordered list of dataset indices
        num_frames_fn (callable): function that returns the number of frames at
            a given index
        max_batch_frames (int, optional): max number of frames in each batch
            (default: 80000).
        max_batch_size (int, optional): max number of sentences in each
            batch (default: 48).
        required_batch_size_multiple: require the batch size to be multiple
            of a given number
    """
    bsz_mult = required_batch_size_multiple

    if isinstance(indices, types.GeneratorType):
        indices = np.fromiter(indices, dtype=np.int64, count=-1)

    sample_len = 0
    sample_lens = []
    batch = []
    batches = []
    for i in range(len(indices)):
        idx = indices[i]
        num_frames = num_frames_fn(idx)
        sample_lens.append(num_frames)
        sample_len = max(sample_len, num_frames)
        assert sample_len <= max_batch_frames, (
            "sentence at index {} of size {} exceeds max_batch_samples "
            "limit of {}!".format(idx, sample_len, max_batch_frames)
        )
        num_frames = (len(batch) + 1) * sample_len

        if _is_batch_full(batch, num_frames, max_batch_frames, max_batch_size):
            mod_len = max(
                bsz_mult * (len(batch) // bsz_mult),
                len(batch) % bsz_mult,
            )
            batches.append(batch[:mod_len])
            batch = batch[mod_len:]
            sample_lens = sample_lens[mod_len:]
            sample_len = max(sample_lens) if len(sample_lens) > 0 else 0
        batch.append(idx)
    if len(batch) > 0:
        batches.append(batch)
    return batches


def make_positions(tensor, padding_idx):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx


def softmax(x, dim):
    return F.softmax(x, dim=dim, dtype=torch.float32)


def unpack_dict_to_list(samples):
    samples_ = []
    bsz = samples.get('outputs').size(0)
    for i in range(bsz):
        res = {}
        for k, v in samples.items():
            try:
                res[k] = v[i]
            except:
                pass
        samples_.append(res)
    return samples_


def filter_kwargs(dict_to_filter, kwarg_obj):
    import inspect

    sig = inspect.signature(kwarg_obj)
    filter_keys = [param.name for param in sig.parameters.values() if param.kind == param.POSITIONAL_OR_KEYWORD]
    filtered_dict = {filter_key: dict_to_filter[filter_key] for filter_key in filter_keys if
                     filter_key in dict_to_filter}
    return filtered_dict


def load_ckpt(
        cur_model, ckpt_base_dir, ckpt_steps=None,
        prefix_in_ckpt='model', key_in_ckpt='state_dict',
        strict=True, device='cpu'
):
    if not isinstance(ckpt_base_dir, pathlib.Path):
        ckpt_base_dir = pathlib.Path(ckpt_base_dir)
    if ckpt_base_dir.is_file():
        checkpoint_path = [ckpt_base_dir]
    elif ckpt_steps is not None:
        checkpoint_path = [ckpt_base_dir / f'model_ckpt_steps_{int(ckpt_steps)}.ckpt']
    else:
        base_dir = ckpt_base_dir
        checkpoint_path = sorted(
            [
                ckpt_file
                for ckpt_file in base_dir.iterdir()
                if ckpt_file.is_file() and re.fullmatch(r'model_ckpt_steps_\d+\.ckpt', ckpt_file.name)
            ],
            key=lambda x: int(re.search(r'\d+', x.name).group(0))
        )
    assert len(checkpoint_path) > 0, f'| ckpt not found in {ckpt_base_dir}.'
    checkpoint_path = checkpoint_path[-1]
    ckpt_loaded = torch.load(checkpoint_path, map_location=device)
    if isinstance(cur_model, CategorizedModule):
        cur_model.check_category(ckpt_loaded.get('category'))
    if key_in_ckpt is None:
        state_dict = ckpt_loaded
    else:
        state_dict = ckpt_loaded[key_in_ckpt]
    if prefix_in_ckpt is not None:
        state_dict = OrderedDict({
            k[len(prefix_in_ckpt) + 1:]: v
            for k, v in state_dict.items() if k.startswith(f'{prefix_in_ckpt}.')
        })
    if not strict:
        cur_model_state_dict = cur_model.state_dict()
        unmatched_keys = []
        for key, param in state_dict.items():
            if key in cur_model_state_dict:
                new_param = cur_model_state_dict[key]
                if new_param.shape != param.shape:
                    unmatched_keys.append(key)
                    print('| Unmatched keys: ', key, new_param.shape, param.shape)
        for key in unmatched_keys:
            del state_dict[key]
    cur_model.load_state_dict(state_dict, strict=strict)
    shown_model_name = 'state dict'
    if prefix_in_ckpt is not None:
        shown_model_name = f'\'{prefix_in_ckpt}\''
    elif key_in_ckpt is not None:
        shown_model_name = f'\'{key_in_ckpt}\''
    print(f'| load {shown_model_name} from \'{checkpoint_path}\'.')







    # return load_pre_train_model()


def remove_padding(x, padding_idx=0):
    if x is None:
        return None
    assert len(x.shape) in [1, 2]
    if len(x.shape) == 2:  # [T, H]
        return x[np.abs(x).sum(-1) != padding_idx]
    elif len(x.shape) == 1:  # [T]
        return x[x != padding_idx]


class Timer:
    timer_map = {}

    def __init__(self, name, print_time=False):
        if name not in Timer.timer_map:
            Timer.timer_map[name] = 0
        self.name = name
        self.print_time = print_time

    def __enter__(self):
        self.t = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        Timer.timer_map[self.name] += time.time() - self.t
        if self.print_time:
            print(self.name, Timer.timer_map[self.name])


def print_arch(model, model_name='model'):
    print(f"| {model_name} Arch: ", model)
    # num_params(model, model_name=model_name)


def num_params(model, print_out=True, model_name="model"):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    if print_out:
        print(f'| {model_name} Trainable Parameters: %.3fM' % parameters)
    return parameters


def build_object_from_class_name(cls_str, parent_cls, *args, **kwargs):
    import importlib

    pkg = ".".join(cls_str.split(".")[:-1])
    cls_name = cls_str.split(".")[-1]
    cls_type = getattr(importlib.import_module(pkg), cls_name)
    if parent_cls is not None:
        assert issubclass(cls_type, parent_cls), f'| {cls_type} is not subclass of {parent_cls}.'

    return cls_type(*args, **filter_kwargs(kwargs, cls_type))

def build_lr_scheduler_from_config(optimizer, scheduler_args):
    def helper(params):
        if isinstance(params, list):
            return [helper(s) for s in params]
        elif isinstance(params, dict):
            resolved = {k: helper(v) for k, v in params.items()}
            resolved['optimizer'] = optimizer
            if 'cls' in resolved:
                obj = build_object_from_class_name(
                    resolved['cls'],
                    torch.optim.lr_scheduler.LRScheduler,
                    **resolved
                )
                if not hasattr(obj, 'last_epoch'):
                    obj.last_epoch = -1
                return obj
            return resolved
        else:
            return params
    resolved = helper(scheduler_args)
    resolved['optimizer'] = optimizer
    return build_object_from_class_name(
        scheduler_args['scheduler_cls'],
        torch.optim.lr_scheduler.LRScheduler,
        **resolved
    )

def simulate_lr_scheduler(optimizer_args, scheduler_args, step_count, num_param_groups=1):
    optimizer = build_object_from_class_name(
        optimizer_args['optimizer_cls'],
        torch.optim.Optimizer,
        [{'params': torch.nn.Parameter(), 'initial_lr': optimizer_args['lr']} for _ in range(num_param_groups)],
        **optimizer_args
    )
    scheduler = build_lr_scheduler_from_config(optimizer, scheduler_args)
    scheduler._initial_step()
    optimizer._step_count = 1
    for _ in range(step_count):
        scheduler.step()
    return scheduler.state_dict()


def remove_suffix(string: str, suffix: str):
    #  Just for Python 3.8 compatibility, since `str.removesuffix()` API of is available since Python 3.9
    if string.endswith(suffix):
        string = string[:-len(suffix)]
    return string
