import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.fabric.loggers.tensorboard import _TENSORBOARD_AVAILABLE
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_only
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.distributed import Sampler

import utils
from utils.hparams import hparams


# ==========LR schedulers==========

class RSQRTSchedule(object):
    def __init__(self, optimizer):
        super().__init__()
        self.optimizer = optimizer
        self.constant_lr = hparams['lr']
        self.warmup_updates = hparams['warmup_updates']
        self.hidden_size = hparams['hidden_size']
        self.lr = hparams['lr']
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr
        self.step(0)

    def step(self, num_updates):
        constant_lr = self.constant_lr
        warmup = min(num_updates / self.warmup_updates, 1.0)
        rsqrt_decay = max(self.warmup_updates, num_updates) ** -0.5
        rsqrt_hidden = self.hidden_size ** -0.5
        self.lr = max(constant_lr * warmup * rsqrt_decay * rsqrt_hidden, 1e-7)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        return self.lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
        `eta_min` (default=0.0) corresponds to the minimum learning rate reached by the scheduler.
    """

    def __init__(self, optimizer, warmup_steps, t_total, eta_min=0.0, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.eta_min = eta_min
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return step / max(1.0, self.warmup_steps)
        # progress after warmup
        progress = (step - self.warmup_steps) / max(1, self.t_total - self.warmup_steps)
        return max(self.eta_min, 0.5 * (1. + math.cos(math.pi * self.cycles * 2.0 * progress)))


# ==========Torch samplers==========

class DsBatchSampler(Sampler):
    def __init__(self, dataset, max_batch_frames, max_batch_size, sub_indices=None,
                 num_replicas=None, rank=None,
                 required_batch_count_multiple=1, batch_by_size=True, sort_by_similar_size=True,
                 size_reversed=False, shuffle_sample=False, shuffle_batch=False,
                 disallow_empty_batch=True, pad_batch_assignment=True, seed=0, drop_last=False) -> None:
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")
        self.dataset = dataset
        self.max_batch_frames = max_batch_frames
        self.max_batch_size = max_batch_size
        self.sub_indices = sub_indices
        self.num_replicas = num_replicas
        self.rank = rank
        self.required_batch_count_multiple = required_batch_count_multiple
        self.batch_by_size = batch_by_size
        self.sort_by_similar_size = sort_by_similar_size
        self.size_reversed = size_reversed
        self.shuffle_sample = shuffle_sample
        self.shuffle_batch = shuffle_batch
        self.disallow_empty_batch = disallow_empty_batch
        self.pad_batch_assignment = pad_batch_assignment
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.batches = None
        self.formed = None

    def __form_batches(self):
        if self.formed == self.epoch + self.seed:
            return
        rng = np.random.default_rng(self.seed + self.epoch)
        # Create indices
        if self.shuffle_sample:
            if self.sub_indices is not None:
                rng.shuffle(self.sub_indices)
                indices = np.array(self.sub_indices)
            else:
                indices = rng.permutation(len(self.dataset))

            if self.sort_by_similar_size:
                grid = int(hparams.get('sampler_frame_count_grid', 6))
                assert grid > 0
                sizes = (np.round(np.array(self.dataset.sizes)[indices] / grid) * grid).clip(grid, None)
                sizes *= (-1 if self.size_reversed else 1)
                indices = indices[np.argsort(sizes, kind='mergesort')]

            indices = indices.tolist()
        else:
            indices = self.sub_indices if self.sub_indices is not None else list(range(len(self.dataset)))

        # Batching
        if self.batch_by_size:
            batches = utils.batch_by_size(
                indices, self.dataset.num_frames,
                max_batch_frames=self.max_batch_frames,
                max_batch_size=self.max_batch_size
            )
        else:
            batches = [indices[i:i + self.max_batch_size] for i in range(0, len(indices), self.max_batch_size)]
        if len(batches) < self.num_replicas and self.disallow_empty_batch:
            raise RuntimeError("There is not enough batch to assign to each node.")

        # Either drop_last or separate the leftovers.
        floored_total_batch_count = (len(batches) // self.num_replicas) * self.num_replicas
        if self.drop_last and len(batches) > floored_total_batch_count:
            batches = batches[:floored_total_batch_count]
            leftovers = []
            if len(batches) == 0:
                raise RuntimeError("There is no batch left after dropping the last batch.")
        elif self.shuffle_batch:
            leftovers = (rng.permutation(len(batches) - floored_total_batch_count) + floored_total_batch_count).tolist()
        else:
            leftovers = list(range(floored_total_batch_count, len(batches)))

        # Initial batch assignment to current rank.
        batch_assignment = np.arange(floored_total_batch_count).reshape(-1, self.num_replicas).transpose()
        if self.shuffle_batch:
            batch_assignment = rng.permuted(batch_assignment, axis=0)[self.rank].tolist()
        else:
            batch_assignment = batch_assignment[self.rank].tolist()

        # Assign leftovers or pad the batch assignment.
        floored_batch_count = len(batch_assignment)
        if self.rank < len(leftovers):
            batch_assignment.append(leftovers[self.rank])
            floored_batch_count += 1
        elif len(leftovers) > 0 and self.pad_batch_assignment:
            if not batch_assignment:
                raise RuntimeError("Cannot pad empty batch assignment.")
            batch_assignment.append(batch_assignment[self.epoch % floored_batch_count])
        # Ensure the batch count is multiple of required_batch_count_multiple.
        if self.required_batch_count_multiple > 1 and len(batch_assignment) % self.required_batch_count_multiple != 0:
            ceiled_batch_count = math.ceil(
                len(batch_assignment) / self.required_batch_count_multiple
            ) * self.required_batch_count_multiple
            for i in range(ceiled_batch_count - len(batch_assignment)):
                batch_assignment.append(
                    batch_assignment[(i + self.epoch * self.required_batch_count_multiple) % floored_batch_count])

        if batch_assignment:
            self.batches = [deepcopy(batches[i]) for i in batch_assignment]
        else:
            self.batches = [[]]
        self.formed = self.epoch + self.seed

        del indices
        del batches
        del batch_assignment

    def __iter__(self):
        self.__form_batches()
        return iter(self.batches)

    def __len__(self):
        self.__form_batches()
        if self.batches is None:
            raise RuntimeError("Batches are not initialized. Call __form_batches first.")
        return len(self.batches)

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.__form_batches()



# ==========PL related==========

class DsModelCheckpoint(ModelCheckpoint):
    def __init__(
            self,
            *args,
            permanent_ckpt_start,
            permanent_ckpt_interval,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.permanent_ckpt_start = permanent_ckpt_start or 0
        self.permanent_ckpt_interval = permanent_ckpt_interval or 0
        self.enable_permanent_ckpt = self.permanent_ckpt_start > 0 and self.permanent_ckpt_interval > 9

        self._verbose = self.verbose
        self.verbose = False

    def state_dict(self):
        ret = super().state_dict()
        ret.pop('dirpath')
        return ret

    def load_state_dict(self, state_dict) -> None:
        super().load_state_dict(state_dict)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.lightning_module.skip_immediate_ckpt_save:
            trainer.lightning_module.skip_immediate_ckpt_save = False
            return
        self.last_val_step = trainer.global_step
        super().on_validation_end(trainer, pl_module)

    def _update_best_and_save(
            self, current: torch.Tensor, trainer: "pl.Trainer", monitor_candidates: Dict[str, torch.Tensor]
    ) -> None:
        k = len(self.best_k_models) + 1 if self.save_top_k == -1 else self.save_top_k

        del_filepath = None
        _op = max if self.mode == "min" else min
        while len(self.best_k_models) > k and k > 0:
            self.kth_best_model_path = _op(self.best_k_models, key=self.best_k_models.get)  # type: ignore[arg-type]
            self.kth_value = self.best_k_models[self.kth_best_model_path]

            del_filepath = self.kth_best_model_path
            self.best_k_models.pop(del_filepath)
            filepath = self._get_metric_interpolated_filepath_name(monitor_candidates, trainer, del_filepath)
            if del_filepath is not None and filepath != del_filepath:
                self._remove_checkpoint(trainer, del_filepath)

        if len(self.best_k_models) == k and k > 0:
            self.kth_best_model_path = _op(self.best_k_models, key=self.best_k_models.get)  # type: ignore[arg-type]
            self.kth_value = self.best_k_models[self.kth_best_model_path]

        super()._update_best_and_save(current, trainer, monitor_candidates)

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        filepath = (Path(self.dirpath) / Path(filepath).name).resolve()
        super()._save_checkpoint(trainer, str(filepath))
        if self._verbose:
            relative_path = filepath
            # Avoid using `is_relative_to` because Python 3.8 does not support this
            if Path('.').resolve() in filepath.parents:
                relative_path = filepath.relative_to(Path('.').resolve())
            rank_zero_info(f'Checkpoint {relative_path} saved.')

    def _remove_checkpoint(self, trainer: "pl.Trainer", filepath: str):
        filepath = (Path(self.dirpath) / Path(filepath).name).resolve()
        relative_path = filepath
        # Avoid using `is_relative_to` because Python 3.8 does not support this
        if Path('.').resolve() in filepath.parents:
            relative_path = filepath.relative_to(Path('.').resolve())
        search = re.search(r'steps_\d+', relative_path.stem)
        if search:
            step = int(search.group(0)[6:])
            if self.enable_permanent_ckpt and \
                    step >= self.permanent_ckpt_start and \
                    (step - self.permanent_ckpt_start) % self.permanent_ckpt_interval == 0:
                rank_zero_info(f'Checkpoint {relative_path} is now permanent.')
                return
        super()._remove_checkpoint(trainer, filepath)
        if self._verbose:
            rank_zero_info(f'Removed checkpoint {relative_path}.')


def get_latest_checkpoint_path(work_dir):
    if not isinstance(work_dir, Path):
        work_dir = Path(work_dir)
    if not work_dir.exists():
        return None

    last_step = -1
    last_ckpt_name = None

    for ckpt in work_dir.glob('model_ckpt_steps_*.ckpt'):
        search = re.search(r'steps_\d+', ckpt.name)
        if search:
            step = int(search.group(0)[6:])
            if step > last_step:
                last_step = step
                last_ckpt_name = str(ckpt)

    return last_ckpt_name if last_ckpt_name is not None else None


class DsTQDMProgressBar(TQDMProgressBar):
    def __init__(self, refresh_rate: int = 1, process_position: int = 0, show_steps: bool = True):
        super().__init__(refresh_rate, process_position)
        self.show_steps = show_steps

    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        if 'batch_size' in items:
            items['batch_size'] = int(items['batch_size'])
        if self.show_steps:
            items['steps'] = str(trainer.global_step)
        for k, v in items.items():
            if isinstance(v, float):
                if np.isnan(v):
                    items[k] = 'nan'
                elif 0.001 <= v < 10:
                    items[k] = np.format_float_positional(v, unique=True, precision=5, trim='-')
                elif 0.00001 <= v < 0.001:
                    if len(np.format_float_positional(v, unique=True, precision=8, trim='-')) > 8:
                        items[k] = np.format_float_scientific(v, precision=3, unique=True, min_digits=2, trim='-')
                    else:
                        items[k] = np.format_float_positional(v, unique=True, precision=5, trim='-')
                elif v < 0.00001:
                    items[k] = np.format_float_scientific(v, precision=3, unique=True, min_digits=2, trim='-')
        items.pop("v_num", None)
        return items


class DsTensorBoardLogger(TensorBoardLogger):
    @property
    def all_rank_experiment(self):
        if rank_zero_only.rank == 0:
            return self.experiment
        if hasattr(self, "_all_rank_experiment") and self._all_rank_experiment is not None:
            return self._all_rank_experiment

        assert rank_zero_only.rank != 0
        if self.root_dir:
            self._fs.makedirs(self.root_dir, exist_ok=True)

        if _TENSORBOARD_AVAILABLE:
            from torch.utils.tensorboard import SummaryWriter
        else:
            from tensorboardX import SummaryWriter  # type: ignore[no-redef]

        self._all_rank_experiment = SummaryWriter(log_dir=self.log_dir, **self._kwargs)
        return self._all_rank_experiment

    def finalize(self, status: str) -> None:
        if rank_zero_only.rank == 0:
            super().finalize(status)
        elif hasattr(self, "_all_rank_experiment") and self._all_rank_experiment is not None:
            self.all_rank_experiment.flush()
            self.all_rank_experiment.close()

    def __getstate__(self):
        state = super().__getstate__()
        if "_all_rank_experiment" in state:
            del state["_all_rank_experiment"]
        return state


def get_strategy(strategy):
    if strategy['name'] == 'auto':
        return 'auto'

    from lightning.pytorch.strategies import StrategyRegistry
    if strategy['name'] not in StrategyRegistry:
        available_names = ", ".join(sorted(StrategyRegistry.keys())) or "none"
        raise ValueError(f"Invalid strategy name {strategy['name']}. Available names: {available_names}")

    data = StrategyRegistry[strategy['name']]
    params = data['init_params']
    params.update({k: v for k, v in strategy.items() if k != 'name'})
    return data['strategy'](**utils.filter_kwargs(params, data['strategy']))
