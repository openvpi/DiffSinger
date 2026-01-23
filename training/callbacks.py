import base64
from collections import deque
from typing import Literal, Any

import lightning.pytorch
import lightning.pytorch.callbacks
import numpy
import sympy
import torch
from lightning_utilities.core.rank_zero import rank_zero_only

from lib import logging


class FriendlyModelCheckpoint(lightning.pytorch.callbacks.ModelCheckpoint):
    def state_dict(self) -> dict[str, Any]:
        state_dict = super().state_dict()
        # Lightning stores dirpath as an absolute path with platform-specific
        # separators. This is not portable across machines, so we remove it.
        state_dict.pop("dirpath", None)
        return state_dict

    def _save_checkpoint(
            self,
            trainer: lightning.pytorch.Trainer,
            filepath: str
    ) -> None:
        @rank_zero_only
        def progress_bar_print(s):
            trainer.progress_bar_callback.print(s)

        super()._save_checkpoint(trainer, filepath)
        logging.info("Saved checkpoint: " + filepath, callback=progress_bar_print)

    def _remove_checkpoint(
            self,
            trainer: lightning.pytorch.Trainer,
            filepath: str
    ) -> None:
        @rank_zero_only
        def progress_bar_print(s):
            trainer.progress_bar_callback.print(s)

        super()._remove_checkpoint(trainer, filepath)
        logging.info("Removed checkpoint: " + filepath, callback=progress_bar_print)


class PeriodicModelCheckpoint(FriendlyModelCheckpoint):
    def __init__(
            self,
            dirpath: str,
            tag: str,
            unit: Literal["step", "epoch"],
            every_n_units: int,
            since_m_units: int = 0,
            save_last_k: int = 1,
            save_weights_only: bool = False,
    ):
        if unit == "step":
            every_n_train_steps = every_n_units
            every_n_epochs = None
            save_on_epoch_end = False
        elif unit == "epoch":
            every_n_train_steps = None
            every_n_epochs = every_n_units
            save_on_epoch_end = True
        else:
            raise ValueError(f"Unit must be 'step' or 'epoch', got '{unit}'.")
        super().__init__(
            dirpath=dirpath,
            filename=f"model-{tag}-steps={{step:07d}}-epochs={{epoch:04d}}",
            verbose=True,
            save_last=False,
            save_top_k=-1,
            save_weights_only=save_weights_only,
            auto_insert_metric_name=False,
            every_n_train_steps=every_n_train_steps,
            every_n_epochs=every_n_epochs,
            save_on_train_epoch_end=save_on_epoch_end,
            enable_version_counter=False,
        )
        self.unit = unit
        self.since_m_units = since_m_units
        self.save_last_k = save_last_k
        self.last_k_models = deque()

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["last_k_models"] = list(self.last_k_models)
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.last_k_models = deque(state.pop("last_k_models", []))
        super().load_state_dict(state)

    def on_train_batch_end(
            self,
            trainer: lightning.pytorch.Trainer,
            pl_module: lightning.pytorch.LightningModule,
            *args, **kwargs
    ) -> None:
        if self.unit == "step" and trainer.global_step < self.since_m_units:
            return
        super().on_train_batch_end(trainer, pl_module, *args, **kwargs)

    def on_train_epoch_end(
            self,
            trainer: lightning.pytorch.Trainer,
            pl_module: lightning.pytorch.LightningModule
    ) -> None:
        if self.unit == "epoch" and trainer.current_epoch + 1 < self.since_m_units:
            return
        super().on_train_epoch_end(trainer, pl_module)

    def _save_checkpoint(
            self,
            trainer: lightning.pytorch.Trainer,
            filepath: str
    ) -> None:
        if self.save_last_k == 0:
            return
        self.last_k_models.append(filepath)
        super()._save_checkpoint(trainer, filepath)
        if self.save_last_k == -1:
            return
        while len(self.last_k_models) > self.save_last_k:
            self._remove_checkpoint(trainer, self.last_k_models.popleft())


class ExpressionModelCheckpoint(FriendlyModelCheckpoint):
    def __init__(
            self,
            dirpath: str,
            tag: str,
            expression: str,
            mode: Literal["min", "max"],
            save_top_k: int = 1,
            save_weights_only: bool = False,
    ):
        parsed_expr = sympy.sympify(expression)
        if isinstance(parsed_expr, sympy.Number):
            raise ValueError(f"Expression '{expression}' is a constant.")
        if isinstance(parsed_expr, sympy.Symbol):
            metric_name = expression
        else:
            metric_name = "expr"
        metric_key = f"expr_{base64.b16encode(expression.encode('utf8')).decode()}"
        super().__init__(
            dirpath=dirpath,
            filename=f"model-{tag}-steps={{step:07d}}-epochs={{epoch:04d}}-{metric_name}={{{metric_key}:06.4f}}",
            monitor=metric_key,
            verbose=False,
            save_last=False,
            save_top_k=save_top_k,
            save_weights_only=save_weights_only,
            mode=mode,
            auto_insert_metric_name=False,
            enable_version_counter=False,
        )
        self.expression = parsed_expr
        self.metric_key = metric_key

    def _save_topk_checkpoint(
            self,
            trainer: lightning.pytorch.Trainer,
            monitor_candidates: dict[str, torch.Tensor]
    ) -> None:
        eval_result = self.expression.evalf(subs=monitor_candidates)
        if not isinstance(eval_result, sympy.Number):
            raise ValueError(
                f"Expression '{self.expression}' not fully evaluated. "
                f"Valid candidates: {list(monitor_candidates.keys())}"
            )
        eval_result = torch.tensor(float(eval_result), dtype=torch.float32)
        monitor_candidates[self.metric_key] = eval_result
        super()._save_topk_checkpoint(trainer, monitor_candidates)


class FriendlyTQDMProgressBar(lightning.pytorch.callbacks.TQDMProgressBar):
    def __init__(self, refresh_rate: int = 1, process_position: int = 0, show_steps: bool = True):
        super().__init__(refresh_rate, process_position)
        self.show_steps = show_steps

    def get_metrics(self, trainer: lightning.pytorch.Trainer, model: Any):
        items = super().get_metrics(trainer, model)
        if "batch_size" in items:
            items["batch_size"] = int(items["batch_size"])
        if self.show_steps:
            items["steps"] = str(trainer.global_step)
        for k, v in items.items():
            if isinstance(v, float):
                if numpy.isnan(v):
                    items[k] = "nan"
                elif 0.001 <= v < 10:
                    items[k] = numpy.format_float_positional(v, unique=True, precision=5, trim="-")
                elif 0.00001 <= v < 0.001:
                    if len(numpy.format_float_positional(v, unique=True, precision=8, trim="-")) > 8:
                        items[k] = numpy.format_float_scientific(v, precision=3, unique=True, min_digits=2, trim="-")
                    else:
                        items[k] = numpy.format_float_positional(v, unique=True, precision=5, trim="-")
                elif v < 0.00001:
                    items[k] = numpy.format_float_scientific(v, precision=3, unique=True, min_digits=2, trim="-")
        items.pop("v_num", None)
        return items
