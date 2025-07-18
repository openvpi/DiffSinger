from torch import Tensor
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT
from dataclasses import dataclass
from typing import Any, Dict, List, Type, Callable, Optional, Iterable


@dataclass
class OptimizerSpec:
    """Spec for creating an optimizer that is part of a `ChainedOptimizer`."""

    class_type: Type[Optimizer]
    init_args: Dict[str, Any]
    param_filter: Optional[Callable[[Tensor], bool]]


class ChainedOptimizer(Optimizer):
    """
    A wrapper around multiple optimizers that allows for chaining them together.
    The optimizers are applied in the order they are passed in the constructor.
    Each optimizer is responsible for updating a subset of the parameters, which
    is determined by the `param_filter` function. If no optimizer is found for a
    parameter group, an exception is raised.
    """

    def __init__(
        self,
        params: ParamsT,
        optimizer_specs: List[OptimizerSpec],
        lr: float,
        weight_decay: float = 0.0,
        optimizer_selection_callback: Optional[Callable[[Tensor, int], None]] = None,
        **common_kwargs,
    ):
        self.optimizer_specs = optimizer_specs
        self.optimizer_selection_callback = optimizer_selection_callback
        self.optimizers: List[Optimizer] = []
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # Split the params for each optimzier
        params_for_optimizers = [[] for _ in optimizer_specs]
        for param_group in self.param_groups:
            params = param_group["params"]
            indices = param_group["optimizer_and_param_group_indices"] = set()
            for param in params:
                assert isinstance(param, Tensor), f"Expected a Tensor, got {type(param)}"
                for index, spec in enumerate(optimizer_specs):
                    if spec.param_filter is None or spec.param_filter(param):
                        if self.optimizer_selection_callback is not None:
                            self.optimizer_selection_callback(param, index)
                        params_for_optimizers[index].append(param)
                        indices.add((index, 0))
                        break

        # Initialize the optimizers
        for spec, selected_params in zip(optimizer_specs, params_for_optimizers):
            optimizer_args = {
                'lr': lr,
                'weight_decay': weight_decay,
            }
            optimizer_args.update(common_kwargs)
            optimizer_args.update(spec.init_args)
            optimizer = spec.class_type(selected_params, **optimizer_args)
            self.optimizers.append(optimizer)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "optimizers": [opt.state_dict() for opt in self.optimizers],
            **super().state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        optimizers = state_dict.pop("optimizers")
        super().load_state_dict(state_dict)
        for i in range(len(self.optimizers)):
            self.optimizers[i].load_state_dict(optimizers[i])

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def _copy_lr_to_optimizers(self) -> None:
        for param_group in self.param_groups:
            indices = param_group["optimizer_and_param_group_indices"]
            for optimizer_idx, param_group_idx in indices:
                self.optimizers[optimizer_idx].param_groups[param_group_idx]["lr"] = param_group["lr"]

    def step(self, closure=None) -> None:
        self._copy_lr_to_optimizers()
        for opt in self.optimizers:
            opt.step(closure)

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        super().add_param_group(param_group)

        # If optimizer has not been initialized, skip adding the param groups
        if not self.optimizers:
            return

        # Split the params for each optimzier
        params_for_optimizers = [[] for _ in self.optimizer_specs]
        params = param_group["params"]
        indices = param_group["optimizer_and_param_group_indices"] = set()
        for param in params:
            assert isinstance(param, Tensor), f"Expected a Tensor, got {type(param)}"
            found_optimizer = False
            for index, spec in enumerate(self.optimizer_specs):
                if spec.param_filter is None or spec.param_filter(param):
                    if self.optimizer_selection_callback is not None:
                        self.optimizer_selection_callback(param, index)
                    params_for_optimizers[index].append(param)
                    indices.add((index, len(self.optimizers[index].param_groups)))
                    found_optimizer = True
                    break
            if not found_optimizer:
                raise ValueError("No valid optimizer found for the given parameter group")

        # Add the selected param group to the optimizers
        for optimizer, selected_params in zip(self.optimizers, params_for_optimizers):
            if selected_params:
                optimizer.add_param_group({"params": selected_params})
