import importlib
import inspect
from typing import Any

import torch.optim

from .config.schema import LRSchedulerConfig, OptimizerConfig, _walk_lr_scheduler_configs


def get_object_by_module_path(path: str):
    """
    Get an object by module path. The path should be in the format like 'module.submodule.name'.
    """
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    obj = getattr(module, class_name)
    return obj


def filter_kwargs_by_class(cls: type, kwargs: dict[str, Any]):
    """
    Filter the kwargs dictionary to only include keys that are valid arguments for the given class.
    """
    parameters = inspect.signature(cls).parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        # If the class accepts **kwargs, there's no need to filter
        return kwargs.copy()
    valid_keys = [
        name
        for name, param in parameters.items()
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    ]
    filtered_kwargs = {name: kwargs[name] for name in valid_keys if name in kwargs}
    return filtered_kwargs


def build_object_from_class_name(cls_name: str, required_parent_cls=None, /, *args, **kwargs):
    """
    Build an object from a class name. The class name should be in the format 'module.ClassName'.
    If `required_parent_cls` is specified, the class should be a subclass of required_parent_cls.
    """
    cls = get_object_by_module_path(cls_name)
    if required_parent_cls and not issubclass(cls, required_parent_cls):
        raise TypeError(f"Class {cls_name} is not a subclass of {required_parent_cls.__name__}.")
    return cls(*args, **filter_kwargs_by_class(cls, kwargs))


def build_optimizer_from_config(
        module: torch.nn.Module, config: OptimizerConfig
) -> torch.optim.Optimizer:
    """
    Build an optimizer from a configuration object, recursively.
    """
    if config.wraps == "parameters":
        wrapped = module.parameters()
    elif config.wraps == "module":
        wrapped = module
    else:
        raise ValueError(f"Optimizer must wrap 'parameters' or 'module', got '{config.wraps}'.")
    optimizer = build_object_from_class_name(
        config.cls,
        torch.optim.Optimizer,
        wrapped,
        **config.kwargs
    )
    return optimizer


def build_lr_scheduler_from_config(
        optimizer: torch.optim.Optimizer, config: LRSchedulerConfig
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Build a learning rate scheduler from a configuration object, recursively.
    """
    kwargs = _walk_lr_scheduler_configs(
        config.kwargs,
        lambda cfg: build_lr_scheduler_from_config(optimizer, cfg),
    )
    return build_object_from_class_name(
        config.cls,
        torch.optim.lr_scheduler.LRScheduler,
        optimizer=optimizer,
        **kwargs
    )
