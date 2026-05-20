from lightning.pytorch.strategies import StrategyRegistry

from lib.reflection import filter_kwargs_by_class


def get_strategy(name: str = "auto", **kwargs):
    """Create a Lightning strategy instance from a name and keyword arguments."""
    if name == "auto":
        return "auto"

    if name not in StrategyRegistry:
        available = ", ".join(sorted(StrategyRegistry.available_strategies())) or "none"
        raise KeyError(
            f"Invalid strategy name '{name}'. Available names: {available}"
        )

    data = StrategyRegistry[name]
    strategy_cls = data["strategy"]
    params = {**data.get("init_params", {}), **kwargs}
    return strategy_cls(**filter_kwargs_by_class(strategy_cls, params))
