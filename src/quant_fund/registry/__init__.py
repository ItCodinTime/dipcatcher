"""Run registry. MLflow loads with the tracking helpers, not with this package."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "configure_tracking",
    "log_run",
    "promotion_decision",
    "promotion_is_approved",
    "set_alias",
]

_EXPORTS: dict[str, str] = {
    "configure_tracking": "quant_fund.registry.mlflow_store",
    "log_run": "quant_fund.registry.mlflow_store",
    "promotion_decision": "quant_fund.registry.mlflow_store",
    "promotion_is_approved": "quant_fund.registry.mlflow_store",
    "set_alias": "quant_fund.registry.mlflow_store",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
