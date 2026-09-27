"""Training and forecast pipeline.

Submodules stay unloaded until a name is used, so importing
``pipeline.doctor`` does not import training, sklearn, or MLflow.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "HISTORY_SORT_KEYS",
    "MARKET_RISK_OVERLAY_GARCH",
    "MARKET_RISK_OVERLAY_REALIZED_GARCH",
    "build_causal_weight_panel",
    "build_event_time_day_index",
    "history_for_calibration",
    "history_prefix_upto",
    "history_upto",
    "sort_for_history",
    "under_history_sort_contract",
    "slice_day",
    "build_gold",
    "clear_forecast_caches",
    "clear_wrappee_cache",
    "wrappee_cal_fingerprint",
    "wrappee_fit_cache_key",
    "wrappee_cache_size",
    "resolve_wrappee_reselect_cached",
    "doctor",
    "forecast_asof",
    "garch_market_forecast_asof",
    "garch_name_forecasts_asof",
    "garch_name_walk_forward",
    "apply_market_variance_overlay_to_covariance",
    "market_risk_overlay_asof",
    "estimate_optimizer_covariance_asof",
    "optimize_asof",
    "overlay_covariance_with_garch_market",
    "panel",
    "realized_garch_market_forecast_asof",
    "resolve_market_variance_overlay_asof",
    "train_family",
]

_FORECAST = "quant_fund.pipeline.forecast"
_EXPORTS: dict[str, str] = {
    "HISTORY_SORT_KEYS": _FORECAST,
    "MARKET_RISK_OVERLAY_GARCH": _FORECAST,
    "MARKET_RISK_OVERLAY_REALIZED_GARCH": _FORECAST,
    "apply_market_variance_overlay_to_covariance": _FORECAST,
    "build_causal_weight_panel": _FORECAST,
    "build_event_time_day_index": _FORECAST,
    "clear_forecast_caches": _FORECAST,
    "clear_wrappee_cache": _FORECAST,
    "estimate_optimizer_covariance_asof": _FORECAST,
    "forecast_asof": _FORECAST,
    "garch_market_forecast_asof": _FORECAST,
    "garch_name_forecasts_asof": _FORECAST,
    "history_for_calibration": _FORECAST,
    "history_prefix_upto": _FORECAST,
    "history_upto": _FORECAST,
    "market_risk_overlay_asof": _FORECAST,
    "optimize_asof": _FORECAST,
    "overlay_covariance_with_garch_market": _FORECAST,
    "realized_garch_market_forecast_asof": _FORECAST,
    "resolve_market_variance_overlay_asof": _FORECAST,
    "resolve_wrappee_reselect_cached": _FORECAST,
    "slice_day": _FORECAST,
    "sort_for_history": _FORECAST,
    "under_history_sort_contract": _FORECAST,
    "wrappee_cache_size": _FORECAST,
    "wrappee_cal_fingerprint": _FORECAST,
    "wrappee_fit_cache_key": _FORECAST,
    "build_gold": "quant_fund.pipeline.dataset",
    "panel": "quant_fund.pipeline.dataset",
    "doctor": "quant_fund.pipeline.doctor",
    "garch_name_walk_forward": "quant_fund.pipeline.train",
    "train_family": "quant_fund.pipeline.train",
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
