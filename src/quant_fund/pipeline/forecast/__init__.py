"""Produce AssetForecasts and target weights for a decision date."""

from __future__ import annotations

import hashlib
import types
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_fund.config.models import AppConfig
from quant_fund.data.point_in_time import filter_trailing_returns_asof
from quant_fund.data.universe import require_valid_membership_panel, restrict_to_membership
from quant_fund.fusion.engine import fuse_signals
from quant_fund.metrics.conformal import assign_terciles
from quant_fund.metrics.cross_section import _date_keys as _date_keys
from quant_fund.models.base import JoblibMixin, load_joblib_artifact
from quant_fund.models.calibration import ProbabilityCalibrator
from quant_fund.models.conformal import MondrianCQR, SplitCQR
from quant_fund.models.covariance import (
    DCC_COVARIANCE_OBJECT_ONE_STEP,
    DCC_FAMILY_ADCC,
    DCC_FAMILY_AGDCC,
    DCC_FAMILY_AGDCC_FULL,
    DCC_FAMILY_CCC,
    DCC_FAMILY_GAUSSIAN,
    DCC_FAMILY_STUDENT_T,
    DCC_SPEC_ADCC,
    DCC_SPEC_AGDCC,
    DCC_SPEC_AGDCC_FULL,
    DCC_SPEC_CCC,
    DCC_SPEC_ENGLE_2002,
    DCC_SPEC_STUDENT_T,
    DCC_STAGE1_MIN_OBS,
    EWMA_MIN_OBS,
    EWMA_SPEC_RISKMETRICS,
    IMPLEMENTED_OPTIMIZER_DCC_FAMILIES,
    IMPLEMENTED_OPTIMIZER_NAMED_SPECS,
    IMPLEMENTED_OPTIMIZER_ONE_STEP_SPECS,
    NLSHRINK_MIN_OBS,
    OAS_SPEC_CHEN_2010,
    OPTIMIZER_COVARIANCE_EWMA,
    OPTIMIZER_COVARIANCE_HOMOSKEDASTIC_PROXY,
    OPTIMIZER_COVARIANCE_LEDOIT_WOLF,
    OPTIMIZER_COVARIANCE_LEDOIT_WOLF_NONLINEAR,
    OPTIMIZER_COVARIANCE_OAS,
    OPTIMIZER_COVARIANCE_OBJECT_DIAGONAL_PROXY,
    OPTIMIZER_COVARIANCE_OBJECT_TRAILING,
    OPTIMIZER_COVARIANCE_SAMPLE,
    OPTIMIZER_COVARIANCE_SPEC_DIAGONAL_PROXY,
    OPTIMIZER_COVARIANCE_SPEC_LEDOIT_WOLF,
    OPTIMIZER_COVARIANCE_SPEC_LEDOIT_WOLF_NONLINEAR,
    SAMPLE_SPEC_UNBIASED,
    adcc,
    agdcc,
    agdcc_full,
    ccc,
    dcc_gaussian,
    dcc_student_t,
    dcc_trailing_complete_window,
    ewma,
    ledoit_wolf,
    ledoit_wolf_nonlinear,
    oas,
    repair_psd,
    require_implemented_optimizer_covariance,
    sample,
)
from quant_fund.models.distribution import (
    GaussianDistribution,
    ScaledGaussianDistribution,
    ScaledStudentTDistribution,
    fit_scaled_wrappee,
    select_wrappee_family_name,
)
from quant_fund.models.ranking import available_features
from quant_fund.models.realized_garch import (
    REALIZED_GARCH_FAMILY,
    REALIZED_GARCH_MEASURE,
    RealizedGARCHVol,
)
from quant_fund.models.robinhood_plus.constants import ENGINE_NAME, MODEL_VERSION, STATUS_OK
from quant_fund.models.robinhood_plus.engine import (
    RobinhoodPlusNameForecast,
    forecast_robinhood_plus_cross_section,
    kline_columns_present,
)
from quant_fund.models.volatility import (
    GARCH_DATE_LEVEL_SCOPE,
    GARCH_SECURITY_LEVEL_SCOPE,
    GARCHVol,
)
from quant_fund.pipeline.dataset import design_matrix, panel
from quant_fund.pipeline.train import (
    _garch_name_return_history as _garch_name_return_history,
)
from quant_fund.pipeline.train import (
    _garch_return_history as _garch_return_history,
)
from quant_fund.pipeline.train import (
    _realized_garch_history as _realized_garch_history,
)
from quant_fund.pipeline.train import (
    _require_garch_security_keys as _require_garch_security_keys,
)
from quant_fund.pipeline.train import (
    _stamp_strictly_before as _stamp_strictly_before,
)
from quant_fund.portfolio.interval_risk import apply_interval_caps, interval_refs
from quant_fund.portfolio.optimizer import optimize_mean_variance
from quant_fund.schemas.errors import OptimizationInfeasible, PointInTimeError
from quant_fund.schemas.forecast import (
    MARKET_RISK_OVERLAY_GARCH,
    MARKET_RISK_OVERLAY_REALIZED_GARCH,
    AssetForecast,
    IntervalMethod,
    MarketState,
)
from quant_fund.utils.hashing import hash_file
from quant_fund.utils.logging import get_logger

from . import artifacts as _mod_artifacts
from . import conformal as _mod_conformal
from . import covariance as _mod_covariance
from . import decide as _mod_decide
from . import garch as _mod_garch
from . import history as _mod_history
from . import realized as _mod_realized
from . import state as _mod_state
from . import wrappee as _mod_wrappee
from .artifacts import (
    _joblib_artifact_digest as _joblib_artifact_digest,
)
from .artifacts import (
    _load_probability_calibrator as _load_probability_calibrator,
)
from .artifacts import (
    _load_ranker_cached as _load_ranker_cached,
)
from .artifacts import (
    _load_rl_cached as _load_rl_cached,
)
from .artifacts import (
    _paper_challenger_stamp as _paper_challenger_stamp,
)
from .artifacts import (
    _ranker_artifact_path as _ranker_artifact_path,
)
from .conformal import (
    conformal_sets_asof,
)
from .covariance import (
    OptimizerCovarianceEstimate,
    estimate_optimizer_covariance_asof,
    market_risk_overlay_asof,
)
from .covariance import (
    _market_overlay_diagnostics as _market_overlay_diagnostics,
)
from .covariance import (
    _market_overlay_note as _market_overlay_note,
)
from .covariance import (
    _named_dcc_optimizer_estimate as _named_dcc_optimizer_estimate,
)
from .covariance import (
    _named_ewma_optimizer_estimate as _named_ewma_optimizer_estimate,
)
from .covariance import (
    _named_ledoit_wolf_nonlinear_optimizer_estimate as _named_ledoit_wolf_nonlinear_optimizer_estimate,
)
from .covariance import (
    _named_oas_optimizer_estimate as _named_oas_optimizer_estimate,
)
from .covariance import (
    _named_sample_optimizer_estimate as _named_sample_optimizer_estimate,
)
from .covariance import (
    _trailing_return_matrix as _trailing_return_matrix,
)
from .covariance import (
    _unmeasured_optimizer_covariance as _unmeasured_optimizer_covariance,
)
from .decide import (
    _align_w_prev as _align_w_prev,
)
from .decide import (
    _alpha_vector as _alpha_vector,
)
from .decide import (
    _apply_forecast_interval_caps as _apply_forecast_interval_caps,
)
from .decide import (
    _count_robinhood_plus_ok as _count_robinhood_plus_ok,
)
from .decide import (
    _panel_cached as _panel_cached,
)
from .decide import (
    _robinhood_plus_asof as _robinhood_plus_asof,
)
from .decide import (
    build_causal_weight_panel,
    decision_dates,
    forecast_asof,
    optimize_asof,
)
from .garch import (
    GarchMarketForecast,
    garch_market_forecast_asof,
    garch_name_forecasts_asof,
    overlay_covariance_with_garch_market,
)
from .garch import (
    _clone_garch_spec as _clone_garch_spec,
)
from .garch import (
    _garch_artifact_digest as _garch_artifact_digest,
)
from .garch import (
    _garch_artifact_path as _garch_artifact_path,
)
from .garch import (
    _garch_asof_from_returns as _garch_asof_from_returns,
)
from .garch import (
    _garch_history_digest as _garch_history_digest,
)
from .garch import (
    _garch_overlay_return_frame as _garch_overlay_return_frame,
)
from .garch import (
    _garch_requested_security_ids as _garch_requested_security_ids,
)
from .garch import (
    _load_garch_spec_cached as _load_garch_spec_cached,
)
from .garch import (
    _require_date_level_garch_spec as _require_date_level_garch_spec,
)
from .garch import (
    _universe_artifact_path as _universe_artifact_path,
)
from .history import (
    HISTORY_SORT_KEYS,
    ForecastIntervals,
    build_event_time_day_index,
    history_for_calibration,
    history_prefix_upto,
    history_upto,
    latest_decision,
    slice_day,
    sort_for_history,
    under_history_sort_contract,
)
from .history import (
    _align_col as _align_col,
)
from .history import (
    _date_train_cal as _date_train_cal,
)
from .history import (
    _decision_x as _decision_x,
)
from .history import (
    _distribution_label as _distribution_label,
)
from .history import (
    _horizon_bars as _horizon_bars,
)
from .history import (
    _horizon_name as _horizon_name,
)
from .history import (
    _require_assume_sorted_contract as _require_assume_sorted_contract,
)
from .realized import (
    RealizedGarchMarketForecast,
    apply_market_variance_overlay_to_covariance,
    realized_garch_market_forecast_asof,
    resolve_market_variance_overlay_asof,
)
from .realized import (
    _clone_realized_garch_spec as _clone_realized_garch_spec,
)
from .realized import (
    _load_realized_garch_spec_cached as _load_realized_garch_spec_cached,
)
from .realized import (
    _realized_garch_artifact_path as _realized_garch_artifact_path,
)
from .realized import (
    _require_date_level_realized_garch_spec as _require_date_level_realized_garch_spec,
)
from .state import (
    _CONFORMAL_CACHE as _CONFORMAL_CACHE,
)
from .state import (
    _GARCH_ASOF_CACHE as _GARCH_ASOF_CACHE,
)
from .state import (
    _GARCH_NAME_ASOF_CACHE as _GARCH_NAME_ASOF_CACHE,
)
from .state import (
    _GARCH_SPEC_CACHE as _GARCH_SPEC_CACHE,
)
from .state import (
    _PANEL_ASOF_CACHE as _PANEL_ASOF_CACHE,
)
from .state import (
    _RANKER_CACHE as _RANKER_CACHE,
)
from .state import (
    _REALIZED_GARCH_ASOF_CACHE as _REALIZED_GARCH_ASOF_CACHE,
)
from .state import (
    _REALIZED_GARCH_SPEC_CACHE as _REALIZED_GARCH_SPEC_CACHE,
)
from .state import (
    _RL_POLICY_CACHE as _RL_POLICY_CACHE,
)
from .state import (
    _WRAPPEE_CACHE as _WRAPPEE_CACHE,
)
from .state import (
    INTERVAL_ALPHA,
    Array,
    log,
)
from .wrappee import (
    _array_content_digest as _array_content_digest,
)
from .wrappee import (
    _require_wrappee_resolve_inputs as _require_wrappee_resolve_inputs,
)
from .wrappee import (
    clear_forecast_caches,
    clear_wrappee_cache,
    resolve_wrappee_reselect_cached,
    wrappee_cache_size,
    wrappee_cal_fingerprint,
    wrappee_fit_cache_key,
)

_FACADE_MODULES = (
    _mod_state,
    _mod_wrappee,
    _mod_history,
    _mod_conformal,
    _mod_artifacts,
    _mod_garch,
    _mod_realized,
    _mod_covariance,
    _mod_decide,
)

_MISSING = object()


def _publish_split_modules() -> None:
    """Copy split-module globals here so rebound functions resolve patches."""
    facade = globals()
    for module in _FACADE_MODULES:
        for key, value in vars(module).items():
            if key.startswith("__"):
                continue
            current = facade.get(key, _MISSING)
            if current is not _MISSING and current is not value:
                raise RuntimeError(
                    f"facade global conflict on {key!r} while publishing {module.__name__}"
                )
            facade[key] = value


def _rebind_to_facade(fn: types.FunctionType) -> types.FunctionType:
    rebound = types.FunctionType(
        fn.__code__,
        globals(),
        fn.__name__,
        fn.__defaults__,
        fn.__closure__,
    )
    rebound.__kwdefaults__ = fn.__kwdefaults__
    rebound.__annotations__ = dict(getattr(fn, "__annotations__", {}))
    rebound.__dict__.update(fn.__dict__)
    rebound.__module__ = __name__
    rebound.__qualname__ = fn.__qualname__
    return rebound


def _rebind_split_functions() -> None:
    facade = globals()
    prefix = __name__ + "."
    for key, value in list(facade.items()):
        module_name = getattr(value, "__module__", "") or ""
        if isinstance(value, types.FunctionType) and module_name.startswith(prefix):
            facade[key] = _rebind_to_facade(value)


_publish_split_modules()
_rebind_split_functions()

__all__ = [
    "AppConfig",
    "Array",
    "AssetForecast",
    "Callable",
    "DCC_COVARIANCE_OBJECT_ONE_STEP",
    "DCC_FAMILY_ADCC",
    "DCC_FAMILY_AGDCC",
    "DCC_FAMILY_AGDCC_FULL",
    "DCC_FAMILY_CCC",
    "DCC_FAMILY_GAUSSIAN",
    "DCC_FAMILY_STUDENT_T",
    "DCC_SPEC_ADCC",
    "DCC_SPEC_AGDCC",
    "DCC_SPEC_AGDCC_FULL",
    "DCC_SPEC_CCC",
    "DCC_SPEC_ENGLE_2002",
    "DCC_SPEC_STUDENT_T",
    "DCC_STAGE1_MIN_OBS",
    "ENGINE_NAME",
    "EWMA_MIN_OBS",
    "EWMA_SPEC_RISKMETRICS",
    "ForecastIntervals",
    "GARCHVol",
    "GARCH_DATE_LEVEL_SCOPE",
    "GARCH_SECURITY_LEVEL_SCOPE",
    "GarchMarketForecast",
    "GaussianDistribution",
    "HISTORY_SORT_KEYS",
    "IMPLEMENTED_OPTIMIZER_DCC_FAMILIES",
    "IMPLEMENTED_OPTIMIZER_NAMED_SPECS",
    "IMPLEMENTED_OPTIMIZER_ONE_STEP_SPECS",
    "INTERVAL_ALPHA",
    "IntervalMethod",
    "JoblibMixin",
    "MARKET_RISK_OVERLAY_GARCH",
    "MARKET_RISK_OVERLAY_REALIZED_GARCH",
    "MODEL_VERSION",
    "MarketState",
    "MondrianCQR",
    "NDArray",
    "NLSHRINK_MIN_OBS",
    "OAS_SPEC_CHEN_2010",
    "OPTIMIZER_COVARIANCE_EWMA",
    "OPTIMIZER_COVARIANCE_HOMOSKEDASTIC_PROXY",
    "OPTIMIZER_COVARIANCE_LEDOIT_WOLF",
    "OPTIMIZER_COVARIANCE_LEDOIT_WOLF_NONLINEAR",
    "OPTIMIZER_COVARIANCE_OAS",
    "OPTIMIZER_COVARIANCE_OBJECT_DIAGONAL_PROXY",
    "OPTIMIZER_COVARIANCE_OBJECT_TRAILING",
    "OPTIMIZER_COVARIANCE_SAMPLE",
    "OPTIMIZER_COVARIANCE_SPEC_DIAGONAL_PROXY",
    "OPTIMIZER_COVARIANCE_SPEC_LEDOIT_WOLF",
    "OPTIMIZER_COVARIANCE_SPEC_LEDOIT_WOLF_NONLINEAR",
    "OptimizationInfeasible",
    "OptimizerCovarianceEstimate",
    "OrderedDict",
    "Path",
    "PointInTimeError",
    "ProbabilityCalibrator",
    "REALIZED_GARCH_FAMILY",
    "REALIZED_GARCH_MEASURE",
    "RealizedGARCHVol",
    "RealizedGarchMarketForecast",
    "RobinhoodPlusNameForecast",
    "SAMPLE_SPEC_UNBIASED",
    "STATUS_OK",
    "ScaledGaussianDistribution",
    "ScaledStudentTDistribution",
    "SplitCQR",
    "adcc",
    "agdcc",
    "agdcc_full",
    "apply_interval_caps",
    "apply_market_variance_overlay_to_covariance",
    "assign_terciles",
    "available_features",
    "build_causal_weight_panel",
    "build_event_time_day_index",
    "decision_dates",
    "ccc",
    "clear_forecast_caches",
    "clear_wrappee_cache",
    "conformal_sets_asof",
    "dataclass",
    "datetime",
    "dcc_gaussian",
    "dcc_student_t",
    "dcc_trailing_complete_window",
    "design_matrix",
    "estimate_optimizer_covariance_asof",
    "ewma",
    "filter_trailing_returns_asof",
    "fit_scaled_wrappee",
    "forecast_asof",
    "forecast_robinhood_plus_cross_section",
    "fuse_signals",
    "garch_market_forecast_asof",
    "garch_name_forecasts_asof",
    "get_logger",
    "hash_file",
    "hashlib",
    "history_for_calibration",
    "history_prefix_upto",
    "history_upto",
    "interval_refs",
    "kline_columns_present",
    "latest_decision",
    "ledoit_wolf",
    "ledoit_wolf_nonlinear",
    "load_joblib_artifact",
    "log",
    "market_risk_overlay_asof",
    "np",
    "oas",
    "optimize_asof",
    "optimize_mean_variance",
    "overlay_covariance_with_garch_market",
    "panel",
    "pl",
    "realized_garch_market_forecast_asof",
    "repair_psd",
    "replace",
    "require_implemented_optimizer_covariance",
    "require_valid_membership_panel",
    "resolve_market_variance_overlay_asof",
    "resolve_wrappee_reselect_cached",
    "restrict_to_membership",
    "sample",
    "select_wrappee_family_name",
    "slice_day",
    "sort_for_history",
    "under_history_sort_contract",
    "wrappee_cache_size",
    "wrappee_cal_fingerprint",
    "wrappee_fit_cache_key",
]
