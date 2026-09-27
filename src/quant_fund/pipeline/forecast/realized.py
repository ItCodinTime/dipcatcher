"""Causal realized-GARCH market forecasts and variance overlays.

Split out of the original module. Import the parent path; it re-exports these names.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from quant_fund.config.models import AppConfig
from quant_fund.models.realized_garch import (
    REALIZED_GARCH_FAMILY,
    REALIZED_GARCH_MEASURE,
    RealizedGARCHVol,
)
from quant_fund.models.volatility import GARCH_DATE_LEVEL_SCOPE
from quant_fund.pipeline.train import _realized_garch_history
from quant_fund.schemas.forecast import (
    MARKET_RISK_OVERLAY_GARCH,
    MARKET_RISK_OVERLAY_REALIZED_GARCH,
)

from .garch import (
    GarchMarketForecast,
    _garch_artifact_digest,
    _garch_history_digest,
    _garch_overlay_return_frame,
    garch_market_forecast_asof,
    overlay_covariance_with_garch_market,
)
from .history import _horizon_bars
from .state import _REALIZED_GARCH_ASOF_CACHE, _REALIZED_GARCH_SPEC_CACHE, Array, log


def _realized_garch_artifact_path(config: AppConfig) -> Path:
    return Path(config.data.root) / "metadata" / "vol_realized_garch.joblib"


def _load_realized_garch_spec_cached(config: AppConfig) -> tuple[RealizedGARCHVol, str] | None:
    """Load the trained Realized GARCH spec; cache identity is artifact bytes."""
    path = _realized_garch_artifact_path(config)
    if not path.exists():
        return None
    digest = _garch_artifact_digest(path)
    key = (str(path.resolve()), digest)
    cached = _REALIZED_GARCH_SPEC_CACHE.get(key)
    if cached is not None:
        return cached, digest
    model = RealizedGARCHVol.load(path)
    _REALIZED_GARCH_SPEC_CACHE[key] = model
    if len(_REALIZED_GARCH_SPEC_CACHE) > 8:
        oldest = next(iter(_REALIZED_GARCH_SPEC_CACHE))
        _REALIZED_GARCH_SPEC_CACHE.pop(oldest, None)
    return model, digest


def _clone_realized_garch_spec(
    model: RealizedGARCHVol, *, series_scope: str | None = None
) -> RealizedGARCHVol:
    """Unfitted clone so as-of fits cannot reuse post-origin parameters."""
    return RealizedGARCHVol(
        min_obs=int(model.min_obs),
        mean=str(model.mean),
        series_scope=str(model.series_scope if series_scope is None else series_scope),
        realized_measure=str(model.realized_measure),
    )


def _require_date_level_realized_garch_spec(spec: RealizedGARCHVol) -> str:
    scope = str(getattr(spec, "series_scope", "")).strip()
    if scope != GARCH_DATE_LEVEL_SCOPE:
        raise ValueError(
            f"vol_realized_garch.joblib series_scope must be {GARCH_DATE_LEVEL_SCOPE!r}; "
            f"got {scope!r}"
        )
    if str(spec.realized_measure) != REALIZED_GARCH_MEASURE:
        raise ValueError(
            f"vol_realized_garch.joblib realized_measure must be {REALIZED_GARCH_MEASURE!r}; "
            f"got {spec.realized_measure!r}"
        )
    if bool(spec.intraday_realized_variance):
        raise ValueError("vol_realized_garch.joblib cannot claim intraday realized variance")
    return scope


@dataclass(frozen=True)
class RealizedGarchMarketForecast:
    """Causal Realized GARCH as-of forecast. Parkinson daily OHLC, not HF RV."""

    sigma: float
    variance: float
    cumulative_variance: float
    horizon: int
    series_scope: str
    fit_status: str
    n_obs: int
    fallback_reason: str | None
    realized_measure: str
    intraday_realized_variance: bool
    variance_family: str


def realized_garch_market_forecast_asof(
    config: AppConfig,
    frame: pl.DataFrame,
    asof: datetime,
) -> RealizedGarchMarketForecast | None:
    """Causal Realized GARCH market forecast from Parkinson/return pairs before ``asof``.

    Clones ``vol_realized_garch.joblib`` and refits on date-level equal-weight
    ``ret_1`` paired with same-name one-day Parkinson. Origin-bar OHLC cannot
    enter the fit. ``forecast_asof`` / ``optimize_asof`` / paper-backtest
    ``check_order`` prefer this overlay when the artifact is present. It does
    not replace name-level ``vol_20`` for impact/cost. Missing artifacts
    return None. A present artifact with the wrong scope or a high-frequency
    RV claim fails closed. PIT universe membership and ``available_time``
    follow the GARCH overlay contract.
    """
    loaded = _load_realized_garch_spec_cached(config)
    if loaded is None:
        return None
    spec, spec_digest = loaded
    scope = _require_date_level_realized_garch_spec(spec)
    overlay = _garch_overlay_return_frame(config, frame)
    hist_dates, hist_values, hist_measures = _realized_garch_history(overlay, asof=asof)
    if hist_values.size == 0 or hist_measures.size == 0:
        log.warning(
            "realized_garch_asof_skipped",
            reason="no_strictly_prior_parkinson_pairs",
            asof=str(asof),
        )
        return None
    horizon = _horizon_bars(str(config.train.volatility_target), config)
    path = _realized_garch_artifact_path(config)
    cache_key = (
        str(path.resolve()),
        spec_digest,
        str(asof),
        int(horizon),
        _garch_history_digest(hist_dates, hist_values),
        _garch_history_digest(hist_dates, hist_measures),
        scope,
        REALIZED_GARCH_MEASURE,
    )
    cached = _REALIZED_GARCH_ASOF_CACHE.get(cache_key)
    if cached is not None:
        _REALIZED_GARCH_ASOF_CACHE.move_to_end(cache_key)
        return cached  # type: ignore[return-value]
    model = _clone_realized_garch_spec(spec, series_scope=scope)
    model.fit_returns(hist_values, hist_measures)
    forecast = model.forecast(horizon=horizon)
    variance = np.asarray(forecast["variance"], dtype=float).reshape(-1)
    cumulative = np.asarray(forecast["cumulative_variance"], dtype=float).reshape(-1)
    if variance.size < 1 or not np.isfinite(variance[0]) or float(variance[0]) <= 0.0:
        raise ValueError("Realized GARCH produced an invalid one-step variance")
    if (
        cumulative.size < horizon
        or not np.isfinite(cumulative[horizon - 1])
        or float(cumulative[horizon - 1]) <= 0.0
    ):
        raise ValueError("Realized GARCH produced an invalid horizon-cumulative variance")
    result = RealizedGarchMarketForecast(
        sigma=float(np.sqrt(variance[0])),
        variance=float(variance[0]),
        cumulative_variance=float(cumulative[horizon - 1]),
        horizon=int(horizon),
        series_scope=scope,
        fit_status=str(forecast.get("fit_status", model.fit_status)),
        n_obs=int(model.n_obs),
        fallback_reason=model.fallback_reason,
        realized_measure=REALIZED_GARCH_MEASURE,
        intraday_realized_variance=False,
        variance_family=REALIZED_GARCH_FAMILY,
    )
    _REALIZED_GARCH_ASOF_CACHE[cache_key] = result
    if len(_REALIZED_GARCH_ASOF_CACHE) > 64:
        _REALIZED_GARCH_ASOF_CACHE.popitem(last=False)
    return result


def resolve_market_variance_overlay_asof(
    config: AppConfig,
    frame: pl.DataFrame,
    asof: datetime,
) -> tuple[GarchMarketForecast | RealizedGarchMarketForecast | None, str | None]:
    """Causal date-level market overlay for forecast, optimize, and check_order.

    Prefers Hansen–Huang–Shek Realized GARCH when ``vol_realized_garch.joblib``
    is present. A present Realized GARCH artifact still fail-closes on missing
    OHLC, a high-frequency RV claim, or the wrong ``series_scope``; it does not
    silently fall back to return-only GARCH because the panel lacks ranges.
    When the Realized GARCH artifact is absent, or the as-of history has no
    strictly-prior Parkinson pairs, the return-only GARCH overlay is used.
    Neither overlay replaces name-level ``vol_20`` for impact/cost. This does
    not invent high-frequency realized variance.
    """
    rgarch = realized_garch_market_forecast_asof(config, frame, asof)
    if rgarch is not None:
        return rgarch, MARKET_RISK_OVERLAY_REALIZED_GARCH
    garch = garch_market_forecast_asof(config, frame, asof)
    if garch is not None:
        return garch, MARKET_RISK_OVERLAY_GARCH
    return None, None


def apply_market_variance_overlay_to_covariance(
    config: AppConfig,
    frame: pl.DataFrame,
    asof: datetime,
    sigma: Array,
) -> tuple[Array, GarchMarketForecast | RealizedGarchMarketForecast | None, str | None]:
    """Scale trailing name-covariance to the causal GARCH/RGARCH overlay.

    ``optimize_asof`` and ``/risk/portfolio`` share this helper so the research
    risk diagnostic cannot report unscaled Ledoit–Wolf or OAS while the
    optimizer sized on the overlay. Precedence and fail-closed OHLC / scope /
    high-frequency-RV gates match ``forecast_asof``. Missing artifacts leave
    ``sigma`` unchanged and stamp None. Named one-step DCC/EWMA paths do not
    call this helper. This does not invent high-frequency RV.
    """
    overlay, source = resolve_market_variance_overlay_asof(config, frame, asof)
    if overlay is None or source is None:
        return np.asarray(sigma, dtype=float), None, None
    return (
        overlay_covariance_with_garch_market(sigma, float(overlay.variance)),
        overlay,
        source,
    )


__all__ = [
    "RealizedGarchMarketForecast",
    "apply_market_variance_overlay_to_covariance",
    "realized_garch_market_forecast_asof",
    "resolve_market_variance_overlay_asof",
]
