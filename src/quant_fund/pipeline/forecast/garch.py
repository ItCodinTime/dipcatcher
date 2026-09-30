"""Causal GARCH market and name forecasts.

Split out of the original module. Import the parent path; it re-exports these names.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from quant_fund.config.models import AppConfig
from quant_fund.data.parquet import read_derived_parquet
from quant_fund.data.universe import require_valid_membership_panel, restrict_to_membership
from quant_fund.models.covariance import repair_psd
from quant_fund.models.volatility import (
    GARCH_DATE_LEVEL_SCOPE,
    GARCH_SECURITY_LEVEL_SCOPE,
    GARCHVol,
)
from quant_fund.pipeline.train import (
    _garch_name_return_history,
    _garch_return_history,
    _require_garch_security_keys,
    _stamp_strictly_before,
)
from quant_fund.schemas.errors import PointInTimeError

from .artifacts import _joblib_artifact_digest
from .history import _horizon_bars
from .state import (
    _GARCH_ASOF_CACHE,
    _GARCH_NAME_ASOF_CACHE,
    _GARCH_SPEC_CACHE,
    Array,
    log,
)


def _garch_artifact_path(config: AppConfig) -> Path:
    return Path(config.data.root) / "metadata" / "vol_garch.joblib"


def _universe_artifact_path(config: AppConfig) -> Path:
    return Path(config.data.root) / "silver" / "universe.parquet"


def _garch_history_digest(dates: np.ndarray, values: np.ndarray) -> str:
    """Fingerprint the full causal return path, not only the last observation.

    A last-date / last-value / length key can reuse a stale overlay after an
    earlier membership or return rewrite that leaves the terminal mean unchanged.
    """
    digest = hashlib.sha256()
    for stamp in np.asarray(dates).tolist():
        if isinstance(stamp, datetime):
            encoded = stamp.isoformat().encode("utf-8")
        else:
            encoded = str(stamp).encode("utf-8")
        digest.update(encoded)
        digest.update(b"\x1e")
    digest.update(b"\x1f")
    vals = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    digest.update(str(vals.shape).encode("ascii"))
    digest.update(vals.tobytes())
    return digest.hexdigest()


def _garch_overlay_return_frame(config: AppConfig, frame: pl.DataFrame) -> pl.DataFrame:
    """Restrict the GARCH overlay series to the current PIT universe.

    Training scores date-level equal-weight ``ret_1`` on the gold/PIT panel
    (Waves 108–109). Paper/backtest execution bars may still carry ineligible
    ADV/listing names for marks. Those names must not move the market vol
    overlay. Missing universe keeps the caller frame (legacy/test fixtures).
    An empty or invalid universe fails closed.
    """
    path = _universe_artifact_path(config)
    if not path.is_file():
        return frame
    membership = read_derived_parquet(
        path,
        dataset="PIT universe membership",
        required_columns=(),
        allow_empty=True,
    )
    require_valid_membership_panel(membership)
    if membership.is_empty():
        raise PointInTimeError(
            "universe membership is empty; refusing unfiltered GARCH market overlay"
        )
    if "security_id" not in frame.columns:
        raise PointInTimeError(
            "GARCH overlay frame missing security_id; cannot apply PIT membership"
        )
    return restrict_to_membership(frame, membership)


def _garch_artifact_digest(path: Path) -> str:
    """SHA-256 of a GARCH/RGARCH joblib. Mtime is not identity."""
    return _joblib_artifact_digest(path)


def _load_garch_spec_cached(config: AppConfig) -> tuple[GARCHVol, str] | None:
    """Load the trained GARCH spec; cache identity is artifact bytes, not mtime.

    An in-place replacement that preserves mtime (Windows timestamp resolution,
    ``os.utime``, or a same-size rewrite) must not reuse a stale variance
    family, mean, or power. Missing artifacts return None.
    """
    path = _garch_artifact_path(config)
    if not path.exists():
        return None
    digest = _garch_artifact_digest(path)
    key = (str(path.resolve()), digest)
    cached = _GARCH_SPEC_CACHE.get(key)
    if cached is not None:
        return cached, digest
    model = GARCHVol.load(path)
    _GARCH_SPEC_CACHE[key] = model
    if len(_GARCH_SPEC_CACHE) > 8:
        oldest = next(iter(_GARCH_SPEC_CACHE))
        _GARCH_SPEC_CACHE.pop(oldest, None)
    return model, digest


def _clone_garch_spec(model: GARCHVol, *, series_scope: str | None = None) -> GARCHVol:
    """Unfitted clone of a persisted GARCH spec so asof fits cannot reuse future params."""
    return GARCHVol(
        p=int(model.p),
        q=int(model.q),
        dist=str(model.dist),
        vol=str(model.vol),
        min_obs=int(model.min_obs),
        mean=str(model.mean),
        power=float(model.power),
        series_scope=str(model.series_scope if series_scope is None else series_scope),
    )


@dataclass(frozen=True)
class GarchMarketForecast:
    """Causal univariate GARCH as-of forecast. Scope is stamped, never implied."""

    sigma: float
    variance: float
    cumulative_variance: float
    horizon: int
    series_scope: str
    fit_status: str
    n_obs: int
    fallback_reason: str | None


def overlay_covariance_with_garch_market(sigma: Array, garch_variance: float) -> Array:
    """Scale a name-covariance so equal-weight market variance matches the overlay.

    The overlay supplies the *level* of date-level equal-weight variance; the
    input matrix keeps relative covariances. One-step GARCH or Parkinson
    Realized GARCH variance is in the same daily decimal-squared units as
    ``ret_1`` sample covariance. PSD repair is reapplied after scaling. A
    non-positive sample equal-weight variance skips the overlay rather than
    inventing a scale.
    """
    sig = np.asarray(sigma, dtype=float)
    if sig.ndim != 2 or sig.shape[0] != sig.shape[1] or sig.shape[0] == 0:
        raise ValueError("sigma must be a non-empty square covariance")
    if not np.isfinite(sig).all():
        raise ValueError("sigma must contain only finite values")
    var = float(garch_variance)
    if not np.isfinite(var) or var <= 0.0:
        raise ValueError("GARCH market variance must be finite and strictly positive")
    n = int(sig.shape[0])
    weights = np.full(n, 1.0 / n, dtype=float)
    sample_mkt_var = float(weights @ sig @ weights)
    if not np.isfinite(sample_mkt_var) or sample_mkt_var <= 0.0:
        log.warning(
            "garch_cov_overlay_skipped",
            reason="nonpositive_sample_ew_variance",
            sample_mkt_var=sample_mkt_var,
        )
        return sig
    scaled = sig * (var / sample_mkt_var)
    repaired, _ = repair_psd(scaled)
    return repaired


def _require_date_level_garch_spec(spec: GARCHVol) -> str:
    scope = str(getattr(spec, "series_scope", "")).strip()
    if scope != GARCH_DATE_LEVEL_SCOPE:
        raise ValueError(
            f"vol_garch.joblib series_scope must be {GARCH_DATE_LEVEL_SCOPE!r}; got {scope!r}"
        )
    return scope


def _garch_asof_from_returns(
    spec: GARCHVol,
    hist_values: np.ndarray,
    *,
    horizon: int,
    series_scope: str,
) -> GarchMarketForecast:
    """Refit the cloned spec on causal returns and stamp a scoped as-of forecast."""
    model = _clone_garch_spec(spec, series_scope=series_scope)
    model.fit_returns(hist_values)
    forecast = model.forecast(horizon=horizon)
    variance = np.asarray(forecast["variance"], dtype=float).reshape(-1)
    cumulative = np.asarray(forecast["cumulative_variance"], dtype=float).reshape(-1)
    if variance.size < 1 or not np.isfinite(variance[0]) or float(variance[0]) <= 0.0:
        raise ValueError("GARCH produced an invalid one-step variance")
    if (
        cumulative.size < horizon
        or not np.isfinite(cumulative[horizon - 1])
        or float(cumulative[horizon - 1]) <= 0.0
    ):
        raise ValueError("GARCH produced an invalid horizon-cumulative variance")
    return GarchMarketForecast(
        sigma=float(np.sqrt(variance[0])),
        variance=float(variance[0]),
        cumulative_variance=float(cumulative[horizon - 1]),
        horizon=int(horizon),
        series_scope=series_scope,
        fit_status=str(forecast.get("fit_status", model.fit_status)),
        n_obs=int(model.n_obs),
        fallback_reason=model.fallback_reason,
    )


def garch_market_forecast_asof(
    config: AppConfig,
    frame: pl.DataFrame,
    asof: datetime,
) -> GarchMarketForecast | None:
    """Causal GARCH market forecast from returns strictly before ``asof``.

    Uses the persisted ``vol_garch.joblib`` *specification* (p, q, dist, vol,
    series_scope) and refits on the date-level equal-weight ``ret_1`` history
    with ``event_time < asof``. When ``available_time`` is present, late
    restatements (``available_time > asof``) cannot enter that history even
    when their ``event_time`` is earlier. When ``silver/universe.parquet`` is
    present the overlay series is restricted to current PIT members so
    execution-bar extras cannot move the gate. The as-of cache is keyed by a
    digest of the full causal history and the SHA-256 of the spec artifact
    bytes, not mtime or a partial ``(p, q, dist, vol)`` tuple. Mean/power
    replacements therefore cannot reuse a stale overlay. The persisted
    fitted parameters are never reused, so a full-sample artifact cannot
    leak post-asof returns into a historical decision. Missing artifacts
    return None. A present artifact with the wrong ``series_scope`` fails
    closed: it is not an asset-specific vol.
    """
    loaded = _load_garch_spec_cached(config)
    if loaded is None:
        return None
    spec, spec_digest = loaded
    scope = _require_date_level_garch_spec(spec)
    overlay = _garch_overlay_return_frame(config, frame)
    dates, values = _garch_return_history(overlay, asof=asof)
    origin_mask = np.array(
        [_stamp_strictly_before(stamp, asof) for stamp in np.asarray(dates).tolist()],
        dtype=bool,
    )
    if not np.any(origin_mask):
        log.warning("garch_asof_skipped", reason="no_strictly_prior_returns", asof=str(asof))
        return None
    hist_dates = np.asarray(dates)[origin_mask]
    hist_values = np.asarray(values, dtype=float)[origin_mask]
    horizon = _horizon_bars(str(config.train.volatility_target), config)
    path = _garch_artifact_path(config)
    cache_key = (
        str(path.resolve()),
        spec_digest,
        str(asof),
        int(horizon),
        _garch_history_digest(hist_dates, hist_values),
        scope,
    )
    cached = _GARCH_ASOF_CACHE.get(cache_key)
    if cached is not None:
        _GARCH_ASOF_CACHE.move_to_end(cache_key)
        return cached  # type: ignore[return-value]
    result = _garch_asof_from_returns(spec, hist_values, horizon=horizon, series_scope=scope)
    _GARCH_ASOF_CACHE[cache_key] = result
    if len(_GARCH_ASOF_CACHE) > 64:
        _GARCH_ASOF_CACHE.popitem(last=False)
    return result


def _garch_requested_security_ids(
    overlay: pl.DataFrame,
    security_ids: list[str] | tuple[str, ...] | None,
) -> tuple[list[str], bool]:
    """Return unique requested ids and whether the caller listed them explicitly."""
    requested: list[str]
    if security_ids is None:
        requested = sorted(
            {str(sid).strip() for sid in overlay["security_id"].to_list() if str(sid).strip()}
        )
        return requested, False
    requested = []
    seen: set[str] = set()
    for raw in security_ids:
        if not isinstance(raw, str) or isinstance(raw, bool):
            raise ValueError("per-security GARCH security_ids must be strings")
        sid = raw.strip()
        if not sid:
            raise ValueError("per-security GARCH security_ids must be non-empty")
        if sid in seen:
            raise ValueError("per-security GARCH security_ids must be unique")
        seen.add(sid)
        requested.append(sid)
    return requested, True


def garch_name_forecasts_asof(
    config: AppConfig,
    frame: pl.DataFrame,
    asof: datetime,
    security_ids: list[str] | tuple[str, ...] | None = None,
) -> dict[str, GarchMarketForecast]:
    """Causal per-security GARCH one-step forecasts from returns strictly before ``asof``.

    Clones the persisted date-level ``vol_garch.joblib`` specification and refits
    each name on that name's ``ret_1`` only. Output ``series_scope`` is
    ``security_level_ret_1`` so these forecasts cannot be mistaken for the
    equal-weight market overlay. They do not replace ``vol_20`` or
    ``max_predicted_vol``. Missing artifacts return an empty mapping. A present
    artifact with the wrong scope fails closed. Explicit ``security_ids`` with
    no strictly-prior observable returns fail closed; implicit scans omit
    names without history. PIT universe membership and ``available_time``
    follow the market overlay contract. Spec/as-of cache identity is the
    artifact SHA-256, not mtime.
    """
    loaded = _load_garch_spec_cached(config)
    if loaded is None:
        return {}
    spec, spec_digest = loaded
    _require_date_level_garch_spec(spec)
    overlay = _garch_overlay_return_frame(config, frame)
    if "security_id" not in overlay.columns:
        raise PointInTimeError("per-security GARCH frame missing security_id")
    _require_garch_security_keys(overlay)
    requested, explicit = _garch_requested_security_ids(overlay, security_ids)
    horizon = _horizon_bars(str(config.train.volatility_target), config)
    path = _garch_artifact_path(config)
    path_key = str(path.resolve())
    out: dict[str, GarchMarketForecast] = {}
    for sid in requested:
        hist_dates, hist_values = _garch_name_return_history(overlay, sid, asof=asof)
        if hist_values.size == 0:
            if explicit:
                raise ValueError(f"per-security GARCH has no strictly prior returns for {sid!r}")
            continue
        cache_key = (
            path_key,
            spec_digest,
            str(asof),
            sid,
            int(horizon),
            _garch_history_digest(hist_dates, hist_values),
            GARCH_SECURITY_LEVEL_SCOPE,
        )
        cached = _GARCH_NAME_ASOF_CACHE.get(cache_key)
        if cached is not None:
            _GARCH_NAME_ASOF_CACHE.move_to_end(cache_key)
            out[sid] = cached  # type: ignore[assignment]
            continue
        result = _garch_asof_from_returns(
            spec,
            hist_values,
            horizon=horizon,
            series_scope=GARCH_SECURITY_LEVEL_SCOPE,
        )
        _GARCH_NAME_ASOF_CACHE[cache_key] = result
        if len(_GARCH_NAME_ASOF_CACHE) > 256:
            _GARCH_NAME_ASOF_CACHE.popitem(last=False)
        out[sid] = result
    return out


__all__ = [
    "GarchMarketForecast",
    "garch_market_forecast_asof",
    "garch_name_forecasts_asof",
    "overlay_covariance_with_garch_market",
]
