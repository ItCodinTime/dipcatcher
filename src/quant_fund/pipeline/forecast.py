"""Produce AssetForecasts and target weights for a decision date."""

from __future__ import annotations

import hashlib
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
from quant_fund.metrics.cross_section import _date_keys
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
    _garch_name_return_history,
    _garch_return_history,
    _realized_garch_history,
    _require_garch_security_keys,
    _stamp_strictly_before,
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

Array = NDArray[np.float64]

log = get_logger()

INTERVAL_ALPHA = 0.10

# Process-local caches for causal panel / multi-asof hot paths
_RANKER_CACHE: dict[tuple[str, str], object] = {}
_RL_POLICY_CACHE: dict[tuple[str, float], object] = {}
_GARCH_SPEC_CACHE: dict[tuple[str, str], GARCHVol] = {}
_GARCH_ASOF_CACHE: OrderedDict[tuple, object] = OrderedDict()
_GARCH_NAME_ASOF_CACHE: OrderedDict[tuple, object] = OrderedDict()
_REALIZED_GARCH_SPEC_CACHE: dict[tuple[str, str], RealizedGARCHVol] = {}
_REALIZED_GARCH_ASOF_CACHE: OrderedDict[tuple, object] = OrderedDict()
_PANEL_ASOF_CACHE: dict[tuple[str, float, float], pl.DataFrame] = {}
_CONFORMAL_CACHE: dict[tuple, ForecastIntervals] = {}
# Wrappee train-fit reuse: key = train-primary fingerprint + selected family name.
# Family is re-selected on current cal each call; fit reused only when family matches.
_WRAPPEE_CACHE: OrderedDict[tuple, object] = OrderedDict()


def _array_content_digest(*arrays: NDArray[np.float64]) -> str:
    """Return a deterministic digest for numeric inputs used by cached fits."""
    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def clear_wrappee_cache() -> None:
    """Drop cached operational wrappee fits (Student-t / scaled Gaussian)."""
    _WRAPPEE_CACHE.clear()


def wrappee_cache_size() -> int:
    """Number of entries in the process-local wrappee cache (tests / diagnostics)."""
    return len(_WRAPPEE_CACHE)


def wrappee_cal_fingerprint(
    *,
    train_date_keys: tuple[object, ...],
    cal_date_keys: tuple[object, ...] = (),
    alpha: float,
    n_tr: int,
    n_cal: int = 0,
    y_tr_mean: float,
    scale_tr_mean: float,
    label: str = "",
    include_cal: bool = False,
    train_content_digest: str = "",
) -> tuple:
    """Fingerprint for ``_WRAPPEE_CACHE``.

    **When reuse is valid (train-primary, default ``include_cal=False``)**

    Train-fit reuse is keyed by this fingerprint **plus** the selected family
    name (see ``wrappee_fit_cache_key`` / ``resolve_wrappee_reselect_cached``).
    Family selection always uses the current cal/holdout; the fit itself is
    train-only. Adjacent asofs that share the train window may reuse the fitted
    wrappee when the re-selected family matches. Cal-window slides alone do
    **not** invalidate the train-primary key (family may still change).

    **Stale-reuse guards (always in the key)**

    Alpha (interval config), label column, train date keys, train sample size,
    and train y/scale means always participate. Production callers also pass
    a content digest so same-mean data cannot collide. Pass
    ``include_cal=True`` for a strict train+cal identity key (diagnostics / tests
    only).
    """
    base = (
        tuple(train_date_keys),
        float(alpha),
        str(label),
        int(n_tr),
        float(y_tr_mean),
        float(scale_tr_mean),
        str(train_content_digest),
    )
    if include_cal:
        return base + (tuple(cal_date_keys), int(n_cal))
    return base


def wrappee_fit_cache_key(
    *,
    train_date_keys: tuple[object, ...],
    alpha: float,
    n_tr: int,
    y_tr_mean: float,
    scale_tr_mean: float,
    label: str,
    family: str,
    cal_date_keys: tuple[object, ...] = (),
    n_cal: int = 0,
    include_cal: bool = False,
    train_content_digest: str = "",
) -> tuple:
    """Train-primary fit key including selected family (Wave 7).

    Cal keys are ignored unless ``include_cal=True`` (diagnostics). Alpha/label
    and train identity always participate so config changes invalidate.
    """
    fp = wrappee_cal_fingerprint(
        train_date_keys=train_date_keys,
        cal_date_keys=cal_date_keys,
        alpha=alpha,
        n_tr=n_tr,
        n_cal=n_cal,
        y_tr_mean=y_tr_mean,
        scale_tr_mean=scale_tr_mean,
        label=label,
        include_cal=include_cal,
        train_content_digest=train_content_digest,
    )
    return fp + (str(family),)


def _require_wrappee_resolve_inputs(
    taus: list[float],
    y_train: NDArray[np.float64],
    scale_train: NDArray[np.float64],
    y_cal: NDArray[np.float64],
    scale_cal: NDArray[np.float64],
    *,
    alpha: float,
    min_coverage: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Wave 50: fail-closed guards for wrappee resolve (research diagnostic path).

    Rejects empty taus, empty/mismatched y/scale arrays, and alpha/min_coverage
    outside (0, 1). Returns contiguous float64 views. Not a live fill claim.
    """
    if not taus:
        raise ValueError("taus must be a non-empty list")
    yt = np.asarray(y_train, dtype=np.float64).reshape(-1)
    st = np.asarray(scale_train, dtype=np.float64).reshape(-1)
    yc = np.asarray(y_cal, dtype=np.float64).reshape(-1)
    sc = np.asarray(scale_cal, dtype=np.float64).reshape(-1)
    if yt.size == 0 or st.size == 0:
        raise ValueError("y_train and scale_train must be non-empty")
    if yc.size == 0 or sc.size == 0:
        raise ValueError("y_cal and scale_cal must be non-empty")
    if yt.size != st.size:
        raise ValueError("y_train and scale_train length mismatch")
    if yc.size != sc.size:
        raise ValueError("y_cal and scale_cal length mismatch")
    a = float(alpha)
    if not np.isfinite(a) or not (0.0 < a < 1.0):
        raise ValueError("alpha must be finite and in (0, 1)")
    mc = float(min_coverage)
    if not np.isfinite(mc) or not (0.0 < mc < 1.0):
        raise ValueError("min_coverage must be finite and in (0, 1)")
    return yt, st, yc, sc


def resolve_wrappee_reselect_cached(
    taus: list[float],
    y_train: NDArray[np.float64],
    scale_train: NDArray[np.float64],
    y_cal: NDArray[np.float64],
    scale_cal: NDArray[np.float64],
    *,
    train_date_keys: tuple[object, ...],
    cal_date_keys: tuple[object, ...] = (),
    alpha: float,
    label: str = "",
    min_coverage: float = 0.85,
    nominal_coverage: float = 0.90,
) -> tuple[str, ScaledGaussianDistribution | ScaledStudentTDistribution, dict[str, object]]:
    """Re-select wrappee family on current cal; reuse train fit when fingerprint allows.

    Selection always runs on the provided cal/holdout (freshness when cal grows).
    The expensive train MLE is reused from ``_WRAPPEE_CACHE`` when the train-primary
    fingerprint and selected family match a prior entry.

    Returns ``(family_name, fitted_model, meta)`` with ``meta`` keys:
    ``cache_hit``, ``family``, ``reselected``, ``live_pnl_claim`` (always False).
    """
    _ = nominal_coverage
    y_train, scale_train, y_cal, scale_cal = _require_wrappee_resolve_inputs(
        taus,
        y_train,
        scale_train,
        y_cal,
        scale_cal,
        alpha=alpha,
        min_coverage=min_coverage,
    )
    family = select_wrappee_family_name(
        y_train,
        scale_train,
        y_cal,
        scale_cal,
        min_coverage=min_coverage,
    )
    fit_key = wrappee_fit_cache_key(
        train_date_keys=train_date_keys,
        cal_date_keys=cal_date_keys,
        alpha=float(alpha),
        n_tr=int(y_train.size),
        n_cal=int(y_cal.size),
        y_tr_mean=float(np.nanmean(y_train)),
        scale_tr_mean=float(np.nanmean(scale_train)),
        label=str(label),
        family=family,
        include_cal=False,
        train_content_digest=_array_content_digest(y_train, scale_train),
    )
    cached = _WRAPPEE_CACHE.get(fit_key)
    meta: dict[str, object] = {
        "family": family,
        "reselected": True,
        "live_pnl_claim": False,
        "research_only": True,
    }
    if isinstance(cached, (ScaledGaussianDistribution, ScaledStudentTDistribution)):
        _WRAPPEE_CACHE.move_to_end(fit_key)
        meta["cache_hit"] = True
        return family, cached, meta
    model = fit_scaled_wrappee(family, taus, y_train, scale_train)
    # Keep the bounded cache warm across broad multi-asof runs. Clearing the
    # entire cache at the capacity boundary creates a needless thundering herd
    # of refits; evict the least-recently-used fit instead.
    if len(_WRAPPEE_CACHE) >= 64:
        _WRAPPEE_CACHE.popitem(last=False)
    _WRAPPEE_CACHE[fit_key] = model
    meta["cache_hit"] = False
    return family, model, meta


def clear_forecast_caches() -> None:
    """Clear ranker / conformal / wrappee / GARCH process-local caches (panel cache separate)."""
    _RANKER_CACHE.clear()
    _RL_POLICY_CACHE.clear()
    _GARCH_SPEC_CACHE.clear()
    _GARCH_ASOF_CACHE.clear()
    _GARCH_NAME_ASOF_CACHE.clear()
    _REALIZED_GARCH_SPEC_CACHE.clear()
    _REALIZED_GARCH_ASOF_CACHE.clear()
    _PANEL_ASOF_CACHE.clear()
    _CONFORMAL_CACHE.clear()
    _WRAPPEE_CACHE.clear()


@dataclass(frozen=True)
class ForecastIntervals:
    """CQR / Mondrian sets. Separate from raw quantile PIT, pinball, and CRPS."""

    lower: dict[str, float]
    upper: dict[str, float]
    alpha: float
    method: IntervalMethod
    horizon: str
    cal_event_times: tuple[object, ...]


def latest_decision(frame: pl.DataFrame) -> datetime:
    value = frame["event_time"].max()
    if not isinstance(value, datetime):
        raise TypeError("event_time max is not a datetime")
    return value


# Wave 10: documented sort contract for order-preserving history_upto wiring.
# Production panels (dataset.panel / build_gold) already sort by these keys.
HISTORY_SORT_KEYS: tuple[str, ...] = ("event_time", "security_id")


def under_history_sort_contract(frame: pl.DataFrame) -> bool:
    """True when ``frame`` row order matches stable sort by HISTORY_SORT_KEYS.

    Empty frames and frames missing required columns are treated as non-contract
    (callers should fall back to explicit ``<=`` filters).
    """
    if frame.is_empty():
        return True
    for col in HISTORY_SORT_KEYS:
        if col not in frame.columns:
            return False
    keys = list(HISTORY_SORT_KEYS)
    keyed = frame.select(keys)
    return keyed.equals(keyed.sort(keys, maintain_order=True))


def _require_assume_sorted_contract(frame: pl.DataFrame) -> None:
    """Wave 43: fail-closed when ``assume_sorted=True`` but frame is unsorted.

    Empty frames and frames without ``event_time`` are allowed (prefix path is
    a no-op / empty). Nonempty frames with ``event_time`` that violate
    ``HISTORY_SORT_KEYS`` raise ``ValueError`` so a lying caller cannot take the
    binary-search prefix path. Cost is one ``under_history_sort_contract`` check.
    """
    if frame.is_empty() or "event_time" not in frame.columns:
        return
    if not under_history_sort_contract(frame):
        raise ValueError(
            "assume_sorted=True requires under_history_sort_contract(frame) "
            f"(HISTORY_SORT_KEYS={HISTORY_SORT_KEYS}); unsorted frame refused "
            "for history_prefix_upto — call sort_for_history or omit assume_sorted"
        )


def sort_for_history(frame: pl.DataFrame) -> pl.DataFrame:
    """Stable sort by HISTORY_SORT_KEYS so history_upto matches filter order."""
    missing = [c for c in HISTORY_SORT_KEYS if c not in frame.columns]
    if missing:
        raise ValueError(
            f"sort_for_history requires columns {HISTORY_SORT_KEYS}; missing {missing}"
        )
    return frame.sort(list(HISTORY_SORT_KEYS), maintain_order=True)


def build_event_time_day_index(frame: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Partition ``frame`` by ``event_time`` once for O(1) exact day slices.

    Correctness: ``slice_day(..., day_index=index)`` returns the same rows as
    ``frame.filter(pl.col("event_time") == asof)`` for matching keys (iso
    fingerprint via ``_date_keys``). Safe for PIT day extracts.

    Cumulative ``history_upto`` is order-preserving under the HISTORY_SORT_KEYS
    contract (Wave 10); ``history_for_calibration`` uses the day-index path only
    when that contract holds. Callers that loop many asofs (e.g.
    ``build_causal_weight_panel``) should build the index once and pass it
    through for ``slice_day`` / calibration history.
    """
    if frame.is_empty() or "event_time" not in frame.columns:
        return {}
    out: dict[str, pl.DataFrame] = {}
    for key_tup, group in frame.group_by("event_time", maintain_order=True):
        raw = key_tup[0] if isinstance(key_tup, tuple) else key_tup
        out[_date_keys([raw])[0]] = group
    return out


def slice_day(
    frame: pl.DataFrame,
    asof: datetime,
    *,
    day_index: dict[str, pl.DataFrame] | None = None,
) -> pl.DataFrame:
    """Exact ``event_time == asof`` day frame; optional prebuilt day index."""
    if day_index is not None:
        hit = day_index.get(_date_keys([asof])[0])
        if hit is not None:
            return hit
        return frame.head(0)
    return frame.filter(pl.col("event_time") == asof)


def history_prefix_upto(frame: pl.DataFrame, asof: datetime) -> pl.DataFrame:
    """Contiguous prefix ``event_time <= asof`` for HISTORY_SORT_KEYS-sorted frames.

    Caller must guarantee ``under_history_sort_contract(frame)`` (or have just
    run ``sort_for_history``). Under that contract this is order-identical to
    ``frame.filter(event_time <= asof)`` and much cheaper than day-index concat
    (Wave 12: concat was ~16× slower than filter on the SYNTHETIC lab panel).

    Uses O(log n) Series index probes + ``slice`` (no full-column ``to_list``).
    Wave 54: non-datetime ``asof`` fail-closed (TypeError).
    """
    if not isinstance(asof, datetime):
        raise TypeError("history_prefix_upto asof must be a datetime")
    if frame.is_empty() or "event_time" not in frame.columns:
        return frame.head(0)
    et = frame.get_column("event_time")
    n = frame.height
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if et[mid] <= asof:
            lo = mid + 1
        else:
            hi = mid
    if lo <= 0:
        return frame.head(0)
    if lo >= n:
        return frame
    return frame.slice(0, lo)


def history_upto(
    frame: pl.DataFrame,
    asof: datetime,
    *,
    day_index: dict[str, pl.DataFrame] | None = None,
    assume_sorted: bool = False,
) -> pl.DataFrame:
    """Cumulative ``event_time <= asof`` via prefix, day-index concat, or filter.

    When ``day_index`` is provided **and** the frame is under the HISTORY_SORT_KEYS
    contract (or ``assume_sorted=True`` after contract check), uses ``history_prefix_upto`` — order-
    identical to filter, without O(days) concat (Wave 12 fast path).

    When ``day_index`` is provided on an **unsorted** frame, concatenates day
    slices whose event_time compares ``<= asof`` using **datetime** comparison
    (not ISO string order). That path preserves the row multiset but may
    reorder vs ``filter`` — production hot paths must not use it unsorted.

    With no ``day_index``, falls back to ``frame.filter(event_time <= asof)``.
    Wave 43: ``assume_sorted=True`` on a nonempty unsorted frame with
    ``event_time`` raises ``ValueError`` (fail-closed; one contract check).
    Mixed naive/aware columns are rejected by Polars at frame build.
    """
    if day_index is None or not day_index:
        return frame.filter(pl.col("event_time") <= asof)
    # Wave 12/43: prefix under sort contract; assume_sorted fail-closed if lie.
    if assume_sorted:
        _require_assume_sorted_contract(frame)
        return history_prefix_upto(frame, asof)
    if under_history_sort_contract(frame):
        return history_prefix_upto(frame, asof)
    parts: list[pl.DataFrame] = []
    # Chronological walk of unique times present in the index groups.
    # Prefer datetime keys from each group rather than ISO string order.
    keyed: list[tuple[datetime, str, pl.DataFrame]] = []
    for key, group in day_index.items():
        if group.is_empty() or "event_time" not in group.columns:
            continue
        raw = group["event_time"][0]
        if not isinstance(raw, datetime):
            # Fall back to filter — refuse unsafe key-only compares.
            return frame.filter(pl.col("event_time") <= asof)
        keyed.append((raw, key, group))
    keyed.sort(key=lambda item: item[0])
    for raw, _key, group in keyed:
        if raw <= asof:
            parts.append(group)
        else:
            # Sorted by datetime — remaining days are strictly after asof.
            break
    if not parts:
        return frame.head(0)
    return pl.concat(parts, how="vertical")


def _distribution_label(frame: pl.DataFrame, config: AppConfig) -> str | None:
    label = config.train.distribution_target
    if label in frame.columns:
        return label
    for prefix in ("future_log_return", "future_idio_return", "future_return"):
        cands = [c for c in frame.columns if c.startswith(prefix)]
        if cands:
            return cands[0]
    return None


def _horizon_bars(label: str, config: AppConfig) -> int:
    tail = label.rsplit("_", 1)[-1]
    if tail.isdigit():
        return int(tail)
    return max(config.horizons.bars)


def _horizon_name(bars: int, config: AppConfig) -> str:
    for b, name in zip(config.horizons.bars, config.horizons.names, strict=True):
        if b == bars:
            return name
    return f"{bars}d"


def history_for_calibration(
    frame: pl.DataFrame,
    asof: datetime,
    horizon_bars: int,
    *,
    day_index: dict[str, pl.DataFrame] | None = None,
    assume_sorted: bool = False,
    event_times: list[datetime] | None = None,
) -> pl.DataFrame:
    """Rows whose forward labels are realized before ``asof``. Never the decision bar.

    Under the HISTORY_SORT_KEYS contract (or ``assume_sorted=True``), uses
    ``history_prefix_upto`` — order-identical to ``filter(event_time <= cutoff)``
    and faster than day-index concat (Wave 12). Wave 43:
    ``assume_sorted=True`` on an unsorted nonempty frame raises
    ``ValueError`` (fail-closed). Unsorted callers without the flag fall
    back to an explicit ``<=`` filter so they cannot silently reorder rows.

    ``day_index`` is accepted for API compatibility / optional unique-time
    derivation; cumulative history no longer concatenates day slices on the
    hot path. Pass ``event_times`` (unique sorted) from causal loops to skip
    per-asof ``unique().sort()``.
    """
    try:
        horizon_value = float(horizon_bars)
    except (TypeError, ValueError):
        horizon_value = float("nan")
    if not np.isfinite(horizon_value) or not horizon_value.is_integer() or horizon_value < 0:
        raise ValueError("horizon_bars must be a non-negative integer")
    horizon = int(horizon_value)
    if event_times is not None:
        times = event_times
        if any(not isinstance(value, datetime) for value in times):
            raise ValueError("event_times must contain datetime values")
        if any(left >= right for left, right in zip(times, times[1:], strict=False)):
            raise ValueError("event_times must be strictly increasing")
    elif day_index:
        # Unique event times from day groups (already partitioned).
        collected: list[datetime] = []
        for group in day_index.values():
            if group.is_empty() or "event_time" not in group.columns:
                continue
            raw = group["event_time"][0]
            if isinstance(raw, datetime):
                collected.append(raw)
        times = sorted(collected)
    else:
        times = frame["event_time"].unique().sort().to_list()
    if not times:
        return frame.head(0)
    idx = next((i for i, t in enumerate(times) if t >= asof), len(times))
    last = idx - horizon - 1
    if last < 0:
        last = idx - 1
    if last < 0:
        return frame.head(0)
    cutoff = times[last]
    if assume_sorted:
        _require_assume_sorted_contract(frame)
        return history_prefix_upto(frame, cutoff)
    if under_history_sort_contract(frame):
        return history_prefix_upto(frame, cutoff)
    return frame.filter(pl.col("event_time") <= cutoff)


def _align_col(frame: pl.DataFrame, dates: np.ndarray, ids: np.ndarray, name: str) -> Array | None:
    if name not in frame.columns:
        return None
    sub = frame.select(["event_time", "security_id", name]).drop_nulls()
    lookup = {
        (d, str(i)): float(v)
        for d, i, v in zip(
            _date_keys(sub["event_time"].to_numpy()),
            sub["security_id"].to_numpy(),
            sub[name].to_numpy().astype(float),
            strict=False,
        )
    }
    return np.array(
        [lookup.get((d, str(i)), np.nan) for d, i in zip(_date_keys(dates), ids, strict=True)],
        dtype=float,
    )


def _date_train_cal(dates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split on unique dates so a session is never half-train, half-cal."""
    keys = _date_keys(dates)
    uniq = sorted(set(keys))
    empty = np.zeros(len(keys), dtype=bool)
    if len(uniq) < 6:
        return empty, empty
    cut = min(max(int(0.7 * len(uniq)), 3), len(uniq) - 2)
    train_keys = set(uniq[:cut])
    cal_keys = set(uniq[cut:])
    tr = np.array([k in train_keys for k in keys], dtype=bool)
    cal = np.array([k in cal_keys for k in keys], dtype=bool)
    return tr, cal


def _decision_x(day: pl.DataFrame, feats: list[str]) -> Array:
    if not feats:
        return np.zeros((day.height, 1))
    use = [c for c in feats if c in day.columns]
    if not use:
        return np.zeros((day.height, len(feats)))
    arr = day.select(use).fill_null(0.0).to_numpy().astype(float)
    if arr.shape[1] == len(feats):
        return arr
    out = np.zeros((day.height, len(feats)))
    for j, col in enumerate(use):
        out[:, feats.index(col)] = arr[:, j]
    return out


def conformal_sets_asof(
    frame: pl.DataFrame,
    asof: datetime,
    config: AppConfig,
    *,
    alpha: float = INTERVAL_ALPHA,
    day_index: dict[str, pl.DataFrame] | None = None,
    assume_sorted: bool = False,
    event_times: list[datetime] | None = None,
) -> ForecastIntervals | None:
    """Fit scaled (t) bands on train dates; calibrate CQR on later dates; apply at ``asof``.

    Decision-bar ``y`` is never in the calibration window. Cross-section sets are
    predicted on the asof date only (dates are not stacked for the set update).
    Homoskedastic Gaussian is used only when vol_20 is missing.
    """
    # The label/horizon drive the fit, so resolve them before the cache lookup:
    # two configs with different distribution targets on the same frame/asof
    # must not share cached intervals.
    label = _distribution_label(frame, config)
    if label is None:
        return None
    horizon_bars = _horizon_bars(label, config)
    horizon = _horizon_name(horizon_bars, config)
    # Cache key uses asof + alpha + label + frame height fingerprint (Phase 18).
    # Never key by ``id(frame)``: Python may reuse an object id after a prior
    # panel is collected, which can return intervals fitted on a different
    # dataset.  A content fingerprint keeps the cache an optimization only.
    row_hash = tuple(int(value) for value in frame.hash_rows().to_list())
    cache_key = (
        str(asof),
        float(alpha),
        str(label),
        int(horizon_bars),
        int(frame.height),
        tuple(frame.columns),
        row_hash,
    )
    hit = _CONFORMAL_CACHE.get(cache_key)
    if hit is not None:
        return hit
    hist = history_for_calibration(
        frame,
        asof,
        horizon_bars,
        day_index=day_index,
        assume_sorted=assume_sorted,
        event_times=event_times,
    )
    if hist.is_empty():
        return None
    x, y, dates, feats, ids = design_matrix(hist, label)
    if x.shape[0] < 24:
        return None
    tr, cal = _date_train_cal(dates)
    if int(tr.sum()) < 8 or int(cal.sum()) < 8:
        return None
    taus = [alpha / 2.0, 1.0 - alpha / 2.0]
    gauss: ScaledGaussianDistribution | ScaledStudentTDistribution | GaussianDistribution
    vol_tr = _align_col(hist, dates[tr], ids[tr], "vol_20")
    vol_cal = _align_col(hist, dates[cal], ids[cal], "vol_20")
    if (
        vol_tr is not None
        and vol_cal is not None
        and bool(np.isfinite(vol_tr).any())
        and bool(np.isfinite(vol_cal).any())
    ):
        med_tr = float(np.nanmedian(vol_tr[np.isfinite(vol_tr)]))
        scale_tr = np.where(np.isfinite(vol_tr), vol_tr, med_tr)
        med = float(np.nanmedian(vol_cal[np.isfinite(vol_cal)]))
        scale_cal = np.where(np.isfinite(vol_cal), vol_cal, med)
        # Wave 7: re-select family on current cal; reuse train fit when
        # (train-primary fp, family) hits. Cal growth can change family without
        # forcing a train MLE when the family is unchanged. Student-t MLE is hot.
        _, gauss, _wrap_meta = resolve_wrappee_reselect_cached(
            taus,
            y[tr],
            scale_tr,
            y[cal],
            scale_cal,
            train_date_keys=tuple(sorted(set(_date_keys(dates[tr])))),
            cal_date_keys=tuple(sorted(set(_date_keys(dates[cal])))),
            alpha=float(alpha),
            label=str(label),
            nominal_coverage=1.0 - alpha,
        )
        _ = _wrap_meta
        q_cal = gauss.predict(scale_cal)
    else:
        gauss = GaussianDistribution(taus).fit(x[tr], y[tr])
        q_cal = gauss.predict(x[cal])
        scale_cal = None
        med = 0.01
    day = slice_day(frame, asof, day_index=day_index)
    if day.is_empty():
        return None
    day_ids = [str(v) for v in day["security_id"].to_list()]
    scaled = isinstance(gauss, ScaledGaussianDistribution | ScaledStudentTDistribution)
    if scaled and "vol_20" in day.columns:
        day_vol_pre = day["vol_20"].to_numpy().astype(float)
        day_vol_pre = np.where(np.isfinite(day_vol_pre), day_vol_pre, med)
        q_day = gauss.predict(day_vol_pre)
    else:
        q_day = gauss.predict(_decision_x(day, feats))
    if scale_cal is not None and bool(np.isfinite(scale_cal).any()):
        lab_cal, cuts = assign_terciles(scale_cal, prefix="vol_20")
        mondrian = MondrianCQR(alpha).calibrate(
            y[cal], q_cal[:, 0], q_cal[:, 1], lab_cal, scale_cal
        )
        if "vol_20" in day.columns:
            day_vol = day["vol_20"].to_numpy().astype(float)
        else:
            day_vol = np.full(day.height, med)
        day_vol = np.where(np.isfinite(day_vol), day_vol, med)
        lab_day, _ = assign_terciles(day_vol, cuts, prefix="vol_20")
        lo, hi = mondrian.predict_sets(q_day[:, 0], q_day[:, 1], lab_day, day_vol)
        method: IntervalMethod = "mondrian_cqr"
    else:
        cqr = SplitCQR(alpha).calibrate(y[cal], q_cal[:, 0], q_cal[:, 1])
        lo, hi = cqr.predict_sets(q_day[:, 0], q_day[:, 1])
        method = "split_cqr"
    result = ForecastIntervals(
        lower={sid: float(lo[i]) for i, sid in enumerate(day_ids)},
        upper={sid: float(hi[i]) for i, sid in enumerate(day_ids)},
        alpha=float(alpha),
        method=method,
        horizon=horizon,
        cal_event_times=tuple(sorted(set(dates[cal].tolist()), key=str)),
    )
    if len(_CONFORMAL_CACHE) > 256:
        _CONFORMAL_CACHE.clear()
    _CONFORMAL_CACHE[cache_key] = result
    return result


def _ranker_artifact_path(config: AppConfig) -> Path:
    return Path(config.data.root) / "metadata" / "ranker_ridge.joblib"


def _joblib_artifact_digest(path: Path) -> str:
    """SHA-256 of joblib artifact bytes. Mtime is not identity."""
    return hash_file(path)


def _load_ranker_cached(config: AppConfig):
    """Load the strongest available supervised ranker artifact.

    Candidate artifacts are searched in priority order; cache identity is
    artifact bytes, not mtime, so an in-place replacement that preserves mtime
    (Windows timestamp resolution, ``os.utime``, or a same-size rewrite) must
    not reuse a stale ranker for ``forecast_asof`` / ``optimize_asof``. Missing
    artifacts return None so the momentum heuristic remains the explicit
    no-model path.
    """
    root = Path(config.data.root) / "metadata"
    rank_path = next(
        (
            candidate
            for candidate in (
                root / "ranker_auto.joblib",
                root / "ranker_ensemble.joblib",
                root / "ranker_neural.joblib",
                root / "ranker_ridge.joblib",
                root / "ranker_elasticnet.joblib",
                root / "ranker_lambdarank.joblib",
                root / "ranker_xendcg.joblib",
                root / "ranker_xgboost.joblib",
                root / "ranker_lightgbm.joblib",
            )
            if candidate.exists()
        ),
        _ranker_artifact_path(config),
    )
    if not rank_path.exists():
        return None
    digest = _joblib_artifact_digest(rank_path)
    key = (str(rank_path.resolve()), digest)
    cached = _RANKER_CACHE.get(key)
    if cached is not None:
        return cached
    # The artifact name is a routing hint only: the serialized object may be
    # any ranking implementation. JoblibMixin.load retains checksum
    # verification while avoiding a Ridge-only type assertion.
    model = JoblibMixin.load(rank_path)
    _RANKER_CACHE[key] = model
    if len(_RANKER_CACHE) > 8:
        oldest = next(iter(_RANKER_CACHE))
        _RANKER_CACHE.pop(oldest, None)
    return model


def _load_rl_cached(config: AppConfig):
    """Load the strongest available persisted RL policy and feature contract."""
    root = Path(config.data.root) / "metadata"
    path = next(
        (
            candidate
            for candidate in (
                root / "rl_auto.joblib",
                root / "rl_policy_gradient.joblib",
                root / "rl_quantile_thompson.joblib",
                root / "rl_linucb.joblib",
                root / "rl_thompson.joblib",
            )
            if candidate.exists()
        ),
        None,
    )
    if path is None:
        return None
    key = (str(path.resolve()), path.stat().st_mtime)
    cached = _RL_POLICY_CACHE.get(key)
    if cached is not None:
        return cached
    artifact = load_joblib_artifact(path)
    if not isinstance(artifact, dict) or "policy" not in artifact or "features" not in artifact:
        raise ValueError("RL artifact is malformed: expected policy and features")
    policy = artifact["policy"]
    features = artifact["features"]
    if (
        not (hasattr(policy, "scores") or hasattr(policy, "predict"))
        or not isinstance(features, list)
        or not features
    ):
        raise ValueError("RL artifact has an invalid policy or feature contract")
    artifact_name = str(artifact.get("policy_name", path.stem.removeprefix("rl_"))).upper()
    loaded = (policy, [str(feature) for feature in features], f"RL_{artifact_name}")
    _RL_POLICY_CACHE[key] = loaded
    if len(_RL_POLICY_CACHE) > 8:
        oldest = next(iter(_RL_POLICY_CACHE))
        _RL_POLICY_CACHE.pop(oldest, None)
    return loaded


def _load_probability_calibrator(
    config: AppConfig, *, asof: datetime | None = None
) -> ProbabilityCalibrator:
    """Load an explicitly requested calibrator and validate its score identity."""
    name = str(config.fusion.probability_calibrator)
    path = Path(config.data.root) / "metadata" / f"calibrator_{name}.joblib"
    if name == "auto":
        path = Path(config.data.root) / "metadata" / "calibrator_auto.joblib"
    if not path.is_file():
        raise ValueError(f"probability calibration is enabled but artifact is missing: {path}")
    calibrator = JoblibMixin.load(path)
    if not isinstance(calibrator, ProbabilityCalibrator) or not calibrator.fitted:
        raise ValueError("probability calibrator artifact is invalid or unfitted")
    if calibrator.score_feature != "cs_pct_mom_20":
        raise ValueError("probability calibrator score identity mismatch: expected cs_pct_mom_20")
    expected_label = config.fusion.probability_calibration_label
    if expected_label is not None and calibrator.label != expected_label:
        raise ValueError("probability calibrator label identity mismatch")
    expected_horizon = config.fusion.probability_calibration_horizon
    if expected_horizon is not None and getattr(calibrator, "horizon", None) != expected_horizon:
        raise ValueError("probability calibrator horizon identity mismatch")
    for window_field in ("fit_start", "fit_end", "oos_start", "oos_end"):
        value = getattr(calibrator, window_field, None)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"probability calibrator {window_field} is missing")
    max_age = config.fusion.probability_calibration_max_age_days
    if asof is not None and max_age is not None:
        try:
            fit_end = datetime.fromisoformat(str(calibrator.fit_end).replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("probability calibrator fit_end is not parseable") from exc
        left = fit_end.date()
        right = asof.date()
        if (right - left).days < 0 or (right - left).days > max_age:
            raise ValueError("probability calibrator is stale for forecast asof")
    return calibrator


def _paper_challenger_stamp(
    config: AppConfig,
    x: NDArray[np.float64],
    ridge_scores: NDArray[np.float64],
    dates: NDArray[np.float64] | None = None,
    ids: NDArray[np.float64] | None = None,
) -> tuple[str, dict[str, float | str]]:
    """List fitted paper rankers. Spearman vs ridge only. Never blended into alpha."""
    import joblib
    from scipy.stats import spearmanr

    from quant_fund.pipeline.train import _predict_ranker

    present: list[str] = []
    extra: dict[str, float | str] = {"paper_challengers_blend": 0.0}
    root = Path(config.data.root) / "metadata"
    for name in config.train.paper_rankers:
        path = root / f"ranker_{name}.joblib"
        if not path.is_file():
            continue
        try:
            model = joblib.load(path)
            pred = np.asarray(_predict_ranker(model, name, x, dates, ids), dtype=float)
        except (TypeError, ValueError, OSError, AttributeError):
            continue
        present.append(name)
        if pred.shape[0] == ridge_scores.shape[0] and pred.size >= 4:
            rho = spearmanr(ridge_scores, pred, nan_policy="omit").correlation
            if rho is not None and np.isfinite(rho):
                extra[f"paper_{name}_spearman_vs_ridge"] = float(rho)
    extra["paper_challengers"] = ",".join(present) if present else "none"
    return f"paper_challengers={extra['paper_challengers']}", extra


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
    membership = pl.read_parquet(path)
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


@dataclass(frozen=True)
class OptimizerCovarianceEstimate:
    """Named trailing covariance used by ``optimize_asof`` and ``/risk/portfolio``."""

    sigma: Array
    security_ids: list[str]
    estimator: str
    covariance_object: str
    spec: str
    n_obs: int
    market_overlay: str | None
    overlay: GarchMarketForecast | RealizedGarchMarketForecast | None
    params: dict[str, float | str]
    unmeasured_reason: str | None = None
    fallback_reason: str | None = None


def _trailing_return_matrix(hist: pl.DataFrame, ids: list[str]) -> tuple[list[str], Array]:
    if "ret_1" not in hist.columns or hist.is_empty() or not ids:
        return [], np.empty((0, 0), dtype=float)
    wide = (
        hist.select(["event_time", "security_id", "ret_1"])
        .sort(["event_time", "security_id"])
        .pivot(on="security_id", index="event_time", values="ret_1")
        .sort("event_time")
    )
    cols = [sid for sid in ids if sid in wide.columns]
    if not cols:
        return [], np.empty((0, 0), dtype=float)
    return cols, np.asarray(wide.select(cols).to_numpy(), dtype=float)


def _named_dcc_optimizer_estimate(
    estimator: str,
    mat: Array,
    cols: list[str],
    finite_rows: int,
    psd_tol: float,
) -> OptimizerCovarianceEstimate:
    """Fit a named DCC optimizer path. Look up fitters at call time.

    Patches of ``dcc_gaussian`` / ``dcc_student_t`` / ``adcc`` / ``ccc`` /
    ``agdcc`` / ``agdcc_full`` on this module must still bind. A family or
    covariance-object mismatch fails closed so unrestricted AG-DCC cannot
    silently size as diagonal AG-DCC, scalar ADCC, Gaussian DCC,
    Student-t DCC, or CCC, and diagonal AG-DCC cannot silently size as
    those other specs either.
    """
    fitter: Callable[[Array], tuple[Array, dict[str, float | str]]]
    if estimator == DCC_FAMILY_GAUSSIAN:
        fitter = dcc_gaussian
        default_spec = DCC_SPEC_ENGLE_2002
    elif estimator == DCC_FAMILY_STUDENT_T:
        fitter = dcc_student_t
        default_spec = DCC_SPEC_STUDENT_T
    elif estimator == DCC_FAMILY_ADCC:
        fitter = adcc
        default_spec = DCC_SPEC_ADCC
    elif estimator == DCC_FAMILY_CCC:
        fitter = ccc
        default_spec = DCC_SPEC_CCC
    elif estimator == DCC_FAMILY_AGDCC:
        fitter = agdcc
        default_spec = DCC_SPEC_AGDCC
    elif estimator == DCC_FAMILY_AGDCC_FULL:
        fitter = agdcc_full
        default_spec = DCC_SPEC_AGDCC_FULL
    else:
        raise ValueError(f"unknown_optimizer_dcc_family:{estimator}")
    prefix = f"optimizer_covariance_failed:{estimator}"
    if len(cols) < 2:
        raise ValueError(f"{prefix}:fewer_than_two_securities")
    if finite_rows < DCC_STAGE1_MIN_OBS:
        raise ValueError(f"{prefix}:insufficient_finite_rows:{finite_rows}")
    try:
        dcc_trailing_complete_window(mat, min_rows=DCC_STAGE1_MIN_OBS)
    except ValueError as exc:
        raise ValueError(f"{prefix}:{exc}") from exc
    try:
        sigma, params = fitter(mat)
    except ValueError as exc:
        raise ValueError(f"{prefix}:{exc}") from exc
    sigma, _ = repair_psd(sigma, psd_tol)
    family = str(params.get("family", ""))
    if family != estimator:
        raise ValueError(f"{prefix}:unexpected_family:{family}")
    object_name = str(params.get("covariance_object", DCC_COVARIANCE_OBJECT_ONE_STEP))
    spec_name = str(params.get("spec", default_spec))
    if object_name != DCC_COVARIANCE_OBJECT_ONE_STEP:
        raise ValueError(f"{prefix}:unexpected_covariance_object:{object_name}")
    return OptimizerCovarianceEstimate(
        sigma=np.asarray(sigma, dtype=float),
        security_ids=cols,
        estimator=estimator,
        covariance_object=object_name,
        spec=spec_name,
        n_obs=int(float(params.get("n_obs", finite_rows))),
        market_overlay=None,
        overlay=None,
        params=dict(params),
    )


def _named_ewma_optimizer_estimate(
    mat: Array,
    cols: list[str],
    finite_rows: int,
    psd_tol: float,
    lam: float,
) -> OptimizerCovarianceEstimate:
    """Fit named RiskMetrics EWMA. Look up ``ewma`` at call time so patches bind.

    A family or covariance-object mismatch fails closed so EWMA cannot
    silently size as Ledoit–Wolf or DCC. The matrix is not GARCH-overlaid.
    """
    prefix = f"optimizer_covariance_failed:{OPTIMIZER_COVARIANCE_EWMA}"
    if len(cols) < 2:
        raise ValueError(f"{prefix}:fewer_than_two_securities")
    if finite_rows < EWMA_MIN_OBS:
        raise ValueError(f"{prefix}:insufficient_finite_rows:{finite_rows}")
    try:
        dcc_trailing_complete_window(mat, min_rows=EWMA_MIN_OBS)
    except ValueError as exc:
        raise ValueError(f"{prefix}:{exc}") from exc
    try:
        sigma, params = ewma(mat, lam=lam)
    except ValueError as exc:
        raise ValueError(f"{prefix}:{exc}") from exc
    sigma, _ = repair_psd(sigma, psd_tol)
    family = str(params.get("family", ""))
    if family != OPTIMIZER_COVARIANCE_EWMA:
        raise ValueError(f"{prefix}:unexpected_family:{family}")
    object_name = str(params.get("covariance_object", DCC_COVARIANCE_OBJECT_ONE_STEP))
    spec_name = str(params.get("spec", EWMA_SPEC_RISKMETRICS))
    if object_name != DCC_COVARIANCE_OBJECT_ONE_STEP:
        raise ValueError(f"{prefix}:unexpected_covariance_object:{object_name}")
    return OptimizerCovarianceEstimate(
        sigma=np.asarray(sigma, dtype=float),
        security_ids=cols,
        estimator=OPTIMIZER_COVARIANCE_EWMA,
        covariance_object=object_name,
        spec=spec_name,
        n_obs=int(float(params.get("n_obs", finite_rows))),
        market_overlay=None,
        overlay=None,
        params=dict(params),
    )


def _named_sample_optimizer_estimate(
    config: AppConfig,
    frame: pl.DataFrame,
    asof: datetime,
    mat: Array,
    cols: list[str],
    finite_rows: int,
    psd_tol: float,
) -> OptimizerCovarianceEstimate:
    """Fit named unbiased sample covariance. Look up ``sample`` at call time.

    A family or covariance-object mismatch fails closed so sample cannot
    silently size as Ledoit–Wolf, OAS, EWMA, or DCC. The matrix is
    GARCH/RGARCH overlay-scaled like trailing Ledoit–Wolf. When T>N the
    estimator stays sample rather than switching to Ledoit–Wolf.
    """
    prefix = f"optimizer_covariance_failed:{OPTIMIZER_COVARIANCE_SAMPLE}"
    if len(cols) < 2:
        raise ValueError(f"{prefix}:fewer_than_two_securities")
    if finite_rows < 2:
        raise ValueError(f"{prefix}:insufficient_finite_rows:{finite_rows}")
    try:
        sigma, params = sample(mat)
    except ValueError as exc:
        raise ValueError(f"{prefix}:{exc}") from exc
    sigma, _ = repair_psd(sigma, psd_tol)
    family = str(params.get("family", ""))
    if family != OPTIMIZER_COVARIANCE_SAMPLE:
        raise ValueError(f"{prefix}:unexpected_family:{family}")
    object_name = str(params.get("covariance_object", OPTIMIZER_COVARIANCE_OBJECT_TRAILING))
    spec_name = str(params.get("spec", SAMPLE_SPEC_UNBIASED))
    if object_name != OPTIMIZER_COVARIANCE_OBJECT_TRAILING:
        raise ValueError(f"{prefix}:unexpected_covariance_object:{object_name}")
    sigma, overlay, overlay_kind = apply_market_variance_overlay_to_covariance(
        config, frame, asof, sigma
    )
    return OptimizerCovarianceEstimate(
        sigma=np.asarray(sigma, dtype=float),
        security_ids=cols,
        estimator=OPTIMIZER_COVARIANCE_SAMPLE,
        covariance_object=object_name,
        spec=spec_name,
        n_obs=int(float(params.get("n_obs", finite_rows))),
        market_overlay=overlay_kind,
        overlay=overlay,
        params=dict(params),
    )


def _named_oas_optimizer_estimate(
    config: AppConfig,
    frame: pl.DataFrame,
    asof: datetime,
    mat: Array,
    cols: list[str],
    finite_rows: int,
    psd_tol: float,
) -> OptimizerCovarianceEstimate:
    """Fit named Chen OAS. Look up ``oas`` at call time so patches bind.

    A family or covariance-object mismatch fails closed so OAS cannot
    silently size as Ledoit–Wolf, sample, EWMA, or DCC. The matrix is
    GARCH/RGARCH overlay-scaled like trailing Ledoit–Wolf. When T<=N the
    estimator stays OAS rather than switching to sample.
    """
    prefix = f"optimizer_covariance_failed:{OPTIMIZER_COVARIANCE_OAS}"
    if len(cols) < 2:
        raise ValueError(f"{prefix}:fewer_than_two_securities")
    if finite_rows < 2:
        raise ValueError(f"{prefix}:insufficient_finite_rows:{finite_rows}")
    try:
        sigma, params = oas(mat)
    except ValueError as exc:
        raise ValueError(f"{prefix}:{exc}") from exc
    sigma, _ = repair_psd(sigma, psd_tol)
    family = str(params.get("family", ""))
    if family != OPTIMIZER_COVARIANCE_OAS:
        raise ValueError(f"{prefix}:unexpected_family:{family}")
    object_name = str(params.get("covariance_object", OPTIMIZER_COVARIANCE_OBJECT_TRAILING))
    spec_name = str(params.get("spec", OAS_SPEC_CHEN_2010))
    if object_name != OPTIMIZER_COVARIANCE_OBJECT_TRAILING:
        raise ValueError(f"{prefix}:unexpected_covariance_object:{object_name}")
    sigma, overlay, overlay_kind = apply_market_variance_overlay_to_covariance(
        config, frame, asof, sigma
    )
    return OptimizerCovarianceEstimate(
        sigma=np.asarray(sigma, dtype=float),
        security_ids=cols,
        estimator=OPTIMIZER_COVARIANCE_OAS,
        covariance_object=object_name,
        spec=spec_name,
        n_obs=int(float(params.get("n_obs", finite_rows))),
        market_overlay=overlay_kind,
        overlay=overlay,
        params=dict(params),
    )


def _named_ledoit_wolf_nonlinear_optimizer_estimate(
    config: AppConfig,
    frame: pl.DataFrame,
    asof: datetime,
    mat: Array,
    cols: list[str],
    finite_rows: int,
    psd_tol: float,
) -> OptimizerCovarianceEstimate:
    """Fit named analytical nonlinear Ledoit-Wolf. Look up at call time.

    A family or covariance-object mismatch fails closed so 2020 analytical
    nonlinear shrinkage cannot silently size as 2004 linear Ledoit-Wolf,
    OAS, sample, EWMA, or DCC. The matrix is GARCH/RGARCH overlay-scaled
    like trailing Ledoit-Wolf. When T<=N the estimator stays nonlinear
    rather than switching to sample or 2004 linear shrinkage.
    """
    prefix = f"optimizer_covariance_failed:{OPTIMIZER_COVARIANCE_LEDOIT_WOLF_NONLINEAR}"
    if len(cols) < 2:
        raise ValueError(f"{prefix}:fewer_than_two_securities")
    if finite_rows < NLSHRINK_MIN_OBS:
        raise ValueError(f"{prefix}:insufficient_finite_rows:{finite_rows}")
    try:
        sigma, params = ledoit_wolf_nonlinear(mat)
    except ValueError as exc:
        raise ValueError(f"{prefix}:{exc}") from exc
    sigma, _ = repair_psd(sigma, psd_tol)
    family = str(params.get("family", ""))
    if family != OPTIMIZER_COVARIANCE_LEDOIT_WOLF_NONLINEAR:
        raise ValueError(f"{prefix}:unexpected_family:{family}")
    object_name = str(params.get("covariance_object", OPTIMIZER_COVARIANCE_OBJECT_TRAILING))
    spec_name = str(params.get("spec", OPTIMIZER_COVARIANCE_SPEC_LEDOIT_WOLF_NONLINEAR))
    if object_name != OPTIMIZER_COVARIANCE_OBJECT_TRAILING:
        raise ValueError(f"{prefix}:unexpected_covariance_object:{object_name}")
    if spec_name == OPTIMIZER_COVARIANCE_SPEC_LEDOIT_WOLF:
        raise ValueError(f"{prefix}:unexpected_spec:{spec_name}")
    sigma, overlay, overlay_kind = apply_market_variance_overlay_to_covariance(
        config, frame, asof, sigma
    )
    return OptimizerCovarianceEstimate(
        sigma=np.asarray(sigma, dtype=float),
        security_ids=cols,
        estimator=OPTIMIZER_COVARIANCE_LEDOIT_WOLF_NONLINEAR,
        covariance_object=object_name,
        spec=spec_name,
        n_obs=int(float(params.get("n_obs", finite_rows))),
        market_overlay=overlay_kind,
        overlay=overlay,
        params=dict(params),
    )


def _unmeasured_optimizer_covariance(reason: str, ids: list[str]) -> OptimizerCovarianceEstimate:
    n = len(ids)
    sigma = np.empty((0, 0), dtype=float) if n == 0 else np.diag(np.ones(n) * 0.02**2)
    return OptimizerCovarianceEstimate(
        sigma=sigma,
        security_ids=list(ids),
        estimator=OPTIMIZER_COVARIANCE_HOMOSKEDASTIC_PROXY,
        covariance_object=OPTIMIZER_COVARIANCE_OBJECT_DIAGONAL_PROXY,
        spec=OPTIMIZER_COVARIANCE_SPEC_DIAGONAL_PROXY,
        n_obs=0,
        market_overlay=None,
        overlay=None,
        params={},
        unmeasured_reason=reason,
        fallback_reason=reason,
    )


def estimate_optimizer_covariance_asof(
    config: AppConfig,
    frame: pl.DataFrame,
    asof: datetime,
    ids: list[str],
    hist: pl.DataFrame,
) -> OptimizerCovarianceEstimate:
    r"""Estimate the named optimizer covariance on PIT-filtered trailing returns.

    ``optimizer.covariance=ledoit_wolf`` (default) keeps trailing Ledoit-Wolf
    2004 plus the GARCH/RGARCH overlay and stays Ledoit-Wolf when T<=N
    rather than silently switching to sample. ``dcc_gaussian``,
    ``dcc_student_t``, ``adcc``, ``ccc``, ``agdcc``, ``agdcc_full``, and
    ``ewma`` return one-step-ahead H_{t+1} and do **not** apply that overlay.
    Named ``oas`` returns trailing Chen OAS plus the overlay and must not
    silently size as Ledoit-Wolf, sample, EWMA, or DCC; when T<=N it stays
    OAS rather than switching to sample. Named ``ledoit_wolf_nonlinear``
    returns trailing analytical 2020 nonlinear shrinkage plus the overlay
    and must not silently size as 2004 linear Ledoit-Wolf, OAS, sample,
    EWMA, or DCC; when T<=N it stays nonlinear rather than switching to
    sample or 2004. Named ``sample`` returns trailing unbiased sample
    covariance plus the overlay and must not silently size as Ledoit-Wolf,
    OAS, EWMA, or DCC; when T>N it stays sample rather than switching to
    Ledoit-Wolf. Sequential one-step samples use the trailing contiguous
    complete-case window; holes are not concatenated and an incomplete
    asof row fails closed rather than dropping \(r_t\) / \(z_t\).
    OAS, named sample, named nonlinear Ledoit-Wolf, and default
    Ledoit-Wolf listwise-delete. Those named paths are distinct and must
    not substitute for each other or for Ledoit-Wolf 2004. A failed or
    short named fit fails closed rather than silently substituting
    Ledoit-Wolf. Named unrestricted AG-DCC must not silently size as
    diagonal AG-DCC. Named diagonal AG-DCC must not silently size as
    scalar ADCC. Named CCC must not silently size as Gaussian DCC or
    Ledoit-Wolf. Factor stays unwired. This does not invent high-frequency
    RV or wire factor covariance.
    """
    estimator = require_implemented_optimizer_covariance(config.optimizer.covariance)
    cols, mat = _trailing_return_matrix(hist, ids)
    finite_rows = int(np.isfinite(mat).all(axis=1).sum()) if mat.size else 0
    if estimator == OPTIMIZER_COVARIANCE_SAMPLE:
        return _named_sample_optimizer_estimate(
            config,
            frame,
            asof,
            mat,
            cols,
            finite_rows,
            config.train.psd_eigen_tol,
        )
    if estimator == OPTIMIZER_COVARIANCE_OAS:
        return _named_oas_optimizer_estimate(
            config,
            frame,
            asof,
            mat,
            cols,
            finite_rows,
            config.train.psd_eigen_tol,
        )
    if estimator == OPTIMIZER_COVARIANCE_LEDOIT_WOLF_NONLINEAR:
        return _named_ledoit_wolf_nonlinear_optimizer_estimate(
            config,
            frame,
            asof,
            mat,
            cols,
            finite_rows,
            config.train.psd_eigen_tol,
        )
    if estimator == OPTIMIZER_COVARIANCE_EWMA:
        return _named_ewma_optimizer_estimate(
            mat,
            cols,
            finite_rows,
            config.train.psd_eigen_tol,
            float(config.features.ewma_lambda),
        )
    if estimator in IMPLEMENTED_OPTIMIZER_DCC_FAMILIES:
        return _named_dcc_optimizer_estimate(
            estimator,
            mat,
            cols,
            finite_rows,
            config.train.psd_eigen_tol,
        )

    if "ret_1" not in hist.columns:
        return _unmeasured_optimizer_covariance("no_ret_1", ids)
    if len(cols) < 2:
        return _unmeasured_optimizer_covariance("fewer_than_two_securities", ids)
    if finite_rows < 2:
        return _unmeasured_optimizer_covariance("insufficient_finite_rows", ids)
    try:
        sigma, params = ledoit_wolf(mat)
    except ValueError:
        log.warning("covariance_fallback", reason="estimation_failed", n_ids=len(ids))
        return _unmeasured_optimizer_covariance("estimation_failed", ids)
    family = str(params.get("family", ""))
    prefix = f"optimizer_covariance_failed:{OPTIMIZER_COVARIANCE_LEDOIT_WOLF}"
    if family != OPTIMIZER_COVARIANCE_LEDOIT_WOLF:
        raise ValueError(f"{prefix}:unexpected_family:{family}")
    object_name = str(params.get("covariance_object", OPTIMIZER_COVARIANCE_OBJECT_TRAILING))
    spec_name = str(params.get("spec", OPTIMIZER_COVARIANCE_SPEC_LEDOIT_WOLF))
    if object_name != OPTIMIZER_COVARIANCE_OBJECT_TRAILING:
        raise ValueError(f"{prefix}:unexpected_covariance_object:{object_name}")
    sigma, _ = repair_psd(sigma, config.train.psd_eigen_tol)
    sigma, overlay, overlay_kind = apply_market_variance_overlay_to_covariance(
        config, frame, asof, sigma
    )
    return OptimizerCovarianceEstimate(
        sigma=np.asarray(sigma, dtype=float),
        security_ids=cols,
        estimator=OPTIMIZER_COVARIANCE_LEDOIT_WOLF,
        covariance_object=object_name,
        spec=spec_name,
        n_obs=int(float(params.get("n_obs", finite_rows))),
        market_overlay=overlay_kind,
        overlay=overlay,
        params=dict(params),
    )


def market_risk_overlay_asof(
    config: AppConfig,
    frame: pl.DataFrame,
    asof: datetime,
) -> tuple[float | None, str | None]:
    """Causal date-level market sigma for paper/backtest ``check_order``."""
    overlay, source = resolve_market_variance_overlay_asof(config, frame, asof)
    if overlay is None:
        return None, None
    return float(overlay.sigma), source


def _market_overlay_note(overlay_kind: str) -> str:
    if overlay_kind == MARKET_RISK_OVERLAY_REALIZED_GARCH:
        return "realized_garch_market_cross_section"
    return "garch_market_cross_section"


def _market_overlay_diagnostics(
    overlay: GarchMarketForecast | RealizedGarchMarketForecast,
    overlay_kind: str,
) -> dict[str, float | str]:
    diagnostics: dict[str, float | str] = {
        "garch_market_sigma": float(overlay.sigma),
        "garch_market_variance": float(overlay.variance),
        "garch_cumulative_variance": float(overlay.cumulative_variance),
        "garch_horizon": float(overlay.horizon),
        "garch_n_obs": float(overlay.n_obs),
        "garch_series_scope": overlay.series_scope,
        "garch_fit_status": overlay.fit_status,
        "market_risk_overlay": overlay_kind,
    }
    if overlay_kind == MARKET_RISK_OVERLAY_REALIZED_GARCH:
        diagnostics["realized_measure"] = REALIZED_GARCH_MEASURE
        diagnostics["intraday_realized_variance"] = "false"
        diagnostics["garch_variance_family"] = REALIZED_GARCH_FAMILY
    return diagnostics


def _panel_cached(config: AppConfig) -> pl.DataFrame:
    """Thin wrapper — dataset.panel already caches; keep call site explicit."""
    return panel(config)


def _count_robinhood_plus_ok(
    rh_map: dict[str, RobinhoodPlusNameForecast],
    security_ids: list[str],
) -> int:
    n_ok = 0
    for sid in security_ids:
        hit = rh_map.get(sid)
        if hit is None or hit.forecast.status != STATUS_OK:
            continue
        mu = float(hit.forecast.expected_returns.get("5d", hit.forecast.rank_score))
        if np.isfinite(mu):
            n_ok += 1
    return n_ok


def _robinhood_plus_asof(
    frame: pl.DataFrame,
    asof: datetime,
    config: AppConfig,
    security_ids: list[str],
) -> dict[str, RobinhoodPlusNameForecast]:
    """Causal robinhood+ K-line forecasts. Torch backend is local-weights only."""
    cfg = config.robinhood_plus
    if not cfg.enabled:
        return {}
    if cfg.backend.value == "torch":
        from quant_fund.models.robinhood_plus.torch_backend import forecast_cross_section_torch

        return forecast_cross_section_torch(frame, asof, config, security_ids)
    if not kline_columns_present(frame.columns):
        return {}
    return forecast_robinhood_plus_cross_section(
        frame,
        asof,
        lookback=cfg.lookback,
        pred_len=cfg.pred_len,
        sample_count=cfg.sample_count,
        s1_bits=cfg.s1_bits,
        s2_bits=cfg.s2_bits,
        clip=cfg.clip,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_context=cfg.max_context,
        seed=config.train.random_seed,
        decoder=cfg.decoder.value,
        security_ids=security_ids,
        quantile_levels=tuple(float(q) for q in config.quantiles.levels),
        horizons=tuple(int(h) for h in config.horizons.bars),
    )


def forecast_asof(
    config: AppConfig,
    asof: datetime | None = None,
    *,
    interval_alpha: float = INTERVAL_ALPHA,
    frame: pl.DataFrame | None = None,
    day_index: dict[str, pl.DataFrame] | None = None,
    assume_sorted: bool = False,
    event_times: list[datetime] | None = None,
) -> MarketState:
    df = frame if frame is not None else _panel_cached(config)
    # day_index is optional: build once in causal loops and pass through.
    # Single-asof calls keep a plain filter (building a full index is slower).
    if asof is None:
        asof = latest_decision(df)
    day = slice_day(df, asof, day_index=day_index)
    if day.is_empty():
        # An explicitly requested decision date with no panel rows must fail
        # closed: silently substituting the latest date would persist
        # end-of-sample weights for the wrong as-of (look-ahead).
        raise ValueError(
            f"no panel rows for requested asof={asof!r} (refusing latest-date fallback)"
        )
    model = _load_ranker_cached(config)
    ranker_features = getattr(model, "features", None) if model is not None else None
    if ranker_features is not None:
        if not isinstance(ranker_features, list) or not ranker_features:
            raise ValueError("ranker artifact has an invalid feature contract")
        missing = [feature for feature in ranker_features if feature not in day.columns]
        if missing:
            raise ValueError(f"ranker artifact feature contract missing columns: {missing}")
        feats = [str(feature) for feature in ranker_features]
    else:
        feats = available_features(day.columns)
    x = (
        day.select(feats).fill_null(0.0).to_numpy().astype(float)
        if feats
        else np.zeros((day.height, 1))
    )
    notes: list[str] = []
    if model is not None:
        # A stale joblib / feature-set mismatch must surface: silently degrading
        # to the momentum heuristic would masquerade a fabricated alpha as model
        # output. The heuristic is used only when no ranker is loaded at all.
        scores = model.predict(x)
    else:
        rl_artifact = _load_rl_cached(config)
        if rl_artifact is not None:
            policy, rl_features, policy_note = rl_artifact
            missing = [feature for feature in rl_features if feature not in day.columns]
            if missing:
                raise ValueError(f"RL artifact feature contract missing columns: {missing}")
            rl_x = day.select(rl_features).fill_null(0.0).to_numpy().astype(float)
            score_fn = getattr(policy, "scores", None) or getattr(policy, "predict", None)
            if not callable(score_fn):
                raise ValueError("RL artifact policy does not expose a callable scorer")
            scores = score_fn(rl_x)
            notes = [policy_note]
        else:
            scores = (
                day["cs_pct_mom_20"].fill_null(0.5).to_numpy().astype(float)
                if "cs_pct_mom_20" in day.columns
                else np.zeros(day.height)
            )
    # percentile ranks within the day
    order = scores.argsort().argsort()
    pct = (order + 0.5) / max(len(scores), 1)
    if config.fusion.apply_probability_calibration:
        calibrator = _load_probability_calibrator(config, asof=asof)
        if "cs_pct_mom_20" not in day.columns:
            raise ValueError("calibration requires cs_pct_mom_20 in the forecast panel")
        raw_probability = day["cs_pct_mom_20"].fill_null(0.5).to_numpy().astype(float)
        pct = np.asarray(calibrator.predict(raw_probability), dtype=float)
        notes.append(f"CALIBRATED_{calibrator.method.upper()}")
    vol = (
        day["vol_20"].fill_null(0.02).to_numpy().astype(float)
        if "vol_20" in day.columns
        else np.full(day.height, 0.02)
    )
    alpha = (pct - 0.5) * 2.0 * config.fusion.alpha_scale
    conf = np.clip(0.5 + np.abs(pct - 0.5), 0.2, 1.0)
    security_ids = [str(row["security_id"]) for row in day.iter_rows(named=True)]
    blend = float(config.robinhood_plus.blend_weight) if config.robinhood_plus.enabled else 0.0
    # blend_weight=0: keep the engine as a stamped challenger and do not size
    # the book. Still emit n_ok / n_fallback so a missing OHLC window cannot
    # masquerade as a Kronos path when the engine *is* applied.
    rh_map: dict[str, RobinhoodPlusNameForecast] = {}
    if config.robinhood_plus.enabled and blend > 0.0:
        rh_map = _robinhood_plus_asof(df, asof, config, security_ids)
    rh_ok = _count_robinhood_plus_ok(rh_map, security_ids)
    n_fallback = int(len(security_ids) - rh_ok)
    if rh_map and blend > 0.0:
        for i, sid in enumerate(security_ids):
            hit = rh_map.get(sid)
            if hit is None or hit.forecast.status != STATUS_OK:
                continue
            mu = float(hit.forecast.expected_returns.get("5d", hit.forecast.rank_score))
            if not np.isfinite(mu):
                continue
            scores[i] = (1.0 - blend) * float(scores[i]) + blend * float(hit.forecast.rank_score)
            alpha[i] = (1.0 - blend) * float(alpha[i]) + blend * mu
            conf[i] = (1.0 - blend) * float(conf[i]) + blend * float(hit.forecast.confidence)
        order = scores.argsort().argsort()
        pct = (order + 0.5) / max(len(scores), 1)
        conf = np.clip(conf, 0.2, 1.0)
    regime = np.ones(day.height)
    tail = np.clip(vol * 0.1, 0, None)
    liq = np.zeros(day.height)
    fused = fuse_signals(alpha, conf, regime, vol, tail, liq, config.fusion)
    if config.data.source == "synthetic":
        notes.append("SYNTHETIC")
    notes.append(f"robinhood_plus_n_ok={rh_ok}")
    notes.append(f"robinhood_plus_n_fallback={n_fallback}")
    notes.append(f"robinhood_plus_blend_weight={blend}")
    paper_note, paper_diag = _paper_challenger_stamp(
        config,
        x,
        scores,
        dates=np.asarray([asof] * len(scores), dtype=object),
        ids=np.asarray(security_ids, dtype=object),
    )
    notes.append(paper_note)
    if config.robinhood_plus.enabled and blend <= 0.0:
        notes.append("robinhood_plus_challenger")
    if rh_ok and blend > 0.0:
        notes.append(ENGINE_NAME)
    overlay, overlay_kind = resolve_market_variance_overlay_asof(config, df, asof)
    if overlay is not None and overlay_kind is not None:
        notes.append(_market_overlay_note(overlay_kind))
    # Research-only weight smoke: skip conformal when fusion.skip_intervals is set.
    if bool(getattr(config.fusion, "skip_intervals", False)):
        intervals = None
    else:
        intervals = conformal_sets_asof(
            df,
            asof,
            config,
            alpha=interval_alpha,
            day_index=day_index,
            assume_sorted=assume_sorted,
            event_times=event_times,
        )
    forecasts = []
    for i, row in enumerate(day.iter_rows(named=True)):
        q05 = float(alpha[i] - 1.65 * vol[i])
        q50 = float(alpha[i])
        q95 = float(alpha[i] + 1.65 * vol[i])
        sid = str(row["security_id"])
        lo_map: dict[str, float] = {}
        hi_map: dict[str, float] = {}
        method: IntervalMethod | None = None
        i_alpha: float | None = None
        if intervals is not None and sid in intervals.lower:
            lo_map = {intervals.horizon: intervals.lower[sid]}
            hi_map = {intervals.horizon: intervals.upper[sid]}
            method = intervals.method
            i_alpha = intervals.alpha
        diagnostics: dict[str, float | str] = {
            "fused": float(fused[i]),
            "robinhood_plus_n_ok": float(rh_ok),
            "robinhood_plus_n_fallback": float(n_fallback),
            "robinhood_plus_blend_weight": float(blend),
            **paper_diag,
        }
        if overlay is not None and overlay_kind is not None:
            diagnostics.update(_market_overlay_diagnostics(overlay, overlay_kind))
        expected = {"5d": float(alpha[i])}
        q_map: dict[str, dict[float, float]] = {"5d": {0.05: q05, 0.5: q50, 0.95: q95}}
        p_map = {"5d": float(pct[i])}
        hit = rh_map.get(sid)
        if hit is not None and hit.forecast.status == STATUS_OK:
            diagnostics["core_engine"] = ENGINE_NAME
            diagnostics["robinhood_plus_status"] = hit.forecast.status
            diagnostics["robinhood_plus_model_version"] = MODEL_VERSION
            diagnostics["robinhood_plus_backend"] = str(
                hit.forecast.diagnostics.get("backend", "numpy")
            )
            diagnostics["robinhood_plus_decoder"] = str(hit.forecast.diagnostics.get("decoder", ""))
            if hit.forecast.expected_returns:
                expected = dict(hit.forecast.expected_returns)
                expected["5d"] = float(alpha[i])
            if hit.forecast.quantiles:
                q_map = {
                    hz: {float(level): float(val) for level, val in levels.items()}
                    for hz, levels in hit.forecast.quantiles.items()
                }
            if hit.forecast.probability_positive:
                p_map = {
                    hz: float(np.clip(val, 0.0, 1.0))
                    for hz, val in hit.forecast.probability_positive.items()
                }
        elif config.robinhood_plus.enabled:
            diagnostics["core_engine"] = "ridge"
            diagnostics["robinhood_plus_status"] = (
                hit.forecast.status
                if hit is not None
                else ("challenger" if blend <= 0.0 else "absent")
            )
        forecasts.append(
            AssetForecast(
                security_id=sid,
                symbol=row.get("symbol", sid),
                asof=asof,
                model_version="fusion.v1",
                expected_returns=expected,
                quantiles=q_map,
                probability_positive=p_map,
                alpha={"5d": float(alpha[i])},
                rank_score={"5d": float(scores[i])},
                rank_percentile={"5d": float(pct[i])},
                volatility={"5d": float(vol[i])},
                confidence={"5d": float(conf[i])},
                diagnostics=diagnostics,
                interval_lo=lo_map,
                interval_hi=hi_map,
                interval_alpha=i_alpha,
                interval_method=method,
            )
        )
    if overlay is None or overlay_kind is None:
        return MarketState(asof=asof, forecasts=forecasts, notes=notes)
    return MarketState(
        asof=asof,
        forecasts=forecasts,
        notes=notes,
        garch_market_sigma=float(overlay.sigma),
        garch_market_variance=float(overlay.variance),
        garch_cumulative_variance=float(overlay.cumulative_variance),
        garch_horizon=int(overlay.horizon),
        garch_series_scope=overlay.series_scope,
        garch_fit_status=overlay.fit_status,
        garch_n_obs=int(overlay.n_obs),
        market_risk_overlay=overlay_kind,
    )


def _alpha_vector(state: MarketState, config: AppConfig, *, use_fused: bool) -> Array:
    """Raw alpha or fused diagnostics used for mean-variance sizing."""
    if use_fused:
        return np.array(
            [float(f.diagnostics.get("fused", f.alpha.get("5d", 0.0))) for f in state.forecasts],
            dtype=float,
        )
    return np.array([float(f.alpha.get("5d", 0.0)) for f in state.forecasts], dtype=float)


def _align_w_prev(ids: list[str], w_prev: Array | dict[str, float] | None) -> Array:
    if w_prev is None:
        return np.zeros(len(ids), dtype=float)
    if isinstance(w_prev, dict):
        return np.array([float(w_prev.get(i, 0.0)) for i in ids], dtype=float)
    arr = np.asarray(w_prev, dtype=float).reshape(-1)
    if arr.size != len(ids):
        # Shape mismatches must raise, never silently reset the prior book to
        # flat (that would optimize from cash against a different universe).
        raise ValueError(
            f"w_prev length {arr.size} != n_ids {len(ids)}; "
            "pass a security_id-keyed dict when the universe changes"
        )
    return arr


def _apply_forecast_interval_caps(
    w: Array,
    state: MarketState,
    ids: list[str],
    config: AppConfig,
    *,
    w_prev: Array | None = None,
) -> Array:
    """Apply conformal name caps without violating the hard turnover contract."""
    if not config.fusion.apply_interval_caps:
        return w
    by_id = {f.security_id: f for f in state.forecasts}
    lo_list: list[float] = []
    hi_list: list[float] = []
    for sid in ids:
        f = by_id.get(sid)
        if f is None or not f.interval_lo or not f.interval_hi:
            lo_list.append(float("nan"))
            hi_list.append(float("nan"))
            continue
        hz = next(iter(f.interval_lo))
        lo_list.append(float(f.interval_lo[hz]))
        hi_list.append(float(f.interval_hi[hz]))
    lo = np.asarray(lo_list, dtype=float)
    hi = np.asarray(hi_list, dtype=float)
    if not np.isfinite(lo).any() or not np.isfinite(hi).any():
        # A configured risk control silently not applying must be visible.
        log.warning("interval_caps_skipped", reason="no_finite_intervals", n_ids=len(ids))
        return w
    valid = np.isfinite(lo) & np.isfinite(hi) & (hi >= lo)
    if int(valid.sum()) == 0:
        log.warning("interval_caps_skipped", reason="no_valid_intervals", n_ids=len(ids))
        return w
    width_ref, downside_ref = interval_refs(lo[valid], hi[valid])
    capped, caps = apply_interval_caps(
        w,
        lo,
        hi,
        max_weight=float(config.constraints.name_max),
        width_ref=width_ref,
        downside_ref=downside_ref,
    )
    if w_prev is None:
        return capped
    prior = np.asarray(w_prev, dtype=float).reshape(-1)
    if prior.shape != capped.shape or not np.isfinite(prior).all():
        raise ValueError("w_prev must be finite and aligned with interval-capped weights")
    turnover_limit = float(config.constraints.turnover_limit)
    capped_turnover = float(np.abs(capped - prior).sum())
    if capped_turnover <= turnover_limit + 1e-12:
        return capped

    # The closest point to the prior book that satisfies the interval boxes is
    # the coordinate-wise projection onto [-caps, caps]. If that minimum move
    # already exceeds the documented hard turnover limit, emitting capped
    # weights would silently violate the optimizer contract; fail closed.
    nearest = np.clip(prior, -caps, caps)
    minimum_turnover = float(np.abs(nearest - prior).sum())
    if minimum_turnover > turnover_limit + 1e-12:
        raise OptimizationInfeasible(
            "interval caps are incompatible with the hard turnover limit: "
            f"minimum turnover {minimum_turnover:.12g} > {turnover_limit:.12g}"
        )

    # Otherwise move from the nearest feasible point toward the capped target
    # until the L1 turnover budget is exactly respected. Both endpoints obey
    # the interval boxes, so the interpolation remains inside them.
    low, high = 0.0, 1.0
    for _ in range(60):
        mid = (low + high) / 2.0
        candidate = nearest + mid * (capped - nearest)
        if float(np.abs(candidate - prior).sum()) <= turnover_limit:
            low = mid
        else:
            high = mid
    return nearest + low * (capped - nearest)


def optimize_asof(
    config: AppConfig,
    asof: datetime | None = None,
    *,
    persist: bool = True,
    w_prev: Array | dict[str, float] | None = None,
    use_fused_alpha: bool | None = None,
    frame: pl.DataFrame | None = None,
    day_index: dict[str, pl.DataFrame] | None = None,
    assume_sorted: bool = False,
    event_times: list[datetime] | None = None,
) -> pl.DataFrame:
    """Mean-variance weights as-of ``asof`` (no look-ahead beyond that decision time).

    OptimizationInfeasible and other genuine failures propagate — never silently
    flatten to zeros (see docs: no silent relaxation / auto-flatten).

    When ``w_prev`` is provided (dict by security_id or aligned array), turnover
    penalties use the prior book. Alpha sizing defaults to fused diagnostics
    (``fusion.use_fused_for_optimize``); set ``use_fused_alpha=False`` to force
    raw alpha. Conformal interval caps apply when intervals exist and
    ``fusion.apply_interval_caps`` is true. Trailing name-covariance uses the
    same ``available_time <= asof`` contract as the GARCH overlay: unpublished
    restatements cannot move Ledoit–Wolf/sample, OAS, or named DCC risk. Null
    availability among usable ``ret_1`` rows fails closed. Default covariance
    is trailing Ledoit-Wolf 2004 plus the GARCH/RGARCH overlay helper shared
    with ``/risk/portfolio``; when T<=N it stays Ledoit-Wolf rather than
    switching to sample. Set ``optimizer.covariance=dcc_gaussian``,
    ``dcc_student_t``, ``adcc``, ``ccc``, ``agdcc``, ``agdcc_full``, or
    ``ewma`` for the named one-step path; those paths do not overlay
    H_{t+1} and use the trailing contiguous complete-case window (holes
    are not concatenated; an incomplete asof row fails closed). Set
    ``optimizer.covariance=oas`` for trailing Chen OAS plus the overlay;
    that path must not silently size as Ledoit–Wolf, sample, EWMA, or DCC,
    and when T<=N it stays OAS. Set
    ``optimizer.covariance=ledoit_wolf_nonlinear`` for trailing analytical
    2020 nonlinear shrinkage plus the overlay; that path must not silently
    size as 2004 linear Ledoit–Wolf, OAS, sample, EWMA, or DCC, and when
    T<=N it stays nonlinear. Set ``optimizer.covariance=sample`` for
    trailing unbiased sample covariance plus the overlay; that path must
    not silently size as Ledoit–Wolf, OAS, EWMA, or DCC, and when T>N it
    stays sample. Named paths fail closed rather than substituting the
    default. Set ``optimizer.covariance=agdcc`` for diagonal CES AG-DCC
    one-step H_{t+1} without overlay; that path must not silently size as
    scalar ADCC or unrestricted AG-DCC. Set
    ``optimizer.covariance=agdcc_full`` for unrestricted CES AG-DCC
    one-step H_{t+1} without overlay; that path must not silently size as
    diagonal AG-DCC. Named CCC must not silently size as Gaussian DCC.
    Factor stays unwired.
    """
    state = forecast_asof(
        config,
        asof,
        frame=frame,
        day_index=day_index,
        assume_sorted=assume_sorted,
        event_times=event_times,
    )
    ids = [f.security_id for f in state.forecasts]
    use_fused = (
        bool(config.fusion.use_fused_for_optimize)
        if use_fused_alpha is None
        else bool(use_fused_alpha)
    )
    alpha = _alpha_vector(state, config, use_fused=use_fused)
    # covariance from trailing returns if present
    base = frame if frame is not None else _panel_cached(config)
    if assume_sorted:
        _require_assume_sorted_contract(base)
        hist = history_prefix_upto(base, state.asof)
    elif under_history_sort_contract(base):
        hist = history_prefix_upto(base, state.asof)
    else:
        hist = base.filter(pl.col("event_time") <= state.asof)
    hist = filter_trailing_returns_asof(hist, state.asof)
    estimator = require_implemented_optimizer_covariance(config.optimizer.covariance)
    if estimator in IMPLEMENTED_OPTIMIZER_NAMED_SPECS:
        estimate = estimate_optimizer_covariance_asof(config, base, state.asof, ids, hist)
        alpha_map = dict(zip(ids, alpha, strict=False))
        ids = estimate.security_ids
        alpha = np.array([alpha_map.get(c, 0.0) for c in ids], dtype=float)
        sig = estimate.sigma
    elif "ret_1" in hist.columns and hist.height > 20:
        estimate = estimate_optimizer_covariance_asof(config, base, state.asof, ids, hist)
        if estimate.unmeasured_reason is None:
            alpha_map = dict(zip(ids, alpha, strict=False))
            ids = estimate.security_ids
            alpha = np.array([alpha_map.get(c, 0.0) for c in ids], dtype=float)
            sig = estimate.sigma
        else:
            log.warning(
                "covariance_fallback",
                reason=estimate.unmeasured_reason,
                finite_rows=int(estimate.n_obs),
                n_bars=int(hist.height),
            )
            sig = np.diag(np.ones(len(ids)) * 0.02**2)
            sig, overlay, overlay_kind = apply_market_variance_overlay_to_covariance(
                config, base, state.asof, sig
            )
            estimate = replace(estimate, sigma=sig, overlay=overlay, market_overlay=overlay_kind)
    else:
        reason = "insufficient_history" if "ret_1" in hist.columns else "no_ret_1"
        log.warning("covariance_fallback", reason=reason, n_bars=int(hist.height))
        sig = np.diag(np.ones(len(ids)) * 0.02**2)
        sig, overlay, overlay_kind = apply_market_variance_overlay_to_covariance(
            config, base, state.asof, sig
        )
        estimate = replace(
            _unmeasured_optimizer_covariance(reason, ids),
            sigma=sig,
            overlay=overlay,
            market_overlay=overlay_kind,
        )
    wp = _align_w_prev(ids, w_prev)
    # Planner cost vector: the same modeled one-way cost the execution engine
    # charges (commission + half spread + per-turnover bps), as a fraction of
    # notional. The scalar ``optimizer.lambda_tc`` stays a dimensionless
    # multiplier; passing unit costs instead of modeled costs made the planner
    # see a 100% penalty and return a dead book.
    one_way_cost = (
        float(config.costs.commission_bps)
        + float(config.costs.half_spread_bps)
        + float(config.costs.bps_per_turnover)
    ) / 1e4
    tc_linear = np.full(len(ids), max(one_way_cost, 0.0), dtype=float)
    w, diag = optimize_mean_variance(alpha, sig, wp, config, tc_linear=tc_linear)
    w = _apply_forecast_interval_caps(w, state, ids, config, w_prev=wp)
    payload: dict[str, object] = {
        "event_time": [state.asof] * len(ids),
        "security_id": ids,
        "target_weight": w.tolist(),
        "alpha": alpha.tolist(),
        "covariance_estimator": [estimate.estimator] * len(ids),
        "covariance_object": [estimate.covariance_object] * len(ids),
        "covariance_spec": [estimate.spec] * len(ids),
    }
    if (
        estimate.unmeasured_reason is None
        and estimate.estimator in IMPLEMENTED_OPTIMIZER_ONE_STEP_SPECS
    ):
        payload["covariance_horizon"] = [1] * len(ids)
    elif (
        estimate.estimator not in IMPLEMENTED_OPTIMIZER_ONE_STEP_SPECS
        and state.garch_market_sigma is not None
    ):
        payload["garch_market_sigma"] = [float(state.garch_market_sigma)] * len(ids)
        payload["garch_market_variance"] = [float(state.garch_market_variance or 0.0)] * len(ids)
        payload["garch_series_scope"] = [str(state.garch_series_scope)] * len(ids)
        payload["market_risk_overlay"] = [str(state.market_risk_overlay)] * len(ids)
    out = pl.DataFrame(payload)
    if persist:
        Path(config.data.root).joinpath("gold").mkdir(parents=True, exist_ok=True)
        out.write_parquet(Path(config.data.root) / "gold" / "target_weights.parquet")
    _ = diag
    return out


def build_causal_weight_panel(
    config: AppConfig,
    dates: list[datetime] | None = None,
) -> pl.DataFrame:
    """Point-in-time weight panel: ``optimize_asof(asof=d)`` per decision date.

    Does not broadcast end-of-sample weights across history (look-ahead).
    Passes the prior date's target weights as ``w_prev`` so turnover and TC
    terms are causal rather than always-from-cash.
    """
    if dates is None:
        dates = panel(config)["event_time"].unique().sort().to_list()
    if not dates:
        return pl.DataFrame(
            schema={
                "event_time": pl.Datetime,
                "security_id": pl.Utf8,
                "target_weight": pl.Float64,
                "alpha": pl.Float64,
            }
        )
    # Load gold panel + ranker once; reuse across asof dates (Phase 18 hot path).
    # Wave 8: build event_time day index once for exact day slices (PIT-safe).
    # Wave 12: ensure HISTORY_SORT_KEYS once; pass assume_sorted + event_times
    # so calibration history uses prefix slice (not per-asof day-index concat).
    shared = _panel_cached(config)
    if not under_history_sort_contract(shared):
        shared = sort_for_history(shared)
    day_index = build_event_time_day_index(shared)
    event_times = shared["event_time"].unique().sort().to_list()
    _ = _load_ranker_cached(config)
    frames: list[pl.DataFrame] = []
    prev_w: dict[str, float] = {}
    for i, d in enumerate(dates):
        frame = optimize_asof(
            config,
            d,
            persist=(i == len(dates) - 1),
            w_prev=prev_w or None,
            frame=shared,
            day_index=day_index,
            assume_sorted=True,
            event_times=event_times,
        )
        frames.append(frame)
        prev_w = {
            str(r["security_id"]): float(r["target_weight"]) for r in frame.iter_rows(named=True)
        }
    return pl.concat(frames)
