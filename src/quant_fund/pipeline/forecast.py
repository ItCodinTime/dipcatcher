"""Produce AssetForecasts and target weights for a decision date."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_fund.config.models import AppConfig
from quant_fund.fusion.engine import fuse_signals
from quant_fund.metrics.conformal import assign_terciles
from quant_fund.metrics.cross_section import _date_keys
from quant_fund.models.conformal import MondrianCQR, SplitCQR
from quant_fund.models.covariance import ledoit_wolf_cov, repair_psd, sample_cov
from quant_fund.models.distribution import (
    GaussianDistribution,
    ScaledGaussianDistribution,
    ScaledStudentTDistribution,
    fit_scaled_wrappee,
    select_wrappee_family_name,
)
from quant_fund.models.ranking import RidgeRanker, available_features
from quant_fund.pipeline.dataset import design_matrix, panel
from quant_fund.portfolio.interval_risk import apply_interval_caps, interval_refs
from quant_fund.portfolio.optimizer import optimize_mean_variance
from quant_fund.schemas.errors import OptimizationInfeasible
from quant_fund.schemas.forecast import AssetForecast, IntervalMethod, MarketState
from quant_fund.utils.logging import get_logger

Array = NDArray[np.float64]

log = get_logger()

INTERVAL_ALPHA = 0.10

# Process-local caches for causal panel / multi-asof hot paths
_RANKER_CACHE: dict[tuple[str, float], object] = {}
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
    """Clear ranker / conformal / wrappee process-local caches (panel cache separate)."""
    _RANKER_CACHE.clear()
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


def _load_ranker_cached(config: AppConfig):
    """Load ridge ranker joblib once per (path, mtime); None if missing."""
    rank_path = Path(config.data.root) / "metadata" / "ranker_ridge.joblib"
    if not rank_path.exists():
        return None
    key = (str(rank_path.resolve()), rank_path.stat().st_mtime)
    cached = _RANKER_CACHE.get(key)
    if cached is not None:
        return cached
    model = RidgeRanker.load(rank_path)
    _RANKER_CACHE[key] = model
    # Bound cache size
    if len(_RANKER_CACHE) > 8:
        oldest = next(iter(_RANKER_CACHE))
        _RANKER_CACHE.pop(oldest, None)
    return model


def _panel_cached(config: AppConfig) -> pl.DataFrame:
    """Thin wrapper — dataset.panel already caches; keep call site explicit."""
    return panel(config)


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
    feats = available_features(day.columns)
    x = (
        day.select(feats).fill_null(0.0).to_numpy().astype(float)
        if feats
        else np.zeros((day.height, 1))
    )
    model = _load_ranker_cached(config)
    if model is not None:
        # A stale joblib / feature-set mismatch must surface: silently degrading
        # to the momentum heuristic would masquerade a fabricated alpha as model
        # output. The heuristic is used only when no ranker is loaded at all.
        scores = model.predict(x)
    else:
        scores = (
            day["cs_pct_mom_20"].fill_null(0.5).to_numpy().astype(float)
            if "cs_pct_mom_20" in day.columns
            else np.zeros(day.height)
        )
    # percentile ranks within the day
    order = scores.argsort().argsort()
    pct = (order + 0.5) / max(len(scores), 1)
    vol = (
        day["vol_20"].fill_null(0.02).to_numpy().astype(float)
        if "vol_20" in day.columns
        else np.full(day.height, 0.02)
    )
    alpha = (pct - 0.5) * 2.0 * config.fusion.alpha_scale
    conf = np.clip(0.5 + np.abs(pct - 0.5), 0.2, 1.0)
    regime = np.ones(day.height)
    tail = np.clip(vol * 0.1, 0, None)
    liq = np.zeros(day.height)
    fused = fuse_signals(alpha, conf, regime, vol, tail, liq, config.fusion)
    notes = []
    if config.data.source == "synthetic":
        notes.append("SYNTHETIC")
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
        forecasts.append(
            AssetForecast(
                security_id=sid,
                symbol=row.get("symbol", sid),
                asof=asof,
                model_version="fusion.v1",
                expected_returns={"5d": float(alpha[i])},
                quantiles={"5d": {0.05: q05, 0.5: q50, 0.95: q95}},
                probability_positive={"5d": float(pct[i])},
                alpha={"5d": float(alpha[i])},
                rank_score={"5d": float(scores[i])},
                rank_percentile={"5d": float(pct[i])},
                volatility={"5d": float(vol[i])},
                confidence={"5d": float(conf[i])},
                diagnostics={"fused": float(fused[i])},
                interval_lo=lo_map,
                interval_hi=hi_map,
                interval_alpha=i_alpha,
                interval_method=method,
            )
        )
    return MarketState(asof=asof, forecasts=forecasts, notes=notes)


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
    ``fusion.apply_interval_caps`` is true.
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
    if "ret_1" in hist.columns and hist.height > 20:
        wide = hist.select(["event_time", "security_id", "ret_1"]).pivot(
            on="security_id", index="event_time", values="ret_1"
        )
        cols = [c for c in ids if c in wide.columns]
        mat = wide.select(cols).to_numpy() if cols else np.empty((0, 0))
        finite_rows = int(np.isfinite(mat).all(axis=1).sum()) if mat.size else 0
        if cols and finite_rows >= 2:
            try:
                sig = ledoit_wolf_cov(mat) if mat.shape[0] > mat.shape[1] else sample_cov(mat)
                sig, _ = repair_psd(sig)
                alpha_map = dict(zip(ids, alpha, strict=False))
                alpha = np.array([alpha_map.get(c, 0.0) for c in cols], dtype=float)
                ids = cols
            except ValueError:
                # Documented fallback, but never silent: the homoskedastic 2%
                # proxy is recorded so downstream diagnostics can flag it.
                log.warning("covariance_fallback", reason="estimation_failed", n_ids=len(ids))
                sig = np.diag(np.ones(len(ids)) * 0.02**2)
        else:
            log.warning(
                "covariance_fallback",
                reason="insufficient_finite_rows",
                finite_rows=finite_rows,
            )
            sig = np.diag(np.ones(len(ids)) * 0.02**2)
    else:
        log.warning("covariance_fallback", reason="insufficient_history", n_bars=int(hist.height))
        sig = np.diag(np.ones(len(ids)) * 0.02**2)
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
    out = pl.DataFrame(
        {
            "event_time": [state.asof] * len(ids),
            "security_id": ids,
            "target_weight": w.tolist(),
            "alpha": alpha.tolist(),
        }
    )
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
