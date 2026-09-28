"""Point-in-time history slicing for forecast panels.

Split out of the original module. Import the parent path; it re-exports these names.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from quant_fund.config.models import AppConfig
from quant_fund.metrics.cross_section import _date_keys
from quant_fund.schemas.forecast import IntervalMethod

if TYPE_CHECKING:
    from .state import Array


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
    # A label issued at session i is realized at session i+horizon; it is
    # available at asof only when i <= idx-horizon-1. When fewer than
    # horizon+1 sessions precede asof no label can be realized — return empty
    # rather than falling back to unrealized (future) labels.
    last = idx - horizon - 1
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


__all__ = [
    "ForecastIntervals",
    "HISTORY_SORT_KEYS",
    "build_event_time_day_index",
    "history_for_calibration",
    "history_prefix_upto",
    "history_upto",
    "latest_decision",
    "slice_day",
    "sort_for_history",
    "under_history_sort_contract",
]
