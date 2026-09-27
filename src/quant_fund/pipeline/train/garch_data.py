"""Point-in-time return and OHLC histories for GARCH fits.

Split out of the original module. Import the parent path; it re-exports these names.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from quant_fund.models.realized_garch import parkinson_daily_variance
from quant_fund.schemas.errors import PointInTimeError

from .splits import _available_stamp_is_missing, _stamp_at_or_before, _stamp_strictly_before


def _require_garch_security_keys(frame: Any) -> None:
    """Fail closed on blank ids or duplicate ``(security_id, event_time)`` keys.

    Equal-weight date-level means and per-name histories are undefined when a
    name is double-counted. Frames without ``security_id`` keep the legacy
    date-only path used by univariate fixtures.
    """
    columns = set(frame.columns)
    if "security_id" not in columns or "event_time" not in columns:
        return
    if frame.is_empty():
        return
    blank = frame.filter(
        pl.col("security_id").is_null()
        | (pl.col("security_id").cast(pl.String).str.strip_chars() == "")
    )
    if blank.height:
        raise PointInTimeError("GARCH return history contains blank security_id")
    if frame.select(["security_id", "event_time"]).is_duplicated().any():
        raise PointInTimeError(
            "GARCH return history contains duplicate security_id/event_time rows"
        )


def _garch_name_return_history(
    frame: Any,
    security_id: str,
    asof: object | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Causal ``ret_1`` history for one security, never a pooled cross-section.

    Duplicate panel keys fail closed on the full frame so a colliding name
    cannot silently inflate another name's series. ``available_time`` uses the
    same as-of contract as the date-level overlay.
    """
    if not isinstance(security_id, str) or not security_id.strip():
        raise ValueError("per-security GARCH requires a non-empty security_id")
    columns = set(frame.columns)
    if "security_id" not in columns:
        raise PointInTimeError("per-security GARCH frame missing security_id")
    _require_garch_security_keys(frame)
    sid = security_id.strip()
    name_frame = frame.filter(pl.col("security_id").cast(pl.String) == sid)
    if name_frame.is_empty():
        return np.asarray([]), np.asarray([], dtype=float)
    return _garch_return_history(name_frame, asof=asof)


def _garch_return_history(
    frame: Any,
    asof: object | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite equal-weight ``ret_1`` observations aggregated by date.

    This deliberately reads the full feature panel instead of the label-filtered
    design matrix: the latest returns are known even when their forward variance
    labels are structurally unavailable. When ``asof`` is supplied, only rows
    with ``event_time < asof`` enter the series. A present ``available_time``
    column is PIT-filtered at the same origin (``available_time <= asof``);
    null availability among otherwise usable rows fails closed rather than
    treating an unpublished restatement as observable. Frames without
    ``available_time`` keep the legacy event-time path.
    """
    _require_garch_security_keys(frame)
    columns = set(frame.columns)
    if not {"event_time", "ret_1"}.issubset(columns):
        raise ValueError("GARCH requires event_time and ret_1 in the full feature panel")
    has_availability = "available_time" in columns
    selected = ["event_time", "ret_1"]
    if has_availability:
        selected.append("available_time")
    history = frame.select(selected).drop_nulls(subset=["event_time", "ret_1"])
    raw_dates = np.asarray(history["event_time"].to_numpy())
    raw_values = np.asarray(history["ret_1"].to_numpy(), dtype=float)
    finite = np.isfinite(raw_values)
    if not np.any(finite):
        raise ValueError("GARCH return history contains no finite ret_1 observations")
    raw_dates = raw_dates[finite]
    raw_values = raw_values[finite]
    raw_available = (
        np.asarray(history["available_time"].to_numpy())[finite] if has_availability else None
    )
    if asof is not None:
        keep = np.array(
            [_stamp_strictly_before(stamp, asof) for stamp in raw_dates.tolist()],
            dtype=bool,
        )
        if raw_available is not None:
            available_stamps = raw_available.tolist()
            missing = [
                _available_stamp_is_missing(stamp)
                for stamp, origin_ok in zip(available_stamps, keep.tolist(), strict=True)
                if origin_ok
            ]
            if any(missing):
                raise PointInTimeError(
                    "GARCH return history has null available_time; "
                    "refusing unobservable market overlay"
                )
            keep &= np.array(
                [
                    (not _available_stamp_is_missing(stamp)) and _stamp_at_or_before(stamp, asof)
                    for stamp in available_stamps
                ],
                dtype=bool,
            )
        raw_dates = raw_dates[keep]
        raw_values = raw_values[keep]
        if raw_dates.size == 0:
            return np.asarray([]), np.asarray([], dtype=float)
    ordered_dates = sorted(set(raw_dates.tolist()))
    dates = np.asarray(ordered_dates)
    values = np.asarray(
        [np.mean(raw_values[raw_dates == date]) for date in ordered_dates], dtype=float
    )
    return dates, values


def _realized_garch_ohlc_columns(frame: Any) -> tuple[str, str]:
    """Prefer split-adjusted high/low; never invent a close-to-close RV proxy."""
    columns = set(frame.columns)
    if {"high_split_adjusted", "low_split_adjusted"}.issubset(columns):
        return "high_split_adjusted", "low_split_adjusted"
    if {"high", "low"}.issubset(columns):
        return "high", "low"
    raise PointInTimeError(
        "Realized GARCH requires daily OHLC high/low; refusing close-to-close squared proxy"
    )


def _realized_garch_history(
    frame: Any,
    asof: object | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Date-level equal-weight ``ret_1`` paired with same-name Parkinson variance.

    One-day Parkinson is computed from daily OHLC. The 20-day ``vol_parkinson``
    feature is never the realized measure. Names missing a finite return or
    Parkinson drop from *both* means so the pair stays aligned. PIT filters
    match ``_garch_return_history``.
    """
    _require_garch_security_keys(frame)
    columns = set(frame.columns)
    if not {"event_time", "ret_1"}.issubset(columns):
        raise ValueError("Realized GARCH requires event_time and ret_1")
    high_col, low_col = _realized_garch_ohlc_columns(frame)
    has_availability = "available_time" in columns
    selected = ["event_time", "ret_1", high_col, low_col]
    if has_availability:
        selected.append("available_time")
    history = frame.select(selected).drop_nulls(subset=["event_time", "ret_1", high_col, low_col])
    if history.is_empty():
        raise ValueError("Realized GARCH history contains no finite OHLC/return pairs")
    park = parkinson_daily_variance(history[high_col].to_numpy(), history[low_col].to_numpy())
    raw_dates = np.asarray(history["event_time"].to_numpy())
    raw_values = np.asarray(history["ret_1"].to_numpy(), dtype=float)
    finite = np.isfinite(raw_values) & np.isfinite(park) & (park > 0.0)
    if not np.any(finite):
        raise ValueError("Realized GARCH history contains no finite Parkinson/return pairs")
    raw_dates = raw_dates[finite]
    raw_values = raw_values[finite]
    raw_park = park[finite]
    raw_available = (
        np.asarray(history["available_time"].to_numpy())[finite] if has_availability else None
    )
    if asof is not None:
        keep = np.array(
            [_stamp_strictly_before(stamp, asof) for stamp in raw_dates.tolist()],
            dtype=bool,
        )
        if raw_available is not None:
            available_stamps = raw_available.tolist()
            missing = [
                _available_stamp_is_missing(stamp)
                for stamp, origin_ok in zip(available_stamps, keep.tolist(), strict=True)
                if origin_ok
            ]
            if any(missing):
                raise PointInTimeError(
                    "Realized GARCH history has null available_time; "
                    "refusing unobservable Parkinson overlay"
                )
            keep &= np.array(
                [
                    (not _available_stamp_is_missing(stamp)) and _stamp_at_or_before(stamp, asof)
                    for stamp in available_stamps
                ],
                dtype=bool,
            )
        raw_dates = raw_dates[keep]
        raw_values = raw_values[keep]
        raw_park = raw_park[keep]
        if raw_dates.size == 0:
            return np.asarray([]), np.asarray([], dtype=float), np.asarray([], dtype=float)
    ordered_dates = sorted(set(raw_dates.tolist()))
    dates = np.asarray(ordered_dates)
    values = np.asarray(
        [np.mean(raw_values[raw_dates == date]) for date in ordered_dates], dtype=float
    )
    measures = np.asarray(
        [np.mean(raw_park[raw_dates == date]) for date in ordered_dates], dtype=float
    )
    return dates, values, measures
