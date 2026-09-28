"""Chronological and walk-forward split helpers.

Split out of the original module. Import the parent path; it re-exports these names.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

import numpy as np

from quant_fund.pipeline.dataset import design_frame
from quant_fund.validation.purging import purge_mask
from quant_fund.validation.walk_forward import Fold, timestamp_ns, walk_forward


def _chronological_split(
    dates: Any,
    *,
    train_fraction: float = 0.7,
    horizon_bars: int = 1,
    embargo_bars: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Split whole decision dates, purging labels and an explicit embargo.

    This compatibility helper is intentionally date-level: no panel rows from
    a date are split across train and test.  ``horizon_bars`` removes dates
    whose forward labels could reach the test boundary; ``embargo_bars`` adds
    the configured post-training gap independently of the label horizon.
    """
    values = np.asarray(dates)
    unique = sorted(set(values.tolist()))
    if not 0.0 < float(train_fraction) < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if int(horizon_bars) < 0:
        raise ValueError("horizon_bars must be non-negative")
    if int(embargo_bars) < 0:
        raise ValueError("embargo_bars must be non-negative")
    if len(unique) < 2:
        raise ValueError("chronological_split requires at least 2 unique dates")
    cut = int(np.floor(len(unique) * float(train_fraction)))
    cut = min(max(cut, 1), len(unique) - 1)
    train_end = max(0, cut - int(horizon_bars) - int(embargo_bars))
    train_dates = set(unique[:train_end])
    test_dates = set(unique[cut:])
    date_ns = timestamp_ns(values)
    return (
        np.isin(date_ns, timestamp_ns(list(train_dates))),
        np.isin(date_ns, timestamp_ns(list(test_dates))),
    )


def _walk_forward_splits(
    dates: Any,
    config: Any,
    *,
    horizon_bars: int,
    label_end_times: Any | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return disjoint, date-level train/test masks with purging and embargo."""
    values = np.asarray(dates)
    ends = None if label_end_times is None else np.asarray(label_end_times)
    if ends is not None and len(ends) != len(values):
        raise ValueError("label_end_times must align with dates")
    unique = sorted(set(values.tolist()))
    validation = getattr(config, "validation", config)
    embargo_method = getattr(config, "embargo_bars", None)
    configured = (
        embargo_method() if callable(embargo_method) else getattr(validation, "embargo_bars", None)
    )
    # Explicit embargo_bars=0 must stay 0 (``or`` would substitute the horizon).
    embargo = int(configured) if configured is not None else int(horizon_bars)
    folds = walk_forward(
        values.tolist(),
        validation,
        horizon_bars=int(horizon_bars),
        embargo_bars=embargo,
        label_end_times=None if ends is None else ends.tolist(),
    )
    if not folds:
        # Short samples still need a genuine untouched test block. Purge the
        # candidate train dates before the test boundary instead of splitting
        # rows. When observed label endpoints are available, use them as the
        # source of truth; session arithmetic is only the compatibility fallback.
        cut = max(1, len(unique) // 2)
        test_times = unique[cut:]
        if ends is not None and test_times:
            end_by_time: dict[Any, Any] = {}
            for date, end in zip(values.tolist(), ends.tolist(), strict=True):
                previous = end_by_time.get(date)
                if previous is None or end > previous:
                    end_by_time[date] = end
            train_candidates = unique[:cut]
            session_idx = {date: i for i, date in enumerate(unique)}
            keep = purge_mask(
                train_candidates,
                test_times[0],
                test_times[-1],
                int(horizon_bars),
                session_index=session_idx,
                label_end_times=[end_by_time[date] for date in train_candidates],
            )
            train_times = [
                date
                for date, safe in zip(train_candidates, keep, strict=True)
                if safe and session_idx[date] + embargo < cut
            ]
        else:
            train_end = max(0, cut - int(horizon_bars) - embargo)
            train_times = unique[:train_end]
        folds = [Fold(train_times=train_times, val_times=[], test_times=test_times)]
    date_ns = timestamp_ns(values)
    return [
        (
            np.isin(date_ns, timestamp_ns(fold.train_times)),
            np.isin(date_ns, timestamp_ns(fold.test_times)),
        )
        for fold in folds
        if fold.train_times and fold.test_times
    ]


def _split_fold(
    dates: Any, x: np.ndarray, y: np.ndarray, fold: Fold
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    date_ns = timestamp_ns(dates)
    tr = np.isin(date_ns, timestamp_ns(fold.train_times))
    va = np.isin(date_ns, timestamp_ns(fold.val_times))
    te = np.isin(date_ns, timestamp_ns(fold.test_times))
    return tr, va, te


def _label_horizon(label: str, default: int = 5) -> int:
    """Extract the trailing bar horizon encoded by a forward label name."""
    match = re.search(r"_(\d+)$", label)
    return int(match.group(1)) if match else int(default)


def _aligned_label_end_times(frame: Any, label: str, features: list[str]) -> np.ndarray | None:
    """Return row-aligned observed label endpoints when the label engine provides them."""
    horizon = _label_horizon(label)
    endpoint = f"label_end_time_{horizon}"
    if endpoint not in frame.columns:
        return None
    # Match design_matrix's selected columns and null filtering exactly.  The
    # endpoint is non-null whenever a forward label is usable.
    sub = design_frame(frame, label, features, extra_columns=[endpoint])
    return sub[endpoint].to_numpy()


def _require_model(name: str, catalog: set[str], family: str) -> str:
    """Fail closed on unknown model names before any panel I/O."""
    if name not in catalog:
        raise ValueError(f"unknown {family} model {name!r}")
    return name


def _as_utc(value: datetime) -> datetime:
    """Normalize comparison stamps; naive wall-clock values mean UTC."""
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _stamp_naive(value: datetime) -> bool:
    return value.tzinfo is None or value.utcoffset() is None


def _stamp_strictly_before(stamp: object, asof: object) -> bool:
    """Compare panel timestamps to asof without mixing naive/aware datetimes."""
    if isinstance(stamp, datetime) and isinstance(asof, datetime):
        if _stamp_naive(stamp) != _stamp_naive(asof):
            return _as_utc(stamp) < _as_utc(asof)
        return stamp < asof
    return bool(stamp < asof)  # type: ignore[operator]


def _stamp_at_or_before(stamp: object, asof: object) -> bool:
    """True when ``stamp`` is observable at the decision origin."""
    if isinstance(stamp, datetime) and isinstance(asof, datetime):
        if _stamp_naive(stamp) != _stamp_naive(asof):
            return _as_utc(stamp) <= _as_utc(asof)
        return stamp <= asof
    return bool(stamp <= asof)  # type: ignore[operator]


def _available_stamp_is_missing(stamp: object) -> bool:
    if stamp is None:
        return True
    if isinstance(stamp, datetime):
        return False
    try:
        if stamp != stamp:  # NaT / NaN
            return True
    except (TypeError, ValueError):
        return True
    return False
