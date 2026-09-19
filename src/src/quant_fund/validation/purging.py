"""Purging overlapping labels. Horizon in bars."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta


def overlaps(
    train_decision: datetime,
    test_start: datetime,
    horizon_bars: int,
    bar_delta: timedelta,
) -> bool:
    """True if the label window (decision, decision+horizon] intersects [test_start, ...)."""
    if horizon_bars < 0:
        raise ValueError("horizon_bars must be non-negative")
    label_end = train_decision + horizon_bars * bar_delta
    # A decision on or after the test start is itself in the holdout window.
    return train_decision < test_start <= label_end or train_decision >= test_start


def purge_mask(
    decision_times: list[datetime],
    test_start: datetime,
    test_end: datetime,
    horizon_bars: int,
    *,
    session_index: dict[datetime, int] | None = None,
    label_end_times: list[datetime] | None = None,
) -> list[bool]:
    """Return True for observations that are SAFE to keep in train.

    The label interval is ``(decision, decision+horizon]`` in session-index
    space and the test interval is inclusive.  Endpoint insertion points make
    the result conservative when a test boundary is not an observed session.
    """
    if horizon_bars < 0:
        raise ValueError("horizon_bars must be non-negative")
    if label_end_times is not None and len(label_end_times) != len(decision_times):
        raise ValueError("label_end_times must align with decision_times")
    if test_end < test_start:
        raise ValueError("test_end must be on or after test_start")
    if session_index is None:
        unique = sorted(set(decision_times) | {test_start, test_end})
        session_index = {t: i for i, t in enumerate(unique)}
    keys = sorted(session_index)
    if not keys:
        return [True] * len(decision_times)

    test_lo = bisect_left(keys, test_start)
    test_hi = bisect_right(keys, test_end) - 1
    keep: list[bool] = []
    for t in decision_times:
        i = session_index.get(t)
        if i is None:
            i = bisect_left(keys, t)
        label_end = i + horizon_bars
        if label_end_times is not None:
            end_time = label_end_times[len(keep)]
            if end_time < t:
                raise ValueError("label_end_times must not precede decision_times")
            # Explicit timestamps are the source of truth for sparse/asynchronous
            # labels; session-index arithmetic is only the compatibility fallback.
            pre_test_overlap = t < test_start <= end_time
        else:
            pre_test_overlap = i < test_lo and label_end >= test_lo
        # A row is unsafe for train when either
        #   (a) it lies inside the inclusive holdout [test_lo, test_hi] —
        #       including the final holdout session and a one-session holdout
        #       (consistent with overlaps()), or
        #   (b) its label window (i, label_end] reaches test_lo.
        in_holdout = test_lo <= i <= test_hi
        keep.append(not (in_holdout or pre_test_overlap))
    return keep
