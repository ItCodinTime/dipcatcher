"""Embargo after a test/validation block."""

from __future__ import annotations

from datetime import datetime


def embargo_mask(
    decision_times: list[datetime],
    block_end: datetime,
    embargo_bars: int,
    session_index: dict[datetime, int],
) -> list[bool]:
    """True = keep. Drop the next embargo_bars sessions after block_end."""
    if embargo_bars <= 0:
        return [True] * len(decision_times)
    end_i = session_index.get(block_end)
    if end_i is None:
        keys = sorted(session_index)
        end_i = session_index[min(keys, key=lambda k: abs((k - block_end).total_seconds()))]
    keep = []
    for t in decision_times:
        i = session_index.get(t)
        if i is None:
            keep.append(True)
            continue
        drop = end_i < i <= end_i + embargo_bars
        keep.append(not drop)
    return keep
