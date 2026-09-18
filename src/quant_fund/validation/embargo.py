"""Embargo after a test/validation block."""

from __future__ import annotations

from datetime import datetime


def embargo_mask(
    decision_times: list[datetime],
    block_end: datetime,
    embargo_bars: int,
    session_index: dict[datetime, int],
) -> list[bool]:
    """True = keep. Drop the next embargo_bars sessions after block_end.

    Edge rules (fail-closed / identity-preserving):
    - empty decision_times → []
    - embargo_bars <= 0 (incl. negative) → keep all
    - empty session_index → keep all (no sessions to map)
    - unknown block_end → nearest session key by absolute time delta
    - decision times missing from session_index → kept
    - returned mask length always equals len(decision_times)
    """
    n = len(decision_times)
    if n == 0:
        return []
    if embargo_bars <= 0 or not session_index:
        return [True] * n
    end_i = session_index.get(block_end)
    if end_i is None:
        keys = sorted(session_index)
        end_i = session_index[min(keys, key=lambda k: abs((k - block_end).total_seconds()))]
    keep: list[bool] = []
    for t in decision_times:
        i = session_index.get(t)
        if i is None:
            keep.append(True)
            continue
        drop = end_i < i <= end_i + embargo_bars
        keep.append(not drop)
    assert len(keep) == n
    return keep
