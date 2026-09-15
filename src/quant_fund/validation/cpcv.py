"""Combinatorial purged cross-validation (López de Prado)."""

from __future__ import annotations

from datetime import datetime
from itertools import combinations

from quant_fund.validation.purging import purge_mask
from quant_fund.validation.walk_forward import Fold, session_index


def combinatorial_purged_cv(
    times: list[datetime],
    n_groups: int,
    n_test_groups: int,
    horizon_bars: int,
    embargo_bars: int,
) -> list[Fold]:
    uniq = sorted(set(times))
    n = len(uniq)
    if n_groups < 2 or n_test_groups < 1 or n_test_groups >= n_groups:
        raise ValueError("invalid CPCV group counts")
    bounds = [int(i * n / n_groups) for i in range(n_groups + 1)]
    groups = [uniq[bounds[i] : bounds[i + 1]] for i in range(n_groups)]
    idx = session_index(uniq)
    folds: list[Fold] = []
    for test_ids in combinations(range(n_groups), n_test_groups):
        test_times: list[datetime] = []
        for g in test_ids:
            test_times.extend(groups[g])
        train_times: list[datetime] = []
        for g in range(n_groups):
            if g not in test_ids:
                train_times.extend(groups[g])
        if not test_times or not train_times:
            continue
        keep = purge_mask(
            train_times, test_times[0], test_times[-1], horizon_bars, session_index=idx
        )
        # also purge each contiguous test block separately
        purged = [t for t, k in zip(train_times, keep, strict=True) if k]
        # embargo around each test group
        for g in test_ids:
            if not groups[g]:
                continue
            g_idx = [idx[t] for t in groups[g]]
            lo, hi = min(g_idx), max(g_idx)
            purged = [
                t
                for t in purged
                if not (lo - embargo_bars <= idx[t] <= hi + embargo_bars and lo <= idx[t] <= hi)
                and not (hi < idx[t] <= hi + embargo_bars)
                and not (lo - embargo_bars <= idx[t] < lo)
            ]
        folds.append(Fold(train_times=purged, val_times=[], test_times=test_times))
    return folds
