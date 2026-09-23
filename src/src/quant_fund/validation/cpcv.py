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
    """Split the timeline into ``n_groups`` sequential groups; every combination
    of ``n_test_groups`` is the test set, train is the purged/embargoed complement.

    Purge and embargo are applied **per contiguous test group**. Using the
    min/max span of a non-contiguous test combination would incorrectly wipe
    intervening train groups.

    Raises
    ------
    ValueError
        Invalid group counts, negative horizon/embargo, or fewer unique dates
        than ``n_groups`` (empty groups would break fold-count integrity).
    """
    if n_groups < 2 or n_test_groups < 1 or n_test_groups >= n_groups:
        raise ValueError("invalid CPCV group counts")
    if horizon_bars < 0:
        raise ValueError("horizon_bars must be non-negative")
    if embargo_bars < 0:
        raise ValueError("embargo_bars must be non-negative")
    uniq = sorted(set(times))
    n = len(uniq)
    if n == 0:
        return []
    if n < n_groups:
        raise ValueError(f"need at least n_groups={n_groups} unique dates for CPCV, got {n}")
    bounds = [int(i * n / n_groups) for i in range(n_groups + 1)]
    groups = [uniq[bounds[i] : bounds[i + 1]] for i in range(n_groups)]
    if any(len(g) == 0 for g in groups):
        raise ValueError("CPCV group bounds produced an empty group")
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
        purged = list(train_times)
        for g in test_ids:
            block = groups[g]
            if not block:
                continue
            keep = purge_mask(purged, block[0], block[-1], horizon_bars, session_index=idx)
            purged = [t for t, k in zip(purged, keep, strict=True) if k]
            lo, hi = idx[block[0]], idx[block[-1]]
            purged = [
                t
                for t in purged
                if not (hi < idx[t] <= hi + embargo_bars) and not (lo - embargo_bars <= idx[t] < lo)
            ]
        if not purged:
            continue
        folds.append(Fold(train_times=purged, val_times=[], test_times=test_times))
    return folds
