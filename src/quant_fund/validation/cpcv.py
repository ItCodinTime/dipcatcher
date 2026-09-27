"""Combinatorial purged cross-validation (López de Prado).

Path count follows López de Prado, *Advances in Financial Machine Learning*
(2018), chapter 12: with ``N`` groups and ``k`` test groups there are
``C(N, k)`` splits and ``φ = C(N-1, k-1)`` backtest paths.
"""

from __future__ import annotations

from datetime import datetime
from itertools import combinations
from math import comb

import numpy as np
from numpy.typing import NDArray

from quant_fund.validation.purging import purge_mask
from quant_fund.validation.walk_forward import Fold, session_index

IntArray = NDArray[np.int64]


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


def _require_group_counts(n_groups: int, n_test_groups: int) -> None:
    if n_groups < 2 or n_test_groups < 1 or n_test_groups >= n_groups:
        raise ValueError("invalid CPCV group counts")


def cpcv_n_splits(n_groups: int, n_test_groups: int) -> int:
    """Number of train/test splits, ``C(N, k)``."""
    _require_group_counts(n_groups, n_test_groups)
    return int(comb(n_groups, n_test_groups))


def cpcv_n_paths(n_groups: int, n_test_groups: int) -> int:
    """Number of reconstructed backtest paths, ``C(N-1, k-1)``.

    López de Prado (AFML, 2018, ch. 12) shows this equals
    ``C(N, k) * k / N``: each group is tested in that many splits, and each
    path tests the group exactly once.
    """
    _require_group_counts(n_groups, n_test_groups)
    return int(comb(n_groups - 1, n_test_groups - 1))


def cpcv_path_assignments(n_groups: int, n_test_groups: int) -> IntArray:
    """Assign each group on each backtest path to one split that tests it.

    Returns an integer array of shape ``(n_paths, n_groups)``. Entry
    ``[p, g]`` is the index of a combination in lexicographic order (the
    same order as :func:`itertools.combinations`) whose test set contains
    group ``g``. Each path covers every group once. Each incidence
    ``(split, group)`` with ``group`` in the split appears on exactly one
    path, so stitched paths do not reuse a test forecast.
    """
    n_paths = cpcv_n_paths(n_groups, n_test_groups)
    splits = list(combinations(range(n_groups), n_test_groups))
    by_group: list[list[int]] = [[] for _ in range(n_groups)]
    for split_id, combo in enumerate(splits):
        for group in combo:
            by_group[group].append(split_id)
    assignments = np.empty((n_paths, n_groups), dtype=np.int64)
    for group, split_ids in enumerate(by_group):
        if len(split_ids) != n_paths:
            raise RuntimeError("CPCV path incidence is not balanced")
        assignments[:, group] = np.asarray(split_ids, dtype=np.int64)
    return assignments


def stitch_group_paths(
    group_scores: NDArray[np.float64], assignments: IntArray
) -> NDArray[np.float64]:
    """Stitch per-split group scores into backtest paths.

    ``group_scores`` has shape ``(n_splits, n_groups)`` and stores the
    out-of-sample score of each group under each split. Only entries whose
    group belongs to that split are read. Returns shape ``(n_paths, n_groups)``.
    """
    scores = np.asarray(group_scores, dtype=float)
    paths = np.asarray(assignments)
    if scores.ndim != 2 or paths.ndim != 2:
        raise ValueError("group_scores and assignments must be 2-d")
    if paths.shape[1] != scores.shape[1]:
        raise ValueError("assignments and group_scores disagree on n_groups")
    if paths.size and int(paths.max()) >= scores.shape[0]:
        raise ValueError("assignment split id is outside group_scores")
    n_paths, n_groups = paths.shape
    stitched = np.empty((n_paths, n_groups), dtype=float)
    for group in range(n_groups):
        stitched[:, group] = scores[paths[:, group], group]
    return stitched


def combinatorial_purged_indices(
    n: int,
    n_groups: int,
    n_test_groups: int,
    *,
    label_horizon: int,
    embargo: int,
) -> list[tuple[IntArray, IntArray]]:
    """Index-space CPCV: every ``C(N, k)`` split, purged by label horizon.

    Observations are the contiguous positions ``0 .. n-1``, partitioned into
    ``n_groups`` sequential groups. For each combination of ``n_test_groups``,
    the test set is the union of those groups and the train set is the
    complement after per-group purge and embargo.

    Purge drops a train index ``i`` whose label window ``(i, i+label_horizon]``
    reaches a test block. Embargo drops the ``embargo`` bars immediately
    before and after each test block. Folds whose train set is emptied are
    omitted (the same fail-closed rule as :func:`combinatorial_purged_cv`).

    The label horizon is measured in observation steps, matching a session
    index on a gap-free timeline.
    """
    _require_group_counts(n_groups, n_test_groups)
    if label_horizon < 0:
        raise ValueError("label_horizon must be non-negative")
    if embargo < 0:
        raise ValueError("embargo must be non-negative")
    if n < n_groups:
        raise ValueError(f"need at least n_groups={n_groups} observations for CPCV, got {n}")
    bounds = [int(i * n / n_groups) for i in range(n_groups + 1)]
    groups = [np.arange(bounds[i], bounds[i + 1], dtype=np.int64) for i in range(n_groups)]
    if any(group.size == 0 for group in groups):
        raise ValueError("CPCV group bounds produced an empty group")
    folds: list[tuple[IntArray, IntArray]] = []
    for test_ids in combinations(range(n_groups), n_test_groups):
        chosen = set(test_ids)
        test = np.concatenate([groups[group] for group in test_ids])
        train_parts = [groups[group] for group in range(n_groups) if group not in chosen]
        train = np.concatenate(train_parts) if train_parts else np.empty(0, dtype=np.int64)
        if test.size == 0 or train.size == 0:
            continue
        keep = np.ones(train.size, dtype=bool)
        for group in test_ids:
            block = groups[group]
            lo = int(block[0])
            hi = int(block[-1])
            idx = train
            in_holdout = (idx >= lo) & (idx <= hi)
            label_reaches = (idx < lo) & ((idx + label_horizon) >= lo)
            embargo_after = (idx > hi) & (idx <= hi + embargo)
            embargo_before = (idx >= lo - embargo) & (idx < lo)
            keep &= ~(in_holdout | label_reaches | embargo_after | embargo_before)
        kept = train[keep]
        if kept.size == 0:
            continue
        folds.append((kept.astype(np.int64, copy=False), test.astype(np.int64, copy=False)))
    return folds
