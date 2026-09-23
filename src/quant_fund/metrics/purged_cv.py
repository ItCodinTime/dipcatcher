"""Purged and embargoed k-fold cross-validation
(Lopez de Prado 2018, ch. 7).

Financial labels derived from overlapping windows (e.g. triple-barrier
outcomes with t1 end times) leak information when a train sample's
label interval overlaps a test sample's interval. This module yields
(train, test) index arrays where, for each test block:

  1. Purge: drop train indices i whose label span [t_i, t1_i]
     intersects the test span.
  2. Embargo: additionally drop train indices in the embargo window
     immediately following the test block.

Fail-closed: mismatched lengths, non-integer folds, t1 < t.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def purged_kfold_indices(
    n: int,
    n_folds: int = 5,
    t1: Array | None = None,
    embargo_frac: float = 0.0,
) -> list[tuple[Array, Array]]:
    """Yield (train_idx, test_idx) for purged embargoed k-fold.

    ``t1`` is the integer array of label end indices (exclusive) for
    each sample; default t1 = i+1 (non-overlapping labels -> plain
    contiguous k-fold). ``embargo_frac`` is the fraction of n embargoed
    after each test block.
    """
    if n_folds < 2 or n < 2 * n_folds:
        raise ValueError("need n >= 2 * n_folds")
    if t1 is None:
        t1a = np.arange(1, n + 1)
    else:
        t1a = np.asarray(t1, dtype=float).ravel()
        if t1a.size != n or not np.isfinite(t1a).all():
            raise ValueError("t1 must be finite and length n")
        if (t1a < np.arange(1, n + 1)).any():
            raise ValueError("t1 must be >= i+1 (labels look ahead)")
        t1a = np.minimum(np.round(t1a), n)
    if embargo_frac < 0 or embargo_frac >= 0.5:
        raise ValueError("embargo_frac must be in [0, 0.5)")
    embargo = int(np.floor(embargo_frac * n))

    bounds = np.linspace(0, n, n_folds + 1).astype(int)
    out: list[tuple[Array, Array]] = []
    for k in range(n_folds):
        test = np.arange(bounds[k], bounds[k + 1])
        t_start, t_end = bounds[k], bounds[k + 1]
        # purge: train samples whose label interval [i, t1_i) overlaps
        # [t_start, t_end)
        starts = np.arange(n)
        overlap = (starts < t_end) & (t1a > t_start)
        train_mask = ~overlap
        # embargo: drop the window after the test block
        if embargo > 0:
            emb_lo = t_end
            emb_hi = min(n, t_end + embargo)
            train_mask[emb_lo:emb_hi] = False
        train = np.nonzero(train_mask & ~np.isin(starts, test))[0]
        out.append((train, test))
    return out


def purged_cv_score(
    x: Array,
    y: Array,
    n_folds: int = 5,
    t1: Array | None = None,
    embargo_frac: float = 0.01,
) -> dict[str, Array | float]:
    """Ridge-regression purged-CV R2 (deterministic reference scorer).

    Useful as a leakage-safe default scorer: ordinary k-fold on
    overlapping labels inflates OOS R2; the purged score does not.
    """
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float).ravel()
    if xx.ndim == 1:
        xx = xx[:, None]
    if xx.shape[0] != yy.size or not np.isfinite(xx).all() or not np.isfinite(yy).all():
        raise ValueError("non-finite or misaligned x, y")
    n = yy.size
    splits = purged_kfold_indices(n, n_folds, t1, embargo_frac)
    preds = np.full(n, np.nan)
    for train, test in splits:
        xt = np.column_stack([np.ones(train.size), xx[train]])
        beta = np.linalg.solve(xt.T @ xt + 1e-6 * np.eye(xt.shape[1]), xt.T @ yy[train])
        xv = np.column_stack([np.ones(test.size), xx[test]])
        preds[test] = xv @ beta
    valid = np.isfinite(preds)
    ss_res = float(np.sum((yy[valid] - preds[valid]) ** 2))
    ss_tot = float(np.sum((yy[valid] - yy[valid].mean()) ** 2))
    return {
        "r2": 1.0 - ss_res / max(ss_tot, 1e-14),
        "preds": preds,
        "n_test": float(valid.sum()),
    }
