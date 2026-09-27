"""Tests for metrics/purged_cv.py — Lopez de Prado purged k-fold."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.purged_cv import purged_cv_score, purged_kfold_indices


def test_no_overlap_purge() -> None:
    n, k = 100, 5
    # labels span 10 obs -> overlapping; purge must drop boundary trains
    t1 = np.minimum(np.arange(1, n + 1) + 9, n)
    splits = purged_kfold_indices(n, k, t1=t1)
    assert len(splits) == k
    for train, test in splits:
        assert np.intersect1d(train, test).size == 0
        # every train sample's label interval must not intersect test span
        t_start, t_end = test[0], test[-1] + 1
        for i in train:
            assert i + 9 <= t_start or i >= t_end


def test_embargo_drops_following_window() -> None:
    n, k = 200, 4
    splits = purged_kfold_indices(n, k, embargo_frac=0.05)
    embargo = int(0.05 * n)
    for train, test in splits:
        emb_lo = test[-1] + 1
        emb_hi = min(n, emb_lo + embargo)
        assert np.intersect1d(train, np.arange(emb_lo, emb_hi)).size == 0


def test_plain_kfold_when_no_overlap() -> None:
    n = 100
    splits = purged_kfold_indices(n, 5)  # t1 default -> non-overlapping
    # union of all train+test should cover everything, test disjoint
    all_idx = np.concatenate([np.concatenate(s) for s in splits])
    assert set(np.unique(all_idx)) == set(range(n))


def test_purged_cv_score() -> None:
    rng = np.random.default_rng(0)
    n = 300
    x = rng.standard_normal(n)
    # AR-ish target with overlapping labels
    y = np.convolve(x, np.ones(5) / 5, mode="same") + 0.2 * rng.standard_normal(n)
    t1 = np.minimum(np.arange(1, n + 1) + 4, n)
    out = purged_cv_score(x, y, n_folds=5, t1=t1, embargo_frac=0.02)
    assert np.isfinite(out["r2"])
    assert out["n_test"] == float(n)


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        purged_kfold_indices(10, 8)  # too many folds
    with pytest.raises(ValueError):
        purged_kfold_indices(100, 5, t1=np.arange(1, 100))  # t1 < i+1
    with pytest.raises(ValueError):
        purged_kfold_indices(100, 5, embargo_frac=0.6)
    with pytest.raises(ValueError):
        purged_cv_score(np.full(50, np.nan), np.arange(50.0))
