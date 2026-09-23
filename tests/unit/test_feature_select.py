"""Tests for metrics/feature_select.py — mRMR + screens."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.feature_select import (
    forward_orthogonalized,
    mrmr_select,
    univariate_screen,
)


def _feature_panel(
    n: int = 400, p: int = 20, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """p features; first 3 are true signals, rest are correlated noise."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, 3))
    x = np.empty((n, p))
    x[:, :3] = base + 0.1 * rng.standard_normal((n, 3))
    # noise features highly correlated with signal 0 (redundant)
    for j in range(3, p):
        x[:, j] = base[:, j % 3] * 0.9 + rng.standard_normal(n)
    y = 2.0 * x[:, 0] - 1.5 * x[:, 1] + 1.0 * x[:, 2] + 0.5 * rng.standard_normal(n)
    true_idx = np.array([0, 1, 2])
    return x, y, true_idx


def test_mrmr_finds_signals() -> None:
    x, y, true_idx = _feature_panel()
    out = mrmr_select(x, y, k=3)
    sel = set(np.asarray(out["selected"], dtype=int).tolist())
    # mRMR should prefer the true signals over redundant copies
    assert len(sel & set(true_idx.tolist())) >= 2


def test_univariate_screen() -> None:
    x, y, true_idx = _feature_panel(seed=1)
    out = univariate_screen(x, y, k=3)
    sel = set(np.asarray(out["selected"], dtype=int).tolist())
    assert len(sel & set(true_idx.tolist())) >= 1


def test_forward_orthogonalized() -> None:
    x, y, true_idx = _feature_panel(seed=2)
    out = forward_orthogonalized(x, y, k=3)
    sel = set(np.asarray(out["selected"], dtype=int).tolist())
    assert len(sel & set(true_idx.tolist())) >= 2


def test_mrmr_penalizes_redundancy() -> None:
    # x2 = x1 + eps (redundant): mRMR should not pick both when a
    # distinct signal exists
    rng = np.random.default_rng(3)
    n = 300
    x1 = rng.standard_normal(n)
    x2 = x1 + 0.01 * rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    x = np.column_stack([x1, x2, x3])
    y = x1 + x3 + 0.1 * rng.standard_normal(n)
    out = mrmr_select(x, y, k=2)
    sel = set(np.asarray(out["selected"], dtype=int).tolist())
    # should pick one of {0,1} plus feature 2
    assert 2 in sel
    assert not ({0, 1} <= sel)


def test_fail_closed() -> None:
    x, y, _ = _feature_panel()
    with pytest.raises(ValueError):
        mrmr_select(x, y, k=0)
    with pytest.raises(ValueError):
        mrmr_select(x, y, k=99)
    bad = x.copy()
    bad[:, 0] = 1.0
    with pytest.raises(ValueError):
        mrmr_select(bad, y, 3)
    with pytest.raises(ValueError):
        univariate_screen(x, np.full_like(y, np.nan), 2)
