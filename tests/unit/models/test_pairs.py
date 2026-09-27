"""Tests for pairs selection and spread diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.pairs import (
    cointegration_screen,
    gatev_distance,
    gatev_select,
    ou_optimal_bands,
    pair_quality_score,
    zscore_spread_stats,
)


def _make_panel(n: int = 600, n_assets: int = 8, seed: int = 0):
    """One strongly cointegrated pair + independent random walks."""
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.normal(size=n))
    spr = np.zeros(n)
    for t in range(1, n):
        spr[t] = 0.85 * spr[t - 1] + rng.normal(scale=0.2)
    p = np.zeros((n, n_assets))
    p[:, 0] = np.exp(base / 10 + 3)
    p[:, 1] = np.exp((base + spr) / 10 + 3)
    for i in range(2, n_assets):
        p[:, i] = np.exp(np.cumsum(rng.normal(size=n)) / 10 + 3)
    return p


class TestGatev:
    def test_distance_matrix(self):
        p = _make_panel()
        ssd = gatev_distance(p)
        assert ssd.shape == (8, 8)
        assert np.allclose(np.diag(ssd), 0.0)
        assert np.allclose(ssd, ssd.T)
        # The cointegrated pair should have the lowest SSD.
        assert ssd[0, 1] == pytest.approx(np.min(ssd[np.triu_indices(8, 1)]))

    def test_select_top_pairs(self):
        p = _make_panel()
        out = gatev_select(p, n_pairs=3)
        assert out["pairs"].shape == (3, 2)
        assert set(out["pairs"][0]) == {0, 1}
        assert np.all(np.diff(out["distances"]) >= 0)

    def test_failclosed(self):
        with pytest.raises(ValueError):
            gatev_distance(np.ones((10, 3)))
        with pytest.raises(ValueError):
            gatev_select(np.ones((50, 3)), n_pairs=100)


class TestScreen:
    def test_cointegration_screen_finds_pair(self):
        p = _make_panel()
        out = cointegration_screen(p)
        assert out["passes"][0, 1]
        assert out["tau"][0, 1] < -2.5
        # Most independent pairs should not pass.
        indep_pass = out["passes"][2:, 2:].sum()
        assert indep_pass <= 6

    def test_zscore_stats(self):
        p = _make_panel()
        out = zscore_spread_stats(p[:, 0], p[:, 1])
        assert abs(out["hedge_ratio"] - 1.0) < 0.3
        assert out["stationary_5pct"] == 1.0
        assert 2 < out["half_life"] < 60
        assert out["zero_crossings"] > 5

    def test_pair_quality(self):
        p = _make_panel()
        good = pair_quality_score(p[:, 0], p[:, 1])
        bad = pair_quality_score(p[:, 2], p[:, 3])
        assert good["score"] > bad["score"]
        assert 0 <= good["score"] <= 1.0


class TestOU:
    def test_optimal_bands_ordered(self):
        out = ou_optimal_bands(theta=0.5, mu=0.0, sigma=0.1, r=0.01)
        assert np.isfinite(out["entry"])
        assert out["entry"] < out["exit"]
        assert out["entry"] < 0.0  # buy below the mean

    def test_failclosed(self):
        with pytest.raises(ValueError):
            ou_optimal_bands(theta=-1.0, mu=0.0, sigma=0.1)
        with pytest.raises(ValueError):
            zscore_spread_stats(np.ones(20), np.ones(20))
