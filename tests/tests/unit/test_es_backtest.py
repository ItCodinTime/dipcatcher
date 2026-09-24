"""Tests for ES backtesting."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.es_backtest import (
    acerbi_szekely_test,
    du_escanciano_test,
    exceedance_residuals,
    mcneil_frey_test,
)


def _calibrated(n=800, seed=0, alpha=0.99):
    """Losses ~ Student-t-ish; VaR/ES set to the correct tail quantiles."""
    rng = np.random.default_rng(seed)
    from scipy import stats as st

    r = st.t.rvs(df=6, size=n, random_state=rng)
    var = np.full(n, st.t.ppf(alpha, df=6))
    # t ES closed form for loss convention.
    v = var[0]
    es = np.full(n, (st.t.pdf(v, df=6) / (1 - alpha)) * ((6 + v * v) / 5.0))
    return r, var, es


class TestAcerbiSzekely:
    def test_correct_es_not_rejected(self):
        r, v, e = _calibrated()
        out = acerbi_szekely_test(r, v, e, n_boot=300, seed=1)
        assert 0 <= out["pvalue"] <= 1
        assert abs(out["mean_ratio"] - 1.0) < 0.4

    def test_overestimated_es_detected(self):
        r, v, e = _calibrated()
        e_big = e * 1.5  # ES too deep in the tail -> mean ratio < 1
        out = acerbi_szekely_test(r, v, e_big, n_boot=300, seed=2)
        assert out["mean_ratio"] < 0.85

    def test_failclosed(self):
        r, v, e = _calibrated()
        with pytest.raises(ValueError):
            acerbi_szekely_test(r[:10], v[:10], e[:10])
        with pytest.raises(ValueError):
            acerbi_szekely_test(r, v, e * 0.5)  # ES < VaR violates ordering


class TestMcNeilFrey:
    def test_correct_not_rejected(self):
        r, v, e = _calibrated(seed=3)
        out = mcneil_frey_test(r, v, e, n_boot=300, seed=4)
        assert 0 <= out["pvalue"] <= 1
        assert out["n_exceedances"] >= 3

    def test_failclosed(self):
        with pytest.raises(ValueError):
            mcneil_frey_test(np.ones(60), np.ones(60), np.ones(60) * 2)


class TestExceedanceResiduals:
    def test_unit_mean_for_calibrated(self):
        r, v, e = _calibrated(seed=5)
        out = exceedance_residuals(r, v, e)
        assert abs(out["mean"] - 1.0) < 0.6
        assert np.all(out["residuals"] > 0)

    def test_failclosed(self):
        with pytest.raises(ValueError):
            exceedance_residuals(np.ones(40), np.ones(40) * 2, np.ones(40))


class TestDuEscanciano:
    def test_iid_hits_pass(self):
        rng = np.random.default_rng(6)
        hits = (rng.random(500) < 0.01).astype(float)
        out = du_escanciano_test(hits, alpha=0.99)
        assert out["pvalue"] > 0.01

    def test_clustered_hits_rejected(self):
        h = np.zeros(500)
        h[50:70] = 1.0  # violation cluster
        out = du_escanciano_test(h, alpha=0.90)
        assert out["pvalue"] < 0.05

    def test_failclosed(self):
        with pytest.raises(ValueError):
            du_escanciano_test(np.zeros(100))
