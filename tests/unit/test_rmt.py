"""Tests for random-matrix-theory module."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.rmt import (
    absorption_ratio,
    correlation_eigenvalues,
    detone_cov,
    effective_rank,
    eigenvalue_clip,
    inverse_participation_ratio,
    marchenko_pastur_bounds,
    marchenko_pastur_pdf,
    noise_fraction,
    risk_in_eigenmodes,
)


class TestMP:
    def test_bounds(self):
        lo, hi = marchenko_pastur_bounds(0.5)
        assert abs(lo - (1 - np.sqrt(0.5)) ** 2) < 1e-12
        assert abs(hi - (1 + np.sqrt(0.5)) ** 2) < 1e-12
        assert lo < 1.0 < hi
        # sigma2 scales both edges.
        lo2, hi2 = marchenko_pastur_bounds(0.5, sigma2=4.0)
        assert np.isclose(lo2 / lo, 4.0) and np.isclose(hi2 / hi, 4.0)

    def test_pdf_integrates(self):
        lam = np.linspace(0.05, 3.0, 4000)
        pdf = marchenko_pastur_pdf(lam, 0.5)
        mass = np.trapezoid(pdf, lam)
        assert 0.95 < mass < 1.01

    def test_pdf_zero_outside(self):
        lo, hi = marchenko_pastur_bounds(0.5)
        assert marchenko_pastur_pdf(np.array([lo - 0.5]), 0.5)[0] == 0.0
        assert marchenko_pastur_pdf(np.array([hi + 0.5]), 0.5)[0] == 0.0

    def test_failclosed(self):
        with pytest.raises(ValueError):
            marchenko_pastur_bounds(0.0)
        with pytest.raises(ValueError):
            marchenko_pastur_bounds(0.5, sigma2=-1.0)


class TestEigenDecomp:
    def test_iid_data_eigs_in_band(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(400, 200))  # q = 0.5
        vals, vecs = correlation_eigenvalues(x)
        assert vals.shape == (200,)
        assert vecs.shape == (200, 200)
        # For pure noise, most eigenvalues inside MP band.
        frac = noise_fraction(vals, q=0.5)
        assert frac > 0.85

    def test_spiked_data_has_outliers(self):
        rng = np.random.default_rng(1)
        f = rng.normal(size=(400, 1))
        beta = rng.normal(size=(1, 30))
        x = f @ beta + 0.3 * rng.normal(size=(400, 30))
        vals, _ = correlation_eigenvalues(x)
        lo, hi = marchenko_pastur_bounds(30.0 / 400.0)
        assert vals[0] > hi  # market factor pops out of the MP band

    def test_failclosed(self):
        with pytest.raises(ValueError):
            correlation_eigenvalues(np.ones((5, 3)))
        with pytest.raises(ValueError):
            correlation_eigenvalues(np.random.default_rng(0).normal(size=(50, 3)) * 0.0)


class TestDenoising:
    def test_clip_preserves_trace(self):
        rng = np.random.default_rng(2)
        a = rng.normal(size=(30, 30))
        cov = a.T @ a / 30.0 + np.eye(30) * 0.5
        out = eigenvalue_clip(cov, q=0.5)
        assert np.isclose(np.trace(out), np.trace(cov), rtol=0.05)
        assert np.all(np.linalg.eigvalsh(out) >= -1e-8)

    def test_detone_kills_market(self):
        rng = np.random.default_rng(3)
        f = rng.normal(size=(400, 1))
        x = f @ np.ones((1, 10)) + 0.2 * rng.normal(size=(400, 10))
        cov = np.cov(x.T)
        d = detone_cov(cov, n_market=1)
        vals = np.linalg.eigvalsh(cov)
        dvals = np.linalg.eigvalsh(d)
        assert dvals[-1] < vals[-1] * 0.2  # top eigenvalue removed

    def test_failclosed(self):
        with pytest.raises(ValueError):
            eigenvalue_clip(np.ones((3, 4)), 0.5)
        with pytest.raises(ValueError):
            eigenvalue_clip(np.array([[1.0, 0.9], [0.1, 1.0]]), 0.5)  # asymmetric
        with pytest.raises(ValueError):
            detone_cov(np.eye(4), n_market=4)
        with pytest.raises(ValueError):
            detone_cov(np.eye(4), n_market=0)


class TestDiagnostics:
    def test_absorption_ratio(self):
        v = np.array([4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        ar = absorption_ratio(v, top_k=1)
        assert abs(ar - 4.0 / 10.0) < 1e-12
        assert 0.0 < absorption_ratio(v) < 1.0

    def test_effective_rank_bounds(self):
        uniform = np.ones(10)
        assert abs(effective_rank(uniform) - 10.0) < 1e-6
        spike = np.array([100.0, 0.01, 0.01, 0.01, 0.01])
        assert effective_rank(spike) < 2.0

    def test_ipr(self):
        delocalized = np.ones(16) / 4.0
        assert abs(inverse_participation_ratio(delocalized) - 1.0 / 16.0) < 1e-12
        localized = np.zeros(16)
        localized[0] = 1.0
        assert inverse_participation_ratio(localized) == 1.0

    def test_risk_in_eigenmodes_sums_to_one(self):
        rng = np.random.default_rng(4)
        a = rng.normal(size=(8, 8))
        cov = a.T @ a + np.eye(8)
        w = rng.normal(size=8)
        shares = risk_in_eigenmodes(w, cov)
        assert shares.shape == (8,)
        assert np.isclose(shares.sum(), 1.0)

    def test_failclosed(self):
        with pytest.raises(ValueError):
            absorption_ratio(np.array([-1.0, 2.0, 3.0]))
        with pytest.raises(ValueError):
            effective_rank(np.zeros(5))
        with pytest.raises(ValueError):
            inverse_participation_ratio(np.zeros(4))
        with pytest.raises(ValueError):
            risk_in_eigenmodes(np.zeros(3), np.eye(3))
