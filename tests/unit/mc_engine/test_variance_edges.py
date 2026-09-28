"""mc_engine.variance edge-branch coverage: every fail-closed return,
validation raise, and degenerate-variance path."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.mc_engine import variance as V


class TestCrossStats:
    def test_sums_match_direct_computation(self) -> None:
        rng = np.random.default_rng(0)
        y, x = rng.normal(size=50), rng.normal(size=50)
        s = V.cross_stats(y, x)
        assert s[0] == 50
        assert s[1] == pytest.approx(y.sum())
        assert s[5] == pytest.approx((y * x).sum())

    def test_empty_and_mismatch(self) -> None:
        assert V.cross_stats(np.array([]), np.array([])) == (0, 0.0, 0.0, 0.0, 0.0, 0.0)
        with pytest.raises(ValueError, match="same shape"):
            V.cross_stats(np.ones(3), np.ones(4))
        with pytest.raises(ValueError, match="finite"):
            V.cross_stats(np.array([np.nan]), np.array([1.0]))

    def test_add_cross(self) -> None:
        a = V.cross_stats(np.ones(3), np.ones(3))
        b = V.cross_stats(np.ones(2) * 2, np.ones(2) * 2)
        s = V.add_cross(a, b)
        assert s[0] == 5 and s[1] == pytest.approx(7.0)


class TestAntithetic:
    def test_perfect_pairs_infinite(self) -> None:
        rng = np.random.default_rng(1)
        base = rng.normal(size=50)
        vals = np.column_stack([base, -base]).ravel()
        out = V.antithetic_mean_vrf(vals)
        assert out["variance_reduction_factor_infinite"] is True
        assert out["variance_reduction_factor"] is None

    def test_uncorrelated_pairs_finite(self) -> None:
        rng = np.random.default_rng(2)
        vals = rng.normal(size=200)
        out = V.antithetic_mean_vrf(vals)
        factor = out["variance_reduction_factor"]
        assert isinstance(factor, float) and factor > 0

    def test_too_few_pairs(self) -> None:
        out = V.antithetic_mean_vrf(np.array([1.0, -1.0]))
        assert out["variance_reduction_factor"] is None
        assert "2 antithetic pairs" in str(out["reason"])


class TestControlVariate:
    def _stats(self, seed: int, n: int, b: float = 0.8) -> V.CrossStats:
        rng = np.random.default_rng(seed)
        x = rng.normal(2.0, 1.0, n)
        y = b * (x - 2.0) + rng.normal(0, 0.1, n)
        return V.cross_stats(y, x)

    def test_strong_control_large_vrf(self) -> None:
        out = V.control_variate_from_stats(
            self._stats(1, 200), self._stats(2, 200), control_mean=2.0
        )
        factor = out["variance_reduction_factor"]
        assert isinstance(factor, float) and factor > 10.0
        assert out["coefficient"] == pytest.approx(0.8, abs=0.1)

    def test_edge_branches(self) -> None:
        pilot = self._stats(1, 200)
        empty = V._empty_cross()
        assert V.control_variate_from_stats(empty, pilot, 0.0)["reason"] == (
            "pilot sample smaller than 2"
        )
        # degenerate control: x constant -> den == 0
        const = V.cross_stats(np.arange(10.0), np.ones(10))
        out = V.control_variate_from_stats(const, pilot, 1.0)
        assert "zero variation" in str(out["reason"])
        # tiny evaluation
        tiny_eval = V.cross_stats(np.array([1.0]), np.array([1.0]))
        out = V.control_variate_from_stats(pilot, tiny_eval, 2.0)
        assert "evaluation sample smaller than 2" in str(out["reason"])
        with pytest.raises(ValueError, match="finite"):
            V.control_variate_from_stats(pilot, pilot, float("nan"))

    def test_exact_cancellation_infinite(self) -> None:
        rng = np.random.default_rng(5)
        x = rng.normal(0.0, 1.0, 100)
        pilot = V.cross_stats(x + rng.normal(0, 0.3, 100), x)
        # evaluation y identical to shifted x -> adjusted residuals ~ const
        eval_stats = V.cross_stats(x.copy(), x.copy())
        out = V.control_variate_from_stats(pilot, eval_stats, control_mean=0.0)
        # y - b*(x-mu) = (1-b)x + mu*b has variance ~0 when b~1
        assert (
            out["variance_reduction_factor"] is not None
            or out["variance_reduction_factor_infinite"]
        )


class TestImportanceSampling:
    def test_shift_vector_factor0_only(self) -> None:
        mu = V.importance_shift_vector(4, 3, -0.5)
        assert mu.shape == (12,)
        assert np.allclose(mu[0::3], -0.5)
        assert np.allclose(np.delete(mu, np.arange(0, 12, 3)), 0.0)
        with pytest.raises(ValueError, match="positive"):
            V.importance_shift_vector(0, 3, 0.1)
        with pytest.raises(ValueError, match="finite"):
            V.importance_shift_vector(2, 2, float("inf"))

    def test_weights_formula(self) -> None:
        rng = np.random.default_rng(0)
        z = rng.normal(size=(10, 6))
        mu = np.full(6, 0.1)
        w = V.importance_weights(z, mu)
        lw = V.importance_log_weights(z, mu)
        assert np.allclose(np.log(w), lw)
        # analytic: w = exp(-mu·z + 0.5||mu||^2)
        expect = np.exp(-z @ mu + 0.5 * mu @ mu)
        assert np.allclose(w, expect)
        with pytest.raises(ValueError, match="shape"):
            V.importance_weights(rng.normal(size=(4, 3)), np.ones(5))
        with pytest.raises(ValueError, match="finite"):
            V.importance_weights(np.array([[np.nan] * 6]), np.ones(6))

    def test_ess_edges(self) -> None:
        assert V.effective_sample_size(np.array([])) == 0.0
        assert V.effective_sample_size(np.zeros(4)) == 0.0
        assert V.effective_sample_size(np.ones(10)) == pytest.approx(10.0)
        w = np.array([1.0, 0.0, 0.0, 0.0])
        assert V.effective_sample_size(w) == pytest.approx(1.0)
        with pytest.raises(ValueError, match="non-negative"):
            V.effective_sample_size(np.array([1.0, -0.5]))

    def test_mean_vrf_edges(self) -> None:
        rng = np.random.default_rng(3)
        crude = rng.normal(size=100)
        out = V.importance_mean_vrf(crude[:1], crude[:1])
        assert "same length >= 2" in str(out["reason"])
        out = V.importance_mean_vrf(crude, np.full(100, np.inf))
        assert "non-finite" in str(out["reason"])
        # zero-variance weighted outcomes -> infinite
        out = V.importance_mean_vrf(crude, np.ones(100))
        assert out["variance_reduction_factor_infinite"] is True
        # normal case
        w = rng.normal(size=100) * 0.3
        out = V.importance_mean_vrf(crude, w)
        assert out["variance_reduction_factor"] is not None


class TestRqmc:
    def test_edges_and_factor(self) -> None:
        rng = np.random.default_rng(4)
        crude = rng.normal(size=64)
        assert "at least 2" in str(V.rqmc_mean_vrf(np.array([1.0]), crude)["reason"])
        assert "non-finite" in str(V.rqmc_mean_vrf(np.array([np.nan, 1.0]), crude)["reason"])
        tight = np.full(8, 0.5)
        out = V.rqmc_mean_vrf(tight, crude)
        assert out["variance_reduction_factor_infinite"] is True
        means = rng.normal(0, 0.1, 8)
        out = V.rqmc_mean_vrf(means, crude)
        assert out["variance_reduction_factor"] is not None
        assert out["n_scrambles"] == 8


class TestSobolAndDraws:
    def test_sobol_validation(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            V.sobol_normals(-1, 4, 2, seed=0, scramble=True)
        with pytest.raises(ValueError, match="non-negative"):
            V.sobol_normals(0, -1, 2, seed=0, scramble=True)
        with pytest.raises(ValueError, match="positive"):
            V.sobol_normals(0, 4, 0, seed=0, scramble=True)
        with pytest.raises(ValueError, match="exceeds scipy"):
            V.sobol_normals(0, 4, 99999, seed=0, scramble=True)
        with pytest.raises(ValueError, match="non-negative"):
            V.sobol_normals(0, 4, 2, seed=-1, scramble=True)
        with pytest.raises(ValueError, match="non-negative"):
            V.sobol_normals(True, 4, 2, seed=0, scramble=True)  # bool rejected

    def test_sobol_slice_and_empty(self) -> None:
        a = V.sobol_normals(0, 8, 3, seed=11, scramble=False)
        b = V.sobol_normals(0, 16, 3, seed=0, scramble=False)
        assert np.allclose(a, b[:8])  # contiguous slices consistent
        assert V.sobol_normals(0, 0, 3, seed=0, scramble=True).shape == (0, 3)
        scr1 = V.sobol_normals(0, 8, 3, seed=7, scramble=True)
        scr2 = V.sobol_normals(0, 8, 3, seed=7, scramble=True)
        assert np.allclose(scr1, scr2)

    def test_draw_modes(self) -> None:
        idx = np.arange(6, dtype=np.int64)
        s, w = V.draw_standard_normals(
            idx,
            3,
            2,
            seed=1,
            shock_mode="crude",
            importance_shift=0.0,
            qmc_scramble=True,
            qmc_seed=5,
        )
        assert s.shape == (6, 3, 2) and w is None
        s2, _ = V.draw_standard_normals(
            idx,
            3,
            2,
            seed=1,
            shock_mode="qmc_sobol",
            importance_shift=0.0,
            qmc_scramble=True,
            qmc_seed=5,
        )
        assert s2.shape == (6, 3, 2)
        sa, _ = V.draw_standard_normals(
            idx,
            3,
            2,
            seed=1,
            shock_mode="antithetic",
            importance_shift=0.0,
            qmc_scramble=True,
            qmc_seed=5,
        )
        assert np.allclose(sa[1], -sa[0])  # antithetic pair negation
        si, wi = V.draw_standard_normals(
            idx,
            3,
            2,
            seed=1,
            shock_mode="importance",
            importance_shift=-0.1,
            qmc_scramble=True,
            qmc_seed=5,
        )
        assert wi is not None and wi.shape == (6,)
        assert (wi > 0).all()

    def test_draw_errors(self) -> None:
        with pytest.raises(ValueError, match="unknown shock_mode"):
            V.draw_standard_normals(
                np.arange(4),
                2,
                2,
                seed=0,
                shock_mode="bogus",
                importance_shift=0.0,
                qmc_scramble=True,
                qmc_seed=0,
            )
        with pytest.raises(ValueError, match="contiguous"):
            V.draw_standard_normals(
                np.array([0, 2, 5]),
                2,
                2,
                seed=0,
                shock_mode="qmc_sobol",
                importance_shift=0.0,
                qmc_scramble=True,
                qmc_seed=0,
            )
        with pytest.raises(ValueError, match="even/odd"):
            V.draw_standard_normals(
                np.array([1, 2, 3]),
                2,
                2,
                seed=0,
                shock_mode="antithetic",
                importance_shift=0.0,
                qmc_scramble=True,
                qmc_seed=0,
            )
        # empty index array is allowed for qmc
        s, _ = V.draw_standard_normals(
            np.array([], dtype=np.int64),
            2,
            2,
            seed=0,
            shock_mode="qmc_sobol",
            importance_shift=0.0,
            qmc_scramble=True,
            qmc_seed=0,
        )
        assert s.shape == (0, 2, 2)
