"""P6.4 audit KATs: regression tests for findings in docs/AUDIT_P64_DIST.md."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.egarch import egarch_fit, gjr_garch_fit
from quant_fund.models.garch_ext import _fracdiff_weights, aparch_variance, figarch_variance
from quant_fund.models.localized_conformal import (
    LocalizedCQR,
    effective_sample_size,
    rbf_weights,
)
from quant_fund.models.regime import GaussianHMMRegime, VolThresholdRegime
from quant_fund.models.rough_vol import variance_curve_fit
from quant_fund.models.stoch_vol import stoch_vol_fit
from quant_fund.models.weighted_conformal import bench_weighted_cqr


def _figarch_lam_brute(phi: float, d: float, beta: float, n: int) -> np.ndarray:
    """lam_k via direct series expansion of 1 - (1-phi L)(1-L)^d/(1-beta L)."""
    pi = _fracdiff_weights(d, n + 1)
    f = np.zeros(n + 1)
    f[0] = 1.0
    for k in range(1, n + 1):
        f[k] = pi[k] - phi * pi[k - 1]
    lam = np.empty(n)
    for k in range(1, n + 1):
        lam[k - 1] = -sum(f[j] * beta ** (k - j) for j in range(k + 1))
    return lam


def _figarch_lam_from_variance(phi: float, d: float, beta: float, n: int) -> np.ndarray:
    """Recover lam weights: sigma2_t = omega_bar + sum_{k<=t} lam_{k-1} e2_{t-k};
    a unit impulse at t=0 contributes lam_{t-1} to sigma2_t."""
    s0 = 0.5
    s2_base = figarch_variance(np.zeros(n), phi, d, beta, sigma2_0=s0)
    e = np.zeros(n)
    e[0] = 1.0
    s2 = figarch_variance(e, phi, d, beta, sigma2_0=s0)
    return s2[1:] - s2_base[1:]


class TestFigarchLambda:
    def test_lambda_matches_series_expansion(self) -> None:
        n = 60
        lam = _figarch_lam_from_variance(0.2, 0.4, 0.3, n)
        np.testing.assert_allclose(lam, _figarch_lam_brute(0.2, 0.4, 0.3, n - 1), atol=1e-10)

    def test_lambda_weights_nonnegative_for_valid_params(self) -> None:
        # BBM positivity: with phi <= beta and these orders, lam_k >= 0.
        lam = _figarch_lam_from_variance(0.1, 0.5, 0.3, 60)
        assert np.all(lam >= 0.0)

    def test_pure_fracdiff_limit(self) -> None:
        # phi=beta=0: sigma2_t = s0 + e2_t - (1-L)^d e2_t => lam_k = -pi_k > 0.
        d = 0.4
        n = 60
        lam = _figarch_lam_from_variance(0.0, d, 0.0, n)
        pi = _fracdiff_weights(d, n)
        np.testing.assert_allclose(lam, -pi[1:n], atol=1e-12)


class TestAparchDivergence:
    def test_divergent_path_raises(self) -> None:
        v = np.zeros(100)
        v[10] = 1e100  # news^delta overflows -> sdelta = inf
        with pytest.raises(FloatingPointError):
            aparch_variance(v, 0.02, 0.5, 0.0, 0.4, 3.5)


class TestVarianceCurveFit:
    def test_lags_are_lag_values(self) -> None:
        x = np.random.default_rng(0).normal(size=400).cumsum()
        out = variance_curve_fit(x, lags=np.arange(1.0, 15.0))
        np.testing.assert_array_equal(out["lags"], np.arange(1.0, 15.0)[out["lags"] >= 0])
        assert np.all(out["lags"] >= 1.0)
        # lag values are integer-valued spacings, not log variances
        np.testing.assert_allclose(out["lags"], np.round(out["lags"]))


class TestLocalizedESSGate:
    def test_far_query_falls_back_to_global(self) -> None:
        # Calibration mass near 0; query at x=100 -> all raw RBF weights ~0
        # -> ESS ~ 0 -> must fall back to the global quantile even though
        # clip_weights would floor every weight at 1e-3 (ESS ~= n_cal).
        rng = np.random.default_rng(0)
        x_cal = rng.normal(0.0, 0.1, 200)
        y = rng.normal(size=200)
        lo, hi = np.full(200, -0.5), np.full(200, 0.5)
        m = LocalizedCQR(0.10, bandwidth=1.0, min_ess=12.0).calibrate(y, lo, hi, x_cal)
        raw = rbf_weights(x_cal, 100.0, m.bandwidth_)
        clipped_raw = np.clip(raw, 1e-3, 1e3)
        assert effective_sample_size(raw) < m.min_ess
        assert effective_sample_size(clipped_raw) >= m.min_ess  # the defeated gate
        plo, phi = m.predict_sets(np.array([0.0]), np.array([0.0]), np.array([100.0]))
        assert float(phi[0] - plo[0]) == pytest.approx(2.0 * m.global_qhat)


class TestWeightedBenchLabels:
    def test_bench_weighted_cqr_labels(self) -> None:
        row = bench_weighted_cqr(n_cal=200, n_test=100, seed=11)
        assert row["claim"] == "research_metric_only"
        assert row["dgp"] == "fixture"
        assert row["seed"] == 11.0


class TestQMLEConvergenceGuards:
    def test_egarch_raises_on_garbage_objective(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import quant_fund.models.egarch as eg

        class _Res:
            x = np.array([0.0, 0.0, 0.0, 0.5, 0.0])
            fun = 1e12
            success = False

        monkeypatch.setattr(eg.optimize, "minimize", lambda *a, **k: _Res())
        with pytest.raises(ValueError, match="converge"):
            egarch_fit(np.random.default_rng(0).normal(size=150))

    def test_gjr_raises_on_garbage_objective(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import quant_fund.models.egarch as eg

        class _Res:
            x = np.array([0.01, 0.05, 0.05, 0.5, 0.0])
            fun = 1e12
            success = False

        monkeypatch.setattr(eg.optimize, "minimize", lambda *a, **k: _Res())
        with pytest.raises(ValueError, match="converge"):
            gjr_garch_fit(np.random.default_rng(0).normal(size=150))

    def test_stoch_vol_raises_on_garbage_objective(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import quant_fund.models.stoch_vol as sv

        class _Res:
            x = np.array([0.0, 0.5, 0.3])
            fun = 1e12
            success = False

        monkeypatch.setattr(sv.optimize, "minimize", lambda *a, **k: _Res())
        with pytest.raises(ValueError, match="converge"):
            stoch_vol_fit(np.random.default_rng(0).normal(scale=0.01, size=150) + 1e-9)


class TestVolThresholdFailClosed:
    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            VolThresholdRegime().predict_proba(np.zeros((3, 1)))

    def test_all_nan_vol_raises(self) -> None:
        with pytest.raises(ValueError):
            VolThresholdRegime().fit(np.full((10, 1), np.nan))


class TestHMMScoringFailClosed:
    def test_aic_bic_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            GaussianHMMRegime(n_states=2).aic_bic(np.zeros((20, 2)))
