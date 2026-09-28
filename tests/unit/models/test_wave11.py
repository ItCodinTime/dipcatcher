"""Canon wave 11 tests: bvar, msgarch, qvar, predictive_regression,
port_sorts, smooth_transition. Deterministic seeds; known-answer and
invariant checks; degenerate-input raises."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models import (
    bvar,
    msgarch,
    port_sorts,
    predictive_regression,
    qvar,
    smooth_transition,
)


def _var_panel(n: int = 3, t: int = 160, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = 0.4 * np.eye(n)
    e = rng.normal(0.0, 0.2, (t, n))
    y = np.zeros((t, n))
    y[0] = e[0]
    for i in range(1, t):
        y[i] = a @ y[i - 1] + e[i]
    return y


class TestBvar:
    def test_fit_shapes_and_prior_pull(self):
        y = _var_panel()
        out = bvar.bvar_fit(y, p=2, theta=0.3)
        assert out["B"].shape == (1 + 3 * 2, 3)
        assert out["sigma_u"].shape == (3, 3)
        assert out["n_obs"] == 158
        # tighter prior shrinks coefficients toward the rw prior harder
        tight = bvar.bvar_fit(y, p=2, theta=0.02)
        cross = np.abs(tight["B"][3:].ravel() - np.zeros_like(tight["B"][3:].ravel()))
        cross_l = np.abs(out["B"][3:].ravel())
        assert cross.mean() < cross_l.mean()

    def test_irf_diagonal(self):
        y = _var_panel()
        f = bvar.bvar_fit(y, p=1)
        irf = bvar.bvar_irf(f["B"], f["sigma_u"], n_steps=10, p=1)
        assert irf.shape == (10, 3, 3)
        # shock j only moves its own equation at h=0 for cholesky-first var
        assert irf[0, 0, 0] > 0

    def test_fevd_rows_sum(self):
        y = _var_panel()
        f = bvar.bvar_fit(y, p=1)
        fevd = bvar.bvar_fevd(f["B"], f["sigma_u"], n_steps=8, p=1)
        assert np.allclose(np.nansum(fevd, axis=2)[-1], 1.0)

    def test_forecast_shapes(self):
        y = _var_panel()
        f = bvar.bvar_fit(y, p=2)
        fc = bvar.bvar_forecast(f["B"], y[-2:], p=2, n_steps=6)
        assert fc.shape == (6, 3)

    def test_validation(self):
        with pytest.raises(ValueError, match="T, N"):
            bvar.bvar_fit(np.ones(10))
        with pytest.raises(ValueError, match="observations"):
            bvar.bvar_fit(np.ones((6, 3)), p=2)
        with pytest.raises(ValueError, match="theta"):
            bvar.bvar_fit(_var_panel(), p=1, theta=0.0)
        f = bvar.bvar_fit(_var_panel(), p=1)
        with pytest.raises(ValueError, match="inconsistent"):
            bvar.bvar_irf(f["B"], f["sigma_u"], p=2)
        with pytest.raises(ValueError, match="history"):
            bvar.bvar_forecast(f["B"], np.ones((1, 3)), p=2, n_steps=3)


class TestMsgarch:
    def _params(self):
        return (
            np.array([0.05, 0.5]),
            np.array([0.05, 0.2]),
            np.array([0.9, 0.6]),
            np.array([[0.98, 0.02], [0.05, 0.95]]),
        )

    def test_filter_shapes_and_probabilities(self):
        w, a, b, p = self._params()
        sim = msgarch.msgarch_simulate(w, a, b, p, 400, seed=3)
        out = msgarch.msgarch_filter(sim["y"], w, a, b, p)
        post = out["state_prob"]
        assert post.shape == (400, 2)
        assert np.allclose(post.sum(axis=1), 1.0)
        assert (post >= 0).all()
        assert np.isfinite(out["loglik"])
        assert out["h"].shape == (400, 2)
        assert (out["h"] > 0).all()

    def test_filter_separates_high_vol_regime(self):
        rng = np.random.default_rng(1)
        y = np.concatenate(
            [rng.normal(0, 0.1, 200), rng.normal(0, 1.5, 200), rng.normal(0, 0.1, 200)]
        )
        w, a, b, p = (
            np.array([0.01, 0.5]),
            np.array([0.05, 0.3]),
            np.array([0.9, 0.5]),
            np.array([[0.99, 0.01], [0.02, 0.98]]),
        )
        out = msgarch.msgarch_filter(y, w, a, b, p)
        hi_mid = out["state_prob"][250:350, 1].mean()
        lo_early = out["state_prob"][50:150, 0].mean()
        lo_late = out["state_prob"][450:550, 0].mean()
        assert hi_mid > 0.6
        assert lo_early > 0.9
        assert lo_late > 0.9

    def test_stationary_vol_known_answer(self):
        out = msgarch.msgarch_stationary_vol(np.array([0.1]), np.array([0.2]), np.array([0.7]))
        assert out[0] == pytest.approx(0.1 / 0.1)

    def test_simulate_deterministic_seed(self):
        w, a, b, p = self._params()
        s1 = msgarch.msgarch_simulate(w, a, b, p, 50, seed=7)
        s2 = msgarch.msgarch_simulate(w, a, b, p, 50, seed=7)
        assert np.array_equal(s1["y"], s2["y"])
        assert np.array_equal(s1["states"], s2["states"])

    def test_fit_runs_and_recovers_structure(self):
        w, a, b, p = self._params()
        sim = msgarch.msgarch_simulate(w, a, b, p, 600, seed=5)
        fit = msgarch.msgarch_fit(sim["y"], n_regimes=2, max_iter=50)
        assert fit["omega"].shape == (2,)
        assert np.isfinite(fit["nll"])
        assert np.allclose(fit["p"].sum(axis=1), 1.0)

    def test_validation(self):
        w, a, b, p = self._params()
        y = np.random.default_rng(0).normal(0, 1, 100)
        with pytest.raises(ValueError, match="30"):
            msgarch.msgarch_filter(np.ones(10), w, a, b, p)
        with pytest.raises(ValueError, match="alpha"):
            msgarch.msgarch_filter(y, w, np.array([0.8, 0.2]), b, p)
        with pytest.raises(ValueError, match="stochastic"):
            msgarch.msgarch_filter(y, w, a, b, np.array([[2.0, 0.0], [0.1, 0.9]]))
        with pytest.raises(ValueError, match="regimes"):
            msgarch.msgarch_filter(y, w[:1], a[:1], b[:1], np.array([[1.0]]))
        with pytest.raises(ValueError, match="pi"):
            msgarch.msgarch_filter(y, w, a, b, p, pi=np.array([0.5, 0.6]))


class TestQvar:
    def test_fit_shapes_and_median_close_to_ols(self):
        y = _var_panel(n=2, t=200, seed=2)
        fit = qvar.qvar_fit(y, p=1, tau=0.5, smooth=1e-3)
        assert fit["B"].shape == (1 + 2, 2)
        # median regression slope close to OLS on symmetric innovations
        x = np.column_stack([np.ones(199), y[:-1]])
        ols = np.linalg.lstsq(x, y[1:], rcond=None)[0]
        assert np.abs(fit["B"][1:, 0] - ols[1:, 0]).max() < 0.15

    def test_lower_quantile_has_negative_intercept(self):
        rng = np.random.default_rng(1)
        y = rng.normal(0, 1, (400, 1))
        fit_lo = qvar.qvar_fit(y, p=1, tau=0.1)
        fit_med = qvar.qvar_fit(y, p=1, tau=0.5)
        assert fit_lo["B"][0, 0] < fit_med["B"][0, 0]

    def test_forecast_and_irf(self):
        y = _var_panel(n=2, t=200, seed=4)
        fit = qvar.qvar_fit(y, p=1, tau=0.2)
        fc = qvar.qvar_forecast(fit["B"], y[-1:], p=1, n_steps=5)
        assert fc.shape == (5, 2)
        irf = qvar.qvar_irf(y, p=1, tau=0.5, shock=0, size=1.0, n_steps=4)
        assert irf.shape == (4, 2)
        # own-shock response at h=0 equals the fitted AR coefficient ~ 0.4
        assert 0.2 < irf[0, 0] < 0.8
        # response decays toward zero along the horizon
        assert abs(irf[0, 0]) > abs(irf[-1, 0])

    def test_validation(self):
        with pytest.raises(ValueError, match="tau"):
            qvar.qvar_fit(_var_panel(t=100), p=1, tau=1.2)
        with pytest.raises(ValueError, match="observations"):
            qvar.qvar_fit(np.ones((8, 2)), p=3)
        with pytest.raises(ValueError, match="history"):
            qvar.qvar_forecast(np.ones((3, 2)), np.ones((1, 2)), p=2, n_steps=2)
        with pytest.raises(ValueError, match="shock"):
            qvar.qvar_irf(_var_panel(t=100), tau=0.5, shock=9)


class TestPredictiveRegression:
    def _pair(self, rho: float = 0.97, b: float = 0.0, t: int = 500, seed: int = 0):
        rng = np.random.default_rng(seed)
        v = rng.normal(0, 1, t)
        x = np.zeros(t)
        for i in range(1, t):
            x[i] = rho * x[i - 1] + v[i]
        u = -0.8 * v + np.sqrt(1 - 0.64) * rng.normal(0, 1, t)
        y = b * np.roll(x, 1) + u
        y[0] = u[0]
        return x, y

    def test_stambaugh_correct_shrinks_toward_zero(self):
        x, y = self._pair(b=0.0, rho=0.98, t=600, seed=1)
        out = predictive_regression.stambaugh_correct(x, y)
        assert np.isfinite(out["b_bc"])
        # endogeneity makes OLS slope biased; corrected should be closer to 0
        assert abs(out["b_bc"]) <= abs(out["b_ols"]) + 0.02

    def test_bonferroni_interval_wide_on_persistent_x(self):
        x, y = self._pair(rho=0.95, t=300, seed=2)
        out = predictive_regression.bonferroni_test(x, y)
        assert out["ci_hi"] > out["ci_lo"]
        assert out["rho_ci"][1] <= 1.0
        assert 0 < out["rho"] < 1.1

    def test_long_horizon_detects_signal(self):
        rng = np.random.default_rng(3)
        x = rng.normal(0, 1, 400)
        noise = rng.normal(0, 1, 400)
        # y_{t+k} = 0.4*x_t + e => sum over h=3 has slope 1.2 on x_t
        y = np.zeros(400)
        for i in range(397):
            y[i + 1] += 0.4 * x[i] + noise[i + 1]
            y[i + 2] += 0.4 * x[i] + noise[i + 2]
            y[i + 3] += 0.4 * x[i] + noise[i + 3]
        out = predictive_regression.long_horizon_predict(x, y, h=3)
        assert out["b"] == pytest.approx(1.2, abs=0.4)
        assert abs(out["t"]) > 1.5
        assert out["n_used"] == 397

    def test_validation(self):
        with pytest.raises(ValueError, match="equal-length"):
            predictive_regression.stambaugh_correct(np.ones(10), np.ones(5))
        with pytest.raises(ValueError, match="degenerate"):
            predictive_regression.stambaugh_correct(np.ones(30), np.ones(30))
        x = np.random.default_rng(0).normal(0, 1, 100)
        y = np.random.default_rng(1).normal(0, 1, 100)
        with pytest.raises(ValueError, match="1 <= h"):
            predictive_regression.long_horizon_predict(x, y, h=200)
        with pytest.raises(ValueError, match="nw_lags"):
            predictive_regression.long_horizon_predict(x, y, h=2, nw_lags=200)


class TestPortSorts:
    def _panel(self, t: int = 30, n: int = 40, seed: int = 0):
        rng = np.random.default_rng(seed)
        char = rng.normal(0, 1, (t, n))
        fwd = char * 0.05 + rng.normal(0, 0.2, (t, n))  # predictive char
        return char, fwd

    def test_breakpoints_nyse_subset(self):
        x = np.linspace(0, 99, 40)
        mask = np.zeros(40, bool)
        mask[:10] = True
        bp = port_sorts.breakpoints_nyse(x, mask, 4)
        assert bp.size == 3
        assert np.all(bp < 30)  # cuts from the NYSE subset only

    def test_sorts_monotone_spread_positive(self):
        char, fwd = self._panel()
        out = port_sorts.sort_portfolios(char, fwd, n_bins=5)
        assert out["bucket_returns"].shape == (30, 5)
        assert out["spread_stats"]["mean"] > 0
        assert out["counts"].sum(axis=1).min() == 40

    def test_spread_invariant_under_shuffle(self):
        char, fwd = self._panel(seed=5)
        rng = np.random.default_rng(9)
        perm = rng.permutation(char.shape[1])
        out1 = port_sorts.sort_portfolios(char, fwd, n_bins=5)
        out2 = port_sorts.sort_portfolios(char[:, perm], fwd[:, perm], n_bins=5)
        assert np.allclose(out1["spread"], out2["spread"])

    def test_bivariate_shapes(self):
        char, fwd = self._panel()
        c2 = np.random.default_rng(2).normal(0, 1, char.shape)
        out = port_sorts.bivariate_sort(char, c2, fwd, n1=4, n2=4)
        assert out["means"].shape == (4, 4)
        assert out["counts"].sum() == 30 * 40

    def test_sort_tstat(self):
        out = port_sorts.sort_tstat(np.array([0.1, -0.05, 0.2, 0.15, 0.1]))
        assert out["n"] == 5
        assert np.isfinite(out["t"])
        assert port_sorts.sort_tstat(np.array([np.nan, 1.0]))["n"] == 1.0

    def test_validation(self):
        char, fwd = self._panel(t=10, n=10)
        with pytest.raises(ValueError, match="equal shape"):
            port_sorts.sort_portfolios(char, fwd[:, :5])
        with pytest.raises(ValueError, match="n_bins"):
            port_sorts.breakpoints_nyse(char[0], np.ones(10, bool), 1)
        with pytest.raises(ValueError, match="NYSE"):
            port_sorts.breakpoints_nyse(char[0], np.ones(10, bool), 20)
        with pytest.raises(ValueError, match="positive"):
            port_sorts.sort_portfolios(char, fwd, n_bins=3, value_weight=-np.ones_like(char))


class TestSmoothTransition:
    def _lstar(self, t: int = 400, seed: int = 0):
        rng = np.random.default_rng(seed)
        y = np.zeros(t)
        for i in range(2, t):
            g = 1.0 / (1.0 + np.exp(-6.0 * y[i - 1]))
            y[i] = 0.7 * y[i - 1] * (1 - g) - 0.6 * y[i - 1] * g + rng.normal(0, 0.15)
        return y

    def test_fit_recovers_switch_point(self):
        y = self._lstar()
        fit = smooth_transition.star_fit(y, p=2, d=1, kind="lstar")
        assert fit["gamma"] > 0
        assert fit["sse"] < np.var(y) * y.size
        assert np.isfinite(fit["resid"]).all()

    def test_forecast_shape(self):
        y = self._lstar(t=250)
        fit = smooth_transition.star_fit(y, p=2, d=1)
        fc = smooth_transition.star_forecast(fit, y[-2:], 5)
        assert fc.shape == (5,)
        assert np.isfinite(fc).all()

    def test_linearity_rejects_on_strong_nonlinearity(self):
        y = self._lstar(t=600)
        out = smooth_transition.star_linearity(y, p=2, d=1)
        assert out["p"] < 0.1

    def test_estar_runs(self):
        rng = np.random.default_rng(4)
        y = np.zeros(300)
        for i in range(2, 300):
            g = 1 - np.exp(-2.0 * y[i - 1] ** 2)
            y[i] = (0.3 * y[i - 1]) * (1 - g) + (-0.5 * y[i - 1]) * g + rng.normal(0, 0.2)
        fit = smooth_transition.star_fit(y, p=1, d=1, kind="estar")
        assert np.isfinite(fit["sse"])

    def test_validation(self):
        with pytest.raises(ValueError, match="d <= p"):
            smooth_transition.star_fit(np.ones(100), p=1, d=2)
        with pytest.raises(ValueError, match="kind"):
            smooth_transition.star_fit(np.ones(100), p=1, d=1, kind="bad")
        fit = smooth_transition.star_fit(self._lstar(t=120), p=2, d=1)
        with pytest.raises(ValueError, match="history"):
            smooth_transition.star_forecast(fit, np.ones(1), 3)
        with pytest.raises(ValueError, match="n_steps"):
            smooth_transition.star_forecast(fit, np.ones(3), 0)
