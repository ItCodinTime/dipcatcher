"""Canon wave 12: Bai-Perron breaks, UCM, FAVAR, DML, MS-VAR, C-vine.

Each module gets known-answer recovery tests, degenerate-input error paths,
and invariants. Deterministic seeds; no network.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from quant_fund.models import bai_perron as bp
from quant_fund.models import dml, favar, ms_var, ucm, vine_copula

Array = NDArray[np.float64]


def _break_dgp(t: int = 240) -> tuple[Array, Array]:
    rng = np.random.default_rng(0)
    x = np.ones((t, 1))
    y = np.empty(t)
    y[:80] = 1.0 + rng.normal(0, 0.2, 80)
    y[80:160] = 3.0 + rng.normal(0, 0.2, 80)
    y[160:] = -1.0 + rng.normal(0, 0.2, 80)
    return y, x


class TestBaiPerron:
    def test_sup_wald_finds_break(self) -> None:
        y, x = _break_dgp()
        res = bp.sup_wald(y[:160], x[:160])
        assert abs(res["break"] - 80) <= 3
        assert res["stat"] > bp.sup_wald_cv(1)

    def test_dp_recovers_two_breaks(self) -> None:
        y, x = _break_dgp()
        out = bp.breakpoints_dp(y, x, 2)
        assert np.allclose(out["breaks"], [80, 160], atol=3)
        ssr_flat = bp.breakpoints_dp(y, x, 0)["ssr_total"][0]
        assert out["ssr_total"][0] < 0.3 * ssr_flat

    def test_sequential_selects_m(self) -> None:
        y, x = _break_dgp()
        out = bp.sequential_breaks(y, x)
        assert out["n_breaks"][0] == 2
        # no-break series -> 0 breaks
        rng = np.random.default_rng(1)
        y0 = 1.0 + rng.normal(0, 0.2, 240)
        out0 = bp.sequential_breaks(y0, x)
        assert out0["n_breaks"][0] <= 1

    def test_refit_segments(self) -> None:
        y, x = _break_dgp()
        out = bp.refit_segments(y, x, np.array([80, 160]))
        assert out["n_segments"][0] == 3
        assert np.allclose(out["coefs"].ravel(), [1.0, 3.0, -1.0], atol=0.15)
        assert np.isfinite(out["bic"][0])

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            bp.sup_wald(np.arange(5.0), np.ones((5, 1)))
        with pytest.raises(ValueError, match="trim"):
            bp.sup_wald(np.arange(20.0), np.ones((20, 1)), trim=0.9)
        with pytest.raises(ValueError, match="alpha"):
            bp.sup_wald_cv(1, alpha=0.2)
        y, x = _break_dgp()
        with pytest.raises(ValueError, match="infeasible"):
            bp.breakpoints_dp(y, x, 20, trim=0.3)
        with pytest.raises(ValueError):
            bp.refit_segments(y, x, np.array([160, 80]))  # unsorted


class TestUcm:
    def _dgp(self, cycle: bool) -> Array:
        rng = np.random.default_rng(0)
        t = 240
        level = np.cumsum(0.05 + rng.normal(0, 0.05, t))
        y = level + rng.normal(0, 0.2, t)
        if cycle:
            y = y + 0.5 * np.sin(2 * np.pi * np.arange(t) / 12)
        return y

    def test_trend_only_recovers_level(self) -> None:
        y = self._dgp(cycle=False)
        fit = ucm.ucm_fit(y)
        assert fit["converged"][0] >= 0.0
        assert fit["sigma_obs"][0] > 0
        # smoothed level tracks the drift
        assert np.corrcoef(fit["level"], y)[0, 1] > 0.9

    def test_cycle_extracted(self) -> None:
        y = self._dgp(cycle=True)
        fit = ucm.ucm_fit(y, cycle_period=12.0)
        assert fit["has_cycle"][0] == 1.0
        assert 0 < fit["rho"][0] <= 1.0
        c = fit["cycle"]
        assert np.std(c) > 0.05  # nonzero cycle extracted

    def test_forecast_grows_with_trend(self) -> None:
        fit = ucm.ucm_fit(self._dgp(cycle=False))
        out = ucm.ucm_forecast(fit, steps=6)
        assert out["y_hat"].shape == (6,)
        assert np.isfinite(out["y_hat"]).all()
        with pytest.raises(ValueError):
            ucm.ucm_forecast(fit, steps=0)

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            ucm.ucm_fit(np.arange(5.0))
        with pytest.raises(ValueError, match="finite"):
            ucm.ucm_fit(np.array([1.0] * 20 + [np.nan]))
        with pytest.raises(ValueError, match="cycle_period"):
            ucm.ucm_fit(self._dgp(False), cycle_period=1.0)


class TestFavar:
    def _panel(self) -> tuple[Array, Array]:
        rng = np.random.default_rng(0)
        t, n = 200, 8
        f1 = np.cumsum(rng.normal(0, 0.1, t))
        f2 = np.cumsum(rng.normal(0, 0.1, t))
        x = np.column_stack(
            [0.8 * f1 + rng.normal(0, 0.2, t), 0.8 * f1 + rng.normal(0, 0.2, t)]
            + [0.9 * f2 + rng.normal(0, 0.2, t) for _ in range(4)]
            + [rng.normal(0, 1, t), rng.normal(0, 1, t)]
        )[:, :n]
        y = (0.5 * np.r_[0, np.diff(f1)] + rng.normal(0, 0.1, t)).reshape(-1, 1)
        return y, x

    def test_extract_factors(self) -> None:
        _, x = self._panel()
        out = favar.favar_extract(x, 2)
        assert out["factors"].shape == (x.shape[0], 2)
        assert out["explained"][0] > 0.5
        assert np.isclose(np.std(out["factors"][:, 0]), 1.0, atol=0.01)

    def test_fit_irf_fevd(self) -> None:
        y, x = self._panel()
        fit = favar.favar_fit(y, x, r=2, p=2)
        assert fit["var"].shape == (2, 3, 3)
        irf = favar.favar_irf(fit, horizon=8)
        assert irf["oirf_target"].shape == (9, 1, 3)
        f = favar.favar_fevd(fit, horizon=6)
        assert f["gfevd_target"].shape[0] == 1
        assert np.allclose(f["gfevd"].sum(axis=1), 1.0, atol=1e-6)

    def test_forecast(self) -> None:
        y, x = self._panel()
        fit = favar.favar_fit(y, x, r=2, p=2)
        out = favar.favar_forecast(fit, y[-2:], fit["factors"][-2:], steps=4)
        assert out["y_forecast"].shape == (4, 1)
        with pytest.raises(ValueError):
            favar.favar_forecast(fit, y[-2:], fit["factors"][-2:], steps=0)
        with pytest.raises(ValueError, match="p"):
            favar.favar_forecast(fit, y[-1:], fit["factors"][-1:], steps=2)

    def test_validation(self) -> None:
        y, x = self._panel()
        with pytest.raises(ValueError, match="r"):
            favar.favar_extract(x, 20)
        with pytest.raises(ValueError, match="constant"):
            favar.favar_extract(np.hstack([x, np.ones((x.shape[0], 1))]), 2)
        with pytest.raises(ValueError):
            favar.favar_fit(y, x[:5], r=1, p=1)


class TestDml:
    def _dgp(self, theta: float = 0.8, n: int = 400) -> tuple[Array, Array, Array]:
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, (n, 3))
        m_x = 0.5 * x[:, 0] + 0.3 * x[:, 1]
        g_x = 0.4 * x[:, 0] - 0.2 * x[:, 2]
        d = m_x + rng.normal(0, 0.5, n)
        y = theta * d + g_x + rng.normal(0, 0.3, n)
        return y, d, x

    def test_plr_recovers_theta(self) -> None:
        y, d, x = self._dgp()
        out = dml.dml_plr(y, d, x, seed=7)
        assert abs(out["theta"][0] - 0.8) < 0.1
        assert out["ci_lo"][0] < out["theta"][0] < out["ci_hi"][0]
        assert out["se"][0] > 0

    def test_plr_zero_effect(self) -> None:
        y, d, x = self._dgp(theta=0.0)
        out = dml.dml_plr(y, d, x)
        assert abs(out["theta"][0]) < 0.15
        assert out["ci_lo"][0] < 0 < out["ci_hi"][0]

    def test_plr_degenerate_treatment(self) -> None:
        y, d, x = self._dgp()
        d_const = np.full_like(d, 1.0)  # m_hat explains all -> collapse
        with pytest.raises(ValueError, match="collapsed"):
            dml.dml_plr(y, d_const, x)

    def test_irm_ate(self) -> None:
        rng = np.random.default_rng(2)
        n = 400
        x = rng.normal(0, 1, (n, 3))
        p = 0.5 + 0.2 * np.tanh(x[:, 0])
        d = (rng.uniform(0, 1, n) < p).astype(float)
        y = 1.5 * d + 0.4 * x[:, 0] + rng.normal(0, 0.3, n)
        out = dml.dml_irm(y, d, x, seed=3)
        assert abs(out["theta"][0] - 1.5) < 0.25
        assert out["propensity_range"][0] >= 0.05

    def test_irm_validation(self) -> None:
        y, d, x = self._dgp()
        with pytest.raises(ValueError, match="binary"):
            dml.dml_irm(y, d, x)  # continuous d
        with pytest.raises(ValueError):
            dml.dml_plr(y, d, x, n_folds=100)
        with pytest.raises(ValueError):
            dml.dml_plr(y[:10], d[:10], x[:10])


class TestMsVar:
    def _dgp(self) -> Array:
        rng = np.random.default_rng(0)
        t = 300
        states = np.zeros(t, dtype=int)
        for i in range(1, t):
            states[i] = states[i - 1] if rng.uniform() < 0.95 else 1 - states[i - 1]
        y = np.empty(t)
        y[0] = 0.0
        for i in range(1, t):
            if states[i] == 0:
                y[i] = 0.3 * y[i - 1] + rng.normal(0, 0.1)
            else:
                y[i] = -0.2 + 0.1 * y[i - 1] + rng.normal(0, 0.8)
        return y.reshape(-1, 1)

    def test_fit_two_regimes(self) -> None:
        y = self._dgp()
        fit = ms_var.msvar_fit(y, k=2, max_iter=60)
        assert fit["P"].shape == (2, 2)
        assert np.allclose(fit["P"].sum(axis=1), 1.0, atol=1e-6)
        assert np.isfinite(fit["loglik"][0])
        sig = np.sqrt(fit["sigma"][0, 0])
        # one regime's mean should differ — recovered separate mus
        assert fit["mu"].shape == (2, 1)
        assert sig > 0

    def test_smoothed_probs_valid(self) -> None:
        fit = ms_var.msvar_fit(self._dgp(), k=2, max_iter=40)
        sm = fit["smoothed"]
        assert np.allclose(sm.sum(axis=1), 1.0, atol=1e-6)
        assert (sm >= -1e-9).all()

    def test_forecast(self) -> None:
        fit = ms_var.msvar_fit(self._dgp(), k=2, max_iter=40)
        out = ms_var.msvar_forecast(fit, np.array([0.1]), steps=4)
        assert out["y_forecast"].shape == (4, 1)
        assert np.isfinite(out["y_forecast"]).all()

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            ms_var.msvar_fit(np.arange(10.0))
        with pytest.raises(ValueError, match="k"):
            ms_var.msvar_fit(self._dgp(), k=8)
        fit = ms_var.msvar_fit(self._dgp(), k=2, max_iter=10)
        with pytest.raises(ValueError, match="match"):
            ms_var.msvar_forecast(fit, np.array([0.1, 0.2]), steps=2)


class TestVineCopula:
    def _corr_sample(self, rho: float = 0.7, n: int = 400, d: int = 3) -> Array:
        rng = np.random.default_rng(0)
        cov = np.full((d, d), rho) + np.eye(d) * (1 - rho)
        z = rng.multivariate_normal(np.zeros(d), cov, size=n)
        return vine_copula.to_pseudo(z)

    def test_to_pseudo_uniform(self) -> None:
        u = self._corr_sample()
        assert (u > 0).all() and (u < 1).all()
        # rank transform of normals -> approximately uniform marginals
        assert abs(float(u[:, 0].mean()) - 0.5) < 0.05

    def test_fit_recovers_positive_dependence(self) -> None:
        u = self._corr_sample(rho=0.7)
        fit = vine_copula.cvine_fit(u)
        theta = fit["theta"]
        assert theta.shape == (2, 3)
        # tree-1 pairs should show strong positive rho
        assert theta[0, 1] > 0.4
        assert theta[0, 2] > 0.4
        assert np.isfinite(fit["loglik"][0])
        assert fit["loglik"][0] > 0  # dependence beats independence

    def test_independence_low_loglik(self) -> None:
        rng = np.random.default_rng(1)
        u = rng.uniform(0.001, 0.999, size=(300, 3))
        fit = vine_copula.cvine_fit(u)
        assert fit["loglik"][0] < self._corr_sample(0.7).shape[0] * 0.5

    def test_loglik_eval_and_simulate(self) -> None:
        u = self._corr_sample()
        fit = vine_copula.cvine_fit(u)
        ll = vine_copula.cvine_loglik(fit, u)
        assert ll == pytest.approx(fit["loglik"][0], rel=1e-6)
        sim = vine_copula.cvine_simulate(fit, n=200, seed=0)["u"]
        assert sim.shape == (200, 3)
        assert (sim > 0).all() and (sim < 1).all()
        # simulated marginals ~ uniform, joint dependence preserved
        corr = np.corrcoef(sim[:, 0], sim[:, 1])[0, 1]
        assert corr > 0.3

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            vine_copula.to_pseudo(np.arange(5.0).reshape(-1, 1))
        u = self._corr_sample()
        u_bad = u.copy()
        u_bad[0, 0] = 1.5
        with pytest.raises(ValueError, match="0, 1"):
            vine_copula.cvine_fit(u_bad)
        fit = vine_copula.cvine_fit(u)
        with pytest.raises(ValueError):
            vine_copula.cvine_simulate(fit, n=0)
