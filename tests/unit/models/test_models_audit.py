"""Regression tests for defects found in the models audit (docs/AUDIT_MODELS.md)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.ar_estimation import levinson_durbin
from quant_fund.models.diffusion_index import diffusion_index_forecast, sw_factors
from quant_fund.models.duration import _nll_weibull
from quant_fund.models.factor_models import bai_ng_factors, double_sorted_factors
from quant_fund.models.fgls_ar1 import prais_winsten
from quant_fund.models.gas import gas_vol_fit, gas_vol_forecast
from quant_fund.models.nonlinear_filters import particle_filter
from quant_fund.models.nowcasting import beta_weights, fit_midas
from quant_fund.models.state_space import ou_mle
from quant_fund.models.tail import DrawdownClassifier, ScaledHistoricalTail
from quant_fund.models.var_coint import engle_granger, johansen_test, vecm_fit


def test_engle_granger_rejects_nonfinite_x():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=200))
    x[50] = np.nan
    y = x + rng.normal(size=200)
    with pytest.raises(ValueError):
        engle_granger(y, x)


def test_engle_granger_still_rejects_nonfinite_y():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=200))
    y = x + rng.normal(size=200)
    y[50] = np.inf
    with pytest.raises(ValueError):
        engle_granger(y, x)


def test_johansen_cv_indexed_by_n_minus_r():
    # n=2: under H0 rank<=0 there are n-r=2 independent combinations ->
    # the 5% trace CV must be 9.24, not the n-r=1 entry 3.76.
    rng = np.random.default_rng(1)
    eps = rng.normal(size=(300, 2))
    beta = np.array([1.0, -1.0])
    alpha = np.array([[-0.1], [0.0]])
    y = np.zeros((300, 2))
    for t in range(1, 300):
        y[t] = y[t - 1] + (alpha @ np.array([[beta @ y[t - 1] - 0.0]])).ravel() + eps[t]
    out = johansen_test(y, p=2)
    assert out["cv_trace5"][0] == pytest.approx(9.24)
    assert out["cv_max5"][0] == pytest.approx(11.22)
    # rank = first r whose statistic does not reject; all-rejected -> n.
    assert 0 <= out["rank_trace"][0] <= 2
    # A genuinely cointegrated pair should detect rank 1.
    assert out["rank_trace"][0] == 1.0 or out["rank_max"][0] == 1.0


def test_vecm_gamma_lag_ordering():
    # dy_t = Gamma1 dy_{t-1} + 0 * dy_{t-2} + small EC pull; lag-1 block must
    # land in gamma[0], not be scrambled across lags.
    rng = np.random.default_rng(7)
    n, tt = 2, 600
    g1 = np.diag([0.5, 0.3])
    alpha = np.array([[-0.05], [0.0]])
    beta = np.array([[1.0], [-1.0]])
    dy = np.zeros((tt, n))
    y = np.zeros((tt, n))
    for t in range(2, tt):
        ect = float((beta.T @ y[t - 1]).item())
        dy[t] = g1 @ dy[t - 1] + (alpha * ect).ravel() + 0.1 * rng.normal(size=n)
        y[t] = y[t - 1] + dy[t]
    out = vecm_fit(y, p=3, rank=1)
    gamma = np.asarray(out["gamma"])
    assert gamma.shape == (2, n, n)
    # True Gamma1[1,1] ~ 0.3; the buggy reshape scrambled lag-1/lag-2 rows so
    # gamma[0][1,:] picked up the ~0 lag-2 coefficients instead.
    assert gamma[0][1, 1] == pytest.approx(0.3, abs=0.15)
    assert np.abs(gamma[1]).max() < 0.2


def test_diffusion_index_forecast_origin():
    # A one-step forecast must use the true origin row t = T-1 ([1, F_{T-1},
    # y_T-1...]), not the last fitted row t = T-2.
    rng = np.random.default_rng(3)
    panel = rng.normal(size=(60, 8))
    f = np.asarray(sw_factors(panel, 1)["factors"])[:, 0]
    y = 1.5 + 2.0 * f
    out = diffusion_index_forecast(panel, y, k=1, ar_lags=1, horizon=1)
    fc = float(out["forecast"])
    beta = np.asarray(out["coefs"])
    correct_row = np.array([1.0, f[-1], y[-1]])
    buggy_row = np.array([1.0, f[-2], y[-2]])
    assert fc == pytest.approx(float(correct_row @ beta), abs=1e-12)
    assert abs(fc - float(buggy_row @ beta)) > 1e-8


def test_bai_ng_zero_variance_panel_raises():
    panel = np.full((50, 4), 3.0)
    with pytest.raises(ValueError, match="zero total variance"):
        bai_ng_factors(panel)


def test_levinson_durbin_non_pd_raises():
    # r1 > r0 is impossible for a valid autocovariance; must raise, not
    # silently floor the innovation variance at 1e-12.
    with pytest.raises(ValueError, match="positive-definite"):
        levinson_durbin(np.array([1.0, 2.0, 0.5]), 2)


def test_prais_winsten_rho_outside_unit_interval_raises():
    # Exponentially growing residuals give num/den > 1; the PW transform is
    # invalid there and must fail closed.
    t = np.linspace(0.0, 3.0, 50)
    y = np.exp(t)
    x = np.column_stack([np.random.default_rng(0).normal(size=50)])
    with pytest.raises(ValueError, match="outside"):
        prais_winsten(y, x)


def test_ou_mle_non_stationary_raises():
    # Explosive transition (phi > 1) must raise rather than clamp to 0.999999.
    x = np.exp(np.linspace(0.0, 2.0, 100))
    with pytest.raises(ValueError, match="mean-reverting"):
        ou_mle(x)


def test_gas_fit_exposes_scaling_and_forecast_uses_it():
    rng = np.random.default_rng(11)
    eps = rng.normal(size=400)
    h = np.empty(400)
    h[0] = 1.0
    for t in range(1, 400):
        h[t] = 0.05 + 0.1 * eps[t - 1] ** 2 + 0.85 * h[t - 1]
    y = np.sqrt(h) * eps
    fit = gas_vol_fit(y, dist="gauss", scaling="unit")
    assert fit["unit_scaling"] == 1.0
    f_last = float(np.asarray(fit["f"])[-1])
    y_last = float(y[-1])
    s = (y_last * y_last - f_last) / (2.0 * f_last * f_last)
    u_unit = s * 2.0 * f_last
    expected = float(fit["omega"]) + float(fit["A"]) * u_unit + float(fit["B"]) * f_last
    assert gas_vol_forecast(fit, y_last, dist="gauss") == pytest.approx(expected)


def test_acd_weibull_nll_includes_psi_jacobian():
    # Correct WACD loglik: ln g - g ln(psi*lam) + (g-1) ln(x/(psi*lam))... i.e.
    # -psi coefficient must be -gamma, not -1. For gamma=1 both agree; for
    # gamma=2 the missing (g-1)*ln psi term shifts the value.
    x = np.array([1.0, 1.2, 0.8, 1.1, 0.9, 1.3, 1.0, 0.95, 1.05, 1.1])
    theta = np.array([0.3, 0.2, 0.5, 2.0])  # omega, alpha, beta, gamma=2
    omega, alpha, beta, gamma = theta
    psi = np.empty(x.size)
    psi[0] = x.mean()  # matches _psi_path's initialization
    for t in range(1, x.size):
        psi[t] = omega + alpha * x[t - 1] + beta * psi[t - 1]
    lam = math.exp(math.lgamma(1.0 + 1.0 / gamma))
    z = x / psi / lam
    # Derived: eps = x/psi, f_x = f_eps(x/psi)/psi =>
    # ln f = ln g - g ln lam - ln psi + (g-1)(ln x - ln psi) - z^g
    #      = ln g - g ln(psi*lam) + (g-1) ln x - z^g.
    expected = np.sum(
        np.log(gamma) - gamma * np.log(psi * lam) + (gamma - 1.0) * np.log(x) - z**gamma
    )
    # The pre-fix code had (g-1) ln x and only -ln psi, i.e. psi weight -1
    # instead of -gamma.
    assert _nll_weibull(theta, x) == pytest.approx(float(-expected))


def test_particle_filter_carries_weights_without_resample():
    # With resample_frac=0 the filter never resamples; weights must be the
    # running product w_t ~ w_{t-1} * p(y_t|x_t), not reset each step.
    n_p = 50
    seed = 4
    q_std = 0.3
    y = np.array([0.8, -0.6, 0.2, 0.5, -0.1])

    out = particle_filter(
        y,
        f=lambda x, rng: x,
        obs_loglik=lambda x, yy: -0.5 * (x - yy) ** 2,
        q_std=q_std,
        n_particles=n_p,
        x0=0.0,
        p0_std=1.0,
        seed=seed,
        resample_frac=0.0,
    )
    rng = np.random.default_rng(seed)
    parts = rng.normal(0.0, 1.0, n_p)
    w = np.full(n_p, 1.0 / n_p)
    means, loglik = [], 0.0
    for t in range(y.size):
        parts = parts + rng.normal(0.0, q_std, n_p)
        ll = -0.5 * (parts - y[t]) ** 2
        mx = ll.max()
        w = w * np.exp(ll - mx)
        tot = w.sum()
        w /= tot
        loglik += mx + math.log(tot)
        means.append(float(w @ parts))
    np.testing.assert_allclose(out["mean"], means, rtol=1e-12, atol=1e-12)
    assert out["loglik"] == pytest.approx(loglik, rel=1e-12, abs=1e-12)


def test_midas_ar_lag_false_actually_optimizes():
    # With ar_lag=False the NaN in yl[0] used to poison every SSE evaluation
    # (0 * NaN), so all starts returned the 1e12 penalty and the optimizer
    # never moved off its seed point.
    rng = np.random.default_rng(0)
    n, k = 200, 6
    x = rng.normal(size=(n, k))
    w_true = beta_weights(k, 1.0, 8.0)
    y = 0.5 + 2.0 * (x @ w_true) + rng.normal(scale=0.1, size=n)
    fit = fit_midas(y, x, k_lags=k, ar_lag=False)
    assert fit["sse"] < 1e12
    assert abs(fit["slope"] - 2.0) < 0.4


def test_scaled_historical_tail_unfitted_raises():
    m = ScaledHistoricalTail()
    with pytest.raises(RuntimeError, match="not been fitted"):
        m.predict_var_es(np.ones(3))


def test_drawdown_classifier_unfitted_raises():
    m = DrawdownClassifier()
    with pytest.raises(RuntimeError, match="not been fitted"):
        m.predict_proba(np.zeros((3, 2)))


# --- Optimizer-penalty guards -------------------------------------------------
# Each objective returns a 1e12 penalty on invalid regions; the guard is that
# this penalty must never be accepted as a converged optimum.


class _PenaltyResult:
    def __init__(self, x):
        self.x = x
        self.fun = 1e12
        self.success = True


def test_skew_t_fit_rejects_penalty(monkeypatch):
    import quant_fund.models.skew_t as mod

    monkeypatch.setattr(
        mod, "minimize", lambda *a, **k: _PenaltyResult(np.array([8.0, 0.0, 0.0, 0.0]))
    )
    with pytest.raises(ValueError, match="skew-t MLE"):
        mod.skew_t_fit(np.random.default_rng(0).normal(size=100))


def test_joe_fit_rejects_penalty(monkeypatch):
    import quant_fund.models.archimedean_extra as mod

    monkeypatch.setattr(mod, "minimize_scalar", lambda *a, **k: _PenaltyResult(np.array(2.0)))
    u = np.random.default_rng(0).uniform(0.01, 0.99, size=(60, 2))
    with pytest.raises(ValueError, match="Joe copula"):
        mod.joe_fit(u)


def test_whittle_estimators_reject_penalty(monkeypatch):
    import quant_fund.models.long_memory as mod

    monkeypatch.setattr(mod.opt, "minimize_scalar", lambda *a, **k: _PenaltyResult(np.array(0.3)))
    rng = np.random.default_rng(0)
    series = np.cumsum(rng.normal(size=300))
    with pytest.raises(ValueError, match="Whittle"):
        mod.local_whittle(series)
    with pytest.raises(ValueError, match="Whittle"):
        mod.whittle_arfima(series)


def test_dcc_stage2_rejects_penalty(monkeypatch):
    import quant_fund.models.dcc as mod

    monkeypatch.setattr(
        mod.optimize,
        "minimize",
        lambda f, x0, **k: _PenaltyResult(np.asarray(x0, dtype=float)),
    )
    r = np.random.default_rng(0).normal(size=(150, 3))
    with pytest.raises(ValueError, match="DCC stage-2"):
        mod.dcc_fit(r)


def test_svensson_rejects_penalty(monkeypatch):
    import quant_fund.models.term_structure as mod

    monkeypatch.setattr(
        mod.optimize,
        "minimize",
        lambda f, x0, **k: _PenaltyResult(np.asarray(x0, dtype=float)),
    )
    m = np.array([1.0, 2.0, 5.0, 10.0, 30.0])
    y = np.array([2.0, 2.2, 2.5, 2.8, 3.0])
    with pytest.raises(ValueError, match="Svensson"):
        mod.svensson_fit(m, y)


def test_garch_midas_rejects_penalty(monkeypatch):
    import quant_fund.models.garch_midas as mod

    monkeypatch.setattr(
        mod.optimize,
        "minimize",
        lambda f, x0, **k: _PenaltyResult(np.asarray(x0, dtype=float)),
    )
    rng = np.random.default_rng(0)
    r = rng.normal(size=300)
    with pytest.raises(ValueError, match="GARCH-MIDAS"):
        mod.garch_midas_fit(r, block=21, k_lag=5)


def test_figarch_aparch_reject_penalty(monkeypatch):
    import quant_fund.models.garch_ext as mod

    monkeypatch.setattr(
        mod.opt, "minimize", lambda *a, **k: _PenaltyResult(np.asarray(a[1], dtype=float))
    )
    r = np.random.default_rng(0).normal(size=300)
    with pytest.raises(ValueError, match="FIGARCH"):
        mod.fit_figarch(r)
    with pytest.raises(ValueError, match="APARCH"):
        mod.fit_aparch(r)


def test_double_sorted_factors_permutation_invariant_under_ties():
    # tie group straddling a cell boundary must land in one cell regardless
    # of storage order: permuting asset order must not change the output.
    rng = np.random.default_rng(7)
    t, n = 6, 12
    a = rng.normal(size=(t, n))
    b = rng.normal(size=(t, n))
    # inject tie groups that straddle the cuts=3 bin boundaries
    a[:, 3:6] = 0.5
    b[:, 8:11] = -0.2
    r = rng.normal(size=(t, n))
    base = double_sorted_factors(r, a, b, cuts=3)
    for _ in range(20):
        perm = rng.permutation(n)
        out = double_sorted_factors(r[:, perm], a[:, perm], b[:, perm], cuts=3)
        assert np.allclose(base["factor_a"], out["factor_a"])
        assert np.allclose(base["factor_b"], out["factor_b"])
        assert np.allclose(
            np.nan_to_num(base["grid"], nan=-1.0),
            np.nan_to_num(out["grid"], nan=-1.0),
        )


def test_double_sorted_factors_no_asset_dropped():
    # every asset lands in exactly one grid cell — with 1-based midranks the
    # naive rank*cuts//n formula pushes rank n into cell `cuts` and drops it.
    # oracle: scipy rankdata('average') partition vs the returned grid means.
    from scipy.stats import rankdata

    rng = np.random.default_rng(3)
    t, n, cuts = 5, 12, 3
    r = rng.normal(size=(t, n))
    a = rng.normal(size=(t, n))
    b = rng.normal(size=(t, n))
    a[0, :3] = 0.9  # tie group across the top boundary
    out = double_sorted_factors(r, a, b, cuts=cuts)
    for s in range(t):
        qa = ((rankdata(a[s], method="average") - 1.0) * cuts // n).astype(int)
        qb = ((rankdata(b[s], method="average") - 1.0) * cuts // n).astype(int)
        for i in range(cuts):
            for j in range(cuts):
                sel = (qa == i) & (qb == j)
                cell = out["grid"][s, i, j]
                if np.any(sel):
                    assert np.isfinite(cell)
                    assert cell == pytest.approx(float(r[s, sel].mean()))
                else:
                    assert not np.isfinite(cell)


def test_gp_regression_penalty_falls_back_to_defaults(monkeypatch):
    import scipy.optimize as so

    import quant_fund.models.kernel as mod

    monkeypatch.setattr(so, "minimize", lambda *a, **k: _PenaltyResult(np.zeros(3)))
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 2))
    y = x[:, 0] * 0.5 + rng.normal(scale=0.1, size=30)
    fit = mod.fit_gp_regression(x, y)
    # fallback to the median-length heuristic, not the penalized point
    assert fit["length"] == pytest.approx(mod._median_length(x))
