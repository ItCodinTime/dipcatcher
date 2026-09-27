"""Tests for regression estimation and diagnostics."""

from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm

from quant_fund.metrics.regression import (
    breusch_godfrey,
    breusch_pagan,
    cooks_distance,
    cusum_recursive,
    cusum_sq,
    durbin_watson,
    goldfeld_quandt,
    hac_covariance,
    hc_covariance,
    huber_regression,
    ols,
    ols_summary,
    quantile_regression,
    ramsey_reset,
    sargan_hansen_j,
    theil_sen,
    two_sls,
    vif,
    white_test,
)


def _gen(seed: int = 0, n: int = 300, k: int = 3, beta=None):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, k))
    b = np.arange(1.0, k + 1.0) if beta is None else beta
    y = 0.5 + x @ b + rng.normal(scale=0.5, size=n)
    return x, y, b


class TestOLS:
    def test_betas_recover(self):
        x, y, b = _gen()
        fit = ols(y, x)
        assert np.allclose(fit["beta"][1:], b, atol=0.15)
        assert abs(fit["beta"][0] - 0.5) < 0.15
        assert fit["r2"][0] > 0.95

    def test_matches_statsmodels(self):
        x, y, _ = _gen(seed=1)
        fit = ols_summary(y, x, se="OLS")
        sm_fit = sm.OLS(y, sm.add_constant(x)).fit()
        assert np.allclose(fit["beta"], sm_fit.params, atol=1e-8)
        assert np.allclose(fit["se"], sm_fit.bse, atol=1e-8)

    def test_hc_covariance_vs_statsmodels(self):
        x, y, _ = _gen(seed=2)
        e = ols(y, x)["resid"]
        m = sm.add_constant(x)
        for kind in ("HC0", "HC1", "HC2", "HC3"):
            cov = hc_covariance(e, m, kind)
            sm_cov = sm.OLS(y, m).fit(cov_type=kind).cov_params()
            assert np.allclose(np.diag(cov), np.diag(sm_cov), rtol=1e-6)

    def test_hac_covariance_positive(self):
        x, y, _ = _gen(seed=3)
        e = ols(y, x)["resid"]
        cov = hac_covariance(e, sm.add_constant(x), lag=4)
        assert np.all(np.diag(cov) > 0)
        sm_cov = (
            sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": 4}).cov_params()
        )
        assert np.allclose(np.diag(cov), np.diag(sm_cov), rtol=1e-6)

    def test_failclosed(self):
        with pytest.raises(ValueError):
            ols(np.ones(3), np.ones((3, 2)))
        with pytest.raises(ValueError):
            hc_covariance(np.ones(4), np.ones((4, 6)), "HC3")
        with pytest.raises(ValueError):
            hac_covariance(np.ones(20), np.ones((20, 2)), lag=25)


class TestDiagnostics:
    def test_dw_detects_ar1(self):
        rng = np.random.default_rng(4)
        e = np.zeros(300)
        for i in range(1, 300):
            e[i] = 0.9 * e[i - 1] + rng.normal()
        dw = durbin_watson(e)
        assert dw < 0.5
        dw_iid = durbin_watson(rng.normal(size=300))
        assert 1.5 < dw_iid < 2.5

    def test_bg_detects_serial_corr(self):
        rng = np.random.default_rng(5)
        n = 300
        x = rng.normal(size=(n, 2))
        e = np.zeros(n)
        for i in range(1, n):
            e[i] = 0.8 * e[i - 1] + rng.normal()
        y = x @ np.array([1.0, 2.0]) + e
        out = breusch_godfrey(y, x, lags=1)
        assert out["pvalue"] < 0.001
        y2 = x @ np.array([1.0, 2.0]) + rng.normal(size=n)
        assert breusch_godfrey(y2, x, lags=1)["pvalue"] > 0.01

    def test_het_tests(self):
        rng = np.random.default_rng(6)
        n = 300
        x = rng.normal(size=(n, 1))
        y_het = x[:, 0] + x[:, 0] * rng.normal(size=n)  # variance ~ x^2
        assert breusch_pagan(y_het, x)["pvalue"] < 0.01
        y_hom = x[:, 0] + rng.normal(size=n)
        assert breusch_pagan(y_hom, x)["pvalue"] > 0.01
        assert white_test(y_het, x)["pvalue"] < 0.05

    def test_reset_detects_omitted_nonlinearity(self):
        rng = np.random.default_rng(7)
        x = rng.uniform(-2, 2, 300)
        y_nonlin = x**2 + rng.normal(scale=0.3, size=300)
        assert ramsey_reset(y_nonlin, x.reshape(-1, 1))["pvalue"] < 0.01
        y_lin = 2 * x + rng.normal(scale=0.3, size=300)
        assert ramsey_reset(y_lin, x.reshape(-1, 1))["pvalue"] > 0.01

    def test_goldfeld_quandt(self):
        rng = np.random.default_rng(8)
        x = np.sort(rng.normal(size=300))
        y = x + np.where(x < np.median(x), 0.5, 3.0) * rng.normal(size=300)
        out = goldfeld_quandt(y, x.reshape(-1, 1))
        assert out["f"] > 1.0
        assert out["pvalue"] < 0.05

    def test_cusum_detects_break(self):
        rng = np.random.default_rng(9)
        n = 200
        x = rng.normal(size=n)
        # Intercept break: signed recursive residuals drift one way -> CUSUM power.
        y = np.concatenate(
            [
                1.0 + x[:100] + rng.normal(scale=0.3, size=100),
                -2.0 + x[100:] + rng.normal(scale=0.3, size=100),
            ]
        )
        out = cusum_recursive(y, x.reshape(-1, 1))
        assert np.any(np.abs(out["cusum"]) > out["upper"])
        # Slope sign flip: mean-zero post-break residuals -> CUSUMSQ detects.
        y_flip = np.concatenate(
            [x[:100] + rng.normal(scale=0.3, size=100), -x[100:] + rng.normal(scale=0.3, size=100)]
        )
        sq = cusum_sq(y_flip, x.reshape(-1, 1))
        dev = np.abs(sq["w"] - sq["frac"])
        assert np.any(dev > sq["upper"] - sq["frac"])
        # stable regression stays inside bounds
        y2 = 2 * x + rng.normal(scale=0.3, size=n)
        out2 = cusum_recursive(y2, x.reshape(-1, 1))
        assert np.mean(np.abs(out2["cusum"]) > out2["upper"]) < 0.2
        sq2 = cusum_sq(y2, x.reshape(-1, 1))
        assert np.mean(np.abs(sq2["w"] - sq2["frac"]) > sq2["upper"] - sq2["frac"]) < 0.3

    def test_vif_and_cooks(self):
        rng = np.random.default_rng(10)
        x1 = rng.normal(size=200)
        x = np.column_stack([x1, x1 * 0.99 + 0.1 * rng.normal(size=200)])
        v = vif(x)
        assert v[0] > 10.0  # collinearity inflates VIF
        _, y, _ = _gen(seed=11)
        d = cooks_distance(y, _gen(seed=11)[0])
        assert np.all(d >= 0)
        assert d.max() < 1.5


class TestRobust:
    def test_theil_sen(self):
        rng = np.random.default_rng(12)
        x = rng.normal(size=100)
        y = 2.0 * x + 1.0 + rng.normal(scale=0.5, size=100)
        out = theil_sen(x, y)
        assert abs(out["slope"] - 2.0) < 0.2
        assert abs(out["intercept"] - 1.0) < 0.3
        # Robust to outlier.
        y[0] += 100.0
        out2 = theil_sen(x, y)
        assert abs(out2["slope"] - 2.0) < 0.3

    def test_huber_resists_outliers(self):
        rng = np.random.default_rng(13)
        x = rng.normal(size=100)
        y = 1.0 + 2.0 * x + rng.normal(scale=0.3, size=100)
        y[5] += 50.0
        hub = huber_regression(x.reshape(-1, 1), y)
        ols_fit = ols(y, x.reshape(-1, 1))
        assert abs(hub["beta"][1] - 2.0) < abs(ols_fit["beta"][1] - 2.0)

    def test_quantile_regression_median(self):
        rng = np.random.default_rng(14)
        x = rng.normal(size=200)
        y = 1.0 + 3.0 * x + rng.normal(scale=0.5, size=200)
        fit = quantile_regression(x.reshape(-1, 1), y, tau=0.5)
        assert abs(fit["beta"][0] - 1.0) < 0.2
        assert abs(fit["beta"][1] - 3.0) < 0.2
        # Median pinball residual ~ 0 weighted count.
        resid = fit["resid"]
        assert abs(np.mean(resid > 0) - 0.5) < 0.1


class TestIV:
    def test_2sls_recovers_beta(self):
        rng = np.random.default_rng(15)
        n = 1000
        z = rng.normal(size=(n, 2))
        u = rng.normal(size=n) * 3.0
        xe = z[:, 0] + z[:, 1] + 0.5 * u + rng.normal(size=n)
        y = 1.0 + 2.0 * xe + u  # endogeneity via shared u
        fit = two_sls(y, xe.reshape(-1, 1), np.zeros((n, 1)), z)
        assert abs(fit["beta"][1] - 2.0) < 0.25
        # OLS is biased upward.
        ols_b = ols(y, xe.reshape(-1, 1))["beta"][1]
        assert abs(ols_b - 2.0) > 0.3

    def test_sargan_valid_instruments(self):
        rng = np.random.default_rng(16)
        n = 800
        z = rng.normal(size=(n, 3))
        xe = z.sum(axis=1) + rng.normal(size=n)
        y = 1.0 + xe + rng.normal(size=n)
        out = sargan_hansen_j(y, xe.reshape(-1, 1), np.zeros((n, 1)), z)
        assert out["df"] == 2.0
        assert out["pvalue"] > 0.01

    def test_underidentified_fails(self):
        rng = np.random.default_rng(17)
        n = 100
        xe = rng.normal(size=n)
        with pytest.raises(ValueError):
            two_sls(
                rng.normal(size=n),
                xe.reshape(-1, 1),
                np.zeros((n, 1)),
                rng.normal(size=(n, 0)).reshape(n, 0),
            )
