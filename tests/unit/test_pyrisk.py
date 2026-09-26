"""pyRisk / pyriskmgmt — VaR/ES/EVT estimators and portfolio risk."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as st

from quant_fund.metrics.risk import gaussian_var, historical_es, historical_var
from quant_fund.risk.pyrisk import (
    BackTesting,
    ExpectedShortfall,
    Leadbetter,
    PickandsEstimator,
    Statistics,
    ValueAtRisk,
    hill_es,
    hill_var,
    pickands_xi,
)
from quant_fund.risk.pyriskmgmt import (
    component_var,
    ewma_var_es,
    ewma_variance,
    portfolio_returns,
    portfolio_var_es,
    scale_weights_to_es,
)


def test_statistics_moments() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    s = Statistics(x)
    assert s.minimum() == 1.0 and s.maximum() == 5.0
    assert s.mean() == pytest.approx(3.0)
    assert s.var() == pytest.approx(2.5)
    assert s.std() == pytest.approx(np.sqrt(2.5))
    assert s.skewness() == pytest.approx(0.0, abs=1e-12)
    assert s.kurtosis() == pytest.approx(st.kurtosis(x, fisher=False, bias=False))
    assert s.kurtosis(fisher=True) == pytest.approx(st.kurtosis(x, fisher=True, bias=False))


def test_statistics_tiny_samples_nan() -> None:
    s = Statistics(np.array([1.0]))
    assert np.isnan(s.var()) and np.isnan(s.std())
    assert np.isnan(s.skewness())
    s0 = Statistics(np.array([]))
    assert np.isnan(s0.mean()) and np.isnan(s0.minimum())


def test_var_and_es_match_metrics() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(0.0, 0.01, 500)
    var = ValueAtRisk(r, alpha=0.05)
    assert var.loss_alpha == pytest.approx(0.95)
    assert var.empirical_var() == pytest.approx(historical_var(-r, 0.95))
    assert var.parametrical_var() == pytest.approx(gaussian_var(-r, 0.95))
    # Bootstrap VaR is deterministic under a fixed seed.
    assert var.non_parametrical_var() == pytest.approx(
        ValueAtRisk(r, alpha=0.05).non_parametrical_var()
    )
    es = ExpectedShortfall(r, alpha=0.05)
    assert es.empirical_cvar() == pytest.approx(historical_es(-r, 0.95))
    assert es.empirical_cvar() >= var.empirical_var()
    # Too few observations -> fail-closed NaN.
    assert np.isnan(ValueAtRisk(r[:5], alpha=0.05).non_parametrical_var())


def test_var_rejects_bad_alpha() -> None:
    for bad in (0.0, 1.0, -0.1, np.nan):
        with pytest.raises(ValueError, match="alpha"):
            ValueAtRisk(np.ones(10), alpha=bad)


def test_pickands_and_hill() -> None:
    rng = np.random.default_rng(0)
    losses = np.abs(rng.standard_t(3, 2000))  # heavy tail -> xi > 0
    xi = pickands_xi(np.sort(losses), k=50)
    assert np.isfinite(xi)
    assert np.isnan(pickands_xi(np.ones(10), k=5))  # 4k >= n
    # Constant tail fails monotonicity -> NaN.
    assert np.isnan(pickands_xi(np.ones(100), k=5))
    hv = hill_var(losses, 0.99, k=50)
    he = hill_es(losses, 0.99, k=50)
    assert np.isfinite(hv) and np.isfinite(he) and he >= hv
    # Short sample falls back to the historical estimators.
    small = losses[:10]
    assert hill_var(small, 0.95, k=5) == pytest.approx(historical_var(small, 0.95))
    assert hill_es(small, 0.95, k=5) == pytest.approx(historical_es(small, 0.95))


def test_leadbetter_extremal_index_clusters() -> None:
    # Clustered exceedances: 6 exceedances in 2 clusters -> theta = 1/3.
    x = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    th = Leadbetter(x, threshold=0.5).extremal_index()
    assert th == pytest.approx(2.0 / 6.0)
    # Independent exceedances -> theta ~ 1.
    x2 = np.array([1.0, 0.0] * 4)
    assert Leadbetter(x2, threshold=0.5).extremal_index() == pytest.approx(1.0)
    assert np.isnan(Leadbetter(np.zeros(4), threshold=0.5).extremal_index())
    assert np.isnan(Leadbetter(np.ones(3), threshold=0.5).extremal_index())


def test_pickands_estimator_wrapper() -> None:
    x = np.abs(np.random.default_rng(2).standard_t(3, 500))
    assert PickandsEstimator(x, k=40).gev_parameter() == pytest.approx(pickands_xi(np.sort(x), 40))


def test_backtesting_hits_and_tests() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(0.0, 0.01, 400)
    bt = BackTesting(r)
    var99 = np.quantile(-r, 0.99)
    kup = bt.kupiec_test(var99, alpha=0.01)
    assert 0.0 <= kup["hit_rate"] <= 1.0
    assert kup["n"] == 400 and np.isfinite(kup["lr"])
    cc = bt.christoffersen_test(var99, alpha=0.01)
    assert np.isfinite(cc["lr"]) and 0.0 <= cc["p_value"] <= 1.0
    both = bt.kupiec_christoffersen_test(var99, alpha=0.01)
    assert set(both) == {"kupiec", "christoffersen"}
    st_out = bt.student_test(0.0)
    nt_out = bt.normal_test(0.0)
    assert st_out["n"] == nt_out["n"] == 400
    # Non-finite threshold -> zero hits path.
    assert bt.kupiec_test(np.nan, alpha=0.01)["n"] == 0
    # Diameter over VaR estimates.
    d = bt.var_diameter([0.02, 0.03, 0.025])
    assert d["diameter"] == pytest.approx(0.01)
    assert bt.cvar_diameter([0.02, 0.03])["max"] == 0.03
    assert np.isnan(bt.var_diameter([np.nan])["diameter"])


def test_ewma_variance_and_var_es() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(0.0, 0.01, 300)
    v = ewma_variance(r, lam=0.94)
    assert v > 0
    var, es = ewma_var_es(r, alpha=0.95)
    assert np.isfinite(var) and es >= var
    # Bad inputs fail closed.
    assert np.isnan(ewma_variance(r, lam=1.5))
    assert np.isnan(ewma_variance(r[:1]))
    assert all(np.isnan(x) for x in ewma_var_es(r[:5]))


def test_portfolio_returns_and_validation() -> None:
    a = np.array([[0.01, 0.02], [0.03, 0.04], [np.nan, 0.01]])
    w = np.array([0.5, 0.5])
    pr = portfolio_returns(a, w)
    assert pr.tolist() == pytest.approx([0.015, 0.035, 0.005])
    with pytest.raises(ValueError, match="T, N"):
        portfolio_returns(np.ones(5), w)
    with pytest.raises(ValueError, match="length N"):
        portfolio_returns(a, np.ones(3))


def test_portfolio_var_es_methods() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 0.01, (300, 3))
    w = np.array([0.4, 0.4, 0.2])
    for m in ("historical", "gaussian", "ewma"):
        out = portfolio_var_es(a, w, alpha=0.95, method=m)
        assert out["method"] == m and out["n"] == 300
        assert out["es"] >= out["var"]
        assert out["research_only"] is True
        assert out["execution_claim"] == "paper_backtest"
    with pytest.raises(ValueError, match="method"):
        portfolio_var_es(a, w, method="bogus")


def test_component_var_sums_to_total() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 0.01, (200, 3))
    w = np.array([0.5, 0.3, 0.2])
    cv = component_var(a, w, alpha=0.95)
    assert cv.shape == (3,)
    # Euler decomposition: sum of component VaR = portfolio parametric VaR.
    sigma = np.sqrt(w @ np.cov(a, rowvar=False, ddof=1) @ w)
    assert cv.sum() == pytest.approx(sigma * st.norm.ppf(0.95), rel=1e-9)
    # Fail-closed paths.
    assert np.isnan(component_var(a[:5], w)).all()
    assert np.isnan(component_var(a, np.ones(2))).all()


def test_scale_weights_to_es_caps() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 0.02, (300, 2))
    w = np.array([0.6, 0.4])
    scaled, blob = scale_weights_to_es(a, w, es_limit=0.001)
    assert blob["capped"] is True
    assert np.linalg.norm(scaled) < np.linalg.norm(w)
    uncapped, blob2 = scale_weights_to_es(a, w, es_limit=1.0)
    assert blob2["capped"] is False
    assert np.allclose(uncapped, w)
    with pytest.raises(ValueError, match="es_limit"):
        scale_weights_to_es(a, w, es_limit=0.0)
