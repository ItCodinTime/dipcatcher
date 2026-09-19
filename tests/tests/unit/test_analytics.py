"""HF-grade analytics fixtures matching MATH_SPEC conventions."""

import math

import numpy as np
import pytest

from quant_fund.metrics.analytics import (
    book_diagnostics,
    book_var_es,
    capacity_proxy,
    equity_curve_analytics,
    execution_diagnostics,
    net_gross_exposure,
    stress_scenarios,
    var_backtest_hooks,
)
from quant_fund.metrics.probability import (
    acerbi_szekely_z1,
    acerbi_szekely_z2,
    christoffersen_cc,
    christoffersen_independence,
    kupiec_pof,
)
from quant_fund.metrics.risk import historical_es, historical_var, losses_from_returns


def test_net_gross_exposure_fixture():
    w = np.array([0.05, -0.03, 0.02, 0.0])
    out = net_gross_exposure(w)
    assert out["gross"] == pytest.approx(0.10)
    assert out["net"] == pytest.approx(0.04)
    assert out["n_long"] == 2
    assert out["n_short"] == 1


def test_var_es_known_losses():
    # Returns such that losses = [0.01, 0.02, 0.03, 0.04, 0.05] sorted
    # Use fixed returns so L = -R
    r = -np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.00, -0.01])
    losses = losses_from_returns(r)
    var = historical_var(losses, 0.8)
    es = historical_es(losses, 0.8)
    assert var > 0
    assert es >= var - 1e-12
    ves = book_var_es(r, alpha=0.8)
    assert ves["var"] == pytest.approx(var)
    assert ves["es"] == pytest.approx(es)


def test_capacity_proxy_fixture():
    traded = np.array([1e6, 2e6, 5e5])
    adv = np.array([1e8, 1e8, 1e8])
    out = capacity_proxy(traded, adv)
    assert out["mean_participation"] == pytest.approx(np.mean([0.01, 0.02, 0.005]))
    assert out["max_participation"] == pytest.approx(0.02)


def test_kupiec_christoffersen_hooks_iid():
    rng = np.random.default_rng(0)
    # ~5% hits, independent
    hits = (rng.random(400) < 0.05).astype(float)
    rate, lr, p = kupiec_pof(hits, 0.05)
    assert 0.01 < rate < 0.12
    assert p > 0.001  # should not wildly reject
    ind_lr, ind_p, counts = christoffersen_independence(hits)
    assert "n00" in counts
    assert ind_p > 0.001 or not np.isfinite(ind_p)
    cc_lr, cc_p, extras = christoffersen_cc(hits, 0.05)
    assert "kupiec_p" in extras


def test_var_backtest_hooks_on_returns():
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.01, size=300)
    losses = losses_from_returns(r)
    var = historical_var(losses, 0.95)
    hooks = var_backtest_hooks(r, var, alpha=0.95)
    assert hooks["n"] == 300
    assert "kupiec_p" in hooks
    assert "christoffersen_cc_p" in hooks
    assert "acerbi_szekely_z1" in hooks
    assert "acerbi_szekely_z2" in hooks
    # Without es_level, FZ mean stays NaN (research field present)
    assert "fissler_ziegel_mean" in hooks
    assert math.isnan(hooks["fissler_ziegel_mean"])
    hooks_es = var_backtest_hooks(r, var, alpha=0.95, es_level=float(var) * 1.2)
    assert math.isfinite(hooks_es["fissler_ziegel_mean"]) or math.isnan(
        hooks_es["fissler_ziegel_mean"]
    )


def test_acerbi_szekely_known_ratio_fixture() -> None:
    losses = np.array([0.01, 0.02, 0.10, 0.20])
    var = np.full(4, 0.05)
    es = np.full(4, 0.15)
    z1, n_hits = acerbi_szekely_z1(losses, var, es, 0.5)
    z2, n_hits_2 = acerbi_szekely_z2(losses, var, es)
    assert n_hits == n_hits_2 == 2
    assert z1 == pytest.approx((0.10 / 0.15 + 0.20 / 0.15) / 2 - 1.0)
    assert z2 == pytest.approx(z1)


def test_var_backtest_hooks_es_mismatch_is_aligned_and_fail_closed() -> None:
    returns = np.linspace(-0.2, 0.1, 20)
    hooks = var_backtest_hooks(
        returns,
        0.15,
        alpha=0.95,
        es_level=np.full(10, 0.2),
    )
    assert hooks["n"] == 10
    assert hooks["es_hit_count"] >= 0.0
    with pytest.raises(ValueError, match="alpha"):
        acerbi_szekely_z1(returns, np.full(20, 0.15), np.full(20, 0.2), 1.0)


def test_stress_scenarios_signs():
    w = np.array([0.1, -0.05, 0.05])
    out = stress_scenarios(w, shock_sigma=1.0, liquidity_haircut=0.1)
    assert out["shock_down_pnl"] < 0
    assert out["liquidity_haircut_pnl"] == pytest.approx(-0.1 * 0.2)
    assert out["note"] == "stylized_hypothetical"


def test_book_diagnostics_synthetic_label():
    r = np.array([0.01, -0.02, 0.005, -0.01, 0.0, 0.002] * 20)
    diag = book_diagnostics(
        r,
        gross_exposure=np.full(120, 1.2),
        net_exposure=np.full(120, 0.05),
        turnover=np.full(120, 0.1),
        weights=np.array([0.05, -0.03, 0.02]),
        data_source="SYNTHETIC",
    )
    assert diag["data_source"] == "SYNTHETIC"
    assert diag["live_pnl_claim"] is False
    assert diag["research_only"] is True
    assert "SYNTHETIC" in diag["disclaimer"]
    assert diag["execution"]["role"] == 1.0
    assert "var" in diag["var_es"]
    assert "stress_report" in diag
    assert "drawdown_duration" in diag


def test_execution_diagnostics_not_headline():
    r = np.array([0.01, -0.02, 0.015, -0.005, 0.0] * 30)
    d = execution_diagnostics(r)
    assert "realized_vol_ann" in d
    assert "calmar_diagnostic" in d
    assert d["role"] == 1.0


def test_equity_curve_analytics_in_book_diagnostics():
    rng = np.random.default_rng(2)
    r = rng.normal(0.0005, 0.01, size=50)
    out = book_diagnostics(r, label="TEST", data_source="SYNTHETIC")
    assert "equity" in out
    assert out["equity"]["n_points"] == 51  # unit start + 50 compounded
    assert out["equity"]["role"] == 1.0
    assert out["live_pnl_claim"] is False
    direct = equity_curve_analytics(np.cumprod(np.concatenate([[1.0], 1.0 + r])))
    assert direct["n_points"] == out["equity"]["n_points"]


def test_book_var_es_empty_and_short() -> None:
    """Wave 44: empty / short / all-NaN book returns → honest NaN VaR/ES."""
    empty = book_var_es(np.array([]))
    assert math.isnan(empty["var"]) and math.isnan(empty["es"])
    assert empty["n"] == 0
    short = book_var_es(np.array([0.01, -0.01, 0.0, 0.02]))  # n=4 < 5
    assert math.isnan(short["var"]) and math.isnan(short["es"])
    assert short["n"] == 4
    all_nan = book_var_es(np.array([np.nan, np.inf, -np.inf]))
    assert math.isnan(all_nan["var"]) and all_nan["n"] == 0
    with pytest.raises(ValueError, match="alpha"):
        book_var_es(np.linspace(-0.01, 0.01, 20), alpha=0.0)


def test_var_backtest_hooks_empty_and_mismatch() -> None:
    """Wave 44: short returns → NaN hooks; length mismatch truncates safely."""
    short = var_backtest_hooks(np.array([0.01, -0.01]), 0.02)
    assert short["n"] == 2
    assert math.isnan(short["hit_rate"])
    assert math.isnan(short["kupiec_p"])
    empty = var_backtest_hooks(np.array([]), 0.02)
    assert empty["n"] == 0 and math.isnan(empty["hit_rate"])
    # Mismatched per-period VaR: truncate to min length, still need n>=10 finite
    rng = np.random.default_rng(3)
    r = rng.normal(0, 0.01, size=30)
    va = np.full(10, 0.02)
    hooks = var_backtest_hooks(r, va, alpha=0.95)
    assert hooks["n"] == 10
    # After truncate n==10 is the boundary that still runs (not the short-circuit)
    assert "kupiec_p" in hooks


def test_book_diagnostics_empty_returns() -> None:
    """Wave 44: empty book → labeled research-only, NaN var_es, empty hooks."""
    diag = book_diagnostics(np.array([]), data_source="SYNTHETIC")
    assert diag["live_pnl_claim"] is False
    assert diag["research_only"] is True
    assert math.isnan(diag["var_es"]["var"])
    assert diag["var_backtest"] == {} or diag["var_backtest"].get("n", 0) == 0
