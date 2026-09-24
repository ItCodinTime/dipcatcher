"""pyRisk / pyriskmgmt in-repo engines and the paper-book overlay."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.risk import historical_es, historical_var
from quant_fund.risk.overlay import BookRiskOverlay
from quant_fund.risk.pyrisk import BackTesting, ExpectedShortfall, ValueAtRisk, pickands_xi
from quant_fund.risk.pyriskmgmt import ewma_var_es, portfolio_var_es, scale_weights_to_es


def test_pyrisk_empirical_matches_dipcatcher_historical() -> None:
    rng = np.random.default_rng(7)
    rets = rng.normal(0.0002, 0.01, size=400)
    engine = ValueAtRisk(rets, alpha=0.01)
    losses = -rets
    assert engine.empirical_var() == historical_var(losses, 0.99)
    es = ExpectedShortfall(rets, alpha=0.01)
    assert es.empirical_cvar() == historical_es(losses, 0.99)


def test_pyrisk_parametric_and_evt_finite() -> None:
    rng = np.random.default_rng(3)
    rets = rng.normal(0.0, 0.012, size=300)
    var = ValueAtRisk(rets, alpha=0.05)
    es = ExpectedShortfall(rets, alpha=0.05)
    assert np.isfinite(var.parametrical_var())
    assert np.isfinite(var.non_parametrical_var(n_iter=2_000))
    assert np.isfinite(var.extreme_var(k=8))
    assert np.isfinite(es.parametrical_cvar())
    assert np.isfinite(es.extreme_cvar(k=8))
    xi = pickands_xi(np.sort(-rets), 8)
    assert isinstance(xi, float)


def test_pyrisk_kupiec_on_calibrated_gaussian() -> None:
    rng = np.random.default_rng(11)
    rets = rng.normal(0.0, 0.01, size=800)
    var = ValueAtRisk(rets, alpha=0.05).empirical_var()
    out = BackTesting(rets).kupiec_test(var, alpha=0.05)
    assert out["n"] == 800
    assert np.isfinite(out["hit_rate"])


def test_pyriskmgmt_scale_to_es() -> None:
    rng = np.random.default_rng(2)
    r = rng.normal(0.0, 0.02, size=(120, 4))
    w = np.array([0.4, 0.3, 0.2, 0.1])
    blob = portfolio_var_es(r, w, alpha=0.95, method="historical")
    assert np.isfinite(blob["es"])
    scaled, info = scale_weights_to_es(r, w, es_limit=1e-6, alpha=0.95, method="historical")
    assert float(np.abs(scaled).sum()) <= float(np.abs(w).sum()) + 1e-12
    assert info["capped"] is True
    ewma = ewma_var_es(r @ w, alpha=0.95)
    assert np.isfinite(ewma[0]) and np.isfinite(ewma[1])


def test_overlay_uses_prior_only_during_warmup() -> None:
    overlay = BookRiskOverlay(vol_target=0.10, dd_limit=0.50, es_limit=1.0, lookback=8)
    prior = 0.10 / 0.20
    overlay.observe(100.0)
    assert overlay.preview_scale() == pytest.approx(prior)
    for nav in (100.5, 99.5, 100.25, 99.75, 100.1, 99.9, 100.0, 100.2, 99.8):
        overlay.observe(nav)
    # Nine observations produce eight returns: the first full volatility window.
    scale = overlay.preview_scale()
    assert scale > prior
    assert scale <= 1.0


def test_overlay_low_and_high_finite_volatility_are_not_clipped_by_prior() -> None:
    low = BookRiskOverlay(vol_target=0.10, dd_limit=0.50, es_limit=1.0, lookback=8)
    for nav in (100.0, 100.01, 100.02, 100.03, 100.04, 100.05, 100.06, 100.07, 100.08):
        low.observe(nav)
    assert low.preview_scale() == pytest.approx(1.0)

    high = BookRiskOverlay(vol_target=0.10, dd_limit=0.50, es_limit=1.0, lookback=8)
    for nav in (100.0, 110.0, 99.0, 111.0, 98.0, 112.0, 97.0, 113.0, 96.0):
        high.observe(nav)
    scale = high.preview_scale()
    assert 0.0 < scale < 0.5


def test_overlay_flattens_when_drawdown_budget_spent() -> None:
    overlay = BookRiskOverlay(vol_target=1.0, dd_limit=0.05, es_limit=1.0, lookback=8)
    overlay.observe(100.0)
    assert overlay.preview_scale() == 1.0
    overlay.observe(100.0)
    overlay.observe(93.0)
    assert overlay.preview_scale() == 0.0
    snap = overlay.snapshot()
    assert snap["n_halt"] >= 1
    assert snap["dd_limit"] == 0.05
