"""quant-models engines: BSM, CRR, Heston, HRP, GEX, TSMOM, Krauss, GKX."""

from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.quant_models.beta import blume_beta, cost_of_equity, market_beta
from quant_fund.quant_models.binomial import crr_american, crr_european
from quant_fund.quant_models.black_scholes import (
    bs_price,
    implied_volatility,
    put_call_parity_gap,
)
from quant_fund.quant_models.gex import gex_at, last_hour_decide
from quant_fund.quant_models.gkx import r2_oos
from quant_fund.quant_models.greeks import strangle_volga, validate_greeks
from quant_fund.quant_models.heston import heston_call, heston_put
from quant_fund.quant_models.hrp import (
    generate_ldp_example,
    hcaa_weights,
    hrp_weights,
    inverse_variance_weights,
    long_only_min_variance,
)
from quant_fund.quant_models.krauss import krauss_hit_rate, krauss_walk_forward_proba
from quant_fund.quant_models.monte_carlo import gbm_paths, one_day_short_call_hedge
from quant_fund.quant_models.mvo import long_only_mean_variance
from quant_fund.quant_models.nss import nss_forward, nss_yield
from quant_fund.quant_models.risk_parity import equal_risk_contribution, risk_contributions
from quant_fund.quant_models.svi import fit_svi, svi_butterfly_ok, svi_total_variance
from quant_fund.quant_models.tsmom import tsmom_weights


def test_bs_atm_call_and_parity() -> None:
    s, k, t, r, q, sig = 100.0, 100.0, 1.0, 0.05, 0.0, 0.20
    call = float(bs_price(s, k, t, r, q, sig, "call"))
    put = float(bs_price(s, k, t, r, q, sig, "put"))
    assert call == pytest.approx(10.450583572185565, rel=1e-10)
    assert float(put_call_parity_gap(call, put, s, k, t, r, q)) == pytest.approx(0.0, abs=1e-12)


def test_implied_vol_roundtrip() -> None:
    s, k, t, r, q, sig = 100.0, 100.0, 1.0, 0.05, 0.0, 0.247
    px = float(bs_price(s, k, t, r, q, sig, "call"))
    iv = implied_volatility(s, k, t, r, q, px, "call")
    assert iv == pytest.approx(sig, rel=1e-8)


def test_greeks_match_finite_difference_of_price() -> None:
    worst = validate_greeks(verbose=False)
    assert worst < 2e-4


def test_crr_european_approaches_bs_and_american_put_ge() -> None:
    s, k, t, r, q, sig = 100.0, 100.0, 1.0, 0.05, 0.0, 0.20
    bs = float(bs_price(s, k, t, r, q, sig, "call"))
    crr = crr_european(s, k, t, r, q, sig, n=400, option_type="call")
    assert crr == pytest.approx(bs, rel=5e-3)
    am_put = crr_american(s, k, t, r, q, sig, n=80, option_type="put")
    eu_put = crr_european(s, k, t, r, q, sig, n=80, option_type="put")
    assert am_put >= eu_put - 1e-10


def test_heston_recovers_bs_when_vol_of_vol_is_tiny() -> None:
    s, k, t, r, q, v = 100.0, 100.0, 1.0, 0.05, 0.0, 0.04
    bs = float(bs_price(s, k, t, r, q, np.sqrt(v), "call"))
    h = heston_call(k, t, S0=s, r=r, q=q, v0=v, kappa=8.0, theta=v, xi=1e-3, rho=0.0)
    assert h == pytest.approx(bs, rel=0.03)
    p = heston_put(k, t, S0=s, r=r, q=q, v0=v, kappa=8.0, theta=v, xi=1e-3, rho=0.0)
    gap = float(put_call_parity_gap(h, p, s, k, t, r, q))
    assert gap == pytest.approx(0.0, abs=0.05)


def test_gbm_mean_and_one_day_hedge_mean_near_zero() -> None:
    rng = np.random.default_rng(0)
    paths = gbm_paths(100.0, 0.05, 0.2, 1.0, n_steps=12, n_paths=2000, rng=rng, antithetic=True)
    assert paths.shape == (2000, 13)
    assert float(paths[:, -1].mean()) == pytest.approx(100.0 * np.exp(0.05), rel=0.03)
    _m, pnl = one_day_short_call_hedge(5, n_paths=4000, seed=11)
    assert abs(float(pnl.mean())) < 0.05


def test_svi_fit_and_nss_short_rate() -> None:
    k = np.linspace(-0.6, 0.6, 21)
    w = svi_total_variance(k, 0.04, 0.15, -0.4, 0.0, 0.2)
    fit = fit_svi(k, w)
    assert fit["butterfly_ok"]
    assert svi_butterfly_ok(fit["a"], fit["b"], fit["rho"], fit["m"], fit["sigma"])
    assert fit["rmse"] < 1e-4
    y0 = float(nss_yield(1e-8, 0.03, -0.01, 0.01, 0.005, 1.5, 8.0))
    assert y0 == pytest.approx(0.02, rel=1e-3)
    f = float(nss_forward(1.0, 0.03, -0.01, 0.01, 0.005, 1.5, 8.0))
    assert np.isfinite(f)


def test_hrp_more_diversified_than_minvar_on_ldp_example() -> None:
    panel, _ = generate_ldp_example(n_obs=2000)
    cov = panel.cov().to_numpy()
    corr = panel.corr().to_numpy()
    hrp = hrp_weights(cov, corr)
    ivp = inverse_variance_weights(cov)
    mv = long_only_min_variance(cov)
    hcaa = hcaa_weights(cov, corr)
    for w in (hrp, ivp, mv, hcaa):
        assert w.shape[0] == cov.shape[0]
        assert w.sum() == pytest.approx(1.0, abs=1e-8)
        assert np.all(w >= -1e-12)
    top_hrp = float(np.sort(hrp)[-5:].sum())
    top_mv = float(np.sort(mv)[-5:].sum())
    assert top_hrp < top_mv + 1e-9


def test_erc_risk_contributions_nearly_equal() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(size=(80, 4))
    cov = np.cov(a, rowvar=False)
    w = equal_risk_contribution(cov)
    rc = risk_contributions(w, cov)
    assert rc.sum() == pytest.approx(1.0, abs=1e-8)
    assert rc.std() < 0.02


def test_mvo_prefers_high_mean() -> None:
    mu = np.array([0.01, 0.08])
    cov = np.array([[0.04, 0.0], [0.0, 0.04]])
    w = long_only_mean_variance(mu, cov, risk_aversion=1.0)
    assert w[1] > w[0]


def test_gex_sign_and_last_hour_rule() -> None:
    gamma = np.array([0.01, 0.01])
    oi = np.array([1000.0, 1000.0])
    g = gex_at(gamma, oi, ["C", "P"], 5000.0)
    assert g[0] > 0
    assert g[1] < 0
    follow = last_hour_decide(-1.0, 0.004, 1_000_000, 5000.0)
    assert follow.action == "LONG" and follow.leg == "follow"
    fade = last_hour_decide(1.0, 0.004, 1_000_000, 5000.0)
    assert fade.action == "SHORT" and fade.leg == "fade"
    off = last_hour_decide(1.0, 0.004, 1_000_000, 5000.0, fade_long_gamma=False)
    assert off.action == "FLAT"


def test_tsmom_blume_gkx_strangle() -> None:
    rng = np.random.default_rng(4)
    r = rng.normal(0.0, 0.01, size=(300, 2))
    r[-252:-21, 0] += 0.004
    r[-252:-21, 1] -= 0.004
    w = tsmom_weights(r, lookback=252, skip=21, vol_lookback=60, target_vol=0.4)
    assert w[0] > 0
    assert w[1] < 0
    raw = market_beta(np.array([0.02, 0.03, 0.01, 0.04]), np.array([0.01, 0.02, 0.00, 0.03]))
    assert blume_beta(2.0) == pytest.approx(5.0 / 3.0)
    assert cost_of_equity(0.04, 1.1, 0.05) == pytest.approx(0.095)
    assert np.isfinite(raw)
    y = np.array([0.1, -0.2, 0.3])
    assert r2_oos(y, y) == pytest.approx(1.0)
    assert r2_oos(y, np.zeros_like(y)) == pytest.approx(0.0)
    sv = strangle_volga(100, 110, 90, 0.25, 0.04, 0.0, 0.25, 0.22)
    assert sv["volga_raw"] > 0


def test_krauss_window_beats_chance_on_planted_cs_signal() -> None:
    rng = np.random.default_rng(2)
    n_dates, n_names = 40, 16
    dates = np.repeat(np.arange(n_dates), n_names)
    x = rng.normal(size=(n_dates * n_names, 3))
    y = 0.4 * x[:, 0] + 0.05 * rng.normal(size=x.shape[0])
    proba = krauss_walk_forward_proba(x, y, dates, window_dates=12, min_names=8)
    stats = krauss_hit_rate(y, proba, dates)
    assert stats["n_dates"] >= 10
    assert stats["hit_rate"] > 0.55
    assert stats["live_pnl_claim"] is False


def test_qm_cli_bs_and_gex_decide() -> None:
    runner = CliRunner()
    px = runner.invoke(app, ["qm", "bs-price", "--spot", "100", "--strike", "100"])
    assert px.exit_code == 0
    blob = json.loads(px.stdout)
    assert blob["price"] > 0
    assert blob["live_pnl_claim"] is False
    d = runner.invoke(
        app,
        ["qm", "gex-decide", "--gex", "-1", "--sofar", "0.004", "--spot", "5000"],
    )
    assert d.exit_code == 0
    dec = json.loads(d.stdout)
    assert dec["action"] == "LONG"
    assert dec["broker"] is None
