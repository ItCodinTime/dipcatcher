"""Causal risk gates: delay 1, no sign-flip, Sharpe scale-invariance."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.returns import sharpe_ratio
from quant_fund.risk.gates import (
    GateSpec,
    apply_gate_stack,
    crash_leverage,
    crc_leverage,
    dd_halt,
    kelly_leverage,
    stepm_leverage,
    vol_target,
)
from quant_fund.risk.overlay import BookRiskOverlay


def test_constant_leverage_leaves_sharpe_unchanged() -> None:
    rng = np.random.default_rng(7)
    r = rng.normal(0.0004, 0.01, size=500)
    sr = float(sharpe_ratio(r)["sharpe"])
    sr2 = float(sharpe_ratio(2.0 * r)["sharpe"])
    sr_neg = float(sharpe_ratio(-r)["sharpe"])
    assert sr2 == pytest.approx(sr, rel=1e-12, abs=1e-12)
    assert sr_neg == pytest.approx(-sr, rel=1e-12, abs=1e-12)


def test_vol_target_is_delay_one() -> None:
    rng = np.random.default_rng(3)
    r = rng.normal(0.0, 0.01, size=120)
    vt = vol_target(r, target=0.025, lookback=60)
    assert float(vt[60]) == pytest.approx(0.0)
    r2 = r.copy()
    r2[80] = 0.5
    vt2 = vol_target(r2, target=0.025, lookback=60)
    # Bar 80's leverage uses vol through 79; mutating r[80] scales the
    # output but must not change the scale factor or earlier bars.
    assert float(vt[79]) == pytest.approx(float(vt2[79]))
    scale = float(vt[80] / r[80])
    scale2 = float(vt2[80] / r2[80])
    assert scale == pytest.approx(scale2, rel=1e-12, abs=1e-12)
    assert abs(float(vt[81]) - float(vt2[81])) > 1e-12


def test_vol_target_on_white_noise_does_not_mint_sharpe5() -> None:
    rng = np.random.default_rng(11)
    r = rng.normal(0.0, 0.012, size=800)
    vt = vol_target(r, target=0.025, lookback=60)
    sr = float(sharpe_ratio(vt[60:])["sharpe"])
    assert abs(sr) < 2.0


def test_dd_halt_stays_cash_after_breach() -> None:
    r = np.concatenate([np.full(40, 0.001), np.full(30, -0.02)])
    halted = dd_halt(r, limit=0.05)
    assert float(np.sum(np.abs(halted[55:]))) == pytest.approx(0.0)


def test_kelly_clips_negative_mean_to_zero() -> None:
    r = np.full(80, -0.001)
    lev = kelly_leverage(r, fraction=0.25, lookback=20)
    assert float(np.max(lev[19:])) == pytest.approx(0.0)


def test_kelly_does_not_sign_flip_a_loser() -> None:
    r = np.concatenate([np.full(40, -0.002), np.full(40, -0.001)])
    gated = apply_gate_stack(r, GateSpec(vol_target=None, dd_limit=None, es_limit=None, crc_alpha=None, crash_lookback=0))
    assert float(np.max(gated.returns)) <= 1e-12


def test_crc_scale_in_unit_interval() -> None:
    rng = np.random.default_rng(5)
    r = rng.normal(0.0002, 0.01, size=120)
    lev = crc_leverage(r, alpha=0.05, lookback=40)
    assert np.all(np.isfinite(lev))
    assert float(np.min(lev)) >= -1e-12
    assert float(np.max(lev)) <= 1.0 + 1e-12


def test_crash_flattens_after_twenty_percent_drop() -> None:
    r = np.zeros(30)
    r[10:20] = -0.03
    lev = crash_leverage(r, lookback=10, crash_return=-0.2)
    assert float(lev[9]) == pytest.approx(1.0)
    assert float(np.min(lev[20:])) == pytest.approx(0.0)


def test_stepm_on_noise_does_not_allow_the_book() -> None:
    rng = np.random.default_rng(2)
    r = rng.normal(0.0, 0.01, size=150)
    lev = stepm_leverage(r, min_obs=80, step=20, n_boot=64, alpha=0.05, seed=2)
    assert np.all((lev == 0.0) | (lev == 1.0))
    # White noise vs 0 should almost never reject; size stays cash.
    assert float(np.mean(lev)) < 0.5


def test_gate_stack_delay_and_snapshot() -> None:
    rng = np.random.default_rng(9)
    r = rng.normal(0.0003, 0.015, size=200)
    out = apply_gate_stack(r, GateSpec(crash_lookback=10))
    assert out.returns.shape == r.shape
    assert out.scale.shape == r.shape
    snap = out.snapshot()
    assert snap["sharpe_scale_invariant_for_constant_leverage"] is True
    assert snap["vol_target_cannot_mint_sharpe5_from_ic0"] is True
    assert snap["live_pnl_claim"] is False
    # First bar cannot see any history.
    assert float(out.scale[0]) == pytest.approx(0.0)


def test_overlay_kelly_crc_do_not_trip_short_warmup() -> None:
    overlay = BookRiskOverlay(vol_target=0.10, dd_limit=0.50, es_limit=1.0, lookback=8)
    prior = 0.10 / 0.20
    overlay.observe(100.0)
    assert overlay.preview_scale() == pytest.approx(prior)
    for nav in (100.5, 99.5, 100.25, 99.75, 100.1, 99.9, 100.0, 100.2, 99.8):
        overlay.observe(nav)
    scale = overlay.preview_scale()
    assert scale > prior
    assert scale <= 1.0
    snap = overlay.snapshot()
    assert snap["kelly_fraction"] == pytest.approx(0.25)
    assert snap["crc_alpha"] == pytest.approx(0.05)
