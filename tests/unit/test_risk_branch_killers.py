"""Mutmut kill tests: boundary semantics of the causal risk gates/overlay.

Every scale here is a delay-1, fail-closed quantity: dropping a finiteness
check, flipping a comparison, or shifting the lookback off by one must change
an asserted number.  Values are pinned to closed-form expectations.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.risk.gates import (
    _EPS,
    GateResult,
    GateSpec,
    _delay1,
    _min_lev,
    apply_gate_stack,
    crash_leverage,
    crc_leverage,
    dd_halt,
    dd_remaining_leverage,
    es_leverage,
    kelly_leverage,
    stepm_leverage,
    vol_target,
    vol_target_leverage,
)
from quant_fund.risk.overlay import BookRiskOverlay

# ------------------------------- gates.vol_target / vol_target_leverage -------


def test_vol_target_writes_next_bar_only() -> None:
    r = np.array([0.01, -0.01] * 40)
    out = vol_target(r, target=0.025, lookback=8, cap=8.0)
    # Loop starts i=lb and writes i+1: indices <= lb stay zero.
    assert out[8] == 0.0
    # window std(ddof=1) ≈ 0.0106904 → sig ≈ 0.16971 → scale ≈ 0.1473139.
    assert out[9] == pytest.approx(-0.01 * 0.1473139127, rel=1e-6)
    assert out[10] == pytest.approx(0.01 * 0.1473139127, rel=1e-6)


def test_vol_target_cap_binds_and_zero_vol_skips() -> None:
    # Tiny vol → target/sig > cap → clipped at cap.
    r = np.array([0.0001, -0.0001] * 40)
    out = vol_target(r, target=0.025, lookback=8, cap=8.0)
    assert out[9] == pytest.approx(-0.0001 * 8.0, rel=1e-9)
    # Constant returns → sig = 0 < _EPS → bar stays unlevered (0).
    flat = vol_target(np.full(20, 0.01), target=0.025, lookback=8)
    assert np.all(flat == 0.0)


def test_vol_target_leverage_prior_and_window_start() -> None:
    r = np.array([0.01, -0.01] * 40)
    lev = vol_target_leverage(r, target=0.025, lookback=8, cap=8.0)
    # Prior = min(1, target/0.20) fills everything before the first full window.
    assert lev[0] == pytest.approx(0.125)
    assert lev[6] == pytest.approx(0.125)
    # First computable index is i = lb - 1 = 7.
    assert lev[7] == pytest.approx(0.1473139127, rel=1e-6)
    # Zero-vol window falls back to the prior, not the clip.
    lev_flat = vol_target_leverage(np.full(20, 0.01), target=0.025, lookback=8)
    assert np.all(lev_flat == pytest.approx(0.125))
    # target >= 0.20 → prior saturates at 1.0.
    lev_hi = vol_target_leverage(np.full(20, 0.01), target=0.4, lookback=8)
    assert np.all(lev_hi == pytest.approx(1.0))


def test_vol_target_rejects_bad_target() -> None:
    r = np.zeros(20)
    with pytest.raises(ValueError, match="finite and positive"):
        vol_target(r, target=0.0)
    with pytest.raises(ValueError, match="finite and positive"):
        vol_target(r, target=float("nan"))


# ------------------------------- gates.dd_halt / dd_remaining_leverage -------


def test_dd_halt_boundary_inclusive() -> None:
    """dd == -limit exactly halts (``<=``); one tick shallower does not."""
    halted = dd_halt(np.array([-0.05, 0.001]), limit=0.05)
    assert halted[1] == 0.0
    alive = dd_halt(np.array([-0.049, 0.001]), limit=0.05)
    assert alive[1] == pytest.approx(0.001)
    # Once halted it never re-enters.
    out = dd_halt(np.array([-0.05, 0.5, 0.5]), limit=0.05)
    assert out[2] == 0.0


def test_dd_remaining_cushion_boundary() -> None:
    # remaining == cushion → scale 0 (inclusive ``<=``).
    lev = dd_remaining_leverage(np.array([-0.045]), limit=0.05, cushion=0.005)
    assert lev[0] == 0.0
    # remaining just above cushion → remaining / limit.
    lev = dd_remaining_leverage(np.array([-0.0449]), limit=0.05, cushion=0.005)
    assert lev[0] == pytest.approx(0.102, rel=1e-9)
    # No drawdown → full scale.
    lev = dd_remaining_leverage(np.array([0.01]), limit=0.05, cushion=0.005)
    assert lev[0] == pytest.approx(1.0)


def test_dd_remaining_nonfinite_return_preserves_equity() -> None:
    lev = dd_remaining_leverage(np.array([float("nan"), -0.045]), limit=0.05)
    # NaN bar does not move equity; the -4.5% bar then exhausts the budget.
    assert lev[0] == pytest.approx(1.0)
    assert lev[1] == 0.0


# ------------------------------- gates.es_leverage / kelly / crc / crash -----


def test_es_leverage_exact_scale_and_warmup() -> None:
    r = np.full(20, -0.01)
    lev = es_leverage(r, es_limit=0.006, lookback=8)
    # Before the first full window (i < lb - 1) leverage stays 1.
    assert lev[6] == pytest.approx(1.0)
    # ES of a flat -1% book is 0.01 → scale = 0.006/0.01 = 0.6.
    assert lev[7] == pytest.approx(0.6)
    assert lev[19] == pytest.approx(0.6)


def test_es_leverage_degenerate_window_keeps_one() -> None:
    # All-zero returns → ES = 0 ≤ _EPS → leverage untouched.
    lev = es_leverage(np.zeros(20), es_limit=0.006, lookback=8)
    assert np.all(lev == 1.0)


def test_kelly_formula_and_clips() -> None:
    r = np.array([0.02, -0.01] * 10)
    lev = kelly_leverage(r, fraction=0.01, lookback=8)
    assert lev[6] == pytest.approx(1.0)
    # μ = 0.005, σ²(ddof=1) = 0.000257142857 → κ μ/σ² ≈ 0.194444.
    assert lev[7] == pytest.approx(0.1944444444, rel=1e-6)
    # Same window with κ=0.25 → 4.861 → capped at cap=1.
    lev = kelly_leverage(r, fraction=0.25, lookback=8, cap=1.0)
    assert lev[7] == pytest.approx(1.0)


def test_kelly_degenerate_windows_halt() -> None:
    # Zero variance → 0, not prior.
    assert np.all(kelly_leverage(np.full(20, 0.002), fraction=0.25, lookback=8)[7:] == 0.0)
    # Negative mean → 0 (ADR-032: no sign-flip).
    lev = kelly_leverage(np.full(20, -0.001), fraction=0.25, lookback=8)
    assert np.all(lev[7:] == 0.0)
    with pytest.raises(ValueError, match="non-negative"):
        kelly_leverage(np.zeros(10), fraction=-0.1)


def test_crc_leverage_binds_on_heavy_tail() -> None:
    r = np.concatenate([np.full(62, -0.005), np.array([-0.4, 0.02]), np.zeros(10)])
    lev = crc_leverage(r, alpha=0.05, lookback=64, step=2)
    assert lev[62] == pytest.approx(1.0)
    # First recompute at i = lb - 1 = 63: allowed/es = 0.005/0.12844 ≈ 0.0389294.
    assert lev[63] == pytest.approx(0.03892944, rel=1e-7)
    # Held between recomputes (step=2): i=64 does not recompute.
    assert lev[64] == pytest.approx(lev[63])


def test_crc_leverage_step_hold_structure() -> None:
    r = np.concatenate([np.full(62, -0.005), np.array([-0.4, 0.02]), np.zeros(10)])
    lev = crc_leverage(r, alpha=0.05, lookback=64, step=3)
    # Recompute cadence at i ≡ 63 (mod 3): 63, 66, 69 hold constant between.
    assert lev[64] == lev[63]
    assert lev[65] == lev[63]


def test_crash_leverage_boundary_and_dead_base() -> None:
    # Trailing return exactly at the (float-reachable) bound → flatten.
    r = np.array([0.0, -0.2])
    lev = crash_leverage(r, lookback=1, crash_return=-0.19999999999999996)
    assert lev[1] == 0.0
    # One tick shallower → no crash.
    lev = crash_leverage(np.array([0.0, -0.19]), lookback=1, crash_return=-0.2)
    assert lev[1] == pytest.approx(1.0)
    # wealth ≈ 0 → base ≤ _EPS → skip (leverage stays 1, not 0).
    lev = crash_leverage(np.array([-1.0, -0.5]), lookback=1)
    assert lev[1] == pytest.approx(1.0)
    # i < lookback never writes.
    lev = crash_leverage(np.full(30, -0.03), lookback=10, crash_return=-0.2)
    assert np.all(lev[:10] == 1.0)
    assert np.all(lev[19:] == 0.0)


def test_stepm_leverage_is_delay_one() -> None:
    rng = np.random.default_rng(0)
    lev = stepm_leverage(rng.normal(0.02, 0.01, 150), min_obs=80, step=21, n_boot=64, seed=2)
    # Decision at i applies to i+1: bar 0 can never trade, and the allow at
    # min_obs-1=79 lands on bar 80.
    assert lev[0] == 0.0
    assert lev[79] == 0.0
    assert lev[80] == 1.0
    # Noise never gets allowed on.
    lev = stepm_leverage(rng.normal(0.0, 0.01, 150), min_obs=80, step=21, n_boot=64, seed=2)
    assert np.all(lev == 0.0)
    assert stepm_leverage(np.array([])).size == 0


def test_stepm_leverage_default_pins() -> None:
    # Pin the documented defaults (min_obs=80, step=21, n_boot=200, seed=7)
    # and the min_obs/step floor clamps.
    rng = np.random.default_rng(0)
    x = rng.normal(0.02, 0.01, 150)
    lev = stepm_leverage(x)
    assert lev[79] == 0.0
    assert lev[80] == 1.0
    assert np.array_equal(lev[80:101], np.ones(21))
    # min_obs clamps at 10: min_obs=10 starts decisions at index 9.
    lev = stepm_leverage(x, min_obs=10, step=7, n_boot=32, seed=3)
    assert lev[8] == 0.0
    assert lev[9] == 0.0  # decision at i=9 applies at i+1=10
    # step=1 (floor) recomputes every bar after min_obs-1.
    lev = stepm_leverage(x, min_obs=10, step=1, n_boot=32, seed=3)
    assert np.array_equal(lev[:10], np.zeros(10))
    assert np.array_equal(lev[10:], np.ones(140))
    # Odd length survives reshape(-1, 1): any reshape(-2, 1) mutant must die.
    lev = stepm_leverage(x[:-1], min_obs=10, step=7, n_boot=32, seed=3)
    assert lev.shape == (149,)


def test_stepm_leverage_holds_between_recomputes() -> None:
    rng = np.random.default_rng(0)
    # Noise: stepm rejects → all bars stay 0.
    lev = stepm_leverage(rng.normal(0.0, 0.01, 60), min_obs=10, step=5, n_boot=64, seed=4)
    assert np.all(lev == 0.0)


def test_delay1_exact() -> None:
    out = _delay1(np.array([0.3, 0.7]))
    assert np.array_equal(out, np.array([0.0, 0.3]))
    out = _delay1(np.array([0.5, 0.6, 0.7]))
    assert np.array_equal(out, np.array([0.0, 0.5, 0.6]))


def test_min_lev_exact() -> None:
    out = _min_lev(np.array([0.5, 1.0, 0.8]), np.array([1.0, 0.4, 0.9]))
    assert np.array_equal(out, np.array([0.5, 0.4, 0.8]))
    # Elementwise min picks the smaller of each pair — max/sum/other reductions die.
    out = _min_lev(np.array([0.2, 0.9]), np.array([0.8, 0.1]), np.array([0.5, 0.6]))
    assert np.array_equal(out, np.array([0.2, 0.1]))


# ------------------------------- gates.apply_gate_stack -----------------------


def test_gate_stack_all_gates_off() -> None:
    r = np.array([0.01, -0.02, 0.03])
    out = apply_gate_stack(
        r,
        GateSpec(
            vol_target=None,
            dd_limit=None,
            es_limit=None,
            kelly_fraction=None,
            crc_alpha=None,
            crash_lookback=0,
        ),
    )
    np.testing.assert_array_equal(out.scale, [0.0, 1.0, 1.0])
    np.testing.assert_array_equal(out.returns, [0.0, -0.02, 0.03])
    assert out.n_halt == 1
    assert out.n_scaled == 0


def test_gate_stack_dd_halt_zeroes_tail() -> None:
    r = np.concatenate([np.full(5, -0.03), np.zeros(10)])
    spec = GateSpec(
        vol_target=None,
        dd_limit=0.05,
        es_limit=None,
        kelly_fraction=None,
        crc_alpha=None,
        crash_lookback=0,
    )
    out = apply_gate_stack(r, spec)
    # Equity breaches -5% on the second -3% bar; everything after is cash.
    assert out.scale[1] == pytest.approx(1.0)
    assert out.scale[2] == pytest.approx(1.0)
    assert np.all(out.scale[3:] == 0.0)
    assert out.n_halt == 13
    assert out.n_scaled == 0


def test_gate_stack_dd_remaining_partial_scale() -> None:
    r = np.concatenate([np.full(5, -0.03), np.zeros(10)])
    spec = GateSpec(
        vol_target=None,
        dd_limit=0.05,
        dd_mode="remaining",
        es_limit=None,
        kelly_fraction=None,
        crc_alpha=None,
        crash_lookback=0,
    )
    out = apply_gate_stack(r, spec)
    # After one -3% bar: dd=0.03 → remaining 0.02 → scale 0.4 on the next bar.
    assert out.scale[2] == pytest.approx(0.4)
    assert np.all(out.scale[3:] == 0.0)
    assert out.n_halt == 13
    assert out.n_scaled == 1


# ------------------------------- overlay.BookRiskOverlay ----------------------


def test_overlay_init_validation() -> None:
    with pytest.raises(ValueError, match="vol_target"):
        BookRiskOverlay(vol_target=0.0)
    with pytest.raises(ValueError, match="vol_target"):
        BookRiskOverlay(vol_target=float("nan"))
    with pytest.raises(ValueError, match="dd_limit"):
        BookRiskOverlay(dd_limit=1.0)
    with pytest.raises(ValueError, match="dd_limit"):
        BookRiskOverlay(dd_limit=-0.05)
    with pytest.raises(ValueError, match="es_limit"):
        BookRiskOverlay(es_limit=0.0)
    with pytest.raises(ValueError, match="kelly_fraction"):
        BookRiskOverlay(kelly_fraction=-0.1)
    with pytest.raises(ValueError, match="crc_alpha"):
        BookRiskOverlay(crc_alpha=1.0)


def test_overlay_lookback_clamps_to_eight() -> None:
    ov = BookRiskOverlay(lookback=2, kelly_lookback=1, crc_lookback=3)
    assert ov.lookback == 8
    assert ov.kelly_lookback == 8
    assert ov.crc_lookback == 8


def test_overlay_observe_filters_bad_navs() -> None:
    ov = BookRiskOverlay()
    for bad in (float("nan"), float("inf"), -float("inf"), 0.0, -10.0):
        ov.observe(bad)
    assert len(ov.navs) == 0
    ov.observe(100.0)
    assert len(ov.navs) == 1


def test_overlay_prior_before_history() -> None:
    ov = BookRiskOverlay(vol_target=0.1)
    assert ov.preview_scale() == pytest.approx(0.5)
    ov.observe(100.0)
    assert ov.preview_scale() == pytest.approx(0.5)


def test_overlay_halt_cushion_boundary() -> None:
    """remaining <= 50bp cushion halts (fail-closed equality)."""
    ov = BookRiskOverlay(vol_target=0.1, dd_limit=0.05, es_limit=9.0)
    ov.observe(100.0)
    ov.observe(95.5)  # dd = 4.5% → remaining == 0.005 → halt
    assert ov.preview_scale() == 0.0
    assert ov.n_halt == 1
    ov2 = BookRiskOverlay(vol_target=0.1, dd_limit=0.05, es_limit=9.0)
    ov2.observe(100.0)
    ov2.observe(95.51)  # dd = 4.49% → remaining 0.0051 > cushion → prior holds
    assert ov2.preview_scale() == pytest.approx(0.5)


def test_overlay_vol_scale_exact() -> None:
    ov = BookRiskOverlay(vol_target=0.1, dd_limit=0.5, es_limit=9.0, lookback=8)
    for nav in (100.0, 101.0, 100.0, 101.0, 100.0, 101.0, 100.0, 101.0, 100.0, 101.0):
        ov.observe(nav)
    assert ov.preview_scale() == pytest.approx(0.5921872711, rel=1e-6)
    assert ov.n_scaled == 1


def test_overlay_kelly_flattens_loser() -> None:
    ov = BookRiskOverlay(vol_target=0.5, dd_limit=0.9, es_limit=9.0, lookback=8, kelly_lookback=8)
    for nav in (100.0, 99.9, 99.8, 99.7, 99.6, 99.5, 99.4, 99.3, 99.2, 99.1):
        ov.observe(nav)
    assert ov.preview_scale() == 0.0


def test_overlay_crc_binds_heavy_tail() -> None:
    ov = BookRiskOverlay(
        vol_target=5.0,
        dd_limit=0.9,
        es_limit=9.0,
        lookback=8,
        kelly_fraction=0.0,
        crc_lookback=8,
    )
    nav = 100.0
    for ret in (0.0, -0.001, 0.001, -0.001, -0.4, 0.001, -0.002, 0.0, 0.001):
        nav *= 1.0 + ret
        ov.observe(nav)
    assert ov.preview_scale() == pytest.approx(0.3926017, rel=1e-6)


def test_overlay_snapshot_contract() -> None:
    ov = BookRiskOverlay(vol_target=0.1)
    snap = ov.snapshot()
    assert snap["n_observe"] == 0
    assert snap["last_scale"] == 1.0
    assert snap["mean_scale"] == 1.0
    assert snap["research_only"] is True
    assert snap["execution_claim"] == "paper_backtest"
    ov.observe(100.0)
    ov.preview_scale()
    snap = ov.snapshot()
    assert snap["n_observe"] == 1
    assert snap["last_scale"] == pytest.approx(0.5)


# ------------------------------- second wave: pins & counters -----------------


def test_gate_result_snapshot_exact() -> None:
    """The audit snapshot is a fixed-key contract: every key/value pinned."""
    empty = GateResult(
        scale=np.array([]),
        returns=np.array([]),
        n_halt=0,
        n_scaled=0,
        notes=[],
        spec=GateSpec(vol_target=None),
    )
    assert empty.snapshot() == {
        "n_halt": 0,
        "n_scaled": 0,
        "mean_scale": 1.0,
        "last_scale": 1.0,
        "sharpe_scale_invariant_for_constant_leverage": True,
        "vol_target_cannot_mint_sharpe5_from_ic0": True,
        "research_only": True,
        "live_pnl_claim": False,
        "notes": [],
    }


def test_vol_target_leverage_default_pins() -> None:
    r = np.array([0.01, -0.01] * 35)
    lev = vol_target_leverage(r)
    # Defaults: lookback 60, target 0.025, cap 8.0 → first computed bar is 59.
    assert lev[58] == pytest.approx(0.125)  # prior = min(1, 0.025/0.20)
    assert lev[59] == pytest.approx(0.1561673062, rel=1e-6)
    assert lev[69] == pytest.approx(0.1561673062, rel=1e-6)


def test_vol_target_leverage_cap_binds_on_calm() -> None:
    # Alternating ±0.0001 → realized vol ~1.6e-3 → target/sig ≈ 15.8 → capped 8.
    lev = vol_target_leverage(np.array([0.0001, -0.0001] * 35), lookback=60, cap=8.0)
    assert lev[59] == pytest.approx(8.0)
    lev = vol_target_leverage(np.array([0.0001, -0.0001] * 35), lookback=60, cap=2.0)
    assert lev[59] == pytest.approx(2.0)


def test_es_leverage_default_lookback() -> None:
    lev = es_leverage(np.full(80, -0.01))
    # Default lookback 63 → first computed bar is 62; flat -1% → ES 1% → 0.6.
    assert lev[61] == pytest.approx(1.0)
    assert lev[62] == pytest.approx(0.6)


def test_es_leverage_limit_never_exceeds_one() -> None:
    # ES (0.001) < es_limit (0.006) → cap at 1.0, never min(2.0, ·) = 2.
    lev = es_leverage(np.full(20, -0.001), es_limit=0.006, lookback=8)
    assert np.all(lev[7:] == 1.0)


def test_es_leverage_window_boundary() -> None:
    # Non-symmetric series: a one-bar window shift must change the ES estimate.
    r = np.array([0.03, -0.012, 0.004, -0.021, 0.007, 0.011, -0.008, 0.017, -0.002, 0.009])
    lev = es_leverage(r, es_limit=0.006, lookback=8)
    assert lev[7] == pytest.approx(0.2857142857, rel=1e-6)


def test_kelly_leverage_defaults() -> None:
    # μ=0.005, σ²≈0.000257 → κ=0.25·f*≈4.86 → cap 1.
    assert kelly_leverage(np.array([0.02, -0.01] * 40))[63] == pytest.approx(1.0)


def test_kelly_fraction_zero_allowed_but_neutral() -> None:
    # fraction=0.0 passes validation (<0 raises) and every computed bar is 0.
    lev = kelly_leverage(np.array([0.02, -0.01] * 10), fraction=0.0, lookback=8)
    assert lev[7] == pytest.approx(0.0)


def test_kelly_break_equivalence_boundary() -> None:
    # f* in (0, 1]: leverage = fraction·f*, not forced 0.
    # Window of alternating ±returns: μ≈0.001, σ²≈0.001997 → f*≈0.5008.
    r = np.concatenate([np.zeros(4), np.tile([0.0428, -0.0408], 4)])
    lev = kelly_leverage(r, fraction=0.5, lookback=8)
    assert lev[7] == pytest.approx(0.25032327, rel=1e-6)


def test_crc_leverage_nondefault_alpha() -> None:
    r = np.concatenate([np.full(62, -0.005), np.array([-0.4, 0.02]), np.zeros(10)])
    lev = crc_leverage(r, alpha=0.5, lookback=64, step=2)
    assert lev[63] != pytest.approx(0.03892944, rel=1e-3)
    assert 0.0 < lev[63] <= 1.0


def test_crc_leverage_flat_losses_keeps_one() -> None:
    # ES of an all-zero loss window is 0 → lev stays 1 (also exercises the
    # isfinite(crc_es) and es>_EPS guard on the happy path).
    lev = crc_leverage(np.zeros(20), lookback=8, step=2)
    assert np.all(lev == 1.0)


def test_crc_leverage_window_exact() -> None:
    # last8 of the irregular series as the CRC window pins calibrate+ES args.
    r = np.array([0.0] * 8 + [-0.02, 0.01, -0.03, 0.005, -0.04, 0.01, -0.015, 0.02])
    lev = crc_leverage(r, lookback=8, step=1)
    window = np.array([-0.02, 0.01, -0.03, 0.005, -0.04, 0.01, -0.015, 0.02])
    losses = -window
    from quant_fund.models.crc import ConformalRiskControl

    crc = ConformalRiskControl(alpha=0.05).calibrate(losses, np.zeros(8))
    allowed = max(float(crc.lambda_hat), 1e-12)
    from quant_fund.metrics.risk import historical_es

    es = float(historical_es(losses, 0.95))
    expected = min(1.0, allowed / es)
    assert lev[7] == pytest.approx(expected, rel=1e-9)


def test_crash_leverage_ignores_nan_for_wealth() -> None:
    # NaN bars must not poison wealth (they are treated as 0), and a real
    # crash after them still flattens.
    r = np.array([0.0, float("nan"), -0.3, 0.0])
    lev = crash_leverage(r, lookback=1, crash_return=-0.2)
    assert lev[2] == 0.0
    lev = crash_leverage(np.array([0.0, float("nan"), -0.1, 0.0]), lookback=1)
    assert lev[2] == pytest.approx(1.0)


def test_crash_leverage_lookback_boundary() -> None:
    # i < lookback never writes: with lookback=2, index 1 keeps 1.
    lev = crash_leverage(np.array([0.0, -0.5, -0.5]), lookback=2, crash_return=-0.2)
    assert lev[1] == pytest.approx(1.0)
    assert lev[2] == 0.0


def test_crash_leverage_default_pins() -> None:
    lev = crash_leverage(np.full(30, -0.03))
    assert lev[9] == pytest.approx(1.0)
    assert np.all(lev[10:] == 0.0)


def test_apply_gate_stack_full_contract() -> None:
    """Non-default spec pinned end-to-end: scale, returns, counters, notes."""
    r = np.array(
        [
            0.03,
            -0.012,
            0.004,
            -0.021,
            0.007,
            0.011,
            -0.008,
            0.017,
            -0.002,
            0.009,
            -0.015,
            0.006,
            0.001,
            -0.019,
            0.013,
            -0.004,
            0.008,
            -0.011,
            0.002,
            0.016,
            -0.006,
            0.005,
            0.012,
            -0.018,
            0.007,
            -0.009,
            0.004,
            -0.002,
            0.015,
            -0.024,
        ]
    )
    spec = GateSpec(
        vol_target=0.5,
        vol_lookback=8,
        vol_cap=3.0,
        dd_limit=0.4,
        dd_mode="remaining",
        dd_cushion=0.01,
        es_limit=0.02,
        es_lookback=8,
        tail_p=0.2,
        kelly_fraction=0.5,
        kelly_lookback=8,
        crc_alpha=0.2,
        crc_lookback=8,
        crash_lookback=3,
        crash_return=-0.1,
        stepm_enable=False,
        periods_per_year=12.0,
    )
    out = apply_gate_stack(r, spec)
    assert out.spec is spec
    np.testing.assert_allclose(
        out.scale,
        np.array(
            [
                0.0,
                1.0,
                0.97,
                0.97988,
                0.92780252,
                0.9447971376,
                0.9716899062,
                0.9519163869,
                0.9935989655,
                0.0,
                0.9886117676,
                0.0,
                0.951282591,
                0.9537338736,
                0.0,
                0.90711293,
                0.0,
                0.0,
                0.0,
                0.0,
                0.8974844783,
                0.0,
                0.8830995714,
                0.9116967663,
                0.8682862245,
                0.8848642281,
                0.86340045,
                0.8728540518,
                0.0,
                0.8681083437,
            ]
        ),
    )
    np.testing.assert_allclose(out.returns, r * out.scale)
    assert out.n_halt == 10
    assert out.n_scaled == 19
    assert out.notes == [
        "Sharpe is invariant to constant leverage (rf=0).",
        "Vol targeting cannot create Sharpe 5 from IC≈0.",
        "Negative Kelly is clipped to 0 (no sign-flip).",
        "blend_weight stays 0. Not a live P&L claim.",
        "vol_target=Moreira–Muir 2017",
        "kelly=Thorp fractional",
        "crc=Angelopoulos et al. 2022",
        "es_halt=Acerbi–Tasche ES",
        "crash=nautica −20%/10d analog",
        "dd=remaining-budget",
    ]
    snap = out.snapshot()
    assert snap["mean_scale"] == pytest.approx(float(np.mean(out.scale)))
    assert snap["last_scale"] == pytest.approx(0.8681083437)


def test_apply_gate_stack_stepm_branch() -> None:
    """stepm_enable exercises the stepm gate: delay-1 allow/deny in the stack."""
    r = np.array([0.01, -0.005, 0.02, -0.01, 0.015, -0.008, 0.012, -0.002] * 4)
    ex = np.random.default_rng(3).normal(0.02, 0.01, 32)
    spec = GateSpec(
        vol_target=None,
        dd_limit=None,
        es_limit=None,
        kelly_fraction=None,
        crc_alpha=None,
        crash_lookback=0,
        stepm_enable=True,
        stepm_min_obs=20,
        stepm_step=7,
        stepm_n_boot=32,
        stepm_alpha=0.2,
    )
    out = apply_gate_stack(r, spec, stepm_excess=ex)
    expected = np.zeros(32)
    expected[21:] = 1.0
    np.testing.assert_array_equal(out.scale, expected)
    assert "stepm=Romano–Wolf 2005 allow/deny" in out.notes


def test_apply_gate_stack_stepm_defaults_to_returns() -> None:
    # stepm_excess=None → the stack runs StepM on the returns themselves.
    r = np.array([0.01] * 40)
    spec = GateSpec(
        vol_target=None,
        dd_limit=None,
        es_limit=None,
        kelly_fraction=None,
        crc_alpha=None,
        crash_lookback=0,
        stepm_enable=True,
        stepm_min_obs=10,
        stepm_step=7,
        stepm_n_boot=32,
        stepm_alpha=0.2,
    )
    out = apply_gate_stack(r, spec)
    assert out.scale.shape == (40,)
    assert "stepm=Romano–Wolf 2005 allow/deny" in out.notes


def test_apply_gate_stack_kelly_off_at_zero_fraction() -> None:
    r = np.array([0.02, -0.01] * 20)
    spec = GateSpec(
        vol_target=None,
        dd_limit=None,
        es_limit=None,
        kelly_fraction=0.0,
        crc_alpha=None,
        crash_lookback=0,
    )
    out = apply_gate_stack(r, spec)
    # fraction=0 skips the Kelly gate entirely: no note, scale untouched.
    assert "kelly=Thorp fractional" not in out.notes
    assert np.all(out.scale[1:] == 1.0)


def test_apply_gate_stack_crash_lookback_zero_disables() -> None:
    r = np.array([0.0, -0.5, 0.01] * 15)
    spec = GateSpec(
        vol_target=None,
        dd_limit=None,
        es_limit=None,
        kelly_fraction=None,
        crc_alpha=None,
        crash_lookback=0,
    )
    out = apply_gate_stack(r, spec)
    assert "crash=nautica −20%/10d analog" not in out.notes
    assert np.all(out.scale[1:] == 1.0)


def test_apply_gate_stack_crash_lookback_one_flattens() -> None:
    r = np.array([0.0, -0.3, 0.01] * 15)
    spec = GateSpec(
        vol_target=None,
        dd_limit=None,
        es_limit=None,
        kelly_fraction=None,
        crc_alpha=None,
        crash_lookback=1,
        crash_return=-0.2,
    )
    out = apply_gate_stack(r, spec)
    # Every -30% bar is a crash → delay-1 flattens the following bar.
    assert out.scale[2] == 0.0
    assert "crash=nautica −20%/10d analog" in out.notes


def test_apply_gate_stack_dd_halt_mode_rebuilds_scale() -> None:
    r = np.array([-0.03] * 5 + [0.0] * 10)
    spec = GateSpec(
        vol_target=None,
        dd_limit=0.05,
        dd_mode="halt",
        es_limit=None,
        kelly_fraction=None,
        crc_alpha=None,
        crash_lookback=0,
    )
    out = apply_gate_stack(r, spec)
    # dd crosses -5% after bar 2; the delay-1 halt flattens bar 3 onward.
    assert np.all(out.scale[3:] == 0.0)
    assert out.n_halt == 13
    assert "dd=halt-and-stay-cash" in out.notes


# ------------------------------- overlay second wave --------------------------


def test_overlay_defaults_stored() -> None:
    ov = BookRiskOverlay()
    assert ov.vol_target == 0.025
    assert ov.dd_limit == 0.05
    assert ov.es_limit == 0.006
    assert ov.tail_p == 0.01
    assert ov.lookback == 63
    assert ov.kelly_fraction == 0.25
    assert ov.kelly_lookback == 63
    assert ov.crc_alpha == 0.05
    assert ov.crc_lookback == 63


def test_overlay_init_boundary_validation() -> None:
    # Anchored message matches (``XX…XX`` wrappers must not survive) on real
    # boundary values.
    with pytest.raises(ValueError, match=r"^vol_target must be finite and positive$"):
        BookRiskOverlay(vol_target=float("nan"))
    with pytest.raises(ValueError, match=r"^vol_target must be finite and positive$"):
        BookRiskOverlay(vol_target=0.0)
    with pytest.raises(ValueError, match=r"^dd_limit must be in \(0, 1\)$"):
        BookRiskOverlay(dd_limit=0.0)
    with pytest.raises(ValueError, match=r"^dd_limit must be in \(0, 1\)$"):
        BookRiskOverlay(dd_limit=1.0)
    with pytest.raises(ValueError, match=r"^dd_limit must be in \(0, 1\)$"):
        BookRiskOverlay(dd_limit=float("inf"))
    with pytest.raises(ValueError, match=r"^es_limit must be finite and positive$"):
        BookRiskOverlay(es_limit=float("nan"))
    with pytest.raises(ValueError, match=r"^es_limit must be finite and positive$"):
        BookRiskOverlay(es_limit=0.0)
    with pytest.raises(ValueError, match=r"^kelly_fraction must be finite and non-negative$"):
        BookRiskOverlay(kelly_fraction=float("nan"))
    with pytest.raises(ValueError, match=r"^kelly_fraction must be finite and non-negative$"):
        BookRiskOverlay(kelly_fraction=-0.1)
    with pytest.raises(ValueError, match=r"^crc_alpha must be in \[0, 1\)$"):
        BookRiskOverlay(crc_alpha=float("nan"))
    with pytest.raises(ValueError, match=r"^crc_alpha must be in \[0, 1\)$"):
        BookRiskOverlay(crc_alpha=-0.01)
    with pytest.raises(ValueError, match=r"^crc_alpha must be in \[0, 1\)$"):
        BookRiskOverlay(crc_alpha=1.0)
    # Boundaries that stay legal: kelly_fraction=0 disables the gate and
    # crc_alpha=0.0 disables the CRC gate.
    BookRiskOverlay(kelly_fraction=0.0)
    BookRiskOverlay(crc_alpha=0.0)


def test_overlay_observe_accepts_small_positive_nav() -> None:
    ov = BookRiskOverlay()
    ov.observe(0.5)
    ov.observe(1.0)
    assert len(ov.navs) == 2


def test_overlay_prior_capped_at_one() -> None:
    # vol_target > 0.20 → prior = min(1, vt/0.2) = 1, not min(2, ·).
    ov = BookRiskOverlay(vol_target=0.5, dd_limit=0.9, es_limit=9.0)
    assert ov.preview_scale() == pytest.approx(1.0)


def test_overlay_n_halt_counts_each_call() -> None:
    ov = BookRiskOverlay(vol_target=0.1, dd_limit=0.05, es_limit=9.0)
    ov.observe(100.0)
    ov.observe(95.5)
    assert ov.preview_scale() == 0.0
    assert ov.preview_scale() == 0.0
    assert ov.n_halt == 2


def test_overlay_scale_one_is_not_counted_as_scaled() -> None:
    # scale == 1.0 exactly → neither halt nor scaled buckets move.
    ov = BookRiskOverlay(
        vol_target=0.5,
        dd_limit=0.9,
        es_limit=9.0,
        lookback=8,
        kelly_fraction=0.0,
        crc_alpha=0.0,
    )
    for _ in range(10):
        ov.observe(100.0)
    assert ov.preview_scale() == 1.0
    assert ov.n_scaled == 0
    assert ov.n_halt == 0


def test_overlay_n_scaled_counts_each_call() -> None:
    ov = BookRiskOverlay(
        vol_target=0.1,
        dd_limit=0.9,
        es_limit=9.0,
        lookback=8,
        kelly_fraction=0.0,
        crc_alpha=0.0,
    )
    for nav in (100.0, 101.0, 100.0, 101.0, 100.0, 101.0, 100.0, 101.0, 100.0, 101.0):
        ov.observe(nav)
    ov.preview_scale()
    ov.preview_scale()
    assert ov.n_scaled == 2


def test_overlay_finite_zero_vol_keeps_scale_one() -> None:
    # A finite zero-vol window is real evidence → scale stays 1.0, not prior.
    ov = BookRiskOverlay(
        vol_target=0.1,
        dd_limit=0.9,
        es_limit=9.0,
        lookback=8,
        kelly_fraction=0.0,
        crc_alpha=0.0,
    )
    for _ in range(10):
        ov.observe(50.0)
    assert ov.preview_scale() == 1.0


def test_overlay_kelly_fraction_zero_skips_kelly() -> None:
    # fraction=0 disables Kelly: a gently rising book is not force-halted.
    ov = BookRiskOverlay(
        vol_target=0.5,
        dd_limit=0.9,
        es_limit=9.0,
        lookback=8,
        kelly_fraction=0.0,
        crc_alpha=0.0,
    )
    for nav in (100.0, 100.5, 100.9, 101.4, 100.8, 101.6, 101.1, 101.9, 101.3, 102.1):
        ov.observe(nav)
    assert ov.preview_scale() > 0.0


def test_overlay_crc_alpha_zero_skips_crc() -> None:
    # alpha=0 disables CRC: the heavy-tail book is scaled only by the ES
    # term (remaining/(3·es) ≈ 0.3926), not by the 0.0099 CRC bound.
    nav = 100.0
    navs = [nav]
    for ret in (0.0, -0.001, 0.001, -0.001, -0.4, 0.001, -0.002, 0.0, 0.001):
        nav *= 1.0 + ret
        navs.append(nav)
    ov = BookRiskOverlay(
        vol_target=5.0,
        dd_limit=0.9,
        es_limit=9.0,
        lookback=8,
        kelly_fraction=0.0,
        crc_alpha=0.0,
    )
    for x in navs:
        ov.observe(x)
    assert ov.preview_scale() == pytest.approx(0.3926017294, rel=1e-6)


def test_overlay_crc_window_boundary() -> None:
    # Exactly crc_lookback observations must engage the gate (>= boundary).
    nav = 100.0
    navs = [nav]
    for ret in (0.0, -0.001, 0.001, -0.001, -0.4, 0.001, -0.002, 0.0, 0.001):
        nav *= 1.0 + ret
        navs.append(nav)
    ov = BookRiskOverlay(
        vol_target=5.0,
        dd_limit=0.9,
        es_limit=9.0,
        lookback=8,
        kelly_fraction=0.0,
        crc_lookback=8,
        crc_alpha=0.5,
    )
    for x in navs[:-1]:
        ov.observe(x)
    # 9 navs → 8 returns → window.size == crc_lookback → gate fires.
    assert ov.preview_scale() == pytest.approx(0.0099009901, rel=1e-6)


def test_overlay_crc_scale_exact() -> None:
    # CRC binds below every other term at alpha=0.5: allowed/crc_es =
    # 0.001/0.101 = 0.0099.
    nav = 100.0
    navs = [nav]
    for ret in (0.0, -0.001, 0.001, -0.001, -0.4, 0.001, -0.002, 0.0, 0.001):
        nav *= 1.0 + ret
        navs.append(nav)
    ov = BookRiskOverlay(
        vol_target=5.0,
        dd_limit=0.9,
        es_limit=9.0,
        lookback=8,
        kelly_fraction=0.0,
        crc_lookback=8,
        crc_alpha=0.5,
    )
    for x in navs:
        ov.observe(x)
    assert ov.preview_scale() == pytest.approx(0.0099009901, rel=1e-6)


def test_overlay_constant_losses_no_crash() -> None:
    # Constant navs → crc_es == 0 → gate leaves scale alone (no ZeroDivision).
    ov = BookRiskOverlay(
        vol_target=0.5,
        dd_limit=0.9,
        es_limit=9.0,
        lookback=8,
        kelly_fraction=0.0,
        crc_lookback=8,
        crc_alpha=0.5,
    )
    for _ in range(10):
        ov.observe(100.0)
    assert ov.preview_scale() == 1.0


def test_overlay_kelly_scale_exact() -> None:
    # Alternating ±4%-ish returns: μ/var ≈ 0.5008 → κ=0.5 → scale ≈ 0.2504.
    rets = np.tile([0.0428, -0.0408], 4)
    navs = 100.0 * np.cumprod(1.0 + np.concatenate([[0.0], rets]))
    ov = BookRiskOverlay(
        vol_target=0.5,
        dd_limit=0.9,
        es_limit=9.0,
        lookback=8,
        kelly_lookback=8,
        kelly_fraction=0.5,
        crc_alpha=0.0,
    )
    for x in navs:
        ov.observe(float(x))
    assert ov.preview_scale() == pytest.approx(0.2503949085, rel=1e-6)


def test_overlay_kelly_negative_mu_halts() -> None:
    # Negative-μ window → scale 0 (ADR-032: never sign-flip the book).
    navs = [100.0, 100.3, 99.7, 100.6, 100.1, 100.9, 99.8, 100.4, 100.7, 100.2]
    ov = BookRiskOverlay(
        vol_target=0.5,
        dd_limit=0.9,
        es_limit=9.0,
        lookback=8,
        kelly_lookback=8,
        kelly_fraction=0.5,
        crc_alpha=0.0,
    )
    for x in navs:
        ov.observe(x)
    assert ov.preview_scale() == 0.0


def test_overlay_snapshot_full_contract() -> None:
    ov = BookRiskOverlay(vol_target=0.1)
    assert ov.snapshot() == {
        "vol_target": 0.1,
        "dd_limit": 0.05,
        "es_limit": 0.006,
        "tail_p": 0.01,
        "lookback": 63,
        "kelly_fraction": 0.25,
        "crc_alpha": 0.05,
        "n_observe": 0,
        "n_halt": 0,
        "n_scaled": 0,
        "last_scale": 1.0,
        "mean_scale": 1.0,
        "sources": ("lprtk/pyRisk", "GianMarcoOddo/pyriskmgmt"),
        "research_only": True,
        "execution_claim": "paper_backtest",
    }


# ------------------------------- third wave ----------------------------------


def test_vol_target_defaults_pinned() -> None:
    # Default kwargs must produce the documented behavior.
    rng = np.random.default_rng(2)
    r = rng.standard_normal(120) * 0.02
    out = vol_target(r)
    assert out[-2] == pytest.approx(0.0012343284924830515)
    assert out[-1] == pytest.approx(7.347789725246678e-05)


def test_vol_target_writes_last_bar() -> None:
    # The loop runs i in [lb, len(r)-1) — the last bar must be scaled.
    rng = np.random.default_rng(2)
    r = rng.standard_normal(120) * 0.02
    assert vol_target(r, lookback=60)[-1] != 0.0


def test_vol_target_lookback_two_allowed() -> None:
    # lb = max(lookback, 2): lookback=2 is legal.
    r = np.array([0.01, -0.02, 0.03, 0.005, -0.01, 0.02])
    out = vol_target(r, lookback=2)
    assert out[3] == pytest.approx(0.00022271770159368694)
    lev = vol_target_leverage(r, lookback=2)
    assert lev[1] == pytest.approx(0.07423923441043106)


def test_vol_target_flat_then_volatile_recovers() -> None:
    # A degenerate (flat) window must not stop the scan of later windows.
    r = np.concatenate([np.zeros(8), np.array([0.02, -0.03] * 4)])
    lev = vol_target_leverage(r, lookback=8)
    assert lev[8] == pytest.approx(0.22271769235585575)
    out = vol_target(r, lookback=8)
    assert out[9] == pytest.approx(-0.006681532525465461)


def test_vol_target_leverage_defaults_and_priors() -> None:
    rng = np.random.default_rng(2)
    r = rng.standard_normal(120) * 0.02
    lev = vol_target_leverage(r)
    # Unobserved bars keep the 20%-vol prior, not the clip of a wrapped window.
    assert lev[0] == pytest.approx(0.125)
    assert lev[59] == pytest.approx(0.07830591344329725)
    assert lev[60] == pytest.approx(0.0781047059010711)


def test_vol_target_window_left_edge_matters() -> None:
    # r[i-lb+1:i+1] — dropping the oldest obs must change sigma.
    r = np.array([-0.10, 0.01, -0.02, 0.03, 0.01, -0.01, 0.02, -0.03, 0.01, -0.02, 0.03, 0.01])
    lev = vol_target_leverage(r, target=0.05, lookback=8)
    assert lev[7] == pytest.approx(0.07643168931613436)


def test_dd_halt_divide_not_multiply() -> None:
    # peak > 1 separates eq/peak from eq*peak.
    r = np.array([0.2, -0.3, 0.05])
    np.testing.assert_array_equal(dd_halt(r, limit=0.05), [0.2, -0.3, 0.0])


def test_dd_halt_exact_limit_is_halt() -> None:
    # dd <= -limit halts: equality must halt.
    r = np.array([0.0, -0.05, 0.03])
    np.testing.assert_array_equal(dd_halt(r, limit=0.05), [0.0, -0.05, 0.0])


def test_dd_remaining_nan_neutralizes_equity() -> None:
    # Non-finite returns skip the equity update (multiplier 1.0, not 2.0).
    lev = dd_remaining_leverage(np.array([0.5, np.nan, -0.3]), limit=0.1, cushion=0.01)
    np.testing.assert_array_equal(lev, [1.0, 1.0, 0.0])


def test_dd_remaining_peak_above_one() -> None:
    # 1 - eq/peak with peak>1 differs from 1 - eq*peak.
    lev = dd_remaining_leverage(np.array([0.5, -0.2]), limit=0.1, cushion=0.01)
    np.testing.assert_array_equal(lev, [1.0, 0.0])


def test_dd_remaining_cushion_boundary_is_halt() -> None:
    # remaining == cushion → halt (``<=``). Dyadic-exact construction.
    lev = dd_remaining_leverage(np.array([1.0, -0.49609375]), limit=0.5, cushion=0.00390625)
    np.testing.assert_array_equal(lev, [1.0, 0.0])


def test_es_leverage_writes_from_lookback() -> None:
    r = np.array([0.01, -0.08] * 6)
    lev = es_leverage(r, es_limit=0.02, lookback=8, tail_p=0.2)
    assert lev[7] == pytest.approx(0.25)
    assert np.all(lev[:7] == 1.0)


def test_es_leverage_window_edges() -> None:
    # The 8-observation window r[i-7:i+1] pins the left edge exactly.
    r = np.concatenate([[-0.10], np.full(11, -0.01)])
    lev = es_leverage(r, es_limit=0.02, lookback=8, tail_p=0.2)
    assert lev[7] == pytest.approx(0.30188679)


def test_es_leverage_flat_then_tail_recovers() -> None:
    # A degenerate ES window must not break the loop.
    r = np.concatenate([np.zeros(8), np.array([0.02, -0.03] * 4)])
    lev = es_leverage(r, es_limit=0.02, lookback=8)
    assert lev[9] == pytest.approx(2.0 / 3.0)


def test_kelly_nan_window_halts() -> None:
    # nan inside the window → non-finite μ/var → 0, not propagating nan.
    r = np.concatenate([np.tile([0.02, -0.01], 9), [np.nan]])
    lev = kelly_leverage(r, fraction=0.25, lookback=8)
    assert lev[18] == 0.0
    assert np.isfinite(lev[18])


def test_kelly_defaults_pinned() -> None:
    rng = np.random.default_rng(2)
    r = rng.standard_normal(120) * 0.02
    lev = kelly_leverage(r)
    assert lev.shape == r.shape
    assert np.all(lev[:62] == 1.0)


def test_crc_defaults_and_recompute_schedule() -> None:
    # Default step=5 recomputes at i=lb-1, lb+4, ...; (i-(lb-1))/step==0 would
    # only fire once.
    r = np.array([0.05, -0.09] * 6)
    lev = crc_leverage(r, alpha=0.05, lookback=8, step=3)
    assert lev[7] == pytest.approx(crc_leverage(r, alpha=0.05, lookback=8, step=3)[7])
    assert lev[10] == pytest.approx(crc_leverage(r, alpha=0.05, lookback=8, step=3)[10])


def test_crc_step_one_recomputes_every_bar() -> None:
    r = np.concatenate([[0.01] * 8, [-0.4, -0.4, -0.4, -0.4]])
    lev = crc_leverage(r, alpha=0.9, lookback=8, step=1)
    assert lev[11] != lev[7]


def test_crc_flat_losses_capped_at_one() -> None:
    # allowed/es can exceed 1; the cap is min(1.0, ·) not min(2.0, ·).
    lev = crc_leverage(np.full(12, -0.001), alpha=0.05, lookback=8)
    assert np.all(lev == 1.0)


def test_crash_wraparound_window_is_not_used() -> None:
    # i < lb would index wealth[i-lb] from the tail — those bars stay at 1.
    r = np.array([-0.5, 0.3, -0.5, 0.3, -0.5, 0.3])
    lev = crash_leverage(r, lookback=3, crash_return=-0.2)
    assert lev[0] == 1.0 and lev[1] == 1.0 and lev[2] == 1.0


def test_crash_trail_divides_by_base() -> None:
    r = np.array([0.5, -0.1, -0.15, -0.2, 0.1] * 3)
    lev = crash_leverage(r, lookback=3, crash_return=-0.15)
    np.testing.assert_array_equal(lev, [1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0])


def test_crash_nan_treated_as_zero_for_wealth() -> None:
    # np.where(isfinite, r, 0.0): nan contributes 0, not +1.
    r = np.array([0.1, np.nan, 0.1, -0.5, 0.2] * 3)
    lev = crash_leverage(r, lookback=3, crash_return=-0.2)
    np.testing.assert_array_equal(lev, [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0])


def test_stepm_step_boundary_fill() -> None:
    # end = min(t, i + step): the fill covers exactly step bars.
    rng = np.random.default_rng(3)
    f = rng.normal(0.001, 0.01, 40)
    lev = stepm_leverage(f, min_obs=10, step=5, n_boot=32)
    assert lev.shape == (40,)
    assert np.all(np.isfinite(lev))


def test_stepm_last_initialized_zero() -> None:
    # If the first recompute were skipped, lev must still start at 0 (deny).
    rng = np.random.default_rng(4)
    f = rng.normal(0.0, 0.01, 30)
    lev = stepm_leverage(f, min_obs=10, step=21, n_boot=16)
    assert np.all(np.isfinite(lev))
    assert lev[0] == 0.0


def test_apply_gate_stack_vol_isolation_exact() -> None:
    r = np.array([0.02, -0.03, 0.01, 0.04, -0.02, 0.03, -0.01, 0.05, -0.02, 0.01, 0.03, -0.02])
    spec = GateSpec(
        vol_target=0.01,
        vol_lookback=8,
        vol_cap=3.0,
        dd_limit=None,
        es_limit=None,
        kelly_fraction=None,
        crc_alpha=None,
        crash_lookback=0,
        periods_per_year=12.0,
    )
    res = apply_gate_stack(r, spec)
    assert res.scale[8] == pytest.approx(0.09953891742315302)
    assert res.scale[11] == pytest.approx(0.10405543028949823)
    # mutating any vol kwarg (lookback=60, cap=8, ppy=252 defaults) flattens or
    # rescales this — the assertions above pin all three.
    res_def = apply_gate_stack(
        r,
        GateSpec(
            vol_target=0.01,
            dd_limit=None,
            es_limit=None,
            kelly_fraction=None,
            crc_alpha=None,
            crash_lookback=0,
        ),
    )
    assert res_def.scale[8] == pytest.approx(0.05)


def test_apply_gate_stack_kelly_fraction_arg_used() -> None:
    # μ/σ² = 0.005/0.018 ≈ 0.28 → κ·f* ≈ 0.14 < 1 (fraction must reach the call).
    r = np.tile([0.1, -0.09], 50)
    spec = GateSpec(
        vol_target=None,
        dd_limit=None,
        es_limit=None,
        kelly_fraction=0.5,
        kelly_lookback=8,
        crc_alpha=None,
        crash_lookback=0,
    )
    res = apply_gate_stack(r, spec)
    assert np.all(res.scale[8:] != 1.0)


def test_apply_gate_stack_crc_args_used() -> None:
    r = np.concatenate([[0.01] * 8, [-0.4, -0.4, -0.4, -0.4], [0.01, 0.01]])
    spec = GateSpec(
        vol_target=None,
        dd_limit=None,
        es_limit=None,
        kelly_fraction=None,
        crc_alpha=0.9,
        crc_lookback=8,
        crash_lookback=0,
    )
    res = apply_gate_stack(r, spec)
    assert res.scale[13] == pytest.approx(4.591836734693878e-12)


def test_apply_gate_stack_crash_return_arg_used() -> None:
    r = np.array([0.05, -0.15, -0.1, -0.12, 0.1] * 2)
    spec = GateSpec(
        vol_target=None,
        dd_limit=None,
        es_limit=None,
        kelly_fraction=None,
        crc_alpha=None,
        crash_lookback=3,
        crash_return=-0.15,
    )
    res = apply_gate_stack(r, spec)
    np.testing.assert_array_equal(res.scale, [0, 1, 1, 1, 0, 1, 1, 1, 0, 0])


def test_apply_gate_stack_dd_cushion_arg_used() -> None:
    # cushion 0.01 vs default 0.005: remaining=0.008 sits between them.
    r = np.array([0.0, -0.092, 0.0, 0.0])
    spec = GateSpec(
        vol_target=None,
        dd_limit=0.1,
        dd_mode="remaining",
        dd_cushion=0.01,
        es_limit=None,
        kelly_fraction=None,
        crc_alpha=None,
        crash_lookback=0,
    )
    res = apply_gate_stack(r, spec)
    assert res.scale[3] == 0.0
    spec_lo = GateSpec(
        vol_target=None,
        dd_limit=0.1,
        dd_mode="remaining",
        dd_cushion=0.005,
        es_limit=None,
        kelly_fraction=None,
        crc_alpha=None,
        crash_lookback=0,
    )
    assert apply_gate_stack(r, spec_lo).scale[3] > 0.0


def test_apply_gate_stack_dd_limit_arg_used() -> None:
    r = np.array([-0.06, -0.01, 0.0] * 4)
    spec = GateSpec(
        vol_target=None,
        dd_limit=0.4,
        dd_mode="halt",
        es_limit=None,
        kelly_fraction=None,
        crc_alpha=None,
        crash_lookback=0,
    )
    res = apply_gate_stack(r, spec)
    assert res.n_halt == 5


def test_apply_gate_stack_nan_keeps_prior_scale() -> None:
    r = np.array([0.5, np.nan, -0.2])
    spec = GateSpec(
        vol_target=None,
        dd_limit=0.1,
        dd_mode="halt",
        es_limit=None,
        kelly_fraction=None,
        crc_alpha=None,
        crash_lookback=0,
    )
    res = apply_gate_stack(r, spec)
    assert res.scale[1] == 1.0


# ----------------------------- overlay wave 3 --------------------------------


def test_overlay_halt_count_boundary_eps() -> None:
    # scale == 1e-12 exactly: ``<=`` counts a halt, ``<`` must not.
    ov = BookRiskOverlay(vol_target=0.2e-12)
    ov.observe(1.0)
    assert ov.preview_scale() == 1e-12
    assert ov.n_halt == 1
    assert ov.n_scaled == 0


def test_overlay_scaled_count_boundary_one_minus_eps() -> None:
    # scale == 1.0 - 1e-12 exactly: ``<`` must not count it as scaled.
    ov = BookRiskOverlay(vol_target=0.2 * (1.0 - 1e-12))
    ov.observe(1.0)
    assert ov.preview_scale() == 1.0 - 1e-12
    assert ov.n_scaled == 0
    assert ov.n_halt == 0


def test_overlay_peak_at_or_below_one_scales() -> None:
    # NAVs at or below 1.0 are legal inputs; ``peak <= 1`` / ``last <= 1``
    # must not trigger the zero-return guard.
    ov = BookRiskOverlay()
    ov.observe(0.95)
    ov.observe(0.98)
    assert ov.preview_scale() == pytest.approx(0.125)


def test_overlay_prior_cap_at_one() -> None:
    # prior = min(1.0, vt/0.20) — vt=0.5 → 1.0, not min(2.0, ·).
    ov = BookRiskOverlay(vol_target=0.5)
    ov.observe(1.0)
    assert ov.preview_scale() == 1.0


def test_overlay_hist_es_dominant_window() -> None:
    # Graded tail: historical ES (0.5) dominates the EWMA estimate (≈0.26)
    # at tail_p=0.01 — both es branches must be wired to the real window.
    rets = np.concatenate([np.full(17, -0.001), [-0.05, -0.1, -0.5], np.full(10, -0.001)])
    navs = 100.0 * np.cumprod(1.0 + rets)
    ov = BookRiskOverlay(
        vol_target=0.5,
        dd_limit=0.9,
        es_limit=0.006,
        tail_p=0.01,
        lookback=20,
        kelly_fraction=0.0,
        kelly_lookback=8,
        crc_alpha=0.0,
    )
    for nav in navs:
        ov.observe(float(nav))
    assert ov.preview_scale() == pytest.approx(0.012)
    # kelly_fraction=0 disables the gate: a ``>= 0`` condition would apply
    # fractional Kelly with f*=0 and halt — scale must stay ES-bound.
    assert ov.preview_scale() > 0.0


def test_overlay_tail_p_reaches_expected_shortfall() -> None:
    # tail_p=0.5 must reach ExpectedShortfall's alpha kwarg: with it the EWMA
    # estimate (≈0.12) dominates and bounds the scale — the default alpha=0.01
    # would take the deeper 0.5 tail and scale to 0.012 instead.
    rets = np.concatenate([np.full(17, -0.001), [-0.05, -0.1, -0.5], np.full(10, -0.001)])
    navs = 100.0 * np.cumprod(1.0 + rets)
    ov = BookRiskOverlay(
        vol_target=0.5,
        dd_limit=0.9,
        es_limit=0.006,
        tail_p=0.5,
        lookback=20,
        kelly_fraction=0.0,
        crc_alpha=0.0,
    )
    for nav in navs:
        ov.observe(float(nav))
    assert ov.preview_scale() == pytest.approx(0.04950367682800607)


def test_overlay_kelly_halts_on_degenerate_variance() -> None:
    # Drift ~1e-9/bar → σ² ≪ _EPS with μ > 0 → kelly denies (0).
    navs = np.cumprod(1.0 + np.linspace(1e-9, 2e-9, 30))
    ov = BookRiskOverlay(
        kelly_fraction=0.25,
        kelly_lookback=8,
        lookback=20,
        dd_limit=0.5,
        vol_target=0.5,
        es_limit=1.0,
        crc_alpha=0.0,
    )
    for nav in navs:
        ov.observe(float(nav))
    assert ov.preview_scale() == 0.0


def test_overlay_crc_disabled_at_alpha_zero() -> None:
    # crc_alpha=0 disables the gate even once the window fills: a ``>= 0``
    # or ``or`` condition would apply CRC and scale to ~1e-10.
    rng = np.random.default_rng(0)
    navs = 100.0 * np.cumprod(1.0 + rng.standard_normal(25) * 0.02)
    ov = BookRiskOverlay(
        vol_target=0.5,
        dd_limit=0.5,
        es_limit=1.0,
        tail_p=0.01,
        lookback=20,
        kelly_fraction=0.0,
        crc_alpha=0.0,
        crc_lookback=8,
    )
    for nav in navs:
        ov.observe(float(nav))
    assert ov.preview_scale() == 1.0


def test_overlay_crc_gate_binds_when_enabled() -> None:
    # With crc_alpha > 0 the gate applies once the window fills —
    # regression pin for the CRC path used by the alpha=0 test above.
    rng = np.random.default_rng(0)
    navs = 100.0 * np.cumprod(1.0 + rng.standard_normal(25) * 0.02)
    ov = BookRiskOverlay(
        vol_target=0.5,
        dd_limit=0.5,
        es_limit=1.0,
        tail_p=0.01,
        lookback=20,
        kelly_fraction=0.0,
        crc_alpha=0.9,
        crc_lookback=8,
    )
    for nav in navs[:10]:
        ov.observe(float(nav))
    assert ov.preview_scale() == pytest.approx(1.894464783471029e-09)


def test_overlay_mean_scale_reflects_observed_scales() -> None:
    # mean_scale must be the mean of recorded scales, not a constant.
    ov = BookRiskOverlay()
    ov.observe(0.95)
    ov.observe(0.98)
    ov.preview_scale()
    assert ov.snapshot()["mean_scale"] == pytest.approx(0.125)


def test_vol_target_sig_exactly_eps_does_not_skip() -> None:
    # ``sig < _EPS`` must be strict: a window whose annualized sigma lands
    # exactly on _EPS is computed, not skipped. Alternating ±m gives
    # std(ddof=1) == 1e-12 bit-exact.
    m = 9.354143466934853e-13
    window = np.tile(np.array([-m, m]), 4)
    assert float(np.std(window, ddof=1)) == _EPS
    r = np.concatenate([np.zeros(10), window, [0.5]])
    lev = vol_target_leverage(r, target=0.5, lookback=8, cap=10.0, periods_per_year=1.0)
    assert lev[17] == 10.0
    out = vol_target(r, target=0.5, lookback=8, cap=10.0, periods_per_year=1.0)
    assert out[18] == 5.0


def test_vol_target_break_does_not_stop_after_flat() -> None:
    # ``continue`` after a degenerate sigma still processes later bars;
    # ``break`` would leave them all at zero.
    r = np.array([0.0, 0.0, 0.0, 0.5, -0.5, 0.5, -0.5])
    out = vol_target(r, target=0.1, lookback=2, cap=5.0, periods_per_year=1.0)
    assert np.any(out[3:] != 0.0)


def test_vol_target_raises_message_anchored() -> None:
    with pytest.raises(ValueError, match="^vol target must be finite and positive$"):
        vol_target([0.01, -0.01], target=float("nan"))
    with pytest.raises(ValueError, match="^vol target must be finite and positive$"):
        vol_target([0.01, -0.01], target=-0.5)


def test_vol_target_cap_binds_above_default() -> None:
    # Default cap=8 must clip leverage>8; cap=9 would not match.
    r = np.tile(np.array([0.001, -0.001]), 20)
    lev = vol_target_leverage(r, target=0.2, lookback=8)
    assert lev[7] == 8.0
    out = vol_target(np.concatenate([r, [0.01]]), target=0.2, lookback=8)
    assert out[40] == pytest.approx(0.01 * 8.0)


def test_vol_target_leverage_writes_only_from_lookback() -> None:
    # i = lb - 1 is the first computed bar; earlier bars keep the prior.
    r = np.concatenate([np.zeros(7), np.tile(np.array([0.05, -0.05]), 10)])
    lev = vol_target_leverage(r, target=0.5, lookback=8, cap=10.0)
    prior = min(1.0, 0.5 / 0.20)
    assert lev[6] == prior
    assert lev[7] != prior


def test_dd_halt_default_limit_is_005() -> None:
    # Documented default limit=0.05: a -6% bar halts, mutating it to 1.05 doesn't.
    out = dd_halt(np.array([-0.06, 0.02, 0.02]))
    np.testing.assert_array_equal(out, [-0.06, 0.0, 0.0])


def test_dd_halt_exact_limit_boundary() -> None:
    # dd == -limit must halt (inclusive). 0.75/1 - 1 is bit-exact -0.25.
    out = dd_halt(np.array([-0.25, 0.5]), limit=0.25)
    np.testing.assert_array_equal(out, [-0.25, 0.0])


def test_dd_remaining_default_limit_pin() -> None:
    lev = dd_remaining_leverage(np.array([-0.06, 0.01]))
    assert lev[0] == 0.0


def test_dd_remaining_nan_multiplies_by_one_not_two() -> None:
    # A non-finite return must leave equity unchanged (×1.0), not double it:
    # after a 50% drawdown, eq staying 0.5 keeps lev at 0; eq=1.0 resets dd.
    lev = dd_remaining_leverage(np.array([-0.5, np.nan]), limit=0.05, cushion=0.005)
    assert lev[1] == 0.0


def test_es_leverage_keeps_warmup_bars_at_one() -> None:
    # n < lb: the real loop range(lb-1, n) is empty, so lev stays all ones.
    # A mutant starting at i=0 evaluates the clamped window r[0:1] = [-0.05]
    # and would write lev[0] = 0.006/0.05 = 0.12.
    r = np.concatenate([[-0.05], [0.02] * 19, [-0.05] * 5])
    lev = es_leverage(r, es_limit=0.006, lookback=63, tail_p=0.1)
    assert np.all(lev == 1.0)


def test_kelly_default_fraction_and_lookback_pin() -> None:
    # Defaults fraction=0.25, lookback=63: lev[62] is computed, lev pattern
    # changes if either default moves.
    r = np.tile(np.array([0.1, -0.09]), 40)
    lev = kelly_leverage(r)
    mu = float(np.mean(r[:63]))
    var = float(np.var(r[:63], ddof=1))
    expected = float(min(1.0, 0.25 * (mu / var)))
    assert lev[62] == pytest.approx(expected)
    assert lev[62] != pytest.approx(min(1.0, 1.25 * (mu / var)))


def test_kelly_raises_message_anchored() -> None:
    with pytest.raises(ValueError, match="^kelly fraction must be finite and non-negative$"):
        kelly_leverage([0.01, -0.01], fraction=float("nan"))
    with pytest.raises(ValueError, match="^kelly fraction must be finite and non-negative$"):
        kelly_leverage([0.01, -0.01], fraction=-0.1)


def test_crc_default_lookback_is_63() -> None:
    # lookback=63 default: lev[62] reflects the first calibration window.
    r = np.tile(np.array([0.01, -0.4]), 40)
    lev = crc_leverage(r, alpha=0.9, step=1)
    assert lev[62] != 1.0


def test_crc_step_one_recomputes_each_bar_strict() -> None:
    # With step=1 every bar recomputes; a floor of 2 would hold i=8 at i=7's value.
    r = np.concatenate([[0.01] * 8, [-0.4, -0.4, -0.4, -0.4]])
    lev = crc_leverage(r, alpha=0.9, lookback=8, step=1)
    assert lev[8] != lev[7]


def test_crc_computes_from_bar_zero_when_step_one() -> None:
    # With n < lb the real loop never runs (all 1s). A range(0) start would
    # compute at i = 0 (mod 1 == 0) on the clamped window r[:1]; a negative
    # first return there would write lev[0] below 1.
    r = np.concatenate([[-0.4], [0.01] * 6])
    lev = crc_leverage(r, alpha=0.9, lookback=8, step=1)
    np.testing.assert_array_equal(lev, np.ones(7))


def test_crc_unbound_result_clips_to_one() -> None:
    # allowed/es > 1 must clip to 1.0 — min(2.0, ·) would leak leverage > 1.
    rng = np.random.RandomState(0)
    lev = crc_leverage(rng.normal(1e-4, 1e-3, 80), alpha=0.95, lookback=10, step=1)
    assert np.all(lev <= 1.0)


def test_crash_leverage_no_prewarm_writes() -> None:
    # i < lb would read wealth[i-lb] from the tail (wraparound) and could zero
    # the first bars; they must stay 1.
    r = np.array([-0.5, 0.9, 0.9, 0.9, -0.5, -0.5])
    lev = crash_leverage(r, lookback=3, crash_return=-0.2)
    np.testing.assert_array_equal(lev[:3], [1.0, 1.0, 1.0])
    assert lev[5] == 0.0


def test_crash_degenerate_base_continues_not_breaks() -> None:
    # A base below _EPS skips that bar only; later windows still evaluate.
    r = np.array([-0.9999999999999, 1e12, -0.5, -0.5, -0.5])
    lev = crash_leverage(r, lookback=3, crash_return=-0.2)
    assert lev[3] == 1.0
    assert lev[4] == 0.0


def test_stepm_default_fixture_transitions() -> None:
    # Deterministic fixture: first decision window (i=79) does not reject,
    # windows at i=100/121/142 do. Pins step=21, n_boot=200, seed=7, and the
    # f[:i+1] panel boundary — a single bootstrap/param change flips i=79.
    x = np.random.default_rng(3).normal(0.003, 0.012, 150)
    lev = stepm_leverage(x)
    assert np.array_equal(lev[:80], np.zeros(80))
    assert np.array_equal(lev[80:101], np.zeros(21))
    assert np.array_equal(lev[101:], np.ones(49))


def test_stepm_second_fixture_kills_param_changes() -> None:
    # Complementary fixture where the i=80 decision DOES reject under the
    # documented params but flips under n_boot=201 / seed=8 / n_boot=2000.
    x = np.random.default_rng(34).normal(0.003, 0.012, 150)
    lev = stepm_leverage(x)
    assert np.array_equal(lev[:80], np.zeros(80))
    assert np.array_equal(lev[80:], np.ones(70))


def test_stepm_step_one_recomputes_every_bar_fixture() -> None:
    # step floor must be 1: with step=2 consecutive bars would share decisions.
    x = np.random.default_rng(3).normal(0.003, 0.012, 150)
    lev = stepm_leverage(x, min_obs=80, step=1, n_boot=200)
    lev2 = stepm_leverage(x, min_obs=80, step=2, n_boot=200)
    assert not np.array_equal(lev, lev2)


def test_stepm_seed_eight_fixture() -> None:
    # seed must reach the bootstrap: on this fixture the decision at i=121
    # rejects under seed=8 but not seed=7.
    x = np.random.default_rng(10).normal(0.003, 0.012, 150)
    lev = stepm_leverage(x, min_obs=80, step=21, n_boot=200, seed=7)
    lev8 = stepm_leverage(x, min_obs=80, step=21, n_boot=200, seed=8)
    assert not np.array_equal(lev, lev8)


def test_stepm_n_boot_two_thousand_fixture() -> None:
    # n_boot must reach stepm(): n_boot=2000 flips the i=79 decision on this data.
    x = np.random.default_rng(43).normal(0.003, 0.012, 150)
    lev = stepm_leverage(x, min_obs=80, step=21, n_boot=200)
    lev2k = stepm_leverage(x, min_obs=80, step=21, n_boot=2000)
    assert not np.array_equal(lev, lev2k)


def test_stepm_params_reach_stepm_call() -> None:
    # n_boot/alpha/seed must be forwarded to stepm() — dropping any of them
    # silently changes the resample count, threshold, or determinism.
    import quant_fund.risk.gates as gates_mod

    real = gates_mod.stepm
    calls: list[dict[str, object]] = []

    def spy(panel: object, **kwargs: object) -> object:
        calls.append(kwargs)
        return real(panel, **kwargs)  # type: ignore[arg-type]

    gates_mod.stepm = spy  # type: ignore[attr-defined]
    try:
        x = np.random.default_rng(3).normal(0.003, 0.012, 100)
        stepm_leverage(x, min_obs=20, step=5, n_boot=32, alpha=0.1, seed=11)
    finally:
        gates_mod.stepm = real  # type: ignore[attr-defined]
    assert calls
    assert all(c.get("n_boot") == 32 for c in calls)
    assert all(c.get("alpha") == 0.1 for c in calls)
    assert all(c.get("seed") == 11 for c in calls)


def test_apply_gate_stack_vol_cap_arg_used() -> None:
    # spec.vol_cap must reach vol_target_leverage: calm returns push
    # target/sig above the cap, so dropping it to the 8.0 default is visible.
    r = np.tile(np.array([0.001, -0.001]), 40)
    spec = GateSpec(
        vol_target=0.2,
        vol_cap=0.5,
        vol_lookback=8,
        dd_limit=None,
        es_limit=None,
        kelly_fraction=None,
        crc_alpha=None,
        crash_lookback=0,
    )
    res = apply_gate_stack(r, spec)
    assert res.scale[9] == pytest.approx(0.5)


def test_apply_gate_stack_min_composes_across_gates() -> None:
    # _min_lev must take the running min, not replace it: with vol ≈ 0.003
    # and Kelly ≈ 1 the composite stays at the vol level.
    r = np.random.default_rng(2).normal(0.001, 0.17, 80)
    spec = GateSpec(
        vol_target=0.01,
        vol_lookback=8,
        dd_limit=None,
        es_limit=None,
        kelly_fraction=0.5,
        kelly_lookback=8,
        crc_alpha=None,
        crash_lookback=0,
    )
    res = apply_gate_stack(r, spec)
    assert res.scale[8] == pytest.approx(0.0028676849635591656)
    assert np.all(res.scale < 0.5)


def test_apply_gate_stack_kelly_fraction_reaches_call() -> None:
    # Exact pin: dropping fraction= to the 0.25 default halves the scale.
    r = np.tile(np.array([0.1, -0.09]), 50)
    spec = GateSpec(
        vol_target=None,
        dd_limit=None,
        es_limit=None,
        kelly_fraction=0.5,
        kelly_lookback=8,
        crc_alpha=None,
        crash_lookback=0,
    )
    res = apply_gate_stack(r, spec)
    window = r[1:9]
    expected = 0.5 * float(np.mean(window)) / float(np.var(window, ddof=1))
    assert res.scale[9] == pytest.approx(min(1.0, expected))


def test_apply_gate_stack_es_gate_present() -> None:
    # Removing the es_leverage call leaves scale at 1 where ES binds.
    r = np.concatenate([[0.02] * 60, [-0.03] * 20])
    spec = GateSpec(
        vol_target=None,
        dd_limit=None,
        es_limit=0.005,
        es_lookback=10,
        tail_p=0.1,
        kelly_fraction=None,
        crc_alpha=None,
        crash_lookback=0,
    )
    res = apply_gate_stack(r, spec)
    assert res.scale[61] == pytest.approx(1.0 / 6.0)


def test_apply_gate_stack_es_lookback_arg_used() -> None:
    # spec.es_lookback=10 vs the function default 63: binding starts earlier.
    r = np.concatenate([[0.02] * 30, [-0.03] * 20])
    spec = GateSpec(
        vol_target=None,
        dd_limit=None,
        es_limit=0.005,
        es_lookback=10,
        tail_p=0.1,
        kelly_fraction=None,
        crc_alpha=None,
        crash_lookback=0,
    )
    res = apply_gate_stack(r, spec)
    assert res.scale[40] < 1.0
    res63 = apply_gate_stack(
        r,
        GateSpec(
            vol_target=None,
            dd_limit=None,
            es_limit=0.005,
            es_lookback=63,
            tail_p=0.1,
            kelly_fraction=None,
            crc_alpha=None,
            crash_lookback=0,
        ),
    )
    assert not np.array_equal(res.scale, res63.scale)


def test_apply_gate_stack_stepm_excess_arg_used() -> None:
    # stepm_excess must feed the StepM panel; using returns instead changes
    # the decisions where the two series disagree.
    r = np.random.default_rng(1).normal(0.003, 0.012, 150)
    excess = np.random.default_rng(5).normal(-0.004, 0.012, 150)
    spec = GateSpec(
        vol_target=None,
        dd_limit=None,
        es_limit=None,
        kelly_fraction=None,
        crc_alpha=None,
        crash_lookback=0,
        stepm_enable=True,
        stepm_min_obs=30,
        stepm_step=21,
        stepm_n_boot=64,
    )
    res = apply_gate_stack(r, spec, stepm_excess=excess)
    res_r = apply_gate_stack(r, spec)
    assert not np.array_equal(res.scale, res_r.scale)


def test_apply_gate_stack_stepm_min_composes() -> None:
    # _min_lev over [running lev, stepm] must not drop the running lev:
    # where StepM accepts (lev 1) the vol gate's ~0.0087 must still apply.
    r = np.random.default_rng(3).normal(0.01, 0.05, 120)
    spec = GateSpec(
        vol_target=0.01,
        vol_lookback=8,
        dd_limit=None,
        es_limit=None,
        kelly_fraction=None,
        crc_alpha=None,
        crash_lookback=0,
        stepm_enable=True,
        stepm_min_obs=30,
        stepm_step=21,
        stepm_n_boot=64,
    )
    res = apply_gate_stack(r, spec)
    assert res.scale[119] == pytest.approx(0.008677041991059767)


def test_apply_gate_stack_stepm_step_arg_used() -> None:
    # spec.stepm_step must reach stepm_leverage — the 21-bar default would
    # recompute on a different schedule than step=10.
    r = np.random.default_rng(3).normal(0.003, 0.012, 150)
    spec = GateSpec(
        vol_target=None,
        dd_limit=None,
        es_limit=None,
        kelly_fraction=None,
        crc_alpha=None,
        crash_lookback=0,
        stepm_enable=True,
        stepm_min_obs=30,
        stepm_step=10,
        stepm_n_boot=64,
    )
    res = apply_gate_stack(r, spec)
    spec_default = GateSpec(
        vol_target=None,
        dd_limit=None,
        es_limit=None,
        kelly_fraction=None,
        crc_alpha=None,
        crash_lookback=0,
        stepm_enable=True,
        stepm_min_obs=30,
        stepm_step=21,
        stepm_n_boot=64,
    )
    res_d = apply_gate_stack(r, spec_default)
    assert not np.array_equal(res.scale, res_d.scale)


def test_apply_gate_stack_stepm_n_boot_arg_used() -> None:
    # spec.stepm_n_boot=200 vs the dropped-arg 2000: the i=79 decision flips
    # on this fixture (verified against stepm() directly).
    r = np.random.default_rng(43).normal(0.003, 0.012, 150)
    spec = GateSpec(
        vol_target=None,
        dd_limit=None,
        es_limit=None,
        kelly_fraction=None,
        crc_alpha=None,
        crash_lookback=0,
        stepm_enable=True,
        stepm_min_obs=80,
        stepm_step=21,
        stepm_n_boot=200,
    )
    res = apply_gate_stack(r, spec)
    assert res.scale[81] == 1.0


def test_apply_gate_stack_stepm_alpha_arg_used() -> None:
    # spec.stepm_alpha=0.01 tightens the threshold; on this fixture the
    # decision at i=80 rejects at 0.05 (p ≈ 0.0199) but not at 0.01.
    r = np.random.default_rng(1).normal(0.003, 0.012, 100)
    spec = GateSpec(
        vol_target=None,
        dd_limit=None,
        es_limit=None,
        kelly_fraction=None,
        crc_alpha=None,
        crash_lookback=0,
        stepm_enable=True,
        stepm_min_obs=80,
        stepm_step=21,
        stepm_n_boot=64,
        stepm_alpha=0.01,
    )
    res = apply_gate_stack(r, spec)
    res_wide = apply_gate_stack(
        r,
        GateSpec(
            vol_target=None,
            dd_limit=None,
            es_limit=None,
            kelly_fraction=None,
            crc_alpha=None,
            crash_lookback=0,
            stepm_enable=True,
            stepm_min_obs=80,
            stepm_step=21,
            stepm_n_boot=64,
            stepm_alpha=0.05,
        ),
    )
    assert not np.array_equal(res.scale, res_wide.scale)


def test_apply_gate_stack_halt_rebuild_stays_finite() -> None:
    # rebuilt = scaled/r overflows to inf for |r| >> 0 but tiny magnitude:
    # np.where must keep the prior scale there, not insert a non-finite value.
    r = np.array([0.01, -0.3, 1e-300, 0.01] * 5)
    spec = GateSpec(
        vol_target=None,
        dd_limit=0.05,
        es_limit=None,
        kelly_fraction=None,
        crc_alpha=None,
        crash_lookback=0,
    )
    res = apply_gate_stack(r, spec)
    assert np.isfinite(res.scale).all()
    assert res.scale.dtype == np.float64


def test_apply_gate_stack_halt_count_boundary() -> None:
    # scale exactly == 1e-12 counts as a halt (``<=``), not as scaled (``>``):
    # vol_target = 0.2e-12 makes the prior exactly 1e-12.
    r = np.array([0.01, -0.01] * 3)
    spec = GateSpec(
        vol_target=0.2e-12,
        dd_limit=None,
        es_limit=None,
        kelly_fraction=None,
        crc_alpha=None,
        crash_lookback=0,
    )
    res = apply_gate_stack(r, spec)
    assert np.all(res.scale[1:] == pytest.approx(1e-12))
    assert res.n_halt == 6
    assert res.n_scaled == 0


def test_apply_gate_stack_scaled_count_upper_boundary() -> None:
    # scale exactly == 1 - 1e-12 is NOT scaled (``<``), and not a halt.
    r = np.array([0.01, -0.01] * 3)
    spec = GateSpec(
        vol_target=0.2 * (1.0 - 1e-12),
        dd_limit=None,
        es_limit=None,
        kelly_fraction=None,
        crc_alpha=None,
        crash_lookback=0,
    )
    res = apply_gate_stack(r, spec)
    assert np.all(res.scale[1:] == pytest.approx(1.0 - 1e-12))
    assert res.n_halt == 1
    assert res.n_scaled == 0


def test_overlay_prior_scale_capped_at_one() -> None:
    # The vol-target prior is capped at 1.0: vol_target=0.4 > 0.20 must give
    # 1.0, not 2.0 — on both early-return paths (no navs, and <8-bar window).
    assert BookRiskOverlay(vol_target=0.4).preview_scale() == 1.0
    o = BookRiskOverlay(vol_target=0.4)
    o.observe(1.0)
    o.observe(1.01)
    assert o.preview_scale() == 1.0
