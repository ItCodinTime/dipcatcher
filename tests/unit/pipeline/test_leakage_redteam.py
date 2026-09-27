"""Tests for quant_fund.validation.leakage_redteam — leaky-oracle red-team battery.

Every test is SYNTHETIC (seeded, deterministic, no market data). The battery
regression-locks the documented DSR/PBO blind spot (Gençay, 2026,
arXiv:2608.27734) so a future change to the gates must move the contrast
deliberately, not silently.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.validation.leakage_redteam import (
    AuditResult,
    BatteryResult,
    StudyResult,
    _matrix_pbo,
    dsr_pbo_battery,
    make_control_oracle,
    make_leaky_oracle,
    run_leaky_oracle_study,
    structural_lookahead_audit,
    trial_count_deflation,
)

N_DAYS = 504  # 2 trading years of SYNTHETIC daily returns


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _leaky_matrix(n_days: int, leakage: float, n_trials: int, seed: int) -> np.ndarray:
    gen = np.random.default_rng(seed)
    return np.column_stack(
        [make_leaky_oracle(n_days, leakage, gen)["returns"] for _ in range(n_trials)]
    )


def _control_matrix(n_days: int, n_trials: int, seed: int) -> np.ndarray:
    gen = np.random.default_rng(seed)
    return np.column_stack([make_control_oracle(n_days, gen)["returns"] for _ in range(n_trials)])


# ---------------------------------------------------------------------------
# oracle makers: construction, stamps, determinism
# ---------------------------------------------------------------------------


def test_leaky_oracle_stream_shape_and_stamp() -> None:
    gen = np.random.default_rng(0)
    out = make_leaky_oracle(N_DAYS, 0.5, gen)
    assert out["returns"].shape == (N_DAYS,)
    assert out["has_leakage"] is True
    assert out["data_source"] == "SYNTHETIC"
    assert out["research_only"] is True
    # positive mean from the traded leaked magnitude, zero genuine skill
    assert float(np.mean(out["returns"])) > 0.05


def test_leaky_oracle_multidim_and_control() -> None:
    gen = np.random.default_rng(1)
    multi = make_leaky_oracle(N_DAYS, 0.5, gen, n_assets=3)
    assert multi["returns"].shape == (3, N_DAYS)
    ctrl = make_control_oracle(N_DAYS, gen)
    assert ctrl["returns"].shape == (N_DAYS,)
    assert ctrl["has_leakage"] is False
    assert ctrl["data_source"] == "SYNTHETIC"
    assert ctrl["research_only"] is True


def test_oracle_makers_are_seeded_deterministic() -> None:
    a = make_leaky_oracle(N_DAYS, 0.5, np.random.default_rng(7))["returns"]
    b = make_leaky_oracle(N_DAYS, 0.5, np.random.default_rng(7))["returns"]
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# (a) BLIND-SPOT DOCUMENTED: leaky promotion > 0.5, control < 0.10
# ---------------------------------------------------------------------------


def test_blind_spot_contrast_at_high_leakage() -> None:
    study = run_leaky_oracle_study([0.5, 0.75], n_mc=200, seed=0)
    for i in range(len(study.leakage_levels)):
        # leaky oracle sails through DSR despite zero genuine skill
        assert study.promotion_rate_leaky[i] > 0.5, (
            f"leakage={study.leakage_levels[i]}: leaky promotion "
            f"{study.promotion_rate_leaky[i]:.3f} must exceed 0.5"
        )
        # control promotes only at the search-deflated false rate
        assert study.promotion_rate_control[i] < 0.10, (
            f"leakage={study.leakage_levels[i]}: control promotion "
            f"{study.promotion_rate_control[i]:.3f} must stay below 0.10"
        )
        # the contrast is the headline: leaky comfortably above control
        assert study.blind_spot_contrast[i] > 0.4
    assert "SYNTHETIC" in study.headline
    assert "never market evidence" in study.headline


# ---------------------------------------------------------------------------
# (b) PBO of a leaky-oracle trial matrix stays high (> 0.4)
# ---------------------------------------------------------------------------


def test_pbo_of_leaky_trial_matrix_stays_high() -> None:
    # one battery call on a leaky matrix: PBO is a valid finite CSCV estimate
    battery = dsr_pbo_battery(_leaky_matrix(N_DAYS, 0.5, 8, seed=11), n_splits=8)
    assert 0.0 <= battery.pbo <= 1.0
    # aggregated over replicates the leaky-matrix PBO sits at its ~0.5 null:
    # CSCV cannot see the leak (Gençay blind spot), and stays above 0.4
    pbos = np.array([_matrix_pbo(_leaky_matrix(N_DAYS, 0.5, 8, seed=s)) for s in range(48)])
    assert pbos.mean() > 0.4
    # replicated study-level mean is tightly concentrated near its 0.5 null
    study = run_leaky_oracle_study([0.5], n_mc=80, seed=7)
    assert study.pbo_leaky[0] > 0.4
    assert study.pbo_control[0] > 0.4  # PBO cannot separate leaky from control


# ---------------------------------------------------------------------------
# battery: PSR/DSR per-trial outputs and promotion flags
# ---------------------------------------------------------------------------


def test_battery_reports_per_trial_psr_dsr_and_pbo() -> None:
    mat = np.column_stack(
        [_leaky_matrix(N_DAYS, 0.6, 4, seed=13), _control_matrix(N_DAYS, 4, seed=14)]
    )
    battery = dsr_pbo_battery(mat, n_splits=8)
    assert isinstance(battery, BatteryResult)
    assert battery.psr.shape == (8,)
    assert battery.dsr.shape == (8,)
    assert battery.dsr_promoted.shape == (8,)
    assert battery.dsr_promoted.dtype == bool
    assert 0.0 <= battery.promotion_rate <= 1.0
    assert battery.pbo == battery.pbo  # not NaN
    # leaky trials promote far more often than control trials
    leaky_flags = battery.dsr_promoted[:4]
    ctrl_flags = battery.dsr_promoted[4:]
    assert leaky_flags.sum() > ctrl_flags.sum()


# ---------------------------------------------------------------------------
# (c) structural look-ahead audit
# ---------------------------------------------------------------------------


def test_structural_audit_flags_planted_lookahead_names() -> None:
    audit = structural_lookahead_audit(
        ["future_return_5d", "lead_1_volume", "shift(-1)_close", "rsi_14", "mom_20"]
    )
    assert audit.fail is True
    assert set(audit.flagged) == {"future_return_5d", "lead_1_volume", "shift(-1)_close"}


def test_structural_audit_passes_clean_names() -> None:
    audit = structural_lookahead_audit(
        ["mom_20", "vol_60", "basis_zscore", "rsi_14", "ema_spread_12_26"]
    )
    assert audit.fail is False
    assert audit.flagged == ()
    assert audit.n_features == 5


def test_structural_audit_extra_patterns_and_case_insensitivity() -> None:
    audit = structural_lookahead_audit(["FUTURE_ret", "mytarget_proxy"], extra_patterns=["target"])
    assert audit.fail is True
    assert set(audit.flagged) == {"FUTURE_ret", "mytarget_proxy"}
    assert "target" in audit.extra_patterns


# ---------------------------------------------------------------------------
# (d) trial-count deflation
# ---------------------------------------------------------------------------


def test_trial_count_deflation_monotone_and_identity_at_one() -> None:
    alpha = 0.05
    assert trial_count_deflation(1, alpha) == pytest.approx(alpha)
    thresholds = [trial_count_deflation(n, alpha) for n in (1, 2, 4, 8, 16, 64, 512)]
    for lo, hi in zip(thresholds[:-1], thresholds[1:], strict=True):
        assert hi < lo
        assert 0.0 < hi < alpha


# ---------------------------------------------------------------------------
# (e) every emitted artifact carries the SYNTHETIC / research_only stamps
# ---------------------------------------------------------------------------


def test_all_artifacts_carry_synthetic_stamps() -> None:
    gen = np.random.default_rng(3)
    artifacts: list = [
        make_leaky_oracle(N_DAYS, 0.5, gen),
        make_control_oracle(N_DAYS, gen),
        dsr_pbo_battery(_leaky_matrix(N_DAYS, 0.5, 4, seed=5), n_splits=8),
        run_leaky_oracle_study([0.25, 0.5], n_mc=20, seed=6),
        structural_lookahead_audit(["future_x", "clean_y"]),
    ]
    for art in artifacts:
        source = art["data_source"] if isinstance(art, dict) else art.data_source
        research = art["research_only"] if isinstance(art, dict) else art.research_only
        assert source == "SYNTHETIC"
        assert research is True


def test_result_dataclasses_are_frozen_and_stamped() -> None:
    battery = dsr_pbo_battery(_leaky_matrix(N_DAYS, 0.5, 4, seed=17), n_splits=8)
    study = run_leaky_oracle_study([0.5], n_mc=10, seed=8)
    audit = structural_lookahead_audit(["future_x"])
    for res in (battery, study, audit):
        assert isinstance(res, (BatteryResult, StudyResult, AuditResult))
        assert res.data_source == "SYNTHETIC"
        assert res.research_only is True
    with pytest.raises(AttributeError):
        battery.pbo = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# (f) fail-closed edges
# ---------------------------------------------------------------------------


def test_oracle_makers_fail_closed_below_one_year() -> None:
    gen = np.random.default_rng(0)
    with pytest.raises(ValueError):
        make_leaky_oracle(251, 0.5, gen)
    with pytest.raises(ValueError):
        make_control_oracle(100, gen)


def test_battery_fail_closed() -> None:
    mat = _control_matrix(N_DAYS, 4, seed=19)
    with pytest.raises(ValueError):  # n_trials < 2
        dsr_pbo_battery(mat[:, :1])
    bad = mat.copy()
    bad[10, 2] = np.nan  # non-finite returns
    with pytest.raises(ValueError):
        dsr_pbo_battery(bad)
    with pytest.raises(ValueError):  # 1-D input
        dsr_pbo_battery(mat[:, 0])
    with pytest.raises(ValueError):  # bad n_splits
        dsr_pbo_battery(mat, n_splits=1)
    with pytest.raises(ValueError):  # bad dsr_level
        dsr_pbo_battery(mat, dsr_level=0.1)


def test_audit_fail_closed_on_empty_and_bad_patterns() -> None:
    with pytest.raises(ValueError):
        structural_lookahead_audit([])
    with pytest.raises(ValueError):
        structural_lookahead_audit(["clean"], extra_patterns=[""])
    with pytest.raises(ValueError):
        structural_lookahead_audit(["clean"], extra_patterns=["   "])


def test_deflation_fail_closed() -> None:
    with pytest.raises(ValueError):
        trial_count_deflation(0, 0.05)
    with pytest.raises(ValueError):
        trial_count_deflation(3.5, 0.05)  # non-integer-valued count
    with pytest.raises(ValueError):
        trial_count_deflation(4, 1.5)
    with pytest.raises(ValueError):
        trial_count_deflation(4, 0.0)
