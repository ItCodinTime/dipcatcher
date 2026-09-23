"""Data-snooping battery: Reality Check, SPA, StepM, MCS.

Statistical properties (size, power, invariance, ordering, fail-closed edges).
Research-diagnostic only — never a live Sharpe / P&L claim.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.snooping import (
    model_confidence_set,
    reality_check,
    spa_test,
    stepm,
)

T = 320
K = 12
SIG = 0.01


def _null_panel(seed: int, *, t: int = T, k: int = K) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, SIG, size=(t, k))


def _winner_panel(seed: int, *, bump: float = 0.004, col: int = 3) -> np.ndarray:
    f = _null_panel(seed)
    f[:, col] += bump
    return f


# --- Reality Check -----------------------------------------------------------


def test_reality_check_determinism_and_statistic() -> None:
    f = _winner_panel(0)
    a = reality_check(f, n_boot=400, seed=11)
    b = reality_check(f, n_boot=400, seed=11)
    assert a.p_value == b.p_value and a.statistic == b.statistic
    assert a.best_index == 3
    assert a.statistic == pytest.approx(np.sqrt(T) * float(f[:, 3].mean()))
    assert 0.0 <= a.p_value <= 1.0
    assert a.n_obs == T and a.n_strategies == K


def test_reality_check_size_under_null() -> None:
    ps = [reality_check(_null_panel(100 + s), n_boot=300, seed=s).p_value for s in range(40)]
    assert np.mean(ps) > 0.20  # not systematically tiny
    assert float(np.mean([p < 0.05 for p in ps])) <= 0.15


def test_reality_check_power_on_strong_winner() -> None:
    rc = reality_check(_winner_panel(7, bump=0.006), n_boot=500, seed=1)
    assert rc.p_value < 0.01
    assert rc.best_mean > 0.004


def test_reality_check_monotone_in_common_shift() -> None:
    """A common location shift moves the statistic by sqrt(T)·c and lowers p."""
    f = _null_panel(2)
    base = reality_check(f, n_boot=400, seed=2)
    shifted = reality_check(f + 0.003, n_boot=400, seed=2)
    assert shifted.statistic == pytest.approx(base.statistic + np.sqrt(T) * 0.003)
    assert shifted.p_value < base.p_value
    down = reality_check(f - 0.003, n_boot=400, seed=2)
    assert down.p_value > base.p_value


# --- SPA ---------------------------------------------------------------------


def test_spa_variants_and_scale_invariance() -> None:
    f = _winner_panel(3)
    spa = spa_test(f, n_boot=400, seed=5)
    # Upper never recenters → conservative; lower/consistent reject the winner.
    assert spa.p_upper >= spa.p_consistent
    assert spa.p_upper >= spa.p_lower
    assert spa.p_consistent < 0.05
    scaled = spa_test(f * 7.0, n_boot=400, seed=5)
    # Studentization makes SPA invariant to positive rescaling.
    assert scaled.p_consistent == pytest.approx(spa.p_consistent, abs=1e-12)
    assert scaled.p_lower == pytest.approx(spa.p_lower, abs=1e-12)
    assert scaled.statistic == pytest.approx(spa.statistic, abs=1e-9)
    assert spa.best_index == 3


def test_spa_consistent_equals_lower_without_bad_columns() -> None:
    """No significantly-negative column → consistent and lower recentering coincide."""
    f = _winner_panel(4, bump=0.006)
    spa = spa_test(f, n_boot=600, seed=6)
    assert spa.p_consistent == spa.p_lower
    assert spa.p_upper > spa.p_consistent


def test_spa_does_not_reject_when_all_columns_are_bad() -> None:
    """All-bad panel: max E[f] <= 0 holds, so no variant may reject."""
    f = _null_panel(4) - 0.004
    spa = spa_test(f, n_boot=600, seed=6)
    assert spa.p_lower > 0.5 and spa.p_consistent > 0.5 and spa.p_upper > 0.5


def test_spa_size_under_null() -> None:
    ps = [spa_test(_null_panel(200 + s), n_boot=300, seed=s).p_consistent for s in range(30)]
    assert float(np.mean([p < 0.05 for p in ps])) <= 0.15


def test_spa_drops_constant_columns() -> None:
    f = _null_panel(8)
    f = np.column_stack([f, np.full(T, 0.001)])  # zero-variance strategy
    spa = spa_test(f, n_boot=300, seed=8)
    assert spa.n_dropped == 1
    assert spa.n_strategies == K
    assert len(spa.studentized) == K


# --- StepM -------------------------------------------------------------------


def test_stepm_fwer_under_global_null() -> None:
    rejects = 0
    reps = 30
    for s in range(reps):
        st = stepm(_null_panel(300 + s), n_boot=300, alpha=0.10, seed=s)
        rejects += 1 if st.n_rejected > 0 else 0
    assert rejects / reps <= 0.20  # nominal 0.10 with simulation noise


def test_stepm_rejects_only_the_winner() -> None:
    st = stepm(_winner_panel(9, bump=0.006), n_boot=500, alpha=0.05, seed=10)
    assert st.n_rejected >= 1
    assert st.rejected[3] is True
    assert st.adjusted_p[3] < 0.01
    # Every other strategy must have a clearly larger adjusted p-value.
    others = [st.adjusted_p[i] for i in range(K) if i != 3]
    assert min(others) > st.adjusted_p[3]


def test_stepm_adjusted_p_defines_rejections() -> None:
    st = stepm(_winner_panel(11, bump=0.005), n_boot=400, alpha=0.10, seed=12)
    for i in range(K):
        assert st.rejected[i] == (st.adjusted_p[i] <= st.alpha)
    assert st.n_rejected == sum(st.rejected)
    # Adjusted p-values are ordered by the studentized statistic (step-down).
    order = st.order
    assert len(order) == K
    assert order[0] == 3


def test_stepm_single_strategy_edge() -> None:
    f = _winner_panel(13)[:, [3]]
    st = stepm(f, n_boot=300, alpha=0.05, seed=13)
    assert st.n_strategies == 1
    assert st.rejected[0] is True
    assert st.adjusted_p[0] < 0.05


# --- MCS ---------------------------------------------------------------------


def test_mcs_includes_all_under_null() -> None:
    mcs = model_confidence_set(_null_panel(21), n_boot=400, alpha=0.10, seed=21)
    assert mcs.n_included == K
    assert all(p >= mcs.alpha for p in mcs.p_values)


def test_mcs_eliminates_a_bad_model() -> None:
    f = _null_panel(22)
    f[:, 5] -= 0.006
    mcs = model_confidence_set(f, n_boot=500, alpha=0.10, seed=22)
    assert mcs.included[5] is False
    assert 5 in mcs.elimination_order
    assert mcs.n_included == K - 1
    assert mcs.p_values[5] < mcs.alpha
    for i in range(K):
        if mcs.included[i]:
            assert mcs.p_values[i] >= mcs.alpha


# --- shared fail-closed edges ------------------------------------------------


def test_short_panels_return_honest_nan() -> None:
    f = _null_panel(31, t=6)
    rc = reality_check(f, n_boot=200, seed=1)
    assert np.isnan(rc.p_value) and rc.n_obs == 6 and rc.best_index == -1
    spa = spa_test(f, n_boot=200, seed=1)
    assert np.isnan(spa.p_consistent) and spa.studentized == ()
    st = stepm(f, n_boot=200, seed=1)
    assert st.n_rejected == 0 and all(np.isnan(p) for p in st.adjusted_p)
    mcs = model_confidence_set(f, n_boot=200, seed=1)
    assert mcs.n_included == 0


def test_nonfinite_rows_dropped_with_count() -> None:
    f = _winner_panel(41)
    f[0, :] = np.nan
    f[5, 2] = np.inf
    rc = reality_check(f, n_boot=300, seed=2)
    assert rc.n_rows_dropped == 2
    assert rc.n_obs == T - 2
    assert np.isfinite(rc.p_value)


def test_single_strategy_vector_input() -> None:
    f = _winner_panel(51)[:, 3]
    rc = reality_check(f, n_boot=300, seed=3)
    assert rc.n_strategies == 1 and np.isfinite(rc.p_value)
    mcs = model_confidence_set(f, n_boot=300, seed=3)
    assert mcs.n_strategies == 1 and mcs.n_included == 1


def test_validation_fail_closed() -> None:
    f = _null_panel(61)
    with pytest.raises(ValueError, match="n_boot"):
        reality_check(f, n_boot=0)
    with pytest.raises(ValueError, match="n_boot"):
        spa_test(f, n_boot=0)
    with pytest.raises(ValueError, match="alpha"):
        stepm(f, alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        model_confidence_set(f, alpha=1.0)
    with pytest.raises(ValueError, match="block"):
        reality_check(f, block=0.5)
    with pytest.raises(ValueError, match="1-D"):
        reality_check(np.zeros((2, 2, 2)), n_boot=10)
    empty = reality_check(np.empty((0, 3)), n_boot=10)
    assert np.isnan(empty.p_value) and empty.n_obs == 0
