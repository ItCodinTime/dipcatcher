"""Tests for quant_fund.metrics.watch — WATCH conformal test martingales."""

import numpy as np
import pytest

from quant_fund.metrics.watch import (
    CompositeJumper,
    SimpleJumper,
    WATCHMonitor,
    diagnose_shift,
    gaussian_density_ratio,
    nearest_neighbor_scores,
    shiryaev_roberts,
    weighted_conformal_pvalue,
)


def test_uniform_pvalue_is_the_upper_rank() -> None:
    scores = np.array([1.0, 3.0, 2.0, 5.0])
    # Test point 5.0 is strictly largest, so only itself counts: p = 1/4.
    assert weighted_conformal_pvalue(scores, test_index=-1) == pytest.approx(0.25)
    # Ties are counted (conservative u=1): two 3s, test is the last 3.
    tied = np.array([3.0, 1.0, 3.0])
    assert weighted_conformal_pvalue(tied, test_index=-1) == pytest.approx(2.0 / 3.0)
    assert weighted_conformal_pvalue(tied, test_index=-1, tie_breaker=0.25) == pytest.approx(
        1.0 / 6.0
    )


def test_constant_exchangeable_stream_does_not_force_a_watch_alarm() -> None:
    # Conservative p=1 on every tied observation makes the positive-epsilon
    # jumper grow exponentially, even under this null. Online ties are smoothed.
    monitor = WATCHMonitor(alpha=0.05, adapt_threshold=1e6, min_calibration=500, seed=0)
    steps = [monitor.update(0.0, 0.0) for _ in range(40)]
    assert all(0.0 < step.p_y < 1.0 and 0.0 < step.p_x < 1.0 for step in steps)
    assert not steps[-1].alarm
    assert not steps[-1].x_alarm


def test_weights_reweight_the_rank() -> None:
    scores = np.array([0.0, 10.0, 1.0])
    weights = np.array([100.0, 1.0, 1.0])
    # Test score 1.0. Scores >= 1 are the 10 and the test. Their weight is 2/102.
    p = weighted_conformal_pvalue(scores, weights, test_index=-1)
    assert p == pytest.approx(2.0 / 102.0)


def test_jumper_with_unit_rate_stays_at_one() -> None:
    jumper = SimpleJumper(jump_rate=1.0)
    rng = np.random.default_rng(0)
    for p in rng.random(50):
        assert jumper.update(float(p)) == pytest.approx(1.0)


def test_mean_jumper_respects_the_wealth_floor() -> None:
    jumper = CompositeJumper((1e-3, 1e-2, 1e-1, 1.0))
    rng = np.random.default_rng(1)
    for p in rng.random(200):
        wealth = jumper.update(float(p))
        assert wealth >= jumper.floor - 1e-9


def test_jumper_false_alarm_rate_respects_ville() -> None:
    rng = np.random.default_rng(2)
    alpha = 0.05
    level = 1.0 / alpha
    alarms = 0
    n_paths = 200
    horizon = 250
    for _ in range(n_paths):
        jumper = CompositeJumper()
        hit = False
        for p in rng.random(horizon):
            if jumper.update(float(p)) >= level:
                hit = True
                break
        alarms += int(hit)
    assert alarms / n_paths <= alpha + 0.04


def test_small_pvalues_grow_the_jumper() -> None:
    jumper = CompositeJumper()
    for _ in range(40):
        jumper.update(0.01)
    assert jumper.value >= 20.0


def test_nearest_neighbor_score_on_a_line() -> None:
    x = np.array([0.0, 1.0, 3.0])
    scores = nearest_neighbor_scores(x)
    assert np.allclose(scores, [1.0, 1.0, 2.0])


def test_gaussian_plugin_upweights_the_target_mean() -> None:
    rng = np.random.default_rng(3)
    cal = rng.normal(0.0, 1.0, size=(80, 1))
    test = rng.normal(3.0, 1.0, size=(40, 1))
    w_cal, w_test = gaussian_density_ratio(cal, test)
    near = np.abs(cal[:, 0] - 3.0) < 0.5
    far = np.abs(cal[:, 0]) < 0.5
    assert int(near.sum()) > 0 and int(far.sum()) > 0
    assert float(np.mean(w_cal[near])) > float(np.mean(w_cal[far]))
    assert np.all(w_test > 0.0)


def test_diagnose_shift_labels() -> None:
    assert diagnose_shift(False, False, False) == "none"
    assert diagnose_shift(False, True, True) == "benign_covariate"
    assert diagnose_shift(True, False, False) == "concept"
    assert diagnose_shift(True, True, True) == "extreme_covariate"


def test_shiryaev_roberts_on_a_constant_martingale() -> None:
    # M_t = 1 for t = 0..3 ⇒ SR = 1 * (1+1+1) = 3 at the last step.
    assert shiryaev_roberts(np.ones(4)) == pytest.approx(3.0)


def test_concept_shift_alarms_the_label_martingale() -> None:
    rng = np.random.default_rng(4)
    alarms = 0
    for _ in range(12):
        local = np.random.default_rng(rng.integers(1_000_000))
        monitor = WATCHMonitor(alpha=0.1, adapt_threshold=50.0, min_calibration=30)
        alarmed = False
        for _ in range(40):
            step = monitor.update(float(local.normal()), float(local.normal()))
            assert step.diagnosis in {"none", "benign_covariate", "concept", "extreme_covariate"}
        for _ in range(30):
            step = monitor.update(float(local.normal(loc=8.0)), float(local.normal()))
            alarmed = alarmed or step.alarm
        alarms += int(alarmed)
    assert alarms >= 10


def _ones(x_cal: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.ones(x_cal.shape[0]), np.ones(x_test.shape[0])


def test_benign_covariate_shift_adapts_without_a_label_alarm() -> None:
    """Invariant labels under a covariate shift: the X-CTM adapts, Y usually stays quiet."""
    adapted = 0
    quiet = 0
    for seed in range(20):
        rng = np.random.default_rng(100 + seed)
        monitor = WATCHMonitor(
            alpha=0.05, adapt_threshold=1.05, min_calibration=20, weight_fn=_ones
        )
        for _ in range(50):
            monitor.update(float(rng.normal()), float(rng.normal()))
        last = None
        for _ in range(60):
            last = monitor.update(float(rng.normal()), float(rng.normal(loc=4.0)))
        assert last is not None
        adapted += int(last.adapted)
        quiet += int(not last.alarm)
    assert adapted >= 18
    assert quiet >= 12


def test_oracle_weights_alarm_less_often_than_uniform_weights() -> None:
    """Frozen-bag oracle weights (shift 1) keep p-values uniform; uniform weights do not."""
    shift = 1.0
    level = 20.0
    oracle_alarms = 0
    uniform_alarms = 0
    weighted_ps: list[float] = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        x_cal = rng.normal(0.0, 1.0, 200)
        y_cal = x_cal + rng.normal(0.0, 0.3, 200)
        x_test = rng.normal(shift, 1.0, 80)
        y_test = x_test + rng.normal(0.0, 0.3, 80)
        w_cal = np.exp(np.clip(shift * x_cal - 0.5 * shift**2, -20.0, 20.0))
        plain = CompositeJumper()
        weighted = CompositeJumper()
        plain_hit = False
        weighted_hit = False
        for x_value, y_value in zip(x_test, y_test, strict=True):
            scores = np.concatenate([y_cal, [y_value]])
            p_plain = weighted_conformal_pvalue(scores)
            w_test = np.exp(np.clip(shift * x_value - 0.5 * shift**2, -20.0, 20.0))
            p_weighted = weighted_conformal_pvalue(scores, np.concatenate([w_cal, [w_test]]))
            weighted_ps.append(p_weighted)
            plain_hit = plain_hit or plain.update(p_plain) >= level
            weighted_hit = weighted_hit or weighted.update(p_weighted) >= level
        uniform_alarms += int(plain_hit)
        oracle_alarms += int(weighted_hit)
    assert uniform_alarms >= 18
    assert oracle_alarms <= 6
    assert oracle_alarms < uniform_alarms
    assert 0.4 <= float(np.mean(weighted_ps)) <= 0.6


def test_alarm_is_sticky() -> None:
    rng = np.random.default_rng(5)
    monitor = WATCHMonitor(alpha=0.1, adapt_threshold=100.0, min_calibration=500)
    for _ in range(15):
        monitor.update(float(rng.normal()), float(rng.normal()))
    hit = False
    for k in range(1, 40):
        step = monitor.update(float(100 + k), float(rng.normal()))
        hit = hit or step.alarm
    assert hit
    quiet = monitor.update(0.0, float(rng.normal()))
    assert quiet.alarm


def test_fail_closed_edges() -> None:
    with pytest.raises(ValueError):
        WATCHMonitor(alpha=0.0)
    with pytest.raises(ValueError):
        WATCHMonitor(adapt_threshold=1.0)
    with pytest.raises(ValueError):
        SimpleJumper(jump_rate=0.0)
    with pytest.raises(ValueError):
        weighted_conformal_pvalue(np.array([1.0, np.nan]))
    with pytest.raises(ValueError, match="tie_breaker"):
        weighted_conformal_pvalue(np.ones(2), tie_breaker=float("nan"))
    with pytest.raises(ValueError):
        nearest_neighbor_scores(np.array([1.0]))
    monitor = WATCHMonitor()
    with pytest.raises(ValueError):
        monitor.update(np.nan, 0.0)
    monitor.update(0.0, np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="dimension"):
        monitor.update(0.0, 1.0)
    with pytest.raises(ValueError):
        shiryaev_roberts(np.array([1.0]))

    return_bad_shape = False

    def bad(x_cal: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if return_bad_shape:
            return np.ones(x_cal.shape[0] + 1), np.ones(x_test.shape[0])
        return np.ones(x_cal.shape[0]), np.ones(x_test.shape[0])

    broken = WATCHMonitor(alpha=0.2, adapt_threshold=1.05, min_calibration=2, weight_fn=bad)
    for k in range(40):
        broken.update(0.0, float(2**k))
        if broken.adapted:
            break
    assert broken.adapted
    return_bad_shape = True
    with pytest.raises(ValueError, match="wrong shape"):
        broken.update(0.0, 5.0)
    with pytest.raises(RuntimeError, match="cannot continue"):
        broken.update(0.0, 0.0)
