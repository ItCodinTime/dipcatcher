"""Tests for quant_fund.models.watch (WATCH weighted-conformal martingales).

Seeded throughout; the heavy tests are vectorized inside ``run_watch`` and
finish in a few seconds each. Monte Carlo tolerances were calibrated against
the fixed seeds below (see per-test notes).
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.watch import (
    DEFAULT_GAMMA_GRID,
    WatchMartingale,
    WatchResult,
    conformity_from_quantiles,
    run_watch,
)
from quant_fund.models.weighted_conformal import likelihood_ratio_weights

ALPHA = 0.05
T_LONG = 2000
N_REP_FAR = 200  # spec: >= 200 replicates for the false-alarm test


def _null_scores(rng: np.random.Generator, n: int) -> np.ndarray:
    """IID |N(0, 1)| nonconformity scores: exchangeable, continuous."""
    return np.abs(rng.standard_normal(n))


def _shifted_scores(rng: np.random.Generator, n: int, t0: int, scale: float = 2.0) -> np.ndarray:
    """Scale shift 1 -> ``scale`` mid-stream at step ``t0`` (score shift)."""
    s = np.abs(rng.standard_normal(n))
    s[t0:] = np.abs(rng.standard_normal(n - t0)) * scale
    return s


def test_false_alarm_rate_under_exchangeability() -> None:
    """(a) IID scores: empirical FPR by T=2000 <= alpha + 0.03, betting='mix'.

    Ville gives FPR <= alpha = 0.05 exactly; the Monte Carlo margin is 0.03.
    With the seeds below the observed rate is ~0.015.
    """
    alarms = 0
    for r in range(N_REP_FAR):
        rng = np.random.default_rng(10_000 + r)
        res = run_watch(_null_scores(rng, T_LONG), alpha=ALPHA, random_state=r)
        alarms += int(res.alarm_time is not None)
    far = alarms / N_REP_FAR
    assert far <= ALPHA + 0.03


def test_score_shift_detection_power_and_delay() -> None:
    """(b) Scale shift 1 -> 2 at t=1000: detection probability >= 0.85."""
    n_rep = 100
    detections = 0
    delays: list[int] = []
    for r in range(n_rep):
        rng = np.random.default_rng(20_000 + r)
        res = run_watch(_shifted_scores(rng, T_LONG, 1000), alpha=ALPHA, random_state=r)
        if res.alarm_time is not None:
            detections += 1
            delays.append(res.alarm_time - 1000)
    det_prob = detections / n_rep
    median_delay = float(np.median(delays)) if delays else float("nan")
    print(f"\ndetection probability = {det_prob:.3f}, median alarm delay = {median_delay:.0f}")
    assert det_prob >= 0.85
    assert median_delay < T_LONG - 1000  # alarm well before the horizon ends


def test_weighted_ramp_detects_break_faster() -> None:
    """(c) Monotone ramp weight_fn vs uniform on a persistent score shift.

    Direction matters, matching Prinster et al. 2025 (arXiv:2505.04608):

    - ramp DOWN (weight ~ t - i + 1, emphasizes the calibration past): the
      post-shift score looks extreme against the anchored pool, so the
      weighted monitor alarms FASTER than uniform. This is what the test
      asserts.
    - ramp UP (weight ~ i, follows the shifted regime): by Theorem 3.3 the
      weighted-conformal p-values are again uniform under the weighted null,
      i.e. the monitor ADAPTS and alarms slower than uniform. Numbers are
      reported (not asserted) — adaptation is the paper's design goal.
    """
    n_rep, horizon, t0 = 60, 800, 300

    def ramp_up(i: np.ndarray, t: int) -> np.ndarray:
        return np.asarray(i, dtype=float)

    def ramp_down(i: np.ndarray, t: int) -> np.ndarray:
        return (t - np.asarray(i, dtype=float) + 1.0).reshape(-1)

    def median_delay(weight_fn: object) -> tuple[float, float]:
        detections = 0
        delays: list[int] = []
        for r in range(n_rep):
            rng = np.random.default_rng(50_000 + r)
            scores = _shifted_scores(rng, horizon, t0)
            res = run_watch(scores, alpha=ALPHA, weight_fn=weight_fn, random_state=r)
            if res.alarm_time is not None:
                detections += 1
                delays.append(res.alarm_time - t0)
        delay = float(np.median(delays)) if delays else float("inf")
        return detections / n_rep, delay

    det_uni, delay_uni = median_delay(None)
    det_up, delay_up = median_delay(ramp_up)
    det_down, delay_down = median_delay(ramp_down)
    print(
        f"\nuniform: det={det_uni:.2f} median_delay={delay_uni:.0f} | "
        f"ramp_up (adapts): det={det_up:.2f} median_delay={delay_up:.0f} | "
        f"ramp_down (anchored): det={det_down:.2f} median_delay={delay_down:.0f}"
    )
    assert det_down >= det_uni
    assert delay_down < delay_uni
    assert delay_up > delay_uni  # adaptation: following the shift slows the alarm


def test_tibshirani_weights_from_covariate_stream() -> None:
    """(c, cont.) likelihood_ratio_weights reused as a covariate-shift weight_fn.

    Frozen pre-shift calibration reference vs post-shift window (the fixed
    calibration set of Prinster et al. 2025, Sec. 4): weights emphasizing the
    pre-shift regime detect the break faster than uniform. Runtime is kept
    small (n_rep=30, horizon=600) because the weighted path is a per-step loop.
    """
    n_rep, horizon, t0 = 30, 600, 250

    def covariate_weight_fn(x: np.ndarray) -> object:
        pre = x[:t0]
        post = x[t0:]
        # anchored_i = 1 / (p_post(x_i) / p_pre(x_i)): downweights points the
        # post-shift regime visits, anchoring the pool to the pre-shift past
        # (frozen calibration reference, Prinster et al. 2025, Sec. 4). Rows
        # are covered by two likelihood_ratio_weights calls (weights are
        # evaluated on their first argument).
        anchored = np.empty(x.size, dtype=float)
        anchored[:t0] = 1.0 / likelihood_ratio_weights(pre, post, bins=8)
        anchored[t0:] = likelihood_ratio_weights(post, pre, bins=8)

        def fn(i: np.ndarray, t: int) -> np.ndarray:
            return anchored[np.asarray(i, dtype=np.int64) - 1].reshape(-1)

        return fn

    delays_w: list[int] = []
    delays_u: list[int] = []
    for r in range(n_rep):
        rng = np.random.default_rng(60_000 + r)
        x = rng.standard_normal(horizon)
        x[t0:] += 2.0
        scores = _shifted_scores(rng, horizon, t0)
        res_w = run_watch(scores, alpha=ALPHA, weight_fn=covariate_weight_fn(x), random_state=r)
        res_u = run_watch(scores, alpha=ALPHA, random_state=r)
        if res_w.alarm_time is not None:
            delays_w.append(res_w.alarm_time - t0)
        if res_u.alarm_time is not None:
            delays_u.append(res_u.alarm_time - t0)
    delay_w = float(np.median(delays_w)) if delays_w else float("inf")
    delay_u = float(np.median(delays_u)) if delays_u else float("inf")
    print(f"\nTibshirani-weighted median delay = {delay_w:.0f} vs uniform = {delay_u:.0f}")
    assert delays_w  # weighted variant detects
    assert delay_w < delay_u


def test_martingale_is_nonnegative_supermartingale() -> None:
    """(d) Under the null, sample mean of M_t at fixed t <= 1 + tolerance.

    E[M_t] = 1 exactly for the power martingale with uniform p-values
    (E[gamma p^{gamma-1}] = 1). The distribution is extremely right-skewed
    (most paths decay geometrically, rare paths explode and carry the mean),
    so with finite replicates the sample mean UNDERESTIMATES 1 and an upper
    one-sided check against 1 + tolerance is the meaningful direction — it
    fails if a bug inflated the wealth in expectation. Tolerance 0.5, t = 50.
    """
    n_rep, t_fixed = 400, 50
    finals = np.empty(n_rep)
    for r in range(n_rep):
        rng = np.random.default_rng(30_000 + r)
        scores = _null_scores(rng, t_fixed)
        res = run_watch(
            scores,
            alpha=ALPHA,
            betting="power",
            power_gamma=0.5,
            burn_in=5,
            random_state=r,
        )
        assert np.all(res.martingale_path >= 0.0)  # nonnegative
        assert res.martingale_path[0] == 1.0  # M_0 = 1
        finals[r] = res.martingale_path[-1]
    sample_mean = float(finals.mean())
    print(
        f"\nsample mean of M_50 = {sample_mean:.4f} (E = 1, q95 = {np.quantile(finals, 0.95):.3f})"
    )
    assert sample_mean <= 1.0 + 0.5


def test_mix_dominates_badly_tuned_power() -> None:
    """(e) Mixture over the gamma grid beats a badly tuned single gamma.

    Under the 1 -> 2 score shift a power martingale at gamma = 0.9 bets
    almost nothing (factor 0.9 p^{-0.1} ~ 1 for all p), while the mixture
    keeps small-gamma components alive. Loose, documented tolerance: the
    mixture's terminal log-wealth exceeds the single-gamma one in >= 90% of
    replicates (observed: 100%).
    """
    n_rep = 100
    wins = 0
    log_mix = np.empty(n_rep)
    log_single = np.empty(n_rep)
    for r in range(n_rep):
        rng = np.random.default_rng(40_000 + r)
        scores = _shifted_scores(rng, T_LONG, 1000)
        mix = run_watch(scores, alpha=ALPHA, betting="mix", random_state=r)
        single = run_watch(scores, alpha=ALPHA, betting="power", power_gamma=0.9, random_state=r)
        lm = np.log(max(mix.martingale_path[-1], 1e-300))
        ls = np.log(max(single.martingale_path[-1], 1e-300))
        log_mix[r], log_single[r] = lm, ls
        wins += int(lm > ls)
    print(
        f"\nmix wins {wins}/{n_rep}, mean log M mix = {log_mix.mean():.1f} vs single = {log_single.mean():.1f}"
    )
    assert wins / n_rep >= 0.90
    assert log_mix.mean() > log_single.mean()


def test_fail_closed_edges() -> None:
    """(f) Fail-closed: invalid alpha / burn_in / betting / gamma / scores."""
    scores = np.abs(np.random.default_rng(0).standard_normal(50))
    with pytest.raises(ValueError):
        WatchMartingale(alpha=0.0)
    with pytest.raises(ValueError):
        WatchMartingale(alpha=1.0)
    with pytest.raises(ValueError):
        WatchMartingale(alpha=-0.1)
    with pytest.raises(ValueError):
        WatchMartingale(burn_in=4)
    with pytest.raises(ValueError):
        WatchMartingale(betting="grapa")
    with pytest.raises(ValueError):
        WatchMartingale(power_gamma=1.5)
    with pytest.raises(ValueError):
        WatchMartingale(gamma_grid=np.array([0.5]))
    with pytest.raises(ValueError):
        run_watch(np.array([]), alpha=ALPHA)
    with pytest.raises(ValueError):
        run_watch(np.array([1.0, np.nan]), alpha=ALPHA)
    with pytest.raises(ValueError):
        run_watch(np.array([1.0, np.inf]), alpha=ALPHA)
    with pytest.raises(ValueError):
        run_watch(scores, alpha=ALPHA, burn_in=0)
    watch = WatchMartingale(alpha=ALPHA, random_state=0)
    with pytest.raises(ValueError):
        watch.update(float("nan"))
    with pytest.raises(ValueError):
        watch.update(float("inf"))
    # A weight_fn returning non-positive or misaligned weights fails closed.
    bad_watch = WatchMartingale(alpha=ALPHA, weight_fn=lambda i, t: np.zeros(t), random_state=0)
    with pytest.raises(ValueError):
        bad_watch.update(1.0)


def test_conformity_from_quantiles_score() -> None:
    """Exact score: |y - q_0.5| / (q_0.75 - q_0.25), linear interpolation."""
    levels = np.array([0.25, 0.5, 0.75])
    grid = np.array(
        [
            [0.0, 1.0, 2.0],  # med=1, iqr=2
            [10.0, 11.0, 13.0],  # med=11, iqr=3
        ]
    )
    y = np.array([2.0, 8.0])
    s = conformity_from_quantiles(y, grid, levels)
    assert s[0] == pytest.approx(abs(2.0 - 1.0) / 2.0)
    assert s[1] == pytest.approx(abs(8.0 - 11.0) / 3.0)
    with pytest.raises(ValueError):
        conformity_from_quantiles(np.array([1.0]), grid, levels)
    with pytest.raises(ValueError):
        conformity_from_quantiles(y, grid[:, :2], levels)
    with pytest.raises(ValueError):
        conformity_from_quantiles(y, grid, np.array([0.5, 0.25, 0.75]))  # not increasing
    with pytest.raises(ValueError):
        conformity_from_quantiles(y, grid, np.array([0.0, 0.5, 1.0]))  # endpoints not in (0, 1)


def test_conformity_feeds_run_watch_end_to_end() -> None:
    """Median/IQR scores from a quantile grid detect a median break."""
    rng = np.random.default_rng(7)
    n, t0 = 600, 300
    levels = np.array([0.25, 0.5, 0.75])
    med = np.zeros(n)
    med[t0:] = 1.5  # concept shift: forecast median jumps, y stays put
    grid = np.column_stack([med - 1.0, med, med + 1.0])  # q25, q50, q75
    y = rng.standard_normal(n)
    scores = conformity_from_quantiles(y, grid, levels)
    res = run_watch(scores, alpha=ALPHA, random_state=3)
    assert res.alarm_time is not None
    assert np.all((res.p_values > 0.0) & (res.p_values <= 1.0))


def test_seeded_determinism_and_stateful_api() -> None:
    """Same seed -> identical paths; stateful class matches stateless run."""
    rng = np.random.default_rng(11)
    scores = _null_scores(rng, 200)
    res_a = run_watch(scores, alpha=ALPHA, random_state=5)
    res_b = run_watch(scores, alpha=ALPHA, random_state=5)
    np.testing.assert_array_equal(res_a.p_values, res_b.p_values)
    np.testing.assert_array_equal(res_a.martingale_path, res_b.martingale_path)
    assert isinstance(res_a, WatchResult)

    watch_a = WatchMartingale(alpha=ALPHA, random_state=5)
    watch_b = WatchMartingale(alpha=ALPHA, random_state=5)
    for s in scores:
        m_a = watch_a.update(s)
        m_b = watch_b.update(s)
    assert m_a == m_b
    np.testing.assert_array_equal(watch_a.p_value_history, watch_b.p_value_history)
    np.testing.assert_allclose(watch_a.martingale_path, res_a.martingale_path)
    assert watch_a.step == scores.size
    assert watch_a.alarmed == (res_a.alarm_time is not None)
    # M_0 = 1 at index 0; burn-in keeps the martingale flat.
    assert watch_a.martingale_path[0] == 1.0
    np.testing.assert_array_equal(watch_a.martingale_path[1:30], np.ones(29))


def test_default_gamma_grid_valid() -> None:
    assert DEFAULT_GAMMA_GRID.size == 19
    assert np.all(DEFAULT_GAMMA_GRID > 0.0)
    assert np.all(DEFAULT_GAMMA_GRID < 1.0)
