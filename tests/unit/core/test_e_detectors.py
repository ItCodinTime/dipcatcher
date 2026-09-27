"""Tests for e_detectors: anytime-valid sequential changepoint detectors.

Construction under test (Shin, Ramdas & Rinaldo 2023, arXiv:2203.03532):
geometric-prior mixture Shiryaev-Roberts e-detectors; alarm at 1/alpha gives
time-uniform type-I error <= alpha by Ville. Test (a) checks the bound
empirically under the null; (b)-(e) check detection power and monotonicity;
(f) checks fail-closed edges. Every random draw uses a seeded
np.random.default_rng, so runs are deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.e_detectors import (
    DetectionResult,
    EDetectorBernoulli,
    EDetectorBounded,
    EDetectorGaussian,
    alarm_threshold,
    run_detector,
)

SEED = 20260927
ALPHA = 0.05
T_NULL = 2000
N_REPS_FA = 200  # >= 200 replicate streams per spec


def _gauss_shift_stream(rng: np.random.Generator, n0: int, n1: int, shift: float) -> np.ndarray:
    return np.concatenate([rng.standard_normal(n0), rng.standard_normal(n1) + shift])


def _false_alarm_rate(make_detector, sample, n_reps: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    alarms = 0
    for _ in range(n_reps):
        det = make_detector()
        res = run_detector(det, sample(rng), ALPHA)
        alarms += int(res.alarm_time is not None)
    return alarms / n_reps


# ---------------------------------------------------------------------------
# (a) time-uniform false-alarm control under the null
# ---------------------------------------------------------------------------


def test_gaussian_false_alarm_control_time_uniform() -> None:
    """P(any alarm by T=2000 | no change) <= alpha, empirical <= 0.07."""
    rate = _false_alarm_rate(
        lambda: EDetectorGaussian(1.0),
        lambda rng: rng.standard_normal(T_NULL),
        N_REPS_FA,
        SEED,
    )
    assert rate <= 0.07


# ---------------------------------------------------------------------------
# (b) detection of a 2-sigma mean shift
# ---------------------------------------------------------------------------


def test_gaussian_detects_two_sigma_shift_with_high_probability() -> None:
    """Shift 2*sigma at t=1000: detected >= 0.9 of replicates, well before T."""
    rng = np.random.default_rng(SEED + 1)
    detections: list[int] = []
    for _ in range(100):
        res = run_detector(EDetectorGaussian(1.0), _gauss_shift_stream(rng, 1000, 1000, 2.0), ALPHA)
        assert res.alarm_time is None or res.alarm_time >= 1000  # never before the change
        if res.alarm_time is not None:
            detections.append(res.alarm_time)
    assert len(detections) / 100 >= 0.9
    assert int(np.median(detections)) < 1500  # alarm well before T=2000


# ---------------------------------------------------------------------------
# (c) tradeoff: smaller shift -> later alarm (monotone, loose)
# ---------------------------------------------------------------------------


def test_gaussian_smaller_shift_later_alarm() -> None:
    """Median alarm time is strictly monotone in the shift size."""
    rng = np.random.default_rng(SEED + 2)
    medians = []
    for shift in (2.0, 1.0, 0.5):
        alarms = []
        for _ in range(150):
            res = run_detector(
                EDetectorGaussian(1.0), _gauss_shift_stream(rng, 1000, 1000, shift), ALPHA
            )
            if res.alarm_time is not None:
                alarms.append(res.alarm_time)
        assert len(alarms) / 150 >= 0.9
        medians.append(float(np.median(alarms)))
    assert medians[0] < medians[1] < medians[2]


# ---------------------------------------------------------------------------
# (d) Bernoulli VaR-violation detector
# ---------------------------------------------------------------------------


def test_bernoulli_alarms_at_double_violation_rate() -> None:
    """p0 = 0.05 (95% VaR): alarms at true rate 2*p0, stays quiet at p0."""
    p0 = 0.05
    rng = np.random.default_rng(SEED + 3)
    detections = 0
    for _ in range(100):
        stream = (rng.random(T_NULL) < 2 * p0).astype(float)
        res = run_detector(EDetectorBernoulli(p0), stream, ALPHA)
        detections += int(res.alarm_time is not None)
    assert detections / 100 >= 0.9

    fa_rate = _false_alarm_rate(
        lambda: EDetectorBernoulli(p0),
        lambda rng: (rng.random(T_NULL) < p0).astype(float),
        N_REPS_FA,
        SEED + 4,
    )
    assert fa_rate <= 0.07


# ---------------------------------------------------------------------------
# (e) bounded detector on a [0, 1]-scaled stream
# ---------------------------------------------------------------------------


def test_bounded_detector_on_unit_interval_stream() -> None:
    """beta(2, 2) (mean 0.5) -> beta(4, 2) (mean 2/3) at t=500: detects; quiet at null."""
    fa_rate = _false_alarm_rate(
        lambda: EDetectorBounded(0.0, 1.0),
        lambda rng: rng.beta(2, 2, T_NULL),
        150,
        SEED + 5,
    )
    assert fa_rate <= 0.07

    rng = np.random.default_rng(SEED + 6)
    detections = 0
    for _ in range(100):
        stream = np.concatenate([rng.beta(2, 2, 500), rng.beta(4, 2, 1500)])
        res = run_detector(EDetectorBounded(0.0, 1.0), stream, ALPHA)
        detections += int(res.alarm_time is not None and res.alarm_time >= 500)
    assert detections / 100 >= 0.9


def test_bounded_detector_respects_custom_m0() -> None:
    """Stream mean 0.3 against m0 = 0.5 stays quiet; m0 = 0.2 alarms."""
    rng = np.random.default_rng(SEED + 7)
    stream = rng.beta(2, 5, 2000)  # mean ~ 2/7 ~ 0.286
    res_quiet = run_detector(EDetectorBounded(0.0, 1.0, m0=0.5), stream, ALPHA)
    assert res_quiet.alarm_time is None
    res_alarm = run_detector(EDetectorBounded(0.0, 1.0, m0=0.2), stream, ALPHA)
    assert res_alarm.alarm_time is not None


# ---------------------------------------------------------------------------
# mechanics: update/alarmed/reset/run_detector
# ---------------------------------------------------------------------------


def test_detector_state_reset_and_alarmed_property() -> None:
    det = EDetectorGaussian(1.0)
    assert det.value == 1.0
    assert not det.alarmed
    rng = np.random.default_rng(SEED + 8)
    stream = _gauss_shift_stream(rng, 50, 500, 3.0)
    alarmed_at: int | None = None
    for i, x in enumerate(stream):
        m = det.update(float(x))
        assert m == det.value
        assert m >= 0.0
        if det.alarmed:
            alarmed_at = i
            break
    assert alarmed_at is not None
    assert det.alarmed
    assert det.n_seen == (alarmed_at + 1)
    det.reset()
    assert not det.alarmed
    assert det.n_seen == 0
    assert det.value == 1.0


def test_run_detector_result_shape_and_threshold() -> None:
    det = EDetectorBernoulli(0.05)
    res = run_detector(det, [0.0] * 20, ALPHA)
    assert isinstance(res, DetectionResult)
    assert res.alarm_time is None
    assert res.detector_path.shape == (20,)
    assert np.all(np.isfinite(res.detector_path))
    assert np.all(res.detector_path >= 0.0)
    # run_detector resets first: a second pass over a quiet stream starts fresh
    res2 = run_detector(det, [0.0] * 20, ALPHA)
    assert res2.detector_path.shape == (20,)


def test_alarm_threshold_is_inverse_alpha() -> None:
    assert alarm_threshold(0.05) == pytest.approx(20.0)
    assert alarm_threshold(0.10) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# (f) fail-closed edges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ctor",
    [
        lambda: EDetectorGaussian(0.0),
        lambda: EDetectorGaussian(-1.0),
        lambda: EDetectorGaussian(np.nan),
        lambda: EDetectorGaussian(1.0, alpha=0.0),
        lambda: EDetectorGaussian(1.0, alpha=1.0),
        lambda: EDetectorGaussian(1.0, lambda_grid=[]),
        lambda: EDetectorGaussian(1.0, lambda_grid=[np.inf]),
        lambda: EDetectorGaussian(1.0, gamma=1.0),
        lambda: EDetectorBounded(1.0, 1.0),
        lambda: EDetectorBounded(2.0, 1.0),
        lambda: EDetectorBounded(0.0, np.inf),
        lambda: EDetectorBounded(0.0, 1.0, m0=2.0),
        lambda: EDetectorBernoulli(0.0),
        lambda: EDetectorBernoulli(1.0),
        lambda: EDetectorBernoulli(-0.5),
        lambda: alarm_threshold(0.0),
        lambda: alarm_threshold(1.0),
    ],
)
def test_fail_closed_edges_raise(ctor) -> None:
    with pytest.raises(ValueError):
        ctor()


def test_update_rejects_non_finite_and_out_of_support() -> None:
    with pytest.raises(ValueError):
        EDetectorGaussian(1.0).update(np.nan)
    with pytest.raises(ValueError):
        EDetectorBounded(0.0, 1.0).update(1.5)
    with pytest.raises(ValueError):
        EDetectorBernoulli(0.1).update(np.inf)
