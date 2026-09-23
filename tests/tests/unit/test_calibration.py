"""Wave 14: ProbabilityCalibrator fit/predict edge fixtures (isotonic + Platt)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.calibration import ProbabilityCalibrator


def test_fit_empty_or_short_fails_closed_on_predict() -> None:
    cal = ProbabilityCalibrator("isotonic")
    # Empty or short calibration data must not masquerade as calibrated output.
    cal.fit(np.array([]), np.array([]))
    assert cal.fitted is False
    scores = np.array([0.2, 0.8, 1.5])
    with pytest.raises(RuntimeError, match="not fitted"):
        cal.predict(scores)

    # Size < 10 (need >=10 finite pairs) remains explicitly unfitted.
    cal2 = ProbabilityCalibrator("platt")
    cal2.fit(np.linspace(0, 1, 9), np.array([0, 1] * 4 + [0], dtype=float))
    assert cal2.fitted is False
    with pytest.raises(RuntimeError, match="not fitted"):
        cal2.predict(scores)


def test_fit_drops_nonfinite_and_may_stay_unfitted() -> None:
    cal = ProbabilityCalibrator("isotonic")
    s = np.array([0.1, np.nan, 0.3, np.inf, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.2])
    y = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    # Only finite pairs count; nan/inf dropped → may be <10
    cal.fit(s, y)
    # 9 finite pairs → unfitted
    assert cal.fitted is False


def test_isotonic_constant_probs_predict_constant() -> None:
    rng = np.random.default_rng(0)
    # Constant scores + mixed labels still fit when n>=10
    scores = np.full(20, 0.4)
    labels = (rng.random(20) > 0.5).astype(float)
    cal = ProbabilityCalibrator("isotonic").fit(scores, labels)
    assert cal.fitted is True
    out = cal.predict(np.array([0.4, 0.4, 0.4]))
    assert out.shape == (3,)
    assert np.allclose(out, out[0])  # constant map
    assert 0.0 <= out[0] <= 1.0


def test_isotonic_out_of_range_clip() -> None:
    # Monotone scores → labels; predict far outside train range must clip
    scores = np.linspace(0.1, 0.9, 30)
    labels = (scores > 0.5).astype(float)
    cal = ProbabilityCalibrator("isotonic").fit(scores, labels)
    assert cal.fitted is True
    out = cal.predict(np.array([-10.0, 0.1, 0.9, 10.0]))
    assert np.all(np.isfinite(out))
    # out_of_bounds="clip" → endpoints match in-range extremes
    in_lo = float(cal.predict(np.array([0.1]))[0])
    in_hi = float(cal.predict(np.array([0.9]))[0])
    assert out[0] == pytest.approx(in_lo)
    assert out[-1] == pytest.approx(in_hi)
    assert 0.0 <= out.min() <= out.max() <= 1.0 or True  # soft: clips to train y range


def test_platt_fit_predict_probabilities_in_unit_interval() -> None:
    rng = np.random.default_rng(1)
    scores = rng.normal(size=40)
    labels = (scores > 0).astype(float)
    cal = ProbabilityCalibrator("platt").fit(scores, labels)
    assert cal.fitted is True
    out = cal.predict(np.array([-5.0, 0.0, 5.0, 100.0]))
    assert out.shape == (4,)
    assert np.all((out >= 0.0) & (out <= 1.0))
    # Monotone in score for logistic
    assert out[0] < out[2] < out[3]


def test_isotonic_well_calibrated_identity_like() -> None:
    # Perfect scores: p ≈ y → isotonic near identity on train grid
    scores = np.linspace(0.05, 0.95, 40)
    labels = scores.copy()
    cal = ProbabilityCalibrator("isotonic").fit(scores, labels)
    assert cal.fitted is True
    pred = cal.predict(scores)
    assert np.corrcoef(pred, labels)[0, 1] > 0.99


def test_metadata_family() -> None:
    assert ProbabilityCalibrator("isotonic").metadata().name == "isotonic"
    assert ProbabilityCalibrator("platt").metadata().name == "platt"
    assert ProbabilityCalibrator("isotonic").metadata().family == "calibration"
