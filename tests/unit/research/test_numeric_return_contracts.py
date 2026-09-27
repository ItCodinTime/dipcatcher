"""Numeric boundary contracts: float64 outputs, formulas, and nonmutation."""

import numpy as np
import pytest

from quant_fund.execution.almgren_chriss import almgren_chriss_trajectory
from quant_fund.metrics.conformal import cqr_scores, onesided_scores
from quant_fund.metrics.evalues import e_process
from quant_fund.models.calibration import ProbabilityCalibrator
from quant_fund.models.covariance import factor_cov
from quant_fund.portfolio.interval_risk import downside
from quant_fund.utils.numeric import clip_positive


@pytest.mark.parametrize("shape", [(), (0,), (3,), (2, 3)])
def test_array_guards_preserve_shape_dtype_and_input(shape: tuple[int, ...]) -> None:
    values = np.full(shape, -2.0, dtype=np.float64)
    original = values.copy()
    for result, expected in [(clip_positive(values, floor=0.5), 0.5), (downside(values), 2.0)]:
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert result.shape == shape
        np.testing.assert_array_equal(result, np.full(shape, expected))
    np.testing.assert_array_equal(values, original)


def test_clip_positive_nonfinite_values_and_scalar_contract() -> None:
    values = np.array([np.nan, np.inf, -np.inf, -1.0, 3.0])
    original = values.copy()
    np.testing.assert_array_equal(clip_positive(values, floor=0.5), [0.5, 0.5, 0.5, 0.5, 3.0])
    np.testing.assert_array_equal(values, original)
    for value, expected in [(np.nan, 0.5), (-1.0, 0.5), (3.0, 3.0)]:
        result = clip_positive(value, floor=0.5)
        assert isinstance(result, float)
        assert result == expected


def test_conformal_score_formulas_and_normalization() -> None:
    y = np.array([-2.0, 0.0, 3.0])
    lo = np.array([-1.0, -1.0, -1.0])
    hi = np.array([1.0, 1.0, 1.0])
    scale = np.array([2.0, 0.0, 4.0])
    cases = [
        (cqr_scores(y, lo, hi), [1.0, -1.0, 2.0]),
        (cqr_scores(y, lo, hi, scale), [0.5, -1e12, 0.5]),
        (onesided_scores(y, hi), [-3.0, -1.0, 2.0]),
    ]
    for actual, expected in cases:
        assert actual.dtype == np.float64
        assert actual.shape == (3,)
        np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(y, [-2.0, 0.0, 3.0])
    np.testing.assert_array_equal(scale, [2.0, 0.0, 4.0])


def test_e_process_matches_hand_computed_likelihood_products() -> None:
    actual = e_process(np.array([0.0, 1.0, np.nan, 0.0]), alpha=0.1)
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, [8 / 9, 16 / 9, 128 / 81])


@pytest.mark.parametrize("risk_aversion", [0.0, 0.5])
def test_execution_trajectory_matches_closed_form(risk_aversion: float) -> None:
    actual = almgren_chriss_trajectory(
        100.0, 4, sigma=0.2, eta=0.1, gamma=0.0, risk_aversion=risk_aversion
    )
    if risk_aversion == 0.0:
        expected = np.array([100.0, 75.0, 50.0, 25.0, 0.0])
    else:
        kappa = np.sqrt(risk_aversion * 0.2**2 / 0.1)
        expected = 100.0 * np.sinh(kappa * np.array([4.0, 3.0, 2.0, 1.0, 0.0])) / np.sinh(kappa * 4)
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected)


def test_factor_covariance_matches_known_matrix() -> None:
    actual = factor_cov(np.array([[1.0], [2.0]]), np.array([[0.5]]), np.array([0.1, 0.2]))
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, [[0.6, 1.0], [1.0, 2.2]])


@pytest.mark.parametrize("method", ["isotonic", "platt"])
def test_calibration_has_float64_probability_contract(method: str) -> None:
    scores = np.linspace(-1.0, 1.0, 30)
    model = ProbabilityCalibrator(method).fit(scores, (scores > 0).astype(float))
    actual = model.predict(np.array([-0.8, 0.0, 0.8]))
    assert actual.dtype == np.float64
    assert actual.shape == (3,)
    assert np.all(np.isfinite(actual))
    assert np.all((actual >= 0.0) & (actual <= 1.0))
    assert np.all(np.diff(actual) >= 0.0)
