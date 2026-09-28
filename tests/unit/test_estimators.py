"""Reference estimator unit tests (WAVE2.md §4.3). Synthetic data only."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.proof.estimators import (
    ALLOWLISTED_ESTIMATORS,
    ESTIMATOR_REGISTRY,
    EwmaSignal,
    LinearRegressionNumpy,
)


def test_allowlist_is_static_frozenset() -> None:
    assert isinstance(ALLOWLISTED_ESTIMATORS, frozenset)
    assert {"linear_regression_np", "ewma_signal"} == ALLOWLISTED_ESTIMATORS
    assert set(ESTIMATOR_REGISTRY) == set(ALLOWLISTED_ESTIMATORS)


# -- LinearRegressionNumpy ----------------------------------------------------


def test_linear_regression_recovers_known_plane() -> None:
    rng = np.arange(1.0, 9.0)
    X = np.column_stack([rng, rng**2 / 10.0])
    y = 1.5 + 2.0 * X[:, 0] - 0.5 * X[:, 1]
    est = LinearRegressionNumpy()
    est.fit(X, y)
    prediction = est.predict(np.array([10.0, 10.0]))
    assert abs(prediction - (1.5 + 20.0 - 5.0)) < 1e-4


def test_linear_regression_unfitted_predicts_zero() -> None:
    assert LinearRegressionNumpy().predict(np.array([1.0, 2.0])) == 0.0


def test_linear_regression_fit_validation() -> None:
    est = LinearRegressionNumpy()
    with pytest.raises(ValueError):
        est.fit(np.zeros((0, 2)), np.zeros(0))
    with pytest.raises(ValueError):
        est.fit(np.zeros((3, 2)), np.zeros(2))


def test_linear_regression_determinism_state_bytes() -> None:
    X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 3.0], [3.0, 3.0]])
    y = np.array([0.5, 1.5, 2.5, 3.5])
    first = LinearRegressionNumpy()
    second = LinearRegressionNumpy()
    first.fit(X, y)
    second.fit(X, y)
    assert first.state_bytes() == second.state_bytes()
    # Unfitted and fitted states differ; refit on same data is byte-identical.
    assert LinearRegressionNumpy().state_bytes() != first.state_bytes()
    first.fit(X, y)
    assert first.state_bytes() == second.state_bytes()


def test_linear_regression_alpha_changes_state() -> None:
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.0, 3.0])
    a = LinearRegressionNumpy(alpha=1e-8)
    b = LinearRegressionNumpy(alpha=1e-2)
    a.fit(X, y)
    b.fit(X, y)
    assert a.state_bytes() != b.state_bytes()


def test_linear_regression_state_vector_layout() -> None:
    est = LinearRegressionNumpy()
    assert est.state_vector().tolist() == [0.0, 1e-8]
    est.fit(np.array([[1.0], [2.0]]), np.array([0.0, 1.0]))
    vector = est.state_vector()
    assert vector[0] == 1.0  # fitted flag
    assert vector.shape == (4,)  # flag + alpha + intercept + 1 coef


# -- EwmaSignal ----------------------------------------------------------------


def test_ewma_matches_hand_computation() -> None:
    est = EwmaSignal(span=3.0)  # alpha = 0.5
    est.fit(np.zeros((3, 1)), np.array([1.0, 0.0, 2.0]))
    # 0.5*1+0.5*0 = 0.5 ; 0.5*0+0.5*0.5 = 0.25 ; 0.5*2+0.5*0.25 = 1.125
    assert est.predict(np.zeros(1)) == pytest.approx(1.125)


def test_ewma_cold_start_predicts_zero() -> None:
    assert EwmaSignal().predict(np.zeros(2)) == 0.0


def test_ewma_rejects_bad_span_and_empty_fit() -> None:
    with pytest.raises(ValueError):
        EwmaSignal(span=0.5)
    with pytest.raises(ValueError):
        EwmaSignal().fit(np.zeros((0, 1)), np.zeros(0))


def test_ewma_determinism_state_bytes() -> None:
    y = np.array([0.1, -0.2, 0.3, 0.05])
    first = EwmaSignal(span=2.0)
    second = EwmaSignal(span=2.0)
    first.fit(np.zeros((4, 1)), y)
    second.fit(np.zeros((4, 1)), y)
    assert first.state_bytes() == second.state_bytes()
    assert first.state_vector().tolist() == pytest.approx([4.0, 2.0, first.predict(np.zeros(1))])
    other = EwmaSignal(span=5.0)
    other.fit(np.zeros((4, 1)), y)
    assert other.state_bytes() != first.state_bytes()


def test_state_bytes_are_npy_payloads() -> None:
    est = EwmaSignal()
    est.fit(np.zeros((1, 1)), np.array([1.0]))
    payload = est.state_bytes()
    assert payload.startswith(b"\x93NUMPY")
    loaded = np.load(__import__("io").BytesIO(payload))
    np.testing.assert_array_equal(loaded, est.state_vector())
