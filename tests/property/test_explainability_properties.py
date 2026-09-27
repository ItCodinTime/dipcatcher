"""Hypothesis properties for explainability math (SYNTHETIC — correctness only).

- JS divergence is bounded in [0, ln 2], symmetric, and identity-zero.
- ε-smoothed normalized importance shares always sum to 1.
- 1-D partial dependence of an additive linear model is monotone in the
  grid direction of the coefficient's sign (exact, not statistical).
"""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_fund.research.explainability.drift import _normalized_shares, js_divergence
from quant_fund.research.explainability.partial_dependence import partial_dependence_1d

_vector = st.lists(
    st.floats(-1e4, 1e4, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=16,
)


def _arr(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=float)


@given(p=_vector, q=_vector)
@settings(max_examples=80, deadline=None)
def test_js_divergence_bounds_symmetry(p: list[float], q: list[float]) -> None:
    if len(p) != len(q):
        return
    pa, qa = _arr(p), _arr(q)
    value = js_divergence(pa, qa)
    assert math.isfinite(value)
    assert 0.0 <= value <= math.log(2.0) + 1e-9
    assert js_divergence(pa, qa) == js_divergence(qa, pa)


@given(p=_vector)
@settings(max_examples=80, deadline=None)
def test_js_divergence_identity_zero(p: list[float]) -> None:
    assert js_divergence(_arr(p), _arr(p)) <= 1e-9


@given(p=_vector)
@settings(max_examples=80, deadline=None)
def test_normalized_shares_sum_to_one(p: list[float]) -> None:
    shares = _normalized_shares(_arr(p))
    assert np.all(shares >= 0.0)
    assert float(shares.sum()) <= 1.0 + 1e-9
    assert float(shares.sum()) >= 1.0 - 1e-9


@given(
    coef=st.floats(0.01, 1e3, allow_nan=False, allow_infinity=False),
    n=st.integers(8, 60),
    seed=st.integers(0, 10_000),
)
@settings(max_examples=40, deadline=None)
def test_pd_monotone_for_positive_linear_coefficient(coef: float, n: int, seed: int) -> None:
    """PD_j of x@w is w_j * v + const — non-decreasing grid ⇒ non-decreasing curve."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    w = np.asarray([0.0, coef, 0.0])

    def predict(block: np.ndarray) -> np.ndarray:
        return np.asarray(block, dtype=float) @ w

    curve = partial_dependence_1d(predict, x, 1, "w", grid_points=6)
    values = np.asarray(curve.mean_curve, dtype=float)
    assert np.all(np.diff(values) >= -1e-9)


@given(
    coef=st.floats(-1e3, -0.01, allow_nan=False, allow_infinity=False),
    n=st.integers(8, 60),
    seed=st.integers(0, 10_000),
)
@settings(max_examples=40, deadline=None)
def test_pd_antitone_for_negative_linear_coefficient(coef: float, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    w = np.asarray([0.0, coef, 0.0])

    def predict(block: np.ndarray) -> np.ndarray:
        return np.asarray(block, dtype=float) @ w

    curve = partial_dependence_1d(predict, x, 1, "w", grid_points=6)
    values = np.asarray(curve.mean_curve, dtype=float)
    assert np.all(np.diff(values) <= 1e-9)
