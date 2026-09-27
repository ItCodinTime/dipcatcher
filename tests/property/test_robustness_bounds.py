"""Property checks for the closed-form robustness bounds."""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_fund.robustness.analytic import (
    gelbrich_worst_case_ratio,
    linear_l2_radius,
    linear_positive_probability,
    smoothing_radius,
    wasserstein_worst_case_mean,
)


@given(
    mean=st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False),
    radius=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=40)
def test_worst_case_mean_is_mean_minus_radius(mean: float, radius: float) -> None:
    assert wasserstein_worst_case_mean(mean, radius) == mean - radius


@given(
    weights=st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=4,
    ),
    sample=st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=4,
    ),
    sigma=st.floats(min_value=0.05, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=30)
def test_population_smoothing_radius_matches_the_margin(
    weights: list[float], sample: list[float], sigma: float
) -> None:
    dimension = min(len(weights), len(sample))
    w = np.asarray(weights[:dimension], dtype=float)
    x = np.asarray(sample[:dimension], dtype=float)
    if float(np.linalg.norm(w)) == 0.0:
        return
    distance = linear_l2_radius(w, 0.0, x)
    if math.isinf(distance) or distance < 1e-8:
        return
    positive = linear_positive_probability(w, 0.0, x, sigma)
    top = max(positive, 1.0 - positive)
    if top <= 0.5:
        return
    # Phi saturates at 1, where the certificate radius is infinite.
    if top >= 1.0 - 1e-12:
        return
    certified = smoothing_radius(top, 1.0 - top, sigma)
    assert certified == pytest_approx(distance)


def pytest_approx(value: float) -> object:
    import pytest

    return pytest.approx(value, rel=1e-8, abs=1e-8)


@given(
    mean=st.floats(min_value=0.2, max_value=2.0, allow_nan=False, allow_infinity=False),
    scale=st.floats(min_value=0.5, max_value=2.0, allow_nan=False, allow_infinity=False),
    fraction=st.floats(min_value=0.0, max_value=0.8, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=25)
def test_gelbrich_ratio_matches_the_disk_grid(mean: float, scale: float, fraction: float) -> None:
    radius = fraction * scale
    if math.isclose(radius, scale):
        return
    value, kind = gelbrich_worst_case_ratio(mean, scale, radius)
    assert kind != "unbounded_below"
    assert value is not None
    thetas = np.linspace(0.0, 2.0 * math.pi, 4000, endpoint=False)
    delta = radius * np.cos(thetas)
    eta = radius * np.sin(thetas)
    scale_prime = scale + eta
    mean_prime = mean + delta
    ok = scale_prime > 1e-6
    grid = float(np.min(mean_prime[ok] / scale_prime[ok]))
    assert value == pytest_approx_loose(grid)


def pytest_approx_loose(value: float) -> object:
    import pytest

    return pytest.approx(value, rel=1e-3, abs=1e-3)
