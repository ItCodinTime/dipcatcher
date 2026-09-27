"""Black-box and gradient attacks against toys with a known L2 radius."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.robustness.analytic import linear_l2_radius
from quant_fund.robustness.attacks import black_box_attack, cmaes_minimize, gradient_attack
from quant_fund.robustness.certify import _StrategyMarginBackend, certify
from quant_fund.robustness.gradients import (
    CallableBackend,
    get_gradient_backend,
    register_gradient_backend,
    unregister_gradient_backend,
)
from quant_fund.robustness.strategies import LinearMargin, OpaqueSign


def test_cmaes_locates_a_shifted_sphere() -> None:
    target = np.array([0.3, -0.2, 0.1])

    def objective(point: np.ndarray) -> float:
        return float(np.sum((point - target) ** 2))

    best, value = cmaes_minimize(objective, np.zeros(3), 0.4, generations=30, seed=1)
    assert value < 1e-3
    assert np.allclose(best, target, atol=2e-2)


def test_cmaes_attack_upper_bounds_the_linear_radius() -> None:
    weights = np.array([1.0, 0.0, 0.0, 0.25])
    sample = np.array([0.6, 0.2, -0.1, 0.0])
    strategy = LinearMargin(weights, name="attack_linear")
    analytic = linear_l2_radius(weights, 0.0, sample)
    found = black_box_attack(
        strategy,
        sample,
        method="cmaes",
        norm="l2",
        max_radius=analytic * 3,
        steps=8,
        generations=8,
        seed=2,
    )
    assert found["flipped"] is True
    assert found["proven"] is False
    assert found["status"] == "empirical"
    assert found["value"] >= analytic - 1e-8
    assert found["value"] <= analytic * 1.5


def test_tpe_attack_upper_bounds_the_linear_radius() -> None:
    weights = np.array([1.0, 0.0])
    sample = np.array([0.5, 0.25])
    strategy = LinearMargin(weights)
    analytic = strategy.analytic_l2_radius(sample)
    found = black_box_attack(
        strategy,
        sample,
        method="tpe",
        norm="l2",
        max_radius=analytic * 3,
        steps=6,
        trials=30,
        seed=4,
    )
    assert found["flipped"] is True
    assert found["value"] >= analytic - 1e-8
    assert found["value"] <= analytic * 1.75


def test_gradient_attack_matches_the_linear_radius() -> None:
    weights = np.array([0.5, -0.5, 0.25])
    sample = np.array([0.4, 0.1, 0.8])
    strategy = LinearMargin(weights)
    analytic = strategy.analytic_l2_radius(sample)
    found = gradient_attack(strategy, sample, _StrategyMarginBackend(strategy), steps=3)
    assert found["flipped"] is True
    assert found["value"] == pytest.approx(analytic, rel=1e-6, abs=1e-6)


def test_gradient_attack_is_unavailable_without_a_backend() -> None:
    strategy = OpaqueSign(2)
    found = gradient_attack(strategy, np.array([0.4, -0.2]), None)
    assert found["status"] == "unavailable"
    assert found["value"] is None
    assert found["proven"] is False


def test_registered_backend_supplies_the_gradient() -> None:
    weights = np.array([1.0, 0.0, 0.0])
    sample = np.array([0.7, 0.2, -0.3])
    analytic = linear_l2_radius(weights, 0.0, sample)

    def value_and_grad(path: np.ndarray) -> tuple[float, np.ndarray]:
        vector = np.asarray(path, dtype=float).reshape(-1)
        return float(vector[0]), weights.copy()

    register_gradient_backend(CallableBackend("jax_backtester", value_and_grad))
    try:
        assert "jax_backtester" in get_gradient_backend("jax_backtester").name
        card = certify(
            OpaqueSign(3, name="opaque"),
            sample,
            draws=16,
            alpha=0.05,
            seed=0,
            run_attack=False,
            gradient="jax_backtester",
            gradient_steps=2,
            outcome_mean=0.2,
            outcome_scale=1.0,
            reference="gaussian",
            evidence_class="SYNTHETIC",
        )
    finally:
        unregister_gradient_backend("jax_backtester")
    assert card["gradient_attack"]["flipped"] is True
    assert card["gradient_attack"]["value"] == pytest.approx(analytic, rel=1e-6, abs=1e-6)
    assert card["gradient_attack"]["proven"] is False
    assert math.isfinite(card["certified_radius"]["value"])


def test_finite_difference_backend_matches_the_linear_radius() -> None:
    weights = np.array([0.3, 0.4])
    sample = np.array([1.0, -0.25])
    strategy = LinearMargin(weights)
    analytic = strategy.analytic_l2_radius(sample)
    card = certify(
        strategy,
        sample,
        draws=8,
        seed=1,
        run_attack=False,
        gradient="finite_difference",
        gradient_steps=2,
        outcome_mean=0.1,
        outcome_scale=1.0,
        reference="empirical",
        evidence_class="SYNTHETIC",
    )
    assert card["gradient_attack"]["value"] == pytest.approx(analytic, abs=1e-4)
    assert card["distributional_robustness"]["worst_case_ratio"]["status"] == "outer_bound"
