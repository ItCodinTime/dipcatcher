"""Build one strategy's robustness scorecard.

The scorecard keeps proven statements and empirical searches in different
fields. It is a research diagnostic. It does not route orders.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.robustness.attacks import black_box_attack, gradient_attack
from quant_fund.robustness.dro import distributional_bounds
from quant_fund.robustness.gradients import (
    FiniteDifferenceBackend,
    GradientBackend,
    get_gradient_backend,
)
from quant_fund.robustness.sensitivity import assemble_sensitivity
from quant_fund.robustness.smoothing import certify_decision

FloatArray = NDArray[np.float64]

_LIMITATIONS: tuple[str, ...] = (
    "Randomized smoothing certifies the smoothed decision inside an L2 ball in the noised coordinates, at the stated confidence. It does not certify other threat models.",
    "A black-box or gradient search that flips the decision is an empirical upper bound on the minimal perturbation. A search that does not flip is not a certificate.",
    "The worst-case mean inside a Wasserstein ball is exact for any reference law. The mean-to-scale ratio is the Gelbrich disk minimum: tight for a univariate Gaussian, and only a lower bound otherwise.",
    "Exact-on-path enumerations cover one finite array. They are not a general theorem about other paths.",
    "SYNTHETIC results are correctness checks. They are not market evidence and not a live-trading claim.",
)


class _StrategyMarginBackend:
    """Gradient backend that calls ``strategy.margin_and_grad``."""

    name = "strategy_margin"

    def __init__(self, strategy: Any) -> None:
        self._strategy = strategy

    def value_and_grad(self, path: FloatArray) -> tuple[float, FloatArray]:
        margin, gradient = self._strategy.margin_and_grad(path)
        return float(margin), np.asarray(gradient, dtype=float).reshape(-1)


def _unit_scale(scale: FloatArray | None, dimension: int) -> bool:
    if scale is None:
        return True
    vol = np.asarray(scale, dtype=float).reshape(-1)
    return vol.shape == (dimension,) and bool(np.all(np.isclose(vol, 1.0)))


def _resolve_backend(strategy: Any, gradient: str | None) -> GradientBackend | None:
    if gradient is None:
        if callable(getattr(strategy, "margin_and_grad", None)):
            return _StrategyMarginBackend(strategy)
        return None
    if gradient == "finite_difference":
        if not callable(getattr(strategy, "margin_and_grad", None)):
            return None

        def _margin(path: FloatArray) -> float:
            margin, _gradient = strategy.margin_and_grad(path)
            return float(margin)

        return FiniteDifferenceBackend(_margin)
    if gradient == "none":
        return None
    return get_gradient_backend(gradient)


def _with_proven(certificate: dict[str, Any]) -> dict[str, Any]:
    stamped = dict(certificate)
    stamped["proven"] = stamped.get("status") == "proven"
    return stamped


def _unavailable_attack(norm: str) -> dict[str, Any]:
    return {
        "value": None,
        "norm": norm,
        "unit": "volatility_scaled",
        "status": "unavailable",
        "method": "not_run",
        "flipped": False,
        "proven": False,
        "guarantee": "no_certificate",
    }


def certify(
    strategy: Any,
    sample: FloatArray,
    *,
    sigma: float = 0.5,
    draws: int = 128,
    alpha: float = 0.001,
    seed: int = 0,
    norm: str = "l2",
    attack: str = "cmaes",
    max_radius: float | None = None,
    attack_steps: int = 8,
    generations: int = 6,
    trials: int = 24,
    run_attack: bool = True,
    gradient: str | None = None,
    gradient_steps: int = 4,
    scale: FloatArray | None = None,
    outcome_mean: float | None = None,
    outcome_scale: float | None = None,
    outcomes: FloatArray | None = None,
    reference: str = "empirical",
    wasserstein_radius: float = 0.1,
    base_cost: float = 0.0,
    evidence_class: str = "SYNTHETIC",
) -> dict[str, Any]:
    """Certify ``strategy`` on one array and return a stampable scorecard.

    ``evidence_class`` is ``SYNTHETIC`` for formula checks and
    ``path_diagnostic`` for a simulated path that is not a market result.
    ``gradient=None`` uses ``margin_and_grad`` when the strategy has it, and
    otherwise leaves the gradient attack unavailable so a JAX backtester can
    be registered explicitly.
    """
    if evidence_class not in {"SYNTHETIC", "path_diagnostic"}:
        raise ValueError("evidence_class must be SYNTHETIC or path_diagnostic")
    if reference not in {"gaussian", "empirical"}:
        raise ValueError("reference must be 'gaussian' or 'empirical'")
    path = np.asarray(sample, dtype=float).reshape(-1)
    if path.size == 0 or not np.all(np.isfinite(path)):
        raise ValueError("sample must be a non-empty finite vector")
    certificate = _with_proven(
        certify_decision(strategy, path, sigma, draws=draws, alpha=alpha, seed=seed)
    )
    if run_attack:
        upper = max_radius
        if upper is None:
            analytic = getattr(strategy, "analytic_l2_radius", None)
            if callable(analytic) and norm == "l2" and _unit_scale(scale, path.size):
                known = float(analytic(path))
                upper = 4.0 if math.isinf(known) else max(4.0 * known, 1e-3)
            else:
                upper = 2.0
        empirical = black_box_attack(
            strategy,
            path,
            norm=norm,
            scale=scale,
            method=attack,
            max_radius=upper,
            steps=attack_steps,
            generations=generations,
            trials=trials,
            seed=seed,
        )
    else:
        empirical = _unavailable_attack(norm)
    backend = _resolve_backend(strategy, gradient)
    gradient_result = gradient_attack(
        strategy,
        path,
        backend,
        norm=norm,
        scale=scale,
        steps=gradient_steps,
    )
    if outcomes is not None:
        from quant_fund.robustness.analytic import plug_in_moments

        mean, fitted_scale = plug_in_moments(np.asarray(outcomes, dtype=float))
        moment_source = "plug_in_sample"
        if outcome_mean is not None or outcome_scale is not None:
            raise ValueError("pass outcomes or moments, not both")
        used_scale: float | None = fitted_scale
    else:
        if outcome_mean is None:
            mean = float(np.mean(path))
            moment_source = "plug_in_sample"
        else:
            mean = float(outcome_mean)
            moment_source = "population"
        if outcome_scale is None and moment_source == "population":
            used_scale = None
        elif outcome_scale is None:
            used_scale = float(np.std(path, ddof=1)) if path.size >= 2 else None
        else:
            used_scale = float(outcome_scale)
            moment_source = "population" if outcome_mean is not None else moment_source
    distribution = distributional_bounds(
        mean=mean,
        scale=used_scale,
        radius=wasserstein_radius,
        reference=reference,
        moment_source=moment_source,
    )
    regime_value = distribution["radius_to_nonpositive_mean"]["value"]
    sensitivity = assemble_sensitivity(
        strategy,
        path,
        path_attack=empirical,
        regime_radius=float(regime_value),
        scale=scale,
        base_cost=base_cost,
    )
    certified_value = certificate.get("value")
    empirical_value = empirical.get("value")
    comparable = (
        certificate.get("status") == "proven"
        and empirical.get("flipped") is True
        and certificate.get("norm") == empirical.get("norm") == "l2"
        and _unit_scale(scale, path.size)
        and isinstance(certified_value, (int, float))
        and not isinstance(certified_value, bool)
        and isinstance(empirical_value, (int, float))
        and not isinstance(empirical_value, bool)
        and float(certified_value) <= float(empirical_value) + 1e-6
    )
    analytic_block: dict[str, Any] | None = None
    analytic_fn = getattr(strategy, "analytic_l2_radius", None)
    if callable(analytic_fn) and _unit_scale(scale, path.size):
        known_radius = float(analytic_fn(path))
        analytic_block = {
            "value": None if math.isinf(known_radius) else known_radius,
            "norm": "l2",
            "status": "proven",
            "proven": True,
            "formula": "linear_margin_over_weight_norm",
        }
    return {
        "strategy": str(getattr(strategy, "name", "strategy")),
        "evidence_class": evidence_class,
        "research_only": True,
        "live_trading_claim": False,
        "certified_radius": certificate,
        "empirical_attack_radius": empirical,
        "gradient_attack": gradient_result,
        "analytic_radius": analytic_block,
        "distributional_robustness": distribution,
        "sensitivity": sensitivity,
        "radii_comparable": comparable,
        "limitations": list(_LIMITATIONS),
    }
