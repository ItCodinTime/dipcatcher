"""Randomized smoothing certificates for a discrete strategy decision.

The certificate is Cohen, Rosenfeld, and Kolter (ICML 2019), Theorem 1,
applied to isotropic Gaussian perturbations of the array the strategy reads.
It certifies the *smoothed* decision inside an L2 ball in those coordinates.
It does not certify missing bars, stale prints, spikes outside that ball,
timing jitter, cost shocks, or a Wasserstein shift of the outcome law.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from quant_fund.robustness.analytic import (
    clopper_pearson_lower,
    linf_radius_from_l2,
    smoothing_radius,
)

FloatArray = NDArray[np.float64]


def _class_counts_loop(
    strategy: Any, sample: FloatArray, sigma: float, draws: int, rng: np.random.Generator
) -> dict[int, int]:
    counts: dict[int, int] = {}
    noise = rng.normal(0.0, sigma, size=(draws, sample.shape[0]))
    for row in noise:
        decision = int(strategy.decision(sample + row))
        counts[decision] = counts.get(decision, 0) + 1
    return counts


def _class_counts(
    strategy: Any, sample: FloatArray, sigma: float, draws: int, rng: np.random.Generator
) -> dict[int, int]:
    decide_many = getattr(strategy, "decide_many", None)
    if not callable(decide_many):
        return _class_counts_loop(strategy, sample, sigma, draws, rng)
    noise = rng.normal(0.0, sigma, size=(draws, sample.shape[0]))
    decisions = np.asarray(decide_many(sample + noise), dtype=int).reshape(-1)
    if decisions.shape != (draws,):
        raise ValueError("decide_many must return one decision per draw")
    counts: dict[int, int] = {}
    for decision in decisions.tolist():
        counts[int(decision)] = counts.get(int(decision), 0) + 1
    return counts


def population_certificate(
    strategy: Any, sample: FloatArray, sigma: float
) -> dict[str, Any] | None:
    """Exact certificate when the strategy exposes a Gaussian class probability.

    Returns ``None`` when the strategy has no closed form. The radius then
    equals Cohen's formula at the population probability, with no sampling
    error. For a linear margin this equals the Euclidean distance to the
    boundary (see ``analytic.linear_l2_radius``).
    """
    probability = getattr(strategy, "population_positive_probability", None)
    if not callable(probability):
        return None
    positive = float(probability(sample, sigma))
    if not math.isfinite(positive) or not 0.0 <= positive <= 1.0:
        raise ValueError("population probability must lie in [0, 1]")
    if positive > 0.5:
        top = 1
        p_top = positive
    elif positive < 0.5:
        top = -1
        p_top = 1.0 - positive
    else:
        top = 0
        p_top = 0.5
    p_runner = 1.0 - p_top if top != 0 else 0.5
    # At a tie both classes have probability 1/2; the third class has probability 0
    # under a continuous linear margin. Abstention radius is 0.
    if top == 0:
        p_lower = 0.5
        p_runner = 0.5
    else:
        p_lower = p_top
    radius = smoothing_radius(p_lower, p_runner, sigma) if top != 0 else 0.0
    dimension = int(np.asarray(sample, dtype=float).reshape(-1).size)
    return {
        "value": None if math.isinf(radius) else radius,
        "infinite": math.isinf(radius),
        "norm": "l2",
        "unit": "input_coordinates",
        "status": "proven",
        "formula": "cohen2019_theorem1",
        "sigma": float(sigma),
        "confidence": None,
        "top_class": top,
        "p_lower": p_lower,
        "p_runner_upper": p_runner,
        "abstained": radius <= 0.0,
        "guarantee": "exact_population",
        "derived_linf_radius": None
        if math.isinf(radius)
        else linf_radius_from_l2(radius, dimension),
        "derived_linf_formula": "norm_comparison_l2_contains_linf",
        "derived_linf_status": "corollary",
    }


def monte_carlo_certificate(
    strategy: Any,
    sample: FloatArray,
    sigma: float,
    *,
    draws: int,
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    """High-probability certificate from ``draws`` Gaussian perturbations.

    The top-class count is turned into a one-sided Clopper–Pearson lower
    bound at level ``alpha``. Cohen, Rosenfeld, and Kolter (ICML 2019) plug
    that bound into Theorem 1. The guarantee is over the Monte Carlo draw:
    with probability at least ``1 - alpha`` the returned radius is a valid
    lower bound on the smoothed decision's L2 radius. It can exceed the true
    radius with probability up to ``alpha``. It is not a deterministic proof.
    """
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise ValueError("draws must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    path = np.asarray(sample, dtype=float).reshape(-1)
    counts = _class_counts(strategy, path, sigma, draws, np.random.default_rng(seed))
    if sum(counts.values()) != draws:
        raise ValueError("class counts do not add up to the number of draws")
    top = max(counts, key=lambda label: (counts[label], -abs(int(label))))
    top_count = counts[top]
    runner_count = max((count for label, count in counts.items() if label != top), default=0)
    # Bonferroni split: the lower bound on the top class and the upper bound
    # on the runner-up each fail with probability at most alpha/2, so the
    # pair fails with probability at most alpha. Cohen, Rosenfeld, and Kolter
    # (ICML 2019) use this split when more than one class is estimated.
    p_lower = clopper_pearson_lower(top_count, draws, alpha / 2.0)
    if runner_count == 0:
        p_runner = 1.0 - p_lower
    else:
        # Upper Clopper-Pearson endpoint at level alpha/2.
        p_runner = float(stats.beta.ppf(1.0 - alpha / 2.0, runner_count + 1, draws - runner_count))
    p_runner = min(p_runner, 1.0 - p_lower)
    if p_lower < p_runner:
        radius = 0.0
        abstained = True
    else:
        radius = smoothing_radius(p_lower, p_runner, sigma)
        abstained = radius <= 0.0
    dimension = int(path.size)
    return {
        "value": None if math.isinf(radius) else radius,
        "infinite": bool(math.isinf(radius)),
        "norm": "l2",
        "unit": "input_coordinates",
        "status": "high_probability",
        "formula": "cohen2019_theorem1",
        "sigma": float(sigma),
        "confidence": 1.0 - alpha,
        "alpha": alpha,
        "draws": draws,
        "top_class": int(top),
        "top_count": int(top_count),
        "p_lower": p_lower,
        "p_runner_upper": p_runner,
        "abstained": abstained,
        "guarantee": "monte_carlo_clopper_pearson",
        "derived_linf_radius": None
        if math.isinf(radius)
        else linf_radius_from_l2(radius, dimension),
        "derived_linf_formula": "norm_comparison_l2_contains_linf",
        "derived_linf_status": "corollary",
    }


def certify_decision(
    strategy: Any,
    sample: FloatArray,
    sigma: float,
    *,
    draws: int = 256,
    alpha: float = 0.001,
    seed: int = 0,
) -> dict[str, Any]:
    """Prefer the population certificate and always attach a Monte Carlo check.

    The scorecard's certified radius is the population value when the strategy
    provides one (``status=proven``). The Monte Carlo radius is reported
    alongside it and is not promoted to a proof.
    """
    population = population_certificate(strategy, sample, sigma)
    monte_carlo = monte_carlo_certificate(
        strategy, sample, sigma, draws=draws, alpha=alpha, seed=seed
    )
    if population is None:
        chosen = dict(monte_carlo)
        chosen["monte_carlo"] = {
            key: monte_carlo[key]
            for key in ("value", "p_lower", "draws", "alpha", "guarantee", "status")
        }
        return chosen
    chosen = dict(population)
    chosen["monte_carlo"] = {
        "value": monte_carlo["value"],
        "p_lower": monte_carlo["p_lower"],
        "draws": monte_carlo["draws"],
        "alpha": monte_carlo["alpha"],
        "guarantee": monte_carlo["guarantee"],
        "status": monte_carlo["status"],
        "top_class": monte_carlo["top_class"],
        "abstained": monte_carlo["abstained"],
    }
    return chosen
