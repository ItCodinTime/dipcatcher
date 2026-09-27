"""Closed forms used by the robustness certifier.

Each function states the claim it actually proves. Monte Carlo searches and
optimizer outputs do not live here.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]

# Gelbrich disk comparisons treat radii this close to the boundary as on it.
_REL_TOL = 1e-12


def clopper_pearson_lower(successes: int, trials: int, alpha: float) -> float:
    """One-sided Clopper–Pearson lower bound on a binomial probability.

    Clopper and Pearson (Biometrika, 1934). Cohen, Rosenfeld, and Kolter
    (ICML 2019) use the beta inversion
    ``proportion_confint(count, n, alpha=2*alpha, method="beta")``, whose
    lower endpoint is ``Beta(successes, trials - successes + 1).ppf(alpha)``.
    """
    if isinstance(successes, bool) or isinstance(trials, bool):
        raise TypeError("counts must be integers")
    if not isinstance(successes, int) or not isinstance(trials, int):
        raise TypeError("counts must be integers")
    if trials < 1 or successes < 0 or successes > trials:
        raise ValueError("successes must lie in 0..trials with trials >= 1")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if successes == 0:
        return 0.0
    bound = float(stats.beta.ppf(alpha, successes, trials - successes + 1))
    if not math.isfinite(bound):
        raise ValueError("Clopper-Pearson bound was not finite")
    return bound


def smoothing_radius(p_lower: float, p_runner_upper: float, sigma: float) -> float:
    """L2 radius of a randomized-smoothing certificate.

    Cohen, Rosenfeld, and Kolter (ICML 2019), Theorem 1. If the top class
    probability is at least ``p_lower`` and every other class is at most
    ``p_runner_upper``, with ``p_lower >= p_runner_upper``, isotropic Gaussian
    noise of standard deviation ``sigma`` certifies the smoothed decision
    inside the Euclidean ball of radius

        sigma / 2 * (Phi^{-1}(p_lower) - Phi^{-1}(p_runner_upper)).

    The binary case ``p_runner_upper = 1 - p_lower`` simplifies to
    ``sigma * Phi^{-1}(p_lower)``. A non-positive radius means the sample
    does not certify a class (abstention), which this function reports as 0.
    """
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    for name, value in (("p_lower", p_lower), ("p_runner_upper", p_runner_upper)):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")
    if p_lower < p_runner_upper:
        raise ValueError("p_lower must be at least p_runner_upper")
    if p_lower <= 0.5:
        return 0.0
    if p_runner_upper <= 0.0 and p_lower >= 1.0:
        return math.inf
    upper = p_runner_upper if p_runner_upper > 0.0 else 0.0
    # ppf(0) is -inf; a runner-up bounded by 0 makes the radius infinite
    # only together with p_lower == 1, already returned above. A positive
    # but tiny runner bound stays finite.
    if upper == 0.0:
        return math.inf
    radius = 0.5 * sigma * (float(stats.norm.ppf(p_lower)) - float(stats.norm.ppf(upper)))
    if not math.isfinite(radius):
        return math.inf if radius > 0 else 0.0
    return max(0.0, radius)


def linf_radius_from_l2(l2_radius: float, dimension: int) -> float:
    """Largest L-infinity ball contained in an L2 ball of the given radius.

    ``||z||_2 <= sqrt(d) ||z||_inf``, so ``||z||_inf < R/sqrt(d)`` implies
    ``||z||_2 < R``. This is a corollary of the norm comparison, not a tight
    certificate for the infinity norm.
    """
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise ValueError("dimension must be a positive integer")
    if not math.isfinite(l2_radius) or l2_radius < 0.0:
        raise ValueError("l2_radius must be finite and non-negative")
    if math.isinf(l2_radius):
        return math.inf
    return float(l2_radius / math.sqrt(dimension))


def linear_margin(weights: FloatArray, bias: float, sample: FloatArray) -> float:
    """Signed margin ``weights · sample + bias``."""
    w = np.asarray(weights, dtype=float).reshape(-1)
    x = np.asarray(sample, dtype=float).reshape(-1)
    if w.shape != x.shape:
        raise ValueError("weights and sample must share a shape")
    if not np.all(np.isfinite(w)) or not np.all(np.isfinite(x)) or not math.isfinite(bias):
        raise ValueError("margin inputs must be finite")
    return float(w @ x + bias)


def linear_l2_radius(weights: FloatArray, bias: float, sample: FloatArray) -> float:
    """Euclidean distance from ``sample`` to the linear decision boundary.

    The minimal L2 perturbation that zeros ``w · x + b`` is ``|w·x+b| / ||w||``.
    A zero weight vector never crosses a nonzero margin (infinite radius) and
    is already on the boundary when the margin is zero.
    """
    margin = linear_margin(weights, bias, sample)
    norm = float(np.linalg.norm(np.asarray(weights, dtype=float).reshape(-1)))
    if norm == 0.0:
        return 0.0 if margin == 0.0 else math.inf
    return abs(margin) / norm


def linear_positive_probability(
    weights: FloatArray, bias: float, sample: FloatArray, sigma: float
) -> float:
    """Probability that Gaussian smoothing keeps a positive linear margin.

    With noise ``ε ~ N(0, sigma^2 I)``, ``w·ε`` is univariate normal with
    standard deviation ``sigma ||w||``, so

        P(w·(x+ε)+b > 0) = Phi(margin / (sigma ||w||)).

    The population smoothed top-class probability is the larger of this value
    and its complement. Cohen's radius then equals ``linear_l2_radius``.
    """
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    margin = linear_margin(weights, bias, sample)
    norm = float(np.linalg.norm(np.asarray(weights, dtype=float).reshape(-1)))
    if norm == 0.0:
        if margin > 0.0:
            return 1.0
        if margin < 0.0:
            return 0.0
        return 0.5
    return float(stats.norm.cdf(margin / (sigma * norm)))


def linear_spike_radius(
    weights: FloatArray,
    bias: float,
    sample: FloatArray,
    scale: FloatArray | None = None,
) -> float:
    """Smallest absolute one-coordinate spike, in volatility-scaled units, that
    zeros a linear margin.

    An additive spike ``s`` on coordinate ``i`` changes the margin by
    ``w_i * scale_i * s``. The minimal ``|s|`` is ``|margin| / max_i |w_i scale_i|``.
    """
    margin = linear_margin(weights, bias, sample)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if scale is None:
        scaled = np.abs(w)
    else:
        vol = np.asarray(scale, dtype=float).reshape(-1)
        if vol.shape != w.shape or not np.all(np.isfinite(vol)) or np.any(vol < 0.0):
            raise ValueError("scale must be finite, non-negative, and aligned")
        scaled = np.abs(w * vol)
    largest = float(np.max(scaled)) if scaled.size else 0.0
    if largest == 0.0:
        return 0.0 if margin == 0.0 else math.inf
    return abs(margin) / largest


def wasserstein_worst_case_mean(mean: float, radius: float) -> float:
    """Exact worst-case mean inside a Wasserstein ball.

    For every order ``p >= 1`` and ground cost ``|u - v|`` on the real line,

        inf { E_Q[X] : W_p(Q, P) <= radius } = E_P[X] - radius.

    The shift coupling that moves every outcome down by ``radius`` has
    transport cost ``radius`` and attains the value. No coupling can move the
    mean by more than ``W_1``, and ``W_1 <= W_p``. See the Kantorovich–Rubinstein
    theorem (Villani, Optimal Transport, 2009) and the linear case of
    Mohajerin Esfahani and Kuhn (Mathematical Programming, 2018).
    """
    if not math.isfinite(mean) or not math.isfinite(radius) or radius < 0.0:
        raise ValueError("mean must be finite and radius must be finite and non-negative")
    return float(mean - radius)


def lipschitz_worst_case(value: float, lipschitz: float, radius: float, *, lower: bool) -> float:
    """Worst-case expectation of an L-Lipschitz score over a Wasserstein ball.

    Kantorovich–Rubinstein: the W1 ball of radius ``radius`` moves the
    expectation of an L-Lipschitz function by at most ``radius * L``. Esfahani
    and Kuhn (Mathematical Programming, 2018) record the same dual for the
    Lipschitz case. The bound is tight on an unbounded space. ``lipschitz``
    is the Lipschitz constant in the ground metric the ball uses.
    """
    for name, number in (("value", value), ("lipschitz", lipschitz), ("radius", radius)):
        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite")
    if lipschitz < 0.0 or radius < 0.0:
        raise ValueError("lipschitz constant and radius must be non-negative")
    shift = lipschitz * radius
    return float(value - shift) if lower else float(value + shift)


def gelbrich_worst_case_ratio(mean: float, scale: float, radius: float) -> tuple[float | None, str]:
    """Worst-case mean-to-scale ratio over the Gelbrich moment disk.

    Gelbrich (Mathematische Nachrichten, 1990) proved that univariate laws
    satisfy ``W_2^2(P, Q) >= (μ_P - μ_Q)^2 + (σ_P - σ_Q)^2``. Every law inside
    the W2 ball of radius ``radius`` therefore has moments inside that disk.
    Minimizing ``μ' / σ'`` over the disk lower-bounds the worst-case ratio.
    Equality holds for Gaussians, because the Gelbrich distance between
    univariate Gaussians is the W2 distance, so the bound is tight when the
    reference law is Gaussian and vacuous when the disk minimum is unbounded
    and the reference is not Gaussian.

    For ``0 <= radius < hypot(mean, scale)`` and ``radius != scale`` the
    tangent condition ``|mean - t scale| = radius sqrt(1 + t^2)`` gives

        t = (mean * scale - radius * sqrt(mean^2 + scale^2 - radius^2))
            / (scale^2 - radius^2).

    On ``radius == scale`` with ``mean > 0`` the same geometry collapses to
    ``(mean^2 - scale^2) / (2 mean scale)``. On the origin-touching radius
    ``radius == hypot(mean, scale)`` with ``mean > 0`` the finite infimum is
    ``-scale / mean``. Otherwise, when the disk can drive the scale to zero
    while the mean stays strictly negative, the infimum is unbounded below
    and this function returns ``(None, "unbounded_below")``.
    """
    if not all(math.isfinite(number) for number in (mean, scale, radius)):
        raise ValueError("mean, scale, and radius must be finite")
    if scale <= 0.0 or radius < 0.0:
        raise ValueError("scale must be positive and radius non-negative")
    if radius == 0.0:
        return float(mean / scale), "nominal"
    gap = math.hypot(mean, scale)
    if radius > gap and not math.isclose(radius, gap, rel_tol=_REL_TOL, abs_tol=_REL_TOL):
        return None, "unbounded_below"
    if mean <= 0.0 and radius >= scale:
        return None, "unbounded_below"
    if math.isclose(radius, scale, rel_tol=_REL_TOL, abs_tol=_REL_TOL * max(1.0, scale)):
        value = (mean * mean - scale * scale) / (2.0 * mean * scale)
        return float(value), "scale_boundary"
    if math.isclose(radius, gap, rel_tol=_REL_TOL, abs_tol=_REL_TOL):
        return float(-scale / mean), "origin_boundary"
    disc = gap * gap - radius * radius
    if disc < 0.0:
        return None, "unbounded_below"
    denom = scale * scale - radius * radius
    value = (mean * scale - radius * math.sqrt(disc)) / denom
    return float(value), "tangent"


def cost_shock_radius(gross: float, turnover: float, base_cost: float) -> float | None:
    """Additive cost-rate increase that makes ``gross - cost * turnover`` non-positive.

    Algebra on one fixed simulated path. ``None`` means no finite increase
    can flip the sign: turnover is zero and the gross sum is still positive.
    A zero radius means the path is already non-positive at ``base_cost``.
    This is not a market result.
    """
    for name, number in (("gross", gross), ("turnover", turnover), ("base_cost", base_cost)):
        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite")
    if turnover < 0.0 or base_cost < 0.0:
        raise ValueError("turnover and base_cost must be non-negative")
    net = gross - base_cost * turnover
    if net <= 0.0:
        return 0.0
    if turnover == 0.0:
        return None
    return float(net / turnover)


def plug_in_moments(outcomes: FloatArray) -> tuple[float, float | None]:
    """Sample mean and sample standard deviation (ddof=1).

    Scale is ``None`` when fewer than two finite outcomes exist. Non-finite
    outcomes are rejected rather than dropped.
    """
    sample = np.asarray(outcomes, dtype=float).reshape(-1)
    if sample.size == 0 or not np.all(np.isfinite(sample)):
        raise ValueError("outcomes must be non-empty and finite")
    mean = float(np.mean(sample))
    if sample.size < 2:
        return mean, None
    scale = float(np.std(sample, ddof=1))
    if scale == 0.0:
        return mean, 0.0
    return mean, scale
