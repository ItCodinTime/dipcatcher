"""Empirical attacks that search for a small decision-flipping perturbation.

Black-box search uses CMA-ES (Hansen, arXiv:1604.00772, 2016) or
Optuna's TPE sampler (Akiba et al., KDD 2019; the sampler follows Bergstra,
Bardenet, Bengio, and Kégl, NeurIPS 2011). Gradient search uses projected
gradient steps (Madry et al., ICLR 2018) through a :class:`GradientBackend`.

A successful attack is an upper bound on the minimal perturbation: the
returned vector was checked to flip the decision, and its norm is the
reported radius. Failure to flip is not a certificate of robustness.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.robustness.gradients import GradientBackend
from quant_fund.robustness.threats import apply_vol_scaled, ball_norm, project

FloatArray = NDArray[np.float64]


def _as_path(sample: FloatArray) -> FloatArray:
    path = np.asarray(sample, dtype=float).reshape(-1)
    if path.size == 0 or not np.all(np.isfinite(path)):
        raise ValueError("sample must be a non-empty finite vector")
    return path


def _flipped(
    strategy: Any, base: FloatArray, decision: int, delta: FloatArray, scale: FloatArray | None
) -> bool:
    return int(strategy.decision(apply_vol_scaled(base, delta, scale))) != decision


def cmaes_minimize(
    objective: Any,
    x0: FloatArray,
    sigma: float,
    *,
    generations: int,
    seed: int,
) -> tuple[FloatArray, float]:
    """Minimize ``objective`` with the rank-μ CMA-ES update.

    Parameters follow Hansen's tutorial (arXiv:1604.00772), Table 1, without
    active covariance updates. The return value is the best evaluated point,
    which is an empirical result.
    """
    if not callable(objective):
        raise TypeError("objective must be callable")
    if isinstance(generations, bool) or not isinstance(generations, int) or generations < 1:
        raise ValueError("generations must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    mean = np.asarray(x0, dtype=float).reshape(-1).copy()
    dimension = int(mean.size)
    if dimension < 1 or not np.all(np.isfinite(mean)):
        raise ValueError("x0 must be a non-empty finite vector")
    rng = np.random.default_rng(seed)
    lam = 4 + int(math.floor(3.0 * math.log(dimension)))
    mu = lam // 2
    raw = np.array([math.log(mu + 0.5) - math.log(i + 1.0) for i in range(mu)])
    weights = raw / np.sum(raw)
    mu_eff = float(1.0 / np.sum(weights**2))
    c_sigma = (mu_eff + 2.0) / (dimension + mu_eff + 5.0)
    d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (dimension + 1.0)) - 1.0) + c_sigma
    c_c = (4.0 + mu_eff / dimension) / (dimension + 4.0 + 2.0 * mu_eff / dimension)
    c_1 = 2.0 / ((dimension + 1.3) ** 2 + mu_eff)
    c_mu = min(1.0 - c_1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((dimension + 2.0) ** 2 + mu_eff))
    chi = math.sqrt(dimension) * (
        1.0 - 1.0 / (4.0 * dimension) + 1.0 / (21.0 * dimension * dimension)
    )
    pc = np.zeros(dimension)
    ps = np.zeros(dimension)
    covariance = np.eye(dimension)
    step = float(sigma)
    best_x = mean.copy()
    best_value = float(objective(best_x))
    for generation in range(generations):
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 1e-20)
        inv_sqrt = (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T
        samples: list[FloatArray] = []
        steps: list[FloatArray] = []
        values: list[float] = []
        for _ in range(lam):
            z = rng.normal(size=dimension)
            y = eigenvectors @ (np.sqrt(eigenvalues) * z)
            point = mean + step * y
            samples.append(point)
            steps.append(y)
            values.append(float(objective(point)))
        order = np.argsort(values)
        y_w = np.zeros(dimension)
        for index in range(mu):
            y_w += weights[index] * steps[int(order[index])]
        mean = mean + step * y_w
        ps = (1.0 - c_sigma) * ps + math.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff) * (inv_sqrt @ y_w)
        ps_norm = float(np.linalg.norm(ps))
        step *= math.exp((c_sigma / d_sigma) * (ps_norm / chi - 1.0))
        # Stall the covariance path when the step-size path is too large
        # (Hansen's h_sig indicator).
        indicator_scale = ps_norm / math.sqrt(1.0 - (1.0 - c_sigma) ** (2.0 * (generation + 1)))
        h_sig = 1.0 if indicator_scale < (1.4 + 2.0 / (dimension + 1.0)) * chi else 0.0
        pc = (1.0 - c_c) * pc + h_sig * math.sqrt(c_c * (2.0 - c_c) * mu_eff) * y_w
        rank_update = np.zeros((dimension, dimension))
        for index in range(mu):
            step_i = steps[int(order[index])]
            rank_update += weights[index] * np.outer(step_i, step_i)
        covariance = (
            (1.0 - c_1 - c_mu) * covariance
            + c_1 * (np.outer(pc, pc) + (1.0 - h_sig) * c_c * (2.0 - c_c) * covariance)
            + c_mu * rank_update
        )
        elite = int(order[0])
        if values[elite] < best_value:
            best_value = values[elite]
            best_x = samples[elite].copy()
    return best_x, best_value


def tpe_minimize(
    objective: Any,
    lower: FloatArray,
    upper: FloatArray,
    *,
    trials: int,
    seed: int,
) -> tuple[FloatArray, float]:
    """Minimize ``objective`` with Optuna's TPE sampler inside a box."""
    if not callable(objective):
        raise TypeError("objective must be callable")
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise ValueError("trials must be a positive integer")
    lo = np.asarray(lower, dtype=float).reshape(-1)
    hi = np.asarray(upper, dtype=float).reshape(-1)
    if lo.shape != hi.shape or lo.size == 0 or np.any(lo > hi):
        raise ValueError("bounds must be aligned with lower <= upper")
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))

    def _trial(trial: optuna.Trial) -> float:
        point = np.array(
            [
                trial.suggest_float(f"x{index}", float(lo[index]), float(hi[index]))
                for index in range(lo.size)
            ]
        )
        return float(objective(point))

    study.optimize(_trial, n_trials=trials, show_progress_bar=False)
    best = np.array([float(study.best_params[f"x{index}"]) for index in range(lo.size)])
    return best, float(study.best_value)


def _direction_to_delta(direction: FloatArray, radius: float, norm: str) -> FloatArray:
    vector = np.asarray(direction, dtype=float).reshape(-1)
    length = float(np.linalg.norm(vector))
    if length == 0.0 or radius == 0.0:
        return np.zeros_like(vector)
    if norm == "l2":
        return vector * (radius / length)
    # L-infinity: use the signed unit box corner in the proposed direction.
    signed = np.sign(vector)
    signed[signed == 0.0] = 1.0
    return project(radius * signed, radius, "linf")


def _search_once(
    strategy: Any,
    sample: FloatArray,
    decision: int,
    radius: float,
    norm: str,
    scale: FloatArray | None,
    method: str,
    *,
    seed: int,
    generations: int,
    trials: int,
) -> FloatArray | None:
    """Return a perturbation of the requested radius that flips, if one is found."""

    def objective(direction: FloatArray) -> float:
        delta = _direction_to_delta(direction, radius, norm)
        perturbed = apply_vol_scaled(sample, delta, scale)
        # Score-based black box when the strategy exposes a margin. Hard-label
        # queries are the fallback: every non-flip ties, so the search only
        # learns from samples that already flip.
        scorer = getattr(strategy, "margin_and_grad", None)
        if callable(scorer):
            margin, _gradient = scorer(perturbed)
            sign = float(decision) if decision != 0 else 1.0
            return float(sign * margin)
        return 0.0 if int(strategy.decision(perturbed)) != decision else 1.0

    if method == "cmaes":
        start = np.zeros(sample.shape[0])
        best, _value = cmaes_minimize(
            objective, start, sigma=1.0, generations=generations, seed=seed
        )
        rng = np.random.default_rng(seed + 1)
        restart = np.asarray(rng.normal(size=int(sample.shape[0])), dtype=float)
        other, _other_value = cmaes_minimize(
            objective,
            restart,
            sigma=1.0,
            generations=generations,
            seed=seed + 1,
        )
        first = _direction_to_delta(best, radius, norm)
        second = _direction_to_delta(other, radius, norm)
        for candidate in (first, second):
            if _flipped(strategy, sample, decision, candidate, scale):
                return candidate
        return None
    if method == "tpe":
        bound = np.ones(sample.shape[0])
        best, _value = tpe_minimize(objective, -bound, bound, trials=trials, seed=seed)
        candidate = _direction_to_delta(best, radius, norm)
        if _flipped(strategy, sample, decision, candidate, scale):
            return candidate
        return None
    raise ValueError("method must be 'cmaes' or 'tpe'")


def black_box_attack(
    strategy: Any,
    sample: FloatArray,
    *,
    norm: str = "l2",
    scale: FloatArray | None = None,
    method: str = "cmaes",
    max_radius: float = 2.0,
    steps: int = 12,
    generations: int = 8,
    trials: int = 40,
    seed: int = 0,
) -> dict[str, Any]:
    """Smallest radius at which black-box search flips ``strategy.decision``.

    Binary search on the radius. Each probe asks CMA-ES or TPE for a
    perturbation inside that ball. The reported radius is the smallest probe
    that flipped, so it is an empirical upper bound on the minimal radius
    when ``status`` is ``flipped``.
    """
    if method not in {"cmaes", "tpe"}:
        raise ValueError("method must be 'cmaes' or 'tpe'")
    if norm not in {"l2", "linf"}:
        raise ValueError("norm must be 'l2' or 'linf'")
    if not math.isfinite(max_radius) or max_radius <= 0.0:
        raise ValueError("max_radius must be finite and positive")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    path = _as_path(sample)
    base_decision = int(strategy.decision(path))
    if _flipped(strategy, path, base_decision, np.zeros_like(path), scale):
        return _attack_record(method, norm, 0.0, True, "flipped")
    top = _search_once(
        strategy,
        path,
        base_decision,
        max_radius,
        norm,
        scale,
        method,
        seed=seed,
        generations=generations,
        trials=trials,
    )
    if top is None or not _flipped(strategy, path, base_decision, top, scale):
        return _attack_record(method, norm, None, False, "not_found")
    lo = 0.0
    hi = max_radius
    found = top
    for step_index in range(steps):
        mid = 0.5 * (lo + hi)
        candidate = _search_once(
            strategy,
            path,
            base_decision,
            mid,
            norm,
            scale,
            method,
            seed=seed + 17 * (step_index + 1),
            generations=generations,
            trials=trials,
        )
        if candidate is not None and _flipped(strategy, path, base_decision, candidate, scale):
            hi = mid
            found = candidate
        else:
            lo = mid
    radius = ball_norm(found, norm)
    # The binary search accepts ``hi``, whose feasible perturbation has norm
    # ``hi`` by construction of ``_direction_to_delta``. Report that probe,
    # not a shorter vector the optimizer happened to return.
    if not _flipped(strategy, path, base_decision, found, scale):
        return _attack_record(method, norm, None, False, "not_found")
    return _attack_record(method, norm, hi if hi <= max_radius else radius, True, "flipped")


def _attack_record(
    method: str, norm: str, radius: float | None, flipped: bool, status: str
) -> dict[str, Any]:
    return {
        "value": radius,
        "norm": norm,
        "unit": "volatility_scaled",
        "status": "empirical" if status == "flipped" else status,
        "method": method,
        "flipped": flipped,
        "proven": False,
        "guarantee": "upper_bound_when_flipped" if flipped else "no_certificate",
    }


def gradient_attack(
    strategy: Any,
    sample: FloatArray,
    backend: GradientBackend | None,
    *,
    norm: str = "l2",
    scale: FloatArray | None = None,
    steps: int = 8,
) -> dict[str, Any]:
    """Minimal perturbation by gradient steps, when a backend is available.

    L2 uses one or more Newton steps on the linearized margin (exact for a
    linear margin). L-infinity uses the signed gradient step of Madry et al.
    (ICLR 2018) inside a binary search on the radius. Without a backend the
    result is ``unavailable`` — finite differences are not assumed.
    """
    if norm not in {"l2", "linf"}:
        raise ValueError("norm must be 'l2' or 'linf'")
    path = _as_path(sample)
    if backend is None:
        return {
            "value": None,
            "norm": norm,
            "unit": "volatility_scaled",
            "status": "unavailable",
            "method": "pgd",
            "flipped": False,
            "proven": False,
            "reason": "no_gradient_backend",
            "guarantee": "none",
        }
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    base_decision = int(strategy.decision(path))
    if norm == "l2":
        return _newton_l2(strategy, path, base_decision, backend, scale, steps)
    return _pgd_linf(strategy, path, base_decision, backend, scale, steps)


def _scaled_grad(gradient: FloatArray, scale: FloatArray | None) -> FloatArray:
    grad = np.asarray(gradient, dtype=float).reshape(-1)
    if scale is None:
        return grad
    vol = np.asarray(scale, dtype=float).reshape(-1)
    return np.asarray(grad * vol, dtype=float)


def _newton_l2(
    strategy: Any,
    path: FloatArray,
    base_decision: int,
    backend: GradientBackend,
    scale: FloatArray | None,
    steps: int,
) -> dict[str, Any]:
    delta = np.zeros_like(path)
    for _ in range(steps):
        margin, gradient = backend.value_and_grad(apply_vol_scaled(path, delta, scale))
        grad = _scaled_grad(gradient, scale)
        gram = float(np.dot(grad, grad))
        if not math.isfinite(margin) or gram == 0.0 or not math.isfinite(gram):
            break
        # Move in eta-space. Chain rule: d(margin)/d(eta) = grad_path * scale.
        # A pure Newton step lands on the boundary. Floating-point residual can
        # leave the decision unchanged, so a 1e-8 push finishes the crossing.
        # The reported radius is then a hair above the geometric distance.
        delta = delta - (margin / gram) * grad
        if not _flipped(strategy, path, base_decision, delta, scale):
            delta = delta - math.copysign(1e-8, margin) * grad / math.sqrt(gram)
        if _flipped(strategy, path, base_decision, delta, scale):
            break
    if not _flipped(strategy, path, base_decision, delta, scale):
        return {
            "value": None,
            "norm": "l2",
            "unit": "volatility_scaled",
            "status": "not_found",
            "method": "pgd",
            "flipped": False,
            "proven": False,
            "guarantee": "no_certificate",
        }
    return {
        "value": ball_norm(delta, "l2"),
        "norm": "l2",
        "unit": "volatility_scaled",
        "status": "empirical",
        "method": "pgd",
        "flipped": True,
        "proven": False,
        "guarantee": "upper_bound_when_flipped",
    }


def _pgd_linf(
    strategy: Any,
    path: FloatArray,
    base_decision: int,
    backend: GradientBackend,
    scale: FloatArray | None,
    steps: int,
) -> dict[str, Any]:
    # Expand the box until signed-gradient PGD flips, then binary-search it.
    radius = 1e-3
    found: FloatArray | None = None
    found_radius = 0.0
    for _ in range(20):
        delta = _linf_pgd(path, backend, scale, radius, steps)
        if _flipped(strategy, path, base_decision, delta, scale):
            found = delta
            found_radius = radius
            break
        radius *= 2.0
        if radius > 1e6:
            break
    if found is None:
        return {
            "value": None,
            "norm": "linf",
            "unit": "volatility_scaled",
            "status": "not_found",
            "method": "pgd",
            "flipped": False,
            "proven": False,
            "guarantee": "no_certificate",
        }
    lo = 0.0
    hi = found_radius
    best = found
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        delta = _linf_pgd(path, backend, scale, mid, steps)
        if _flipped(strategy, path, base_decision, delta, scale):
            hi = mid
            best = delta
        else:
            lo = mid
    if not _flipped(strategy, path, base_decision, best, scale):
        return {
            "value": None,
            "norm": "linf",
            "unit": "volatility_scaled",
            "status": "not_found",
            "method": "pgd",
            "flipped": False,
            "proven": False,
            "guarantee": "no_certificate",
        }
    return {
        "value": hi,
        "norm": "linf",
        "unit": "volatility_scaled",
        "status": "empirical",
        "method": "pgd",
        "flipped": True,
        "proven": False,
        "guarantee": "upper_bound_when_flipped",
    }


def _linf_pgd(
    path: FloatArray,
    backend: GradientBackend,
    scale: FloatArray | None,
    radius: float,
    steps: int,
) -> FloatArray:
    delta = np.zeros_like(path)
    alpha = max(radius / max(steps, 1), 1e-6)
    for _ in range(steps):
        _margin, gradient = backend.value_and_grad(apply_vol_scaled(path, delta, scale))
        grad = _scaled_grad(gradient, scale)
        # Descend the margin. Sign steps are the L-infinity PGD update.
        delta = project(delta - alpha * np.sign(grad), radius, "linf")
    return delta
