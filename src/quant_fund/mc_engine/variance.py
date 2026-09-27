"""Variance-reduction designs and the estimators of their factors.

A factor is ``Var(crude mean) / Var(reduced mean)`` on a declared sample.
It is allowed to be below 1. It is null when the reduced variance is zero
(exact cancellation) or when the design does not identify a crude reference.
Factors for different techniques are not multiplied.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm, qmc

from quant_fund.mc_engine.philox import STREAM_SHOCK, philox_normals

FloatArray = NDArray[np.float64]
CrossStats = tuple[int, float, float, float, float, float]
_SOBOL_MAX_DIM = int(qmc.Sobol.MAXDIM)


def _empty_cross() -> CrossStats:
    return (0, 0.0, 0.0, 0.0, 0.0, 0.0)


def cross_stats(y: FloatArray, x: FloatArray) -> CrossStats:
    """Sufficient statistics ``(n, sum y, sum x, sum y^2, sum x^2, sum xy)``."""
    yy = np.asarray(y, dtype=np.float64).ravel()
    xx = np.asarray(x, dtype=np.float64).ravel()
    if yy.shape != xx.shape:
        raise ValueError("y and x must have the same shape")
    if yy.size == 0:
        return _empty_cross()
    if not np.isfinite(yy).all() or not np.isfinite(xx).all():
        raise ValueError("cross_stats requires finite values")
    return (
        int(yy.size),
        float(yy.sum()),
        float(xx.sum()),
        float(np.square(yy).sum()),
        float(np.square(xx).sum()),
        float((yy * xx).sum()),
    )


def add_cross(left: CrossStats, right: CrossStats) -> CrossStats:
    return (
        left[0] + right[0],
        left[1] + right[1],
        left[2] + right[2],
        left[3] + right[3],
        left[4] + right[4],
        left[5] + right[5],
    )


def antithetic_mean_vrf(values: FloatArray) -> dict[str, object]:
    """VRF of the sample mean when even/odd path pairs are antithetic.

    ``values[2k]`` and ``values[2k+1]`` are one pair. The crude variance uses
    the marginal sample variance of those paths (each margin is still a draw
    from the target law). The reduced variance is the variance of the pair
    averages, scaled to the same path count.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    n_pairs = int(arr.size // 2)
    if n_pairs < 2:
        return {
            "method": "antithetic_pairs",
            "variance_reduction_factor": None,
            "variance_reduction_factor_infinite": False,
            "reason": "need at least 2 antithetic pairs",
        }
    paired = arr[: n_pairs * 2].reshape(n_pairs, 2)
    pair_mean = paired.mean(axis=1)
    var_marginal = float(np.var(paired.ravel(), ddof=1))
    var_pair = float(np.var(pair_mean, ddof=1))
    var_crude = var_marginal / (2.0 * n_pairs)
    var_reduced = var_pair / n_pairs
    infinite = False
    factor: float | None
    reason: str | None = None
    if var_reduced == 0.0 and var_crude > 0.0:
        factor = None
        infinite = True
        reason = "antithetic pair averages have zero sample variance"
    elif var_reduced <= 0.0 or not math.isfinite(var_reduced):
        factor = None
        reason = "reduced variance is not positive"
    else:
        factor = var_crude / var_reduced
    return {
        "method": "antithetic_pairs",
        "estimator": "mean",
        "n_pairs": n_pairs,
        "marginal_variance": var_marginal,
        "pair_mean_variance": var_pair,
        "variance_crude_estimator": var_crude,
        "variance_reduced_estimator": var_reduced,
        "variance_reduction_factor": factor,
        "variance_reduction_factor_infinite": infinite,
        "reason": reason,
        "note": (
            "Same path budget. Marginal variance is estimated on the antithetic "
            "sample; each path is still a draw from the target distribution."
        ),
    }


def _centered_control_terms(
    stats: CrossStats, control_mean: float
) -> tuple[int, float, float, float, float]:
    n, sum_y, sum_x, sum_y2, sum_x2, sum_xy = stats
    if n <= 0:
        return 0, 0.0, 0.0, 0.0, 0.0
    ybar = sum_y / n
    sum_dx = sum_x - n * control_mean
    sum_dx2 = sum_x2 - 2.0 * control_mean * sum_x + n * control_mean * control_mean
    sum_y_dx = sum_xy - control_mean * sum_y
    numer = sum_y_dx - ybar * sum_dx
    return n, numer, sum_dx2, sum_y2, sum_y


def control_variate_from_stats(
    pilot: CrossStats,
    evaluation: CrossStats,
    control_mean: float,
) -> dict[str, object]:
    """Out-of-sample control variate for the mean.

    ``b`` is fit on the pilot only, using the known control mean. The factor
    is the ratio of sample variances of the raw and adjusted evaluation
    outcomes. Pilot paths are not reused for that ratio.
    """
    mu = float(control_mean)
    if not math.isfinite(mu):
        raise ValueError("control_mean must be finite")
    n_pilot, numer, den, _, _ = _centered_control_terms(pilot, mu)
    if n_pilot < 2:
        return {
            "method": "control_variate_pilot_eval",
            "variance_reduction_factor": None,
            "variance_reduction_factor_infinite": False,
            "reason": "pilot sample smaller than 2",
        }
    if den <= 0.0 or not math.isfinite(den):
        return {
            "method": "control_variate_pilot_eval",
            "variance_reduction_factor": None,
            "variance_reduction_factor_infinite": False,
            "reason": "control has zero variation on the pilot",
        }
    b = numer / den
    n_eval, sum_y, sum_x, sum_y2, _, _ = evaluation
    if n_eval < 2:
        return {
            "method": "control_variate_pilot_eval",
            "coefficient": float(b),
            "variance_reduction_factor": None,
            "variance_reduction_factor_infinite": False,
            "reason": "evaluation sample smaller than 2",
        }
    sum_dx2 = evaluation[4] - 2.0 * mu * evaluation[2] + n_eval * mu * mu
    sum_y_dx = evaluation[5] - mu * sum_y
    sum_z = sum_y - b * (sum_x - n_eval * mu)
    sum_z2 = sum_y2 + (b * b) * sum_dx2 - 2.0 * b * sum_y_dx
    var_raw = (sum_y2 - (sum_y * sum_y) / n_eval) / (n_eval - 1)
    var_cv = (sum_z2 - (sum_z * sum_z) / n_eval) / (n_eval - 1)
    raw_mean = sum_y / n_eval
    cv_mean = sum_z / n_eval
    infinite = False
    factor: float | None
    reason: str | None = None
    if not math.isfinite(var_raw) or var_raw < 0.0:
        factor = None
        reason = "raw evaluation variance is not finite"
    elif var_cv == 0.0 and var_raw > 0.0:
        factor = None
        infinite = True
        reason = "control-variate residuals have zero sample variance"
    elif var_cv <= 0.0 or not math.isfinite(var_cv):
        factor = None
        reason = "control-variate variance is not positive"
    else:
        factor = float(var_raw / var_cv)
    return {
        "method": "control_variate_pilot_eval",
        "estimator": "mean",
        "coefficient": float(b),
        "control_mean": mu,
        "n_pilot": int(pilot[0]),
        "n_evaluation": int(n_eval),
        "mean_raw_evaluation": float(raw_mean),
        "mean_control_variate_evaluation": float(cv_mean),
        "variance_raw_evaluation": float(var_raw),
        "variance_control_variate_evaluation": float(var_cv) if math.isfinite(var_cv) else None,
        "variance_reduction_factor": factor,
        "variance_reduction_factor_infinite": infinite,
        "reason": reason,
        "note": (
            "b is estimated on the pilot paths only. The factor is the ratio of "
            "evaluation-sample variances and can be below 1."
        ),
    }


def importance_shift_vector(n_steps: int, n_factors: int, shift: float) -> FloatArray:
    """Per-shock mean shift on factor 0 only.

    ``shift`` is added to every factor-0 shock. ``||mu||^2 = n_steps * shift^2``.
    The default caller passes ``shift = -1/sqrt(n_steps)`` so that squared norm is 1.
    Shifting every coordinate by a large constant makes the likelihood ratio
    degenerate; this one-factor shift is the supported parameterization.
    """
    if n_steps < 1 or n_factors < 1:
        raise ValueError("n_steps and n_factors must be positive")
    if not math.isfinite(shift):
        raise ValueError("shift must be finite")
    mu = np.zeros(n_steps * n_factors, dtype=np.float64)
    mu[0::n_factors] = float(shift)
    return mu


def importance_weights(proposal: FloatArray, shift: FloatArray) -> FloatArray:
    """Likelihood ratio ``phi(z) / phi(z; mean=shift)`` for ``z`` drawn under the shift.

    ``proposal`` has shape ``(n_paths, n_coordinates)`` and already includes the
    shift. ``weight = exp(-shift · z + 0.5 ||shift||^2)``.
    """
    z = np.asarray(proposal, dtype=np.float64)
    mu = np.asarray(shift, dtype=np.float64).ravel()
    if z.ndim != 2 or z.shape[1] != mu.size:
        raise ValueError("proposal must have shape (n_paths, shift.size)")
    if not np.isfinite(z).all() or not np.isfinite(mu).all():
        raise ValueError("proposal and shift must be finite")
    log_w = -z @ mu + 0.5 * float(np.dot(mu, mu))
    # Clip only the exponent used for the exp, and report the raw log via the caller.
    return np.asarray(np.exp(log_w), dtype=np.float64)


def importance_log_weights(proposal: FloatArray, shift: FloatArray) -> FloatArray:
    z = np.asarray(proposal, dtype=np.float64)
    mu = np.asarray(shift, dtype=np.float64).ravel()
    if z.ndim != 2 or z.shape[1] != mu.size:
        raise ValueError("proposal must have shape (n_paths, shift.size)")
    log_w = -z @ mu + 0.5 * float(np.dot(mu, mu))
    return np.asarray(log_w, dtype=np.float64)


def effective_sample_size(weights: FloatArray) -> float:
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.size == 0:
        return 0.0
    if np.any(w < 0.0) or not np.isfinite(w).all():
        raise ValueError("weights must be finite and non-negative")
    sum_sq = float(np.square(w).sum())
    if sum_sq <= 0.0:
        return 0.0
    total = float(w.sum())
    return float((total * total) / sum_sq)


def importance_mean_vrf(crude: FloatArray, weighted_outcomes: FloatArray) -> dict[str, object]:
    """VRF of the unbiased IS mean against a same-size crude sample.

    ``weighted_outcomes`` is ``weight * loss(proposal)``. The crude array is a
    separate sample of the same length, not the proposal sample.
    """
    c = np.asarray(crude, dtype=np.float64).ravel()
    w = np.asarray(weighted_outcomes, dtype=np.float64).ravel()
    if c.size != w.size or c.size < 2:
        return {
            "method": "importance_sampling_unbiased_mean",
            "variance_reduction_factor": None,
            "variance_reduction_factor_infinite": False,
            "reason": "crude and weighted samples must have the same length >= 2",
        }
    if not np.isfinite(c).all() or not np.isfinite(w).all():
        return {
            "method": "importance_sampling_unbiased_mean",
            "variance_reduction_factor": None,
            "variance_reduction_factor_infinite": False,
            "reason": "non-finite importance-sampling outcomes",
        }
    n = int(c.size)
    var_crude = float(np.var(c, ddof=1))
    var_is = float(np.var(w, ddof=1))
    var_crude_mean = var_crude / n
    var_is_mean = var_is / n
    infinite = False
    factor: float | None
    reason: str | None = None
    if var_is_mean == 0.0 and var_crude_mean > 0.0:
        factor = None
        infinite = True
        reason = "importance-weighted outcomes have zero sample variance"
    elif var_is_mean <= 0.0 or not math.isfinite(var_is_mean):
        factor = None
        reason = "importance-sampling variance is not positive"
    else:
        factor = var_crude_mean / var_is_mean
    return {
        "method": "importance_sampling_unbiased_mean",
        "estimator": "unbiased_mean",
        "n": n,
        "variance_crude_estimator": var_crude_mean,
        "variance_reduced_estimator": var_is_mean,
        "variance_reduction_factor": factor,
        "variance_reduction_factor_infinite": infinite,
        "reason": reason,
        "note": (
            "Common seed, two generator evaluations per path: unshifted crude "
            "loss and shifted proposal loss. The factor refers to the unbiased "
            "mean mean(weight * loss), not the self-normalized mean."
        ),
    }


def rqmc_mean_vrf(scramble_means: FloatArray, crude_values: FloatArray) -> dict[str, object]:
    """Compare one Sobol scramble's mean to crude MC at the same path count.

    ``scramble_means`` has one entry per independent scramble, each the mean
    loss of ``n`` paths. ``crude_values`` is one crude sample of those ``n``
    paths. The factor is ``(s^2 / n) / Var(scramble mean)``.
    """
    means = np.asarray(scramble_means, dtype=np.float64).ravel()
    crude = np.asarray(crude_values, dtype=np.float64).ravel()
    if means.size < 2 or crude.size < 2:
        return {
            "method": "rqmc_vs_crude_mean",
            "variance_reduction_factor": None,
            "variance_reduction_factor_infinite": False,
            "reason": "need at least 2 scrambles and 2 crude paths",
        }
    if not np.isfinite(means).all() or not np.isfinite(crude).all():
        return {
            "method": "rqmc_vs_crude_mean",
            "variance_reduction_factor": None,
            "variance_reduction_factor_infinite": False,
            "reason": "non-finite RQMC or crude values",
        }
    n = int(crude.size)
    var_scramble = float(np.var(means, ddof=1))
    var_crude_mean = float(np.var(crude, ddof=1) / n)
    se = math.sqrt(var_scramble / means.size) if var_scramble >= 0.0 else None
    infinite = False
    factor: float | None
    reason: str | None = None
    if var_scramble == 0.0 and var_crude_mean > 0.0:
        factor = None
        infinite = True
        reason = "scramble means have zero sample variance"
    elif var_scramble <= 0.0 or not math.isfinite(var_scramble):
        factor = None
        reason = "RQMC scramble variance is not positive"
    else:
        factor = var_crude_mean / var_scramble
    return {
        "method": "rqmc_vs_crude_mean",
        "estimator": "mean",
        "n_scrambles": int(means.size),
        "n_paths_per_scramble": n,
        "scramble_means": [float(v) for v in means.tolist()],
        "variance_one_scramble_mean": var_scramble,
        "variance_crude_estimator": var_crude_mean,
        "rqmc_mean_standard_error": None if se is None else float(se),
        "variance_reduction_factor": factor,
        "variance_reduction_factor_infinite": infinite,
        "reason": reason,
        "note": (
            "The factor compares one scramble of n paths with crude Monte Carlo "
            "of n paths. Extra scrambles are the measurement instrument. They are "
            "not pooled into the headline loss distribution. The crude reference "
            "is not pooled either."
        ),
    }


def sobol_normals(
    start: int,
    count: int,
    n_normals: int,
    *,
    seed: int,
    scramble: bool,
) -> FloatArray:
    """Normals from a contiguous slice of a scrambled (or plain) Sobol sequence.

    Point ``start + i`` is row ``i``. ``scipy.stats.qmc.Sobol.fast_forward``
    makes the slice independent of which chunk asks for it. Coordinates are
    mapped with ``norm.ppf`` after clipping to ``(eps, 1-eps)``.
    """
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise ValueError("start must be a non-negative int")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a non-negative int")
    if isinstance(n_normals, bool) or not isinstance(n_normals, int) or n_normals < 1:
        raise ValueError("n_normals must be a positive int")
    if n_normals > _SOBOL_MAX_DIM:
        raise ValueError(
            f"Sobol dimension {n_normals} exceeds scipy's limit {_SOBOL_MAX_DIM}. "
            "Use shock_mode='crude' or 'antithetic' for this path length."
        )
    if count == 0:
        return np.zeros((0, n_normals), dtype=np.float64)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative int")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The balance properties of Sobol",
            category=UserWarning,
        )
        engine = qmc.Sobol(d=n_normals, scramble=scramble, seed=seed if scramble else None)
        # scipy 1.18 raises OverflowError on fast_forward(0); the engine is already there.
        if start > 0:
            engine.fast_forward(start)
        uniforms = np.asarray(engine.random(count), dtype=np.float64)
    eps = np.finfo(np.float64).eps
    uniforms = np.clip(uniforms, eps, 1.0 - eps)
    return np.asarray(norm.ppf(uniforms), dtype=np.float64)


def draw_standard_normals(
    indices: NDArray[np.int64],
    n_steps: int,
    n_factors: int,
    *,
    seed: int,
    shock_mode: str,
    importance_shift: float,
    qmc_scramble: bool,
    qmc_seed: int,
) -> tuple[FloatArray, FloatArray | None]:
    """Standard or proposal normals, plus importance weights (or ``None``).

    Shape of the returned shocks is ``(n_paths, n_steps, n_factors)``.
    Antithetic path ``2k+1`` is the negation of the Philox draw for base ``k``.
    """
    if shock_mode not in {"crude", "antithetic", "qmc_sobol", "importance"}:
        raise ValueError(f"unknown shock_mode {shock_mode!r}")
    idx = np.asarray(indices, dtype=np.int64).ravel()
    n_normals = n_steps * n_factors
    if shock_mode == "qmc_sobol":
        if idx.size > 0:
            start = int(idx[0])
            if not np.array_equal(idx, np.arange(start, start + idx.size, dtype=np.int64)):
                raise ValueError("QMC chunks must be contiguous path indices")
        else:
            start = 0
        flat = sobol_normals(start, int(idx.size), n_normals, seed=qmc_seed, scramble=qmc_scramble)
        shocks = flat.reshape(idx.size, n_steps, n_factors)
        return shocks, None
    if shock_mode == "antithetic":
        if idx.size > 0 and (int(idx[0]) % 2 != 0 or int(idx[-1]) % 2 == 0):
            raise ValueError("antithetic chunks must cover whole even/odd pairs")
        base = idx // 2
        sign = np.where(idx % 2 == 0, 1.0, -1.0)
        flat = philox_normals(seed, base, n_normals, stream_id=STREAM_SHOCK)
        flat = flat * sign[:, None]
        shocks = flat.reshape(idx.size, n_steps, n_factors)
        return shocks, None
    flat = philox_normals(seed, idx, n_normals, stream_id=STREAM_SHOCK)
    if shock_mode == "importance":
        mu = importance_shift_vector(n_steps, n_factors, importance_shift)
        proposal = flat + mu.reshape(1, -1)
        weights = importance_weights(proposal, mu)
        shocks = proposal.reshape(idx.size, n_steps, n_factors)
        return shocks, weights
    shocks = flat.reshape(idx.size, n_steps, n_factors)
    return shocks, None
