"""Pathwise tail summaries, intervals, and peaks-over-threshold diagnostics.

The numeraire starts at 1. Loss is ``1 - terminal level`` (positive means the
path finished lower). Expected shortfall uses the same spectral definition as
:func:`quant_fund.metrics.spectral_risk.expected_shortfall_srm` on an
unweighted sample.

Intervals are Monte Carlo error under the scenario design. They are not
sampling error of a historical market estimate.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sstats

from quant_fund.metrics.spectral_risk import expected_shortfall_srm

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

NUMERAIRE_START = 1.0


def normal_z(ci_level: float) -> float:
    level = float(ci_level)
    if not math.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("ci_level must be in (0, 1)")
    return float(sstats.norm.ppf(0.5 + 0.5 * level))


def wilson_interval(successes: float, n: int, z: float) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (unweighted counts)."""
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive int")
    if not math.isfinite(successes) or not 0.0 <= successes <= n:
        raise ValueError("successes must lie in [0, n]")
    if not math.isfinite(z) or z <= 0.0:
        raise ValueError("z must be positive")
    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    margin = z * math.sqrt((phat * (1.0 - phat) + z2 / (4.0 * n)) / n) / denom
    low = center - margin
    high = center + margin
    # 0 successes and n successes have closed-form endpoints. Float subtraction
    # leaves a few ulps; snap those two cases back to the boundary.
    if successes == 0.0:
        low = 0.0
    if successes == float(n):
        high = 1.0
    return float(min(1.0, max(0.0, low))), float(min(1.0, max(0.0, high)))


def path_risk_stats(
    returns: FloatArray,
    *,
    ruin_level: float,
) -> dict[str, FloatArray | IntArray]:
    """Per-path loss, max drawdown, ruin, and recovery.

    Recovery is the step count from the max-drawdown trough back to the peak
    that was in force at that trough. Paths that never draw down are not
    recoveries and are not censored failures. Paths that draw down and do not
    get back to that peak by the horizon are censored (``recovery_steps = -1``).
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim != 2 or r.shape[1] < 1:
        raise ValueError("returns must have shape (n_paths, n_steps)")
    if not np.isfinite(r).all():
        raise ValueError("returns must be finite")
    if not math.isfinite(ruin_level):
        raise ValueError("ruin_level must be finite")
    n_paths, _n_steps = r.shape
    growth = np.cumprod(1.0 + r, axis=1)
    wealth = np.concatenate(
        [np.full((n_paths, 1), NUMERAIRE_START, dtype=np.float64), growth],
        axis=1,
    )
    peak = np.maximum.accumulate(wealth, axis=1)
    drawdown = 1.0 - wealth / peak
    max_drawdown = np.max(drawdown, axis=1)
    trough = np.argmax(drawdown, axis=1).astype(np.int64)
    peak_at = peak[np.arange(n_paths), trough]
    times = np.arange(wealth.shape[1], dtype=np.int64)
    after = times.reshape(1, -1) > trough.reshape(-1, 1)
    back = wealth >= peak_at.reshape(-1, 1)
    horizon = wealth.shape[1]
    hit = np.where(after & back, times.reshape(1, -1), horizon)
    rec_idx = np.min(hit, axis=1)
    no_drawdown = max_drawdown <= 1e-15
    recovered = (rec_idx < horizon) & ~no_drawdown
    recovery_steps = np.where(recovered, rec_idx - trough, -1).astype(np.int64)
    ruined = np.any(wealth <= float(ruin_level), axis=1)
    terminal_level = wealth[:, -1]
    loss = NUMERAIRE_START - terminal_level
    return {
        "loss": np.asarray(loss, dtype=np.float64),
        "terminal_level": np.asarray(terminal_level, dtype=np.float64),
        "max_drawdown": np.asarray(max_drawdown, dtype=np.float64),
        "ruined": np.asarray(ruined, dtype=np.uint8),
        "no_drawdown": np.asarray(no_drawdown, dtype=np.uint8),
        "recovered": np.asarray(recovered, dtype=np.uint8),
        "recovery_steps": recovery_steps,
    }


def spectral_es(losses: FloatArray, alpha: float) -> float:
    """Unweighted expected shortfall, spectral definition, positive-is-loss."""
    return float(expected_shortfall_srm(np.asarray(losses, dtype=np.float64), alpha))


def weighted_expected_shortfall(losses: FloatArray, weights: FloatArray, alpha: float) -> float:
    """Self-normalized spectral ES on a weighted empirical measure.

    Equal weights reproduce :func:`spectral_es`.
    """
    x = np.asarray(losses, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    a = float(alpha)
    if x.shape != w.shape:
        raise ValueError("losses and weights must have the same shape")
    if x.size < 5:
        raise ValueError("weighted expected shortfall needs at least 5 outcomes")
    if not np.isfinite(x).all() or not np.isfinite(w).all():
        raise ValueError("losses and weights must be finite")
    if np.any(w < 0.0):
        raise ValueError("weights must be non-negative")
    if not 0.0 < a < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("weights must have positive mass")
    order = np.argsort(x, kind="mergesort")
    ordered = x[order]
    ordered_w = w[order]
    hi = np.cumsum(ordered_w) / total
    lo = hi - ordered_w / total
    overlap = np.clip(np.minimum(hi, 1.0) - np.maximum(lo, a), 0.0, None)
    mass = float(overlap.sum())
    if mass <= 0.0:
        return float(ordered[-1])
    return float(np.dot(ordered, overlap) / mass)


def batch_size_for(alpha: float) -> int:
    """Block length so a block's tail has about five expected points, and at least 20."""
    a = float(alpha)
    if not 0.0 < a < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return max(20, int(math.ceil(5.0 / (1.0 - a))))


def batch_means_es_interval(
    losses: FloatArray,
    alpha: float,
    ci_level: float,
    *,
    weights: FloatArray | None = None,
) -> dict[str, object]:
    """Batch-means interval centered on the pooled expected-shortfall estimate.

    Contiguous blocks are treated as independent Monte Carlo replicates. That
    is a property of the scenario design (independent path blocks), not of a
    market sample. Fewer than eight usable blocks yields no interval.
    """
    x = np.asarray(losses, dtype=np.float64).ravel()
    a = float(alpha)
    block = batch_size_for(a)
    n_batches = int(x.size // block)
    point = (
        spectral_es(x, a)
        if weights is None
        else weighted_expected_shortfall(x, np.asarray(weights, dtype=np.float64), a)
    )
    if n_batches < 8:
        return {
            "estimate": point,
            "ci_low": None,
            "ci_high": None,
            "standard_error": None,
            "batch_count": n_batches,
            "batch_size": block,
            "reason": "fewer than 8 batches",
        }
    estimates: list[float] = []
    w = None if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    for i in range(n_batches):
        sl = slice(i * block, (i + 1) * block)
        if w is None:
            estimates.append(spectral_es(x[sl], a))
        else:
            if float(w[sl].sum()) <= 0.0:
                continue
            estimates.append(weighted_expected_shortfall(x[sl], w[sl], a))
    if len(estimates) < 8:
        return {
            "estimate": point,
            "ci_low": None,
            "ci_high": None,
            "standard_error": None,
            "batch_count": len(estimates),
            "batch_size": block,
            "reason": "fewer than 8 batches with positive weight",
        }
    arr = np.asarray(estimates, dtype=np.float64)
    se = float(np.std(arr, ddof=1) / math.sqrt(arr.size))
    z = normal_z(ci_level)
    return {
        "estimate": point,
        "ci_low": float(point - z * se),
        "ci_high": float(point + z * se),
        "standard_error": se,
        "batch_count": int(arr.size),
        "batch_size": block,
        "reason": None,
        "note": (
            "Centered on the pooled estimate. The half-width is the batch-means "
            "standard error of within-block expected shortfall. Simulation noise only."
        ),
    }


def gpd_var_es(
    xi: float,
    sigma: float,
    threshold: float,
    phi_u: float,
    alpha: float,
) -> tuple[float, float]:
    """GPD tail quantile and expected shortfall above a high threshold.

    ``VaR = u + sigma/xi * (((1-alpha)/phi_u)^(-xi) - 1)`` for ``xi != 0``.
    ES uses the mean-excess identity and is undefined for ``xi >= 1``.
    """
    for name, val in (
        ("xi", xi),
        ("sigma", sigma),
        ("threshold", threshold),
        ("phi_u", phi_u),
        ("alpha", alpha),
    ):
        if not math.isfinite(val):
            raise ValueError(f"{name} must be finite")
    if sigma <= 0.0 or not 0.0 < phi_u <= 1.0 or not 0.0 < alpha < 1.0:
        raise ValueError("sigma > 0, 0 < phi_u <= 1, and 0 < alpha < 1 are required")
    if (1.0 - alpha) >= phi_u:
        raise ValueError("alpha must exceed the threshold coverage 1 - phi_u")
    ratio = (1.0 - alpha) / phi_u
    if xi == 0.0:
        var = threshold - sigma * math.log(ratio)
    else:
        var = threshold + (sigma / xi) * (ratio ** (-xi) - 1.0)
    if xi < 1.0:
        es = (var + sigma - xi * threshold) / (1.0 - xi)
    else:
        es = float("nan")
    return float(var), float(es)


def _observed_information_se(
    excess: FloatArray, xi: float, sigma: float
) -> tuple[float | None, float | None]:
    """Standard errors from a numerical Hessian of the GPD negative log-likelihood.

    Returns ``(None, None)`` when the observed information is not positive definite.
    The errors assume i.i.d. excesses; the report says so.
    """

    def nll(theta: NDArray[np.float64]) -> float:
        shape = float(theta[0])
        scale = float(math.exp(float(theta[1])))
        if scale <= 0.0 or not math.isfinite(scale):
            return 1e12
        if shape < 0.0:
            upper = -scale / shape
            if float(np.max(excess)) >= upper:
                return 1e12
        val = sstats.genpareto.nnlf((shape, 0.0, scale), excess)
        if not np.isfinite(val):
            return 1e12
        return float(val)

    theta = np.array([xi, math.log(sigma)], dtype=np.float64)
    step = 1e-4
    hess = np.zeros((2, 2), dtype=np.float64)
    for i in range(2):
        for j in range(i, 2):
            step_i = np.zeros(2, dtype=np.float64)
            step_j = np.zeros(2, dtype=np.float64)
            step_i[i] = step
            step_j[j] = step
            val = (
                nll(theta + step_i + step_j)
                - nll(theta + step_i - step_j)
                - nll(theta - step_i + step_j)
                + nll(theta - step_i - step_j)
            ) / (4.0 * step * step)
            hess[i, j] = val
            hess[j, i] = val
    if not np.isfinite(hess).all():
        return None, None
    sign, logdet = np.linalg.slogdet(hess)
    if sign <= 0.0 or not np.isfinite(logdet):
        return None, None
    cov = np.linalg.inv(hess)
    if np.any(np.diag(cov) <= 0.0) or not np.isfinite(cov).all():
        return None, None
    se_xi = float(math.sqrt(float(cov[0, 0])))
    se_sigma = float(sigma * math.sqrt(float(cov[1, 1])))
    return se_xi, se_sigma


def _fit_excess(excess: FloatArray) -> tuple[float, float] | None:
    if excess.size < 20:
        return None
    try:
        xi, _loc, sigma = sstats.genpareto.fit(excess, floc=0.0)
    except (RuntimeError, ValueError, FloatingPointError):
        return None
    if not np.isfinite(xi) or not np.isfinite(sigma) or float(sigma) <= 0.0:
        return None
    return float(xi), float(sigma)


def pot_gpd(
    losses: FloatArray,
    *,
    threshold: float | None = None,
    threshold_quantile: float = 0.95,
) -> dict[str, object]:
    """Peaks-over-threshold GPD fit with diagnostics that can fail closed.

    ``diagnostics_ok`` requires at least 50 exceedances, a finite positive
    scale, shape below 1 (so ES is finite), and mean-excess relative error
    at most 0.25. The KS p-value is reported and is not a pass/fail gate:
    at large n it rejects trivial discrepancies. Shape stability across
    nearby thresholds is reported as a range, with a warning above 0.5.
    """
    v = np.asarray(losses, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size < 20:
        return {
            "available": False,
            "reason": "fewer than 20 finite losses",
            "diagnostics_ok": False,
            "diagnostic_failures": ["fewer than 20 finite losses"],
        }
    if threshold is None:
        if not 0.5 < threshold_quantile < 1.0:
            raise ValueError("threshold_quantile must be in (0.5, 1)")
        u = float(np.quantile(v, threshold_quantile))
        rule = f"quantile_{threshold_quantile}"
    else:
        u = float(threshold)
        rule = "absolute"
        if not math.isfinite(u):
            raise ValueError("threshold must be finite")
    excess = v[v > u] - u
    n_exceed = int(excess.size)
    if n_exceed < 20:
        return {
            "available": False,
            "reason": "fewer than 20 exceedances",
            "threshold": u,
            "threshold_rule": rule,
            "n_exceedances": n_exceed,
            "diagnostics_ok": False,
            "diagnostic_failures": ["fewer than 20 exceedances"],
        }
    fitted = _fit_excess(excess)
    if fitted is None:
        return {
            "available": False,
            "reason": "gpd fit failed",
            "threshold": u,
            "threshold_rule": rule,
            "n_exceedances": n_exceed,
            "diagnostics_ok": False,
            "diagnostic_failures": ["gpd fit failed"],
        }
    xi, sigma = fitted
    phi_u = float(n_exceed / v.size)
    mean_excess = float(excess.mean())
    failures: list[str] = []
    warnings: list[str] = []
    if n_exceed < 50:
        failures.append("fewer than 50 exceedances")
    es_finite = xi < 1.0
    if not es_finite:
        failures.append("shape >= 1 so expected shortfall is not finite")
    if xi >= 1.0:
        model_mean: float | None = None
        rel_err: float | None = None
    else:
        model_mean = float(sigma / (1.0 - xi))
        rel_err = abs(mean_excess - model_mean) / max(abs(model_mean), 1e-12)
        if rel_err > 0.25:
            failures.append("mean excess relative error above 0.25")
    se_xi, se_sigma = _observed_information_se(excess, xi, sigma)
    if se_xi is None:
        warnings.append("observed information is not positive definite; standard errors omitted")
    ks_stat, ks_p = sstats.kstest(excess, "genpareto", args=(xi, 0.0, sigma))
    stability: dict[str, float | None] = {}
    if threshold is None:
        shapes: list[float] = []
        for q in (0.90, 0.95, 0.97):
            u_q = float(np.quantile(v, q))
            fit_q = _fit_excess(v[v > u_q] - u_q)
            stability[f"{q}"] = None if fit_q is None else fit_q[0]
            if fit_q is not None:
                shapes.append(fit_q[0])
        if len(shapes) >= 2 and max(shapes) - min(shapes) > 0.5:
            warnings.append("shape moves by more than 0.5 across thresholds 0.90, 0.95, 0.97")
    tail: dict[str, float | None] = {}
    for alpha in (0.975, 0.99):
        try:
            var_a, es_a = gpd_var_es(xi, sigma, u, phi_u, alpha)
        except ValueError:
            tail[f"var_{alpha}"] = None
            tail[f"es_{alpha}"] = None
        else:
            tail[f"var_{alpha}"] = var_a
            tail[f"es_{alpha}"] = None if not math.isfinite(es_a) else es_a
    return _gpd_payload(
        threshold=u,
        rule=rule,
        n_total=int(v.size),
        n_exceed=n_exceed,
        phi_u=phi_u,
        xi=xi,
        sigma=sigma,
        se_xi=se_xi,
        se_sigma=se_sigma,
        mean_excess=mean_excess,
        model_mean=model_mean,
        rel_err=rel_err,
        ks_stat=float(ks_stat),
        ks_p=float(ks_p),
        stability=stability or None,
        es_finite=es_finite,
        failures=failures,
        warnings=warnings,
        tail=tail,
    )


def pot_from_exceedances(
    exceeding_losses: FloatArray,
    threshold: float,
    n_total: int,
) -> dict[str, object]:
    """GPD fit when workers retained only the losses strictly above ``threshold``.

    Threshold-stability across quantiles needs the body of the sample, so it is
    omitted. ``phi_u = n_exceed / n_total``.
    """
    if isinstance(n_total, bool) or not isinstance(n_total, int) or n_total <= 0:
        raise ValueError("n_total must be a positive int")
    u = float(threshold)
    if not math.isfinite(u):
        raise ValueError("threshold must be finite")
    raw = np.asarray(exceeding_losses, dtype=np.float64).ravel()
    raw = raw[np.isfinite(raw)]
    excess = raw[raw > u] - u
    n_exceed = int(excess.size)
    if n_exceed < 20:
        return {
            "available": False,
            "reason": "fewer than 20 exceedances",
            "threshold": u,
            "threshold_rule": "absolute",
            "n_exceedances": n_exceed,
            "diagnostics_ok": False,
            "diagnostic_failures": ["fewer than 20 exceedances"],
        }
    fitted = _fit_excess(excess)
    if fitted is None:
        return {
            "available": False,
            "reason": "gpd fit failed",
            "threshold": u,
            "threshold_rule": "absolute",
            "n_exceedances": n_exceed,
            "diagnostics_ok": False,
            "diagnostic_failures": ["gpd fit failed"],
        }
    xi, sigma = fitted
    mean_excess = float(excess.mean())
    failures: list[str] = []
    warnings: list[str] = ["threshold stability was not computed from a retained body sample"]
    if n_exceed < 50:
        failures.append("fewer than 50 exceedances")
    es_finite = xi < 1.0
    if not es_finite:
        failures.append("shape >= 1 so expected shortfall is not finite")
        model_mean: float | None = None
        rel_err: float | None = None
    else:
        model_mean = float(sigma / (1.0 - xi))
        rel_err = abs(mean_excess - model_mean) / max(abs(model_mean), 1e-12)
        if rel_err > 0.25:
            failures.append("mean excess relative error above 0.25")
    se_xi, se_sigma = _observed_information_se(excess, xi, sigma)
    if se_xi is None:
        warnings.append("observed information is not positive definite; standard errors omitted")
    ks_stat, ks_p = sstats.kstest(excess, "genpareto", args=(xi, 0.0, sigma))
    phi_u = float(n_exceed / n_total)
    tail: dict[str, float | None] = {}
    for alpha in (0.975, 0.99):
        try:
            var_a, es_a = gpd_var_es(xi, sigma, u, phi_u, alpha)
        except ValueError:
            tail[f"var_{alpha}"] = None
            tail[f"es_{alpha}"] = None
        else:
            tail[f"var_{alpha}"] = var_a
            tail[f"es_{alpha}"] = None if not math.isfinite(es_a) else es_a
    return _gpd_payload(
        threshold=u,
        rule="absolute",
        n_total=n_total,
        n_exceed=n_exceed,
        phi_u=phi_u,
        xi=xi,
        sigma=sigma,
        se_xi=se_xi,
        se_sigma=se_sigma,
        mean_excess=mean_excess,
        model_mean=model_mean,
        rel_err=rel_err,
        ks_stat=float(ks_stat),
        ks_p=float(ks_p),
        stability=None,
        es_finite=es_finite,
        failures=failures,
        warnings=warnings,
        tail=tail,
    )


def _gpd_payload(
    *,
    threshold: float,
    rule: str,
    n_total: int,
    n_exceed: int,
    phi_u: float,
    xi: float,
    sigma: float,
    se_xi: float | None,
    se_sigma: float | None,
    mean_excess: float,
    model_mean: float | None,
    rel_err: float | None,
    ks_stat: float,
    ks_p: float,
    stability: dict[str, float | None] | None,
    es_finite: bool,
    failures: list[str],
    warnings: list[str],
    tail: dict[str, float | None],
) -> dict[str, object]:
    return {
        "available": True,
        "threshold": threshold,
        "threshold_rule": rule,
        "n": n_total,
        "n_exceedances": n_exceed,
        "phi_u": phi_u,
        "xi": xi,
        "sigma": sigma,
        "xi_se": se_xi,
        "sigma_se": se_sigma,
        "standard_errors_assume_iid": True,
        "mean_excess_empirical": mean_excess,
        "mean_excess_model": model_mean,
        "mean_excess_relative_error": rel_err,
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_p,
        "ks_pvalue_reported_not_gated": True,
        "xi_at_quantiles": stability,
        "es_finite": es_finite,
        "tail": tail,
        "diagnostics_ok": len(failures) == 0,
        "diagnostic_failures": failures,
        "diagnostic_warnings": warnings,
    }
