"""Survival analysis for event-duration modeling (e.g., time-to-exit,
drawdown spells, order-fill times, position lifetimes).

References:
- Kaplan & Meier (1958): product-limit estimator.
- Nelson (1972) / Aalen (1978): cumulative hazard estimator.
- Mantel (1966): log-rank test.
- Cox (1972): proportional-hazards partial likelihood (Breslow ties).
- Greenwood (1926): variance of the KM estimator.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import optimize as opt
from scipy import stats

Array = NDArray[np.float64]


def _td(t: Array, d: Array, n: int = 10) -> tuple[Array, Array]:
    tv = np.asarray(t, dtype=float).reshape(-1)
    dv = np.asarray(d, dtype=float).reshape(-1)
    if tv.size != dv.size or tv.size < n:
        raise ValueError(f"durations/events must share length >= {n}")
    if not (np.all(np.isfinite(tv)) and np.all(np.isfinite(dv))):
        raise ValueError("inputs must be finite")
    if np.any(tv <= 0):
        raise ValueError("durations must be positive")
    if not np.all((dv == 0.0) | (dv == 1.0)):
        raise ValueError("event indicators must be binary")
    if dv.sum() < 1:
        raise ValueError("no events observed")
    return tv, dv


def kaplan_meier(durations: Array, events: Array) -> dict[str, Array]:
    """Kaplan–Meier product-limit survival curve with Greenwood SE."""
    t, d = _td(durations, events, n=5)
    times = np.unique(t[d == 1])
    if times.size == 0:
        raise ValueError("no event times")
    S = 1.0
    surv = np.empty(times.size)
    se = np.empty(times.size)
    greenwood = 0.0
    for i, tt in enumerate(times):
        at_risk = float(np.sum(t >= tt))
        d_i = float(np.sum((t == tt) & (d == 1)))
        if at_risk <= 0:
            raise ValueError("empty risk set")
        S *= 1.0 - d_i / at_risk
        surv[i] = S
        if at_risk - d_i > 0:
            greenwood += d_i / (at_risk * (at_risk - d_i))
        se[i] = S * math.sqrt(max(greenwood, 0.0))
    return {"times": times, "survival": surv, "se": se}


def nelson_aalen(durations: Array, events: Array) -> dict[str, Array]:
    """Nelson–Aalen cumulative hazard: H(t) = sum_{t_i <= t} d_i / Y_i."""
    t, d = _td(durations, events, n=5)
    times = np.unique(t[d == 1])
    H = np.empty(times.size)
    var = np.empty(times.size)
    acc = 0.0
    vacc = 0.0
    for i, tt in enumerate(times):
        at_risk = float(np.sum(t >= tt))
        d_i = float(np.sum((t == tt) & (d == 1)))
        acc += d_i / at_risk
        vacc += d_i / (at_risk * at_risk)
        H[i] = acc
        var[i] = vacc
    return {"times": times, "hazard": H, "se": np.sqrt(var)}


def log_rank_test(t1: Array, d1: Array, t2: Array, d2: Array) -> dict[str, float]:
    """Mantel (1966) log-rank test comparing two survival curves.

    ``Z = (O1 - E1) / sqrt(V)`` over pooled event times."""
    a_t, a_d = _td(t1, d1, n=5)
    b_t, b_d = _td(t2, d2, n=5)
    t = np.concatenate([a_t, b_t])
    d = np.concatenate([a_d, b_d])
    g = np.concatenate([np.zeros(a_t.size), np.ones(b_t.size)])
    times = np.unique(t[d == 1])
    O1 = E1 = V = 0.0
    for tt in times:
        risk = t >= tt
        n_r = float(risk.sum())
        n1 = float(np.sum(risk & (g == 0)))
        d_i = float(np.sum((t == tt) & (d == 1)))
        d1_i = float(np.sum((t == tt) & (d == 1) & (g == 0)))
        if n_r < 2:
            continue
        O1 += d1_i
        E1 += d_i * n1 / n_r
        V += n1 * (n_r - n1) * d_i * (n_r - d_i) / (n_r * n_r * max(n_r - 1.0, 1e-9))
    if V <= 0:
        raise ValueError("degenerate log-rank variance")
    z = float((O1 - E1) / math.sqrt(V))
    return {
        "statistic": z,
        "pvalue": float(2.0 * (1.0 - stats.norm.cdf(abs(z)))),
        "observed": O1,
        "expected": E1,
        "n_events": float(np.sum(d == 1)),
    }


def fit_cox_ph(X: Array, durations: Array, events: Array) -> dict[str, Array]:
    """Cox (1972) proportional hazards via Breslow partial likelihood.

    Returns coefficients, hazard ratios, robust-ish SEs from the inverse
    information matrix, and the concordance index."""
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A[:, None]
    t, d = _td(durations, events)
    if A.ndim != 2 or A.shape[0] != t.size or not np.all(np.isfinite(A)):
        raise ValueError("X must be finite (n, k) matching durations")
    n, k = A.shape
    if np.any(A.std(axis=0) == 0):
        raise ValueError("constant covariate column")
    event_times = np.unique(t[d == 1])

    def nll_grad(beta: Array) -> tuple[float, Array, Array]:
        xb = A @ beta
        exb = np.exp(np.clip(xb - xb.max(), -50, 50))
        ll = 0.0
        grad = np.zeros(k)
        info = np.zeros((k, k))
        for tt in event_times:
            events_at = (t == tt) & (d == 1)
            risk = t >= tt
            s0 = float(exb[risk].sum())
            s1 = (A[risk] * exb[risk][:, None]).sum(axis=0)
            xb_r = A[risk] * exb[risk][:, None]
            s2 = xb_r.T @ A[risk]
            for idx in np.where(events_at)[0]:
                ll += xb[idx] - math.log(max(s0, 1e-300))
                grad += A[idx] - s1 / s0
                E2 = s2 / s0 - np.outer(s1, s1) / (s0 * s0)
                info += E2
        return -ll, -grad, info

    res = opt.minimize(
        lambda b: nll_grad(b)[:2],
        np.zeros(k),
        jac=True,
        method="BFGS",
        options={"maxiter": 200},
    )
    if not np.isfinite(res.fun):
        raise ValueError("Cox PH optimization failed")
    beta = np.asarray(res.x, dtype=float)
    _, _, info = nll_grad(beta)
    try:
        cov = np.linalg.inv(info)
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)
    # Harrell's concordance on linear predictor.
    xb = A @ beta
    conc = 0.0
    tot = 0.0
    ev = np.where(d == 1)[0]
    for i in ev:
        for j in range(n):
            if t[j] > t[i]:
                tot += 1.0
                conc += 0.5 if xb[i] == xb[j] else float(xb[i] > xb[j])
    cidx = conc / tot if tot > 0 else np.nan
    return {
        "beta": beta,
        "hazard_ratio": np.exp(np.clip(beta, -20, 20)),
        "se": se,
        "loglik": np.array([-res.fun]),
        "concordance": np.array([cidx]),
        "converged": np.array([float(res.success)]),
    }
