"""Smooth transition autoregression (LSTAR / ESTAR).

Terasvirta (1994) STR(p) with logistic or exponential transition:

    y_t = phi0' z_t + (phi1' z_t) * G(gamma, c; s_t) + e_t,
    z_t = (1, y_{t-1}, ..., y_{t-p})',  s_t = y_{t-d} (delay d).

G_logistic = (1 + exp(-gamma (s - c)))^{-1};  G_estar = 1 - exp(-gamma (s-c)^2).

``star_fit`` estimates by nonlinear least squares over (phi0, phi1, gamma, c)
with a small grid on (gamma, c) for a robust start. ``star_forecast``
iterates one-step conditional means. ``star_linearity`` runs the
Luukkonen-Saikkonen-Terasvirta linearity F-test (auxiliary regression of
residuals on z_t * s_t^k, k=1..3 approximated by a first-order expansion).

Fail-closed: p >= 1, d in [1, p], finite input, T > p + params.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.optimize import OptimizeResult, minimize

Array = NDArray[np.float64]


def _check(y: Array, p: int, d: int) -> Array:
    yy = np.asarray(y, dtype=float).ravel()
    if yy.size < p + 30 or not np.isfinite(yy).all():
        raise ValueError(f"y must be finite with T >= {p + 30}")
    if p < 1 or d < 1 or d > p:
        raise ValueError("need 1 <= d <= p")
    return yy


def _zmat(y: Array, p: int) -> tuple[Array, Array]:
    t = y.size
    z = np.column_stack([np.ones(t - p)] + [y[p - lag_i - 1 : t - lag_i - 1] for lag_i in range(p)])
    return z, y[p:]


def _g(s: Array, gamma: float, c: float, kind: str) -> Array:
    if kind == "lstar":
        z = np.clip(gamma * (s - c), -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-z))
    return 1.0 - np.exp(-gamma * np.clip((s - c) ** 2, 0, 1e12))


def star_fit(
    y: Array,
    p: int = 2,
    d: int = 1,
    kind: str = "lstar",
) -> dict[str, Array | float | str]:
    """NLS fit of STR(p) with grid start on (gamma, c)."""
    yy = _check(y, p, d)
    if kind not in ("lstar", "estar"):
        raise ValueError("kind must be lstar or estar")
    z, dep = _zmat(yy, p)
    s = yy[p - d : yy.size - d]  # transition variable = y_{t-d}
    scale = float(np.std(s))

    def _sse(theta: Array) -> float:
        gamma = np.exp(theta[0])
        c = theta[1]
        coefs = theta[2:]
        n_coef = z.shape[1]
        phi0 = coefs[:n_coef]
        phi1 = coefs[n_coef:]
        g = _g(s, gamma, c, kind)
        resid = dep - z @ phi0 - (z @ phi1) * g
        return float(resid @ resid)

    n_coef = z.shape[1]
    phi_ols = np.linalg.lstsq(z, dep, rcond=None)[0]
    best: OptimizeResult | None = None
    for g0 in (1.0, 5.0, 20.0):
        for c0 in (np.quantile(s, 0.25), np.quantile(s, 0.5), np.quantile(s, 0.75)):
            theta0 = np.concatenate(
                [[np.log(g0 / max(scale, 1e-9)), c0], phi_ols, np.zeros(n_coef)]
            )
            res = minimize(_sse, theta0, method="BFGS", options={"maxiter": 400})
            if best is None or res.fun < best.fun:
                best = res
    assert best is not None  # grid always runs at least one fit
    th = best.x
    gamma = float(np.exp(th[0]))
    c = float(th[1])
    n_coef = z.shape[1]
    phi0 = th[2 : 2 + n_coef]
    phi1 = th[2 + n_coef :]
    g = _g(s, gamma, c, kind)
    resid = dep - z @ phi0 - (z @ phi1) * g
    return {
        "kind": kind,
        "gamma": gamma,
        "c": c,
        "phi0": phi0,
        "phi1": phi1,
        "resid": resid,
        "g": g,
        "sse": float(resid @ resid),
        "d": float(d),
        "p": float(p),
        "converged": bool(best.success),
    }


def star_forecast(
    fit: dict[str, Array | float | str],
    history: Array,
    n_steps: int,
) -> Array:
    """Iterate the fitted STR one-step-ahead conditional mean."""
    hist = np.asarray(history, dtype=float).ravel()
    p = int(fit["p"])
    d = int(fit["d"])
    if hist.size < p or not np.isfinite(hist).all():
        raise ValueError(f"history must be finite with >= {p} values")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    phi0 = np.asarray(fit["phi0"], dtype=float)
    phi1 = np.asarray(fit["phi1"], dtype=float)
    gamma = float(fit["gamma"])
    c = float(fit["c"])
    kind = str(fit["kind"])
    work = hist.tolist()
    out = np.zeros(n_steps)
    for h in range(n_steps):
        z = np.array([1.0] + [work[-1 - lag_i] for lag_i in range(p)])
        s = work[-d]
        out[h] = float(z @ phi0 + (z @ phi1) * _g(np.asarray(s), gamma, c, kind))
        work.append(float(out[h]))
    return out


def star_linearity(y: Array, p: int = 2, d: int = 1) -> dict[str, float]:
    """LST linearity test: aux regression residuals ~ z + z*s^k terms.

    Reports the F-stat and p-value; reject linearity when small.
    """
    yy = _check(y, p, d)
    z, dep = _zmat(yy, p)
    s = yy[p - d : yy.size - d]
    resid = dep - z @ np.linalg.lstsq(z, dep, rcond=None)[0]
    n_obs, k = z.shape
    aux = np.column_stack([z] + [z * (s**pow_)[:, None] for pow_ in (1, 2, 3)])
    n_rest = aux.shape[1] - k
    beta = np.linalg.lstsq(aux, resid, rcond=None)[0]
    res2 = resid - aux @ beta
    rss_r = float(resid @ resid)
    rss_u = float(res2 @ res2)
    dof2 = n_obs - aux.shape[1]
    if rss_u <= 0 or dof2 <= 0:
        return {"f": float("nan"), "p": float("nan"), "dof1": float(n_rest), "dof2": float(dof2)}
    f_stat = ((rss_r - rss_u) / n_rest) / (rss_u / dof2)
    p_val = float(stats.f.sf(f_stat, n_rest, dof2))
    return {"f": float(f_stat), "p": p_val, "dof1": float(n_rest), "dof2": float(dof2)}
