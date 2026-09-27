"""Sparse regression: LARS path, orthogonal matching pursuit, adaptive
LASSO (Zou 2006), and EBIC selection (Chen & Chen 2008).

- ``lars_path``: least-angle regression with the lasso modification —
  piecewise-linear coefficient path in lambda.
- ``omp``: greedy orthogonal matching pursuit with LS refit.
- ``lasso_cd``: cyclic coordinate descent for the lasso.
- ``adaptive_lasso``: OLS-weighted l1 via ``lasso_cd`` on scaled columns.
- ``ebic_select``: pick the path point minimizing extended BIC.

Fail-closed: non-finite input, degenerate columns, invalid sizes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _prep(x: Array, y: Array) -> tuple[Array, Array, Array, Array]:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float).ravel()
    if xx.ndim != 2 or yy.shape[0] != xx.shape[0]:
        raise ValueError("y and X must have matching rows")
    n, p = xx.shape
    if n < 10 or p < 1 or not np.isfinite(xx).all() or not np.isfinite(yy).all():
        raise ValueError("finite inputs, n >= 10")
    mx = xx.mean(axis=0)
    my = float(yy.mean())
    xc = xx - mx
    yc = yy - my
    norms = np.linalg.norm(xc, axis=0)
    if (norms <= 0).any():
        raise ValueError("constant column in design")
    xc = xc / norms
    return xc, yc, mx, np.asarray(my)


def lars_path(x: Array, y: Array, max_steps: int | None = None) -> dict[str, Array]:
    """LARS-Lasso coefficient path (Efron et al. 2004, Alg. 3.2a).

    Returns ``beta_path`` (p, n_steps), ``lambdas`` (n_steps,) — the
    L1-norm of each path point — and ``active_sets`` as a padded int
    matrix (-1 padded).
    """
    xc, yc, _, _ = _prep(x, y)
    n, p = xc.shape
    if max_steps is None:
        max_steps = min(n - 1, p) + p
    active: list[int] = []
    beta = np.zeros(p)
    path: list[Array] = []
    lambdas: list[float] = []
    acts: list[tuple[int, ...]] = []

    resid = yc.copy()
    steps = 0
    while steps < max_steps and len(active) < min(n - 1, p):
        c = xc.T @ resid
        c_abs = np.abs(c)
        c_max = c_abs.max() if c_abs.size else 0.0
        if c_max < 1e-12:
            break
        if not active:
            active.append(int(c_abs.argmax()))
        a_idx = np.asarray(active)
        s = np.sign(c[a_idx])
        xa = xc[:, a_idx] * s
        g = xa.T @ xa
        try:
            g_inv = np.linalg.inv(g)
        except np.linalg.LinAlgError:
            break
        one = np.ones(len(active))
        aa = 1.0 / np.sqrt(max(float(one @ g_inv @ one), 1e-18))
        w = aa * (g_inv @ one)
        u = xa @ w
        a_vec = xc.T @ u
        # step length to next join — track which inactive column wins
        gam = np.inf
        join_j = -1
        for j in range(p):
            if j in active:
                continue
            d1 = (c_max - c[j]) / (aa - a_vec[j]) if abs(aa - a_vec[j]) > 1e-14 else np.inf
            d2 = (c_max + c[j]) / (aa + a_vec[j]) if abs(aa + a_vec[j]) > 1e-14 else np.inf
            pos = [v for v in (d1, d2) if v > 1e-12]
            cand = min(pos) if pos else np.inf
            if cand < gam:
                gam = cand
                join_j = j
        if not np.isfinite(gam):
            gam = c_max / aa
            join_j = -1
        # lasso drop check
        d = w * s
        gam_drop = np.inf
        drop_j = -1
        for i, j in enumerate(active):
            if abs(d[i]) > 1e-14:
                cand = -beta[j] / d[i]
                if cand > 1e-12 and cand < gam_drop:
                    gam_drop, drop_j = cand, j
        if gam_drop < gam:
            gam = gam_drop
        # update
        beta[a_idx] += gam * d
        resid = yc - xc @ beta
        if gam_drop < np.inf and drop_j >= 0 and abs(gam - gam_drop) < 1e-9:
            active.remove(drop_j)
        elif join_j >= 0 and gam_drop >= gam:
            active.append(join_j)
        path.append(beta.copy())
        lambdas.append(float(np.abs(beta).sum()))
        acts.append(tuple(active))
        steps += 1
    # final OLS point on the active set
    if active:
        a_idx = np.asarray(active)
        coef, *_ = np.linalg.lstsq(xc[:, a_idx], yc, rcond=None)
        beta[a_idx] = coef
        path.append(beta.copy())
        lambdas.append(float(np.abs(beta).sum()))
        acts.append(tuple(active))
    max_a = max((len(a) for a in acts), default=0)
    act_mat = np.full((len(acts), max_a if max_a else 1), -1, dtype=int)
    for i, a in enumerate(acts):
        act_mat[i, : len(a)] = a
    return {
        "beta_path": np.asarray(path).T if path else np.zeros((p, 0)),
        "lambdas": np.asarray(lambdas),
        "active_sets": act_mat,
    }


def omp(x: Array, y: Array, k: int) -> dict[str, Array | float]:
    """Orthogonal matching pursuit: greedily pick k columns by residual
    correlation, LS-refit each step (Pati et al. 1993)."""
    xc, yc, _, _ = _prep(x, y)
    n, p = xc.shape
    if k < 1 or k > min(p, n - 1):
        raise ValueError("k must be in [1, min(p, n-1)]")
    resid = yc.copy()
    chosen: list[int] = []
    sse = float(resid @ resid)
    for _ in range(k):
        c = np.abs(xc.T @ resid)
        c[list(chosen)] = -1.0
        j = int(c.argmax())
        chosen.append(j)
        coef, *_ = np.linalg.lstsq(xc[:, chosen], yc, rcond=None)
        resid = yc - xc[:, chosen] @ coef
        sse = float(resid @ resid)
        if sse < 1e-18:
            break
    beta = np.zeros(p)
    coef, *_ = np.linalg.lstsq(xc[:, chosen], yc, rcond=None)
    beta[np.asarray(chosen)] = coef
    return {
        "beta": beta,
        "selected": np.asarray(chosen, dtype=np.float64),
        "sse": sse,
        "k": float(len(chosen)),
    }


def _soft(v: float, lam: float) -> float:
    return float(np.sign(v) * max(abs(v) - lam, 0.0))


def lasso_cd(
    x: Array,
    y: Array,
    lam: float,
    weights: Array | None = None,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> dict[str, Array | float]:
    """Cyclic coordinate descent for the lasso (Friedman et al. 2007).

    Solves min 0.5||y - Xb||^2 + lam sum_j w_j |b_j| on standardized X.
    """
    xc, yc, _, _ = _prep(x, y)
    n, p = xc.shape
    if not np.isfinite(lam) or lam < 0.0:
        raise ValueError("lam must be >= 0")
    w = np.ones(p) if weights is None else np.asarray(weights, dtype=float).ravel()
    if w.shape != (p,) or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError("weights must be finite, > 0, length p")
    beta = np.zeros(p)
    x2 = (xc * xc).sum(axis=0)
    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            r = yc - xc @ beta + xc[:, j] * beta[j]
            beta[j] = _soft(float(xc[:, j] @ r), lam * w[j]) / x2[j]
        if np.abs(beta - beta_old).max() < tol:
            break
    resid = yc - xc @ beta
    return {
        "beta": beta,
        "sse": float(resid @ resid),
        "n_iter": float(_),
        "lam": float(lam),
        "n_nonzero": float((np.abs(beta) > 1e-10).sum()),
    }


def adaptive_lasso(
    x: Array,
    y: Array,
    lam: float,
    power: float = 1.0,
) -> dict[str, Array | float]:
    """Zou (2006) adaptive lasso: weights w_j = 1/|b_ols_j|^power, then
    weighted lasso via coordinate descent."""
    xc, yc, _, _ = _prep(x, y)
    n, p = xc.shape
    if p >= n:
        raise ValueError("adaptive lasso needs n > p for the OLS pilot")
    if not np.isfinite(power) or power <= 0.0:
        raise ValueError("power must be > 0")
    b_ols, *_ = np.linalg.lstsq(xc, yc, rcond=None)
    w = 1.0 / np.maximum(np.abs(b_ols), 1e-8) ** power
    return lasso_cd(x, y, lam, weights=w)


def ebic_select(x: Array, y: Array, ebic_gamma: float = 0.5) -> dict[str, Array | float]:
    """Pick the LARS-path point minimizing EBIC (Chen & Chen 2008):

    EBIC = n ln(SSE/n) + k ln n + 2 gamma k ln p.
    """
    xc, yc, _, _ = _prep(x, y)
    n, p = xc.shape
    if not np.isfinite(ebic_gamma) or not 0.0 <= ebic_gamma <= 1.0:
        raise ValueError("ebic_gamma in [0,1]")
    path = lars_path(x, y)
    bp = np.asarray(path["beta_path"])
    if bp.shape[1] == 0:
        raise ValueError("empty LARS path")
    scores = np.empty(bp.shape[1])
    for i in range(bp.shape[1]):
        b = bp[:, i]
        resid = yc - xc @ b
        sse = max(float(resid @ resid), 1e-18)
        k = int((np.abs(b) > 1e-10).sum())
        scores[i] = n * np.log(sse / n) + k * np.log(n) + 2.0 * ebic_gamma * k * np.log(p)
    best = int(np.argmin(scores))
    return {
        "beta": bp[:, best],
        "ebic": float(scores[best]),
        "step": float(best),
        "scores": scores,
        "n_selected": float((np.abs(bp[:, best]) > 1e-10).sum()),
    }
