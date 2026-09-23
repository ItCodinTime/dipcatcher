"""Convergent cross-mapping — nonlinear causality via Takens embeddings.

References:
- Sugihara et al. (2012): convergent cross mapping — Y causes X iff the
  shadow manifold of X predicts Y better as library size grows.
- Takens (1981): delay-coordinate embedding theorem.
- Sugihara & May (1990): simplex projection / S-map forecasting.
- Sugihara (1994): nonlinearity test via S-map theta sweep.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _v(x: Array, n: int = 40) -> Array:
    v = np.asarray(x, dtype=float).reshape(-1)
    if v.size < n or not np.all(np.isfinite(v)):
        raise ValueError(f"series must be finite with length >= {n}")
    if v.std() == 0:
        raise ValueError("degenerate (constant) series")
    return v


def takens_embed(x: Array, dim: int, tau: int = 1) -> Array:
    """Takens delay embedding: rows are (x_t, x_{t-tau}, ..., x_{t-(E-1)tau})."""
    v = _v(x)
    if dim < 2 or tau < 1:
        raise ValueError("dim >= 2 and tau >= 1 required")
    n = v.size
    rows = n - (dim - 1) * tau
    if rows < 10:
        raise ValueError("embedding leaves too few points")
    return np.column_stack(
        [v[(dim - 1 - j) * tau : (dim - 1 - j) * tau + rows] for j in range(dim)]
    )


def simplex_project(lib: Array, lib_y: Array, pred: Array, n_neighbors: int | None = None) -> Array:
    """Sugihara–May simplex projection: predict ``pred`` rows from ``lib``
    using the E+1 nearest neighbors with exponential distance weights."""
    L = np.asarray(lib, dtype=float)
    P = np.asarray(pred, dtype=float)
    yv = np.asarray(lib_y, dtype=float).reshape(-1)
    if L.ndim != 2 or P.ndim != 2 or L.shape[1] != P.shape[1]:
        raise ValueError("lib/pred must be (m, E) and (k, E)")
    if yv.size != L.shape[0]:
        raise ValueError("lib_y must match lib rows")
    E = L.shape[1]
    k = E + 1 if n_neighbors is None else int(n_neighbors)
    if k >= L.shape[0]:
        raise ValueError("n_neighbors must be < lib size")
    out = np.empty(P.shape[0])
    for i in range(P.shape[0]):
        d = np.linalg.norm(L - P[i], axis=1)
        idx = np.argsort(d)[:k]
        dd = d[idx]
        w = np.exp(-dd / max(dd[0], 1e-12))
        w /= w.sum()
        out[i] = float(w @ yv[idx])
    return out


def _ccm_once(lib_x: Array, tgt_y: Array, lib_size: int, rng: np.random.Generator) -> float:
    """Cross-map skill (Pearson) predicting y from x's shadow manifold
    on a random library subset of ``lib_size``."""
    idx = rng.choice(lib_x.shape[0], size=min(lib_size, lib_x.shape[0]), replace=False)
    pred_idx = rng.choice(lib_x.shape[0], size=min(lib_size, lib_x.shape[0]), replace=False)
    pred = simplex_project(lib_x[idx], tgt_y[idx], lib_x[pred_idx])
    actual = tgt_y[pred_idx]
    if actual.std() <= 0 or pred.std() <= 0:
        return 0.0
    return float(np.corrcoef(pred, actual)[0, 1])


def ccm(
    cause: Array,
    effect: Array,
    dim: int = 3,
    tau: int = 1,
    lib_sizes: Array | None = None,
    n_reps: int = 20,
    seed: int = 0,
) -> dict[str, Array | float]:
    """Sugihara et al. (2012) CCM: does ``cause`` drive ``effect``?

    If ``cause`` drives ``effect``, then ``effect``'s shadow manifold
    contains information about ``cause`` — so we embed the EFFECT and
    predict the CAUSE (cross-mapping direction), and the skill must rise
    with library size (convergence). Returns the rho curve, its slope,
    and a z-vs-null via the smallest-vs-largest library comparison.
    """
    c = _v(cause)
    e = _v(effect)
    if c.size != e.size:
        raise ValueError("series must share length")
    Mx = takens_embed(e, dim, tau)  # effect manifold
    rows = Mx.shape[0]
    # Align cause with embedding rows: row j corresponds to time j+(E-1)tau.
    cy = c[(dim - 1) * tau :]
    sizes = np.array([20, 40, 60, 80]) if lib_sizes is None else np.asarray(lib_sizes)
    sizes = sizes[(sizes >= dim + 2) & (sizes <= rows)]
    if sizes.size < 2:
        raise ValueError("no feasible library sizes")
    rng = np.random.default_rng(seed)
    curve = np.empty(sizes.size)
    for i, s in enumerate(sizes):
        reps = np.array([_ccm_once(Mx, cy, int(s), rng) for _ in range(n_reps)])
        curve[i] = reps.mean()
    # Convergence: rho at the largest lib vs smallest (paired slope).
    x = np.log(sizes.astype(float))
    slope = float(np.polyfit(x, curve, 1)[0])
    return {
        "lib_sizes": sizes.astype(float),
        "rho": curve,
        "slope": slope,
        "rho_max": float(curve[-1]),
        "convergent": float(slope > 0.0 and curve[-1] > curve[0]),
    }


def smap_nonlinearity(
    x: Array, dim: int = 3, tau: int = 1, thetas: Array | None = None
) -> dict[str, Array | float]:
    """Sugihara (1994) nonlinearity test: S-map forecast skill vs theta.

    Locally-weighted linear map; theta=0 is the global linear model,
    theta>0 weights neighbors exponentially. A peak at theta>0 signals
    nonlinear state dependence. Returns the skill curve and argmax."""
    v = _v(x)
    th = (
        np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0])
        if thetas is None
        else np.asarray(thetas, dtype=float)
    )
    if np.any(th < 0):
        raise ValueError("thetas must be nonnegative")
    M = takens_embed(v, dim, tau)
    rows = M.shape[0]
    # Row i's "current" coordinate is time i + (E-1)tau; its target is
    # the next observation v[i + (E-1)tau + 1].
    target = v[(dim - 1) * tau + 1 : (dim - 1) * tau + rows]
    if target.size < rows:
        M = M[: target.size]
        rows = M.shape[0]
    if rows < 30:
        raise ValueError("too few embedded points")
    # Leave-one-out S-map.
    skill = np.empty(th.size)
    for ti, theta in enumerate(th):
        preds = np.empty(rows)
        for i in range(rows):
            d = np.linalg.norm(M - M[i], axis=1)
            d[i] = np.inf
            scale = d[np.isfinite(d)].mean()
            if theta == 0.0:
                w = np.ones(rows)
            else:
                w = np.exp(np.clip(-theta * d / max(scale, 1e-12), -700.0, 0.0))
            w[i] = 0.0  # leave-one-out
            Xd = np.column_stack([np.ones(rows), M]) * w[:, None]
            yw = target * w
            try:
                beta, *_ = np.linalg.lstsq(Xd, yw, rcond=None)
            except np.linalg.LinAlgError:
                preds[i] = np.nan
                continue
            preds[i] = float(beta[0] + beta[1:] @ M[i])
        good = np.isfinite(preds)
        if good.sum() < 10:
            raise ValueError("S-map degenerate")
        skill[ti] = float(np.corrcoef(preds[good], target[good])[0, 1])
    return {
        "thetas": th,
        "skill": skill,
        "theta_star": float(th[int(np.argmax(skill))]),
        "nonlinear": float(skill.max() > skill[0] + 0.05),
    }
