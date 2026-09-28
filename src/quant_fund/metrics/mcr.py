"""PAVA + Murphy-decomposition miscalibration/discrimination measures.

Canonical corrected decomposition for the Brier / log scores on
probabilistic binary forecasts (Murphy 1972; Pohle et al. / Bogner, Liechti
et al. "PAVA" literature; Murphy & Locher MCR lineage):

    score = WGV + MCB - DSC + UNC - 2 * COV_w

where ``p_iso = PAVA(prob, y)`` is the isotonic recalibration of the issued
probabilities (the reliability curve), groups are the curve's blocks, and

    WGV = mean_g var(prob | g)               (within-group spread)
    MCB = sum_g n_g (p_g - o_g)^2 / N        (miscalibration, >= 0)
    DSC = sum_g n_g (o_g - y_bar)^2 / N      (discrimination, >= 0)
    UNC = var(y)                             (base-rate uncertainty)

For the Brier score the identity holds exactly; the same split applied to
the log score gives the URC/REL/RES form. A calibrated forecaster has
MCB ~ 0; a forecast with no discrimination has DSC ~ 0.

Fail-closed: probabilities outside [0, 1], length mismatch, non-finite.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _check_prob(prob: Array, y: Array) -> tuple[Array, Array]:
    p = np.asarray(prob, dtype=float).ravel()
    yy = np.asarray(y, dtype=float).ravel()
    if p.size < 2 or p.size != yy.size:
        raise ValueError("prob and y must be equal-length arrays with n >= 2")
    if not np.isfinite(p).all() or not np.isfinite(yy).all():
        raise ValueError("non-finite input")
    if (p < 0.0).any() or (p > 1.0).any():
        raise ValueError("probabilities must lie in [0, 1]")
    return p, yy


def pava(y: Array, w: Array | None = None) -> Array:
    """Pool-Adjacent-Violators isotonic (non-decreasing) regression.

    Returns the weighted least-squares fit of ``y`` subject to monotone
    non-decreasing order (Robertson, Wright & Dykstra 1988).
    """
    yy = np.asarray(y, dtype=float).ravel()
    n = yy.size
    if n == 0 or not np.isfinite(yy).all():
        raise ValueError("y must be a finite non-empty array")
    if w is None:
        ww = np.ones(n)
    else:
        ww = np.asarray(w, dtype=float).ravel()
        if ww.size != n or not np.isfinite(ww).all() or (ww <= 0).any():
            raise ValueError("w must be positive and match y")
    level = yy.copy()
    weight = ww.copy()
    starts = np.arange(n)
    ends = np.arange(n)
    n_block = n
    i = 0
    while i < n_block - 1:
        if level[i] > level[i + 1]:
            # pool blocks i and i+1
            new_w = weight[i] + weight[i + 1]
            new_level = (weight[i] * level[i] + weight[i + 1] * level[i + 1]) / new_w
            level[i] = new_level
            weight[i] = new_w
            ends[i] = ends[i + 1]
            # delete block i+1
            level = np.delete(level, i + 1)
            weight = np.delete(weight, i + 1)
            starts = np.delete(starts, i + 1)
            ends = np.delete(ends, i + 1)
            n_block -= 1
            i = max(0, i - 1)
        else:
            i += 1
    out = np.empty(n)
    for s_, e_, lv in zip(starts, ends, level, strict=True):
        out[s_ : e_ + 1] = lv
    return out


def reliability_isotonic(prob: Array, y: Array) -> Array:
    """Isotonic recalibration map evaluated at the issued probabilities."""
    p, yy = _check_prob(prob, y)
    order = np.argsort(p, kind="stable")
    iso_sorted = pava(yy[order])
    out = np.empty_like(p)
    out[order] = iso_sorted
    return out


def mcr_decomposition(prob: Array, y: Array) -> dict[str, float]:
    """Murphy Brier decomposition grouped by isotonic level (exact).

    Groups observations by their PAVA recalibrated value (the reliability
    curve's blocks). With group means ``p_g`` (issued) and ``o_g`` (outcome
    rate, the isotonic level):

        BS = WGV + REL - RES + UNC - 2 * COV_w

        WGV   = mean_g var(p | g)              (within-group forecast spread)
        COV_w = mean_g cov(p, y | g)           (within-group p-y covariance)
        REL   = sum_g n_g (p_g - o_g)^2 / N    (miscalibration, MCB)
        RES   = sum_g n_g (o_g - y_bar)^2 / N  (discrimination, DSC)
        UNC   = var(y)                         (base-rate uncertainty)

    Exact for any outcome (Brocker 2009 corrected Murphy decomposition).
    When issued probabilities are constant inside each block, WGV = COV_w = 0
    and the classic BS = REL - RES + UNC form is recovered.
    """
    p, yy = _check_prob(prob, y)
    p_iso = reliability_isotonic(p, yy)
    levels = np.unique(p_iso)
    n = p.size
    mcb = 0.0
    dsc = 0.0
    wgv = 0.0
    cov_w = 0.0
    y_bar = float(np.mean(yy))
    for lev in levels:
        mask = p_iso == lev
        n_g = int(mask.sum())
        p_g = float(np.mean(p[mask]))
        o_g = float(np.mean(yy[mask]))
        mcb += n_g * (p_g - lev) ** 2
        dsc += n_g * (lev - y_bar) ** 2
        wgv += float(np.sum((p[mask] - p_g) ** 2))
        cov_w += float(np.sum((p[mask] - p_g) * (yy[mask] - o_g)))
    mcb /= n
    dsc /= n
    wgv /= n
    cov_w /= n
    unc = float(np.var(yy))
    bs = float(np.mean((p - yy) ** 2))
    return {
        "brier": bs,
        "mcb": float(mcb),
        "dsc": float(dsc),
        "unc": unc,
        "within_group_variance": float(wgv),
        "cov_within": float(cov_w),
        "residual": bs - (wgv + mcb - dsc + unc - 2.0 * cov_w),
    }


def log_score_urc(prob: Array, y: Array) -> dict[str, float]:
    """Log-score URC split: LS = REL - RES + UNC in natural-log units."""
    p, yy = _check_prob(prob, y)
    p_iso = reliability_isotonic(p, yy)
    eps = 1e-15
    pc = np.clip(p, eps, 1.0 - eps)
    pi = np.clip(p_iso, eps, 1.0 - eps)
    base = np.clip(np.mean(yy), eps, 1.0 - eps)

    def _log_score(q: Array | float) -> float:
        return float(-np.mean(yy * np.log(q) + (1.0 - yy) * np.log(1.0 - q)))

    ls = _log_score(pc)
    ls_rel = _log_score(pc) - _log_score(pi)
    unc = _log_score(base)
    res = _log_score(pi) - unc
    return {
        "log_score": ls,
        "rel": float(ls_rel),
        "res": float(res),
        "unc": float(unc),
        "residual": float(ls - (ls_rel + res + unc)),
    }
