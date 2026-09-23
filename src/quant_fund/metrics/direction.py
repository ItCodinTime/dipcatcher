"""Directional-accuracy evaluation: Pesaran-Timmermann test, confusion
matrix, rank AUC for probabilistic sign forecasts.

Pesaran & Timmermann (1992): for sign forecasts vs realized signs, the
statistic
  PT = sqrt(n) (P_hat - P_star) / sqrt(P_star (1 - P_star))
with P_hat the observed hit rate and P_star the expected rate under
independence (product of marginals), is asymptotically N(0,1) under
H0: no directional predictability.

Fail-closed: mismatched lengths, non-finite inputs, degenerate margins.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm, rankdata

Array = NDArray[np.float64]


def pesaran_timmermann(forecast_sign: Array, realized_sign: Array) -> dict[str, float]:
    """PT directional-predictability test. Signs in {-1,+1} (0 treated
    as its own category is not supported — caller decides convention)."""
    f = np.asarray(forecast_sign, dtype=float).ravel()
    r = np.asarray(realized_sign, dtype=float).ravel()
    if f.shape != r.shape or f.size < 20:
        raise ValueError("forecast and realized must match, >= 20 obs")
    if not np.isfinite(f).all() or not np.isfinite(r).all():
        raise ValueError("inputs must be finite")
    fu = (f > 0).astype(int)
    ru = (r > 0).astype(int)
    p_hat = float((fu == ru).mean())
    p_y = float(ru.mean())
    p_f = float(fu.mean())
    p_star = p_y * p_f + (1.0 - p_y) * (1.0 - p_f)
    denom = p_star * (1.0 - p_star)
    if denom <= 0.0:
        raise ValueError("degenerate margins: cannot normalize PT")
    pt = float(np.sqrt(f.size) * (p_hat - p_star) / np.sqrt(denom))
    return {
        "hit_rate": p_hat,
        "p_star": p_star,
        "pt_stat": pt,
        "pvalue": float(norm.sf(pt)),  # one-sided: positive predictability
        "n": float(f.size),
    }


def direction_confusion(forecast_sign: Array, realized_sign: Array) -> dict[str, float]:
    """2x2 directional confusion: up-up, up-down, down-up, down-down
    plus recall per realized class (hit_when_up = uu/(uu+du))."""
    f = np.asarray(forecast_sign, dtype=float).ravel()
    r = np.asarray(realized_sign, dtype=float).ravel()
    if f.shape != r.shape or f.size == 0:
        raise ValueError("shapes must match, non-empty")
    if not np.isfinite(f).all() or not np.isfinite(r).all():
        raise ValueError("inputs must be finite")
    fu = f > 0
    ru = r > 0
    uu = float((fu & ru).sum())
    ud = float((fu & ~ru).sum())
    du = float((~fu & ru).sum())
    dd = float((~fu & ~ru).sum())
    prec_up = uu / (uu + du) if (uu + du) > 0 else float("nan")
    prec_dn = dd / (dd + ud) if (dd + ud) > 0 else float("nan")
    return {
        "up_up": uu,
        "up_down": ud,
        "down_up": du,
        "down_down": dd,
        "hit_when_up": prec_up,
        "hit_when_down": prec_dn,
    }


def directional_auc(score: Array, realized_sign: Array) -> dict[str, float]:
    """Rank-based AUC: P(score_i > score_j | r_i=+1, r_j=-1).

    ``score`` is any monotone-in-bullishness forecast (e.g. predicted
    probability of an up move). Ties count 0.5 (Mann-Whitney form).
    """
    s = np.asarray(score, dtype=float).ravel()
    r = np.asarray(realized_sign, dtype=float).ravel()
    if s.shape != r.shape or s.size < 10:
        raise ValueError("score and realized must match, >= 10 obs")
    if not np.isfinite(s).all() or not np.isfinite(r).all():
        raise ValueError("inputs must be finite")
    up = r > 0
    n_up = int(up.sum())
    n_dn = int((~up).sum())
    if n_up == 0 or n_dn == 0:
        raise ValueError("need both up and down realizations")
    ranks = rankdata(s)
    auc = float((ranks[up].sum() - n_up * (n_up + 1) / 2.0) / (n_up * n_dn))
    return {"auc": auc, "n_up": float(n_up), "n_down": float(n_dn)}
