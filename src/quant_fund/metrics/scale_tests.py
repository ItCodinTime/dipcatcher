"""Equality-of-variance (scale) tests across groups.

- **Bartlett** (1937): likelihood-ratio chi-squared test, powerful under
  normality but sensitive to departures from it.
- **Levene** (1960): ANOVA on absolute deviations from the group mean.
- **Brown-Forsythe** (1974): Levene using the median, robust to heavy tails.
- **Fligner-Killeen** (1976): rank-based chi-squared test, the most robust.

Each returns the statistic, p-value, and degrees of freedom.  Fail-closed on
fewer than two groups or non-finite input.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2, f, norm, rankdata

Array = NDArray[np.float64]


def _groups(groups: tuple[Array, ...]) -> list[Array]:
    gs = [np.asarray(g, dtype=float).ravel() for g in groups]
    if (
        len(gs) < 2
        or any(g.size < 2 for g in gs)
        or any(not np.isfinite(g).all() for g in gs)
    ):
        raise ValueError("need >= 2 groups each with >= 2 finite observations")
    return gs


def bartlett_test(*groups: Array) -> dict[str, float]:
    """Bartlett's test for homogeneity of variances."""
    gs = _groups(groups)
    k = len(gs)
    n = np.array([g.size for g in gs])
    v = np.array([g.var(ddof=1) for g in gs])
    if not np.isfinite(v).all() or (v <= 0.0).any():
        raise ValueError("Bartlett's test requires positive finite group variances")
    ntot = int(n.sum())
    sp2 = float(np.sum((n - 1) * v) / (ntot - k))
    if not np.isfinite(sp2) or sp2 <= 0.0:
        raise ValueError("degenerate pooled variance")
    num = (ntot - k) * np.log(sp2) - np.sum((n - 1) * np.log(v))
    c = 1.0 + (np.sum(1.0 / (n - 1)) - 1.0 / (ntot - k)) / (3.0 * (k - 1))
    stat = float(num / c)
    if not np.isfinite(stat):
        raise ValueError("non-finite Bartlett statistic")
    return {
        "statistic": stat,
        "pvalue": float(chi2.sf(stat, k - 1)),
        "df": float(k - 1),
    }


def _levene_like(gs: list[Array], center: str) -> dict[str, float]:
    k = len(gs)
    ntot = sum(g.size for g in gs)
    z = []
    for g in gs:
        c = np.median(g) if center == "median" else g.mean()
        z.append(np.abs(g - c))
    zbar_i = np.array([zi.mean() for zi in z])
    zbar = float(np.concatenate(z).mean())
    numer = np.sum([zi.size * (zbar_i[i] - zbar) ** 2 for i, zi in enumerate(z)])
    denom = np.sum([np.sum((zi - zbar_i[i]) ** 2) for i, zi in enumerate(z)])
    if not np.isfinite(numer) or not np.isfinite(denom) or denom <= 0.0:
        raise ValueError("degenerate within-group deviation")
    stat = float((ntot - k) / (k - 1) * numer / denom)
    if not np.isfinite(stat):
        raise ValueError("non-finite Levene statistic")
    return {
        "statistic": stat,
        "pvalue": float(f.sf(stat, k - 1, ntot - k)),
        "df1": float(k - 1),
        "df2": float(ntot - k),
    }


def levene_test(*groups: Array) -> dict[str, float]:
    """Levene's test (deviations from the group mean)."""
    return _levene_like(_groups(groups), center="mean")


def brown_forsythe(*groups: Array) -> dict[str, float]:
    """Brown-Forsythe test (deviations from the group median)."""
    return _levene_like(_groups(groups), center="median")


def fligner_killeen(*groups: Array) -> dict[str, float]:
    """Fligner-Killeen rank-based test for equal variances."""
    gs = _groups(groups)
    k = len(gs)
    centered = [np.abs(g - np.median(g)) for g in gs]
    allc = np.concatenate(centered)
    if not np.isfinite(allc).all():
        raise ValueError("non-finite centered deviations")
    ranks = rankdata(allc)
    scores = norm.ppf(0.5 + ranks / (2.0 * (allc.size + 1)))
    sbar = float(scores.mean())
    v = float(scores.var(ddof=1))
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError("degenerate Fligner score variance")
    idx = 0
    stat = 0.0
    for g in centered:
        s = scores[idx : idx + g.size]
        idx += g.size
        stat += g.size * (s.mean() - sbar) ** 2
    stat = float(stat / v)
    if not np.isfinite(stat):
        raise ValueError("non-finite Fligner statistic")
    return {
        "statistic": stat,
        "pvalue": float(chi2.sf(stat, k - 1)),
        "df": float(k - 1),
    }
