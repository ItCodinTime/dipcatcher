"""Portfolio sorts with NYSE breakpoints.

Fama-French style characteristic sorts: sort assets on a characteristic,
bucket into portfolios at NYSE-only breakpoints (deciles/quintiles),
aggregate equal- or value-weighted next-period returns per portfolio, and
compute the long-short (top minus bottom) spread.

This is a research diagnostic: it reports portfolio mean returns and a
basic t-stat; it never claims promotion or trading value.

- ``breakpoints_nyse``: quantile cut points restricted to a subset mask.
- ``sort_portfolios``: assign each (t, asset) to a bucket and compute
  portfolio forward returns (EW or VW).
- ``bivariate_sort``: independent double sorts (size x bm style) —
  control and measure the interaction.
- ``sort_tstat``: mean/std-err/t on the portfolio spread series.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _check_cs(char: Array, fwd: Array) -> tuple[Array, Array]:
    c = np.asarray(char, dtype=float)
    f = np.asarray(fwd, dtype=float)
    if c.shape != f.shape or c.ndim != 2:
        raise ValueError("char and fwd must be (T, N) with equal shape")
    if c.shape[0] < 3 or c.shape[1] < 4:
        raise ValueError("need T >= 3, N >= 4")
    if not np.isfinite(c).all() or not np.isfinite(f).all():
        raise ValueError("non-finite input")
    return c, f


def breakpoints_nyse(char_row: Array, nyse_mask: Array, n_bins: int) -> Array:
    """Quantile breakpoints computed on the NYSE subset only."""
    x = np.asarray(char_row, dtype=float).ravel()
    m = np.asarray(nyse_mask, dtype=bool).ravel()
    if x.shape != m.shape or m.sum() < n_bins:
        raise ValueError("mask mismatch / too few NYSE names for the bins")
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    sub = x[m]
    if not np.isfinite(sub).all():
        raise ValueError("non-finite NYSE characteristics")
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    return np.asarray(np.quantile(sub, qs), dtype=float)


def sort_portfolios(
    char: Array,
    fwd: Array,
    n_bins: int = 10,
    nyse_mask: Array | None = None,
    value_weight: Array | None = None,
) -> dict[str, Array]:
    """Per-period sorted buckets -> EW/VW forward returns per bucket.

    ``fwd[t, i]`` is the forward (t -> t+1) return attached to asset i.
    Returns bucket return matrix (T, n_bins) and bucket counts.
    """
    c, f = _check_cs(char, fwd)
    t_len, n = c.shape
    if nyse_mask is None:
        nyse_mask = np.ones(n, dtype=bool)
    if value_weight is not None:
        vw = np.asarray(value_weight, dtype=float)
        if vw.shape != c.shape or (vw <= 0).any() or not np.isfinite(vw).all():
            raise ValueError("value_weight must be positive and (T, N)")
    else:
        vw = np.ones_like(c)
    rets = np.full((t_len, n_bins), np.nan)
    counts = np.zeros((t_len, n_bins), dtype=np.int64)
    for t in range(t_len):
        cuts = breakpoints_nyse(c[t], nyse_mask, n_bins)
        bucket = np.clip(np.digitize(c[t], cuts), 0, n_bins - 1)
        for b in range(n_bins):
            sel = bucket == b
            counts[t, b] = int(sel.sum())
            if sel.any():
                w = vw[t, sel] / vw[t, sel].sum()
                rets[t, b] = float(w @ f[t, sel])
    spread = rets[:, -1] - rets[:, 0]
    return {
        "bucket_returns": rets,
        "counts": counts,
        "spread": spread,
        "spread_stats": sort_tstat(spread),
    }


def sort_tstat(spread: Array) -> dict[str, float]:
    """Plain i.i.d. t-stat on a portfolio spread (research diagnostic)."""
    s = np.asarray(spread, dtype=float).ravel()
    s = s[np.isfinite(s)]
    if s.size < 4:
        return {"mean": float("nan"), "se": float("nan"), "t": float("nan"), "n": float(s.size)}
    mean = float(s.mean())
    se = float(s.std(ddof=1) / np.sqrt(s.size))
    return {
        "mean": mean,
        "se": se,
        "t": mean / se if se > 0 else float("nan"),
        "n": float(s.size),
    }


def bivariate_sort(
    char1: Array,
    char2: Array,
    fwd: Array,
    n1: int = 5,
    n2: int = 5,
    nyse_mask: Array | None = None,
) -> dict[str, Array]:
    """Independent double sort: returns (n1, n2) mean forward returns.

    char1 cuts are NYSE-only; char2 cuts computed within each char1
    bucket's NYSE subset (the classic dependent-cut is deliberately
    replaced by the independent form for comparability).
    """
    c1, f = _check_cs(char1, fwd)
    c2 = np.asarray(char2, dtype=float)
    if c2.shape != c1.shape or not np.isfinite(c2).all():
        raise ValueError("char2 must match char1 and be finite")
    t_len, n = c1.shape
    if nyse_mask is None:
        nyse_mask = np.ones(n, dtype=bool)
    # per-t bucket EW mean, averaged across t
    sums = np.zeros((n1, n2))
    tcounts = np.zeros((n1, n2), dtype=np.int64)
    counts = np.zeros((n1, n2), dtype=np.int64)
    for t in range(t_len):
        cuts1 = breakpoints_nyse(c1[t], nyse_mask, n1)
        b1 = np.clip(np.digitize(c1[t], cuts1), 0, n1 - 1)
        cuts2 = breakpoints_nyse(c2[t], nyse_mask, n2)
        b2 = np.clip(np.digitize(c2[t], cuts2), 0, n2 - 1)
        for i in range(n1):
            for j in range(n2):
                sel = (b1 == i) & (b2 == j)
                if sel.any():
                    sums[i, j] += f[t, sel].mean()
                    tcounts[i, j] += 1
                    counts[i, j] += int(sel.sum())
    means = np.where(tcounts > 0, sums / np.maximum(tcounts, 1), np.nan)
    return {"means": means, "counts": counts, "period_counts": tcounts}
