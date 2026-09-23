"""Lopez de Prado (2016) Hierarchical Risk Parity and inverse-variance.

Faithful port of the paper appendix as used in davidalmeida90/quant-models
``hierarchical-risk-parity/model.py``. Long-only min-variance stands in
for CLA. Research allocation; not a live book.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from numpy.typing import ArrayLike, NDArray
from scipy.cluster.hierarchy import ClusterWarning
from scipy.optimize import minimize

Array = NDArray[np.float64]


def inverse_variance_weights(cov: ArrayLike) -> Array:
    c = np.asarray(cov, dtype=float)
    diag = np.diag(c)
    if np.any(diag <= 0) or not np.isfinite(diag).all():
        raise ValueError("covariance diagonal must be positive and finite")
    w = 1.0 / diag
    return np.asarray(w / w.sum(), dtype=float)


def correl_dist(corr: ArrayLike) -> Array:
    c = np.asarray(corr, dtype=float)
    return np.asarray(((1.0 - c) / 2.0) ** 0.5, dtype=float)


def _quasi_diag(link: np.ndarray) -> list[int]:
    link_i = link.astype(int)
    sort_ix = pd.Series([link_i[-1, 0], link_i[-1, 1]])
    n_items = int(link_i[-1, 3])
    while sort_ix.max() >= n_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= n_items]
        i = df0.index
        j = df0.to_numpy() - n_items
        sort_ix.loc[i] = link_i[j, 0]
        extra = pd.Series(link_i[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, extra]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return [int(x) for x in sort_ix.tolist()]


def _cluster_var(cov: pd.DataFrame, items: list) -> float:
    sub = cov.loc[items, items]
    w = inverse_variance_weights(sub.to_numpy()).reshape(-1, 1)
    return float((w.T @ sub.to_numpy() @ w)[0, 0])


def _rec_bipart(cov: pd.DataFrame, sort_ix: list) -> pd.Series:
    w = pd.Series(1.0, index=sort_ix)
    clusters = [sort_ix]
    while clusters:
        clusters = [
            chunk
            for items in clusters
            for chunk in (items[: len(items) // 2], items[len(items) // 2 :])
            if len(items) > 1
        ]
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            a = _cluster_var(cov, left)
            b = _cluster_var(cov, right)
            alpha = 1.0 - a / (a + b)
            w[left] *= alpha
            w[right] *= 1.0 - alpha
    return w


def hrp_weights(cov: ArrayLike, corr: ArrayLike | None = None) -> Array:
    """HRP weights in original asset order. Sum to 1, long-only."""
    v = np.asarray(cov, dtype=float)
    if v.ndim != 2 or v.shape[0] != v.shape[1]:
        raise ValueError("cov must be square")
    n = v.shape[0]
    if corr is None:
        d = np.sqrt(np.diag(v))
        c = v / np.outer(d, d)
    else:
        c = np.asarray(corr, dtype=float)
    cov_df = pd.DataFrame(v)
    corr_df = pd.DataFrame(c)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ClusterWarning)
        link = sch.linkage(correl_dist(corr_df.to_numpy()), "single")
    order = corr_df.index[_quasi_diag(link)].tolist()
    w = _rec_bipart(cov_df, order).sort_index()
    out = w.reindex(range(n)).to_numpy(dtype=float)
    return np.asarray(out / out.sum(), dtype=float)


def long_only_min_variance(cov: ArrayLike) -> Array:
    """Long-only min-variance (SLSQP). Stands in for CLA in the LdP appendix."""
    v = np.asarray(cov, dtype=float)
    n = v.shape[0]
    scale = float(np.mean(np.diag(v)))
    vs = v / scale

    def obj(w: np.ndarray) -> float:
        return float(w @ vs @ w)

    res = minimize(
        obj,
        np.repeat(1.0 / n, n),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=({"type": "eq", "fun": lambda w: float(w.sum() - 1.0)},),
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not res.success:
        raise RuntimeError(f"min-variance failed: {res.message}")
    return np.asarray(res.x, dtype=float)


def generate_ldp_example(
    n_obs: int = 10_000,
    size0: int = 5,
    size1: int = 5,
    sigma1: float = 0.25,
    seed: int = 12345,
) -> tuple[pd.DataFrame, list[int]]:
    """Lopez de Prado appendix A.3 correlated-block panel."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, size=(n_obs, size0))
    cols = [int(rng.integers(0, size0)) for _ in range(size1)]
    y = x[:, cols] + rng.normal(0, sigma1, size=(n_obs, len(cols)))
    panel = np.append(x, y, axis=1)
    return pd.DataFrame(panel, columns=range(1, panel.shape[1] + 1)), cols


def hcaa_weights(cov: ArrayLike, corr: ArrayLike | None = None) -> Array:
    """Hierarchical equal-weight allocation (Raffinot-style HCAA)."""
    v = np.asarray(cov, dtype=float)
    n = v.shape[0]
    if corr is None:
        d = np.sqrt(np.diag(v))
        c = v / np.outer(d, d)
    else:
        c = np.asarray(corr, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ClusterWarning)
        link = sch.linkage(correl_dist(c), "single")
    order = _quasi_diag(link)
    w = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:
        clusters = [
            chunk
            for items in clusters
            for chunk in (items[: len(items) // 2], items[len(items) // 2 :])
            if len(items) > 1
        ]
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            w[left] *= 0.5
            w[right] *= 0.5
    out = w.sort_index().reindex(range(n)).to_numpy(dtype=float)
    return np.asarray(out / out.sum(), dtype=float)
