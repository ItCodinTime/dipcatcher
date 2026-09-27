"""Stationary block-bootstrap scenario paths.

Wraps :func:`quant_fund.metrics.inference.stationary_bootstrap_indices`
(Politis and Romano 1994). Blocks are drawn synchronously across assets, so
cross-sectional dependence inside a block is kept.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.inference import optimal_block_length, stationary_bootstrap_indices

Array = NDArray[np.float64]


def _as_panel(returns: Array) -> Array:
    panel = np.asarray(returns, dtype=float)
    if panel.ndim == 1:
        panel = panel.reshape(-1, 1)
    if panel.ndim != 2 or panel.shape[0] < 10 or panel.shape[1] < 1:
        raise ValueError("returns must have at least 10 rows")
    if not np.isfinite(panel).all():
        raise ValueError("returns must be finite")
    return np.asarray(panel, dtype=np.float64)


def resolve_block_length(returns: Array, mean_block: float | None) -> float:
    """Use the requested block length, or Politis–White on the first column."""
    panel = _as_panel(returns)
    if mean_block is None:
        auto = float(optimal_block_length(panel[:, 0]))
        if not np.isfinite(auto) or auto < 1.0:
            auto = max(1.0, float(panel.shape[0]) ** (1.0 / 3.0))
        mean_block = auto
    if not np.isfinite(mean_block) or not 1.0 <= float(mean_block) <= float(panel.shape[0]):
        raise ValueError("mean_block must be finite and in [1, n]")
    return float(mean_block)


def stationary_bootstrap_paths(
    returns: Array,
    n_paths: int,
    *,
    horizon: int | None = None,
    mean_block: float | None = None,
    seed: int = 0,
) -> Array:
    """Bootstrap scenario tensor with shape ``(n_paths, horizon, n_assets)``.

    ``horizon`` defaults to the sample length. Paths longer than the sample are
    not produced: the index generator is defined on the observed length, and a
    shorter horizon is a prefix of that resample.
    """
    panel = _as_panel(returns)
    n, _n_assets = panel.shape
    if isinstance(n_paths, bool) or not isinstance(n_paths, int) or n_paths < 1:
        raise ValueError("n_paths must be a positive integer")
    length = n if horizon is None else int(horizon)
    if length < 1 or length > n:
        raise ValueError("horizon must be in [1, n]")
    block = resolve_block_length(panel, mean_block)
    rng = np.random.default_rng(seed)
    index = stationary_bootstrap_indices(n, n_paths, block, rng)
    taken = index[:, :length]
    paths = panel[taken]
    return np.asarray(paths, dtype=np.float64)


def path_moments(paths: Array) -> dict[str, Array]:
    """Mean and covariance of one-step returns pooled across paths and time."""
    cube = np.asarray(paths, dtype=float)
    if cube.ndim != 3 or not np.isfinite(cube).all():
        raise ValueError("paths must be a finite (n_paths, horizon, n_assets) array")
    flat = cube.reshape(-1, cube.shape[-1])
    mean = np.asarray(flat.mean(axis=0), dtype=np.float64)
    if flat.shape[0] < 2:
        cov = np.full((flat.shape[1], flat.shape[1]), np.nan, dtype=np.float64)
    else:
        cov = np.asarray(np.cov(flat, rowvar=False, ddof=1), dtype=np.float64)
        if cov.ndim == 0:
            cov = np.asarray([[float(cov)]], dtype=np.float64)
    return {"mean": mean, "cov": cov}
