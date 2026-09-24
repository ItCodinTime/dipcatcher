"""Random projections for dimensionality reduction (Johnson-Lindenstrauss).

The Johnson-Lindenstrauss (1984) lemma states that ``n`` points can be embedded
into ``k = O(log n / eps^2)`` dimensions while preserving pairwise distances to
within a factor ``1 +/- eps``.  Two data-oblivious projections are provided:

- a dense Gaussian projection with entries ``N(0, 1/k)``;
- the Achlioptas (2003) sparse projection with entries
  ``sqrt(s/k) * {+1, 0, -1}`` at probabilities ``{1/2s, 1-1/s, 1/2s}``.

Reference: W. Johnson, J. Lindenstrauss (1984); D. Achlioptas (2003), JCSS.
Fail-closed on invalid shapes or parameters.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def johnson_lindenstrauss_min_dim(n_samples: int, eps: float = 0.1) -> int:
    """Minimum embedding dimension guaranteeing ``eps`` distortion for ``n`` points."""
    if n_samples < 2 or not 0.0 < eps < 1.0:
        raise ValueError("n_samples >= 2 and eps in (0, 1) required")
    denom = eps**2 / 2.0 - eps**3 / 3.0
    return int(np.ceil(4.0 * np.log(n_samples) / denom))


def _check(x: Array, k: int) -> Array:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2 or not np.isfinite(arr).all():
        raise ValueError("x must be a finite (n, d) matrix")
    if k < 1 or k > arr.shape[1]:
        raise ValueError("k must be in [1, d]")
    return arr


def gaussian_random_projection(
    x: Array, k: int, rng: np.random.Generator | None = None
) -> dict[str, Array]:
    """Project ``x`` (n, d) to ``k`` dimensions with a Gaussian random matrix."""
    arr = _check(x, k)
    gen = np.random.default_rng() if rng is None else rng
    r = gen.standard_normal((arr.shape[1], k)) / np.sqrt(k)
    return {"projected": arr @ r, "matrix": r}


def sparse_random_projection(
    x: Array, k: int, density: float = 1.0 / 3.0, rng: np.random.Generator | None = None
) -> dict[str, Array]:
    """Achlioptas (2003) sparse ``{+1,0,-1}`` random projection."""
    arr = _check(x, k)
    if not 0.0 < density <= 1.0:
        raise ValueError("density must be in (0, 1]")
    gen = np.random.default_rng() if rng is None else rng
    s = 1.0 / density
    d = arr.shape[1]
    u = gen.random((d, k))
    r = np.zeros((d, k))
    scale = np.sqrt(s / k)
    r[u < 1.0 / (2.0 * s)] = scale
    r[u > 1.0 - 1.0 / (2.0 * s)] = -scale
    return {"projected": arr @ r, "matrix": r}
