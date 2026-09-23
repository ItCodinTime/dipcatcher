"""MODWT wavelet analysis (Percival & Walden, 2000).

Maximal-overlap discrete wavelet transform with Haar / D4 / LA(8)
filters, circular pyramid algorithm:

  W_{j,t} = sum_l g~_{j,l} X_{(t - l 2^{j-1}) mod N}
  V_{j,t} = sum_l h~_{j,l} X_{(t - l 2^{j-1}) mod N}

where g~ = g/sqrt(2) are the MODWT filters. Provides per-scale
variance decomposition, MRA additive reconstruction, and per-scale
wavelet correlation.

Fail-closed: non-finite input, levels > log2(n), unknown wavelet.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

_SQRT3 = math.sqrt(3.0)
_SQRT2 = math.sqrt(2.0)

# DWT scaling filters h (Daubechies convention); MODWT uses h/sqrt(2).
_DWT_SCALING: dict[str, Array] = {
    "haar": np.array([1.0 / _SQRT2, 1.0 / _SQRT2]),
    "d4": np.array(
        [
            (1.0 + _SQRT3) / (4.0 * _SQRT2),
            (3.0 + _SQRT3) / (4.0 * _SQRT2),
            (3.0 - _SQRT3) / (4.0 * _SQRT2),
            (1.0 - _SQRT3) / (4.0 * _SQRT2),
        ]
    ),
    "la8": np.array(
        [
            -0.010597401784997278,
            0.032883011666982945,
            0.030841381835986965,
            -0.18703481171888114,
            -0.02798376941698385,
            0.6308807679295904,
            0.7148465705525415,
            0.23037781330885523,
        ]
    ),
}


def _filters(name: str) -> tuple[Array, Array]:
    if name not in _DWT_SCALING:
        raise ValueError(f"unknown wavelet {name!r}; choose haar|d4|la8")
    h = _DWT_SCALING[name] / _SQRT2
    g = np.empty_like(h)
    # QMF: g_l = (-1)^{l+1} h_{L-1-l}
    for li in range(h.size):
        g[li] = ((-1.0) ** (li + 1)) * h[h.size - 1 - li]
    return h, g


def modwt(x: Array, wavelet: str = "haar", levels: int | None = None) -> dict[str, Array]:
    """MODWT pyramid. Returns dict with ``details`` (list W_j) and
    ``smooth`` (V_J), each length n (circular)."""
    xx = np.asarray(x, dtype=float).ravel()
    n = xx.size
    if n < 8 or not np.isfinite(xx).all():
        raise ValueError("x must be finite, >= 8 obs")
    max_j = int(math.log2(n))
    if levels is None:
        levels = max_j
    if levels < 1 or levels > max_j:
        raise ValueError("levels must be in [1, log2(n)]")
    h, g = _filters(wavelet)
    v = xx.copy()
    details = []
    for j in range(1, levels + 1):
        step = 1 << (j - 1)
        w = np.zeros(n)
        v_new = np.zeros(n)
        l_ = h.size
        for t in range(n):
            acc_w = 0.0
            acc_v = 0.0
            for li in range(l_):
                idx = (t - li * step) % n
                acc_w += g[li] * v[idx]
                acc_v += h[li] * v[idx]
            w[t] = acc_w
            v_new[t] = acc_v
        details.append(w)
        v = v_new
    return {
        "details": np.asarray(details),
        "smooth": v,
        "levels": np.asarray([float(levels)]),
    }


def modwt_mra(x: Array, wavelet: str = "haar", levels: int | None = None) -> dict[str, Array]:
    """Additive MRA: x = smooth + sum_j detail_j (circular inverse)."""
    xx = np.asarray(x, dtype=float).ravel()
    out = modwt(xx, wavelet, levels)
    h, g = _filters(wavelet)
    n = xx.size
    details_w = np.asarray(out["details"])
    smooth_v = np.asarray(out["smooth"])
    j_levels = details_w.shape[0]

    def _invert(w_j: Array, v_j: Array, j: int) -> Array:
        step = 1 << (j - 1)
        x_rec = np.zeros(n)
        for t in range(n):
            acc = 0.0
            for li in range(h.size):
                idx = (t + li * step) % n
                acc += g[li] * w_j[idx] + h[li] * v_j[idx]
            x_rec[t] = acc
        return x_rec

    # inverse pyramid: X_j = g~ W_j + h~ V_j applied level by level.
    mra_details = []
    for j in range(1, j_levels + 1):
        w_only = details_w[j - 1]
        rec = _invert(w_only, np.zeros(n), j)
        # propagate through remaining levels with zero detail
        v_cur = rec
        for jj in range(j - 1, 0, -1):
            step = 1 << (jj - 1)
            tmp = np.zeros(n)
            for t in range(n):
                acc = 0.0
                for li in range(h.size):
                    idx = (t + li * step) % n
                    acc += h[li] * v_cur[idx]
                tmp[t] = acc
            v_cur = tmp
        mra_details.append(v_cur)
    rec_smooth = smooth_v.copy()
    for jj in range(j_levels, 0, -1):
        step = 1 << (jj - 1)
        tmp = np.zeros(n)
        for t in range(n):
            acc = 0.0
            for li in range(h.size):
                idx = (t + li * step) % n
                acc += h[li] * rec_smooth[idx]
            tmp[t] = acc
        rec_smooth = tmp
    return {
        "details": np.asarray(mra_details),
        "smooth": rec_smooth,
        "levels": np.asarray([float(j_levels)]),
    }


def wavelet_variance(
    x: Array,
    wavelet: str = "haar",
    levels: int | None = None,
) -> dict[str, Array | float]:
    """Per-scale variance decomposition: var(x) ~ sum_j var(W_j) + var(V_J)."""
    xx = np.asarray(x, dtype=float).ravel()
    out = modwt(xx, wavelet, levels)
    w = np.asarray(out["details"])
    v = np.asarray(out["smooth"])
    var_w = w.var(axis=1)
    total = float(var_w.sum() + v.var())
    share = var_w / total if total > 0 else np.full(w.shape[0], np.nan)
    return {
        "scale_variance": var_w,
        "smooth_variance": float(v.var()),
        "total_variance": total,
        "variance_share": np.asarray(share, dtype=float),
        "sample_variance": float(xx.var()),
    }


def wavelet_correlation(
    x: Array,
    y: Array,
    wavelet: str = "haar",
    level: int | None = None,
) -> dict[str, Array | float]:
    """Per-scale correlation of MODWT details between two series."""
    a = np.asarray(x, dtype=float).ravel()
    b = np.asarray(y, dtype=float).ravel()
    if a.shape != b.shape or a.size < 16:
        raise ValueError("two series of equal length >= 16 required")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("series must be finite")
    wa = np.asarray(modwt(a, wavelet, levels=level)["details"])
    wb = np.asarray(modwt(b, wavelet, levels=level)["details"])
    corrs = []
    for j in range(wa.shape[0]):
        sa, sb = wa[j].std(), wb[j].std()
        corrs.append(float(np.corrcoef(wa[j], wb[j])[0, 1]) if sa > 0 and sb > 0 else float("nan"))
    return {"correlations": np.asarray(corrs), "n_levels": float(wa.shape[0])}
