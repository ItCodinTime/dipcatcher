"""Entropy and information-theoretic complexity measures.

Model-free probes of predictability / disorder in a series.  All estimators
fail closed on degenerate input; finite-sample values are approximate by
construction (research diagnostics, not exact information measures).

References:
- Shannon (1948). A Mathematical Theory of Communication.
- Pincus (1991). Approximate entropy as a measure of system complexity. *PNAS*.
- Richman, Moorman (2000). Sample entropy. *Am. J. Physiol.* 278.
- Bandt, Pompe (2002). Permutation entropy. *Phys. Rev. Lett.* 88.
- Lempel, Ziv (1976). On the complexity of finite sequences. *IEEE IT* 22.
- Kaspar, Schuster (1987). Easily calculable measure for the complexity of
  spatiotemporal patterns (normalized LZ complexity).
- Schreiber (2000). Measuring information transfer. *Phys. Rev. Lett.* 85.
- Rostaghi, Azami (2016). Dispersion entropy. *IEEE TBME* 63(11).
- Powell et al. (1979) / Ince et al. — binning for entropy estimates.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _as_vector(x: Array, name: str = "x", *, min_obs: int = 8) -> Array:
    v = np.asarray(x, dtype=float).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size < min_obs:
        raise ValueError(f"{name} must contain at least {min_obs} finite observations")
    return v


def _entropy_from_probs(p: Array) -> float:
    p = p[p > 0.0]
    return float(-np.sum(p * np.log(p)))


def shannon_entropy(x: Array, bins: int = 10) -> dict[str, float]:
    """Histogram Shannon entropy of ``x`` (nats and normalized to [0,1])."""
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer >= 2")
    v = _as_vector(x, min_obs=bins)
    if v.min() == v.max():
        raise ValueError("x must have positive range")
    counts = np.histogram(v, bins=bins)[0].astype(float)
    h = _entropy_from_probs(counts / counts.sum())
    return {"entropy": h, "normalized": h / np.log(bins)}


def _phi(x: Array, m: int, r: float, *, exclude_self: bool) -> tuple[float, float]:
    """Correlation-sum terms for ApEn/SampEn (O(n^2); research scale only)."""
    n = x.size - m + 1
    counts = np.zeros(n)
    emb = np.stack([x[i : i + n] for i in range(m)], axis=1)
    for i in range(n):
        d = np.max(np.abs(emb - emb[i]), axis=1)
        mask = d <= r
        if exclude_self:
            mask[i] = False
        counts[i] = mask.sum()
    if exclude_self:
        denom = n - 1.0
        c = counts / denom
        return float(np.sum(counts)), float(np.sum(c[c > 0.0]))
    c = counts / n
    return float(np.mean(np.log(np.clip(c, 1e-300, None)))), float(np.sum(counts))


def approximate_entropy(x: Array, m: int = 2, r: float | None = None) -> float:
    """Pincus (1991) ApEn ``phi(m) - phi(m+1)``; ``r`` defaults to 0.2*sd."""
    v = _as_vector(x, min_obs=m + 4)
    r_val = 0.2 * float(np.std(v)) if r is None else float(r)
    if r_val <= 0.0 or not np.isfinite(r_val):
        raise ValueError("r must be positive and finite")
    phi_m, _ = _phi(v, m, r_val, exclude_self=False)
    phi_m1, _ = _phi(v, m + 1, r_val, exclude_self=False)
    return phi_m - phi_m1


def sample_entropy(x: Array, m: int = 2, r: float | None = None) -> float:
    """Richman–Moorman (2000) SampEn ``-log(A/B)`` with self-matches excluded.

    Returns ``nan`` when no template pairs match (honest non-estimable).
    """
    v = _as_vector(x, min_obs=m + 4)
    r_val = 0.2 * float(np.std(v)) if r is None else float(r)
    if r_val <= 0.0 or not np.isfinite(r_val):
        raise ValueError("r must be positive and finite")
    b, _ = _phi(v, m, r_val, exclude_self=True)
    a, _ = _phi(v, m + 1, r_val, exclude_self=True)
    if b <= 0.0 or a <= 0.0:
        return float("nan")
    return float(-np.log(a / b))


def _ordinal_patterns(x: Array, order: int, delay: int) -> NDArray[np.intp]:
    n = x.size - (order - 1) * delay
    if n <= 0:
        raise ValueError("series too short for requested order/delay")
    idx = np.arange(n)
    windows = np.stack([x[idx + j * delay] for j in range(order)], axis=1)
    # Lexicographic rank of each window's argsort pattern.
    return np.argsort(np.argsort(windows, axis=1), axis=1) @ (10 ** np.arange(order))


def permutation_entropy(x: Array, order: int = 3, delay: int = 1) -> dict[str, float]:
    """Bandt–Pompe (2002) permutation entropy, raw and normalized by log(order!)."""
    if isinstance(order, bool) or not isinstance(order, int) or order < 2:
        raise ValueError("order must be an integer >= 2")
    if isinstance(delay, bool) or not isinstance(delay, int) or delay < 1:
        raise ValueError("delay must be a positive integer")
    v = _as_vector(x, min_obs=order * delay + 2)
    keys = _ordinal_patterns(v, order, delay)
    counts = np.array(list(Counter(keys).values()), dtype=float)
    h = _entropy_from_probs(counts / counts.sum())
    return {
        "entropy": h,
        "normalized": h / np.log(math.factorial(order)),
        "n_patterns": float(counts.size),
    }


def lempel_ziv_complexity(x: Array, threshold: float | None = None) -> float:
    """Normalized Lempel–Ziv (1976) complexity of the sign-binarized series.

    LZ76 parse count normalized by ``n / log2(n)`` (Kaspar–Schuster 1987
    convention) so stationary iid sequences cluster near 1.
    """
    v = _as_vector(x)
    thr = float(np.median(v)) if threshold is None else float(threshold)
    if not np.isfinite(thr):
        raise ValueError("threshold must be finite")
    s = "".join("1" if z > thr else "0" for z in v)
    n = len(s)
    # LZ76 incremental parsing.
    i, c = 0, 0
    while i < n:
        j = 1
        while i + j <= n:
            sub = s[i : i + j]
            if s[: i + j - 1].find(sub) == -1:
                break
            j += 1
        c += 1
        i += j
    norm = n / np.log2(n) if n > 1 else 1.0
    return float(c / norm)


def spectral_entropy(x: Array, *, normalized: bool = True) -> float:
    """Shannon entropy of the periodogram power distribution (Inouye-style).

    Low values indicate concentration on few frequencies (predictability).
    """
    v = _as_vector(x)
    z = v - v.mean()
    if np.all(z == 0.0):
        raise ValueError("x must have positive variance")
    power = np.abs(np.fft.rfft(z)) ** 2
    power = power[1:]  # drop DC
    p = power / power.sum()
    h = _entropy_from_probs(p)
    return h / np.log(p.size) if normalized else h


def transfer_entropy(x: Array, y: Array, k: int = 1, bins: int = 3) -> dict[str, float]:
    """Schreiber (2000) transfer entropy ``TE_{x -> y}`` on quantile-binned data.

    ``TE = sum p(y_{t+1}, y_t^k, x_t^k) log [p(y_{t+1}|y_t^k, x_t^k) /
    p(y_{t+1}|y_t^k)]``.  Both directions are reported; binned estimates carry
    upward small-sample bias (diagnostic, not calibrated bits).
    """
    a = _as_vector(x, "x", min_obs=4 * bins + k + 2)
    b = _as_vector(y, "y", min_obs=4 * bins + k + 2)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]

    def _bin(v: Array) -> NDArray[np.intp]:
        qs = np.quantile(v, np.linspace(0.0, 1.0, bins + 1))
        qs[0], qs[-1] = -np.inf, np.inf
        return np.clip(np.digitize(v, qs) - 1, 0, bins - 1)

    xb, yb = _bin(a), _bin(b)

    def _te(src: NDArray[np.intp], dst: NDArray[np.intp]) -> float:
        # tuples: (dst_next, dst_hist, src_hist)
        rows = [(dst[t + 1], dst[t], src[t]) for t in range(k - 1, n - 1)]
        joint = Counter(rows)
        cond_full: Counter[tuple[int, int]] = Counter()
        cond_hist: Counter[int] = Counter()
        for (_dn, dh, sh), c in joint.items():
            cond_full[(dh, sh)] += c
            cond_hist[dh] += c
        total = sum(joint.values())
        te = 0.0
        for (dn, dh, sh), c in joint.items():
            p_joint = c / total
            p_full = joint[(dn, dh, sh)] / cond_full[(dh, sh)]
            p_marg = (
                sum(joint[(dn2, dh2, sh2)] for (dn2, dh2, sh2) in joint if dn2 == dn and dh2 == dh)
                / cond_hist[dh]
            )
            te += p_joint * math.log(p_full / p_marg)
        return te

    return {"te_x_to_y": float(_te(xb, yb)), "te_y_to_x": float(_te(yb, xb))}


def dispersion_entropy(
    x: Array, classes: int = 5, order: int = 3, delay: int = 1
) -> dict[str, float]:
    """Rostaghi–Azami (2016) dispersion entropy on normal-CDF class labels."""
    if isinstance(classes, bool) or not isinstance(classes, int) or classes < 2:
        raise ValueError("classes must be an integer >= 2")
    v = _as_vector(x, min_obs=order * delay + classes + 2)
    mu, sd = float(v.mean()), float(v.std(ddof=1))
    if sd <= 0.0:
        raise ValueError("x must have positive variance")
    from scipy.stats import norm

    z = norm.cdf((v - mu) / sd)
    labels = np.clip((z * classes).astype(int), 0, classes - 1)
    keys = _label_patterns(labels, order, delay, classes)
    counts = np.array(list(Counter(keys).values()), dtype=float)
    h = _entropy_from_probs(counts / counts.sum())
    return {"entropy": h, "normalized": h / np.log(classes**order)}


def _label_patterns(
    labels: NDArray[np.intp], order: int, delay: int, base: int
) -> NDArray[np.intp]:
    n = labels.size - (order - 1) * delay
    if n <= 0:
        raise ValueError("series too short for requested order/delay")
    idx = np.arange(n)
    windows = np.stack([labels[idx + j * delay] for j in range(order)], axis=1)
    return windows @ (base ** np.arange(order))
