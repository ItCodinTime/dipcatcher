"""Anytime-valid e-process for prediction-set miss indicators.

One-step betting / likelihood-ratio e-value for Bernoulli(α) (Ramdas,
Grünwald, Shafer–Vovk). The running product is a nonnegative martingale
under the nominal miss rate, so Ville's inequality needs no peeking
correction: P(sup E_t ≥ 1/level) ≤ level. See ADR-010.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

_EPS = 1e-12
_E_MAX = 1e300
_ERR_LEVEL = 0.05


def _clip_prob(p: float, name: str) -> float:
    x = float(p)
    if not 0.0 < x < 1.0:
        raise ValueError(f"{name} must be in (0, 1)")
    return float(np.clip(x, _EPS, 1.0 - _EPS))


def _alt_lambda(alpha: float) -> float:
    """Fixed alternative miss rate λ ∈ (0, 1), predictable from α only.

    λ = 2α detects undercoverage (too many misses). If 2α would leave
    (0, 1), fall back to the midpoint between α and 1.
    """
    a = float(alpha)
    lam = 2.0 * a
    if lam >= 1.0 - _EPS:
        lam = 0.5 * (a + 1.0)
    return float(np.clip(lam, _EPS, 1.0 - _EPS))


def _step_e(miss: Array, alpha: float) -> Array:
    a = _clip_prob(alpha, "alpha")
    lam = _alt_lambda(a)
    x = np.clip(np.asarray(miss, dtype=float), 0.0, 1.0)
    raw = (x / a) * lam + ((1.0 - x) / (1.0 - a)) * (1.0 - lam)
    return np.clip(raw, 0.0, _E_MAX)


def e_value_bernoulli(miss: float, alpha: float) -> float:
    """One-step e-value for a miss indicator versus expected miss α.

    e = (miss / α) λ + ((1 − miss) / (1 − α)) (1 − λ),  λ = 2α (capped).
    """
    return float(_step_e(np.asarray([miss], dtype=float), alpha)[0])


def e_process(misses: Array, alpha: float) -> Array:
    """Running product E_t = ∏_{s≤t} e_s. Implicit E_0 = 1. Anytime-valid."""
    x = np.asarray(misses, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([1.0], dtype=float)
    steps = _step_e(x, alpha)
    log_steps = np.log(np.clip(steps, _EPS, _E_MAX))
    log_run = np.minimum(np.cumsum(log_steps), np.log(_E_MAX))
    return np.exp(log_run).astype(float)


def e_process_threshold(e: Array, level: float = 0.05) -> dict[str, object]:
    """First time the e-process meets 1/level (Ville). No peeking penalty."""
    path = np.asarray(e, dtype=float).reshape(-1)
    thresh = 1.0 / _clip_prob(level, "level")
    hit = path >= thresh
    if path.size > 0 and bool(np.any(hit)):
        first_cross: int | None = int(np.argmax(hit))
        reject = True
    else:
        first_cross = None
        reject = False
    return {
        "first_cross": first_cross,
        "e": path,
        "reject": reject,
        "threshold": float(thresh),
    }


def bench_e_coverage(covered: Array, alpha: float = 0.10) -> dict[str, object]:
    """Empirical coverage plus the anytime-valid miss e-process (level 0.05)."""
    c = np.asarray(covered, dtype=float).reshape(-1)
    c = c[np.isfinite(c)]
    n = int(c.size)
    if n == 0:
        return {"coverage": float("nan"), "e_final": 1.0, "ever_cross": False, "n": 0}
    misses = 1.0 - np.clip(c, 0.0, 1.0)
    path = e_process(misses, alpha)
    decision = e_process_threshold(path, level=_ERR_LEVEL)
    return {
        "coverage": float(np.mean(c)),
        "e_final": float(path[-1]),
        "ever_cross": bool(decision["reject"]),
        "n": n,
    }
