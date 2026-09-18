"""Anytime-valid e-processes: coverage misses + forecast loss differentials.

1. Bernoulli miss e-process vs nominal α (Ramdas / Grünwald / Shafer–Vovk;
   ADR-010). Ville: P(sup E_t ≥ 1/level) ≤ level with no peeking correction.
2. Loss-differential e-process (Choe–Ramdas style) for H0: E[d] ≤ 0 where
   d = L_A − L_B (smaller loss better). Research-diagnostic only.
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
    return np.asarray(np.clip(raw, 0.0, _E_MAX), dtype=np.float64)


def e_value_bernoulli(miss: float, alpha: float) -> float:
    """One-step e-value for a miss indicator versus expected miss α.

    e = (miss / α) λ + ((1 − miss) / (1 − α)) (1 − λ),  λ = 2α (capped).

    Honesty note: ``miss`` is silently clipped to ``[0, 1]`` inside ``_step_e``
    (soft coverage / fractional miss allowed). Out-of-range values are *not*
    rejected — callers that need hard {0,1} indicators must pre-validate.
    ``bench_e_coverage`` likewise clips covered∈[0,1] before forming misses.
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
    return np.asarray(np.exp(log_run), dtype=np.float64)


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


def _clip_lam(lam: float) -> float:
    x = float(lam)
    if not 0.0 < x < 1.0:
        raise ValueError("lam must be in (0, 1)")
    return float(np.clip(x, _EPS, 1.0 - _EPS))


def _psi_exp(lam: float) -> float:
    """Exponential CGF ψ_E(λ) = −log(1−λ) − λ (Choe–Ramdas / sub-exp)."""
    lam_clipped = _clip_lam(lam)
    return float(-np.log(1.0 - lam_clipped) - lam_clipped)


def _align_loss_diff(
    loss_a: Array | None,
    loss_b: Array | None,
    d: Array | None,
) -> Array:
    """Build d_t = L_A,t − L_B,t (smaller loss better). Finite pairs only."""
    if d is not None:
        if loss_a is not None or loss_b is not None:
            raise ValueError("pass either d or (loss_a, loss_b), not both")
        diff = np.asarray(d, dtype=float).reshape(-1)
    else:
        if loss_a is None or loss_b is None:
            raise ValueError("need loss_a and loss_b, or d")
        a = np.asarray(loss_a, dtype=float).reshape(-1)
        b = np.asarray(loss_b, dtype=float).reshape(-1)
        if a.shape != b.shape:
            raise ValueError("loss series must align")
        diff = a - b
    return diff[np.isfinite(diff)]


def _predictable_bound_scale(d: Array, initial_bound: float) -> Array:
    """b_t = max(initial_bound, max_{s<t}|d_s|); b_0 uses initial only."""
    b0 = float(initial_bound)
    if not np.isfinite(b0) or b0 <= 0.0:
        raise ValueError("initial_bound must be finite and > 0")
    n = int(d.size)
    if n == 0:
        return np.asarray([], dtype=float)
    out = np.empty(n, dtype=float)
    run = b0
    out[0] = run
    abs_d = np.abs(d)
    for t in range(1, n):
        run = max(run, float(abs_d[t - 1]))
        out[t] = run
    return out


def e_process_loss_diff(
    loss_a: Array | None = None,
    loss_b: Array | None = None,
    *,
    d: Array | None = None,
    lam: float = 0.25,
    initial_bound: float = 1.0,
) -> Array:
    """Anytime-valid capital process for H0: E[d] ≤ 0 (research-only).

    ``d_t = L_{A,t} − L_{B,t}`` (smaller loss better). Rejecting / wealth
    growth is evidence that A is systematically **worse** than B
    (Choe–Ramdas weak null style on the loss differential).

    Construction (bounded exponential betting):
    1. Align losses → finite ``d``.
    2. Predictable scale ``b_t = max(initial_bound, max_{s<t}|d_s|)``.
    3. Soft observation ``x_t = clip(d_t / b_t, −1, 1)``.
    4. One-step factor ``e_t = exp(λ x_t − ψ_E(λ) x_t²)`` with
       ``ψ_E(λ) = −log(1−λ) − λ``, ``λ ∈ (0,1)``.
    5. Capital ``E_t = ∏_{s≤t} e_s`` (log-cumsum; clipped like coverage e-process).

    Under H0 with |x|≤1, ``(E_t)`` is a nonnegative supermartingale for fixed
    predictable λ; Ville via ``e_process_threshold`` (default level 0.05 → 20).
    Soft truncation / post-hoc unbounded losses → research diagnostic only,
    not a live-performance or promotion claim. Empty → ``[1.0]``.
    """
    diff = _align_loss_diff(loss_a, loss_b, d)
    if diff.size == 0:
        return np.array([1.0], dtype=float)
    lam_clipped = _clip_lam(lam)
    psi = _psi_exp(lam_clipped)
    scale = _predictable_bound_scale(diff, initial_bound)
    x = np.clip(diff / scale, -1.0, 1.0)
    log_steps = lam_clipped * x - psi * (x * x)
    log_steps = np.clip(log_steps, -np.log(_E_MAX), np.log(_E_MAX))
    log_run = np.minimum(np.cumsum(log_steps), np.log(_E_MAX))
    return np.exp(log_run).astype(float)


def e_process_dm(
    loss_a: Array,
    loss_b: Array,
    *,
    lam: float = 0.25,
    initial_bound: float = 1.0,
    level: float = 0.05,
) -> dict[str, object]:
    """DM-style wrapper: e-process on loss_a − loss_b plus Ville decision.

    Keys: ``e``, ``e_final``, ``mean_loss_diff``, ``n``, plus threshold fields
    from ``e_process_threshold`` (``first_cross``, ``reject``, ``threshold``).
    Research-diagnostic only.
    """
    a = np.asarray(loss_a, dtype=float).reshape(-1)
    b = np.asarray(loss_b, dtype=float).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("loss series must align")
    diff = a - b
    finite = np.isfinite(diff)
    d = diff[finite]
    n = int(d.size)
    path = e_process_loss_diff(d=d, lam=lam, initial_bound=initial_bound)
    decision = e_process_threshold(path, level=level)
    return {
        "e": path,
        "e_final": float(path[-1]),
        "mean_loss_diff": float(np.mean(d)) if n else float("nan"),
        "n": n,
        "first_cross": decision["first_cross"],
        "reject": bool(decision["reject"]),
        "threshold": float(decision["threshold"]),  # type: ignore[arg-type]
    }
