"""Deterministic sampling helpers for the research backend."""

from __future__ import annotations

import numpy as np


def nucleus_sample(
    logits, rng: np.random.Generator, *, temperature: float = 1.0, top_p: float = 0.9
) -> int:
    x = np.asarray(logits, dtype=float).reshape(-1)
    if x.size == 0:
        raise ValueError("logits must be non-empty")
    if not np.isfinite(x).any():
        return 0
    t = max(float(temperature), 1e-8)
    x = np.where(np.isfinite(x), x, -np.inf) / t
    order = np.argsort(-x)
    shifted = x[order] - np.max(x[order])
    probs = np.exp(shifted)
    probs /= max(float(probs.sum()), 1e-300)
    p = float(np.clip(top_p, 1e-8, 1.0))
    cum = np.cumsum(probs)
    keep = int(np.searchsorted(cum, p, side="left")) + 1
    keep = max(1, min(keep, len(order)))
    chosen = order[:keep]
    q = probs[:keep]
    q /= q.sum()
    return int(rng.choice(chosen, p=q))
