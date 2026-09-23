"""Cox–Ross–Rubinstein binomial tree, European and American.

davidalmeida90/quant-models ``crr-binomial-tree/model.py``. Research
pricing only; American early-exercise is the reason to use the tree.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def crr_parameters(sigma: float, n: int, r: float, q: float, T: float) -> tuple[float, float, float, float]:
    if n < 1:
        raise ValueError("n must be a positive integer")
    if T <= 0:
        raise ValueError("T must be positive")
    dt = T / n
    u = float(np.exp(sigma * np.sqrt(dt)))
    d = float(np.exp(-sigma * np.sqrt(dt)))
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not np.isfinite(p) or p <= 0.0 or p >= 1.0:
        raise ValueError(f"CRR risk-neutral p={p} is outside (0, 1); check sigma/r/q/n")
    return u, d, float(p), dt


def crr_european(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    n: int,
    option_type: str = "call",
) -> float:
    """European CRR price (backward induction, O(n) per layer)."""
    kind = str(option_type).lower()
    if kind not in {"call", "put"}:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    u, d, p, dt = crr_parameters(sigma, n, r, q, T)
    disc = np.exp(-r * dt)
    j = np.arange(n + 1, dtype=float)
    spot = S * (u ** (n - j)) * (d**j)
    payoff = np.maximum(spot - K, 0.0) if kind == "call" else np.maximum(K - spot, 0.0)
    for i in range(n, 0, -1):
        payoff = disc * (p * payoff[:i] + (1.0 - p) * payoff[1 : i + 1])
    return float(payoff[0])


def crr_american(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    n: int,
    option_type: str = "call",
) -> float:
    """American CRR price with early exercise at every node."""
    kind = str(option_type).lower()
    if kind not in {"call", "put"}:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    u, d, p, dt = crr_parameters(sigma, n, r, q, T)
    disc = np.exp(-r * dt)
    j = np.arange(n + 1, dtype=float)
    spot = S * (u ** (n - j)) * (d**j)
    value = np.maximum(spot - K, 0.0) if kind == "call" else np.maximum(K - spot, 0.0)
    for i in range(n - 1, -1, -1):
        j_i = np.arange(i + 1, dtype=float)
        spot_i = S * (u ** (i - j_i)) * (d**j_i)
        cont = disc * (p * value[: i + 1] + (1.0 - p) * value[1 : i + 2])
        intrinsic = np.maximum(spot_i - K, 0.0) if kind == "call" else np.maximum(K - spot_i, 0.0)
        value = np.maximum(cont, intrinsic)
    return float(value[0])
