"""GBM Monte Carlo and discrete delta-hedge error.

Combines davidalmeida90/quant-models ``monte-carlo-gbm`` with
``delta-hedging-error-monte-carlo``. When realised vol equals implied,
mean hedge P&L is financing noise around zero; the spread is gamma ×
discrete rebalancing. Research diagnostic, not a live book.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.quant_models.black_scholes import bs_price
from quant_fund.quant_models.greeks import greeks

Array = NDArray[np.float64]


def gbm_paths(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    *,
    rng: np.random.Generator | None = None,
    antithetic: bool = True,
) -> Array:
    """Exact GBM transitions. Last axis is time including ``t=0``. Shape ``(n_paths, n_steps+1)``."""
    if n_steps < 1 or n_paths < 1:
        raise ValueError("n_steps and n_paths must be positive")
    gen = rng if rng is not None else np.random.default_rng()
    dt = T / n_steps
    n_draw = (n_paths + 1) // 2 if antithetic else n_paths
    z = gen.standard_normal((n_draw, n_steps))
    if antithetic:
        z = np.concatenate([z, -z], axis=0)[:n_paths]
    drift = (mu - 0.5 * sigma**2) * dt
    incr = drift + sigma * np.sqrt(dt) * z
    log_s = np.concatenate([np.full((n_paths, 1), np.log(S0)), incr], axis=1)
    return np.exp(np.cumsum(log_s, axis=1))


def one_day_short_call_hedge(
    tau_days: int,
    *,
    S0: float = 100.0,
    K: float = 100.0,
    r: float = 0.02,
    sigma: float = 0.18,
    life_days: int = 20,
    n_paths: int = 8_000,
    seed: int = 11,
) -> tuple[Array, Array]:
    """P&L of a short ATM call, delta-hedged once over one day.

    Common random numbers across ``tau_days`` (source notebook). Returns
    ``(moneyness, pnl)``.
    """
    if tau_days < 1:
        raise ValueError("tau_days must be >= 1")
    dt = 1.0 / 252.0
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(n_paths)
    z2 = rng.standard_normal(n_paths)
    elapsed = (life_days - tau_days) * dt
    s = S0 * np.exp((r - 0.5 * sigma**2) * elapsed + sigma * np.sqrt(max(elapsed, 0.0)) * z1)
    s_next = s * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z2)
    t0 = tau_days * dt
    t1 = (tau_days - 1) * dt
    c0 = bs_price(s, K, t0, r, 0.0, sigma, "call")
    if t1 <= 1e-9:
        c1 = np.maximum(s_next - K, 0.0)
    else:
        c1 = bs_price(s_next, K, t1, r, 0.0, sigma, "call")
    hedge = np.asarray(greeks(s, K, t0, r, 0.0, sigma, "call")["delta"], dtype=float)
    cash = c0 - hedge * s
    pnl = -(c1 - c0) + hedge * (s_next - s) + cash * np.expm1(r * dt)
    return np.asarray(s / K, dtype=float), np.asarray(pnl, dtype=float)


def last_day_rebalanced(
    n_rebalances: int,
    *,
    S0: float = 100.0,
    K: float = 100.0,
    r: float = 0.02,
    sigma: float = 0.18,
    n_paths: int = 20_000,
    seed: int = 3,
) -> tuple[float, float]:
    """Last trading day, spot opening at the strike, ``n_rebalances`` delta resets.

    Returns ``(pnl_std, pnl_mean)``.
    """
    if n_rebalances < 1:
        raise ValueError("n_rebalances must be positive")
    dt = 1.0 / 252.0
    h = dt / n_rebalances
    g = np.random.default_rng(seed)
    s = np.full(n_paths, float(S0 if S0 else K))
    cash = np.full(n_paths, float(bs_price(s[0], K, dt, r, 0.0, sigma, "call")))
    shares = np.zeros(n_paths)
    for i in range(n_rebalances):
        tau = dt - i * h
        hedge = np.asarray(greeks(s, K, tau, r, 0.0, sigma, "call")["delta"], dtype=float)
        cash -= (hedge - shares) * s
        shares = hedge
        s = s * np.exp((r - 0.5 * sigma**2) * h + sigma * np.sqrt(h) * g.standard_normal(n_paths))
        cash *= np.exp(r * h)
    pnl = cash + shares * s - np.maximum(s - K, 0.0)
    return float(pnl.std()), float(pnl.mean())


def hedge_error_summary(
    *,
    tau_days: tuple[int, ...] = (10, 5, 2, 1),
    n_paths: int = 8_000,
    seed: int = 11,
) -> dict[str, Any]:
    """Compact receipt of one-day hedge error vs maturity. Not a live P&L claim."""
    rows = []
    for tau in tau_days:
        m, pnl = one_day_short_call_hedge(tau, n_paths=n_paths, seed=seed)
        near = np.abs(m - 1.0) < 0.01
        rows.append(
            {
                "T_days": int(tau),
                "mean_pnl": float(pnl.mean()),
                "std_pnl": float(pnl.std()),
                "atm_std": float(pnl[near].std()) if near.any() else float("nan"),
            }
        )
    return {"research_only": True, "live_pnl_claim": False, "by_maturity": rows}
