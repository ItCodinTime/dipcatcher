"""Cost-aware research comparisons; never a deployment/promotion authority."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.snooping import reality_check, spa_test

Array = NDArray[np.float64]
Allocator = Callable[[Array, Array], Array]


def serial_adjusted_sharpe(
    excess_returns: Array, *, periods: int = 252, max_lag: int = 20
) -> dict[str, float]:
    """Lo (2002) stationary, arithmetic time-aggregation Sharpe estimate.

    SR(q) = sqrt(q)*mean / sqrt(gamma0 + 2 sum_(k=1)^L (1-k/q)*gamma_k).
    Autocovariances beyond L are assumed zero; this is not an IID confidence
    interval, compounded-return Sharpe, or a correction for selection bias.
    All autocovariances use denominator T. Missing dates must be resolved by
    the caller; non-finite observations are rejected rather than compressed.
    """
    r = np.asarray(excess_returns, dtype=float)
    if r.ndim != 1 or r.size < 3 or not np.isfinite(r).all():
        raise ValueError("excess_returns must be finite with at least three observations")
    if isinstance(periods, bool) or not isinstance(periods, int) or periods < 1:
        raise ValueError("periods must be a positive integer")
    if isinstance(max_lag, bool) or not isinstance(max_lag, int) or not 0 <= max_lag < r.size:
        raise ValueError("max_lag must be an integer in [0, T)")
    lag = min(max_lag, periods - 1)
    centered = r - r.mean()
    gamma0 = float(centered @ centered / r.size)
    variance = gamma0 + 2 * sum(
        (1 - k / periods) * float(centered[k:] @ centered[:-k] / r.size) for k in range(1, lag + 1)
    )
    if gamma0 <= 0 or variance <= 0:
        raise ValueError("Sharpe is undefined for nonpositive variance estimates")
    return {
        "iid": float(np.sqrt(periods) * r.mean() / np.sqrt(gamma0)),
        "serial_adjusted": float(np.sqrt(periods) * r.mean() / np.sqrt(variance)),
        "variance_inflation": variance / gamma0,
        "lags": float(lag),
    }


@dataclass(frozen=True)
class ResearchPath:
    start: int
    weights: Array
    gross_returns: Array
    net_returns: Array
    cost_fraction: Array
    turnover: Array


def walk_forward_allocations(
    returns: Array, allocator: Allocator, *, lookback: int, fee_rate: float
) -> ResearchPath:
    """Rebalance a long-only cash-funded book using strictly preceding returns.

    The callback receives a COPY of returns[t-lookback:t] and drifted risky
    weights, never returns[t:]. Weights describe the post-cost portfolio.
    Transaction fees obey self-financing c = fee*sum(abs((1-c)*target-prev));
    net growth is (1-c)*(1+target @ returns[t]). Initial cash and entry costs
    are included; final positions are marked, not liquidated. fee_rate is per
    unit of one-way traded risky notional; cash trades incur no fees. No short,
    leverage, impact, borrow, or financing model is implied.
    """
    r = np.asarray(returns, dtype=float)
    if r.ndim != 2 or r.shape[1] < 1 or not np.isfinite(r).all() or np.any(r < -1):
        raise ValueError("returns must be a finite T x N matrix with entries >= -1")
    if isinstance(lookback, bool) or not isinstance(lookback, int) or not 2 <= lookback < len(r):
        raise ValueError("lookback must be an integer in [2, T)")
    if not np.isfinite(fee_rate) or not 0 <= fee_rate < 1:
        raise ValueError("fee_rate must lie in [0, 1)")
    n = r.shape[1]
    weights, gross, net, fees, turnover = [], [], [], [], []
    previous = np.zeros(n)
    for t in range(lookback, len(r)):
        target = np.asarray(allocator(r[t - lookback : t].copy(), previous.copy()), dtype=float)
        if (
            target.shape != (n,)
            or not np.isfinite(target).all()
            or np.any(target < 0)
            or target.sum() > 1 + 1e-12
        ):
            raise ValueError("allocator must return nonnegative finite N weights summing to <= 1")
        # Bisection of monotone f(c); avoids a near-unit-cost slow fixed point.
        low, high = 0.0, 1.0
        for _ in range(60):
            mid = (low + high) / 2
            f = mid - fee_rate * float(np.abs((1 - mid) * target - previous).sum())
            if f > 0:
                high = mid
            else:
                low = mid
        cost = (low + high) / 2
        traded = float(np.abs((1 - cost) * target - previous).sum())
        if abs(cost - fee_rate * traded) > 1e-10:
            raise ValueError("self-financing fee equation failed")
        gain = float(target @ r[t])
        if 1 + gain <= 0:
            raise ValueError("portfolio is bankrupt")
        weights.append(target.copy())
        gross.append(gain)
        net.append((1 - cost) * (1 + gain) - 1)
        fees.append(cost)
        turnover.append(traded)
        previous = target * (1 + r[t]) / (1 + gain)
    return ResearchPath(
        lookback,
        np.asarray(weights),
        np.asarray(gross),
        np.asarray(net),
        np.asarray(fees),
        np.asarray(turnover),
    )


def compare_research_paths(
    baseline: Array, candidates: Array, *, n_boot: int = 999, block: float = 10.0, seed: int = 0
) -> dict[str, Any]:
    """Compare aligned, net-of-cost OOS returns across ALL attempted candidates.

    White RC / Hansen SPA test positive mean improvement over the baseline.
    They do not establish risk-adjusted superiority or correct omitted trials.
    Caller must supply same dates and retain a separate untouched final holdout.
    """
    b, c = np.asarray(baseline, dtype=float), np.asarray(candidates, dtype=float)
    if (
        b.ndim != 1
        or c.ndim != 2
        or c.shape[0] != b.size
        or b.size < 20
        or c.shape[1] < 1
        or not np.isfinite(b).all()
        or not np.isfinite(c).all()
    ):
        raise ValueError("finite aligned baseline T and candidates T x K required, T >= 20")
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 99:
        raise ValueError("n_boot must be an integer >= 99")
    if not np.isfinite(block) or not 1 <= block <= b.size:
        raise ValueError("block must lie in [1, T]")
    differences = c - b[:, None]
    return {
        "claim": "research_only",
        "promotable": False,
        "n_observations": int(b.size),
        "n_trials_supplied": int(c.shape[1]),
        "mean_net_improvement": differences.mean(axis=0).tolist(),
        "reality_check": asdict(reality_check(differences, n_boot=n_boot, block=block, seed=seed)),
        "spa": asdict(spa_test(differences, n_boot=n_boot, block=block, seed=seed)),
    }
