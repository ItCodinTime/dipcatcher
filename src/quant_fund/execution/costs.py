"""Transaction cost building blocks."""

from __future__ import annotations

import numpy as np

from quant_fund.config.models import CostConfig


def half_spread_cost(notional: float, half_spread_bps: float) -> float:
    if not np.isfinite(notional) or not np.isfinite(half_spread_bps) or half_spread_bps < 0:
        raise ValueError(
            "notional must be finite and half_spread_bps must be finite and non-negative"
        )
    return abs(notional) * half_spread_bps / 1e4


def commission_cost(notional: float, commission_bps: float) -> float:
    if not np.isfinite(notional) or not np.isfinite(commission_bps) or commission_bps < 0:
        raise ValueError(
            "notional must be finite and commission_bps must be finite and non-negative"
        )
    return abs(notional) * commission_bps / 1e4


def sqrt_impact(
    quantity: float,
    price: float,
    adv_dollars: float,
    sigma: float,
    upsilon: float,
) -> float:
    """Impact in dollars: notional * Y * sigma * sqrt(|Q|/ADV_shares) with ADV in shares.

    Using dollar ADV: participation = |qty|*price / ADV_dollars.
    Zero quantity → zero impact.
    """
    q = abs(float(quantity))
    if not np.isfinite(q) or not np.isfinite(price) or price <= 0:
        raise ValueError("quantity must be finite and price must be finite and positive")
    if not np.isfinite(adv_dollars) or adv_dollars <= 0:
        raise ValueError("adv_dollars must be finite and positive")
    if not np.isfinite(upsilon) or upsilon < 0:
        raise ValueError("upsilon must be finite and non-negative")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and non-negative")
    if q == 0:
        return 0.0
    notional = q * float(price)
    adv = max(float(adv_dollars), 1e-12)
    participation = notional / adv
    return float(notional * upsilon * sigma * np.sqrt(participation))


def total_cost(
    quantity: float,
    price: float,
    adv_dollars: float,
    sigma: float,
    config: CostConfig,
) -> dict[str, float | str]:
    if not np.isfinite(quantity) or not np.isfinite(price) or price <= 0:
        raise ValueError("quantity must be finite and price must be finite and positive")
    if not np.isfinite(adv_dollars) or adv_dollars <= 0:
        raise ValueError("adv_dollars must be finite and positive")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and non-negative")
    for name in ("commission_bps", "half_spread_bps", "impact_y", "bps_per_turnover"):
        value = float(getattr(config, name))
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if config.frictionless:
        return {
            "commission": 0.0,
            "spread": 0.0,
            "impact": 0.0,
            "borrow": 0.0,
            "turnover_bps": 0.0,
            "total": 0.0,
            "label": "FRICTIONLESS RESEARCH ONLY",
        }
    notional = abs(quantity) * price
    commission = commission_cost(notional, config.commission_bps)
    spread = half_spread_cost(notional, config.half_spread_bps)
    impact = sqrt_impact(quantity, price, adv_dollars, sigma, config.impact_y)
    bps_to = abs(notional) * config.bps_per_turnover / 1e4
    total = commission + spread + impact + bps_to
    return {
        "commission": commission,
        "spread": spread,
        "impact": impact,
        "borrow": 0.0,
        "turnover_bps": bps_to,
        "total": total,
        "label": "costed",
    }
