"""Transaction cost building blocks."""

from __future__ import annotations

import numpy as np

from quant_fund.config.models import CostConfig


def half_spread_cost(notional: float, half_spread_bps: float) -> float:
    return abs(notional) * half_spread_bps / 1e4


def commission_cost(notional: float, commission_bps: float) -> float:
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
    if q == 0:
        return 0.0
    notional = q * float(price)
    adv = max(float(adv_dollars), 1e-12)
    participation = notional / adv
    return float(notional * upsilon * max(sigma, 0.0) * np.sqrt(participation))


def total_cost(
    quantity: float,
    price: float,
    adv_dollars: float,
    sigma: float,
    config: CostConfig,
) -> dict[str, float | str]:
    if config.frictionless:
        return {
            "commission": 0.0,
            "spread": 0.0,
            "impact": 0.0,
            "borrow": 0.0,
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
