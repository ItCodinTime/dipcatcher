"""Execution cost stack for the research simulator.

Pieces: spread crossing, square-root impact, Almgren–Chriss temporary and
permanent impact, a bps commission schedule, and an Alpaca-style zero
commission plus SEC/TAF pass-through. Parameters are calibratable. This is
not a live fee quote and not a broker connection.

Almgren–Chriss dollar cost matches ``expected_shortfall_ac`` on a one-slice
trade: temporary ``eta / tau * q^2``, permanent ``gamma * q^2 / 2``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from quant_fund.config.models import CostConfig
from quant_fund.execution.costs import half_spread_cost, sqrt_impact, total_cost

FeeScheduleName = Literal["bps", "alpaca"]

# Schedule snapshot used when the caller does not override rates.
# SEC Section 31: dollars per $1,000,000 of sell principal.
# FINRA TAF: dollars per share sold, then ceiling to the cent, with a floor
# and a cap. Alpaca equity commission is zero; both fees apply to sells.
_DEFAULT_SEC_PER_MILLION = 27.80
_DEFAULT_TAF_PER_SHARE = 0.000166
_DEFAULT_TAF_MIN = 0.01
_DEFAULT_TAF_MAX = 8.30


def _ceil_cent(amount: float) -> float:
    if amount <= 0.0:
        return 0.0
    return math.ceil(amount * 100.0 - 1e-9) / 100.0


@dataclass(frozen=True)
class FeeSchedule:
    """Explicit commission schedule. ``alpaca`` is zero commission plus sell-side fees."""

    name: FeeScheduleName = "bps"
    commission_bps: float = 1.0
    sec_per_million: float = _DEFAULT_SEC_PER_MILLION
    taf_per_share: float = _DEFAULT_TAF_PER_SHARE
    taf_min: float = _DEFAULT_TAF_MIN
    taf_max: float = _DEFAULT_TAF_MAX

    def __post_init__(self) -> None:
        for label, value in (
            ("commission_bps", self.commission_bps),
            ("sec_per_million", self.sec_per_million),
            ("taf_per_share", self.taf_per_share),
            ("taf_min", self.taf_min),
            ("taf_max", self.taf_max),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{label} must be finite and non-negative")
        if self.taf_min > self.taf_max:
            raise ValueError("taf_min cannot exceed taf_max")


def alpaca_equity_schedule(**overrides: float) -> FeeSchedule:
    """Zero-commission equity schedule with calibratable SEC and TAF rates."""
    return FeeSchedule(name="alpaca", commission_bps=0.0, **overrides)


def schedule_fee(
    notional: float, quantity: float, *, is_sell: bool, schedule: FeeSchedule
) -> float:
    """Dollar fee for one fill. Buys on the Alpaca schedule pay zero."""
    if not math.isfinite(notional) or not math.isfinite(quantity):
        raise ValueError("notional and quantity must be finite")
    shares = abs(float(quantity))
    principal = abs(float(notional))
    if schedule.name == "bps":
        return principal * float(schedule.commission_bps) / 1e4
    if not is_sell or shares == 0.0 or principal == 0.0:
        return 0.0
    sec = _ceil_cent(principal * (float(schedule.sec_per_million) / 1_000_000.0))
    taf_raw = shares * float(schedule.taf_per_share)
    taf = _ceil_cent(taf_raw)
    if taf > 0.0:
        taf = min(float(schedule.taf_max), max(float(schedule.taf_min), taf))
    return float(sec + taf)


def spread_crossing_cost(
    notional: float,
    half_spread_bps: float,
    *,
    half_spread: float | None = None,
    quantity: float | None = None,
) -> float:
    """Dollar spread. Quote-based half-spread overrides the bps schedule when given."""
    if half_spread is not None and quantity is not None:
        if not math.isfinite(half_spread) or half_spread < 0.0:
            raise ValueError("half_spread must be finite and non-negative")
        if not math.isfinite(quantity):
            raise ValueError("quantity must be finite")
        return abs(float(quantity)) * float(half_spread)
    return float(half_spread_cost(notional, half_spread_bps))


def almgren_chriss_impact(
    quantity: float,
    *,
    eta: float,
    gamma: float,
    tau: float = 1.0,
    sigma: float = 1.0,
) -> dict[str, float]:
    """One-slice temporary and permanent impact in dollars.

    ``sigma`` scales both terms so a caller can fold volatility into the
    calibration (``sigma=1`` leaves the classic ``expected_shortfall_ac``
    coefficients unchanged).
    """
    q = float(quantity)
    if not math.isfinite(q):
        raise ValueError("quantity must be finite")
    for label, value in (("eta", eta), ("gamma", gamma), ("sigma", sigma)):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{label} must be finite and non-negative")
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be finite and > 0")
    if q == 0.0 or (eta == 0.0 and gamma == 0.0) or sigma == 0.0:
        return {"temporary": 0.0, "permanent": 0.0, "total": 0.0}
    scale = float(sigma)
    temporary = scale * (float(eta) / float(tau)) * q * q
    permanent = scale * float(gamma) * q * q / 2.0
    return {
        "temporary": float(temporary),
        "permanent": float(permanent),
        "total": float(temporary + permanent),
    }


def quote_execution_cost(
    quantity: float,
    price: float,
    adv_dollars: float,
    sigma: float,
    config: CostConfig,
    *,
    schedule: FeeSchedule,
    is_sell: bool,
    eta: float = 0.0,
    gamma: float = 0.0,
    tau: float = 1.0,
    ac_sigma: float | None = None,
    half_spread: float | None = None,
    include_spread: bool = True,
) -> dict[str, float | str]:
    """Full dollar cost. Frictionless configs return zeros with the research label.

    The legacy bps path delegates to ``total_cost`` so operation order matches
    the daily backtest when Almgren–Chriss terms are off and no book spread
    is supplied.
    """
    legacy = (
        schedule.name == "bps"
        and float(schedule.commission_bps) == float(config.commission_bps)
        and eta == 0.0
        and gamma == 0.0
        and half_spread is None
        and include_spread
        and (ac_sigma is None or ac_sigma == 1.0)
    )
    if legacy:
        return total_cost(quantity, price, adv_dollars, sigma, config)
    if config.frictionless:
        return total_cost(quantity, price, adv_dollars, sigma, config)
    if not math.isfinite(quantity) or not math.isfinite(price) or price <= 0.0:
        raise ValueError("quantity must be finite and price must be finite and positive")
    notional = abs(float(quantity)) * float(price)
    commission = schedule_fee(notional, quantity, is_sell=is_sell, schedule=schedule)
    if include_spread:
        spread = spread_crossing_cost(
            notional, config.half_spread_bps, half_spread=half_spread, quantity=quantity
        )
    else:
        spread = 0.0
    impact = float(sqrt_impact(quantity, price, adv_dollars, sigma, config.impact_y))
    ac = almgren_chriss_impact(
        quantity,
        eta=eta,
        gamma=gamma,
        tau=tau,
        sigma=1.0 if ac_sigma is None else float(ac_sigma),
    )
    impact += float(ac["total"])
    turnover = abs(notional) * float(config.bps_per_turnover) / 1e4
    total = commission + spread + impact + turnover
    return {
        "commission": float(commission),
        "spread": float(spread),
        "impact": float(impact),
        "impact_temporary": float(ac["temporary"]),
        "impact_permanent": float(ac["permanent"]),
        "borrow": 0.0,
        "turnover_bps": float(turnover),
        "total": float(total),
        "label": "costed",
    }
