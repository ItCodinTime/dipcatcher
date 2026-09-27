"""Simulation-only specification of order lifecycle and accounting.

Nothing in this package submits orders, routes them, or states a live P&L.
"""

from quant_fund.formal.accounting import (
    Account,
    algebraic_total_cost,
    apply_trade,
    dividend_yield,
    split_market_value,
)
from quant_fund.formal.order_lifecycle import check_trace, enumerate_safety

__all__ = [
    "Account",
    "algebraic_total_cost",
    "apply_trade",
    "check_trace",
    "dividend_yield",
    "enumerate_safety",
    "split_market_value",
]
