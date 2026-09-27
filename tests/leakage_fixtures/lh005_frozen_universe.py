"""SEEDED LEAK (LH005): frozen hard-coded universe (survivorship bias).

Deliberately leaky strategy file for the leakage-hunter CI gate
(DESIGN.md §6.4). MUST trip LH005. Do not import from strategy code.
"""

from __future__ import annotations

universe = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "META", "JPM"]


def members() -> list[str]:
    """Leaky: today's survivors hard-coded as the historical universe."""
    return list(universe)
