"""Cross-framework backtest parity harness.

Runs identical reference strategies through this repository's backtest engine
(``quant_fund.backtest.engine.run_backtest``) and optional third-party engines
(vectorbt, backtrader, NautilusTrader), then reconciles fills and equity line
by line. Every emitted artifact is labeled SYNTHETIC and research-only.

Optional engines are imported lazily; a missing library yields a SKIP status,
never a fabricated result.
"""

from __future__ import annotations

__all__ = [
    "canonical",
    "synthetic",
    "strategies",
    "reconcile",
    "report",
    "shootout",
]
