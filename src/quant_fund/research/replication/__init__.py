"""Faithful re-implementations of published equity anomalies/strategies.

Each strategy follows the original paper's signal definition and is scored with
proper scores and descriptive statistics only (the lab honesty contract forbids
Sharpe/Sortino/Calmar/P&L/NAV headlines — see ``research.catalog``).

Strategies (citations in ``strategies.STRATEGY_SPECS`` and docs/PAPER_REPLICATION.md):

- ``tsmom_mop2012`` — time-series momentum (Moskowitz, Ooi & Pedersen 2012).
- ``str_jegadeesh1990`` / ``str_lehmann1990`` — short-term reversal
  (Jegadeesh 1990; Lehmann 1990).
- ``lowvol_bbw2011`` / ``bab_fp2014`` — low-volatility / betting-against-beta
  (Baker, Bradley & Wurgler 2011; Frazzini & Pedersen 2014).
- ``overnight_lps2019`` / ``tug_of_war_lps2019`` — overnight-vs-intraday return
  split (Lou, Polk & Skouras 2019).
- ``tom_mx2008`` — turn-of-the-month timing (Lakonishok & Smidt 1988;
  McConnell & Xu 2008).

All evaluation outputs are research diagnostics. Receipts are sealed via the
``receipt_sha256`` convention shared with ``research.fleet_eval``.
"""

from quant_fund.research.replication.panel import (
    PANEL_SCHEMA_VERSION,
    load_silver_bars,
    market_returns,
)
from quant_fund.research.replication.strategies import STRATEGY_SPECS, StrategySpec

__all__ = [
    "PANEL_SCHEMA_VERSION",
    "STRATEGY_SPECS",
    "StrategySpec",
    "load_silver_bars",
    "market_returns",
]
