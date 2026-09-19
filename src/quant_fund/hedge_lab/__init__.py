"""Artificial Hedge fund lab — economic diagnostics on a paper book."""

from quant_fund.hedge_lab.runner import HedgeLabProtocol, run_hedge_lab
from quant_fund.hedge_lab.scoreboard import book_economic_scoreboard

__all__ = ["HedgeLabProtocol", "book_economic_scoreboard", "run_hedge_lab"]
