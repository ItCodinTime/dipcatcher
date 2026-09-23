"""Artificial Hedge fund lab — economic diagnostics on a paper book."""

from quant_fund.hedge_lab.gated_race import run_gated_race, run_holdout_confirm
from quant_fund.hedge_lab.lightspeed_book import run_lightspeed_file_book
from quant_fund.hedge_lab.mirror import negate_target_weights, pair_book_and_mirror
from quant_fund.hedge_lab.runner import HedgeLabProtocol, run_hedge_lab
from quant_fund.hedge_lab.scoreboard import book_economic_scoreboard
from quant_fund.hedge_lab.target_hunt import run_target_hunt

__all__ = [
    "HedgeLabProtocol",
    "book_economic_scoreboard",
    "negate_target_weights",
    "pair_book_and_mirror",
    "run_gated_race",
    "run_hedge_lab",
    "run_holdout_confirm",
    "run_lightspeed_file_book",
    "run_target_hunt",
]
