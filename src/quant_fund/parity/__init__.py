"""Backtest-to-shadow parity.

Replays one strategy through the simulated broker on a backtest clock and
on a shadow clock, then attributes every divergent bar to a single cause.
No live order submission.
"""

from quant_fund.parity.checker import CAUSE_PRECEDENCE, attribute_pair, check_parity
from quant_fund.parity.reference import run_backtest_session
from quant_fund.parity.replay import ParityRun, ParityValuationError, ReplayOptions, replay_session
from quant_fund.parity.report import assert_clean_report, build_report, write_report
from quant_fund.parity.session import Bar, MarketSession
from quant_fund.parity.shadow import run_shadow_session
from quant_fund.parity.shortfall import attribute_shortfall
from quant_fund.parity.strategy import (
    DecisionContext,
    FixedWeightStrategy,
    TargetDecision,
    round_weight,
)
from quant_fund.parity.trace import code_path_guard

__all__ = [
    "CAUSE_PRECEDENCE",
    "Bar",
    "DecisionContext",
    "FixedWeightStrategy",
    "MarketSession",
    "ParityRun",
    "ParityValuationError",
    "ReplayOptions",
    "TargetDecision",
    "assert_clean_report",
    "attribute_pair",
    "attribute_shortfall",
    "build_report",
    "check_parity",
    "code_path_guard",
    "replay_session",
    "round_weight",
    "run_backtest_session",
    "run_shadow_session",
    "write_report",
]
