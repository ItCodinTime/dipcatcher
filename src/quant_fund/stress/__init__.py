"""Research stress-testing and scenario engine.

Simulation and diagnostics only. This package does not submit orders, talk to
a broker, or rewrite sealed research receipts.
"""

from quant_fund.stress.catalog import CRISIS_CATALOG, Crisis, crisis_by_id
from quant_fund.stress.replay import replay_crisis, replay_portfolio
from quant_fund.stress.report import build_stress_report, render_html, render_markdown
from quant_fund.stress.reverse import reverse_stress, worst_linear_scenario
from quant_fund.stress.strategy import ResearchStrategy

__all__ = [
    "CRISIS_CATALOG",
    "Crisis",
    "ResearchStrategy",
    "build_stress_report",
    "crisis_by_id",
    "render_html",
    "render_markdown",
    "replay_crisis",
    "replay_portfolio",
    "reverse_stress",
    "worst_linear_scenario",
]
