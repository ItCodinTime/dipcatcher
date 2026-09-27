"""Research stress-testing and scenario engine.

Simulation and diagnostics only. This package does not submit orders, talk to
a broker, or rewrite sealed research receipts.

NumPy, SciPy, and scikit-learn load with the report and replay engines, not
when the CLI imports ``stress.cli`` (the crises command only needs the catalog).
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

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

_EXPORTS: dict[str, str] = {
    "CRISIS_CATALOG": "quant_fund.stress.catalog",
    "Crisis": "quant_fund.stress.catalog",
    "crisis_by_id": "quant_fund.stress.catalog",
    "ResearchStrategy": "quant_fund.stress.strategy",
    "build_stress_report": "quant_fund.stress.report",
    "render_html": "quant_fund.stress.report",
    "render_markdown": "quant_fund.stress.report",
    "replay_crisis": "quant_fund.stress.replay",
    "replay_portfolio": "quant_fund.stress.replay",
    "reverse_stress": "quant_fund.stress.reverse",
    "worst_linear_scenario": "quant_fund.stress.reverse",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
