"""Dipcatcher research harness (Python package ``quant_fund``)."""

from __future__ import annotations

# One distribution. Hatch reads ``fx1.__version__`` ([tool.hatch.version]);
# this package does not keep a second literal.
from fx1 import __version__ as __version__

__firm__ = "Artificial Hedge"

from typing import TYPE_CHECKING, cast  # noqa: E402

# Stable library surface. Heavy research imports load on first use so
# ``import quant_fund`` stays light. See quant_fund.public.

if TYPE_CHECKING:
    from quant_fund.public import AppConfig as AppConfig
    from quant_fund.public import BacktestMetrics as BacktestMetrics
    from quant_fund.public import BacktestRun as BacktestRun
    from quant_fund.public import BookRiskOverlay as BookRiskOverlay
    from quant_fund.public import HypothesisResult as HypothesisResult
    from quant_fund.public import IngestPaths as IngestPaths
    from quant_fund.public import JsonValue as JsonValue
    from quant_fund.public import ResearchReceipt as ResearchReceipt
    from quant_fund.public import ResearchRun as ResearchRun
    from quant_fund.public import ResearchVerification as ResearchVerification
    from quant_fund.public import StaleValuationError as StaleValuationError
    from quant_fund.public import ingest as ingest
    from quant_fund.public import load_config as load_config
    from quant_fund.public import read_research_receipt as read_research_receipt
    from quant_fund.public import run_backtest as run_backtest
    from quant_fund.public import run_research as run_research
    from quant_fund.public import verify_research as verify_research

__all__ = [
    "__firm__",
    "__version__",
    "AppConfig",
    "BacktestMetrics",
    "BacktestRun",
    "BookRiskOverlay",
    "HypothesisResult",
    "IngestPaths",
    "JsonValue",
    "ResearchReceipt",
    "ResearchRun",
    "ResearchVerification",
    "StaleValuationError",
    "ingest",
    "load_config",
    "read_research_receipt",
    "run_backtest",
    "run_research",
    "verify_research",
]

_PACKAGE_ATTRS = frozenset({"__firm__", "__version__"})


def __getattr__(name: str) -> object:
    """Resolve a public name without importing the research stack up front."""
    if name not in __all__ or name in _PACKAGE_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import quant_fund.public as public_module

    value = cast(object, getattr(public_module, name))
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
