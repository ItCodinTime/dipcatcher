"""Phase 17 paper / shadow trading loop.

The package initializer deliberately keeps the ledger and clock surfaces lazy.
Importing ``quant_fund.paper.ledger`` must not import the forecasting stack: a
corrupt or optional forecasting backend must not prevent offline ledger
inspection, recovery, or schema validation.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "PaperLedger",
    "PaperLoopResult",
    "StaleValuationError",
    "build_scaled_challenger_weights",
    "ReplayClock",
    "WallClock",
    "latest_run_id",
    "load_broker_state",
    "paper_root",
    "promotion_dry_run",
    "run_paper_loop",
    "validate_ledger_schema",
    "validate_promotion_dry_run_receipt",
]

_LAZY_MODULES = {
    "ReplayClock": ("quant_fund.paper.clock", "ReplayClock"),
    "WallClock": ("quant_fund.paper.clock", "WallClock"),
    "PaperLedger": ("quant_fund.paper.ledger", "PaperLedger"),
    "latest_run_id": ("quant_fund.paper.ledger", "latest_run_id"),
    "load_broker_state": ("quant_fund.paper.ledger", "load_broker_state"),
    "paper_root": ("quant_fund.paper.ledger", "paper_root"),
    "promotion_dry_run": ("quant_fund.paper.ledger", "promotion_dry_run"),
    "validate_ledger_schema": ("quant_fund.paper.ledger", "validate_ledger_schema"),
    "validate_promotion_dry_run_receipt": (
        "quant_fund.paper.ledger",
        "validate_promotion_dry_run_receipt",
    ),
    "PaperLoopResult": ("quant_fund.paper.loop", "PaperLoopResult"),
    "StaleValuationError": ("quant_fund.paper.loop", "StaleValuationError"),
    "build_scaled_challenger_weights": ("quant_fund.paper.loop", "build_scaled_challenger_weights"),
    "run_paper_loop": ("quant_fund.paper.loop", "run_paper_loop"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _LAZY_MODULES[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
