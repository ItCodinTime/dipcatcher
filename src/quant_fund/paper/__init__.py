"""Phase 17 paper / shadow trading loop."""

from quant_fund.paper.clock import ReplayClock, WallClock
from quant_fund.paper.ledger import (
    PaperLedger,
    latest_run_id,
    load_broker_state,
    paper_root,
    promotion_dry_run,
    validate_ledger_schema,
    validate_promotion_dry_run_receipt,
)
from quant_fund.paper.loop import (
    PaperLoopResult,
    StaleValuationError,
    build_scaled_challenger_weights,
    run_paper_loop,
)
from quant_fund.paper.recon import (
    reconcile_broker_states,
    reconcile_equity,
    reconcile_fills,
)

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
    "reconcile_broker_states",
    "reconcile_equity",
    "reconcile_fills",
    "run_paper_loop",
    "validate_ledger_schema",
    "validate_promotion_dry_run_receipt",
]
