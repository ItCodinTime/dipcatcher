"""Fuzz the paper-ledger readers/validators (read-only tooling only).

First byte selects a ledger file slot; the rest of the input becomes that
file's content inside a fresh run dir, then ``validate_ledger_schema`` runs
over the whole dir. The validator is designed to report, never raise — any
exception is a finding. A second mode feeds JSON straight into
``validate_promotion_dry_run_receipt``.

Never touches order paths: this only exercises the read/validate surface.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fuzz_common import run

from quant_fund.paper.ledger import (
    latest_run_id,
    load_broker_state,
    validate_ledger_schema,
    validate_promotion_dry_run_receipt,
)

EXPECTED: tuple[type[BaseException], ...] = ()

SLOTS = (
    "meta.json",
    "broker_state.json",
    "promotion_dry_run.json",
    "analytics_export.json",
    "equity.parquet",
    "orders.parquet",
    "cash_ledger.parquet",
    "positions.parquet",
    "latest_run.json",  # written one level up, into the paper subdir
    "PROMO_JSON",  # feed validate_promotion_dry_run_receipt directly
)


def test_one_input(data: bytes) -> None:
    if len(data) < 2:
        return
    slot = SLOTS[data[0] % len(SLOTS)]
    body = data[1:]
    if slot == "PROMO_JSON":
        try:
            promo = json.loads(body.decode("utf-8", errors="strict"))
        except ValueError:
            return
        validate_promotion_dry_run_receipt(promo)
        return
    with tempfile.TemporaryDirectory(prefix="fuzz_ledger_") as tmp:
        data_root = Path(tmp)
        run_dir = data_root / "metadata" / "paper" / "run-1"
        run_dir.mkdir(parents=True)
        if slot == "latest_run.json":
            run_dir.parent.joinpath(slot).write_bytes(body)
        else:
            run_dir.joinpath(slot).write_bytes(body)
        validate_ledger_schema(run_dir)
        load_broker_state(data_root, "run-1")
        latest_run_id(data_root)


if __name__ == "__main__":
    # isolate: corrupt parquet can abort the polars engine (OOM/panic) —
    # fork per input so the crash is recorded, not fatal to the campaign.
    raise SystemExit(run("ledger", test_one_input, expected=EXPECTED, isolate=True))
