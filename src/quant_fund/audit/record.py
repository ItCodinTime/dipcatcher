"""Record existing research receipts and simulated paper artifacts.

Reading those files does not rewrite them and does not call the broker.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

import polars as pl

from quant_fund.audit.canonical import canonical_json_bytes, json_safe
from quant_fund.audit.errors import AuditError
from quant_fund.audit.ledger import KINDS, AuditLedger, LedgerEntry
from quant_fund.audit.trace import receipt_digest

_REJECTED = frozenset({"rejected", "cancelled", "canceled"})


def record_research_receipt(ledger: AuditLedger, path: Path | str) -> LedgerEntry:
    """Append a ``research_run`` entry bound to the receipt digest and provenance."""
    receipt = Path(path)
    before = receipt.read_bytes()
    try:
        parsed = json_loads(before)
    except (UnicodeError, ValueError) as exc:
        raise AuditError(f"research receipt is not JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise AuditError("research receipt must be a JSON object")
    notebook: dict[str, Any] = parsed
    provenance = notebook.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    payload: dict[str, Any] = {
        "receipt_sha256": receipt_digest(notebook),
        "receipt_name": receipt.name,
        "run_id": _text(provenance.get("run_id")),
        "git_revision": _text(provenance.get("git_revision")),
        "git_worktree_sha256": _text(provenance.get("git_worktree_sha256")),
        "config_sha256": _text(provenance.get("config_sha256")),
        "dataset_sha256": _text(provenance.get("dataset_sha256")),
        "dataset_content_sha256": _text(provenance.get("dataset_content_sha256")),
        "northset_inputs_sha256": _text(provenance.get("northset_inputs_sha256")),
        "execution_claim": _text(provenance.get("execution_claim")),
        "point_in_time": (
            provenance.get("point_in_time")
            if isinstance(provenance.get("point_in_time"), bool)
            else None
        ),
        "data_source": _text(notebook.get("data_source")),
        "live_pnl_claim": False,
    }
    code_sha = notebook.get("code_sha256", provenance.get("code_sha256"))
    if isinstance(code_sha, str):
        payload["code_sha256"] = code_sha
    entry = ledger.append("research_run", payload)
    if receipt.read_bytes() != before:
        raise AuditError("research receipt changed during recording")
    return entry


def record_paper_directory(ledger: AuditLedger, paper_dir: Path | str) -> list[LedgerEntry]:
    """Append paper, order, fill, and risk entries for one simulated paper run.

    ``orders.parquet`` is the source. The parquet bytes are hashed and left
    unchanged. Fills are recorded only when a finite fill price is present.
    """
    root = Path(paper_dir)
    orders_path = root / "orders.parquet"
    if not orders_path.is_file():
        raise AuditError(f"simulated orders not found: {orders_path}")
    before = orders_path.read_bytes()
    frame = pl.read_parquet(io.BytesIO(before))
    rows = frame.to_dicts()
    safe_rows = [json_safe(row) for row in rows]
    if not isinstance(safe_rows, list):
        raise AuditError("orders parquet did not decode to rows")
    equity_path = root / "equity.parquet"
    equity_hash = (
        hashlib.sha256(equity_path.read_bytes()).hexdigest() if equity_path.is_file() else None
    )
    entries: list[LedgerEntry] = [
        ledger.append(
            "paper_decision",
            {
                "paper_dir": root.name,
                "n_orders": len(safe_rows),
                "orders_sha256": hashlib.sha256(canonical_json_bytes(safe_rows)).hexdigest(),
                "orders_file_sha256": hashlib.sha256(before).hexdigest(),
                "equity_file_sha256": equity_hash,
                "simulation_only": True,
                "live_pnl_claim": False,
                "execution_claim": "simulated_paper",
            },
        )
    ]
    for row in safe_rows:
        if not isinstance(row, dict):
            raise AuditError("order row was not an object")
        row_hash = hashlib.sha256(canonical_json_bytes(row)).hexdigest()
        entries.append(
            ledger.append(
                "simulated_order",
                {
                    "order_id": _text(row.get("order_id")),
                    "security_id": _text(row.get("security_id")),
                    "side": _text(row.get("side")),
                    "quantity": row.get("quantity"),
                    "status": _text(row.get("status")),
                    "simulation_only": True,
                    "live_pnl_claim": False,
                    "row_sha256": row_hash,
                },
            )
        )
        if _finite(row.get("fill_price")) and _finite(row.get("fill_qty")):
            entries.append(
                ledger.append(
                    "fill",
                    {
                        "order_id": _text(row.get("order_id")),
                        "security_id": _text(row.get("security_id")),
                        "side": _text(row.get("side")),
                        "price": row.get("fill_price"),
                        "quantity": row.get("fill_qty"),
                        "fill_id": _text(row.get("fill_id")),
                        "simulation_only": True,
                        "live_pnl_claim": False,
                        "row_sha256": row_hash,
                    },
                )
            )
        reason = row.get("reject_reason")
        status = str(row.get("status") or "")
        rejected = bool(reason) or status.lower() in _REJECTED
        entries.append(
            ledger.append(
                "risk_decision",
                {
                    "order_id": _text(row.get("order_id")),
                    "accepted": not rejected,
                    "reason": reason if isinstance(reason, str) and reason else None,
                    "simulation_only": True,
                    "live_pnl_claim": False,
                },
            )
        )
    if orders_path.read_bytes() != before:
        raise AuditError("orders parquet changed during recording")
    if any(entry.kind not in KINDS for entry in entries):
        raise AuditError("recorder emitted an unknown kind")
    return entries


def json_loads(raw: bytes) -> Any:
    return json.loads(raw)


def _text(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _finite(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))
