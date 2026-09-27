"""Recording a simulated paper directory leaves the parquet untouched."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from quant_fund.audit.ledger import AuditLedger
from quant_fund.audit.record import record_paper_directory
from quant_fund.audit.signing import Ed25519Signer


def test_paper_rows_become_order_fill_and_risk_entries(tmp_path: Path) -> None:
    paper = tmp_path / "paper"
    paper.mkdir()
    frame = pl.DataFrame(
        {
            "order_id": ["a", "b"],
            "security_id": ["EURUSD", "EURUSD"],
            "side": ["buy", "sell"],
            "quantity": [1.0, 2.0],
            "status": ["filled", "rejected"],
            "reject_reason": [None, "limit"],
            "fill_price": [1.1, None],
            "fill_qty": [1.0, None],
            "fill_id": ["f1", None],
        }
    )
    orders = paper / "orders.parquet"
    frame.write_parquet(orders)
    equity = paper / "equity.parquet"
    pl.DataFrame({"event_time": ["2020-01-01"], "nav": [1.0]}).write_parquet(equity)
    before_orders = orders.read_bytes()
    before_equity = equity.read_bytes()
    signer = Ed25519Signer.generate()
    ledger = AuditLedger(
        tmp_path / "ledger",
        signer=signer,
        sync=False,
        clock=lambda: "2020-01-01T00:00:00Z",
    )
    entries = record_paper_directory(ledger, paper)
    assert orders.read_bytes() == before_orders
    assert equity.read_bytes() == before_equity
    assert [entry.kind for entry in entries] == [
        "paper_decision",
        "simulated_order",
        "fill",
        "risk_decision",
        "simulated_order",
        "risk_decision",
    ]
    assert entries[3].payload["accepted"] is True
    assert entries[5].payload["accepted"] is False
    raw = ledger.entries_path.read_text(encoding="utf-8")
    assert '"nav"' not in raw
    assert '"sharpe"' not in raw
    assert '"pnl"' not in raw
    assert entries[0].payload["live_pnl_claim"] is False
    assert entries[0].payload["execution_claim"] == "simulated_paper"
