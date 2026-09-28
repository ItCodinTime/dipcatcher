"""P4.4 nautilus conformance lane — the attempt must be honest evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_fund.backtest.nautilus_conformance import (
    NAUTILUS_CONFORMANCE_SCHEMA,
    PINNED_NAUTULUS_VERSION,
    _compare_fills,
    _fill_ledger,
    conformance_equivalence_spec,
    nautilus_conformance_consistency_errors,
    nautilus_conformance_contract_errors,
    nautilus_status,
    run_nautilus_conformance,
    run_nautilus_conformance_eval,
    write_nautilus_conformance_receipt,
)
from quant_fund.research.receipt_v2 import verify_receipt_file

pytestmark = pytest.mark.skipif(
    nautilus_status().installed, reason="blocked-path tests require engine absence"
)


def test_engine_absent_outcome() -> None:
    result = run_nautilus_conformance(seed=0)
    assert result.outcome == "engine_absent"
    assert result.status.installed is False
    assert PINNED_NAUTULUS_VERSION in result.detail


def test_blocked_receipt_is_sealed_and_self_verifying(tmp_path: Path) -> None:
    receipt = run_nautilus_conformance_eval(seed=0)
    assert receipt["verdict"] == "blocked"
    assert receipt["data_label"] == "META"
    payload = receipt["payload"]
    assert payload["schema"] == NAUTILUS_CONFORMANCE_SCHEMA
    assert payload["engine_installed"] is False
    assert payload["outcome"] == "engine_absent"
    assert payload["live_pnl_claim"] is False
    path = write_nautilus_conformance_receipt(receipt, tmp_path)
    assert path.name.startswith("nautilus_conformance_")
    verification = verify_receipt_file(path)
    assert verification["valid"], verification["errors"]


def test_equivalence_spec_is_written_contract() -> None:
    spec = conformance_equivalence_spec()
    assert spec["fill_key"] == ["ts", "asset", "signed_qty", "exec_px", "fee"]
    assert any("pre-update mark" in r for r in spec["fill_rules"])


def test_compare_fills_detects_count_and_value_drift() -> None:
    ref = [
        {"ts": "t1", "asset": "A", "signed_qty": 1.0, "exec_px": 10.0, "fee": 0.0},
        {"ts": "t2", "asset": "B", "signed_qty": -2.0, "exec_px": 20.0, "fee": 0.1},
    ]
    assert _compare_fills(ref, ref) == []
    short = _compare_fills(ref, ref[:1])
    assert short and "fill count differs" in short[0]
    drift = [dict(ref[0]), dict(ref[1], signed_qty=-1.999)]
    assert any("signed_qty" in m for m in _compare_fills(ref, drift))


def test_consistency_errors_catch_forged_match() -> None:
    body = {
        "verdict": "pass",
        "payload": {"outcome": "matched", "engine_installed": False},
    }
    errors = nautilus_conformance_consistency_errors(body)
    assert not any("verdict_outcome_inconsistent" in e for e in errors)
    assert any("outcome_impossible_without_engine" in e for e in errors)
    # A forged outcome on an honest verdict is equally caught.
    body2 = {
        "verdict": "pass",
        "payload": {"outcome": "engine_absent", "engine_installed": False},
    }
    errors2 = nautilus_conformance_consistency_errors(body2)
    assert any("verdict_outcome_inconsistent" in e for e in errors2)


def test_consistency_errors_accept_blocked_attempt() -> None:
    body = {
        "verdict": "blocked",
        "payload": {"outcome": "engine_absent", "engine_installed": False},
    }
    assert nautilus_conformance_consistency_errors(body) == []


def test_contract_errors() -> None:
    assert "payload_missing" in nautilus_conformance_contract_errors({})
    ok = {
        "payload": {
            "schema": NAUTILUS_CONFORMANCE_SCHEMA,
            "engine_installed": False,
            "outcome": "engine_absent",
            "detail": "x",
            "equivalence_spec": {},
            "live_pnl_claim": False,
        }
    }
    assert nautilus_conformance_contract_errors(ok) == []
    bad = json.loads(json.dumps(ok))
    bad["payload"]["outcome"] = "surely_matched"
    assert "payload_outcome_unknown" in nautilus_conformance_contract_errors(bad)


def test_fill_ledger_maps_result_schema() -> None:
    import polars as pl

    class _R:
        fills = pl.DataFrame(
            {
                "fill_time": ["2024-01-03"],
                "security_id": ["A"],
                "quantity": [1.5],
                "price": [10.0],
                "fee": [0.01],
            }
        )

    ledger = _fill_ledger(_R())
    assert ledger == [
        {"ts": "2024-01-03", "asset": "A", "signed_qty": 1.5, "exec_px": 10.0, "fee": 0.01}
    ]
