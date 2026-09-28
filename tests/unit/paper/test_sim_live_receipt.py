"""sim_live receipt: sealed at write, kind-dispatched deep verify."""

from __future__ import annotations

import json
from pathlib import Path

from quant_fund.paper.sim_live import SIM_LIVE_KINDS, sim_live_contract_errors
from quant_fund.research.receipt_v2 import verify_receipt_file
from quant_fund.utils.receipt import seal_errors, seal_receipt


def _sim_live_receipt() -> dict:
    return {
        "kind": "sim_live_receipt",
        "run_id": "test-run",
        "live_pnl_claim": False,
        "simulated_only": True,
        "research_only": True,
        "bench_only": False,
        "data_label": "SYNTHETIC",
        "bars": {"sha256": {"BTC": "ab" * 32}},
        "strategy": {"champion": {"name": "ridge"}},
        "champion_equity_stats": {"status": "degenerate"},
    }


def test_seal_then_verify_roundtrip(tmp_path: Path) -> None:
    sealed = seal_receipt(_sim_live_receipt())
    path = tmp_path / "sim_live_test.json"
    path.write_text(json.dumps(sealed, indent=2), encoding="utf-8")
    result = verify_receipt_file(path)
    assert result["valid"] is True, result["errors"]
    assert result["digest_convention"] == "canonical_json"


def test_post_seal_edit_fails(tmp_path: Path) -> None:
    sealed = seal_receipt(_sim_live_receipt())
    sealed["champion_equity_stats"] = {"status": "great"}
    path = tmp_path / "sim_live_test.json"
    path.write_text(json.dumps(sealed, indent=2), encoding="utf-8")
    result = verify_receipt_file(path)
    assert result["valid"] is False
    assert "receipt_sha256_mismatch" in result["errors"]


def test_resealed_claim_drift_still_fails_contract(tmp_path: Path) -> None:
    drifted = seal_receipt({**_sim_live_receipt(), "simulated_only": False})
    path = tmp_path / "sim_live_test.json"
    path.write_text(json.dumps(drifted, indent=2), encoding="utf-8")
    result = verify_receipt_file(path)
    assert result["valid"] is False
    assert "simulated_only" in result["errors"]
    assert seal_errors(drifted) == []


def test_contract_covers_each_honesty_field() -> None:
    assert sim_live_contract_errors(_sim_live_receipt()) == []
    assert "kind" in sim_live_contract_errors({**_sim_live_receipt(), "kind": "other"})
    assert "research_only" in sim_live_contract_errors(
        {**_sim_live_receipt(), "research_only": False}
    )
    assert "live_pnl_claim" in sim_live_contract_errors(
        {**_sim_live_receipt(), "live_pnl_claim": True}
    )
    bench = {**_sim_live_receipt(), "kind": "sim_live_bench_receipt", "bench_only": True}
    assert sim_live_contract_errors(bench) == []
    assert "sim_live_bench_receipt" in SIM_LIVE_KINDS


def test_utils_seal_exclude_set() -> None:
    base = _sim_live_receipt()
    with_paths = {**base, "receipt_path": "/x.json"}
    assert (
        seal_receipt(base)["receipt_sha256"]
        == seal_receipt(with_paths, exclude=("receipt_path",))["receipt_sha256"]
    )
