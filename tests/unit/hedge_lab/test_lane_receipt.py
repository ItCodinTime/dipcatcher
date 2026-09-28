"""Shared hedge_lab lane-receipt seal + verify (fail-closed)."""

from __future__ import annotations

import json
from pathlib import Path

from quant_fund.hedge_lab._receipt import (
    lane_receipt_contract_errors,
    lane_receipt_seal_errors,
    seal_receipt,
    verify_lane_receipt,
)


def _lane_receipt() -> dict:
    return {
        "catalog": "hedge_lab_analytics",
        "lab_id": "test-lab",
        "research_only": True,
        "live_pnl_claim": False,
        "execution_claim": "paper_backtest",
        "data_source": "SYNTHETIC",
        "champion_alias": False,
        "blend_weight": 0.0,
    }


def test_seal_then_verify_roundtrip(tmp_path: Path) -> None:
    sealed = seal_receipt(_lane_receipt())
    path = tmp_path / "lane.json"
    path.write_text(json.dumps(sealed, indent=2, default=str), encoding="utf-8")
    assert verify_lane_receipt(path) == []


def test_seal_excludes_path_keys() -> None:
    base = _lane_receipt()
    with_paths = {**base, "receipt_path": "/a.json", "artifact_path": "/b.json"}
    assert seal_receipt(base)["receipt_sha256"] == seal_receipt(with_paths)["receipt_sha256"]
    assert lane_receipt_seal_errors(seal_receipt(with_paths)) == []


def test_post_seal_edit_fails(tmp_path: Path) -> None:
    sealed = seal_receipt(_lane_receipt())
    sealed["blend_weight"] = 0.9
    path = tmp_path / "lane.json"
    path.write_text(json.dumps(sealed, indent=2, default=str), encoding="utf-8")
    assert "receipt_sha256" in verify_lane_receipt(path)


def test_resealed_claim_drift_still_fails(tmp_path: Path) -> None:
    drifted = {**_lane_receipt(), "live_pnl_claim": True}
    sealed = seal_receipt(drifted)  # attacker reseals after editing the claim
    path = tmp_path / "lane.json"
    path.write_text(json.dumps(sealed, indent=2, default=str), encoding="utf-8")
    errors = verify_lane_receipt(path)
    assert "live_pnl_claim" in errors
    assert lane_receipt_seal_errors(sealed) == []  # digest is consistent — the contract caught it


def test_contract_errors_cover_each_honesty_field() -> None:
    assert lane_receipt_contract_errors(_lane_receipt()) == []
    assert "research_only" in lane_receipt_contract_errors(
        {**_lane_receipt(), "research_only": False}
    )
    assert "execution_claim" in lane_receipt_contract_errors(
        {**_lane_receipt(), "execution_claim": "live"}
    )


def test_verify_receipt_dispatch_catches_lane_claim_drift(tmp_path: Path) -> None:
    """verify-receipt re-derives the lane contract from the catalog marker."""
    from quant_fund.research.receipt_v2 import verify_receipt_file

    drifted = seal_receipt({**_lane_receipt(), "live_pnl_claim": True})
    path = tmp_path / "lane.json"
    path.write_text(json.dumps(drifted, indent=2, default=str), encoding="utf-8")
    result = verify_receipt_file(path)
    assert result["valid"] is False
    assert "live_pnl_claim" in result["errors"]

    good = seal_receipt(_lane_receipt())
    path.write_text(json.dumps(good, indent=2, default=str), encoding="utf-8")
    assert verify_receipt_file(path)["valid"] is True


def test_verify_fails_closed_on_garbage_and_missing(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("not json{", encoding="utf-8")
    assert verify_lane_receipt(bad)[0].startswith("unreadable:")
    assert verify_lane_receipt(tmp_path / "absent.json")[0].startswith("unreadable:")
    arr = tmp_path / "arr.json"
    arr.write_text("[1,2,3]", encoding="utf-8")
    assert verify_lane_receipt(arr) == ["not_an_object"]
