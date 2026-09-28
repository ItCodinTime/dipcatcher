"""A published number traces to the receipt digest and one ledger entry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_fund.audit.ledger import AuditLedger
from quant_fund.audit.record import record_research_receipt
from quant_fund.audit.signing import Ed25519Signer
from quant_fund.audit.trace import receipt_digest, trace_published_number
from quant_fund.research.verify import _receipt_digest


def _notebook() -> dict[str, object]:
    return {
        "provenance": {
            "run_id": "r1",
            "git_revision": "abc",
            "git_worktree_sha256": "wt",
            "config_sha256": "cfg",
            "dataset_sha256": "ds",
            "dataset_content_sha256": "dc",
            "northset_inputs_sha256": "ns",
            "execution_claim": "research_only",
            "point_in_time": True,
            "code_sha256": "code",
        },
        "data_source": "SYNTHETIC",
        "metrics": {"pinball": 0.2},
        "code_sha256": "code",
    }


def _ledger(tmp_path: Path) -> tuple[AuditLedger, Ed25519Signer]:
    signer = Ed25519Signer.generate()
    ledger = AuditLedger(
        tmp_path / "ledger",
        signer=signer,
        sync=False,
        clock=lambda: "2020-01-01T00:00:00Z",
    )
    return ledger, signer


def test_trace_links_the_receipt_without_rewriting_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("quant_fund.audit.trace.git_revision", lambda: "abc")
    monkeypatch.setattr("quant_fund.audit.trace.git_worktree_sha256", lambda: "wt")
    notebook = _notebook()
    path = tmp_path / "receipt.json"
    raw = json.dumps(notebook).encode()
    path.write_bytes(raw)
    ledger, signer = _ledger(tmp_path)
    record_research_receipt(ledger, path)
    assert path.read_bytes() == raw
    report = trace_published_number(
        path,
        "metrics.pinball",
        ledger,
        trust_public_key=signer.public_key,
        verify_receipt=True,
        compare_worktree=True,
    )
    assert report["linked"] is True
    assert report["value"] == 0.2
    assert report["receipt_sha256"] == receipt_digest(notebook)
    assert report["receipt_sha256"] == _receipt_digest(notebook)
    inclusion = report["inclusion"]
    assert isinstance(inclusion, dict)
    assert inclusion["valid"] is True
    assert report["checkout_matches_recorded_revision"] is True
    assert report["checkout_matches_recorded_worktree"] is True
    assert report["live_pnl_claim"] is False
    verifier = report["research_verifier"]
    assert isinstance(verifier, dict)
    assert verifier["valid"] is False


def test_editing_the_number_breaks_the_digest(tmp_path: Path) -> None:
    notebook = _notebook()
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(notebook), encoding="utf-8")
    ledger, _signer = _ledger(tmp_path)
    record_research_receipt(ledger, path)
    edited = _notebook()
    metrics = edited["metrics"]
    assert isinstance(metrics, dict)
    metrics["pinball"] = 0.9
    path.write_text(json.dumps(edited), encoding="utf-8")
    report = trace_published_number(path, "metrics.pinball", ledger)
    assert report["linked"] is False
    assert "receipt_not_in_ledger" in report["errors"]
    assert report["value"] == 0.9


def test_provenance_mismatch_is_reported(tmp_path: Path) -> None:
    notebook = _notebook()
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(notebook), encoding="utf-8")
    ledger, _signer = _ledger(tmp_path)
    digest = receipt_digest(notebook)
    ledger.append(
        "research_run",
        {
            "receipt_sha256": digest,
            "run_id": "r1",
            "git_revision": "not-the-receipt",
            "git_worktree_sha256": "wt",
            "config_sha256": "cfg",
            "dataset_sha256": "ds",
            "dataset_content_sha256": "dc",
            "northset_inputs_sha256": "ns",
            "code_sha256": "code",
            "live_pnl_claim": False,
        },
    )
    report = trace_published_number(path, "metrics.pinball", ledger)
    assert report["linked"] is False
    assert "provenance_mismatch:git_revision" in report["errors"]


def test_missing_metric_is_not_linked(tmp_path: Path) -> None:
    notebook = _notebook()
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(notebook), encoding="utf-8")
    ledger, _signer = _ledger(tmp_path)
    record_research_receipt(ledger, path)
    report = trace_published_number(path, "metrics.missing", ledger)
    assert report["linked"] is False
    assert "metric_not_found" in report["errors"]


def test_unsigned_suffix_is_never_presented_as_signed_evidence(tmp_path: Path) -> None:
    ledger, signer = _ledger(tmp_path)
    ledger.append("risk_decision", {"accepted": True})
    ledger.signer = None
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(_notebook()))
    record_research_receipt(ledger, path)
    report = trace_published_number(
        path, "metrics.pinball", ledger, trust_public_key=signer.public_key
    )
    assert report["ledger"]["valid"] is True
    assert report["ledger"]["fully_signed"] is False
    assert report["linked"] is False
    assert "ledger_not_fully_signed" in report["errors"]


def test_missing_receipt_returns_unlinked(tmp_path: Path) -> None:
    ledger, _ = _ledger(tmp_path)
    report = trace_published_number(tmp_path / "absent.json", "metrics.pinball", ledger)
    assert report["linked"] is False
    assert report["errors"][0].startswith("receipt_unreadable:")
