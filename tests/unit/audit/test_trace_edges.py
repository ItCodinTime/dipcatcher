"""Coverage for audit.trace — linking published numbers to ledger entries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_fund.audit.ledger import AuditLedger
from quant_fund.audit.signing import Ed25519Signer
from quant_fund.audit.trace import (
    lookup_path,
    receipt_digest,
    trace_published_number,
)

_STAMP = "2025-01-01T00:00:00+00:00"


def _clock() -> str:
    return _STAMP


def _receipt() -> dict:
    return {
        "schema": "research-receipt/1",
        "run_id": "run-1",
        "metrics": {"pinball": [0.01, 0.02], "ece": {"mean": 0.03}},
        "provenance": {
            "run_id": "run-1",
            "git_revision": "abc123",
            "git_worktree_sha256": "0" * 64,
            "config_sha256": "1" * 64,
            "dataset_sha256": "2" * 64,
            "dataset_content_sha256": "3" * 64,
            "northset_inputs_sha256": "4" * 64,
            "code_sha256": "5" * 64,
        },
    }


def _ledger_with_run(root: Path, notebook: dict) -> tuple[AuditLedger, dict]:
    signer = Ed25519Signer.generate()
    ledger = AuditLedger(root, signer=signer, sign_every=1, sync=False, clock=_clock)
    digest = receipt_digest(notebook)
    payload = {
        "receipt_sha256": digest,
        "live_pnl_claim": False,
        **notebook["provenance"],
    }
    ledger.append("research_run", payload)
    return ledger, notebook


def _write(path: Path, notebook: dict) -> Path:
    path.write_text(json.dumps(notebook), encoding="utf-8")
    return path


class TestLookupPath:
    def test_nested_and_indexed(self) -> None:
        doc = {"a": {"b": [{"c": 9}]}}
        assert lookup_path(doc, "a.b.0.c") == 9
        assert lookup_path([1, [2]], "1.0") == 2

    def test_bad_paths(self) -> None:
        with pytest.raises(KeyError):
            lookup_path({}, "")
        with pytest.raises(KeyError):
            lookup_path({}, ".a")
        with pytest.raises(KeyError):
            lookup_path({}, "a.")
        with pytest.raises(KeyError):
            lookup_path({"a": 1}, "b")
        with pytest.raises(KeyError):
            lookup_path({"a": [1]}, "a.5")
        with pytest.raises(KeyError):
            lookup_path({"a": [1]}, "a.x")
        with pytest.raises(KeyError):
            lookup_path({"a": 5}, "a.b")


class TestTrace:
    def test_linked_happy_path(self, tmp_path: Path) -> None:
        notebook = _receipt()
        ledger, _ = _ledger_with_run(tmp_path / "ledger", notebook)
        receipt = _write(tmp_path / "receipt.json", notebook)
        result = trace_published_number(receipt, "metrics.ece.mean", ledger)
        assert result["linked"] is True
        assert result["value"] == 0.03
        assert result["ledger_index"] == 0
        assert result["inclusion"]["valid"] is True
        assert result["errors"] == []
        assert result["live_pnl_claim"] is False

    def test_unreadable_receipt(self, tmp_path: Path) -> None:
        ledger = AuditLedger(
            tmp_path / "ledger",
            signer=Ed25519Signer.generate(),
            sign_every=1,
            sync=False,
            clock=_clock,
        )
        bad = tmp_path / "bad.json"
        bad.write_bytes(b"{not json")
        result = trace_published_number(bad, "x", ledger)
        assert result["linked"] is False
        assert result["errors"][0].startswith("receipt_unreadable")

    def test_non_object_receipt(self, tmp_path: Path) -> None:
        ledger = AuditLedger(
            tmp_path / "ledger",
            signer=Ed25519Signer.generate(),
            sign_every=1,
            sync=False,
            clock=_clock,
        )
        path = _write(tmp_path / "arr.json", [1, 2])
        result = trace_published_number(path, "x", ledger)
        assert result["errors"] == ["receipt_not_object"]

    def test_metric_not_found(self, tmp_path: Path) -> None:
        notebook = _receipt()
        ledger, _ = _ledger_with_run(tmp_path / "ledger", notebook)
        receipt = _write(tmp_path / "receipt.json", notebook)
        result = trace_published_number(receipt, "metrics.missing", ledger)
        assert result["value"] is None
        assert "metric_not_found" in result["errors"]
        assert result["linked"] is False

    def test_receipt_not_in_ledger(self, tmp_path: Path) -> None:
        notebook = _receipt()
        signer = Ed25519Signer.generate()
        ledger = AuditLedger(
            tmp_path / "ledger",
            signer=signer,
            sign_every=1,
            sync=False,
            clock=_clock,
        )
        ledger.append("paper_decision", {"simulation_only": True, "live_pnl_claim": False})
        receipt = _write(tmp_path / "receipt.json", notebook)
        result = trace_published_number(receipt, "metrics.ece.mean", ledger)
        assert "receipt_not_in_ledger" in result["errors"]
        assert result["ledger_index"] is None
        assert result["inclusion"] is None

    def test_provenance_mismatch(self, tmp_path: Path) -> None:
        notebook = _receipt()
        signer = Ed25519Signer.generate()
        ledger = AuditLedger(
            tmp_path / "ledger",
            signer=signer,
            sign_every=1,
            sync=False,
            clock=_clock,
        )
        # Ledger records a different revision than the receipt carries.
        payload = {
            "receipt_sha256": receipt_digest(notebook),
            "live_pnl_claim": False,
            **notebook["provenance"],
            "git_revision": "recorded-elsewhere",
        }
        ledger.append("research_run", payload)
        receipt = _write(tmp_path / "receipt.json", notebook)
        result = trace_published_number(receipt, "metrics.ece.mean", ledger)
        assert "provenance_mismatch:git_revision" in result["errors"]

    def test_provenance_missing(self, tmp_path: Path) -> None:
        notebook = _receipt()
        ledger, _ = _ledger_with_run(tmp_path / "ledger", notebook)
        del notebook["provenance"]
        receipt = _write(tmp_path / "receipt.json", notebook)
        result = trace_published_number(receipt, "metrics.ece.mean", ledger)
        assert "provenance_missing" in result["errors"]

    def test_unsigned_ledger_flags(self, tmp_path: Path) -> None:
        notebook = _receipt()
        ledger = AuditLedger(
            tmp_path / "ledger",
            signer=None,
            sign_every=1,
            sync=False,
            clock=_clock,
        )
        ledger.append(
            "research_run",
            {"receipt_sha256": receipt_digest(notebook), "live_pnl_claim": False},
        )
        receipt = _write(tmp_path / "receipt.json", notebook)
        result = trace_published_number(receipt, "metrics.ece.mean", ledger)
        assert "ledger_not_fully_signed" in result["errors"]
        assert "ledger_invalid" in result["errors"]

    def test_verify_receipt_embeds_verifier(self, tmp_path: Path) -> None:
        notebook = _receipt()
        ledger, _ = _ledger_with_run(tmp_path / "ledger", notebook)
        receipt = _write(tmp_path / "receipt.json", notebook)
        result = trace_published_number(receipt, "metrics.ece.mean", ledger, verify_receipt=True)
        assert "research_verifier" in result

    def test_compare_worktree(self, tmp_path: Path) -> None:
        notebook = _receipt()
        ledger, _ = _ledger_with_run(tmp_path / "ledger", notebook)
        receipt = _write(tmp_path / "receipt.json", notebook)
        result = trace_published_number(receipt, "metrics.ece.mean", ledger, compare_worktree=True)
        assert "checkout_git_worktree_sha256" in result
        assert "checkout_matches_recorded_worktree" in result

    def test_receipt_mutated_between_read_and_end(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        notebook = _receipt()
        ledger, _ = _ledger_with_run(tmp_path / "ledger", notebook)
        receipt = _write(tmp_path / "receipt.json", notebook)

        original = Path.read_bytes
        calls = {"n": 0}

        def flip(self: Path) -> bytes:
            calls["n"] += 1
            # First call (the receipt read) returns real bytes; the
            # re-read at the end pretends the file changed.
            if calls["n"] > 1 and self == receipt:
                return b"tampered"
            return original(self)

        monkeypatch.setattr(Path, "read_bytes", flip)
        result = trace_published_number(receipt, "metrics.ece.mean", ledger)
        assert result["linked"] is False
        assert "receipt_mutated" in result["errors"]
