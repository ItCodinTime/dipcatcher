"""verify-ledger exits 0, 1, and 2, and the Typer commands render help."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant_fund.audit.cli import main
from quant_fund.audit.ledger import AuditLedger
from quant_fund.audit.signing import Ed25519Signer
from quant_fund.cli.main import app

_RUNNER = CliRunner()


def _ledger(tmp_path: Path) -> tuple[Path, Path]:
    signer = Ed25519Signer.generate()
    key = tmp_path / "key"
    signer.write(key)
    ledger = AuditLedger(
        tmp_path / "ledger",
        signer=signer,
        sync=False,
        clock=lambda: "2020-01-01T00:00:00Z",
    )
    ledger.append("paper_decision", {"simulation_only": True, "live_pnl_claim": False})
    return ledger.root, key


def test_verify_ledger_exit_codes(tmp_path: Path) -> None:
    root, key = _ledger(tmp_path)
    with pytest.raises(SystemExit) as ok:
        main([str(root), "--trust-pub", str(key.with_name(key.name + ".pub"))])
    assert ok.value.code == 0
    raw = (root / "entries.jsonl").read_bytes()
    (root / "entries.jsonl").write_bytes(raw.replace(b"{", b"x", 1))
    with pytest.raises(SystemExit) as bad:
        main([str(root)])
    assert bad.value.code == 1
    with pytest.raises(SystemExit) as usage:
        main([str(root), "--trust-pub", str(tmp_path / "missing.pub")])
    assert usage.value.code == 2


def test_dipcatcher_verify_ledger_help_and_report(tmp_path: Path) -> None:
    help_result = _RUNNER.invoke(app, ["verify-ledger", "--help"])
    assert help_result.exit_code == 0
    assert "Merkle" in help_result.stdout
    root, _key = _ledger(tmp_path)
    result = _RUNNER.invoke(app, ["verify-ledger", str(root)])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["valid"] is True
    assert payload["live_pnl_claim"] is False
    record_help = _RUNNER.invoke(app, ["audit-record", "--help"])
    trace_help = _RUNNER.invoke(app, ["audit-trace", "--help"])
    assert record_help.exit_code == 0
    assert trace_help.exit_code == 0


def test_audit_record_and_trace_commands(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "provenance": {
                    "run_id": "r1",
                    "git_revision": "abc",
                    "git_worktree_sha256": "wt",
                    "config_sha256": "cfg",
                    "dataset_sha256": "ds",
                    "dataset_content_sha256": "dc",
                    "northset_inputs_sha256": "ns",
                },
                "metrics": {"pinball": 0.2},
                "code_sha256": "code",
            }
        ),
        encoding="utf-8",
    )
    before = receipt.read_bytes()
    key = tmp_path / "sign.key"
    ledger = tmp_path / "ledger"
    recorded = _RUNNER.invoke(
        app,
        [
            "audit-record",
            "--generate-key",
            str(key),
            "--ledger",
            str(ledger),
            "--receipt",
            str(receipt),
        ],
    )
    assert recorded.exit_code == 0, recorded.stdout
    assert receipt.read_bytes() == before
    assert key.is_file()
    assert key.read_text(encoding="ascii").strip() not in recorded.stdout
    traced = _RUNNER.invoke(
        app,
        [
            "audit-trace",
            "--ledger",
            str(ledger),
            "--receipt",
            str(receipt),
            "--metric",
            "metrics.pinball",
        ],
    )
    assert traced.exit_code == 0, traced.stdout
    body = json.loads(traced.stdout)
    assert body["linked"] is True
    assert body["value"] == 0.2
