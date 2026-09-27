"""CLI tests for `quant reality` (PROOFCORE W4 §7.9)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from quant_fund.proofcore.contracts import RealityReport, TrialLedgerRow
from quant_fund.reality.cli import reality_app

runner = CliRunner()
_HEX = "ab" * 32


def _row(i: int, *, sr: float, family: str = "discovery") -> TrialLedgerRow:
    return TrialLedgerRow(
        trial_id=f"{i:064x}",
        bundle_hash=_HEX,
        family=family,  # type: ignore[arg-type]
        strategy=f"s{i}",
        cluster_id=f"c{i}",
        created_utc="2026-01-01T00:00:00+00:00",
        n_obs=500,
        periods_per_year=252.0,
        sharpe_periodic=sr,
        skew=0.0,
        kurtosis_raw=3.0,
        returns_sha256="cd" * 32,
    )


def _write_ledger(path: Path, rows: list[TrialLedgerRow]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r.model_dump(mode="json")) + "\n")


def test_trial_report_pass_ledger(tmp_path: Path) -> None:
    from scipy.stats import norm

    sr = float(norm.isf(0.0005) / np.sqrt(499))
    ledger = tmp_path / "ledger.jsonl"
    _write_ledger(ledger, [_row(i, sr=sr) for i in range(1, 13)])
    out = tmp_path / "report.json"
    res = runner.invoke(
        reality_app,
        ["trial-report", "--ledger", str(ledger), "--out", str(out)],
    )
    assert res.exit_code == 0, res.output
    rep = RealityReport.model_validate(json.loads(out.read_text()))
    assert rep.verdict == "pass"
    assert "disclaimer" not in rep.model_dump()  # schema stays strict


def test_ledger_gate_exit_codes(tmp_path: Path) -> None:
    from scipy.stats import norm

    strong = float(norm.isf(0.0005) / np.sqrt(499))
    good = tmp_path / "good.jsonl"
    _write_ledger(good, [_row(i, sr=strong) for i in range(1, 13)])
    res = runner.invoke(reality_app, ["ledger-gate", "--ledger", str(good)])
    assert res.exit_code == 0, res.output
    assert "verdict=pass" in res.output

    weak = tmp_path / "weak.jsonl"
    _write_ledger(
        weak, [_row(i, sr=float(s)) for i, s in enumerate(np.linspace(-0.02, 0.03, 12), start=1)]
    )
    res = runner.invoke(reality_app, ["ledger-gate", "--ledger", str(weak)])
    assert res.exit_code == 1
    assert "verdict=deflated" in res.output


def test_missing_ledger_is_error_exit_2(tmp_path: Path) -> None:
    res = runner.invoke(reality_app, ["ledger-gate", "--ledger", str(tmp_path / "nope.jsonl")])
    assert res.exit_code == 2


def test_malformed_ledger_line_fails_closed(tmp_path: Path) -> None:
    ledger = tmp_path / "bad.jsonl"
    ledger.write_text('{"trial_id": "nope"}\n', encoding="utf-8")
    res = runner.invoke(reality_app, ["trial-report", "--ledger", str(ledger)])
    assert res.exit_code == 2


def test_empty_ledger_trial_report_stays_fail_closed(tmp_path: Path) -> None:
    """Scoring commands do not treat 'no rows' as a pass or a skip."""
    ledger = tmp_path / "empty.jsonl"
    ledger.write_text("\n", encoding="utf-8")
    res = runner.invoke(reality_app, ["trial-report", "--ledger", str(ledger)])
    assert res.exit_code == 2
    assert "contains no trial rows" in res.output
    assert "REALITY_FILTER_SKIP" not in res.output


def test_preflight_absent_provenance_db_is_explicit_skip(tmp_path: Path) -> None:
    db = tmp_path / "proofcore.duckdb"
    res = runner.invoke(reality_app, ["preflight", "--db", str(db)])
    assert res.exit_code == 3, res.output
    assert res.output.startswith("REALITY_FILTER_SKIP:")
    assert "absent from the repository" in res.output
    assert "not at HEAD or on main" in res.output
    assert "writes 0 trial rows" in res.output
    assert not db.exists()


def test_preflight_present_db_is_ready(tmp_path: Path) -> None:
    db = tmp_path / "proofcore.duckdb"
    db.write_bytes(b"")
    res = runner.invoke(reality_app, ["preflight", "--db", str(db)])
    assert res.exit_code == 0, res.output
    assert res.output.startswith("REALITY_FILTER_READY: db=")


def test_preflight_requires_db_or_ledger() -> None:
    res = runner.invoke(reality_app, ["preflight"])
    assert res.exit_code == 2
    assert "REALITY_FILTER_ERROR:" in res.output


def test_preflight_empty_ledger_is_explicit_skip(tmp_path: Path) -> None:
    ledger = tmp_path / "empty.jsonl"
    ledger.write_text("\n\n", encoding="utf-8")
    res = runner.invoke(reality_app, ["preflight", "--ledger", str(ledger)])
    assert res.exit_code == 3, res.output
    assert res.output.startswith("REALITY_FILTER_SKIP:")
    assert "contains no trial rows" in res.output
    assert "was not scored" in res.output


def test_preflight_missing_ledger_is_error(tmp_path: Path) -> None:
    res = runner.invoke(reality_app, ["preflight", "--ledger", str(tmp_path / "nope.jsonl")])
    assert res.exit_code == 2
    assert "REALITY_FILTER_ERROR:" in res.output


def test_preflight_ready_does_not_score(tmp_path: Path) -> None:
    ledger = tmp_path / "one.jsonl"
    _write_ledger(ledger, [_row(1, sr=0.0)])
    res = runner.invoke(reality_app, ["preflight", "--ledger", str(ledger)])
    assert res.exit_code == 0, res.output
    assert res.output.strip() == "REALITY_FILTER_READY: n_rows=1"
    assert "verdict=" not in res.output
