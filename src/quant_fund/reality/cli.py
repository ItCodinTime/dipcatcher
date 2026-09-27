"""``quant reality`` sub-typer — reality-filter CLI (PROOFCORE W4 §7.9).

Commands operate on a trial-ledger JSONL file (one
``proofcore.contracts.TrialLedgerRow`` per line; the JSONL export format of
the provenance DB per DESIGN.md §14.4). Prints verdicts and diagnostics only
— per the AGENTS.md honesty contract no Sharpe/P&L/NAV is headlined.

Mounting into the top-level ``quant`` app is owned by W5 (cli glue);
this module only defines the sub-typer.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from quant_fund.proofcore.contracts import RealityFilterError, TrialLedgerRow
from quant_fund.reality.report import DISCLAIMER, build_reality_report

reality_app = typer.Typer(
    help="Reality filter: deflated-Sharpe / CSCV-PBO / FDR honesty diagnostics over the trial ledger."
)


def _load_ledger(path: Path) -> list[TrialLedgerRow]:
    if not path.is_file():
        raise RealityFilterError(f"ledger not found: {path}")
    rows: list[TrialLedgerRow] = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(TrialLedgerRow.model_validate(json.loads(text)))
            except ValueError as exc:
                raise RealityFilterError(f"ledger line {i} is not a valid TrialLedgerRow") from exc
    if not rows:
        raise RealityFilterError(f"ledger {path} contains no trial rows")
    return rows


def _emit(report_json: str) -> None:
    typer.echo(report_json)


@reality_app.command("trial-report")
def trial_report(
    ledger: Path = typer.Option(..., "--ledger", help="Path to trial-ledger JSONL."),
    q: float = typer.Option(0.05, "--q", help="BH-FDR level, in (0, 1)."),
    s_blocks: int = typer.Option(16, "--s-blocks", help="CSCV block count (even)."),
    out: Path | None = typer.Option(None, "--out", help="Optional JSONL/JSON export path."),
) -> None:
    """Build the RealityReport for a trial ledger and print the verdict."""
    try:
        rows = _load_ledger(ledger)
        report = build_reality_report(rows, q=q, s_blocks=s_blocks)
    except (RealityFilterError, ValueError) as exc:
        typer.echo(f"REALITY_FILTER_ERROR: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    text = json.dumps(report.model_dump(mode="json"), sort_keys=True)
    if out is not None:
        out.write_text(text + "\n", encoding="utf-8")
    typer.echo(DISCLAIMER, err=True)
    _emit(text)


@reality_app.command("ledger-gate")
def ledger_gate(
    ledger: Path = typer.Option(..., "--ledger", help="Path to trial-ledger JSONL."),
    q: float = typer.Option(0.05, "--q", help="BH-FDR level, in (0, 1)."),
    s_blocks: int = typer.Option(16, "--s-blocks", help="CSCV block count (even)."),
) -> None:
    """Exit 0 iff the ledger's reality verdict is 'pass', else exit 1."""
    try:
        rows = _load_ledger(ledger)
        report = build_reality_report(rows, q=q, s_blocks=s_blocks)
    except (RealityFilterError, ValueError) as exc:
        typer.echo(f"REALITY_FILTER_ERROR: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(f"verdict={report.verdict} n_trials={report.n_trials} sha256={report.report_sha256}")
    if report.verdict != "pass":
        raise typer.Exit(code=1)
