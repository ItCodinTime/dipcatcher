"""``quant reality`` sub-typer — reality-filter CLI (PROOFCORE W4 §7.9).

Commands operate on a trial-ledger JSONL file (one
``proofcore.contracts.TrialLedgerRow`` per line; the JSONL export format of
the provenance DB per DESIGN.md §14.4). Prints verdicts and diagnostics only
— per the AGENTS.md honesty contract no Sharpe/P&L/NAV is headlined.

``preflight`` only classifies an export. An empty file is an explicit skip
(exit 3), not a scored verdict: the provenance DB is gitignored and no
production path records trials, so a fresh checkout exports zero rows.
``trial-report`` and ``ledger-gate`` stay fail-closed on that same file.

Mounting into the top-level ``quant`` app is owned by W5 (cli glue);
this module only defines the sub-typer.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from quant_fund.proofcore.contracts import RealityFilterError, TrialLedgerRow

reality_app = typer.Typer(
    help="Reality filter: deflated-Sharpe / CSCV-PBO / FDR honesty diagnostics over the trial ledger."
)


# Distinct from ledger-gate's exit 1 (verdict is not 'pass') and from the
# fail-closed exit 2. ``make reality-gate`` maps this to process exit 0 and
# the reality-filter workflow annotates the skip message as a notice.
EMPTY_LEDGER_EXIT = 3


def count_ledger_rows(path: Path) -> int:
    """Count non-blank lines in an exported trial ledger.

    Does not validate row schemas. A missing file is an error, not an empty
    ledger. Blank lines are ignored, matching :func:`_load_ledger`.
    """
    if not path.is_file():
        raise RealityFilterError(f"ledger not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def empty_ledger_skip_message(path: Path) -> str:
    """Stdout contract for an export that has nothing to score."""
    return (
        "REALITY_FILTER_SKIP: "
        f"ledger {path} contains no trial rows; "
        "the provenance DB has no recorded research trials, "
        "so the reality filter was not scored"
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


@reality_app.command("preflight")
def preflight(
    ledger: Path = typer.Option(..., "--ledger", help="Path to trial-ledger JSONL."),
) -> None:
    """Exit 3 when the export has no rows, 0 when it has rows, 2 if missing.

    Does not score the ledger and does not change filter thresholds. Callers
    that need a verdict still run ``trial-report`` / ``ledger-gate``, which
    fail closed on an empty file.
    """
    try:
        n_rows = count_ledger_rows(ledger)
    except RealityFilterError as exc:
        typer.echo(f"REALITY_FILTER_ERROR: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    if n_rows == 0:
        typer.echo(empty_ledger_skip_message(ledger))
        raise typer.Exit(code=EMPTY_LEDGER_EXIT)
    typer.echo(f"REALITY_FILTER_READY: n_rows={n_rows}")


@reality_app.command("trial-report")
def trial_report(
    ledger: Path = typer.Option(..., "--ledger", help="Path to trial-ledger JSONL."),
    q: float = typer.Option(0.05, "--q", help="BH-FDR level, in (0, 1)."),
    s_blocks: int = typer.Option(16, "--s-blocks", help="CSCV block count (even)."),
    out: Path | None = typer.Option(None, "--out", help="Optional JSONL/JSON export path."),
) -> None:
    """Build the RealityReport for a trial ledger and print the verdict."""
    from quant_fund.reality.report import DISCLAIMER, build_reality_report

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
    from quant_fund.reality.report import build_reality_report

    try:
        rows = _load_ledger(ledger)
        report = build_reality_report(rows, q=q, s_blocks=s_blocks)
    except (RealityFilterError, ValueError) as exc:
        typer.echo(f"REALITY_FILTER_ERROR: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(f"verdict={report.verdict} n_trials={report.n_trials} sha256={report.report_sha256}")
    if report.verdict != "pass":
        raise typer.Exit(code=1)
