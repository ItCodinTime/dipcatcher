"""``quant reality`` sub-typer — reality-filter CLI (PROOFCORE W4 §7.9).

Commands operate on a trial-ledger JSONL file (one
``proofcore.contracts.TrialLedgerRow`` per line; the JSONL export format of
the provenance DB per DESIGN.md §14.4). Prints verdicts and diagnostics only
— per the AGENTS.md honesty contract no Sharpe/P&L/NAV is headlined.

``preflight`` classifies the gate inputs. ``data/metadata/proofcore.duckdb``
is not in the repository at HEAD or on main (``data/metadata/**`` is
gitignored), so a checkout has no database file and ``quant proofcore
export`` would create an empty DB and write 0 trial rows. That absence is
an explicit skip (exit 3). An empty export of a database that does exist
is the same skip. ``trial-report`` and ``ledger-gate`` stay fail-closed
on an empty ledger file.

Mounting into the top-level ``quant`` app is owned by W5 (cli glue);
this module only defines the sub-typer.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from quant_fund.proofcore.contracts import RealityFilterError, TrialLedgerRow
from quant_fund.utils.atomicio import atomic_write_text

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


def absent_provenance_db_skip_message(path: Path) -> str:
    """Stdout contract when the provenance DB is not in the checkout.

    The committed tree has no ``data/metadata/proofcore.duckdb`` at HEAD or
    on main. Export opens that path with duckdb, which creates an empty
    database, then writes 0 trial rows.
    """
    return (
        "REALITY_FILTER_SKIP: "
        f"provenance DB {path} is absent from the repository "
        "(not at HEAD or on main; data/metadata/** is gitignored), "
        "so the proofcore export writes 0 trial rows and "
        "the reality filter was not scored"
    )


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
    ledger: Path | None = typer.Option(
        None, "--ledger", help="Path to trial-ledger JSONL. Omit when only checking --db."
    ),
    db: Path | None = typer.Option(
        None,
        "--db",
        help="Provenance DB path. A missing file is the empty-export skip.",
    ),
) -> None:
    """Exit 3 when there is nothing to score, 0 when scoring can proceed, 2 on error.

    A missing ``--db`` file is the repository case: the path is not checked
    in, and export would write 0 rows. Does not change filter thresholds.
    ``trial-report`` and ``ledger-gate`` still fail closed on an empty file.
    """
    if db is None and ledger is None:
        typer.echo("REALITY_FILTER_ERROR: pass --db or --ledger", err=True)
        raise typer.Exit(code=2)
    if db is not None and not db.is_file():
        typer.echo(absent_provenance_db_skip_message(db))
        raise typer.Exit(code=EMPTY_LEDGER_EXIT)
    if ledger is None:
        typer.echo(f"REALITY_FILTER_READY: db={db}")
        return
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
        atomic_write_text(out, text + "\n")
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
