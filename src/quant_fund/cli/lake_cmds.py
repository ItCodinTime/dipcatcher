"""Lake and lineage commands.

``dipcatcher lineage show <dataset>`` prints the DAG.
``dipcatcher lineage verify`` recomputes hashes and exits non-zero on drift.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import typer

from quant_fund.cli._app import app
from quant_fund.data.lakehouse.quality import QualityThresholdError
from quant_fund.schemas.errors import DataContractError

lineage_app = typer.Typer(help="Lineage DAG for derived datasets.")
lake_app = typer.Typer(help="Content-addressed market-data lake. Research storage only.")
app.add_typer(lineage_app, name="lineage")
app.add_typer(lake_app, name="lake")


@lineage_app.command("show")
def lineage_show(
    dataset: str,
    root: Path = typer.Option(Path("data/lake"), help="Lake root."),
) -> None:
    """Print the lineage DAG for a derived dataset."""
    from quant_fund.data.lakehouse.lineage import format_dag

    try:
        text = format_dag(root, dataset)
    except DataContractError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(text, nl=False)


@lineage_app.command("verify")
def lineage_verify(
    dataset: str | None = typer.Argument(None, help="Dataset name. Omit to verify every record."),
    root: Path = typer.Option(Path("data/lake"), help="Lake root."),
) -> None:
    """Recompute file and code hashes and report drift."""
    from quant_fund.data.lakehouse.lineage import verify_lineage

    report = verify_lineage(root, dataset)
    typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    if not report.ok:
        raise typer.Exit(code=1)


@lake_app.command("import")
def lake_import(
    paths: list[Path] = typer.Argument(..., help="Existing data files to copy in unchanged."),
    dataset: str = typer.Option(..., "--dataset", help="Dataset name stored on the snapshot."),
    root: Path = typer.Option(Path("data/lake"), help="Lake root."),
    source: str | None = typer.Option(None, help="Override the source label."),
) -> None:
    """Import files byte-for-byte and print the snapshot id."""
    from quant_fund.data.lakehouse.migrate import import_files

    snapshot = import_files(paths, root, dataset=dataset, source=source)
    typer.echo(snapshot.snapshot_id)


@lake_app.command("quality")
def lake_quality(
    path: Path = typer.Option(..., "--path", help="Parquet file to score."),
    max_abs_log_return: float = typer.Option(0.5, help="Outlier threshold on absolute log return."),
    max_gap_days: float | None = typer.Option(None, help="Fail when a gap exceeds this many days."),
    stale_run_length: int = typer.Option(
        5, help="Unchanged-close run length that counts as stale."
    ),
    enforce: bool = typer.Option(True, help="Exit non-zero when a threshold fails."),
) -> None:
    """Write a machine-readable quality report to stdout."""
    from quant_fund.data.lakehouse.quality import (
        QualityThresholds,
        quality_report_path,
        report_json,
    )

    gap = None if max_gap_days is None else timedelta(days=max_gap_days)
    thresholds = QualityThresholds(
        max_abs_log_return=max_abs_log_return,
        max_gap=gap,
        stale_run_length=stale_run_length,
    )
    try:
        report = quality_report_path(path, thresholds, enforce=enforce)
    except QualityThresholdError as exc:
        typer.echo(report_json(exc.report), nl=False)
        raise typer.Exit(code=1) from exc
    typer.echo(report_json(report), nl=False)
    if enforce and not report["passed"]:
        raise typer.Exit(code=1)
