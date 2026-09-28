"""Verify-research, report, and tearsheet commands.

Split out of the original module. Import the parent path; it re-exports these names.
"""

from __future__ import annotations

from pathlib import Path

import typer

from .app import app
from .support import _cfg


@app.command("verify-research")
def verify_research(
    path: Path = typer.Argument(Path("data/metadata/research/latest.json")),
) -> None:
    """Verify a notebook, completed Phase-1 run directory, or evidence index.

    Soft-verify includes ``northset_session_means_honesty_errors`` (dispatcher
    over session mean helpers). Research diagnostic only; never live Sharpe.
    """
    import json

    if path.is_dir():
        try:
            directory_manifest = json.loads((path / "manifest.json").read_text())
        except (OSError, UnicodeError, json.JSONDecodeError):
            directory_manifest = None
        if (
            isinstance(directory_manifest, dict)
            and directory_manifest.get("kind") == "forward_shadow_manifest"
        ):
            from quant_fund.paper.forward_shadow import verify

            result = verify(path)
        else:
            from quant_fund.research.phase1_verify import verify_phase1_run

            result = verify_phase1_run(path)
    elif path.is_file() and path.name.endswith(".json"):
        try:
            payload = json.loads(path.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict) and payload.get("kind") == "phase1_evidence_index":
            from quant_fund.research.phase1_verify import verify_phase1_index

            result = verify_phase1_index(path)
        else:
            from quant_fund.research.verify import verify_research_artifact

            result = verify_research_artifact(path)
    else:
        from quant_fund.research.verify import verify_research_artifact

        result = verify_research_artifact(path)
    typer.echo(json.dumps(result, indent=2))
    raise typer.Exit(code=0 if result["valid"] else 1)


@app.command(hidden=True)
def lab(config: Path = typer.Option(Path("configs/research.yaml"))) -> None:
    """Legacy compatibility alias for `dipcatcher research`."""
    from quant_fund.research.agent import run_research

    cfg = _cfg(config)
    nb = run_research(cfg)
    if nb.synthetic:
        typer.echo("SYNTHETIC")
    typer.echo(nb.disclaimer)
    typer.echo(nb.artifacts.get("json"))


@app.command()
def report(
    latest: bool = typer.Option(False, "--latest"),
    config: Path = typer.Option(Path("configs/research.yaml")),
) -> None:
    from quant_fund.reporting.report import latest_report_dir, write_report

    cfg = _cfg(config)
    dest = latest_report_dir(Path(cfg.data.root)) / "latest.md"
    write_report(
        dest,
        "Research report",
        {"config": cfg.dump(), "note": "see MLflow for experiment metrics"},
        synthetic=cfg.data.source == "synthetic",
    )
    typer.echo(dest)


@app.command("tearsheet")
def tearsheet_cmd(
    equity: Path = typer.Option(
        ..., "--equity", help="Equity parquet: event_time, nav (research-only label)"
    ),
    fills: Path | None = typer.Option(None, "--fills", help="Optional fills parquet for IS/TCA"),
    weights: Path | None = typer.Option(
        None,
        "--weights",
        help="Optional target-weight panel (event_time, security_id, target_weight)",
    ),
    bars: Path | None = typer.Option(
        None, "--bars", help="Optional per-name bars for per-security attribution"
    ),
    out_md: Path | None = typer.Option(None, "--out-md", help="Markdown output path"),
    out_json: Path | None = typer.Option(None, "--out-json", help="JSON sheet output path"),
    periods_per_year: float = typer.Option(252.0, "--periods-per-year"),
    label: str = typer.Option("BACKTEST_SIM", "--label"),
    synthetic: bool = typer.Option(False, "--synthetic"),
) -> None:
    """Institutional tearsheet: summary stats, drawdowns, period table, costs, attribution.

    Reads backtest/paper artifacts (equity curve, fills, weights) and emits a
    durable report. Never a live-P&L claim — live_pnl_claim=False is stamped.
    """
    import json

    import polars as pl

    from quant_fund.reporting.tearsheet import (
        build_tearsheet,
        tearsheet_markdown,
        write_tearsheet_md,
    )
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

    eq = pl.read_parquet(equity)
    sheet = build_tearsheet(
        eq,
        fills=pl.read_parquet(fills) if fills is not None else None,
        weights=pl.read_parquet(weights) if weights is not None else None,
        bars=pl.read_parquet(bars) if bars is not None else None,
        periods_per_year=periods_per_year,
        label=label,
        synthetic=synthetic,
    )
    if out_md is not None:
        write_tearsheet_md(out_md, sheet)
        typer.echo(f"markdown={out_md}")
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        sealed = {**sheet, "receipt_sha256": hash_bytes(canonical_json_bytes(sheet))}
        out_json.write_text(json.dumps(sealed, indent=2, default=str))
        typer.echo(f"json={out_json}")
    if out_md is None and out_json is None:
        typer.echo(tearsheet_markdown(sheet))


__all__ = [
    "lab",
    "report",
    "tearsheet_cmd",
    "verify_research",
]
