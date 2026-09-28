"""`quant leakage` sub-typer: scan / watch (DESIGN.md §6.5).

Mounting into the root `quant` app is owned by W5 (§9.2); this module only
defines the sub-typer with function-level (lazy) heavy imports.
"""

from __future__ import annotations

from pathlib import Path

import typer

leakage_app = typer.Typer(help="Leakage hunter: AST scan + runtime watchdog.")


def _parse_rules(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    return {tok.strip().upper() for tok in raw.split(",") if tok.strip()}


@leakage_app.command("scan")
def scan_cmd(
    paths: list[Path] | None = typer.Option(None, "--paths", help="Files/dirs to scan."),
    rules: str | None = typer.Option(None, "--rules", help="Comma list, e.g. LH001,LH007."),
    fmt: str = typer.Option("text", "--format", help="text|json"),
    fail_on: str = typer.Option("error", "--fail-on", help="error|warning|none"),
) -> None:
    """Scan paths with the LH rule pack; exit 1 when findings >= --fail-on."""
    from quant_fund.leakage import report_to_json, report_to_text, scan_paths

    scan_targets = paths if paths else [Path("src/quant_fund")]
    try:
        report = scan_paths(scan_targets, rules=_parse_rules(rules))
    except ValueError as exc:
        typer.echo(f"leakage scan: {exc}", err=True)
        raise typer.Exit(2) from exc
    if fmt == "json":
        typer.echo(report_to_json(report))
    elif fmt == "text":
        typer.echo(report_to_text(report))
    else:
        typer.echo(f"unknown --format {fmt!r} (text|json)", err=True)
        raise typer.Exit(2)
    # Fail closed on scan defects, independent of --fail-on: a scan that
    # covered no files, or skipped an unparseable file, checked nothing —
    # a clean report would be a false pass.
    if report.scanned_files == 0:
        typer.echo(
            "leakage scan: no .py files under scan targets; nothing was checked",
            err=True,
        )
        raise typer.Exit(2)
    if any(f.rule_id == "LH012" for f in report.findings):
        typer.echo(
            "leakage scan: unparseable file(s) (LH012) — contents were never scanned",
            err=True,
        )
        raise typer.Exit(2)
    if fail_on == "error":
        violated = report.errors > 0
    elif fail_on == "warning":
        violated = (report.errors + report.warnings) > 0
    elif fail_on == "none":
        violated = False
    else:
        typer.echo(f"unknown --fail-on {fail_on!r} (error|warning|none)", err=True)
        raise typer.Exit(2)
    if violated:
        raise typer.Exit(1)


@leakage_app.command("watch")
def watch_cmd(
    config: Path = typer.Option(..., "--config", exists=True),
    seed: int = typer.Option(..., "--seed"),
    pit_root: Path | None = typer.Option(
        None, "--pit-root", help="Vault root (default: <data.root>/pit)."
    ),
) -> None:
    """Guarded vault read sweep under the runtime watchdog (no bundle).

    Arms a strict LeakageWatchdog on the PIT vault (constructor-injected,
    monkeypatch-free) and re-reads every dataset as-of now: any row whose
    known_at exceeds the decision time trips the watchdog and exits 1.
    Requires the W1 vault (quant_fund.pit).
    """
    from quant_fund.leakage import LeakageError, LeakageWatchdog

    try:
        from quant_fund.pit import PitVault
    except ImportError:
        typer.echo(
            "leakage watch requires the PIT vault (quant_fund.pit, workstream W1); "
            "not available in this checkout.",
            err=True,
        )
        raise typer.Exit(2) from None

    from quant_fund.config import load_config
    from quant_fund.utils.seeds import set_global_seed

    cfg = load_config(config)
    set_global_seed(seed)
    root = pit_root if pit_root is not None else Path(cfg.data.root) / "pit"
    if not root.is_dir():
        typer.echo(f"pit vault root {root} does not exist; run `quant pit init` first.", err=True)
        raise typer.Exit(2)
    watchdog = LeakageWatchdog(strict=True)
    vault = PitVault(root, watchdog=watchdog)
    from datetime import UTC, datetime

    decision_time = datetime.now(UTC)
    datasets = vault.list_datasets()
    if not datasets:
        typer.echo(f"no datasets under {root} (no manifest.json found).", err=True)
        raise typer.Exit(2)
    try:
        for name in datasets:
            vault.asof(name, decision_time)
    except LeakageError as exc:
        typer.echo(f"WATCHDOG TRIP: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        f"watchdog clean: {len(datasets)} dataset(s), {watchdog.n_observed} read(s), "
        f"seed={seed}, watermark={watchdog.watermark()}"
    )
