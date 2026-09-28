"""``quant pit`` sub-typer: init/verify/stats/restate (DESIGN.md §4.5).

Heavy imports stay function-level (lazy) so importing the CLI never pulls the
vault stack into startup (§9.2).
"""

from __future__ import annotations

from pathlib import Path

import typer

pit_app = typer.Typer(
    help="PIT vault: write-once bitemporal point-in-time data store (PROOFCORE W1)."
)


@pit_app.command("init")
def init_cmd(root: Path = typer.Option(..., "--root", help="Vault root directory.")) -> None:
    """Create the vault root directory."""
    root.mkdir(parents=True, exist_ok=True)
    typer.echo(f"pit vault root ready: {root}")


@pit_app.command("verify")
def verify_cmd(
    root: Path = typer.Option(..., "--root", help="Vault root directory."),
    dataset: str | None = typer.Option(None, "--dataset", help="Verify one dataset only."),
) -> None:
    """Re-hash all parts vs manifest; exit 1 on any violation."""
    from quant_fund.pit import PitVault

    vault = PitVault(root)
    names = [dataset] if dataset is not None else vault.list_datasets()
    if not names:
        typer.echo(f"no datasets under {root}")
        raise typer.Exit(code=1)
    violations = 0
    for name in names:
        found = vault.verify(name)
        if found:
            violations += len(found)
            for violation in found:
                typer.echo(f"VIOLATION {name}: {violation}")
        else:
            typer.echo(f"OK {name}")
    if violations:
        typer.echo(f"pit verify FAILED: {violations} violation(s)")
        raise typer.Exit(code=1)
    typer.echo(f"pit verify passed: {len(names)} dataset(s)")


@pit_app.command("stats")
def stats_cmd(root: Path = typer.Option(..., "--root", help="Vault root directory.")) -> None:
    """Per-dataset rows / revisions / time span."""
    from quant_fund.pit import PitVault
    from quant_fund.proofcore.contracts import ManifestError

    vault = PitVault(root)
    names = vault.list_datasets()
    if not names:
        typer.echo(f"no datasets under {root}")
        return
    for name in names:
        violations = vault.verify(name)
        status = "ok" if not violations else f"CORRUPT({len(violations)})"
        from quant_fund.pit.manifest import read_manifest

        try:
            manifest = read_manifest(root, name)
        except ManifestError as exc:
            typer.echo(f"{name}: manifest unreadable ({exc}) status={status}")
            continue
        rows = sum(entry.rows for entry in manifest.files)
        if manifest.files:
            span = (
                f"{min(e.min_event_time for e in manifest.files)} .. "
                f"{max(e.max_known_at for e in manifest.files)}"
            )
        else:
            span = "(empty)"
        typer.echo(
            f"{name}: rows={rows} revisions={manifest.revision} files={len(manifest.files)} "
            f"span={span} status={status}"
        )


@pit_app.command("restate")
def restate_cmd(
    root: Path = typer.Option(..., "--root", help="Vault root directory."),
    dataset: str = typer.Option(..., "--dataset", help="Target dataset."),
    parquet: Path = typer.Option(..., "--parquet", help="Parquet with corrected rows."),
    known_at: str = typer.Option(..., "--known-at", help="ISO-8601 publication time (UTC)."),
) -> None:
    """Append a restatement; old versions are retained forever (§4.2)."""
    from datetime import datetime

    import polars as pl

    from quant_fund.pit import PitVault, VaultError

    try:
        published = datetime.fromisoformat(known_at)
    except ValueError as exc:
        typer.echo(f"invalid --known-at {known_at!r}: {exc}")
        raise typer.Exit(code=2) from exc
    vault = PitVault(root)
    try:
        manifest = vault.restate(dataset, pl.read_parquet(parquet), known_at=published)
    except VaultError as exc:
        typer.echo(f"restate refused: {exc}")
        raise typer.Exit(code=1) from exc
    typer.echo(f"restated {dataset}: revision={manifest.revision} files={len(manifest.files)}")
