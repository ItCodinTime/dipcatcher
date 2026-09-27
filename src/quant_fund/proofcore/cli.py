"""``quant proofcore`` sub-typer (W5): provenance DB log/query/export.

Every command imports its implementation lazily so
``import quant_fund.cli.main`` never eagerly pulls duckdb (DESIGN.md §9.2).
Prints counts, hashes, and verdict-shaped output only — never headline
metrics (AGENTS.md honesty contract; DESIGN.md §13).
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

proofcore_app = typer.Typer(
    help="PROOFCORE provenance ledger (duckdb): log bundles, query, export.",
)

_DB_OPTION = typer.Option(
    Path("data/metadata/proofcore.duckdb"),
    "--db",
    help="Provenance DB path (default: data/metadata/proofcore.duckdb).",
)


@proofcore_app.command("log")
def log_bundle(
    bundle: Path = typer.Option(..., "--bundle", help="Path to a ProofBundleV1 JSON file."),
    db: Path = _DB_OPTION,
    verification: Path | None = typer.Option(
        None, "--verification", help="Reserved until `quant proof verify` supplies bound results."
    ),
) -> None:
    """Insert one proof bundle (and optional verification result) into the DB."""
    from quant_fund.proofcore.contracts import ProofBundleV1
    from quant_fund.proofcore.provenance import ProvenanceDB

    if verification is not None:
        typer.echo("verification ingestion unavailable until proof verifier exists", err=True)
        raise typer.Exit(2)
    bundle_obj = ProofBundleV1.model_validate(json.loads(bundle.read_text()))
    with ProvenanceDB(db) as prov:
        prov.insert_bundle(bundle_obj, None)
        head = prov.chain_head()
    typer.echo(f"logged bundle {bundle_obj.bundle_id}")
    typer.echo(f"chain_head={head}")


@proofcore_app.command("query")
def query(
    db: Path = _DB_OPTION,
    family: str | None = typer.Option(None, "--family", help="Restrict trial rows to one family."),
) -> None:
    """Print a JSON summary: bundle/trial counts, family split, chain head."""
    from quant_fund.proofcore.provenance import ProvenanceDB

    with ProvenanceDB(db) as prov:
        bundles = prov.bundles()
        trials = prov.trials(family=family)
        head = prov.chain_head()
    summary = {
        "n_bundles": len(bundles),
        "n_trials": len(trials),
        "verified_bundles": sum(1 for b in bundles if b["verified_ok"] is True),
        "chain_head": head,
        "trial_ids": [t.trial_id for t in trials],
    }
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@proofcore_app.command("export")
def export(
    db: Path = _DB_OPTION,
    out: Path = typer.Option(
        ..., "--out", help="Trial-ledger JSONL export path (consumed by `quant reality`)."
    ),
    bundles_out: Path | None = typer.Option(
        None, "--bundles-out", help="Optional JSONL export path for the proof_bundles table."
    ),
) -> None:
    """Export the trial ledger (and optionally bundles) as JSONL (DESIGN.md §14.4)."""
    from quant_fund.proofcore.provenance import ProvenanceDB

    with ProvenanceDB(db) as prov:
        trials = prov.trials()
        bundles = prov.bundles()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in trials:
            fh.write(json.dumps(row.model_dump(mode="json"), sort_keys=True) + "\n")
    typer.echo(f"exported {len(trials)} trial rows -> {out}")
    if bundles_out is not None:
        bundles_out.parent.mkdir(parents=True, exist_ok=True)
        with bundles_out.open("w", encoding="utf-8") as fh:
            for bundle_row in bundles:
                fh.write(json.dumps(bundle_row, sort_keys=True, default=str) + "\n")
        typer.echo(f"exported {len(bundles)} bundle rows -> {bundles_out}")


@proofcore_app.command("chain-head")
def chain_head(db: Path = _DB_OPTION) -> None:
    """Print the current bundle-chain head hash (GENESIS_HASH when empty)."""
    from quant_fund.proofcore.provenance import ProvenanceDB

    with ProvenanceDB(db) as prov:
        typer.echo(prov.chain_head())
