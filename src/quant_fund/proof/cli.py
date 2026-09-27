"""``quant proof`` sub-typer (DESIGN.md §5.8).

Honesty contract: this CLI prints bundle ids, hashes, signature scheme, and
verification verdicts/reasons — never headline metrics (AGENTS.md rule 1;
bundle ``metrics_recompute`` is a verification artifact, not output).

Mounting into ``quant_fund.cli._app`` is W5's glue (DESIGN.md §9.2); this
module only defines ``proof_app`` and keeps heavy imports function-level.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

__all__ = ["proof_app"]

proof_app = typer.Typer(
    help="Verify existing proof bundles; historical proven runs are unavailable."
)


@proof_app.command("run")
def run_cmd(
    config: Path = typer.Option(..., "--config", help="Path to experiment config YAML/JSON."),
    seed: int = typer.Option(..., "--seed", help="Run seed (recorded in the bundle)."),
    pit_root: Path = typer.Option(..., "--pit-root", help="PIT vault root (W1)."),
    bundle_dir: Path = typer.Option(..., "--bundle-dir", help="Proof bundle chain directory."),
    replay_engine: str = typer.Option(
        "reference", "--replay-engine", help="reference | fast (engine parity contract)"
    ),
) -> None:
    """Reject runs until a decision schedule drives point-in-time vault reads."""
    typer.echo(
        "proof run unavailable: explicit per-decision as-of vault reads are not implemented",
        err=True,
    )
    raise typer.Exit(code=2)


@proof_app.command("verify")
def verify_cmd(
    bundle: Path = typer.Option(..., "--bundle", help="Path to bundles/<id>.json."),
    bundle_dir: Path | None = typer.Option(None, "--bundle-dir", help="Chain directory."),
    replay: bool = typer.Option(False, "--replay", help="Deterministic replay (needs --pit-root)."),
    strict_signature: bool = typer.Option(
        True, "--strict-signature/--no-strict-signature", help="Fail unsigned bundles."
    ),
    trusted_head: str | None = typer.Option(
        None, "--trusted-head", help="Externally pinned chain head (A1 F13)."
    ),
    pit_root: Path | None = typer.Option(
        None, "--pit-root", help="PIT vault root, required for --replay."
    ),
) -> None:
    """Verify a proof bundle: re-derive every hash AND recompute metrics."""
    from quant_fund.proof.verify import verify_bundle

    result = verify_bundle(
        bundle,
        bundle_dir=bundle_dir,
        replay=replay,
        strict_signature=strict_signature,
        trusted_head=trusted_head,
        pit_root=pit_root,
    )
    typer.echo(
        json.dumps(
            {
                "ok": result.ok,
                "reasons": result.reasons,
                "env_mismatch": result.env_mismatch,
                "metrics_match": result.metrics_match,
            },
            indent=2,
        )
    )
    if not result.ok:
        raise typer.Exit(code=1)


@proof_app.command("chain-head")
def chain_head_cmd(
    dir: Path = typer.Option(..., "--dir", help="Proof bundle chain directory."),
) -> None:
    """Print the current head bundle hash (CI pins it as proofchain-head.txt)."""
    from quant_fund.proof.bundle import chain_head

    typer.echo(chain_head(dir))
