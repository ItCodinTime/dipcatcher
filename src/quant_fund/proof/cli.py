"""``quant proof`` sub-typer (DESIGN.md §5.8, WAVE2.md §9).

Honesty contract: this CLI prints bundle ids, hashes, signature scheme, and
verification/replay verdicts — never headline metrics (AGENTS.md rule 1;
bundle ``metrics_recompute`` is a verification artifact, not output).

Wave 2 turns ``quant proof run`` into the real causal-run entry point
(WAVE2.md amendment A4: ``run_proven``; the wave-1 ``run_backtest_proven``
API stays fail-closed by design) and adds ``quant proof replay`` for the
cryptographic replay engine.

Mounting into ``quant_fund.cli._app`` is W5's glue (DESIGN.md §9.2); this
module only defines ``proof_app`` and keeps heavy imports function-level.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer

__all__ = ["proof_app"]

proof_app = typer.Typer(
    help="Proof bundles: causal proven runs, verification, cryptographic replay."
)


@proof_app.command("run")
def run_cmd(
    spec: Path = typer.Option(..., "--spec", help="Path to a RunSpec JSON document."),
    vault_root: Path = typer.Option(..., "--vault-root", help="PIT vault root (W1)."),
    bundle_dir: Path = typer.Option(..., "--bundle-dir", help="Proof bundle chain directory."),
    signing_key_env: str | None = typer.Option(
        None,
        "--signing-key-env",
        help="Env var holding the HMAC signing key (bundle is unsigned if unset).",
    ),
) -> None:
    """Execute a causal proven run and mint a proof bundle (WAVE2.md §4)."""
    from pydantic import ValidationError

    from quant_fund.pit.vault import PitVault
    from quant_fund.proof.runner import run_proven
    from quant_fund.proofcore.contracts import ProofcoreError, RunSpec

    try:
        run_spec = RunSpec.model_validate(json.loads(spec.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        typer.echo(f"proof run failed: spec unreadable: {exc.__class__.__name__}", err=True)
        raise typer.Exit(code=2) from exc
    except ValidationError as exc:
        typer.echo(f"proof run failed: spec invalid: {exc.error_count()} errors", err=True)
        raise typer.Exit(code=2) from exc

    signing_key: bytes | None = None
    if signing_key_env is not None:
        raw_key = os.environ.get(signing_key_env)
        if not raw_key:
            typer.echo(
                f"proof run failed: signing key env var {signing_key_env!r} is unset or empty",
                err=True,
            )
            raise typer.Exit(code=2)
        signing_key = raw_key.encode()

    try:
        ok, result = run_proven(
            run_spec,
            vault=PitVault(vault_root),
            bundle_dir=bundle_dir,
            signing_key=signing_key,
        )
    except ProofcoreError as exc:
        # Fail-closed run (§4.4): ProofError / LeakageError / VaultError.
        typer.echo(f"proof run failed: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    if not ok:
        typer.echo(f"proof run failed: {result}", err=True)
        raise typer.Exit(code=2)
    typer.echo(json.dumps({"ok": True, "bundle_id": result}))


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


@proof_app.command("replay")
def replay_cmd(
    bundle: Path = typer.Option(..., "--bundle", help="Path to bundles/<id>.json."),
    bundle_dir: Path = typer.Option(..., "--bundle-dir", help="Proof bundle chain directory."),
    vault_root: Path | None = typer.Option(
        None, "--vault-root", help="PIT vault root to re-execute against."
    ),
) -> None:
    """Replay a proven bundle bit-exactly and print the verdict json (WAVE2.md §5)."""
    from quant_fund.proof.replay import replay_bundle

    vault = None
    if vault_root is not None:
        from quant_fund.pit.vault import PitVault

        vault = PitVault(vault_root)
    ok, detail = replay_bundle(
        bundle,
        bundle_dir=bundle_dir,
        pit_root=vault_root,
        vault=vault,
    )
    typer.echo(detail)
    if not ok:
        raise typer.Exit(code=1)


@proof_app.command("chain-head")
def chain_head_cmd(
    dir: Path = typer.Option(..., "--dir", help="Proof bundle chain directory."),
) -> None:
    """Print the current head bundle hash (CI pins it as proofchain-head.txt)."""
    from quant_fund.proof.bundle import chain_head

    typer.echo(chain_head(dir))
