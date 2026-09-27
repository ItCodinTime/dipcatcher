"""``quant proof`` sub-typer (DESIGN.md §5.8).

Honesty contract: this CLI prints bundle ids, hashes, signature scheme, and
verification verdicts/reasons — never headline metrics (AGENTS.md rule 1;
bundle ``metrics_recompute`` is a verification artifact, not output).

Mounting into ``quant_fund.cli._app`` is W5's glue (DESIGN.md §9.2).
``run``, ``verify``, and ``chain-head`` are registered only once
``proof.runner``, ``proof.verify``, and ``proof.bundle`` exist. Exposing
them earlier fails every invocation with ``ModuleNotFoundError``.
"""

from __future__ import annotations

import typer

__all__ = ["proof_app"]

proof_app = typer.Typer(
    help=(
        "Proof-carrying backtester: mint and verify signed, hash-chained proof bundles. "
        "Commands land with the runner, bundle, and verifier modules."
    )
)


@proof_app.callback()
def proof_callback() -> None:
    """Placeholder group until runner, bundle, and verifier commands register."""
