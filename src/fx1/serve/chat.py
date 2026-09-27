"""Serving utilities: receipts-cited chat wrapper around fx-1 backends.

Every fx-1 answer that references research evidence should carry the receipt
hash it rests on. ``cited_complete`` runs the model, validates honesty, and
appends the provenance footer — the serving path enforces in production what
training teaches.
"""

from __future__ import annotations

from fx1.honesty import validate_fx1_output
from fx1.serve.backends import InferenceBackend


def cited_complete(
    backend: InferenceBackend,
    messages: list[dict[str, str]],
    *,
    receipt_hashes: list[str] | None = None,
) -> str:
    """Complete with honesty validation and provenance footer."""
    response = backend.complete(messages)
    validate_fx1_output(response)  # fail-closed on contract violations
    if receipt_hashes:
        footer = (
            "\n\nEvidence: "
            + ", ".join(f"`{h[:16]}…`" for h in receipt_hashes)
            + " — verify with `dipcatcher verify-research`."
        )
        response += footer
    return response
