"""Errors for the append-only audit ledger.

Distinct from ``quant_fund.proofcore`` errors so this package does not import
the proof-bundle layer. Proof bundles commit an unordered set; this ledger is
an ordered Certificate Transparency log.
"""

from __future__ import annotations


class AuditError(Exception):
    """Ledger contract violation, or a payload that cannot be committed."""

    def __init__(self, message: str, *, errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.errors: list[str] = list(errors or [])


class SignatureUnavailableError(AuditError):
    """A signature was required and no key, token, or verifier was available.

    The ledger never substitutes an unsigned checkpoint or a fabricated
    signature when this is raised.
    """


class ProofError(AuditError):
    """An inclusion or consistency proof was malformed or did not verify."""
