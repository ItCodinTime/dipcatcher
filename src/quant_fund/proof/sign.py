"""Proof bundle signing (DESIGN.md §5.4).

HMAC-SHA256 behind a ``Signer`` protocol: ed25519 requires a new heavy
dependency (forbidden by the global constraints), HMAC gives tamper-evidence
for the CI-held key, and the protocol admits ed25519 later without a schema
change. Verification uses ``hmac.compare_digest`` (the ``models/base.py``
precedent). Keys come from the environment, never from repo files.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from typing import Protocol, runtime_checkable

from quant_fund.proofcore.contracts import SignatureUnavailableError

__all__ = ["SIGNING_KEY_ENV", "HmacSha256Signer", "NullSigner", "Signer"]

SIGNING_KEY_ENV = "PROOFCORE_SIGNING_KEY"


@runtime_checkable
class Signer(Protocol):
    """Hex-signature producer over canonical payload bytes."""

    scheme: str
    key_id: str

    def sign(self, payload: bytes) -> str:
        """Return the hex signature over ``payload``."""
        ...


class HmacSha256Signer:
    """HMAC-SHA256 signer. ``key_id = sha256(key)[:16]`` (never the key itself)."""

    scheme = "hmac-sha256"

    def __init__(self, key: bytes) -> None:
        if not key:
            raise SignatureUnavailableError("signing key must be non-empty")
        self._key = bytes(key)
        self.key_id = hashlib.sha256(self._key).hexdigest()[:16]

    @classmethod
    def from_env(cls) -> HmacSha256Signer:
        """Load the key from PROOFCORE_SIGNING_KEY (hex or raw bytes).

        Raises SignatureUnavailableError if the env var is missing or empty.
        """
        raw = os.environ.get(SIGNING_KEY_ENV, "")
        if not raw:
            raise SignatureUnavailableError(
                f"{SIGNING_KEY_ENV} is not set; signature cannot be produced or checked"
            )
        try:
            key = bytes.fromhex(raw)
        except ValueError:
            key = raw.encode("utf-8")
        return cls(key)

    def sign(self, payload: bytes) -> str:
        return hmac.new(self._key, payload, hashlib.sha256).hexdigest()

    def verify(self, payload: bytes, signature_hex: str) -> bool:
        """Constant-time check of ``signature_hex`` against the payload."""
        expected = self.sign(payload)
        return hmac.compare_digest(expected, signature_hex)


class NullSigner:
    """Unsigned bundles for local dev (scheme ``none``)."""

    scheme = "none"
    key_id = "unsigned"

    def sign(self, payload: bytes) -> str:
        return ""
