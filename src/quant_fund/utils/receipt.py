"""Shared receipt-seal primitive.

One convention repo-wide: ``receipt_sha256`` binds the canonical JSON bytes of
the payload, computed with a small exclusion set for keys attached post-seal
(the digest itself and any filesystem-path metadata). Lane packages wrap this
with their own exclusion set and honesty contract.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

ALWAYS_EXCLUDED = ("receipt_sha256",)


def seal_receipt(receipt: Mapping[str, Any], *, exclude: Iterable[str] = ()) -> dict[str, Any]:
    """Copy of *receipt* with ``receipt_sha256`` bound over canonical JSON."""
    dropped = set(exclude) | set(ALWAYS_EXCLUDED)
    unsigned = {key: value for key, value in receipt.items() if key not in dropped}
    return {**receipt, "receipt_sha256": hash_bytes(canonical_json_bytes(unsigned))}


def seal_errors(receipt: Mapping[str, Any], *, exclude: Iterable[str] = ()) -> list[str]:
    """Digest errors for a sealed receipt; ``[]`` when the seal matches."""
    dropped = set(exclude) | set(ALWAYS_EXCLUDED)
    unsigned = {key: value for key, value in receipt.items() if key not in dropped}
    expected = hash_bytes(canonical_json_bytes(unsigned))
    return [] if receipt.get("receipt_sha256") == expected else ["receipt_sha256"]
