"""Canonical JSON for ledger commitments.

The encoding is part of the hash input. Verification re-encodes each parsed
line and rejects bytes that are not exactly this form, so whitespace, key
reordering, and Unicode escaping changes are visible.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime
from typing import Any

from quant_fund.audit.errors import AuditError


def json_safe(value: Any) -> Any:
    """Convert a payload into strict JSON values.

    Non-finite floats become ``None``. Datetimes become ISO-8601 strings.
    This does not invent replacements for missing data; ``None`` is an explicit
    null in the committed bytes.
    """
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, str):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return json_safe(item())
        except (TypeError, ValueError):
            pass
    raise AuditError(f"value of type {type(value).__name__} is not ledger JSON")


def canonical_json_bytes(value: Any) -> bytes:
    """Sorted-key, tight-separator, ASCII-escaped JSON with no NaN tokens."""
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise AuditError(f"value is not canonical JSON: {exc}") from exc
    return text.encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
