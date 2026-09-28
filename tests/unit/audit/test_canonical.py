"""Coverage for audit.canonical — strict ledger JSON encoding."""

from __future__ import annotations

from datetime import UTC, date, datetime

import numpy as np
import pytest

from quant_fund.audit.canonical import (
    canonical_json_bytes,
    json_safe,
    sha256_hex,
)
from quant_fund.audit.errors import AuditError


class TestJsonSafe:
    def test_scalars_and_containers(self) -> None:
        value = {
            "a": [1, 2.5, "x", None, True],
            "b": (1, 2),
            "c": {"nested": 1},
        }
        out = json_safe(value)
        assert out == {
            "a": [1, 2.5, "x", None, True],
            "b": [1, 2],
            "c": {"nested": 1},
        }

    def test_nonfinite_becomes_none(self) -> None:
        assert json_safe(float("nan")) is None
        assert json_safe(float("inf")) is None
        assert json_safe(float("-inf")) is None
        assert json_safe({"x": float("nan")}) == {"x": None}

    def test_datetimes_iso(self) -> None:
        dt = datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)
        assert json_safe(dt) == dt.isoformat()
        d = date(2024, 1, 2)
        assert json_safe(d) == "2024-01-02"

    def test_numpy_scalars_via_item(self) -> None:
        assert json_safe(np.int64(7)) == 7
        assert json_safe(np.float64(1.5)) == 1.5
        assert json_safe(np.float64("nan")) is None
        # Multi-element .item() raises ValueError -> falls through to AuditError.
        with pytest.raises(AuditError, match="not ledger JSON"):
            json_safe(np.array([1, 2, 3]))

    def test_unconvertible_raises(self) -> None:
        with pytest.raises(AuditError, match="not ledger JSON"):
            json_safe(object())
        with pytest.raises(AuditError, match="not ledger JSON"):
            json_safe({"k": b"bytes"})

    def test_item_raises_typeerror_falls_through(self) -> None:
        class Bad:
            def item(self) -> object:
                raise TypeError("nope")

        with pytest.raises(AuditError, match="not ledger JSON"):
            json_safe(Bad())


class TestCanonicalBytes:
    def test_sorted_tight_ascii(self) -> None:
        out = canonical_json_bytes({"b": 1, "a": "ü"})
        assert out == b'{"a":"\\u00fc","b":1}'

    def test_deterministic(self) -> None:
        v = {"x": [3, 1, 2], "y": {"z": 1, "a": 2}}
        assert canonical_json_bytes(v) == canonical_json_bytes(dict(reversed(list(v.items()))))

    def test_nan_rejected(self) -> None:
        with pytest.raises(AuditError, match="canonical JSON"):
            canonical_json_bytes(float("nan"))

    def test_unserializable_rejected(self) -> None:
        with pytest.raises(AuditError, match="canonical JSON"):
            canonical_json_bytes(object())

    def test_sha256_hex(self) -> None:
        import hashlib

        assert sha256_hex(b"abc") == hashlib.sha256(b"abc").hexdigest()
