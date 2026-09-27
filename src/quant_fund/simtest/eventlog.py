"""Compact recorded trace for deterministic simulation replay.

The on-disk form is a versioned binary framing of canonical JSON payloads,
compressed with zlib. Floats are stored as IEEE hex so a replay compares
bits, not decimal formatting.
"""

from __future__ import annotations

import math
import struct
import zlib
from datetime import datetime
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes

MAGIC = b"DST1"
VERSION = 1


class ReplayDivergence(RuntimeError):
    """A replayed call did not match the recorded trace."""


def freeze(value: Any) -> Any:
    """Replace floats, times, and bytes with tagged JSON-safe values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("simulation trace refuses non-finite floats")
        return {"$f": float(value).hex()}
    if isinstance(value, datetime):
        return {"$t": value.isoformat()}
    if isinstance(value, bytes):
        return {"$b": value.hex()}
    if isinstance(value, dict):
        return {str(key): freeze(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [freeze(item) for item in value]
    raise TypeError(f"unsupported trace value: {type(value).__name__}")


def thaw(value: Any) -> Any:
    """Inverse of :func:`freeze`."""
    if isinstance(value, list):
        return [thaw(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {"$f"}:
            return float.fromhex(str(value["$f"]))
        if set(value) == {"$t"}:
            return datetime.fromisoformat(str(value["$t"]))
        if set(value) == {"$b"}:
            return bytes.fromhex(str(value["$b"]))
        return {str(key): thaw(item) for key, item in value.items()}
    return value


class EventLog:
    """Append-only call trace. Replay pops events in record order."""

    def __init__(self) -> None:
        self._events: list[tuple[str, Any, Any]] = []
        self._cursor = 0

    def append(self, kind: str, request: Any, response: Any) -> None:
        self._events.append((kind, freeze(request), freeze(response)))

    def pop(self, kind: str, request: Any) -> Any:
        if self._cursor >= len(self._events):
            raise ReplayDivergence(f"trace exhausted at {kind}")
        got_kind, got_request, response = self._events[self._cursor]
        self._cursor += 1
        frozen_request = freeze(request)
        if got_kind != kind or got_request != frozen_request:
            raise ReplayDivergence(
                f"trace mismatch: expected {kind} {frozen_request!r}, got {got_kind} {got_request!r}"
            )
        return thaw(response)

    def to_bytes(self) -> bytes:
        parts: list[bytes] = [MAGIC, struct.pack("<HI", VERSION, len(self._events))]
        for kind, request, response in self._events:
            kind_raw = kind.encode("utf-8")
            req_raw = canonical_json_bytes(request)
            resp_raw = canonical_json_bytes(response)
            parts.append(struct.pack("<BI", len(kind_raw), len(req_raw)))
            parts.append(kind_raw)
            parts.append(req_raw)
            parts.append(struct.pack("<I", len(resp_raw)))
            parts.append(resp_raw)
        return zlib.compress(b"".join(parts), level=9)

    @classmethod
    def from_bytes(cls, blob: bytes) -> EventLog:
        try:
            raw = zlib.decompress(blob)
        except zlib.error as exc:
            raise ValueError("simulation trace is not a valid compressed log") from exc
        if len(raw) < 10 or raw[:4] != MAGIC:
            raise ValueError("simulation trace missing DST1 header")
        version, count = struct.unpack_from("<HI", raw, 4)
        if version != VERSION:
            raise ValueError(f"unsupported simulation trace version: {version}")
        log = cls()
        offset = 10
        for _ in range(count):
            if offset + 5 > len(raw):
                raise ValueError("truncated simulation trace")
            kind_len, req_len = struct.unpack_from("<BI", raw, offset)
            offset += 5
            kind = raw[offset : offset + kind_len].decode("utf-8")
            offset += kind_len
            request = thaw(_loads(raw[offset : offset + req_len]))
            offset += req_len
            if offset + 4 > len(raw):
                raise ValueError("truncated simulation trace")
            (resp_len,) = struct.unpack_from("<I", raw, offset)
            offset += 4
            response = thaw(_loads(raw[offset : offset + resp_len]))
            offset += resp_len
            log._events.append((kind, freeze(request), freeze(response)))
        if offset != len(raw):
            raise ValueError("simulation trace has trailing bytes")
        return log

    def __len__(self) -> int:
        return len(self._events)


def _loads(raw: bytes) -> Any:
    import json

    value: Any = json.loads(raw)
    return value
