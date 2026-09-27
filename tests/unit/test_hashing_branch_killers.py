"""Example tests aimed at surviving mutants in ``utils/hashing.py``."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import polars as pl

from quant_fund.utils.hashing import (
    canonical_frame_fingerprint,
    canonical_json_bytes,
    fingerprint,
    hash_bytes,
    hash_file,
    receipt_tree,
)


def _fp(**overrides: object) -> str:
    base: dict[str, object] = {
        "row_count": 3,
        "min_timestamp": "2024-01-02",
        "max_timestamp": "2024-01-04",
        "columns": ["a", "b"],
        "feature_version": "f",
        "universe_version": "u",
        "label_version": "l",
    }
    base.update(overrides)
    return fingerprint(**base)  # type: ignore[arg-type]


def test_hash_bytes_is_sha256_and_hash_file_reads_past_one_chunk() -> None:
    payload = b"abc"
    assert hash_bytes(payload) == hashlib.sha256(payload).hexdigest()
    assert hash_bytes(b"") == hashlib.sha256(b"").hexdigest()
    # Larger than the 1 MiB read so a one-chunk mutant drops the tail.
    blob = b"xyz" * (1 << 20)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "blob.bin"
        path.write_bytes(blob)
        assert hash_file(path) == hash_bytes(blob)


def test_canonical_json_bytes_are_compact_sorted_and_utf8() -> None:
    assert canonical_json_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}'
    assert canonical_json_bytes("café") == '"café"'.encode()
    assert canonical_json_bytes(b"\x00\xff") == b'"00ff"'
    assert json.loads(canonical_json_bytes((1, "z"))) == [1, "z"]
    assert json.loads(canonical_json_bytes({3, 1, 2})) == [1, 2, 3]
    stamped = datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)
    assert json.loads(canonical_json_bytes({"t": stamped}))["t"] == stamped.isoformat()
    assert json.loads(canonical_json_bytes({"d": date(2024, 1, 2)}))["d"] == "2024-01-02"
    assert canonical_json_bytes(float("nan")) == b"null"
    assert canonical_json_bytes({"x": float("inf")}) == b'{"x":null}'
    assert canonical_json_bytes(np.float64(1.5)) == b"1.5"
    # A length-1 array has a working ``item()``; skipping it would emit a list.
    assert canonical_json_bytes(np.array([1.5])) == b"1.5"
    assert json.loads(canonical_json_bytes(np.array([1.0, np.nan, 2.0]))) == [1.0, None, 2.0]
    assert canonical_json_bytes({1: "a", "1": "b"}) == canonical_json_bytes({"1": "b"})
    # Sort key uses ensure_ascii=False; "é" and "z" flip order if that flag flips.
    assert canonical_json_bytes({"é", "z"}) == '["z","é"]'.encode()
    assert canonical_json_bytes(frozenset({2, 1})) == b"[1,2]"
    # JSON text order is not Python's tuple order: "[10]" < "[2]".
    assert canonical_json_bytes({(2,), (10,)}) == b"[[10],[2]]"

    class _Box:
        def __str__(self) -> str:
            return "box"

        def __hash__(self) -> int:
            return 0

        def __eq__(self, other: object) -> bool:
            return isinstance(other, _Box)

    assert canonical_json_bytes(_Box()) == b'"box"'
    # Set ordering calls the sort-key dumps, which must accept this object.
    assert json.loads(canonical_json_bytes({_Box()})) == ["box"]
    assert receipt_tree({_Box()}) == [_Box()]


def test_receipt_tree_sorts_sets_and_keeps_nonfinite_floats() -> None:
    assert receipt_tree({2, 1, 3}) == [1, 2, 3]
    nested = receipt_tree({"k": np.array([1.0, np.nan])})
    assert nested["k"][0] == 1.0
    assert math.isnan(nested["k"][1])
    assert math.isnan(receipt_tree(float("nan")))
    assert receipt_tree({2: [1, {3}]}) == {"2": [1, [3]]}
    # Plain tuples and datetimes are not rewritten (digest compatibility).
    when = datetime(2024, 1, 2, tzinfo=UTC)
    assert receipt_tree((when, 1)) == (when, 1)
    assert receipt_tree({"é", "z"}) == ["z", "é"]
    assert receipt_tree({(2,), (10,)}) == [(10,), (2,)]
    assert isinstance(receipt_tree(np.array([1.0, 2.0])), list)
    assert receipt_tree(np.array([1.5])) == 1.5
    assert type(receipt_tree(np.float64(1.5))) is float

    class _Key(str):
        def __str__(self) -> str:
            return "rewritten"

    assert list(receipt_tree({_Key("a"): 1})) == ["a"]


def test_fingerprint_matches_an_independent_sha256() -> None:
    """Key names and JSON separators are part of the digest, not an accident."""
    payload = {
        "row_count": 3,
        "min_timestamp": "2024-01-02",
        "max_timestamp": "2024-01-04",
        "columns": ["a", "b"],
        "feature_version": "f",
        "universe_version": "u",
        "label_version": "l",
        "extra": {"k": 1},
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    assert _fp(extra={"k": 1}) == hashlib.sha256(blob).hexdigest()

    class _Box:
        def __str__(self) -> str:
            return "box"

    boxed = {
        "row_count": 3,
        "min_timestamp": "2024-01-02",
        "max_timestamp": "2024-01-04",
        "columns": ["a", "b"],
        "feature_version": "f",
        "universe_version": "u",
        "label_version": "l",
        "extra": {"k": "box"},
    }
    boxed_blob = json.dumps(boxed, sort_keys=True, default=str).encode()
    assert _fp(extra={"k": _Box()}) == hashlib.sha256(boxed_blob).hexdigest()
    assert _fp() == _fp(extra={})
    assert _fp(extra={"k": 1}) != _fp()
    assert _fp(columns=["b", "a"]) != _fp(columns=["a", "b"])


def test_frame_fingerprint_includes_schema_and_counts_duplicate_rows() -> None:
    frame = pl.DataFrame({"b": [1, 1], "a": [2, 2]})
    flipped = frame.select(["a", "b"])
    assert canonical_frame_fingerprint(frame) == canonical_frame_fingerprint(flipped)
    shorter = pl.DataFrame({"b": [1], "a": [2]})
    assert canonical_frame_fingerprint(frame) != canonical_frame_fingerprint(shorter)
    widened = frame.with_columns(pl.col("a").cast(pl.Int8))
    assert canonical_frame_fingerprint(frame) != canonical_frame_fingerprint(widened)
    one = pl.DataFrame({"b": [1], "a": [2]})
    body = {
        "columns": ["a", "b"],
        "schema": {name: str(one.schema[name]) for name in ("a", "b")},
        "rows": [{"a": 2, "b": 1}],
    }
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    assert canonical_frame_fingerprint(one) == hashlib.sha256(raw).hexdigest()
