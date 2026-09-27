"""W5 contracts smoke tests (DESIGN.md §3, §8.1).

The contracts module is the coordination point for all five workstreams;
these tests pin the canonical-hash determinism contract (§8.1.3: no reliance
on dict/set iteration order) and the Merkle helper's domain separation.
"""

from __future__ import annotations

import pytest

from quant_fund.proofcore.contracts import (
    GENESIS_HASH,
    HASH_HEX_LEN,
    SCHEMA_VERSION,
    ProofError,
    canonical_json_bytes,
    merkle_root_hex,
    sha256_hex_bytes,
    sha256_hex_json,
)


def test_schema_version_and_genesis() -> None:
    assert SCHEMA_VERSION == "proofcore/1"
    assert GENESIS_HASH == "0" * 64


def test_canonical_json_is_key_order_independent() -> None:
    """§8.1.3: permuted insertion order must hash identically (PYTHONHASHSEED
    cannot save us post-interpreter-start, so sorting is the contract)."""
    a = {"z": 1, "a": [3, 2, 1], "m": {"b": True, "a": None}}
    b = {"m": {"a": None, "b": True}, "a": [3, 2, 1], "z": 1}
    assert canonical_json_bytes(a) == canonical_json_bytes(b)
    assert sha256_hex_json(a) == sha256_hex_json(b)


def test_canonical_json_tight_separators() -> None:
    assert canonical_json_bytes({"a": 1, "b": [2]}) == b'{"a":1,"b":[2]}'


def test_sha256_hex_bytes_length() -> None:
    assert len(sha256_hex_bytes(b"")) == HASH_HEX_LEN


def test_merkle_root_empty_hashes_empty_string() -> None:
    assert merkle_root_hex([]) == sha256_hex_bytes(b"")


def test_merkle_root_order_independent() -> None:
    leaves = [sha256_hex_bytes(bytes([i])) for i in range(5)]
    assert merkle_root_hex(leaves) == merkle_root_hex(list(reversed(leaves)))


def test_merkle_root_rejects_non_hex_leaves() -> None:
    with pytest.raises(ProofError):
        merkle_root_hex(["not-a-digest"])


def test_merkle_single_leaf_is_domain_separated() -> None:
    leaf = sha256_hex_bytes(b"x")
    root = merkle_root_hex([leaf])
    assert root == sha256_hex_bytes(b"PC:leaf:" + bytes.fromhex(leaf))
    assert root != leaf  # second-preimage resistance by domain separation
