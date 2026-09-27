"""RFC 6962 structure and random inclusion/consistency round-trips."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from quant_fund.audit.merkle import (
    consistency_proof,
    hash_empty,
    hash_leaf,
    inclusion_proof,
    merkle_root,
    verify_consistency,
    verify_inclusion,
)


def test_empty_and_single_leaf() -> None:
    assert merkle_root([]) == hash_empty()
    leaves = [b"only"]
    root = merkle_root(leaves)
    proof = inclusion_proof(leaves, 0)
    assert proof == []
    assert verify_inclusion(hash_leaf(b"only"), 0, proof, 1, root)
    assert verify_consistency(1, 1, root, root, [])
    assert not verify_inclusion(hash_leaf(b"only"), 0, [b"\x00" * 32], 1, root)


def test_rfc6962_proof_lengths_for_seven_leaves() -> None:
    leaves = [f"d{index}".encode() for index in range(7)]
    assert [len(inclusion_proof(leaves, index)) for index in (0, 3, 4, 6)] == [3, 3, 3, 2]
    assert len(consistency_proof(leaves, 3)) == 4
    assert len(consistency_proof(leaves, 4)) == 1
    assert len(consistency_proof(leaves, 6)) == 3
    root = merkle_root(leaves)
    for index in range(7):
        proof = inclusion_proof(leaves, index)
        assert verify_inclusion(hash_leaf(leaves[index]), index, proof, 7, root)
    for old_size in range(1, 8):
        old_root = merkle_root(leaves[:old_size])
        proof = consistency_proof(leaves, old_size)
        assert verify_consistency(old_size, 7, old_root, root, proof)


def test_reordering_changes_the_root() -> None:
    forward = merkle_root([b"a", b"b", b"c"])
    backward = merkle_root([b"c", b"b", b"a"])
    assert forward != backward


def test_tampered_or_overlong_proof_fails() -> None:
    leaves = [b"a", b"b", b"c", b"d"]
    root = merkle_root(leaves)
    proof = inclusion_proof(leaves, 1)
    assert verify_inclusion(hash_leaf(b"b"), 1, proof, 4, root)
    damaged = bytearray(proof[0])
    damaged[0] ^= 0x01
    proof[0] = bytes(damaged)
    assert not verify_inclusion(hash_leaf(b"b"), 1, proof, 4, root)
    extra = inclusion_proof(leaves, 1) + [b"\x11" * 32]
    assert not verify_inclusion(hash_leaf(b"b"), 1, extra, 4, root)
    old = merkle_root(leaves[:2])
    consistency = consistency_proof(leaves, 2)
    assert not verify_consistency(2, 4, old, root, consistency + [b"\x22" * 32])
    assert not verify_consistency(2, 4, merkle_root([b"nope", b"nope"]), root, consistency)


@given(st.lists(st.binary(min_size=0, max_size=24), min_size=1, max_size=24))
@settings(max_examples=40, deadline=None)
def test_random_inclusion_and_consistency(leaves: list[bytes]) -> None:
    root = merkle_root(leaves)
    for index, leaf in enumerate(leaves):
        proof = inclusion_proof(leaves, index)
        assert verify_inclusion(hash_leaf(leaf), index, proof, len(leaves), root)
        assert not verify_inclusion(hash_leaf(leaf + b"\x00"), index, proof, len(leaves), root)
    for old_size in range(1, len(leaves) + 1):
        old_root = merkle_root(leaves[:old_size])
        proof = consistency_proof(leaves, old_size)
        assert verify_consistency(old_size, len(leaves), old_root, root, proof)
