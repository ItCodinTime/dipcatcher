"""RFC 6962 Merkle trees for an append-only audit log.

Leaves stay in log order. This is not ``quant_fund.proofcore.merkle_root_hex``,
which sorts leaves and therefore cannot prove that an entry kept its position.
Domain separation matches the RFC: ``0x00 || leaf`` for leaves and
``0x01 || left || right`` for nodes. Inclusion and consistency proofs follow
RFC 6962 sections 2.1.1 and 2.1.2. Verification follows the iterative audit-path
algorithm used by the Certificate Transparency reference client, and it rejects
a proof that is shorter or longer than the path the tree size requires.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Callable

from quant_fund.audit.errors import ProofError

_LEAF = b"\x00"
_NODE = b"\x01"


def hash_empty() -> bytes:
    """MTH({}) = SHA-256()."""
    return hashlib.sha256(b"").digest()


def hash_leaf(data: bytes) -> bytes:
    """MTH of a one-element list: SHA-256(0x00 || d)."""
    return hashlib.sha256(_LEAF + data).digest()


def hash_node(left: bytes, right: bytes) -> bytes:
    """SHA-256(0x01 || left || right)."""
    return hashlib.sha256(_NODE + left + right).digest()


def _split(width: int) -> int:
    """Largest power of two strictly smaller than ``width`` (RFC 6962 ``k``)."""
    if width <= 1:
        raise ProofError(f"cannot split a tree of width {width}")
    return 1 << ((width - 1).bit_length() - 1)


def _mth_fn(leaves: list[bytes]) -> Callable[[int, int], bytes]:
    cache: dict[tuple[int, int], bytes] = {}

    def mth(lo: int, hi: int) -> bytes:
        key = (lo, hi)
        found = cache.get(key)
        if found is not None:
            return found
        width = hi - lo
        if width == 1:
            digest = hash_leaf(leaves[lo])
        elif width < 1:
            raise ProofError("empty range has no subtree hash")
        else:
            k = _split(width)
            digest = hash_node(mth(lo, lo + k), mth(lo + k, hi))
        cache[key] = digest
        return digest

    return mth


def merkle_root(leaves: list[bytes]) -> bytes:
    """Merkle tree hash of raw leaf preimages in order.

    An empty log hashes the empty string, as in RFC 6962.
    """
    if not leaves:
        return hash_empty()
    return _mth_fn(leaves)(0, len(leaves))


def inclusion_proof(leaves: list[bytes], index: int) -> list[bytes]:
    """Audit path for ``leaves[index]``, from the leaf toward the root."""
    size = len(leaves)
    if size == 0 or not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < size:
        raise ProofError(f"inclusion index {index} is outside tree size {size}")
    mth = _mth_fn(leaves)
    proof: list[bytes] = []

    def walk(m: int, lo: int, hi: int) -> None:
        width = hi - lo
        if width == 1:
            return
        k = _split(width)
        if m < k:
            walk(m, lo, lo + k)
            proof.append(mth(lo + k, hi))
        else:
            walk(m - k, lo + k, hi)
            proof.append(mth(lo, lo + k))

    walk(index, 0, size)
    return proof


def consistency_proof(leaves: list[bytes], old_size: int) -> list[bytes]:
    """Minimal proof that the first ``old_size`` leaves are a prefix of ``leaves``.

    Defined for ``0 < old_size <= len(leaves)``. Equal sizes produce an empty
    proof. The empty tree is not a previous signed size in this construction.
    """
    size = len(leaves)
    if not isinstance(old_size, int) or isinstance(old_size, bool) or not 0 < old_size <= size:
        raise ProofError(f"consistency size {old_size} is outside 1..{size}")
    if old_size == size:
        return []
    mth = _mth_fn(leaves)
    proof: list[bytes] = []

    def sub(m: int, lo: int, hi: int, complete: bool) -> None:
        width = hi - lo
        if m == width:
            if not complete:
                proof.append(mth(lo, hi))
            return
        k = _split(width)
        if m <= k:
            sub(m, lo, lo + k, complete)
            proof.append(mth(lo + k, hi))
        else:
            sub(m - k, lo + k, hi, False)
            proof.append(mth(lo, lo + k))

    sub(old_size, 0, size, True)
    return proof


def root_from_inclusion(leaf_hash: bytes, index: int, proof: list[bytes], tree_size: int) -> bytes:
    """Recompute the root from one leaf hash and its audit path.

    The other leaves are not an input. A wrong path, a wrong index, or an
    extra hash raises ``ProofError``.
    """
    if (
        not isinstance(index, int)
        or isinstance(index, bool)
        or not isinstance(tree_size, int)
        or isinstance(tree_size, bool)
        or tree_size <= 0
        or index < 0
        or index >= tree_size
    ):
        raise ProofError(f"bad inclusion coordinates index={index} size={tree_size}")
    if len(leaf_hash) != 32 or any(len(node) != 32 for node in proof):
        raise ProofError("Merkle hashes must be 32 bytes")
    calculated = leaf_hash
    node_index = index
    last_node = tree_size - 1
    cursor = 0
    while last_node > 0:
        if node_index % 2 == 1:
            if cursor >= len(proof):
                raise ProofError("inclusion proof is too short")
            calculated = hash_node(proof[cursor], calculated)
            cursor += 1
        elif node_index < last_node:
            if cursor >= len(proof):
                raise ProofError("inclusion proof is too short")
            calculated = hash_node(calculated, proof[cursor])
            cursor += 1
        node_index //= 2
        last_node //= 2
    if cursor != len(proof):
        raise ProofError("inclusion proof is too long")
    return calculated


def verify_inclusion(
    leaf_hash: bytes,
    index: int,
    proof: list[bytes],
    tree_size: int,
    root: bytes,
) -> bool:
    """Return whether ``proof`` places ``leaf_hash`` at ``index`` under ``root``."""
    try:
        calculated = root_from_inclusion(leaf_hash, index, proof, tree_size)
    except ProofError:
        return False
    return hmac.compare_digest(calculated, root)


def verify_consistency(
    old_size: int,
    new_size: int,
    old_root: bytes,
    new_root: bytes,
    proof: list[bytes],
) -> bool:
    """Return whether ``new_root`` extends ``old_root`` by append-only growth."""
    try:
        _verify_consistency(old_size, new_size, old_root, new_root, proof)
    except ProofError:
        return False
    return True


def _verify_consistency(
    old_size: int,
    new_size: int,
    old_root: bytes,
    new_root: bytes,
    proof: list[bytes],
) -> None:
    if (
        not isinstance(old_size, int)
        or isinstance(old_size, bool)
        or not isinstance(new_size, int)
        or isinstance(new_size, bool)
        or old_size < 0
        or new_size < 0
    ):
        raise ProofError("tree sizes must be non-negative integers")
    if len(old_root) != 32 or len(new_root) != 32 or any(len(node) != 32 for node in proof):
        raise ProofError("Merkle hashes must be 32 bytes")
    if old_size > new_size:
        raise ProofError("consistency proof is for a shrinking tree")
    if old_size == new_size:
        if proof:
            raise ProofError("equal tree sizes require an empty consistency proof")
        if not hmac.compare_digest(old_root, new_root):
            raise ProofError("equal sizes with different roots")
        return
    if old_size == 0:
        raise ProofError("consistency from an empty tree is not defined")

    node = old_size - 1
    last_node = new_size - 1
    while node % 2:
        node //= 2
        last_node //= 2
    cursor = 0
    if node:
        if cursor >= len(proof):
            raise ProofError("consistency proof is too short")
        old_hash = proof[cursor]
        new_hash = proof[cursor]
        cursor += 1
    else:
        old_hash = old_root
        new_hash = old_root
    while node:
        if node % 2:
            if cursor >= len(proof):
                raise ProofError("consistency proof is too short")
            sibling = proof[cursor]
            cursor += 1
            old_hash = hash_node(sibling, old_hash)
            new_hash = hash_node(sibling, new_hash)
        elif node < last_node:
            if cursor >= len(proof):
                raise ProofError("consistency proof is too short")
            new_hash = hash_node(new_hash, proof[cursor])
            cursor += 1
        node //= 2
        last_node //= 2
    while last_node:
        if cursor >= len(proof):
            raise ProofError("consistency proof is too short")
        new_hash = hash_node(new_hash, proof[cursor])
        cursor += 1
        last_node //= 2
    if cursor != len(proof):
        raise ProofError("consistency proof is too long")
    if not hmac.compare_digest(new_hash, new_root):
        raise ProofError("consistency proof does not rebuild the new root")
    if not hmac.compare_digest(old_hash, old_root):
        raise ProofError("consistency proof does not rebuild the old root")
