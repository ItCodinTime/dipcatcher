"""Verify an audit ledger the way a Certificate Transparency monitor would.

``valid`` means the bytes that are present have an intact hash chain, every
checkpoint signature matches the Merkle root of that prefix, and consecutive
signed sizes are consistent. It does not by itself prove that a previously
seen tip was not deleted together with its signed tree head. Pass
``expect_size`` and ``expect_root`` (a witness of a root you have already
seen) to detect that truncation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quant_fund.audit.canonical import canonical_json_bytes, sha256_hex
from quant_fund.audit.errors import AuditError, SignatureUnavailableError
from quant_fund.audit.ledger import SCHEMA_VERSION, load_jsonl, read_entries
from quant_fund.audit.merkle import (
    consistency_proof,
    hash_leaf,
    merkle_root,
    verify_consistency,
    verify_inclusion,
)
from quant_fund.audit.signing import key_id_for, verify_ed25519, verify_sigstore_bundle

_CHECKPOINT_KEYS = frozenset(
    {
        "v",
        "scheme",
        "key_id",
        "tree_size",
        "merkle_root",
        "timestamp_utc",
        "signature",
        "public_key",
        "sigstore_bundle_json",
        "sigstore_identity",
        "sigstore_issuer",
    }
)


def signed_tree_head_bytes(checkpoint: dict[str, Any]) -> bytes:
    """Bytes covered by the checkpoint signature. Excludes the signature itself."""
    payload = {
        "v": checkpoint["v"],
        "scheme": checkpoint["scheme"],
        "key_id": checkpoint["key_id"],
        "tree_size": checkpoint["tree_size"],
        "merkle_root": checkpoint["merkle_root"],
        "timestamp_utc": checkpoint["timestamp_utc"],
    }
    return canonical_json_bytes(payload)


def verify_ledger(
    root: Path | str,
    *,
    trust_public_key: bytes | None = None,
    expect_size: int | None = None,
    expect_root: str | None = None,
    sigstore_identity: str | None = None,
    sigstore_issuer: str | None = None,
    sigstore_offline: bool = False,
) -> dict[str, Any]:
    """Verify ``root`` and return a JSON-ready report. Never raises on tamper."""
    directory = Path(root)
    errors: list[str] = []
    if not directory.is_dir():
        return _report(
            valid=False,
            errors=["ledger_missing"],
            tree_size=0,
            signed_tree_size=0,
            merkle_root=None,
            trust_anchor="none",
            checkpoints=0,
        )
    entries, entry_errors = read_entries(directory / "entries.jsonl")
    errors.extend(entry_errors)
    preimages: list[bytes] = []
    for entry in entries:
        try:
            preimages.append(entry.preimage())
        except AuditError:
            errors.append(f"preimage_error:{entry.index}")
            preimages.append(b"")
    root_bytes = merkle_root(preimages) if preimages and all(preimages) else merkle_root([])

    objects, checkpoint_file_errors = load_jsonl(directory / "checkpoints.jsonl")
    for item in checkpoint_file_errors:
        errors.append(item if item.startswith("checkpoint_") else f"checkpoint_{item}")
    checkpoints: list[dict[str, Any]] = []
    for position, obj in enumerate(objects):
        extra = set(obj) - _CHECKPOINT_KEYS
        if extra:
            errors.append(f"unexpected_checkpoint_field:{position}")
        checkpoints.append(obj)

    trust_anchor = "pinned" if trust_public_key is not None else "bundled"
    if trust_public_key is None and not checkpoints:
        trust_anchor = "none"

    signed_tree_size = 0
    previous: tuple[int, bytes] | None = None
    for position, checkpoint in enumerate(checkpoints):
        errors_before = len(errors)
        covered = _check_checkpoint(
            checkpoint,
            position,
            preimages,
            errors,
            trust_public_key=trust_public_key,
            sigstore_identity=sigstore_identity,
            sigstore_issuer=sigstore_issuer,
            sigstore_offline=sigstore_offline,
            previous=previous,
        )
        if covered and len(errors) == errors_before:
            signed_tree_size = max(signed_tree_size, covered)
        size = checkpoint.get("tree_size")
        claimed = checkpoint.get("merkle_root")
        if isinstance(size, int) and not isinstance(size, bool) and isinstance(claimed, str):
            try:
                claimed_bytes = bytes.fromhex(claimed)
            except ValueError:
                claimed_bytes = b""
            if len(claimed_bytes) == 32 and 0 < size <= len(preimages):
                if previous is not None and size < previous[0]:
                    pass
                previous = (size, claimed_bytes)

    if entries and not checkpoints:
        errors.append("unsigned_log")

    if expect_size is not None and expect_size != len(entries):
        errors.append("witness_size_mismatch")
    if expect_root is not None and expect_root != root_bytes.hex():
        errors.append("witness_root_mismatch")

    # Inclusion of every leaf under the computed root is a self-check of the
    # proof code against this log. A broken proof implementation fails closed.
    if entries and not entry_errors:
        for entry, preimage in zip(entries, preimages, strict=True):
            proof = _safe_inclusion(preimages, entry.index)
            if proof is None or not verify_inclusion(
                hash_leaf(preimage),
                entry.index,
                proof,
                len(entries),
                root_bytes,
            ):
                errors.append(f"inclusion_self_check_failed:{entry.index}")
                break

    valid = not errors
    fully_signed = valid and (len(entries) == 0 or signed_tree_size == len(entries))
    return _report(
        valid=valid,
        errors=errors,
        tree_size=len(entries),
        signed_tree_size=signed_tree_size if checkpoints else 0,
        merkle_root=root_bytes.hex() if entries else merkle_root([]).hex(),
        trust_anchor=trust_anchor if checkpoints else "none",
        checkpoints=len(checkpoints),
        fully_signed=fully_signed,
        unsigned_suffix=max(0, len(entries) - signed_tree_size) if checkpoints else len(entries),
    )


def _safe_inclusion(preimages: list[bytes], index: int) -> list[bytes] | None:
    from quant_fund.audit.merkle import inclusion_proof

    try:
        return inclusion_proof(preimages, index)
    except AuditError:
        return None


def _check_checkpoint(
    checkpoint: dict[str, Any],
    position: int,
    preimages: list[bytes],
    errors: list[str],
    *,
    trust_public_key: bytes | None,
    sigstore_identity: str | None,
    sigstore_issuer: str | None,
    sigstore_offline: bool,
    previous: tuple[int, bytes] | None,
) -> int:
    """Validate one checkpoint. Returns the tree size when the root matches."""
    label = str(position)
    version = checkpoint.get("v")
    scheme = checkpoint.get("scheme")
    key_id = checkpoint.get("key_id")
    tree_size = checkpoint.get("tree_size")
    claimed = checkpoint.get("merkle_root")
    timestamp = checkpoint.get("timestamp_utc")
    if version != SCHEMA_VERSION:
        errors.append(f"checkpoint_bad_version:{label}")
        return 0
    if not isinstance(scheme, str) or scheme not in {"ed25519", "sigstore"}:
        errors.append(f"checkpoint_unknown_scheme:{label}")
        return 0
    if not isinstance(key_id, str) or not key_id:
        errors.append(f"checkpoint_bad_key_id:{label}")
        return 0
    if isinstance(tree_size, bool) or not isinstance(tree_size, int) or tree_size < 1:
        errors.append(f"checkpoint_bad_tree_size:{label}")
        return 0
    if tree_size > len(preimages):
        errors.append(f"checkpoint_tree_size:{label}")
        return 0
    if not isinstance(claimed, str):
        errors.append(f"checkpoint_bad_root:{label}")
        return 0
    if not isinstance(timestamp, str) or not timestamp:
        errors.append(f"checkpoint_bad_timestamp:{label}")
        return 0
    try:
        actual = merkle_root(preimages[:tree_size]).hex()
    except AuditError:
        errors.append(f"checkpoint_root_mismatch:{label}")
        return 0
    if actual != claimed:
        errors.append(f"checkpoint_root_mismatch:{label}")
    if previous is not None and tree_size < previous[0]:
        errors.append(f"checkpoint_order:{label}")
    elif previous is not None and tree_size > previous[0] and actual == claimed:
        try:
            claimed_bytes = bytes.fromhex(claimed)
            proof = consistency_proof(preimages[:tree_size], previous[0])
            if not verify_consistency(previous[0], tree_size, previous[1], claimed_bytes, proof):
                errors.append(f"consistency_failed:{label}")
        except (AuditError, ValueError):
            errors.append(f"consistency_failed:{label}")

    try:
        payload = signed_tree_head_bytes(checkpoint)
    except (KeyError, AuditError):
        errors.append(f"checkpoint_unsigned_payload:{label}")
        return tree_size if actual == claimed else 0

    if scheme == "ed25519":
        _check_ed25519(checkpoint, payload, key_id, label, errors, trust_public_key)
    else:
        _check_sigstore(
            checkpoint,
            payload,
            label,
            errors,
            identity=sigstore_identity,
            issuer=sigstore_issuer,
            offline=sigstore_offline,
        )
    return tree_size if actual == claimed else 0


def _check_ed25519(
    checkpoint: dict[str, Any],
    payload: bytes,
    key_id: str,
    label: str,
    errors: list[str],
    trust_public_key: bytes | None,
) -> None:
    embedded = checkpoint.get("public_key")
    signature = checkpoint.get("signature")
    if not isinstance(embedded, str):
        errors.append(f"missing_public_key:{label}")
        return
    try:
        embedded_bytes = bytes.fromhex(embedded)
    except ValueError:
        errors.append(f"missing_public_key:{label}")
        return
    if trust_public_key is not None and embedded_bytes != trust_public_key:
        errors.append(f"trust_anchor_mismatch:{label}")
        verify_key = trust_public_key
    else:
        verify_key = embedded_bytes if trust_public_key is None else trust_public_key
    if key_id != key_id_for(verify_key):
        errors.append(f"key_id_mismatch:{label}")
    if not isinstance(signature, str) or not verify_ed25519(payload, signature, verify_key):
        errors.append(f"signature_invalid:{label}")


def _check_sigstore(
    checkpoint: dict[str, Any],
    payload: bytes,
    label: str,
    errors: list[str],
    *,
    identity: str | None,
    issuer: str | None,
    offline: bool,
) -> None:
    bundle = checkpoint.get("sigstore_bundle_json")
    recorded_identity = checkpoint.get("sigstore_identity")
    recorded_issuer = checkpoint.get("sigstore_issuer")
    signature = checkpoint.get("signature")
    if not isinstance(bundle, str) or not bundle:
        errors.append(f"sigstore_bundle_missing:{label}")
        return
    expected_signature = sha256_hex(bundle.encode("utf-8"))
    if signature != expected_signature:
        errors.append(f"sigstore_bundle_hash_mismatch:{label}")
    pinned_identity = identity or (
        recorded_identity if isinstance(recorded_identity, str) else None
    )
    pinned_issuer = (
        issuer
        if issuer is not None
        else (recorded_issuer if isinstance(recorded_issuer, str) else None)
    )
    if identity is None:
        errors.append(f"sigstore_identity_unpinned:{label}")
    if not isinstance(pinned_identity, str) or not pinned_identity:
        errors.append(f"sigstore_unavailable:{label}")
        return
    try:
        verify_sigstore_bundle(
            payload,
            bundle,
            identity=pinned_identity,
            issuer=pinned_issuer,
            offline=offline,
        )
    except SignatureUnavailableError:
        errors.append(f"sigstore_unavailable:{label}")
    except AuditError:
        errors.append(f"signature_invalid:{label}")


def _report(
    *,
    valid: bool,
    errors: list[str],
    tree_size: int,
    signed_tree_size: int,
    merkle_root: str | None,
    trust_anchor: str,
    checkpoints: int,
    fully_signed: bool = False,
    unsigned_suffix: int = 0,
) -> dict[str, Any]:
    return {
        "schema": "audit-ledger/1",
        "valid": valid,
        "fully_signed": fully_signed,
        "tree_size": tree_size,
        "signed_tree_size": signed_tree_size,
        "unsigned_suffix": unsigned_suffix,
        "merkle_root": merkle_root,
        "trust_anchor": trust_anchor,
        "checkpoints": checkpoints,
        "errors": errors,
        "live_pnl_claim": False,
    }


def explain(report: dict[str, Any]) -> str:
    """Stable JSON text for the CLI."""
    return json.dumps(report, indent=2, sort_keys=True)
