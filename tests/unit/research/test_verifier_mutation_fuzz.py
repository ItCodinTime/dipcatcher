"""Metamorphic mutation testing for the receipt verifier.

The verifier is the evidence boundary: committed sealed receipts and
synthetic sealed payloads are mutated dozens of deterministic ways. The
contract is strict — a mutation that changes the parsed object must never
verify as valid, and no input may crash the verifier outright (a crash
mid-sweep denies verification of every receipt after it).
"""

from __future__ import annotations

import json
import math
import string
from pathlib import Path
from typing import Any

import pytest

from quant_fund.research.receipt_v2 import (
    seal_receipt,
    verify_receipt_file,
    verify_receipt_payload,
)

RECEIPTS_DIR = Path(__file__).resolve().parents[3] / "receipts"


def _sealed_receipts() -> list[Path]:
    """Committed receipts that currently verify clean under their seal."""
    sealed = []
    for path in sorted(RECEIPTS_DIR.glob("*.json")):
        if verify_receipt_file(path)["valid"]:
            sealed.append(path)
    return sealed


def _walk_mutations(node: Any, path: tuple[Any, ...] = ()) -> list[tuple[tuple[Any, ...], Any]]:
    """Deterministic (key-path, replacement) pairs for every scalar leaf."""
    out: list[tuple[tuple[Any, ...], Any]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            out.extend(_walk_mutations(value, (*path, key)))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            out.extend(_walk_mutations(value, (*path, index)))
    elif isinstance(node, bool):
        out.append((path, not node))
    elif isinstance(node, (int, float)):
        out.append((path, node + 1 if isinstance(node, int) else node * 1.001))
        out.append((path, -node if node != 0 else 1))
    elif isinstance(node, str):
        out.append((path, node + "x"))
    elif node is None:
        out.append((path, 0))
    return out


def _set_path(node: Any, path: tuple[Any, ...], value: Any) -> Any:
    clone = dict(node) if isinstance(node, dict) else list(node)
    if len(path) == 1:
        clone[path[0]] = value
    else:
        clone[path[0]] = _set_path(node[path[0]], path[1:], value)
    return clone


def _structured_mutations(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Body mutations bounded to ~40 so the suite stays fast on big receipts."""
    mutations: list[dict[str, Any]] = []
    for key in list(payload):
        removed = dict(payload)
        removed.pop(key)
        mutations.append(removed)
    mutations.append({**payload, "extra_forged_key": True})
    for path, replacement in _walk_mutations(payload)[:40]:
        mutations.append(_set_path(payload, path, replacement))
    return mutations


def test_committed_sealed_receipts_fail_under_body_mutation() -> None:
    sealed = _sealed_receipts()
    assert sealed, "no sealed committed receipts found — fuzz has nothing to mutate"
    for receipt_path in sealed:
        original = json.loads(receipt_path.read_text())
        assert verify_receipt_payload(original, receipt_path)["valid"]
        for index, mutated in enumerate(_structured_mutations(original)):
            result = verify_receipt_payload(mutated, receipt_path)
            assert not result["valid"], f"{receipt_path.name} mutation #{index} verified as valid"


def test_committed_sealed_receipts_fail_under_byte_mutation(tmp_path: Path) -> None:
    """Byte edits that change the parsed object must fail; whitespace may pass."""
    for receipt_path in _sealed_receipts():
        raw = receipt_path.read_bytes()
        original = json.loads(raw)
        semantic = [
            index
            for index, byte in enumerate(raw)
            if byte in b'"{:,}0123456789' + string.ascii_letters.encode()
        ]
        assert semantic
        step = max(1, len(semantic) // 48)
        for index in semantic[::step]:
            mutated = bytearray(raw)
            mutated[index] = ord("9") if raw[index] != ord("9") else ord("8")
            target = tmp_path / "mutated.json"
            target.write_bytes(bytes(mutated))
            try:
                parsed = json.loads(bytes(mutated))
            except json.JSONDecodeError:
                parsed = None
            if parsed is not None and parsed != original:
                result = verify_receipt_file(target)
                assert not result["valid"], (
                    f"{receipt_path.name} byte {index} flip verified as valid"
                )


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("live_pnl_claim",), True),
        (("data_label",), "REAL"),
        (("schema",), "forged.v1"),
    ],
    ids=["live_pnl_claim", "data_label", "schema"],
)
def test_resealed_forgery_fails_on_fleet_eval(path: tuple[Any, ...], replacement: Any) -> None:
    """Re-sealing tampered contract fields cannot rescue a deep-checked receipt."""
    targets = [
        path_
        for path_ in _sealed_receipts()
        if json.loads(path_.read_text()).get("schema") == "fleet_eval.v1"
    ]
    assert targets, "committed fleet_eval.v1 receipt missing"
    for receipt_path in targets:
        payload = json.loads(receipt_path.read_text())
        forged = seal_receipt(_set_path(payload, path, replacement))
        assert forged["receipt_sha256"] != payload["receipt_sha256"]
        assert not verify_receipt_payload(forged, receipt_path)["valid"]


def _minimal_sealed_v1() -> dict[str, Any]:
    return seal_receipt({"schema": "v1", "kind": "fuzz", "value": 1, "live_pnl_claim": False})


def test_synthetic_sealed_payload_mutations_fail() -> None:
    sealed = _minimal_sealed_v1()
    assert verify_receipt_payload(sealed)["valid"]
    for mutated in _structured_mutations(sealed):
        assert not verify_receipt_payload(mutated)["valid"]


@pytest.mark.parametrize(
    "document",
    [
        b"",
        b"   ",
        b"null",
        b"42",
        b'"string"',
        b"[1,2,3]",
        b"{",
        b"\xff\xfe\x00\x01",
        b'{"schema": "v1", "x": NaN, "receipt_sha256": "' + b"0" * 64 + b'"}',
        b'{"schema": "v1", "x": Infinity}',
        b'{"a":' * 2000 + b"1" + b"}" * 2000,
        b'{"schema": "v1", "receipt_sha256": ' + b"9" * 100000 + b"}",
    ],
    ids=[
        "empty",
        "whitespace",
        "null",
        "number",
        "string",
        "array",
        "truncated",
        "binary",
        "nan_literal",
        "infinity_literal",
        "deep_nesting",
        "huge_seal",
    ],
)
def test_verifier_never_crashes_on_adversarial_input(tmp_path: Path, document: bytes) -> None:
    target = tmp_path / "receipt.json"
    target.write_bytes(document)
    result = verify_receipt_file(target)
    assert isinstance(result["valid"], bool)
    assert result["errors"]


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_float_payload_fails_closed(value: float) -> None:
    sealed = seal_receipt({"schema": "v1", "kind": "fuzz", "value": 0.5})
    sealed["value"] = value
    sealed["receipt_sha256"] = "c" * 64
    result = verify_receipt_payload(sealed)
    assert not result["valid"]
    assert result["errors"]
