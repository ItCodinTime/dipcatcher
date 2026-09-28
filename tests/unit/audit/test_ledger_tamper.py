"""Tamper with an audit ledger in every way that should be visible."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_fund.audit.canonical import canonical_json_bytes, sha256_hex
from quant_fund.audit.errors import AuditError
from quant_fund.audit.ledger import GENESIS_HASH, AuditLedger, load_jsonl
from quant_fund.audit.merkle import hash_empty, verify_consistency, verify_inclusion
from quant_fund.audit.merkle import hash_leaf as leaf_hash
from quant_fund.audit.signing import Ed25519Signer
from quant_fund.audit.verify import verify_ledger

_STAMP = "2020-01-01T00:00:00Z"


def _clock() -> str:
    return _STAMP


def _build(tmp_path: Path, *, sign_every: int = 1) -> tuple[AuditLedger, Ed25519Signer]:
    signer = Ed25519Signer.generate()
    ledger = AuditLedger(
        tmp_path / "ledger",
        signer=signer,
        sign_every=sign_every,
        sync=False,
        clock=_clock,
    )
    payloads = [
        ("research_run", {"receipt_sha256": "ab" * 32, "live_pnl_claim": False, "note": "run"}),
        ("paper_decision", {"simulation_only": True, "live_pnl_claim": False}),
        ("simulated_order", {"order_id": "o1", "simulation_only": True, "live_pnl_claim": False}),
        (
            "fill",
            {
                "order_id": "o1",
                "price": 1.25,
                "quantity": 2,
                "simulation_only": True,
                "live_pnl_claim": False,
            },
        ),
        ("risk_decision", {"order_id": "o1", "accepted": True, "simulation_only": True}),
    ]
    for kind, payload in payloads:
        ledger.append(kind, payload)
    return ledger, signer


def _lines(path: Path) -> list[bytes]:
    raw = path.read_bytes()
    parts = raw.split(b"\n")
    if parts and parts[-1] == b"":
        parts.pop()
    return parts


def _write_lines(path: Path, lines: list[bytes]) -> None:
    path.write_bytes(b"\n".join(lines) + (b"\n" if lines else b""))


def _objects(path: Path) -> list[dict[str, object]]:
    objects, errors = load_jsonl(path)
    assert errors == []
    return objects


def _dump(path: Path, objects: list[dict[str, object]]) -> None:
    body = b"\n".join(canonical_json_bytes(obj) for obj in objects) + b"\n"
    path.write_bytes(body)


def _rechain(objects: list[dict[str, object]]) -> None:
    previous = GENESIS_HASH
    for index, obj in enumerate(objects):
        obj["index"] = index
        obj["prev_hash"] = previous
        body = {
            "v": obj["v"],
            "index": obj["index"],
            "kind": obj["kind"],
            "recorded_at": obj["recorded_at"],
            "payload": obj["payload"],
            "prev_hash": obj["prev_hash"],
        }
        obj["entry_hash"] = sha256_hex(canonical_json_bytes(body))
        previous = str(obj["entry_hash"])


def test_honest_ledger_verifies_and_proves(tmp_path: Path) -> None:
    ledger, signer = _build(tmp_path)
    report = verify_ledger(ledger.root, trust_public_key=signer.public_key)
    assert report["valid"] is True
    assert report["fully_signed"] is True
    assert report["live_pnl_claim"] is False
    assert report["trust_anchor"] == "pinned"
    assert report["tree_size"] == 5
    assert report["signed_tree_size"] == 5
    assert report["errors"] == []
    preimages = ledger.leaf_preimages()
    root = bytes.fromhex(report["merkle_root"])
    proof = ledger.inclusion_proof(3)
    assert verify_inclusion(leaf_hash(preimages[3]), 3, proof, 5, root)
    old = ledger.consistency_proof(3)
    from quant_fund.audit.merkle import merkle_root

    assert verify_consistency(3, 5, merkle_root(preimages[:3]), root, old)
    witnessed = verify_ledger(
        ledger.root,
        expect_size=5,
        expect_root=report["merkle_root"],
    )
    assert witnessed["valid"] is True
    kinds = [entry.kind for entry in ledger.entries()]
    assert kinds == [
        "research_run",
        "paper_decision",
        "simulated_order",
        "fill",
        "risk_decision",
    ]
    assert ledger.entries()[0].prev_hash == GENESIS_HASH


def test_append_does_not_rewrite_earlier_bytes(tmp_path: Path) -> None:
    signer = Ed25519Signer.generate()
    ledger = AuditLedger(tmp_path / "ledger", signer=signer, sync=False, clock=_clock)
    ledger.append("paper_decision", {"simulation_only": True, "live_pnl_claim": False})
    entries_before = ledger.entries_path.read_bytes()
    checkpoints_before = ledger.checkpoints_path.read_bytes()
    ledger.append("risk_decision", {"accepted": True, "simulation_only": True})
    assert ledger.entries_path.read_bytes().startswith(entries_before)
    assert ledger.checkpoints_path.read_bytes().startswith(checkpoints_before)
    assert verify_ledger(ledger.root)["valid"] is True


def test_refuses_forbidden_research_headlines_and_keeps_live_claim_false(tmp_path: Path) -> None:
    ledger, _signer = _build(tmp_path)
    for key in ("sharpe", "PnL", "nav", "sortino", "calmar"):
        with pytest.raises(AuditError):
            ledger.append("research_run", {key: 1.0, "live_pnl_claim": False})
    entry = ledger.append(
        "research_run",
        {"live_pnl_claim": False, "note": "café", "nested": {"sharpe": "not a headline key"}},
    )
    assert entry.payload["note"] == "café"
    raw = ledger.entries_path.read_bytes()
    assert b"\\u00e9" in raw
    assert verify_ledger(ledger.root)["valid"] is True


def test_unicode_rewritten_as_utf8_is_noncanonical(tmp_path: Path) -> None:
    ledger, _signer = _build(tmp_path)
    objects = _objects(ledger.entries_path)
    objects[0]["payload"] = {"note": "café"}
    _rechain(objects)
    _dump(ledger.entries_path, objects)
    honest = ledger.entries_path.read_bytes()
    assert b"\\u00e9" in honest
    tampered = honest.replace(b"\\u00e9", "é".encode())
    ledger.entries_path.write_bytes(tampered)
    report = verify_ledger(ledger.root)
    assert report["valid"] is False
    assert any(item.startswith("noncanonical_line") for item in report["errors"])


def test_unsigned_suffix_is_distinct_from_an_unsigned_log(tmp_path: Path) -> None:
    ledger, _signer = _build(tmp_path, sign_every=2)
    # _build appends 5 entries. Checkpoints land on sizes 2 and 4. One entry remains.
    report = verify_ledger(ledger.root)
    assert report["valid"] is True
    assert report["fully_signed"] is False
    assert report["signed_tree_size"] == 4
    assert report["unsigned_suffix"] == 1
    ledger.checkpoints_path.unlink()
    unsigned = verify_ledger(ledger.root)
    assert unsigned["valid"] is False
    assert "unsigned_log" in unsigned["errors"]


def test_empty_directory_is_valid_and_a_missing_directory_is_not(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    report = verify_ledger(empty)
    assert report["valid"] is True
    assert report["tree_size"] == 0
    assert report["trust_anchor"] == "none"
    assert report["merkle_root"] == hash_empty().hex()
    missing = verify_ledger(tmp_path / "missing")
    assert missing["valid"] is False
    assert missing["errors"] == ["ledger_missing"]


def _error_codes(root: Path) -> list[str]:
    return list(verify_ledger(root)["errors"])


def test_edit_payload_without_rehash(tmp_path: Path) -> None:
    ledger, _signer = _build(tmp_path)
    objects = _objects(ledger.entries_path)
    payload = objects[0]["payload"]
    assert isinstance(payload, dict)
    payload["note"] = "edited"
    _dump(ledger.entries_path, objects)
    assert any(item.startswith("entry_hash_mismatch") for item in _error_codes(ledger.root))


def test_flip_entry_hash(tmp_path: Path) -> None:
    ledger, _signer = _build(tmp_path)
    objects = _objects(ledger.entries_path)
    digest = str(objects[2]["entry_hash"])
    last = digest[-1]
    flipped = "0" if last != "0" else "1"
    objects[2]["entry_hash"] = digest[:-1] + flipped
    _dump(ledger.entries_path, objects)
    assert any(item.startswith("entry_hash_mismatch") for item in _error_codes(ledger.root))


def test_rehash_one_entry_without_the_next_prev(tmp_path: Path) -> None:
    ledger, _signer = _build(tmp_path)
    objects = _objects(ledger.entries_path)
    payload = objects[1]["payload"]
    assert isinstance(payload, dict)
    payload["extra"] = True
    body = {
        "v": objects[1]["v"],
        "index": objects[1]["index"],
        "kind": objects[1]["kind"],
        "recorded_at": objects[1]["recorded_at"],
        "payload": objects[1]["payload"],
        "prev_hash": objects[1]["prev_hash"],
    }
    objects[1]["entry_hash"] = sha256_hex(canonical_json_bytes(body))
    _dump(ledger.entries_path, objects)
    assert any(item.startswith("prev_hash_mismatch:2") for item in _error_codes(ledger.root))


def test_rebuild_chain_but_keep_signatures(tmp_path: Path) -> None:
    ledger, _signer = _build(tmp_path)
    objects = _objects(ledger.entries_path)
    payload = objects[0]["payload"]
    assert isinstance(payload, dict)
    payload["note"] = "rebuilt"
    _rechain(objects)
    _dump(ledger.entries_path, objects)
    errors = _error_codes(ledger.root)
    assert any(item.startswith("checkpoint_root_mismatch") for item in errors)
    assert verify_ledger(ledger.root)["valid"] is False


def test_remove_middle_entry(tmp_path: Path) -> None:
    ledger, _signer = _build(tmp_path)
    lines = _lines(ledger.entries_path)
    del lines[2]
    _write_lines(ledger.entries_path, lines)
    errors = _error_codes(ledger.root)
    assert any(item.startswith("index_gap") for item in errors)
    assert any(item.startswith("prev_hash_mismatch") for item in errors)


def test_remove_last_entry_but_keep_checkpoint(tmp_path: Path) -> None:
    ledger, _signer = _build(tmp_path)
    lines = _lines(ledger.entries_path)
    _write_lines(ledger.entries_path, lines[:-1])
    errors = _error_codes(ledger.root)
    assert any(item.startswith("checkpoint_tree_size") for item in errors)


def test_remove_last_entry_and_checkpoint_needs_a_witness(tmp_path: Path) -> None:
    ledger, _signer = _build(tmp_path)
    before = verify_ledger(ledger.root)
    _write_lines(ledger.entries_path, _lines(ledger.entries_path)[:-1])
    _write_lines(ledger.checkpoints_path, _lines(ledger.checkpoints_path)[:-1])
    truncated = verify_ledger(ledger.root)
    assert truncated["valid"] is True
    assert truncated["tree_size"] == 4
    witnessed = verify_ledger(
        ledger.root,
        expect_size=before["tree_size"],
        expect_root=before["merkle_root"],
    )
    assert witnessed["valid"] is False
    assert "witness_size_mismatch" in witnessed["errors"]
    assert "witness_root_mismatch" in witnessed["errors"]


def test_swap_lines_without_rechaining(tmp_path: Path) -> None:
    ledger, _signer = _build(tmp_path)
    lines = _lines(ledger.entries_path)
    lines[1], lines[2] = lines[2], lines[1]
    _write_lines(ledger.entries_path, lines)
    errors = _error_codes(ledger.root)
    assert any(
        item.startswith("index_gap") or item.startswith("prev_hash_mismatch") for item in errors
    )
    assert verify_ledger(ledger.root)["valid"] is False


def test_reorder_and_rechain_without_new_signatures(tmp_path: Path) -> None:
    ledger, _signer = _build(tmp_path)
    objects = _objects(ledger.entries_path)
    objects[1], objects[3] = objects[3], objects[1]
    _rechain(objects)
    _dump(ledger.entries_path, objects)
    assert any(item.startswith("checkpoint_root_mismatch") for item in _error_codes(ledger.root))


def test_insert_middle_entry(tmp_path: Path) -> None:
    ledger, _signer = _build(tmp_path)
    lines = _lines(ledger.entries_path)
    lines.insert(2, lines[0])
    _write_lines(ledger.entries_path, lines)
    assert verify_ledger(ledger.root)["valid"] is False
    assert _error_codes(ledger.root)


def test_duplicate_last_line(tmp_path: Path) -> None:
    ledger, _signer = _build(tmp_path)
    lines = _lines(ledger.entries_path)
    lines.append(lines[-1])
    _write_lines(ledger.entries_path, lines)
    errors = _error_codes(ledger.root)
    assert any(item.startswith("index_gap") for item in errors)
    assert any(item.startswith("prev_hash_mismatch") for item in errors)


def test_truncate_mid_line(tmp_path: Path) -> None:
    ledger, _signer = _build(tmp_path)
    raw = ledger.entries_path.read_bytes()
    ledger.entries_path.write_bytes(raw[:-8])
    errors = _error_codes(ledger.root)
    assert "truncated_record" in errors
    assert any(item.startswith("malformed_line") for item in errors)


def test_blank_line(tmp_path: Path) -> None:
    ledger, _signer = _build(tmp_path)
    raw = ledger.entries_path.read_bytes()
    ledger.entries_path.write_bytes(raw.replace(b"\n", b"\n\n", 1))
    assert any(item.startswith("blank_line") for item in _error_codes(ledger.root))


def test_noncanonical_whitespace(tmp_path: Path) -> None:
    ledger, _signer = _build(tmp_path)
    lines = _lines(ledger.entries_path)
    lines[0] = lines[0].replace(b":", b": ", 1)
    _write_lines(ledger.entries_path, lines)
    assert any(item.startswith("noncanonical_line") for item in _error_codes(ledger.root))


def test_raw_byte_corruption(tmp_path: Path) -> None:
    ledger, _signer = _build(tmp_path)
    raw = bytearray(ledger.entries_path.read_bytes())
    raw[20] = 0xFF
    ledger.entries_path.write_bytes(bytes(raw))
    errors = _error_codes(ledger.root)
    assert any(
        item.startswith("malformed_line")
        or item == "truncated_record"
        or item.startswith("noncanonical")
        for item in errors
    )
    assert verify_ledger(ledger.root)["valid"] is False


def test_unknown_kind(tmp_path: Path) -> None:
    ledger, _signer = _build(tmp_path)
    objects = _objects(ledger.entries_path)
    objects[2]["kind"] = "live_order"
    _dump(ledger.entries_path, objects)
    assert any(item.startswith("unknown_kind") for item in _error_codes(ledger.root))


def test_flip_signature(tmp_path: Path) -> None:
    ledger, _signer = _build(tmp_path)
    objects = _objects(ledger.checkpoints_path)
    signature = str(objects[0]["signature"])
    flipped = "0" if signature[-1] != "0" else "1"
    objects[0]["signature"] = signature[:-1] + flipped
    _dump(ledger.checkpoints_path, objects)
    report = verify_ledger(ledger.root)
    assert report["valid"] is False
    assert report["fully_signed"] is False
    assert any(item.startswith("signature_invalid") for item in report["errors"])


def test_change_merkle_root_in_checkpoint(tmp_path: Path) -> None:
    ledger, _signer = _build(tmp_path)
    objects = _objects(ledger.checkpoints_path)
    root = str(objects[-1]["merkle_root"])
    objects[-1]["merkle_root"] = root[:-1] + ("0" if root[-1] != "0" else "1")
    _dump(ledger.checkpoints_path, objects)
    errors = _error_codes(ledger.root)
    assert any(item.startswith("checkpoint_root_mismatch") for item in errors)
    assert any(item.startswith("signature_invalid") for item in errors)


def test_swap_checkpoints_so_size_decreases(tmp_path: Path) -> None:
    ledger, _signer = _build(tmp_path)
    lines = _lines(ledger.checkpoints_path)
    lines[0], lines[1] = lines[1], lines[0]
    _write_lines(ledger.checkpoints_path, lines)
    errors = _error_codes(ledger.root)
    assert any(
        item.startswith("checkpoint_order") or item.startswith("checkpoint_root_mismatch")
        for item in errors
    )


def test_rekey_is_valid_until_the_public_key_is_pinned(tmp_path: Path) -> None:
    ledger, signer = _build(tmp_path)
    entries = ledger.entries_path.read_bytes()
    other = tmp_path / "rekeyed"
    other.mkdir()
    (other / "entries.jsonl").write_bytes(entries)
    rewritten = AuditLedger(other, signer=Ed25519Signer.generate(), sync=False, clock=_clock)
    rewritten.checkpoint()
    bundled = verify_ledger(other)
    assert bundled["valid"] is True
    assert bundled["trust_anchor"] == "bundled"
    pinned = verify_ledger(other, trust_public_key=signer.public_key)
    assert pinned["valid"] is False
    assert pinned["trust_anchor"] == "pinned"
    assert any(item.startswith("trust_anchor_mismatch") for item in pinned["errors"])


def test_refuse_append_onto_a_corrupt_log(tmp_path: Path) -> None:
    ledger, _signer = _build(tmp_path)
    ledger.entries_path.write_bytes(ledger.entries_path.read_bytes().replace(b"{", b" ", 1))
    with pytest.raises(AuditError):
        ledger.append("paper_decision", {"simulation_only": True})


@pytest.mark.parametrize("line", [b'{"payload":NaN}\n', b'{"payload":Infinity}\n'])
def test_nonfinite_json_corruption_returns_failed_verdict(tmp_path: Path, line: bytes) -> None:
    root = tmp_path / "malformed-ledger"
    root.mkdir()
    (root / "entries.jsonl").write_bytes(line)
    report = verify_ledger(root)
    assert report["valid"] is False
    assert "invalid_json_value:0" in report["errors"]


def test_unattested_extra_entry_field_is_rejected(tmp_path: Path) -> None:
    ledger = AuditLedger(tmp_path / "ledger", sync=False)
    ledger.append("risk_decision", {"accepted": True})
    row = json.loads(ledger.entries_path.read_text())
    row["unattested"] = "forged"
    ledger.entries_path.write_bytes(canonical_json_bytes(row) + b"\n")
    report = verify_ledger(ledger.root)
    assert report["valid"] is False
    assert "unexpected_entry_fields:0" in report["errors"]


def test_ledger_files_are_lf_only_on_every_platform(tmp_path: Path) -> None:
    """On Windows ``os.open`` without ``O_BINARY`` writes CRLF, which the
    verifier reads back as ``carriage_return`` corruption. Pin the byte
    contract: every file the ledger emits is LF-canonical."""
    signer = Ed25519Signer.generate()
    key = tmp_path / "key"
    signer.write(key)
    ledger = AuditLedger(
        tmp_path / "ledger", signer=signer, sign_every=1, clock=lambda: "2020-01-01T00:00:00Z"
    )
    ledger.append("paper_decision", {"simulation_only": True, "live_pnl_claim": False})
    ledger.checkpoint()
    for emitted in (tmp_path / "ledger").iterdir():
        if emitted.name != ".lock":
            raw = emitted.read_bytes()
            assert b"\r" not in raw, f"{emitted.name} contains a carriage return"
    for emitted in tmp_path.iterdir():
        if emitted.is_file():
            assert b"\r" not in emitted.read_bytes()
