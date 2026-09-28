"""Checkpoint failure-matrix coverage for audit.verify."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_fund.audit.canonical import canonical_json_bytes, sha256_hex
from quant_fund.audit.ledger import SCHEMA_VERSION, AuditLedger
from quant_fund.audit.merkle import merkle_root
from quant_fund.audit.signing import Ed25519Signer, key_id_for
from quant_fund.audit.verify import explain, verify_ledger

_STAMP = "2025-01-01T00:00:00+00:00"


def _clock() -> str:
    return _STAMP


def _build(root: Path, *, sign_every: int = 1) -> tuple[AuditLedger, Ed25519Signer]:
    signer = Ed25519Signer.generate()
    ledger = AuditLedger(
        root,
        signer=signer,
        sign_every=sign_every,
        sync=False,
        clock=_clock,
    )
    for i in range(3):
        ledger.append(
            "paper_decision",
            {"seq": i, "simulation_only": True, "live_pnl_claim": False},
        )
    return ledger, signer


def _checkpoints(root: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (root / "checkpoints.jsonl").read_text().splitlines()
        if line.strip()
    ]


def _rewrite_checkpoint(root: Path, position: int, **over: object) -> None:
    """Rewrite one checkpoint line with patched fields."""
    lines = (root / "checkpoints.jsonl").read_text().splitlines()
    objs = [json.loads(line) for line in lines]
    objs[position].update(over)
    (root / "checkpoints.jsonl").write_text(
        "".join(canonical_json_bytes(obj).decode() + "\n" for obj in objs),
        encoding="utf-8",
    )


def _errors(report: dict) -> str:
    return ",".join(report["errors"])


class TestBaseline:
    def test_signed_ledger_valid(self, tmp_path: Path) -> None:
        ledger, signer = _build(tmp_path / "ledger")
        report = verify_ledger(ledger.root, trust_public_key=signer.public_key)
        assert report["valid"] is True
        assert report["fully_signed"] is True
        assert report["tree_size"] == 3
        assert report["signed_tree_size"] == 3
        assert report["trust_anchor"] == "pinned"
        assert report["checkpoints"] == 3
        assert json.loads(explain(report))["valid"] is True

    def test_unsigned_ledger(self, tmp_path: Path) -> None:
        ledger = AuditLedger(
            tmp_path / "ledger", signer=None, sign_every=1, sync=False, clock=_clock
        )
        ledger.append("paper_decision", {"simulation_only": True, "live_pnl_claim": False})
        report = verify_ledger(ledger.root)
        assert report["valid"] is False
        assert "unsigned_log" in report["errors"]

    def test_missing_and_empty_dirs(self, tmp_path: Path) -> None:
        missing = verify_ledger(tmp_path / "nope")
        assert missing["errors"] == ["ledger_missing"]
        assert missing["trust_anchor"] == "none"
        empty = tmp_path / "empty"
        empty.mkdir()
        report = verify_ledger(empty)
        assert report["valid"] is True
        assert report["tree_size"] == 0

    def test_witness_mismatches(self, tmp_path: Path) -> None:
        ledger, _ = _build(tmp_path / "ledger")
        report = verify_ledger(ledger.root, expect_size=9)
        assert "witness_size_mismatch" in report["errors"]
        report = verify_ledger(ledger.root, expect_root="0" * 64)
        assert "witness_root_mismatch" in report["errors"]
        good = verify_ledger(ledger.root)
        report = verify_ledger(ledger.root, expect_size=3, expect_root=good["merkle_root"])
        assert "witness_size_mismatch" not in report["errors"]
        assert "witness_root_mismatch" not in report["errors"]


class TestCheckpointFieldMatrix:
    @pytest.mark.parametrize(
        ("patch", "expected"),
        [
            ({"v": 99}, "checkpoint_bad_version"),
            ({"scheme": "bogus"}, "checkpoint_unknown_scheme"),
            ({"key_id": ""}, "checkpoint_bad_key_id"),
            ({"tree_size": True}, "checkpoint_bad_tree_size"),
            ({"tree_size": 0}, "checkpoint_bad_tree_size"),
            ({"tree_size": 99}, "checkpoint_tree_size"),
            ({"merkle_root": None}, "checkpoint_bad_root"),
            ({"timestamp_utc": ""}, "checkpoint_bad_timestamp"),
            ({"merkle_root": "ab" * 32}, "checkpoint_root_mismatch"),
            ({"extra_field": 1}, "unexpected_checkpoint_field"),
        ],
    )
    def test_each_refusal(self, tmp_path: Path, patch: dict, expected: str) -> None:
        ledger, _ = _build(tmp_path / "ledger")
        _rewrite_checkpoint(ledger.root, 0, **patch)
        errors = _errors(verify_ledger(ledger.root))
        assert expected in errors
        assert verify_ledger(ledger.root)["valid"] is False

    def test_checkpoint_order_violation(self, tmp_path: Path) -> None:
        ledger, _ = _build(tmp_path / "ledger")
        # Last checkpoint claims a smaller tree than the first.
        _rewrite_checkpoint(ledger.root, 2, tree_size=1)
        errors = _errors(verify_ledger(ledger.root))
        assert "checkpoint_root_mismatch" in errors
        assert "checkpoint_order" in errors

    def test_missing_public_key(self, tmp_path: Path) -> None:
        ledger, _ = _build(tmp_path / "ledger")
        _rewrite_checkpoint(ledger.root, 0, public_key=None)
        assert "missing_public_key" in _errors(verify_ledger(ledger.root))
        _rewrite_checkpoint(ledger.root, 0, public_key="not-hex")
        assert "missing_public_key" in _errors(verify_ledger(ledger.root))

    def test_trust_anchor_mismatch(self, tmp_path: Path) -> None:
        ledger, _ = _build(tmp_path / "ledger")
        wrong = Ed25519Signer.generate()
        errors = _errors(verify_ledger(ledger.root, trust_public_key=wrong.public_key))
        assert "trust_anchor_mismatch" in errors
        assert "key_id_mismatch" in errors
        assert "signature_invalid" in errors

    def test_bad_signature(self, tmp_path: Path) -> None:
        ledger, _ = _build(tmp_path / "ledger")
        _rewrite_checkpoint(ledger.root, 0, signature="ab" * 64)
        errors = _errors(verify_ledger(ledger.root))
        assert "signature_invalid" in errors

    def test_non_str_signature(self, tmp_path: Path) -> None:
        ledger, _ = _build(tmp_path / "ledger")
        _rewrite_checkpoint(ledger.root, 0, signature=None)
        assert "signature_invalid" in _errors(verify_ledger(ledger.root))


class TestSigstorePath:
    def _sigstore_checkpoint(self, root: Path) -> dict:
        return {
            "v": SCHEMA_VERSION,
            "scheme": "sigstore",
            "key_id": "sigstore",
            "tree_size": 3,
            "merkle_root": "ab" * 32,
            "timestamp_utc": _STAMP,
            "signature": "x",
            "public_key": None,
            "sigstore_bundle_json": None,
            "sigstore_identity": "a@b.c",
            "sigstore_issuer": "issuer",
        }

    def _write(self, root: Path, checkpoint: dict) -> None:
        (root / "checkpoints.jsonl").write_text(
            canonical_json_bytes(checkpoint).decode() + "\n", encoding="utf-8"
        )

    def test_bundle_missing(self, tmp_path: Path) -> None:
        ledger, _ = _build(tmp_path / "ledger")
        checkpoint = self._sigstore_checkpoint(ledger.root)
        checkpoint["merkle_root"] = _checkpoints(ledger.root)[-1]["merkle_root"]
        self._write(ledger.root, checkpoint)
        assert "sigstore_bundle_missing" in _errors(verify_ledger(ledger.root))

    def test_bundle_hash_mismatch(self, tmp_path: Path) -> None:
        ledger, _ = _build(tmp_path / "ledger")
        checkpoint = self._sigstore_checkpoint(ledger.root)
        checkpoint["merkle_root"] = _checkpoints(ledger.root)[-1]["merkle_root"]
        checkpoint["sigstore_bundle_json"] = '{"fake": true}'
        self._write(ledger.root, checkpoint)
        assert "sigstore_bundle_hash_mismatch" in _errors(verify_ledger(ledger.root))

    def test_unpinned_and_unavailable(self, tmp_path: Path) -> None:
        ledger, _ = _build(tmp_path / "ledger")
        checkpoint = self._sigstore_checkpoint(ledger.root)
        checkpoint["merkle_root"] = _checkpoints(ledger.root)[-1]["merkle_root"]
        bundle = '{"fake": true}'
        checkpoint["sigstore_bundle_json"] = bundle
        checkpoint["signature"] = sha256_hex(bundle.encode("utf-8"))
        self._write(ledger.root, checkpoint)
        # identity=None param + recorded identity present -> pinned to recorded;
        # sigstore package absent -> unavailable.
        errors = _errors(verify_ledger(ledger.root))
        assert "sigstore_unavailable" in errors
        # Explicit identity=None with no recorded identity -> unpinned.
        checkpoint["sigstore_identity"] = None
        self._write(ledger.root, checkpoint)
        errors = _errors(verify_ledger(ledger.root))
        assert "sigstore_identity_unpinned" in errors
        assert "sigstore_unavailable" in errors

    def test_sigstore_invalid_signature_via_audit_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from quant_fund.audit.errors import AuditError

        ledger, _ = _build(tmp_path / "ledger")
        checkpoint = self._sigstore_checkpoint(ledger.root)
        checkpoint["merkle_root"] = _checkpoints(ledger.root)[-1]["merkle_root"]
        bundle = '{"fake": true}'
        checkpoint["sigstore_bundle_json"] = bundle
        checkpoint["signature"] = sha256_hex(bundle.encode("utf-8"))
        self._write(ledger.root, checkpoint)

        def boom(*_a: object, **_k: object) -> None:
            raise AuditError("bundle rejected")

        monkeypatch.setattr("quant_fund.audit.verify.verify_sigstore_bundle", boom)
        errors = _errors(verify_ledger(ledger.root))
        assert "signature_invalid" in errors


class TestConsistencyProof:
    def test_consistency_checked_across_checkpoints(self, tmp_path: Path) -> None:
        ledger, _ = _build(tmp_path / "ledger")
        # A real growing log verifies consistency between consecutive heads.
        report = verify_ledger(ledger.root)
        assert report["valid"] is True
        assert report["checkpoints"] == 3

    def test_consistency_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ledger, _ = _build(tmp_path / "ledger")
        import quant_fund.audit.verify as v

        monkeypatch.setattr(v, "verify_consistency", lambda *a: False)
        errors = _errors(verify_ledger(ledger.root))
        assert "consistency_failed" in errors

    def test_consistency_raises_counts_as_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ledger, _ = _build(tmp_path / "ledger")
        import quant_fund.audit.verify as v

        def boom(*_a: object) -> list[bytes]:
            raise ValueError("nope")

        monkeypatch.setattr(v, "consistency_proof", boom)
        errors = _errors(verify_ledger(ledger.root))
        assert "consistency_failed" in errors


class TestSelfCheckAndMisc:
    def test_inclusion_self_check_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ledger, _ = _build(tmp_path / "ledger")
        import quant_fund.audit.verify as v

        monkeypatch.setattr(v, "verify_inclusion", lambda *a: False)
        errors = _errors(verify_ledger(ledger.root))
        assert "inclusion_self_check_failed" in errors

    def test_safe_inclusion_none(self) -> None:
        from quant_fund.audit.verify import _safe_inclusion

        # inclusion_proof on index out of range raises AuditError -> None.
        assert _safe_inclusion([b"a"], 5) is None

    def test_signed_tree_head_bytes(self, tmp_path: Path) -> None:
        from quant_fund.audit.verify import signed_tree_head_bytes

        ledger, _ = _build(tmp_path / "ledger")
        checkpoint = _checkpoints(ledger.root)[-1]
        payload = signed_tree_head_bytes(checkpoint)
        assert isinstance(payload, bytes)
        # Missing a signed field -> raises.
        bad = dict(checkpoint)
        bad.pop("tree_size")
        with pytest.raises((KeyError, Exception)):
            signed_tree_head_bytes(bad)


class TestPartialCoverage:
    def test_partially_signed_ledger(self, tmp_path: Path) -> None:
        signer = Ed25519Signer.generate()
        ledger = AuditLedger(
            tmp_path / "ledger",
            signer=signer,
            sign_every=2,
            sync=False,
            clock=_clock,
        )
        for i in range(5):
            ledger.append(
                "paper_decision",
                {"seq": i, "simulation_only": True, "live_pnl_claim": False},
            )
        report = verify_ledger(ledger.root)
        assert report["valid"] is True
        assert report["signed_tree_size"] == 4
        assert report["unsigned_suffix"] == 1
        assert report["fully_signed"] is False

    def test_root_computed_over_all_preimages(self, tmp_path: Path) -> None:
        ledger, _ = _build(tmp_path / "ledger")
        preimages = [e.preimage() for e in ledger.entries()]
        expected = merkle_root(preimages).hex()
        assert verify_ledger(ledger.root)["merkle_root"] == expected
        assert key_id_for(b"") != ""
