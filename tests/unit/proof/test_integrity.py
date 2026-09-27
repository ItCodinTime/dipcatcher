"""Integrity cases for proof hashing, digests, signing, and the public surface.

SYNTHETIC correctness checks only: no market evidence.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from quant_fund.proof.cli import proof_app
from quant_fund.proof.recorder import InMemoryRecorder, make_read_record, read_leaf_hash
from quant_fund.proof.sign import SIGNING_KEY_ENV, HmacSha256Signer, NullSigner
from quant_fund.proofcore.contracts import (
    CodeFingerprint,
    DataAccessRecord,
    DataManifestSummary,
    LeakageFinding,
    LeakageReport,
    PitManifest,
    PitManifestFile,
    ProofBundleV1,
    ProofError,
    RealityReport,
    TrialLedgerRow,
    canonical_json_bytes,
    merkle_root_hex,
    sha256_hex_bytes,
)

HEX = "ab" * 32
BAD = "g" * 64
UPPER = "AB" * 32

_UNIMPLEMENTED = (
    "ASOF_SENTINEL",
    "VerificationResult",
    "build_bundle",
    "chain_head",
    "recompute_headline_metrics",
    "replay_bundle",
    "run_backtest_proven",
    "verify_bundle",
)


def _leaf(digest: str) -> str:
    return hashlib.sha256(b"PC:leaf:" + bytes.fromhex(digest)).hexdigest()


def _node(left: str, right: str) -> str:
    return hashlib.sha256(b"PC:node:" + bytes.fromhex(left) + bytes.fromhex(right)).hexdigest()


def _buggy_raw_node_root(leaf_hashes: list[str]) -> str:
    """Pre-fix tree: raw digests enter ``PC:node:`` with no ``PC:leaf:`` step."""
    level = sorted(leaf_hashes)
    while len(level) > 1:
        nxt: list[str] = []
        for i in range(0, len(level), 2):
            left = bytes.fromhex(level[i])
            right = bytes.fromhex(level[min(i + 1, len(level) - 1)])
            nxt.append(hashlib.sha256(b"PC:node:" + left + right).hexdigest())
        level = nxt
    return level[0]


def test_merkle_empty_hashes_empty_bytes() -> None:
    assert merkle_root_hex([]) == sha256_hex_bytes(b"")


def test_merkle_single_leaf_is_domain_separated() -> None:
    assert merkle_root_hex([HEX]) == _leaf(HEX)


def test_merkle_sorts_before_leaf_domain_separation() -> None:
    low = "11" * 32
    high = "ff" * 32
    paired = _node(_leaf(low), _leaf(high))
    assert merkle_root_hex([high, low]) == paired
    assert merkle_root_hex([low, high]) == paired


def test_merkle_odd_level_duplicates_last_node() -> None:
    leaves = ["01" * 32, "02" * 32, "03" * 32]
    ordered = sorted(leaves)
    left = _node(_leaf(ordered[0]), _leaf(ordered[1]))
    right = _node(_leaf(ordered[2]), _leaf(ordered[2]))
    assert merkle_root_hex(leaves) == _node(left, right)


def test_merkle_multi_leaf_is_not_the_raw_node_tree() -> None:
    leaves = ["11" * 32, "22" * 32]
    assert merkle_root_hex(leaves) != _buggy_raw_node_root(leaves)


@pytest.mark.parametrize(
    "leaf",
    [BAD, UPPER, "ab" * 31, "", "abc", "ab" * 32 + "zz"],
)
def test_merkle_malformed_leaf_is_proof_error(leaf: str) -> None:
    with pytest.raises(ProofError, match="sha256 hex digest"):
        merkle_root_hex([leaf])


def test_merkle_non_string_leaf_is_proof_error() -> None:
    with pytest.raises(ProofError):
        merkle_root_hex([123])  # type: ignore[list-item]


def test_canonical_json_nonfinite_is_null() -> None:
    assert canonical_json_bytes(float("nan")) == canonical_json_bytes(None)
    assert canonical_json_bytes(float("inf")) == canonical_json_bytes(None)
    encoded = canonical_json_bytes({"z": 1, "a": float("-inf"), "b": [float("nan"), 1.5]})
    assert json.loads(encoded.decode("utf-8")) == {"a": None, "b": [None, 1.5], "z": 1}
    assert b"NaN" not in encoded
    assert b"Infinity" not in encoded


def test_canonical_json_finite_payload_unchanged() -> None:
    assert canonical_json_bytes({"z": 1, "a": 2}) == b'{"a":2,"z":1}'


def test_code_fingerprint_rejects_non_hex_digest() -> None:
    with pytest.raises(ValidationError):
        CodeFingerprint(git_revision="abc1234", worktree_sha256=BAD, dirty=False)
    with pytest.raises(ValidationError):
        CodeFingerprint(git_revision="abc1234", worktree_sha256=UPPER, dirty=False)
    ok = CodeFingerprint(git_revision="abc1234", worktree_sha256=HEX, dirty=False)
    assert ok.worktree_sha256 == HEX


def test_data_access_record_rejects_non_utc_and_non_hex() -> None:
    with pytest.raises(ValidationError):
        DataAccessRecord(
            dataset="silver/bars",
            asof_utc="2024-06-01T05:30:00+05:30",
            rows=1,
            content_sha256=HEX,
        )
    with pytest.raises(ValidationError):
        DataAccessRecord(
            dataset="silver/bars",
            asof_utc="2024-06-01T00:00:00",
            rows=1,
            content_sha256=HEX,
        )
    with pytest.raises(ValidationError):
        DataAccessRecord(
            dataset="silver/bars",
            asof_utc="2024-06-01T00:00:00+00:00",
            rows=1,
            content_sha256=BAD,
        )
    utc_z = DataAccessRecord(
        dataset="silver/bars",
        asof_utc="2024-06-01T00:00:00Z",
        rows=1,
        content_sha256=HEX,
    )
    utc_offset = DataAccessRecord(
        dataset="silver/bars",
        asof_utc="2024-06-01T00:00:00+00:00",
        rows=1,
        content_sha256=HEX,
    )
    assert utc_z == utc_offset
    assert utc_z.asof_utc == "2024-06-01T00:00:00+00:00"


def _bundle_kwargs() -> dict[str, object]:
    return {
        "bundle_id": HEX,
        "created_utc": "2024-01-01T00:00:00+00:00",
        "run_kind": "backtest",
        "code": {"git_revision": "abc1234", "worktree_sha256": HEX, "dirty": False},
        "data_manifest": {"reads": [], "merkle_root": HEX, "n_reads": 0},
        "config_sha256": HEX,
        "seed": 1,
        "env": {
            "python_version": "3.12.0",
            "python_implementation": "CPython",
            "platform": "linux",
            "machine": "x86_64",
            "byteorder": "little",
            "packages": {"dipcatcher": "0.0.0"},
        },
        "signal_log_sha256": HEX,
        "trade_log_sha256": HEX,
        "metrics_sha256": HEX,
        "metrics_recompute": {"pinball": 0.1},
        "prev_bundle_hash": HEX,
        "signature": {"scheme": "none", "key_id": "unsigned", "value": ""},
    }


@pytest.mark.parametrize(
    "field",
    [
        "bundle_id",
        "config_sha256",
        "signal_log_sha256",
        "trade_log_sha256",
        "metrics_sha256",
        "prev_bundle_hash",
    ],
)
def test_bundle_digest_fields_reject_non_hex(field: str) -> None:
    ProofBundleV1(**_bundle_kwargs())
    bad = _bundle_kwargs()
    bad[field] = BAD
    with pytest.raises(ValidationError):
        ProofBundleV1(**bad)
    upper = _bundle_kwargs()
    upper[field] = UPPER
    with pytest.raises(ValidationError):
        ProofBundleV1(**upper)


def test_manifest_merkle_root_rejects_non_hex() -> None:
    with pytest.raises(ValidationError):
        DataManifestSummary(reads=[], merkle_root=BAD, n_reads=0)


def test_chain_and_trial_ids_reject_non_hex() -> None:
    with pytest.raises(ValidationError):
        PitManifestFile(
            path="part.parquet",
            sha256=BAD,
            rows=0,
            min_known_at="2024-01-01T00:00:00+00:00",
            max_known_at="2024-01-01T00:00:00+00:00",
            min_event_time="2024-01-01T00:00:00+00:00",
            max_event_time="2024-01-01T00:00:00+00:00",
        )
    with pytest.raises(ValidationError):
        PitManifest(
            dataset="silver/bars",
            created_utc="2024-01-01T00:00:00+00:00",
            revision=0,
            prev_manifest_sha256=BAD,
            files=[],
        )
    with pytest.raises(ValidationError):
        TrialLedgerRow(
            trial_id=BAD,
            bundle_hash=HEX,
            family="calibration",
            strategy="synth",
            cluster_id="c",
            created_utc="2024-01-01T00:00:00+00:00",
            n_obs=1,
            periods_per_year=252.0,
            sharpe_periodic=0.0,
            skew=0.0,
            kurtosis_raw=3.0,
            returns_sha256=HEX,
        )
    finding = LeakageFinding(
        rule_id="LH001",
        severity="warning",
        path="a.py",
        line=1,
        col=0,
        message="synthetic",
    )
    with pytest.raises(ValidationError):
        LeakageReport(
            created_utc="2024-01-01T00:00:00+00:00",
            scanned_files=1,
            findings=[finding],
            errors=0,
            warnings=1,
            report_sha256=UPPER,
        )
    with pytest.raises(ValidationError):
        RealityReport(
            created_utc="2024-01-01T00:00:00+00:00",
            n_trials=1,
            n_effective_trials=1.0,
            best_trial_id=BAD,
            psr=0.5,
            min_trl_periods=10.0,
            dsr=0.0,
            pbo=0.5,
            spa_pvalue=0.5,
            fdr_q=0.1,
            verdict="insufficient_evidence",
            report_sha256=HEX,
        )
    with pytest.raises(ValidationError):
        RealityReport(
            created_utc="2024-01-01T00:00:00+00:00",
            n_trials=1,
            n_effective_trials=1.0,
            best_trial_id=HEX,
            psr=0.5,
            min_trl_periods=10.0,
            dsr=0.0,
            pbo=0.5,
            spa_pvalue=0.5,
            bh_fdr_rejects=[BAD],
            fdr_q=0.1,
            verdict="insufficient_evidence",
            report_sha256=HEX,
        )


def test_make_read_record_normalizes_aware_asof_to_utc() -> None:
    east = datetime(2024, 6, 1, 5, 30, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    utc = datetime(2024, 6, 1, 0, 0, tzinfo=UTC)
    params = {"limit": 3}
    left = make_read_record("silver/bars", east, params=params, rows=2, content_sha256=HEX)
    right = make_read_record("silver/bars", utc, params=params, rows=2, content_sha256=HEX)
    assert left.asof_utc == "2024-06-01T00:00:00+00:00"
    assert left == right
    assert left.params == {"limit": "3"}


def test_make_read_record_rejects_naive_asof() -> None:
    with pytest.raises(ProofError, match="timezone-aware"):
        make_read_record(
            "silver/bars",
            datetime(2024, 6, 1),
            rows=0,
            content_sha256=HEX,
        )


def test_recorder_manifest_uses_domain_separated_merkle() -> None:
    recorder = InMemoryRecorder()
    asof = datetime(2024, 6, 1, tzinfo=UTC)
    recorder.record(make_read_record("silver/bars", asof, rows=1, content_sha256=HEX))
    other = "cd" * 32
    recorder.record(make_read_record("silver/quotes", asof, rows=2, content_sha256=other))
    summary = recorder.manifest_summary()
    assert summary.n_reads == 2
    assert summary.merkle_root == merkle_root_hex([read_leaf_hash(read) for read in summary.reads])


def test_hmac_missing_and_empty_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SIGNING_KEY_ENV, raising=False)
    with pytest.raises(ProofError):
        HmacSha256Signer.from_env()
    monkeypatch.setenv(SIGNING_KEY_ENV, "")
    with pytest.raises(ProofError):
        HmacSha256Signer.from_env()
    with pytest.raises(ProofError):
        HmacSha256Signer(b"")


def test_hmac_hex_key_and_raw_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SIGNING_KEY_ENV, "aabb")
    hex_signer = HmacSha256Signer.from_env()
    assert hex_signer.key_id == hashlib.sha256(bytes.fromhex("aabb")).hexdigest()[:16]
    monkeypatch.setenv(SIGNING_KEY_ENV, "not-hex")
    raw_signer = HmacSha256Signer.from_env()
    assert raw_signer.key_id == hashlib.sha256(b"not-hex").hexdigest()[:16]
    assert hex_signer.sign(b"payload") != raw_signer.sign(b"payload")
    assert len(hex_signer.key_id) == 16
    assert hex_signer.scheme == "hmac-sha256"


def test_hmac_detects_payload_and_signature_tampering() -> None:
    signer = HmacSha256Signer(b"ci-key")
    payload = b'{"bundle_id":"abc"}'
    signature = signer.sign(payload)
    assert len(signature) == 64
    assert signer.verify(payload, signature)
    assert not signer.verify(payload + b" ", signature)
    flipped = signature[:-1] + ("0" if signature[-1] != "0" else "1")
    assert not signer.verify(payload, flipped)
    assert not signer.verify(payload, "")
    assert not signer.verify(payload, signature.upper())
    assert not signer.verify(payload, signature[:-2])


def test_null_signer_is_unsigned() -> None:
    signer = NullSigner()
    assert signer.scheme == "none"
    assert signer.key_id == "unsigned"
    assert signer.sign(b"anything") == ""


def test_proof_star_import_only_loads_implemented_surface() -> None:
    proof = importlib.import_module("quant_fund.proof")
    namespace: dict[str, object] = {}
    exec("from quant_fund.proof import *", namespace, namespace)
    exported = set(proof.__all__)
    assert exported.isdisjoint(_UNIMPLEMENTED)
    for name in proof.__all__:
        assert getattr(proof, name) is namespace[name]
    for name in _UNIMPLEMENTED:
        assert name not in dir(proof)
        with pytest.raises(AttributeError):
            getattr(proof, name)


def test_proof_cli_does_not_expose_unimplemented_commands() -> None:
    names = {command.name for command in proof_app.registered_commands}
    assert names.isdisjoint({"run", "verify", "chain-head"})
    runner = CliRunner()
    help_result = runner.invoke(proof_app, ["--help"])
    assert help_result.exit_code == 0
    assert help_result.exception is None
    for name in ("run", "verify", "chain-head"):
        result = runner.invoke(proof_app, [name])
        assert result.exit_code != 0
        assert not isinstance(result.exception, ModuleNotFoundError)
        combined = f"{result.output}{result.exc_info}"
        assert "ModuleNotFoundError" not in combined
