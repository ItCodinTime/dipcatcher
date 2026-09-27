"""Unit tests for proof.verify: per-check failures (DESIGN.md §5.5, §12 W2)."""

from __future__ import annotations

import io
import json

import polars as pl
import pytest

from quant_fund.proof.bundle import sidecar_paths
from quant_fund.proof.sign import HmacSha256Signer
from quant_fund.proof.verify import VerificationResult, verify_bundle
from quant_fund.proofcore.contracts import GENESIS_HASH
from tests.unit.proof_fake_vault import mint_synthetic_bundle


def _bundle_file(bundle_dir, bundle) -> object:
    return bundle_dir / "bundles" / f"{bundle.bundle_id}.json"


def test_verify_happy_path_unsigned_nonstrict(tmp_path) -> None:
    bundle, bundle_dir = mint_synthetic_bundle(tmp_path)
    result = verify_bundle(
        _bundle_file(bundle_dir, bundle), bundle_dir=bundle_dir, strict_signature=False
    )
    assert result.ok, result.reasons
    assert result.reasons == []
    assert all(result.metrics_match.values())


def test_verify_unsigned_strict_fails(tmp_path) -> None:
    bundle, bundle_dir = mint_synthetic_bundle(tmp_path)
    result = verify_bundle(
        _bundle_file(bundle_dir, bundle), bundle_dir=bundle_dir, strict_signature=True
    )
    assert not result.ok
    assert "signature:unsigned_strict" in result.reasons


def test_verify_hmac_signed(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROOFCORE_SIGNING_KEY", "ci-secret-key")
    signer = HmacSha256Signer.from_env()
    bundle, bundle_dir = mint_synthetic_bundle(tmp_path, signer=signer)
    result = verify_bundle(_bundle_file(bundle_dir, bundle), bundle_dir=bundle_dir)
    assert result.ok, result.reasons


def test_verify_hmac_wrong_key_fails(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROOFCORE_SIGNING_KEY", "ci-secret-key")
    bundle, bundle_dir = mint_synthetic_bundle(tmp_path, signer=HmacSha256Signer.from_env())
    monkeypatch.setenv("PROOFCORE_SIGNING_KEY", "attacker-key")
    result = verify_bundle(_bundle_file(bundle_dir, bundle), bundle_dir=bundle_dir)
    assert not result.ok
    assert "signature:key_id_mismatch" in result.reasons


def test_verify_hmac_key_unavailable(tmp_path) -> None:
    bundle, bundle_dir = mint_synthetic_bundle(tmp_path, signer=HmacSha256Signer(b"k"))
    import os

    os.environ.pop("PROOFCORE_SIGNING_KEY", None)
    result = verify_bundle(_bundle_file(bundle_dir, bundle), bundle_dir=bundle_dir)
    assert not result.ok
    assert "signature:key_unavailable" in result.reasons


def test_forged_self_hash_rejected(tmp_path) -> None:
    bundle, bundle_dir = mint_synthetic_bundle(tmp_path)
    path = _bundle_file(bundle_dir, bundle)
    payload = json.loads(path.read_bytes())
    payload["seed"] = payload["seed"] + 1  # tamper a committed field
    path.write_bytes(json.dumps(payload).encode())
    result = verify_bundle(path, bundle_dir=bundle_dir, strict_signature=False)
    assert not result.ok
    assert "bundle_id:self_hash_mismatch" in result.reasons


def test_tampered_trade_log_fails(tmp_path) -> None:
    bundle, bundle_dir = mint_synthetic_bundle(tmp_path)
    trades_path = sidecar_paths(bundle_dir, bundle.bundle_id)["trade_log"]
    frame = pl.read_parquet(trades_path)
    flipped = frame.with_columns(pl.col("price") * 1.0001)
    flipped.write_parquet(trades_path)
    result = verify_bundle(
        _bundle_file(bundle_dir, bundle), bundle_dir=bundle_dir, strict_signature=False
    )
    assert not result.ok
    assert any("trade_log" in reason for reason in result.reasons)


def test_tampered_metrics_json_fails(tmp_path) -> None:
    bundle, bundle_dir = mint_synthetic_bundle(tmp_path)
    metrics_path = sidecar_paths(bundle_dir, bundle.bundle_id)["metrics"]
    payload = json.loads(metrics_path.read_bytes())
    payload["total_return"] = 999.0
    metrics_path.write_bytes(json.dumps(payload).encode())
    result = verify_bundle(
        _bundle_file(bundle_dir, bundle), bundle_dir=bundle_dir, strict_signature=False
    )
    assert not result.ok
    assert "sidecar:metrics_sha256_mismatch" in result.reasons


def test_tampered_signal_log_fails(tmp_path) -> None:
    bundle, bundle_dir = mint_synthetic_bundle(tmp_path)
    signals_path = sidecar_paths(bundle_dir, bundle.bundle_id)["signal_log"]
    frame = pl.read_parquet(signals_path)
    frame.with_columns(pl.col("weight") * 2).write_parquet(signals_path)
    result = verify_bundle(
        _bundle_file(bundle_dir, bundle), bundle_dir=bundle_dir, strict_signature=False
    )
    assert not result.ok
    assert "sidecar:signal_log_sha256_mismatch" in result.reasons


def test_tampered_config_sidecar_fails(tmp_path) -> None:
    """ADVERSARIAL R2 §2-CFG regression: the config sidecar is hash-checked —
    editing the recorded config must fail plain verify (previously skipped)."""
    bundle, bundle_dir = mint_synthetic_bundle(tmp_path)
    config_path = sidecar_paths(bundle_dir, bundle.bundle_id)["config"]
    payload = json.loads(config_path.read_bytes())
    payload["seed"] = 12345  # attacker edits the recorded config
    config_path.write_bytes(json.dumps(payload).encode())
    result = verify_bundle(
        _bundle_file(bundle_dir, bundle), bundle_dir=bundle_dir, strict_signature=False
    )
    assert not result.ok
    assert "sidecar:config_sha256_mismatch" in result.reasons


def test_pristine_config_sidecar_passes(tmp_path) -> None:
    """Control: the untouched config sidecar verifies clean."""
    bundle, bundle_dir = mint_synthetic_bundle(tmp_path)
    result = verify_bundle(
        _bundle_file(bundle_dir, bundle), bundle_dir=bundle_dir, strict_signature=False
    )
    assert result.ok, result.reasons
    assert "sidecar:config_sha256_mismatch" not in result.reasons


def test_tampered_metrics_recompute_fails(tmp_path) -> None:
    """Editing the recorded metrics_recompute breaks self-hash AND comparison."""
    bundle, bundle_dir = mint_synthetic_bundle(tmp_path)
    path = _bundle_file(bundle_dir, bundle)
    payload = json.loads(path.read_bytes())
    payload["metrics_recompute"]["total_return"] = (
        payload["metrics_recompute"]["total_return"] + 0.5
    )
    path.write_bytes(json.dumps(payload).encode())
    result = verify_bundle(path, bundle_dir=bundle_dir, strict_signature=False)
    assert not result.ok
    assert "bundle_id:self_hash_mismatch" in result.reasons
    assert result.metrics_match.get("total_return") is False


def test_tampered_data_manifest_fails(tmp_path) -> None:
    bundle, bundle_dir = mint_synthetic_bundle(tmp_path)
    path = _bundle_file(bundle_dir, bundle)
    payload = json.loads(path.read_bytes())
    payload["data_manifest"]["reads"][0]["rows"] += 1
    path.write_bytes(json.dumps(payload).encode())
    result = verify_bundle(path, bundle_dir=bundle_dir, strict_signature=False)
    assert not result.ok
    assert "data_manifest:merkle_root_mismatch" in result.reasons


def test_missing_sidecar_fails(tmp_path) -> None:
    bundle, bundle_dir = mint_synthetic_bundle(tmp_path)
    sidecar_paths(bundle_dir, bundle.bundle_id)["trade_log"].unlink()
    result = verify_bundle(
        _bundle_file(bundle_dir, bundle), bundle_dir=bundle_dir, strict_signature=False
    )
    assert not result.ok
    assert "sidecar:trade_log:missing" in result.reasons


def test_trusted_head_ok_and_mismatch(tmp_path) -> None:
    bundle1, bundle_dir = mint_synthetic_bundle(tmp_path / "r")
    bundle2, _ = mint_synthetic_bundle(tmp_path / "r", seed=99)
    head = bundle2.bundle_id
    ok = verify_bundle(
        _bundle_file(bundle_dir, bundle2),
        bundle_dir=bundle_dir,
        strict_signature=False,
        trusted_head=head,
    )
    assert ok.ok, ok.reasons
    bad = verify_bundle(
        _bundle_file(bundle_dir, bundle2),
        bundle_dir=bundle_dir,
        strict_signature=False,
        trusted_head=GENESIS_HASH,
    )
    assert not bad.ok
    assert "chain:trusted_head_mismatch" in bad.reasons
    # non-head bundle still verifies against the same trusted head
    mid = verify_bundle(
        _bundle_file(bundle_dir, bundle1),
        bundle_dir=bundle_dir,
        strict_signature=False,
        trusted_head=head,
    )
    assert mid.ok, mid.reasons


def test_chain_line_removal_detected(tmp_path) -> None:
    bundle1, bundle_dir = mint_synthetic_bundle(tmp_path / "r")
    bundle2, _ = mint_synthetic_bundle(tmp_path / "r", seed=99)
    chain_path = bundle_dir / "bundles.jsonl"
    lines = [line for line in chain_path.read_text().splitlines() if line.strip()]
    chain_path.write_text(lines[1] + "\n")  # drop genesis link
    result = verify_bundle(
        _bundle_file(bundle_dir, bundle2), bundle_dir=bundle_dir, strict_signature=False
    )
    assert not result.ok
    assert any(reason.startswith("chain:") for reason in result.reasons)


def test_malformed_bundle_file_never_raises(tmp_path) -> None:
    path = tmp_path / "garbage.json"
    path.write_bytes(b"not json at all")
    result = verify_bundle(path, bundle_dir=tmp_path)
    assert isinstance(result, VerificationResult)
    assert not result.ok
    assert result.reasons[0].startswith("schema:unreadable")
    path.write_bytes(b'{"schema_version": "proofcore/0"}')
    result = verify_bundle(path, bundle_dir=tmp_path)
    assert not result.ok


def test_env_mismatch_is_warning_not_failure(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle, bundle_dir = mint_synthetic_bundle(tmp_path)

    def _shifted_env():
        from quant_fund.proof.bundle import env_fingerprint

        env = env_fingerprint().model_dump(mode="json")
        env["machine"] = "definitely-different-machine"
        from quant_fund.proofcore.contracts import EnvFingerprint

        return EnvFingerprint(**env)

    monkeypatch.setattr("quant_fund.proof.verify.env_fingerprint", _shifted_env)
    result = verify_bundle(
        _bundle_file(bundle_dir, bundle), bundle_dir=bundle_dir, strict_signature=False
    )
    assert result.env_mismatch
    assert result.ok, result.reasons  # A3 F5.2: non-fatal


def test_metric_recompute_agreement(tmp_path) -> None:
    """The verifier's independent recompute matches mint-time values (A1 F2)."""
    bundle, bundle_dir = mint_synthetic_bundle(tmp_path)
    trades_path = sidecar_paths(bundle_dir, bundle.bundle_id)["trade_log"]
    from quant_fund.proof.bundle import recompute_headline_metrics

    recomputed = recompute_headline_metrics(pl.read_parquet(io.BytesIO(trades_path.read_bytes())))
    for key, value in recomputed.items():
        recorded = bundle.metrics_recompute[key]
        assert value == pytest.approx(recorded, rel=1e-9, abs=1e-12, nan_ok=True)
