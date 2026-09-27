"""Unit tests for proof.bundle: build, self-hash, chain (DESIGN.md §5.3)."""

from __future__ import annotations

import json

import polars as pl
import pytest

from quant_fund.proof.bundle import (
    METRIC_KEYS,
    canonical_bundle_bytes,
    chain_head,
    compute_bundle_id,
    load_chain,
    recompute_headline_metrics,
    round_floats,
    sidecar_paths,
    signing_payload_bytes,
    verify_self_hash,
)
from quant_fund.proof.recorder import InMemoryRecorder, make_read_record
from quant_fund.proof.sign import HmacSha256Signer, NullSigner
from quant_fund.proofcore.contracts import (
    GENESIS_HASH,
    ProofBundleV1,
    sha256_hex_bytes,
)
from tests.unit.proof_fake_vault import T0, mint_synthetic_bundle


def test_build_bundle_fields_and_sidecars(tmp_path) -> None:
    bundle, bundle_dir = mint_synthetic_bundle(tmp_path)
    assert bundle.schema_version == "proofcore/1"
    assert bundle.run_kind == "backtest"
    assert bundle.seed == 42
    assert bundle.prev_bundle_hash == GENESIS_HASH
    assert bundle.signature.scheme == "none"
    assert set(bundle.metrics_recompute) == set(METRIC_KEYS)
    assert bundle.data_manifest.n_reads == 2  # bars + weights panel reads
    sidecars = sidecar_paths(bundle_dir, bundle.bundle_id)
    for path in sidecars.values():
        assert path.exists(), path
    assert (bundle_dir / "bundles" / f"{bundle.bundle_id}.json").exists()
    assert sha256_hex_bytes(sidecars["trade_log"].read_bytes()) == bundle.trade_log_sha256
    assert sha256_hex_bytes(sidecars["signal_log"].read_bytes()) == bundle.signal_log_sha256
    assert sha256_hex_bytes(sidecars["metrics"].read_bytes()) == bundle.metrics_sha256


def test_self_hash_and_signing_payload(tmp_path) -> None:
    bundle, bundle_dir = mint_synthetic_bundle(tmp_path)
    payload = json.loads((bundle_dir / "bundles" / f"{bundle.bundle_id}.json").read_bytes())
    assert verify_self_hash(payload)
    assert compute_bundle_id(payload) == bundle.bundle_id
    # created_utc is evidence, not identity: changing it must NOT change the id.
    mutated = dict(payload, created_utc="1999-01-01T00:00:00+00:00")
    assert compute_bundle_id(mutated) == bundle.bundle_id
    # any committed field IS identity:
    mutated = dict(payload, seed=payload["seed"] + 1)
    assert compute_bundle_id(mutated) != bundle.bundle_id
    # signing payload drops only the signature field
    unsigned = json.loads(signing_payload_bytes(payload))
    assert "signature" not in unsigned
    assert unsigned["bundle_id"] == bundle.bundle_id


def test_chain_links_across_bundles(tmp_path) -> None:
    bundle1, bundle_dir = mint_synthetic_bundle(tmp_path / "a")
    bundle2, _ = mint_synthetic_bundle(tmp_path / "a", seed=43)
    # second run wrote into the same bundle_dir via tmp_path / "a"
    assert bundle2.prev_bundle_hash == bundle1.bundle_id
    assert chain_head(bundle_dir) == bundle2.bundle_id
    chain = load_chain(bundle_dir)
    assert [b.bundle_id for b in chain] == [bundle1.bundle_id, bundle2.bundle_id]


def test_chain_head_empty_is_genesis(tmp_path) -> None:
    assert chain_head(tmp_path) == GENESIS_HASH
    assert load_chain(tmp_path) == []


def test_hmac_signed_bundle(tmp_path) -> None:
    signer = HmacSha256Signer(b"ci-key")
    bundle, bundle_dir = mint_synthetic_bundle(tmp_path, signer=signer)
    assert bundle.signature.scheme == "hmac-sha256"
    assert bundle.signature.key_id == signer.key_id
    payload = json.loads((bundle_dir / "bundles" / f"{bundle.bundle_id}.json").read_bytes())
    assert signer.verify(signing_payload_bytes(payload), bundle.signature.value)


def test_bundle_bytes_canonical_roundtrip(tmp_path) -> None:
    bundle, _ = mint_synthetic_bundle(tmp_path)
    data = canonical_bundle_bytes(bundle)
    assert json.loads(data) == json.loads(canonical_json_bytes_roundtrip(bundle))


def canonical_json_bytes_roundtrip(bundle: ProofBundleV1) -> bytes:
    from quant_fund.proofcore.contracts import canonical_json_bytes

    return canonical_json_bytes(bundle.model_dump(mode="json"))


def test_recompute_headline_metrics_synthetic() -> None:
    trade_log = pl.DataFrame(
        {
            "fill_time": [T0, T0, T0.replace(day=2), T0.replace(day=3)],
            "security_id": ["A", "B", "A", "B"],
            "nav": [1_000_000.0, 1_000_000.0, 1_010_000.0, 990_000.0],
        }
    ).with_columns(pl.col("fill_time").cast(pl.Datetime("us", "UTC")))
    metrics = recompute_headline_metrics(trade_log)
    assert metrics["n_trades"] == 4.0
    assert set(metrics) == set(METRIC_KEYS)
    # nav path 1.00 -> 1.01 -> 0.99 (per-fill-time dedupe): product = 0.99
    assert metrics["total_return"] == pytest.approx(-0.01, rel=1e-9)


def test_recompute_headline_metrics_empty_fail_closed() -> None:
    empty = pl.DataFrame(schema={"fill_time": pl.Datetime("us", "UTC"), "nav": pl.Float64})
    metrics = recompute_headline_metrics(empty)
    assert metrics["n_trades"] == 0.0
    assert metrics["total_return"] == 0.0
    assert metrics["max_drawdown"] == 0.0
    assert metrics["sharpe_periodic"] != metrics["sharpe_periodic"]  # NaN, honest


def test_round_floats_policy() -> None:
    assert round_floats(1.23456789012345678) == round(1.23456789012345678, 12)
    assert round_floats({"a": [1.5, "x", None]}) == {"a": [1.5, "x", None]}
    nan = float("nan")
    assert round_floats(nan) != round_floats(nan)  # NaN survives rounding


def test_manifest_recorder_empty_run(tmp_path) -> None:
    # direct build_bundle with zero reads: merkle root = sha256 of empty string
    from quant_fund.proof.bundle import build_bundle

    recorder = InMemoryRecorder()
    summary = recorder.manifest_summary()
    bundle = build_bundle(
        run_kind="research",
        data_manifest=summary,
        config_dump={"a": 1},
        seed=0,
        signal_log=pl.DataFrame(schema={"decision_time": pl.Datetime("us", "UTC")}),
        trade_log=pl.DataFrame(schema={"fill_time": pl.Datetime("us", "UTC"), "nav": pl.Float64}),
        engine_metrics={},
        bundle_dir=tmp_path,
        signer=NullSigner(),
    )
    assert bundle.data_manifest.merkle_root == sha256_hex_bytes(b"")
    assert bundle.prev_bundle_hash == GENESIS_HASH


def test_make_read_record_used_by_fake_vault(tmp_path) -> None:
    bundle, _ = mint_synthetic_bundle(tmp_path)
    reads = bundle.data_manifest.reads
    assert {r.dataset for r in reads} == {"silver/bars", "gold/weights"}
    assert all(len(r.content_sha256) == 64 for r in reads)
    recorder_check = make_read_record("x", T0, rows=1, content_sha256="0" * 64)
    assert recorder_check.params == {}
