"""Synthetic bundle identity and tamper-evasion correctness checks.

Same inputs and seed yield the same bundle identity. These bundle-builder
tests do not claim a causal historical run; that runner fails closed.
"""

from __future__ import annotations

import io
import json
import time

import polars as pl

from quant_fund.proof.bundle import canonical_bundle_bytes, sidecar_paths
from quant_fund.proof.verify import verify_bundle
from tests.unit.proof_fake_vault import mint_synthetic_bundle


def _run(tmp_path, *, seed: int = 123):
    """Build one synthetic bundle in a fresh directory."""
    return mint_synthetic_bundle(tmp_path, seed=seed)


def _canon(path) -> dict:
    return json.loads(path.read_bytes())


def test_double_run_bundle_identity(tmp_path_factory) -> None:
    """Same config + same data + same seed => identical bundle_id.

    Builds the same synthetic inputs twice, with a pinned seed, and compares
    canonical bundle bytes apart from created_utc.
    """
    bundle_a, dir_a = _run(tmp_path_factory.mktemp("run-a"))
    time.sleep(0.01)  # force a distinct wall clock; identity must survive it
    bundle_b, dir_b = _run(tmp_path_factory.mktemp("run-b"))

    assert bundle_a.bundle_id == bundle_b.bundle_id
    assert bundle_a.metrics_recompute == bundle_b.metrics_recompute

    canon_a = _canon(dir_a / "bundles" / f"{bundle_a.bundle_id}.json")
    canon_b = _canon(dir_b / "bundles" / f"{bundle_b.bundle_id}.json")
    assert canon_a.pop("created_utc") != "" and canon_b.pop("created_utc") != ""
    assert canon_a == canon_b
    assert canonical_bundle_bytes(bundle_a) != b""  # sanity

    # sidecar payload bytes are identical too
    for kind in ("signal_log", "trade_log", "metrics", "config"):
        assert (
            sidecar_paths(dir_a, bundle_a.bundle_id)[kind].read_bytes()
            == sidecar_paths(dir_b, bundle_b.bundle_id)[kind].read_bytes()
        )


def test_double_run_different_seed_differs(tmp_path_factory) -> None:
    bundle_a, _ = _run(tmp_path_factory.mktemp("run-a"), seed=1)
    bundle_b, _ = _run(tmp_path_factory.mktemp("run-b"), seed=2)
    assert bundle_a.bundle_id != bundle_b.bundle_id


def test_tamper_evasion_fails(tmp_path) -> None:
    """Flip one float in the trade-log parquet -> verification fails loudly."""
    bundle, bundle_dir = _run(tmp_path)
    bundle_path = bundle_dir / "bundles" / f"{bundle.bundle_id}.json"
    trades_path = sidecar_paths(bundle_dir, bundle.bundle_id)["trade_log"]
    frame = pl.read_parquet(trades_path)
    # flip ONE fill's nav mark (a global scale would be pct_change-invariant)
    tampered = frame.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.col("nav") * 1.5)
        .otherwise(pl.col("nav"))
        .alias("nav")
    )
    tampered.write_parquet(trades_path)

    result = verify_bundle(bundle_path, bundle_dir=bundle_dir, strict_signature=False)
    assert not result.ok
    assert any(
        "trade_log" in reason or reason.startswith("metrics_recompute:")
        for reason in result.reasons
    )
    # the nav flip moves the recomputed metrics as well
    assert not all(result.metrics_match.values())


def test_tamper_metrics_json_only_fails(tmp_path) -> None:
    """Edit metrics JSON only -> metrics_sha256 reason, trade log untouched."""
    bundle, bundle_dir = _run(tmp_path)
    bundle_path = bundle_dir / "bundles" / f"{bundle.bundle_id}.json"
    metrics_path = sidecar_paths(bundle_dir, bundle.bundle_id)["metrics"]
    payload = json.loads(metrics_path.read_bytes())
    payload["sharpe"] = 42.0
    metrics_path.write_bytes(json.dumps(payload).encode())

    result = verify_bundle(bundle_path, bundle_dir=bundle_dir, strict_signature=False)
    assert not result.ok
    assert "sidecar:metrics_sha256_mismatch" in result.reasons
    # trade-log-driven recompute is unaffected by the metrics-file edit
    assert all(result.metrics_match.values())


def test_tamper_recorded_read_fails(tmp_path) -> None:
    """Edit any logged field of a recorded read -> Merkle + self-hash reasons."""
    bundle, bundle_dir = _run(tmp_path)
    bundle_path = bundle_dir / "bundles" / f"{bundle.bundle_id}.json"
    payload = _canon(bundle_path)
    payload["data_manifest"]["reads"][0]["content_sha256"] = "f" * 64
    bundle_path.write_bytes(json.dumps(payload).encode())

    result = verify_bundle(bundle_path, bundle_dir=bundle_dir, strict_signature=False)
    assert not result.ok
    assert "data_manifest:merkle_root_mismatch" in result.reasons
    assert "bundle_id:self_hash_mismatch" in result.reasons


def test_replay_fails_closed_without_causal_runner(tmp_path) -> None:
    """A bundle hash check never masquerades as historical replay evidence."""
    from quant_fund.proof.replay import replay_bundle

    bundle, bundle_dir = _run(tmp_path, seed=777)
    ok, detail = replay_bundle(
        bundle_dir / "bundles" / f"{bundle.bundle_id}.json",
        bundle_dir=bundle_dir,
        pit_root=tmp_path / "pit",
    )
    assert not ok
    assert detail == "runner_unavailable:per_decision_asof_not_implemented"


def test_signal_and_trade_logs_parse(tmp_path) -> None:
    bundle, bundle_dir = _run(tmp_path)
    signals = pl.read_parquet(
        io.BytesIO(sidecar_paths(bundle_dir, bundle.bundle_id)["signal_log"].read_bytes())
    )
    assert {"decision_time", "security_id", "score", "weight"} <= set(signals.columns)
    trades = pl.read_parquet(
        io.BytesIO(sidecar_paths(bundle_dir, bundle.bundle_id)["trade_log"].read_bytes())
    )
    assert {"fill_time", "nav"} <= set(trades.columns)


def test_double_run_bundle_identity_bundle_level(tmp_path_factory) -> None:
    """Engine-free determinism floor: same inputs -> identical bundle_id.

    Covers the A3 #6 identity at the bundle-construction layer even in minimal
    environments where the engine import chain (mlflow) is unavailable.
    """
    bundle_a, dir_a = mint_synthetic_bundle(tmp_path_factory.mktemp("syn-a"))
    time.sleep(0.01)
    bundle_b, dir_b = mint_synthetic_bundle(tmp_path_factory.mktemp("syn-b"))

    assert bundle_a.bundle_id == bundle_b.bundle_id
    assert bundle_a.metrics_recompute == bundle_b.metrics_recompute
    canon_a = _canon(dir_a / "bundles" / f"{bundle_a.bundle_id}.json")
    canon_b = _canon(dir_b / "bundles" / f"{bundle_b.bundle_id}.json")
    canon_a.pop("created_utc")
    canon_b.pop("created_utc")
    assert canon_a == canon_b
    for kind in ("signal_log", "trade_log", "metrics", "config"):
        assert (
            sidecar_paths(dir_a, bundle_a.bundle_id)[kind].read_bytes()
            == sidecar_paths(dir_b, bundle_b.bundle_id)[kind].read_bytes()
        )
