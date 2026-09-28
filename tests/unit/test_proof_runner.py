"""Fail-closed runner checks; synthetic data is correctness evidence only."""

from __future__ import annotations

import json
from datetime import timedelta

import polars as pl
import pytest
from typer.testing import CliRunner

from quant_fund.config.models import AppConfig
from quant_fund.pit.vault import PitVault
from quant_fund.proof.cli import proof_app
from quant_fund.proof.recorder import InMemoryRecorder
from quant_fund.proof.runner import BARS_DATASET, WEIGHTS_DATASET, run_backtest_proven
from quant_fund.proof.verify import verify_bundle
from quant_fund.proofcore.contracts import ProofError
from tests.unit.proof_fake_vault import (
    T0,
    FakeVault,
    mint_synthetic_bundle,
    synthetic_bars,
    synthetic_weights,
)


@pytest.mark.parametrize("future_restatement", [False, True])
def test_runner_rejects_before_any_future_snapshot_read(tmp_path, future_restatement: bool) -> None:
    """A later revision of an old weight cannot enter a claimed proven run."""
    recorder = InMemoryRecorder()
    weights = synthetic_weights(["AAA"], 3)
    if future_restatement:
        revised = weights.filter(pl.col("event_time") == T0).with_columns(
            pl.col("target_weight") + 0.5,
            pl.lit(T0 + timedelta(days=30)).alias("known_at"),
        )
        weights = pl.concat([weights, revised])
    vault = FakeVault(
        recorder,
        {
            BARS_DATASET: synthetic_bars(["AAA"], 3),
            WEIGHTS_DATASET: weights,
        },
    )
    bundle_dir = tmp_path / "proofs"
    with pytest.raises(ProofError, match="per-decision as-of vault reads"):
        run_backtest_proven(
            AppConfig(),
            seed=7,
            pit_root=tmp_path / "pit",
            bundle_dir=bundle_dir,
            vault=vault,
            recorder=recorder,
        )
    assert recorder.reads == []
    assert not bundle_dir.exists()


def test_proof_run_cli_rejects_unreadable_spec_before_writing(tmp_path) -> None:
    """Wave 2 (amendment A4): `quant proof run` is the real causal-run entry
    point; an unreadable spec file fails closed with no bundle written."""
    bundle_dir = tmp_path / "proofs"
    result = CliRunner().invoke(
        proof_app,
        [
            "run",
            "--spec",
            str(tmp_path / "missing.json"),
            "--vault-root",
            str(tmp_path / "pit"),
            "--bundle-dir",
            str(bundle_dir),
        ],
    )
    assert result.exit_code == 2
    assert "spec unreadable" in result.output
    assert not bundle_dir.exists()


def test_proof_run_cli_rejects_invalid_spec_json(tmp_path) -> None:
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({"name": "x"}))
    result = CliRunner().invoke(
        proof_app,
        [
            "run",
            "--spec",
            str(spec_path),
            "--vault-root",
            str(tmp_path / "pit"),
            "--bundle-dir",
            str(tmp_path / "proofs"),
        ],
    )
    assert result.exit_code == 2
    assert "spec invalid" in result.output
    assert not (tmp_path / "proofs").exists()


def test_proof_run_and_replay_cli_end_to_end(tmp_path, monkeypatch) -> None:
    """CLI smoke: run mints a signed bundle; replay prints an identical verdict."""
    monkeypatch.setenv("WAVE2_CLI_TEST_KEY", "cli-test-key")
    vault = PitVault(tmp_path / "pit")
    vault.create_dataset(BARS_DATASET, security_level=False)
    rows = [
        {
            "event_time": T0 + i * timedelta(days=1),
            "known_at": T0 + i * timedelta(days=1),
            "close": 100.0 + i,
        }
        for i in range(12)
    ]
    vault.append(BARS_DATASET, pl.DataFrame(rows))
    spec = {
        "name": "cli-e2e",
        "vault_uri": "vault://main",
        "decision_grid": {
            "start": (T0 + timedelta(days=5)).isoformat(),
            "step": "1d",
            "count": 4,
        },
        "features": [
            {
                "name": "lag1",
                "kind": "vault_column_lag",
                "params": {"dataset": BARS_DATASET, "column": "close", "lag": 1},
            }
        ],
        "estimator": "ewma_signal",
        "estimator_params": {
            "span": 3.0,
            "label": {"dataset": BARS_DATASET, "column": "close", "horizon": 1},
        },
        "seed": 5,
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    bundle_dir = tmp_path / "proofs"
    runner = CliRunner()
    run_result = runner.invoke(
        proof_app,
        [
            "run",
            "--spec",
            str(spec_path),
            "--vault-root",
            str(tmp_path / "pit"),
            "--bundle-dir",
            str(bundle_dir),
            "--signing-key-env",
            "WAVE2_CLI_TEST_KEY",
        ],
    )
    assert run_result.exit_code == 0, run_result.output
    bundle_id = json.loads(run_result.output)["bundle_id"]
    assert (bundle_dir / f"{bundle_id}.trace.json").is_file()

    replay_result = runner.invoke(
        proof_app,
        [
            "replay",
            "--bundle",
            str(bundle_dir / "bundles" / f"{bundle_id}.json"),
            "--bundle-dir",
            str(bundle_dir),
            "--vault-root",
            str(tmp_path / "pit"),
        ],
    )
    assert replay_result.exit_code == 0, replay_result.output
    verdict = json.loads(replay_result.output)
    assert verdict["status"] == "identical"
    assert verdict["bundle_id"] == bundle_id


def test_hash_verification_can_pass_while_replay_fails_closed(tmp_path) -> None:
    bundle, bundle_dir = mint_synthetic_bundle(tmp_path)
    bundle_path = bundle_dir / "bundles" / f"{bundle.bundle_id}.json"
    hashes = verify_bundle(bundle_path, bundle_dir=bundle_dir, strict_signature=False)
    assert hashes.ok, hashes.reasons
    replay = verify_bundle(
        bundle_path,
        bundle_dir=bundle_dir,
        strict_signature=False,
        replay=True,
        pit_root=tmp_path / "pit",
    )
    assert not replay.ok
    assert "replay:runner_unavailable:per_decision_asof_not_implemented" in replay.reasons
