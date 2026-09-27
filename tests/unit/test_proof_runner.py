"""Fail-closed runner checks; synthetic data is correctness evidence only."""

from __future__ import annotations

from datetime import timedelta

import polars as pl
import pytest
from typer.testing import CliRunner

from quant_fund.config.models import AppConfig
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


def test_proof_run_cli_exits_before_loading_config_or_writing_bundle(tmp_path) -> None:
    bundle_dir = tmp_path / "proofs"
    result = CliRunner().invoke(
        proof_app,
        [
            "run",
            "--config",
            str(tmp_path / "missing.yml"),
            "--seed",
            "7",
            "--pit-root",
            str(tmp_path / "pit"),
            "--bundle-dir",
            str(bundle_dir),
        ],
    )
    assert result.exit_code == 2
    assert "per-decision as-of vault reads" in result.output
    assert not bundle_dir.exists()


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
