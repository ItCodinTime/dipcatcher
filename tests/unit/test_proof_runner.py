"""Engine-coupled runner tests (DESIGN.md §5.2).

The backtest engine import chain pulls the repo SCC (mlflow et al.), so these
tests skip cleanly in minimal environments; CI runs the full env.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mlflow")
pytest.importorskip("pyarrow")

from quant_fund.config.models import AppConfig
from quant_fund.proof.recorder import InMemoryRecorder
from quant_fund.proof.runner import (
    ASOF_SENTINEL,
    BARS_DATASET,
    WEIGHTS_DATASET,
    run_backtest_proven,
)
from quant_fund.proof.sign import HmacSha256Signer
from quant_fund.proof.verify import verify_bundle
from quant_fund.proofcore.contracts import ProofError
from tests.unit.proof_fake_vault import make_fake_vault, mint_proven_run


def test_runner_records_every_vault_read(tmp_path) -> None:
    bundle, bundle_dir = mint_proven_run(tmp_path)
    assert bundle.data_manifest.n_reads == 2
    assert {r.dataset for r in bundle.data_manifest.reads} == {BARS_DATASET, WEIGHTS_DATASET}
    assert all(r.asof_utc == ASOF_SENTINEL.isoformat() for r in bundle.data_manifest.reads)
    assert all(r.rows > 0 for r in bundle.data_manifest.reads)
    result = verify_bundle(
        bundle_dir / "bundles" / f"{bundle.bundle_id}.json",
        bundle_dir=bundle_dir,
        strict_signature=False,
    )
    assert result.ok, result.reasons


def test_runner_signed_run_verifies(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROOFCORE_SIGNING_KEY", "runner-ci-key")
    bundle, bundle_dir = mint_proven_run(tmp_path, signer=HmacSha256Signer.from_env())
    result = verify_bundle(
        bundle_dir / "bundles" / f"{bundle.bundle_id}.json", bundle_dir=bundle_dir
    )
    assert result.ok, result.reasons


def test_runner_rejects_bad_engine_and_seed(tmp_path) -> None:
    recorder = InMemoryRecorder()
    vault = make_fake_vault(recorder)
    with pytest.raises(ProofError):
        run_backtest_proven(
            AppConfig(),
            seed=1,
            pit_root=tmp_path / "pit",
            bundle_dir=tmp_path / "b",
            replay_engine="bogus",  # type: ignore[arg-type]
            vault=vault,
            recorder=recorder,
        )
    with pytest.raises(ProofError):
        run_backtest_proven(
            AppConfig(),
            seed=-1,
            pit_root=tmp_path / "pit",
            bundle_dir=tmp_path / "b",
            vault=vault,
            recorder=recorder,
        )


def test_runner_fast_engine_parity_path(tmp_path) -> None:
    """replay_engine='fast' drives the fast_replay engine through the same bundle path."""
    recorder = InMemoryRecorder()
    vault = make_fake_vault(recorder)
    bundle = run_backtest_proven(
        AppConfig(),
        seed=5,
        pit_root=tmp_path / "pit",
        bundle_dir=tmp_path / "fast",
        replay_engine="fast",
        vault=vault,
        recorder=recorder,
    )
    assert bundle.data_manifest.n_reads == 2


def test_runner_against_real_pit_vault(tmp_path) -> None:
    """W1 integration: real PitVault on disk, recorder wired by the runner."""
    from quant_fund.pit import PitVault
    from tests.unit.proof_fake_vault import synthetic_bars, synthetic_weights

    pit_root = tmp_path / "pit"
    vault = PitVault(pit_root)
    vault.create_dataset(BARS_DATASET)
    vault.append(BARS_DATASET, synthetic_bars(["AAA", "BBB", "CCC"], 30))
    vault.create_dataset(WEIGHTS_DATASET)
    vault.append(WEIGHTS_DATASET, synthetic_weights(["AAA", "BBB", "CCC"], 30))

    bundle_dir = tmp_path / "proofs"
    bundle = run_backtest_proven(
        AppConfig(),
        seed=42,
        pit_root=pit_root,
        bundle_dir=bundle_dir,
    )
    assert bundle.data_manifest.n_reads == 2
    assert {r.dataset for r in bundle.data_manifest.reads} == {BARS_DATASET, WEIGHTS_DATASET}
    result = verify_bundle(
        bundle_dir / "bundles" / f"{bundle.bundle_id}.json",
        bundle_dir=bundle_dir,
        strict_signature=False,
    )
    assert result.ok, result.reasons
