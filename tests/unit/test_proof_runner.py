"""Engine-coupled runner tests (DESIGN.md §5.2).

The backtest engine import chain pulls the repo SCC (mlflow et al.), so these
tests skip cleanly in minimal environments; CI runs the full env.
"""

from __future__ import annotations

from datetime import timedelta

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
from tests.unit.proof_fake_vault import T0, make_fake_vault, mint_proven_run

N_DAYS = 30  # mint_proven_run default


def test_runner_records_every_vault_read(tmp_path) -> None:
    bundle, bundle_dir = mint_proven_run(tmp_path)
    # ADVERSARIAL §1b-W1: 1 grid-discovery probe at the sentinel + 2 reads per
    # decision time (bars + weights) on the data-derived decision clock.
    assert bundle.data_manifest.n_reads == 1 + 2 * N_DAYS
    assert {r.dataset for r in bundle.data_manifest.reads} == {BARS_DATASET, WEIGHTS_DATASET}
    asofs = [r.asof_utc for r in bundle.data_manifest.reads]
    assert asofs[0] == ASOF_SENTINEL.isoformat()  # grid discovery probe only
    clock = [(T0 + timedelta(days=i, hours=16)).isoformat() for i in range(N_DAYS)]
    assert sorted(asofs[1:]) == sorted([t for t in clock for _ in (BARS_DATASET, WEIGHTS_DATASET)])
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
    assert bundle.data_manifest.n_reads == 1 + 2 * N_DAYS


def test_runner_against_real_pit_vault(tmp_path) -> None:
    """W1 integration: real PitVault on disk, recorder + W3 watchdog wired.

    The W1<->W3 seam is adjudicated and wired: the vault emits
    ``params["max_known_at"]`` on every read and the strict
    ``LeakageWatchdog`` observes each per-decision read against the decision
    clock (ADVERSARIAL §1b-W1). Honest per-decision reads pass.
    """
    from quant_fund.leakage import LeakageWatchdog
    from quant_fund.pit import PitVault
    from tests.unit.proof_fake_vault import synthetic_bars, synthetic_weights

    recorder = InMemoryRecorder()
    pit_root = tmp_path / "pit"
    vault = PitVault(pit_root, recorder=recorder, watchdog=LeakageWatchdog())
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
        vault=vault,
        recorder=recorder,
    )
    assert bundle.data_manifest.n_reads == 1 + 2 * N_DAYS
    assert {r.dataset for r in bundle.data_manifest.reads} == {BARS_DATASET, WEIGHTS_DATASET}
    result = verify_bundle(
        bundle_dir / "bundles" / f"{bundle.bundle_id}.json",
        bundle_dir=bundle_dir,
        strict_signature=False,
    )
    assert result.ok, result.reasons
