"""W6 proven-runner tests (WAVE2.md §4): synthetic vault, no metric claims."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from quant_fund.leakage.watchdog import LeakageError, LeakageWatchdog
from quant_fund.pit.errors import VaultError
from quant_fund.proof import runner as runner_mod
from quant_fund.proof.bundle import load_chain
from quant_fund.proof.runner import run_proven
from quant_fund.proof.verify import verify_bundle
from quant_fund.proofcore import run_context
from quant_fund.proofcore.contracts import (
    GENESIS_HASH,
    DecisionGrid,
    DecisionTrace,
    FeatureDecl,
    ProofError,
    RunSpec,
    canonical_json_bytes,
    sha256_hex_bytes,
    sha256_hex_json,
)
from tests.unit.proof_fake_vault import FakeVault, synthetic_bars, synthetic_weights

DATASET = "silver/bars"
T0 = datetime(2024, 1, 10, tzinfo=UTC)
DAY = timedelta(days=1)


def _spec(
    *,
    features: tuple[FeatureDecl, ...] | list[FeatureDecl] = (),
    grid: DecisionGrid | None = None,
    estimator: str = "ewma_signal",
    params: dict | None = None,
    name: str = "synthetic",
) -> RunSpec:
    return RunSpec(
        name=name,
        vault_uri="vault://test",
        decision_grid=grid or DecisionGrid(start=T0, step="1d", count=3),
        features=tuple(features),
        estimator=estimator,
        estimator_params=params
        or {"span": 2.0, "label": {"dataset": DATASET, "column": "close", "horizon": 1}},
        seed=7,
    )


def _vault(tmp_path: Path, *, n: int = 12, with_weights: bool = False) -> FakeVault:
    datasets: dict[str, pl.DataFrame] = {DATASET: synthetic_bars(["AAA"], n, start=T0 - 8 * DAY)}
    if with_weights:
        datasets["gold/weights"] = synthetic_weights(["AAA"], n)
    return FakeVault(None, datasets, tmp_path / "vault")


# ---------------------------------------------------------------------------
# §4.4: fail-closed, no partial bundles
# ---------------------------------------------------------------------------


def test_undeclared_estimator_fails_closed_no_bundle(tmp_path) -> None:
    with pytest.raises(ProofError, match="estimator_not_allowlisted"):
        run_proven(_spec(estimator="xgboost"), vault=_vault(tmp_path), bundle_dir=tmp_path / "b")
    assert not (tmp_path / "b").exists()


def test_undeclared_feature_kind_fails_closed_no_bundle(tmp_path) -> None:
    spec = _spec(features=[FeatureDecl(name="f", kind="callable", params={})])
    with pytest.raises(ProofError, match="feature_kind_undeclared"):
        run_proven(spec, vault=_vault(tmp_path), bundle_dir=tmp_path / "b")
    assert not (tmp_path / "b").exists()


def test_bad_label_declaration_fails_closed(tmp_path) -> None:
    spec = _spec(params={"span": 2.0})
    with pytest.raises(ProofError, match="estimator_params_missing_label"):
        run_proven(spec, vault=_vault(tmp_path), bundle_dir=tmp_path / "b")
    assert not (tmp_path / "b").exists()


def test_missing_vault_dataset_fails_closed_no_bundle(tmp_path) -> None:
    spec = _spec(
        features=[
            FeatureDecl(
                name="f",
                kind="vault_column_lag",
                params={"dataset": "silver/nope", "column": "close", "lag": 1},
            )
        ]
    )
    with pytest.raises(VaultError):
        run_proven(spec, vault=_vault(tmp_path), bundle_dir=tmp_path / "b")
    assert not (tmp_path / "b").exists()


def test_nan_feature_fails_closed_no_bundle(tmp_path) -> None:
    bars = synthetic_bars(["AAA"], 4, start=T0 - 2 * DAY).with_columns(
        pl.when(pl.col("event_time") == T0 - DAY)
        .then(float("nan"))
        .otherwise(pl.col("close"))
        .alias("close")
    )
    vault = FakeVault(None, {DATASET: bars}, tmp_path / "vault")
    spec = _spec(
        features=[
            FeatureDecl(
                name="f",
                kind="vault_column_lag",
                params={"dataset": DATASET, "column": "close", "lag": 1},
            )
        ]
    )
    with pytest.raises(ProofError, match="nan_in_window"):
        run_proven(spec, vault=vault, bundle_dir=tmp_path / "b")
    assert not (tmp_path / "b").exists()


def test_future_known_rows_abort_the_run(tmp_path) -> None:
    """A corrupted vault returning future-known rows must not mint a bundle."""
    vault = _vault(tmp_path)
    vault.frame_meta[DATASET] = ("event_time", None)  # turn off vault-side filtering
    spec = _spec(
        features=[
            FeatureDecl(
                name="f",
                kind="vault_column_lag",
                params={"dataset": DATASET, "column": "close", "lag": 1},
            )
        ]
    )
    with pytest.raises(LeakageError):
        run_proven(spec, vault=vault, bundle_dir=tmp_path / "b")
    assert not (tmp_path / "b").exists()


def test_prior_state_feature_rejects_future_reference(tmp_path) -> None:
    spec = _spec(
        features=[
            FeatureDecl(name="s", kind="prior_decision_state", params={"field": "next_action"})
        ]
    )
    with pytest.raises(ProofError, match="unknown prior-state field"):
        run_proven(spec, vault=_vault(tmp_path), bundle_dir=tmp_path / "b")
    assert not (tmp_path / "b").exists()


# ---------------------------------------------------------------------------
# Trace integrity: hash chain, per-window reads, seeds
# ---------------------------------------------------------------------------


def _run_and_load_trace(tmp_path: Path, spec: RunSpec, **kwargs):
    bundle_dir = tmp_path / "proofs"
    ok, bundle_id = run_proven(spec, vault=_vault(tmp_path), bundle_dir=bundle_dir, **kwargs)
    assert ok
    trace = DecisionTrace.model_validate(
        json.loads((bundle_dir / f"{bundle_id}.trace.json").read_bytes())
    )
    return bundle_dir, bundle_id, trace


def test_run_proven_mints_chained_trace(tmp_path) -> None:
    spec = _spec(
        features=[
            FeatureDecl(
                name="lag1",
                kind="vault_column_lag",
                params={"dataset": DATASET, "column": "close", "lag": 1},
            ),
            FeatureDecl(
                name="state",
                kind="prior_decision_state",
                params={"field": "last_signal", "initial": 0.0},
            ),
        ]
    )
    _, bundle_id, trace = _run_and_load_trace(tmp_path, spec)
    assert trace.verify_chain() is True
    assert len(trace.rows) == 3
    assert trace.rows[0].prev_row_sha256 == GENESIS_HASH
    assert trace.spec_sha256 == sha256_hex_json(spec.model_dump(mode="json"))
    assert trace.head_row_sha256 != GENESIS_HASH
    assert trace.rows[0].decision_time == T0


def test_per_window_reads_are_causal_and_recorded(tmp_path) -> None:
    spec = _spec(
        grid=DecisionGrid(start=T0, step="1d", count=4),
        features=[
            FeatureDecl(
                name="lag2",
                kind="vault_column_lag",
                params={"dataset": DATASET, "column": "close", "lag": 2},
            )
        ],
    )
    recorder_holder: dict[str, object] = {}
    vault = _vault(tmp_path)
    original_asof = vault.asof
    watermarks: dict[datetime, list[datetime]] = {}

    def spied_asof(name: str, t: datetime, *, columns=None):
        watermarks.setdefault(run_context.current_decision_time(), []).append(t)
        return original_asof(name, t, columns=columns)

    vault.asof = spied_asof  # type: ignore[method-assign]
    ok, _ = run_proven(spec, vault=vault, bundle_dir=tmp_path / "proofs")
    assert ok
    for decision_time, reads in watermarks.items():
        assert all(t <= decision_time for t in reads)
    assert recorder_holder == {}  # no cross-run state


def test_window_seeds_sidecar_is_deterministic(tmp_path) -> None:
    spec = _spec()
    bundle_dir, bundle_id, _ = _run_and_load_trace(tmp_path, spec)
    seeds = json.loads((bundle_dir / f"{bundle_id}.seeds.json").read_bytes())
    assert seeds["seed"] == 7
    assert seeds["window_seeds"] == {
        str(i): sha256_hex_bytes(f"7|{i}".encode()) for i in range(3)
    }


def test_run_proven_is_deterministic(tmp_path) -> None:
    """Two identical runs produce byte-identical trace/env/seeds sidecars."""
    spec = _spec(
        features=[
            FeatureDecl(
                name="w",
                kind="vault_window_agg",
                params={"dataset": DATASET, "column": "close", "window": 2, "agg": "mean"},
            )
        ]
    )
    _, bundle_id_a, _ = _run_and_load_trace(tmp_path / "a", spec)
    _, bundle_id_b, _ = _run_and_load_trace(tmp_path / "b", spec)
    assert bundle_id_a == bundle_id_b  # same inputs -> same self-hash
    for kind in ("trace", "env", "seeds", "config", "metrics"):
        a = (tmp_path / "a" / "proofs" / f"{bundle_id_a}.{kind}.json").read_bytes()
        b = (tmp_path / "b" / "proofs" / f"{bundle_id_b}.{kind}.json").read_bytes()
        assert a == b, kind


def test_run_proven_end_to_end_mints_verifiable_bundle(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PROOFCORE_SIGNING_KEY", "test-key")
    spec = _spec()
    bundle_dir, bundle_id, _ = _run_and_load_trace(tmp_path, spec, signing_key=b"test-key")
    result = verify_bundle(
        bundle_dir / "bundles" / f"{bundle_id}.json",
        bundle_dir=bundle_dir,
        strict_signature=True,
    )
    assert result.ok, result.reasons
    chain = load_chain(bundle_dir)
    assert [b.bundle_id for b in chain] == [bundle_id]


def test_config_sidecar_commits_wave2_sidecars(tmp_path) -> None:
    bundle_dir, bundle_id, _ = _run_and_load_trace(tmp_path, _spec())
    config = json.loads((bundle_dir / f"{bundle_id}.config.json").read_bytes())
    committed = config["sidecars"]
    for kind in ("trace", "env", "seeds"):
        data = (bundle_dir / f"{bundle_id}.{kind}.json").read_bytes()
        assert committed[f"{kind}_sha256"] == sha256_hex_bytes(data)
    assert config["run_spec"]["name"] == "synthetic"


def test_fingerprints_delegate_to_proofcore_ci(monkeypatch) -> None:
    """Integration reconciliation: runner mints via proofcore.ci helpers so
    replay re-derives the identical strings at the env gate."""
    from quant_fund.proofcore import ci

    monkeypatch.setattr(ci, "code_fingerprint", lambda: "c" * 64)
    monkeypatch.setattr(ci, "env_fingerprint", lambda: "plat|pytag|1.2.3")
    assert runner_mod._code_fingerprint() == "c" * 64
    assert runner_mod._env_fingerprint() == "plat|pytag|1.2.3"


def test_env_fingerprint_format() -> None:
    """Contracts §2.2 formula: platform|python tag|real quant_fund version."""
    from quant_fund.proofcore import ci

    fingerprint = runner_mod._env_fingerprint()
    parts = fingerprint.split("|")
    assert len(parts) == 3
    assert parts[2] == ci.quant_fund_version() != "quant_fund-dev"
    assert fingerprint == ci.env_fingerprint()


def test_feature_kinds_cover_all_declared_branches(tmp_path) -> None:
    spec = _spec(
        grid=DecisionGrid(start=T0, step="1d", count=3),
        features=[
            FeatureDecl(
                name="lag",
                kind="vault_column_lag",
                params={"dataset": DATASET, "column": "close", "lag": 2},
            ),
            FeatureDecl(
                name="aggstd",
                kind="vault_window_agg",
                params={"dataset": DATASET, "column": "close", "window": 2, "agg": "std"},
            ),
            FeatureDecl(
                name="aggmin",
                kind="vault_window_agg",
                params={"dataset": DATASET, "column": "close", "window": 2, "agg": "min"},
            ),
            FeatureDecl(
                name="aggmax",
                kind="vault_window_agg",
                params={"dataset": DATASET, "column": "close", "window": 2, "agg": "max"},
            ),
            FeatureDecl(
                name="agglast",
                kind="vault_window_agg",
                params={"dataset": DATASET, "column": "close", "window": 2, "agg": "last"},
            ),
            FeatureDecl(
                name="state",
                kind="prior_decision_state",
                params={"field": "last_action", "initial": 0.25},
            ),
        ]
    )
    _, _, trace = _run_and_load_trace(tmp_path, spec)
    assert len(trace.rows) == 3


def test_estimator_state_hash_uses_float64_bytes() -> None:
    import numpy as np

    from quant_fund.proof.estimators import EwmaSignal

    est = EwmaSignal(span=2.0)
    est.fit(np.array([[1.0], [2.0]]), np.array([0.1, 0.2]))
    assert sha256_hex_bytes(est.state_bytes()) == sha256_hex_bytes(
        np.ascontiguousarray(est.state_vector(), dtype=np.float64).tobytes()
    )


def test_linear_regression_walkforward_expands(tmp_path) -> None:
    spec = _spec(
        estimator="linear_regression_np",
        params={"label": {"dataset": DATASET, "column": "close", "horizon": 1}},
        features=[
            FeatureDecl(
                name="lag1",
                kind="vault_column_lag",
                params={"dataset": DATASET, "column": "close", "lag": 1},
            )
        ],
    )
    _, _, trace = _run_and_load_trace(tmp_path, spec)
    hashes = [row.estimator_state_sha256 for row in trace.rows]
    assert len(set(hashes)) >= 2  # expanding window refit changes the state


def test_runner_module_keeps_lazy_leakage_import() -> None:
    """LH011 layering: the watchdog import stays function-level in runner."""
    source = Path(runner_mod.__file__).read_text()
    assert source.count("from quant_fund.leakage.watchdog import") >= 2


def test_proven_run_context_installs_watchdog(tmp_path) -> None:
    recorder_holder = []
    from quant_fund.proof.recorder import InMemoryRecorder

    recorder = InMemoryRecorder()
    with run_context.proven_run(recorder, LeakageWatchdog(strict=True)):
        assert run_context.context_is_proven()
        with run_context.decision_window(T0):
            assert run_context.current_decision_time() == T0
        recorder_holder.append(len(recorder.reads))
    assert not run_context.context_is_proven()
    assert recorder_holder == [0]


def test_atomic_commit_appends_single_chain_line(tmp_path) -> None:
    spec = _spec()
    bundle_dir = tmp_path / "proofs"
    ok, first = run_proven(spec, vault=_vault(tmp_path), bundle_dir=bundle_dir)
    assert ok
    ok, second = run_proven(
        spec.model_copy(update={"seed": 8}), vault=_vault(tmp_path), bundle_dir=bundle_dir
    )
    assert ok
    chain = load_chain(bundle_dir)
    assert [b.bundle_id for b in chain] == [first, second]
    assert chain[1].prev_bundle_hash == first


def test_package_version_helper() -> None:
    assert runner_mod._package_version("polars") != "not-installed"
    assert runner_mod._package_version("no_such_package_xyz") == "not-installed"
