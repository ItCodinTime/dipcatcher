"""Proven runner end-to-end + fail-closed tests (WAVE2.md §4).

All vaults are tiny synthetic datasets built in tmp_path; synthetic data is
correctness evidence only, never a performance claim.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError

from quant_fund.leakage.watchdog import LeakageError
from quant_fund.pit.corrections import RestatementPolicy
from quant_fund.pit.frame import PitFrame
from quant_fund.pit.vault import PitVault
from quant_fund.proof import runner as runner_mod
from quant_fund.proof.bundle import load_chain
from quant_fund.proof.runner import run_backtest_proven, run_proven
from quant_fund.proof.verify import verify_bundle
from quant_fund.proofcore.contracts import (
    DecisionGrid,
    DecisionTrace,
    FeatureDecl,
    ProofError,
    RunSpec,
    sha256_hex_bytes,
    sha256_hex_json,
)

DATASET = "silver/bars"
T0 = datetime(2024, 1, 10, tzinfo=UTC)
DAY = timedelta(days=1)


def _bars_rows(n: int = 20) -> list[dict[str, object]]:
    rows = []
    for i in range(n):
        t = datetime(2024, 1, 1, tzinfo=UTC) + i * DAY
        rows.append({"event_time": t, "known_at": t, "close": 100.0 + i + (0.5 if i % 2 else -0.5)})
    return rows


def _build_vault(root: Path, *, extra: list[dict[str, object]] | None = None) -> PitVault:
    vault = PitVault(root)
    vault.create_dataset(DATASET, security_level=False)
    vault.append(DATASET, pl.DataFrame(_bars_rows()))
    if extra:
        vault.append(DATASET, pl.DataFrame(extra))
    return vault


def _spec(estimator: str = "ewma_signal", count: int = 6, seed: int = 42) -> RunSpec:
    params: dict[str, object] = {
        "label": {"dataset": DATASET, "column": "close", "horizon": 1},
    }
    if estimator == "ewma_signal":
        params["span"] = 3.0
    return RunSpec(
        name="runner-test",
        vault_uri="vault://main",
        decision_grid=DecisionGrid(start=T0, step="1d", count=count),
        features=(
            FeatureDecl(
                name="lag1",
                kind="vault_column_lag",
                params={"dataset": DATASET, "column": "close", "lag": 1},
            ),
            FeatureDecl(
                name="mom",
                kind="vault_window_agg",
                params={"dataset": DATASET, "column": "close", "window": 2, "agg": "mean"},
            ),
            FeatureDecl(
                name="prev",
                kind="prior_decision_state",
                params={"field": "last_signal"},
            ),
        ),
        estimator=estimator,
        estimator_params=params,
        seed=seed,
    )


# -- end-to-end ---------------------------------------------------------------


@pytest.mark.parametrize("estimator", ["ewma_signal", "linear_regression_np"])
def test_run_proven_end_to_end_mints_verifiable_bundle(tmp_path, estimator: str) -> None:
    vault = _build_vault(tmp_path / "pit")
    bundle_dir = tmp_path / "proofs"
    ok, bundle_id = run_proven(_spec(estimator=estimator), vault=vault, bundle_dir=bundle_dir)
    assert ok

    # Wave-2 sidecars exist alongside the wave-1 sidecars.
    for suffix in ("trace.json", "env.json", "seeds.json", "config.json", "metrics.json"):
        assert (bundle_dir / f"{bundle_id}.{suffix}").is_file(), suffix

    # Trace chain verifies and commits to the spec.
    trace = DecisionTrace.model_validate_json((bundle_dir / f"{bundle_id}.trace.json").read_bytes())
    trace.verify_chain()
    assert len(trace.rows) == 6
    assert trace.spec_sha256 == sha256_hex_json(_spec(estimator=estimator).model_dump(mode="json"))

    # Cold start: the first window predicts 0.0 (no matured labels yet).
    signals = pl.read_parquet(bundle_dir / f"{bundle_id}.signals.parquet")
    assert signals["target_weight"][0] == 0.0

    # Sidecar hashes are committed in the (hash-checked) config sidecar.
    config = json.loads((bundle_dir / f"{bundle_id}.config.json").read_bytes())
    for kind in ("trace", "env", "seeds"):
        expected = config["sidecars"][f"{kind}_sha256"]
        actual = sha256_hex_bytes((bundle_dir / f"{bundle_id}.{kind}.json").read_bytes())
        assert actual == expected, kind

    # Seeds sidecar derives deterministically from the top-level seed.
    seeds = json.loads((bundle_dir / f"{bundle_id}.seeds.json").read_bytes())
    assert seeds["seed"] == 42
    assert seeds["window_seeds"]["0"] == sha256_hex_bytes(b"42|0")

    # The wave-1 verifier accepts the bundle (hash checks, chain, metrics).
    result = verify_bundle(
        bundle_dir / "bundles" / f"{bundle_id}.json",
        bundle_dir=bundle_dir,
        strict_signature=False,
    )
    assert result.ok, result.reasons


def test_run_proven_is_deterministic(tmp_path) -> None:
    spec = _spec()
    ok1, id1 = run_proven(spec, vault=_build_vault(tmp_path / "pit1"), bundle_dir=tmp_path / "b1")
    ok2, id2 = run_proven(spec, vault=_build_vault(tmp_path / "pit2"), bundle_dir=tmp_path / "b2")
    assert ok1 and ok2
    assert id1 == id2
    assert (tmp_path / "b1" / f"{id1}.trace.json").read_bytes() == (
        tmp_path / "b2" / f"{id2}.trace.json"
    ).read_bytes()


def test_run_proven_chains_into_existing_bundle_dir(tmp_path) -> None:
    bundle_dir = tmp_path / "proofs"
    vault = _build_vault(tmp_path / "pit")
    _, first_id = run_proven(_spec(), vault=vault, bundle_dir=bundle_dir)
    _, second_id = run_proven(_spec(seed=43), vault=vault, bundle_dir=bundle_dir)
    chain = load_chain(bundle_dir)
    assert [b.bundle_id for b in chain] == [first_id, second_id]
    assert chain[1].prev_bundle_hash == first_id


def test_run_proven_signed_bundle_verifies_strict(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PROOFCORE_SIGNING_KEY", "w6-test-key")
    bundle_dir = tmp_path / "proofs"
    ok, bundle_id = run_proven(
        _spec(),
        vault=_build_vault(tmp_path / "pit"),
        bundle_dir=bundle_dir,
        signing_key=b"w6-test-key",
    )
    assert ok
    result = verify_bundle(
        bundle_dir / "bundles" / f"{bundle_id}.json",
        bundle_dir=bundle_dir,
        strict_signature=True,
    )
    assert result.ok, result.reasons


# -- fail closed ---------------------------------------------------------------


def test_unknown_estimator_fails_closed(tmp_path) -> None:
    with pytest.raises(ProofError, match="estimator_not_allowlisted"):
        run_proven(
            _spec(estimator="xgboost_please"),
            vault=_build_vault(tmp_path / "pit"),
            bundle_dir=tmp_path / "proofs",
        )
    assert not (tmp_path / "proofs").exists()


def test_unknown_feature_kind_rejected_by_contract() -> None:
    with pytest.raises(ValidationError):
        FeatureDecl(name="evil", kind="python_callable", params={})


def test_missing_label_declaration_fails_closed(tmp_path) -> None:
    spec = RunSpec(
        name="no-label",
        vault_uri="vault://main",
        decision_grid=DecisionGrid(start=T0, step="1d", count=3),
        features=(),
        estimator="ewma_signal",
        estimator_params={},
        seed=1,
    )
    with pytest.raises(ProofError, match="estimator_params_missing_label"):
        run_proven(spec, vault=_build_vault(tmp_path / "pit"), bundle_dir=tmp_path / "proofs")


def test_invalid_label_horizon_fails_closed(tmp_path) -> None:
    spec = _spec()
    spec = RunSpec(
        **{
            **spec.model_dump(),
            "estimator_params": {"label": {"dataset": DATASET, "column": "close", "horizon": 0}},
        }
    )
    with pytest.raises(ProofError, match="horizon"):
        run_proven(spec, vault=_build_vault(tmp_path / "pit"), bundle_dir=tmp_path / "proofs")


def test_invalid_estimator_params_fail_closed(tmp_path) -> None:
    spec = _spec(estimator="linear_regression_np")
    spec = RunSpec(
        **{
            **spec.model_dump(),
            "estimator_params": {
                "bogus_kwarg": 1,
                "label": {"dataset": DATASET, "column": "close", "horizon": 1},
            },
        }
    )
    with pytest.raises(ProofError, match="estimator_params_invalid"):
        run_proven(spec, vault=_build_vault(tmp_path / "pit"), bundle_dir=tmp_path / "proofs")


def test_bad_feature_params_fail_closed(tmp_path) -> None:
    spec = _spec()
    bad = FeatureDecl(
        name="bad",
        kind="vault_column_lag",
        params={"dataset": DATASET, "column": "close", "lag": 0},
    )
    spec = RunSpec(**{**spec.model_dump(), "features": (*spec.model_dump()["features"], bad)})
    with pytest.raises(ProofError, match="lag"):
        run_proven(spec, vault=_build_vault(tmp_path / "pit"), bundle_dir=tmp_path / "proofs")
    assert not (tmp_path / "proofs").exists()


def test_unknown_window_agg_fails_closed(tmp_path) -> None:
    spec = _spec()
    bad = FeatureDecl(
        name="bad",
        kind="vault_window_agg",
        params={"dataset": DATASET, "column": "close", "window": 2, "agg": "median"},
    )
    spec = RunSpec(**{**spec.model_dump(), "features": (*spec.model_dump()["features"], bad)})
    with pytest.raises(ProofError, match="agg"):
        run_proven(spec, vault=_build_vault(tmp_path / "pit"), bundle_dir=tmp_path / "proofs")


def test_unknown_prior_state_field_fails_closed(tmp_path) -> None:
    spec = _spec()
    bad = FeatureDecl(name="bad", kind="prior_decision_state", params={"field": "future_weight"})
    spec = RunSpec(**{**spec.model_dump(), "features": (*spec.model_dump()["features"], bad)})
    with pytest.raises(ProofError, match="field"):
        run_proven(spec, vault=_build_vault(tmp_path / "pit"), bundle_dir=tmp_path / "proofs")


def test_nan_injection_aborts_with_nan_in_window(tmp_path) -> None:
    nan_row = {
        "event_time": datetime(2024, 1, 5, tzinfo=UTC),
        "known_at": datetime(2024, 1, 5, tzinfo=UTC),
        "close": float("nan"),
    }
    vault = _build_vault(tmp_path / "pit", extra=[nan_row])
    with pytest.raises(ProofError, match="nan_in_window"):
        run_proven(_spec(), vault=vault, bundle_dir=tmp_path / "proofs")
    assert not (tmp_path / "proofs").exists()


def test_seeded_future_knowledge_row_aborts_with_leakage_error(tmp_path) -> None:
    """A vault that hands back rows known AFTER the decision time must abort.

    The honest PitVault filter makes this unreachable through normal reads
    (known_at <= watermark <= decision_time), so the test rigs the vault to
    simulate a corrupted/backdoored read path; the runner's defense-in-depth
    check (on top of the watchdog) fails closed and leaves no bundle behind.
    """

    class RiggedVault(PitVault):
        def asof(
            self,
            name,
            t,
            *,
            columns=None,
            policy=RestatementPolicy.LATEST_KNOWN,
            decision_time=None,
        ):
            frame = super().asof(
                name, t, columns=columns, policy=policy, decision_time=decision_time
            )
            future_known = frame.frame.head(1).with_columns(
                pl.lit(t + timedelta(days=10)).alias("known_at")
            )
            return PitFrame.build(pl.concat([frame.frame, future_known]), dataset=name, asof=t)

    vault = RiggedVault(tmp_path / "pit")
    vault.create_dataset(DATASET, security_level=False)
    vault.append(DATASET, pl.DataFrame(_bars_rows()))
    with pytest.raises(LeakageError, match="leakage"):
        run_proven(_spec(), vault=vault, bundle_dir=tmp_path / "proofs")
    assert not (tmp_path / "proofs").exists()


def test_tampered_trace_fails_verify_chain(tmp_path) -> None:
    bundle_dir = tmp_path / "proofs"
    _, bundle_id = run_proven(_spec(), vault=_build_vault(tmp_path / "pit"), bundle_dir=bundle_dir)
    raw = json.loads((bundle_dir / f"{bundle_id}.trace.json").read_bytes())
    raw["rows"][1]["data_manifest_sha256"] = "f" * 64
    tampered = DecisionTrace.model_validate(raw)
    with pytest.raises(ProofError, match="chain"):
        tampered.verify_chain()
    # A tampered head hash is caught even when rows are untouched.
    raw = json.loads((bundle_dir / f"{bundle_id}.trace.json").read_bytes())
    raw["head_row_sha256"] = "e" * 64
    with pytest.raises(ProofError, match="head"):
        DecisionTrace.model_validate(raw).verify_chain()


def test_vault_unavailable_propagates(tmp_path) -> None:
    """Grid reads before any known data fail closed via the vault."""
    vault = PitVault(tmp_path / "pit")
    vault.create_dataset(DATASET, security_level=False)
    late = [
        {"event_time": T0 + 30 * DAY, "known_at": T0 + 30 * DAY, "close": 1.0},
    ]
    vault.append(DATASET, pl.DataFrame(late))
    from quant_fund.pit.frame import VaultUnavailableError

    with pytest.raises(VaultUnavailableError):
        run_proven(_spec(), vault=vault, bundle_dir=tmp_path / "proofs")
    assert not (tmp_path / "proofs").exists()


def test_wave1_stub_unchanged_fail_closed(tmp_path) -> None:
    from quant_fund.config.models import AppConfig

    with pytest.raises(ProofError, match="per-decision as-of vault reads"):
        run_backtest_proven(
            AppConfig(), seed=7, pit_root=tmp_path / "pit", bundle_dir=tmp_path / "proofs"
        )
    assert not (tmp_path / "proofs").exists()


def test_window_agg_variants_end_to_end(tmp_path) -> None:
    """std/min/max/last aggregations are all legal, deterministic features."""
    spec = _spec()
    aggs = tuple(
        FeatureDecl(
            name=f"agg_{agg}",
            kind="vault_window_agg",
            params={"dataset": DATASET, "column": "close", "window": 2, "agg": agg},
        )
        for agg in ("std", "min", "max", "last")
    )
    spec = RunSpec(**{**spec.model_dump(), "features": (*spec.model_dump()["features"], *aggs)})
    ok, bundle_id = run_proven(
        spec, vault=_build_vault(tmp_path / "pit"), bundle_dir=tmp_path / "proofs"
    )
    assert ok
    trace = DecisionTrace.model_validate_json(
        (tmp_path / "proofs" / f"{bundle_id}.trace.json").read_bytes()
    )
    trace.verify_chain()


def test_feature_missing_params_fail_closed(tmp_path) -> None:
    spec = _spec()
    bad = FeatureDecl(name="bad", kind="vault_column_lag", params={"dataset": DATASET})
    spec = RunSpec(**{**spec.model_dump(), "features": (*spec.model_dump()["features"], bad)})
    with pytest.raises(ProofError, match="missing params"):
        run_proven(spec, vault=_build_vault(tmp_path / "pit"), bundle_dir=tmp_path / "proofs")


def test_window_agg_zero_window_fails_closed(tmp_path) -> None:
    spec = _spec()
    bad = FeatureDecl(
        name="bad",
        kind="vault_window_agg",
        params={"dataset": DATASET, "column": "close", "window": 0, "agg": "mean"},
    )
    spec = RunSpec(**{**spec.model_dump(), "features": (*spec.model_dump()["features"], bad)})
    with pytest.raises(ProofError, match="window"):
        run_proven(spec, vault=_build_vault(tmp_path / "pit"), bundle_dir=tmp_path / "proofs")


def test_label_declaration_missing_keys_fail_closed(tmp_path) -> None:
    spec = _spec()
    spec = RunSpec(**{**spec.model_dump(), "estimator_params": {"label": {"dataset": DATASET}}})
    with pytest.raises(ProofError, match="label declaration missing"):
        run_proven(spec, vault=_build_vault(tmp_path / "pit"), bundle_dir=tmp_path / "proofs")


# -- helpers -------------------------------------------------------------------


def test_code_fingerprint_nogit_fallback(monkeypatch) -> None:
    monkeypatch.setattr(runner_mod.shutil, "which", lambda name: None)
    assert runner_mod._code_fingerprint() == "nogit"


def test_code_fingerprint_git_failure_fallback(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(runner_mod.shutil, "which", lambda name: "/usr/bin/git")

    def _boom(*args, **kwargs):
        raise OSError("no git")

    monkeypatch.setattr(runner_mod.subprocess, "run", _boom)
    assert runner_mod._code_fingerprint() == "nogit"

    class _Proc:
        returncode = 128
        stdout = ""

    monkeypatch.setattr(runner_mod.subprocess, "run", lambda *a, **k: _Proc())
    assert runner_mod._code_fingerprint() == "nogit"


def test_env_fingerprint_format() -> None:
    parts = runner_mod._env_fingerprint().split("|")
    assert len(parts) == 3
    assert parts[2] == "quant_fund-dev"


def test_package_version_missing() -> None:
    assert runner_mod._package_version("definitely_not_installed_xyz") == "not-installed"
    assert runner_mod._package_version("numpy") != "not-installed"
