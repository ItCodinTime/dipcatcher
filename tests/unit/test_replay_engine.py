"""Unit tests for the wave-2 replay engine (WAVE2.md §5).

All fixtures are SYNTHETIC: fake executors derive deterministic traces from
the spec; no real performance claims are made. The proven runner (W6) is
never imported — re-execution is injected via the ``executor`` parameter.

Fixture layout mirrors the W6 runner's mint: the config sidecar carries
``{"run_spec": <RunSpec dump>, "sidecars": {"<kind>_sha256": ...}}`` and the
bundle's frozen ``config_sha256`` field anchors the whole commitment chain.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from quant_fund.proof import replay
from quant_fund.proof.bundle import METRIC_KEYS, canonical_json_bytes
from quant_fund.proof.recorder import InMemoryRecorder, make_read_record
from quant_fund.proof.replay import (
    ROW_HASH_FIELDS,
    derive_window_seeds,
    expected_seeds_sidecar,
    replay_bundle,
    window_seed_sha256,
)
from quant_fund.proofcore.contracts import (
    GENESIS_HASH,
    DecisionGrid,
    DecisionTrace,
    DecisionTraceRow,
    FeatureDecl,
    ProofError,
    ReplayVerdict,
    RunSpec,
    sha256_hex_bytes,
    sha256_hex_json,
    trace_row_hash,
)
from tests.unit.proof_fake_vault import synthetic_signal_log, synthetic_trade_log

T0 = datetime(2024, 1, 1, tzinfo=UTC)

_STEP_UNITS = {
    "d": timedelta(days=1),
    "h": timedelta(hours=1),
    "m": timedelta(minutes=1),
    "s": timedelta(seconds=1),
}


def parse_step(step: str) -> timedelta:
    """Test-local grid step parser (the real one is W6's scheduler)."""
    return int(step[:-1]) * _STEP_UNITS[step[-1]]


def make_spec(
    *, count: int = 4, seed: int = 42, step: str = "1d", name: str = "w7-spec"
) -> RunSpec:
    """Small synthetic RunSpec; float params survive the mint round-trip."""
    return RunSpec(
        name=name,
        vault_uri="vault://synthetic",
        decision_grid=DecisionGrid(start=T0, step=step, count=count),
        features=(
            FeatureDecl(name="lag_close", kind="vault_column_lag", params={"lag": 1}),
            FeatureDecl(name="agg", kind="vault_window_agg", params={"window": 3, "alpha": 0.5}),
        ),
        estimator="ewma_signal",
        estimator_params={"halflife": 2.0},
        seed=seed,
    )


def canonical_spec(spec: RunSpec) -> RunSpec:
    """The spec as the config sidecar round-trips it.

    Integration reconciliation (spec_sha256, ONE canonical form): the mint
    writes the UNROUNDED canonical json dump (``round_config=False`` in
    build_bundle), so the round-trip is a pure json serialize/parse — floats
    keep full double precision, no 12-digit rounding anywhere.
    """
    dumped = json.loads(canonical_json_bytes(spec.model_dump(mode="json")))
    return RunSpec.model_validate(dumped)


def build_trace(spec: RunSpec, *, mutate: tuple[int, str, str] | None = None) -> DecisionTrace:
    """Deterministic synthetic trace derived purely from the spec.

    ``mutate=(seq, field, value)`` rewrites one hash field of one row BEFORE
    chaining, modeling a re-execution that diverges from the stored trace.
    """
    step = parse_step(spec.decision_grid.step)
    rows: list[DecisionTraceRow] = []
    prev_hash = GENESIS_HASH
    for i in range(spec.decision_grid.count):
        moment = spec.decision_grid.start + i * step
        row = DecisionTraceRow(
            seq=i,
            decision_time=moment,
            known_at_ceiling=moment,
            data_manifest_sha256=sha256_hex_json(["data", spec.name, spec.seed, i]),
            feature_set_sha256=sha256_hex_json(["features", spec.name, spec.seed, i]),
            estimator_state_sha256=sha256_hex_json(["estimator", spec.estimator, spec.seed, i]),
            action_sha256=sha256_hex_json(["action", spec.name, spec.seed, i]),
            rng_counter_sha256=sha256_hex_json(["rng", spec.seed, i]),
            prev_row_sha256=prev_hash,
        )
        if mutate is not None and mutate[0] == i:
            row = row.model_copy(update={mutate[1]: mutate[2]})
        rows.append(row)
        prev_hash = trace_row_hash(row)
    return DecisionTrace(
        rows=tuple(rows),
        spec_sha256=sha256_hex_json(spec.model_dump(mode="json")),
        code_fingerprint="synthetic-code",
        env_fingerprint="synthetic-env",
        head_row_sha256=prev_hash if rows else GENESIS_HASH,
    )


def _write_fills(spec: RunSpec, tmp_bundle_dir: Path) -> None:
    """Deterministic synthetic fills parquet so metric recompute has input."""
    import io

    import polars as pl

    step = parse_step(spec.decision_grid.step)
    frame = pl.DataFrame(
        {
            "fill_time": [
                spec.decision_grid.start + i * step for i in range(spec.decision_grid.count)
            ],
            "nav": [1_000_000.0 * (1 + 0.001 * (i + 1)) for i in range(spec.decision_grid.count)],
        }
    ).with_columns(pl.col("fill_time").cast(pl.Datetime("us", "UTC")))
    buffer = io.BytesIO()
    frame.write_parquet(buffer)
    (tmp_bundle_dir / f"{spec.seed:x}.trades.parquet").write_bytes(buffer.getvalue())


def fake_executor(
    spec: RunSpec,
    vault: Any,
    tmp_bundle_dir: Path,
    *,
    mutate: tuple[int, str, str] | None = None,
    drop_last_row: bool = False,
    write_fills: bool = True,
) -> DecisionTrace:
    """Executor stand-in: re-derives the trace deterministically from the spec."""
    trace = build_trace(spec, mutate=mutate)
    if drop_last_row:
        trace = DecisionTrace(
            rows=trace.rows[:-1],
            spec_sha256=trace.spec_sha256,
            code_fingerprint=trace.code_fingerprint,
            env_fingerprint=trace.env_fingerprint,
            head_row_sha256=trace_row_hash(trace.rows[-1]) if trace.rows else GENESIS_HASH,
        )
    if write_fills:
        _write_fills(spec, tmp_bundle_dir)
    return trace


def mint_fixture(
    tmp_path: Path,
    *,
    spec: RunSpec | None = None,
    trace: DecisionTrace | None = None,
    env_doc: Any = None,
    seeds_doc: Any = None,
    run_spec_override: Any = None,
    trace_raw: bytes | None = None,
    env_raw: bytes | None = None,
    seeds_raw: bytes | None = None,
    config_mutator: Any = None,
) -> tuple[Any, Path, Path, DecisionTrace, RunSpec]:
    """Mint a synthetic wave-2 bundle dir, W6-layout (config["sidecars"]).

    Returns (bundle, bundle_dir, bundle_path, stored_trace, canonical_spec).
    Raw-byte overrides re-anchor the config commitments to the bytes actually
    written, modeling a consistent-but-wrong mint (or a malformed sidecar).
    """
    from quant_fund.proof.bundle import build_bundle

    spec_c = canonical_spec(spec or make_spec())
    stored_trace = trace if trace is not None else build_trace(spec_c)
    sidecar_bytes = {
        "trace": trace_raw
        if trace_raw is not None
        else canonical_json_bytes(stored_trace.model_dump(mode="json")),
        "env": env_raw
        if env_raw is not None
        else canonical_json_bytes(env_doc if env_doc is not None else replay.current_env_sidecar()),
        "seeds": seeds_raw
        if seeds_raw is not None
        else canonical_json_bytes(
            seeds_doc if seeds_doc is not None else expected_seeds_sidecar(spec_c)
        ),
    }
    config_dump: dict[str, Any] = {
        "run_spec": run_spec_override
        if run_spec_override is not None
        else spec_c.model_dump(mode="json"),
        "sidecars": {
            f"{kind}_sha256": sha256_hex_bytes(payload) for kind, payload in sidecar_bytes.items()
        },
    }
    if config_mutator is not None:
        config_dump = config_mutator(config_dump)
    bundle_dir = tmp_path / "proofs"
    recorder = InMemoryRecorder()
    recorder.record(
        make_read_record("silver/bars", T0, rows=3, content_sha256=sha256_hex_bytes(b"bars"))
    )
    recorder.record(
        make_read_record("gold/weights", T0, rows=3, content_sha256=sha256_hex_bytes(b"weights"))
    )
    bundle = build_bundle(
        run_kind="backtest",
        data_manifest=recorder.manifest_summary(),
        config_dump=config_dump,
        seed=spec_c.seed,
        signal_log=synthetic_signal_log(5),
        trade_log=synthetic_trade_log(5),
        engine_metrics={"total_return": 0.01, "label": "SYNTHETIC"},
        bundle_dir=bundle_dir,
    )
    bundle_id = bundle.bundle_id
    for kind, payload in sidecar_bytes.items():
        (bundle_dir / f"{bundle_id}.{kind}.json").write_bytes(payload)
    bundle_path = bundle_dir / "bundles" / f"{bundle_id}.json"
    return bundle, bundle_dir, bundle_path, stored_trace, spec_c


def _verdict(detail: str) -> ReplayVerdict:
    return ReplayVerdict.model_validate(json.loads(detail))


# ---------------------------------------------------------------------------
# Happy path + metric recomputation
# ---------------------------------------------------------------------------


def test_replay_identical(tmp_path: Path) -> None:
    bundle, bundle_dir, bundle_path, stored, spec = mint_fixture(tmp_path)
    ok, detail = replay_bundle(
        bundle_path, bundle_dir=bundle_dir, pit_root=None, executor=fake_executor
    )
    assert ok, detail
    verdict = _verdict(detail)
    assert verdict.status == "identical"
    assert verdict.reason is None
    assert verdict.first_divergence is None
    assert verdict.compared_rows == len(stored.rows) == spec.decision_grid.count
    assert verdict.bundle_id == bundle.bundle_id
    # Step 6: recomputed metric hashes come from the RE-EXECUTED fills.
    assert set(verdict.recomputed_metrics) == set(METRIC_KEYS)
    assert all(len(v) == 64 for v in verdict.recomputed_metrics.values())


def test_replay_never_trusts_stored_metrics_or_fills(tmp_path: Path) -> None:
    """§1.2: corrupting stored metric/trade sidecars cannot change the verdict."""
    bundle, bundle_dir, bundle_path, _, _ = mint_fixture(tmp_path)
    (bundle_dir / f"{bundle.bundle_id}.metrics.json").write_bytes(b'{"total_return": 999}')
    (bundle_dir / f"{bundle.bundle_id}.trades.parquet").write_bytes(b"garbage")
    ok, detail = replay_bundle(
        bundle_path, bundle_dir=bundle_dir, pit_root=None, executor=fake_executor
    )
    assert ok, detail
    verdict = _verdict(detail)
    assert verdict.status == "identical"
    assert set(verdict.recomputed_metrics) == set(METRIC_KEYS)


def test_replay_identical_without_fills(tmp_path: Path) -> None:
    """Trace-only executors (no trade log minted) yield empty metric hashes."""
    _, bundle_dir, bundle_path, stored, _ = mint_fixture(tmp_path)
    ok, detail = replay_bundle(
        bundle_path,
        bundle_dir=bundle_dir,
        pit_root=None,
        executor=lambda spec, vault, tmp: fake_executor(spec, vault, tmp, write_fills=False),
    )
    assert ok, detail
    verdict = _verdict(detail)
    assert verdict.status == "identical"
    assert verdict.recomputed_metrics == {}
    assert verdict.compared_rows == len(stored.rows)


# ---------------------------------------------------------------------------
# Step 1: sidecar tamper classes (config anchored by bundle; wave-2 sidecars
# anchored by the config's sidecars commitment map)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["trace", "env", "seeds", "config"])
def test_sidecar_tamper_detected(tmp_path: Path, kind: str) -> None:
    bundle, bundle_dir, bundle_path, _, _ = mint_fixture(tmp_path)
    target = bundle_dir / f"{bundle.bundle_id}.{kind}.json"
    target.write_bytes(target.read_bytes() + b" ")
    ok, detail = replay_bundle(
        bundle_path, bundle_dir=bundle_dir, pit_root=None, executor=fake_executor
    )
    assert not ok
    assert detail == f"sidecar_tampered:{kind}"


def test_sidecar_missing_detected(tmp_path: Path) -> None:
    bundle, bundle_dir, bundle_path, _, _ = mint_fixture(tmp_path)
    (bundle_dir / f"{bundle.bundle_id}.trace.json").unlink()
    ok, detail = replay_bundle(
        bundle_path, bundle_dir=bundle_dir, pit_root=None, executor=fake_executor
    )
    assert not ok
    assert detail == "sidecar_tampered:trace"


def test_wave1_bundle_fails_closed_with_legacy_wording(tmp_path: Path) -> None:
    """A wave-1 bundle (no run_spec/sidecars in config) keeps the stub's
    fail-closed tuple so the wave-1 regression battery stays green."""
    from tests.unit.proof_fake_vault import mint_synthetic_bundle

    bundle, bundle_dir = mint_synthetic_bundle(tmp_path)
    bundle_path = bundle_dir / "bundles" / f"{bundle.bundle_id}.json"
    ok, detail = replay_bundle(
        bundle_path, bundle_dir=bundle_dir, pit_root=None, executor=fake_executor
    )
    assert not ok
    assert detail == "runner_unavailable:per_decision_asof_not_implemented"


def test_sidecar_commitment_map_missing_fails_closed(tmp_path: Path) -> None:
    """A config sidecar without the sidecars commitment map cannot anchor
    the wave-2 sidecars — fail closed on the first kind checked."""
    _, bundle_dir, bundle_path, _, _ = mint_fixture(
        tmp_path, config_mutator=lambda c: {"run_spec": c["run_spec"]}
    )
    ok, detail = replay_bundle(
        bundle_path, bundle_dir=bundle_dir, pit_root=None, executor=fake_executor
    )
    assert not ok
    assert detail == "sidecar_tampered:trace"


def test_sidecar_commitment_entry_missing(tmp_path: Path) -> None:
    def drop_seeds(config: dict[str, Any]) -> dict[str, Any]:
        del config["sidecars"]["seeds_sha256"]
        return config

    _, bundle_dir, bundle_path, _, _ = mint_fixture(tmp_path, config_mutator=drop_seeds)
    ok, detail = replay_bundle(
        bundle_path, bundle_dir=bundle_dir, pit_root=None, executor=fake_executor
    )
    assert not ok
    assert detail == "sidecar_tampered:seeds"


def test_sidecar_commitment_value_not_a_digest(tmp_path: Path) -> None:
    def corrupt(config: dict[str, Any]) -> dict[str, Any]:
        config["sidecars"]["trace_sha256"] = "zz-not-hex"
        return config

    _, bundle_dir, bundle_path, _, _ = mint_fixture(tmp_path, config_mutator=corrupt)
    ok, detail = replay_bundle(
        bundle_path, bundle_dir=bundle_dir, pit_root=None, executor=fake_executor
    )
    assert not ok
    assert detail == "sidecar_tampered:trace"


def test_sidecar_commitments_accept_bare_kind_keys(tmp_path: Path) -> None:
    """The loader tolerates {'trace': sha, ...} as well as '<kind>_sha256'."""

    def bare(config: dict[str, Any]) -> dict[str, Any]:
        config["sidecars"] = {
            kind.removesuffix("_sha256"): value for kind, value in config["sidecars"].items()
        }
        return config

    _, bundle_dir, bundle_path, _, _ = mint_fixture(tmp_path, config_mutator=bare)
    ok, detail = replay_bundle(
        bundle_path, bundle_dir=bundle_dir, pit_root=None, executor=fake_executor
    )
    assert ok, detail


def test_config_sidecar_malformed_but_committed(tmp_path: Path) -> None:
    """Config bytes that hash-match the bundle but do not parse fail closed."""
    bundle, bundle_dir, bundle_path, _, _ = mint_fixture(tmp_path)
    garbage = b"not-json-at-all"
    (bundle_dir / f"{bundle.bundle_id}.config.json").write_bytes(garbage)
    payload = json.loads(bundle_path.read_bytes())
    payload["config_sha256"] = sha256_hex_bytes(garbage)
    bundle_path.write_bytes(canonical_json_bytes(payload))
    ok, detail = replay_bundle(
        bundle_path, bundle_dir=bundle_dir, pit_root=None, executor=fake_executor
    )
    assert not ok
    assert detail == "config_invalid:JSONDecodeError"


def test_config_sidecar_not_a_mapping(tmp_path: Path) -> None:
    bundle, bundle_dir, bundle_path, _, _ = mint_fixture(tmp_path)
    payload_bytes = canonical_json_bytes(["not", "a", "mapping"])
    (bundle_dir / f"{bundle.bundle_id}.config.json").write_bytes(payload_bytes)
    payload = json.loads(bundle_path.read_bytes())
    payload["config_sha256"] = sha256_hex_bytes(payload_bytes)
    bundle_path.write_bytes(canonical_json_bytes(payload))
    ok, detail = replay_bundle(
        bundle_path, bundle_dir=bundle_dir, pit_root=None, executor=fake_executor
    )
    assert not ok
    assert detail == "config_invalid:not_a_mapping"


def test_seeds_derivation_mismatch(tmp_path: Path) -> None:
    """A seeds sidecar that re-hashes cleanly against the config commitment
    but does not reproduce from the spec seed is still tampering (§2.5)."""
    _, bundle_dir, bundle_path, _, spec = mint_fixture(
        tmp_path, seeds_doc={"seed": 999, "window_seeds": {"0": window_seed_sha256(999, 0)}}
    )
    assert spec.seed != 999
    ok, detail = replay_bundle(
        bundle_path, bundle_dir=bundle_dir, pit_root=None, executor=fake_executor
    )
    assert not ok
    assert detail == "sidecar_tampered:seeds"


def test_seeds_sidecar_malformed_json(tmp_path: Path) -> None:
    _, bundle_dir, bundle_path, _, _ = mint_fixture(tmp_path, seeds_raw=b"not-json")
    ok, detail = replay_bundle(
        bundle_path, bundle_dir=bundle_dir, pit_root=None, executor=fake_executor
    )
    assert not ok
    assert detail == "sidecar_tampered:seeds"


def test_trace_sidecar_invalid_schema(tmp_path: Path) -> None:
    _, bundle_dir, bundle_path, _, _ = mint_fixture(
        tmp_path, trace_raw=canonical_json_bytes({"rows": []})
    )
    ok, detail = replay_bundle(
        bundle_path, bundle_dir=bundle_dir, pit_root=None, executor=fake_executor
    )
    assert not ok
    assert detail.startswith("trace_invalid:")


def test_trace_chain_break_fails_closed(tmp_path: Path) -> None:
    spec = make_spec()
    base = build_trace(canonical_spec(spec))
    # Break the chain post-hoc (row 1's prev no longer matches row 0's hash).
    rows = list(base.rows)
    rows[1] = rows[1].model_copy(update={"prev_row_sha256": "f" * 64})
    broken = DecisionTrace(
        rows=tuple(rows),
        spec_sha256=base.spec_sha256,
        code_fingerprint=base.code_fingerprint,
        env_fingerprint=base.env_fingerprint,
        head_row_sha256=base.head_row_sha256,
    )
    # Commitments re-anchor to the broken trace: hash check passes, chain check fires.
    _, bundle_dir, bundle_path, _, _ = mint_fixture(tmp_path, spec=spec, trace=broken)
    ok, detail = replay_bundle(
        bundle_path, bundle_dir=bundle_dir, pit_root=None, executor=fake_executor
    )
    assert not ok
    assert detail.startswith("trace_chain_invalid:")


# ---------------------------------------------------------------------------
# Step 2: spec/config anchoring
# ---------------------------------------------------------------------------


def test_spec_config_mismatch(tmp_path: Path) -> None:
    """trace.spec_sha256 inconsistent with the config sidecar fails closed."""
    spec = make_spec()
    other = canonical_spec(make_spec(name="other-spec"))
    base = build_trace(canonical_spec(spec))
    bad_trace = DecisionTrace(
        rows=base.rows,
        spec_sha256=sha256_hex_json(other.model_dump(mode="json")),
        code_fingerprint="synthetic-code",
        env_fingerprint="synthetic-env",
        head_row_sha256=base.head_row_sha256,
    )
    _, bundle_dir, bundle_path, _, _ = mint_fixture(tmp_path, spec=spec, trace=bad_trace)
    ok, detail = replay_bundle(
        bundle_path, bundle_dir=bundle_dir, pit_root=None, executor=fake_executor
    )
    assert not ok
    assert detail == "spec_mismatch:trace_spec_sha256"


def test_run_spec_missing_from_config(tmp_path: Path) -> None:
    def drop_spec(config: dict[str, Any]) -> dict[str, Any]:
        del config["run_spec"]
        return config

    _, bundle_dir, bundle_path, _, _ = mint_fixture(tmp_path, config_mutator=drop_spec)
    ok, detail = replay_bundle(
        bundle_path, bundle_dir=bundle_dir, pit_root=None, executor=fake_executor
    )
    assert not ok
    assert detail.startswith("config_spec_invalid:")


def test_run_spec_not_a_spec(tmp_path: Path) -> None:
    _, bundle_dir, bundle_path, _, _ = mint_fixture(
        tmp_path, run_spec_override={"not": "a run spec"}
    )
    ok, detail = replay_bundle(
        bundle_path, bundle_dir=bundle_dir, pit_root=None, executor=fake_executor
    )
    assert not ok
    assert detail.startswith("config_spec_invalid:")


# ---------------------------------------------------------------------------
# Step 3: environment gate
# ---------------------------------------------------------------------------


def test_env_mismatch_fingerprint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bundle minted under one env fingerprint, replayed under another."""
    _, bundle_dir, bundle_path, _, _ = mint_fixture(tmp_path)
    monkeypatch.setattr(replay, "_current_env_fingerprint", lambda: "forced|other|env")
    monkeypatch.setattr(replay, "_current_env_fingerprint_contracts", lambda: "forced|other|env")
    ok, detail = replay_bundle(
        bundle_path, bundle_dir=bundle_dir, pit_root=None, executor=fake_executor
    )
    assert not ok
    verdict = _verdict(detail)
    assert verdict.status == "unavailable"
    assert verdict.reason == "env_mismatch"
    assert verdict.compared_rows == 0


def test_env_mismatch_code_fingerprint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, bundle_dir, bundle_path, _, _ = mint_fixture(tmp_path)
    monkeypatch.setattr(replay, "_current_code_fingerprint", lambda: "e" * 40)
    ok, detail = replay_bundle(
        bundle_path, bundle_dir=bundle_dir, pit_root=None, executor=fake_executor
    )
    assert not ok
    assert _verdict(detail).reason == "env_mismatch"


def test_env_mismatch_python_tag(tmp_path: Path) -> None:
    env_doc = replay.current_env_sidecar()
    env_doc["python_tag"] = "CPython9.9.9"
    _, bundle_dir, bundle_path, _, _ = mint_fixture(tmp_path, env_doc=env_doc)
    ok, detail = replay_bundle(
        bundle_path, bundle_dir=bundle_dir, pit_root=None, executor=fake_executor
    )
    assert not ok
    verdict = _verdict(detail)
    assert verdict.status == "unavailable"
    assert verdict.reason == "env_mismatch"


def test_env_mismatch_package_pin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, bundle_dir, bundle_path, _, _ = mint_fixture(tmp_path)
    pins = replay._current_package_pins()
    monkeypatch.setattr(
        replay, "_current_package_pins", lambda: {**pins, "numpy": "0.0.0-tampered"}
    )
    ok, detail = replay_bundle(
        bundle_path, bundle_dir=bundle_dir, pit_root=None, executor=fake_executor
    )
    assert not ok
    assert _verdict(detail).reason == "env_mismatch"


def test_env_sidecar_not_a_mapping_is_unavailable(tmp_path: Path) -> None:
    _, bundle_dir, bundle_path, _, _ = mint_fixture(
        tmp_path, env_raw=canonical_json_bytes(["not", "a", "mapping"])
    )
    ok, detail = replay_bundle(
        bundle_path, bundle_dir=bundle_dir, pit_root=None, executor=fake_executor
    )
    assert not ok
    assert _verdict(detail).reason == "env_mismatch"


def test_env_sidecar_malformed_json(tmp_path: Path) -> None:
    _, bundle_dir, bundle_path, _, _ = mint_fixture(tmp_path, env_raw=b"not-json")
    ok, detail = replay_bundle(
        bundle_path, bundle_dir=bundle_dir, pit_root=None, executor=fake_executor
    )
    assert not ok
    assert _verdict(detail).reason == "env_mismatch"


def test_env_packages_not_a_mapping_ignored(tmp_path: Path) -> None:
    env_doc = replay.current_env_sidecar()
    env_doc["packages"] = "unpinned"
    _, bundle_dir, bundle_path, _, _ = mint_fixture(tmp_path, env_doc=env_doc)
    ok, detail = replay_bundle(
        bundle_path, bundle_dir=bundle_dir, pit_root=None, executor=fake_executor
    )
    assert ok, detail


# ---------------------------------------------------------------------------
# Step 5: divergence reporting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", sorted(ROW_HASH_FIELDS))
def test_one_field_mutation_diverges(tmp_path: Path, field: str) -> None:
    seq, mutated = 2, sha256_hex_json(["tampered", field])
    _, bundle_dir, bundle_path, stored, _ = mint_fixture(tmp_path)

    def executor(spec: RunSpec, vault: Any, tmp: Path) -> DecisionTrace:
        return fake_executor(spec, vault, tmp, mutate=(seq, field, mutated))

    ok, detail = replay_bundle(bundle_path, bundle_dir=bundle_dir, pit_root=None, executor=executor)
    assert not ok
    verdict = _verdict(detail)
    assert verdict.status == "diverged"
    assert verdict.reason == "trace_divergence"
    assert verdict.first_divergence is not None
    assert verdict.first_divergence.seq == seq
    assert verdict.first_divergence.field == field
    assert verdict.first_divergence.expected_sha256 == getattr(stored.rows[seq], field)
    assert verdict.first_divergence.actual_sha256 == mutated
    assert verdict.compared_rows == seq


def test_row_count_divergence(tmp_path: Path) -> None:
    _, bundle_dir, bundle_path, stored, _ = mint_fixture(tmp_path)

    def executor(spec: RunSpec, vault: Any, tmp: Path) -> DecisionTrace:
        return fake_executor(spec, vault, tmp, drop_last_row=True)

    ok, detail = replay_bundle(bundle_path, bundle_dir=bundle_dir, pit_root=None, executor=executor)
    assert not ok
    verdict = _verdict(detail)
    assert verdict.status == "diverged"
    assert verdict.first_divergence is not None
    assert verdict.first_divergence.field == "row_count"
    assert verdict.first_divergence.seq == len(stored.rows) - 1


# ---------------------------------------------------------------------------
# Step 4: executor failure modes + bundle loading
# ---------------------------------------------------------------------------


def test_default_executor_unavailable_on_wave2_base(tmp_path: Path) -> None:
    """Without an injected executor, the default executor drives the real
    W6 runner; this fixture's spec has no label declaration, so the runner
    fails closed and replay maps the raise to runner_unavailable."""
    _, bundle_dir, bundle_path, _, _ = mint_fixture(tmp_path)
    ok, detail = replay_bundle(bundle_path, bundle_dir=bundle_dir, pit_root=None)
    assert not ok
    assert detail.startswith("runner_unavailable:")


def test_executor_proof_error(tmp_path: Path) -> None:
    _, bundle_dir, bundle_path, _, _ = mint_fixture(tmp_path)

    def executor(spec: RunSpec, vault: Any, tmp: Path) -> DecisionTrace:
        raise ProofError("nan_in_window")

    ok, detail = replay_bundle(bundle_path, bundle_dir=bundle_dir, pit_root=None, executor=executor)
    assert not ok
    assert detail == "replay_reexecute_failed:nan_in_window"


def test_executor_unexpected_error(tmp_path: Path) -> None:
    _, bundle_dir, bundle_path, _, _ = mint_fixture(tmp_path)

    def executor(spec: RunSpec, vault: Any, tmp: Path) -> DecisionTrace:
        raise RuntimeError("boom")

    ok, detail = replay_bundle(bundle_path, bundle_dir=bundle_dir, pit_root=None, executor=executor)
    assert not ok
    assert detail == "replay_reexecute_error:RuntimeError"


def test_executor_returns_invalid_trace(tmp_path: Path) -> None:
    _, bundle_dir, bundle_path, _, _ = mint_fixture(tmp_path)
    ok, detail = replay_bundle(
        bundle_path,
        bundle_dir=bundle_dir,
        pit_root=None,
        executor=lambda spec, vault, tmp: {"rows": []},  # type: ignore[return-value]
    )
    assert not ok
    assert detail.startswith("executor_trace_invalid:")


def test_bundle_unreadable(tmp_path: Path) -> None:
    ok, detail = replay_bundle(
        tmp_path / "nope.json",
        bundle_dir=tmp_path,
        pit_root=None,
        executor=fake_executor,
    )
    assert not ok
    assert detail.startswith("bundle_unreadable:")


def test_bundle_invalid_schema(tmp_path: Path) -> None:
    bundle_path = tmp_path / "bad.json"
    bundle_path.write_bytes(canonical_json_bytes({"schema_version": "proofcore/1"}))
    ok, detail = replay_bundle(
        bundle_path, bundle_dir=tmp_path, pit_root=None, executor=fake_executor
    )
    assert not ok
    assert detail.startswith("bundle_invalid:")


# ---------------------------------------------------------------------------
# Helpers + env probes + default executor paths
# ---------------------------------------------------------------------------


def test_derive_window_seeds_matches_spec_commitment() -> None:
    spec = make_spec(count=3, seed=7)
    seeds = derive_window_seeds(spec)
    assert list(seeds) == ["0", "1", "2"]
    assert seeds["1"] == window_seed_sha256(7, 1)
    assert seeds["1"] != window_seed_sha256(7, 0)
    assert expected_seeds_sidecar(spec) == {"seed": 7, "window_seeds": seeds}


def test_env_probe_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Probe failures degrade to deterministic sentinels, never crash."""
    import importlib.metadata

    from quant_fund.proofcore import ci

    # Version probe (integration: importlib.metadata over the fx-1 dist —
    # layering forbids importing the quant_fund root from proofcore).
    def _no_dist(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(ci.importlib.metadata, "version", _no_dist)
    assert ci.quant_fund_version() == "dev"
    assert replay._current_env_fingerprint_contracts().endswith("|dev")
    monkeypatch.undo()
    assert ci.quant_fund_version() != "quant_fund-dev"  # real version or dev

    # Code fingerprint delegates to proofcore.ci (integration reconciliation);
    # the git-missing fallback path itself is covered by W8's
    # test_fingerprint_fallback.py.
    monkeypatch.setattr(ci, "code_fingerprint", lambda: "nogit")
    assert replay._current_code_fingerprint() == "nogit"

    def _missing(name: str) -> str:
        raise replay.importlib_metadata.PackageNotFoundError(name)

    monkeypatch.setattr(replay.importlib_metadata, "version", _missing)
    pins = replay._current_package_pins()
    assert set(pins) == {"numpy", "pandas", "polars"}
    assert all(v == "unknown" for v in pins.values())


def test_default_executor_no_entry_point(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import quant_fund.proof.runner as runner_mod

    monkeypatch.delattr(runner_mod, "run_proven", raising=False)
    _, bundle_dir, bundle_path, _, _ = mint_fixture(tmp_path)
    ok, detail = replay_bundle(bundle_path, bundle_dir=bundle_dir, pit_root=None)
    assert not ok
    assert detail == "runner_unavailable:quant_fund.proof.runner has no run_proven entry point"


def test_code_fingerprint_delegates_to_proofcore_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    """Integration reconciliation: replay's code-fingerprint probe is the
    SAME helper the runner mints with (git sha, or the W8 §7.2 src-tree
    fallback outside worktrees — covered by test_fingerprint_fallback.py)."""
    from quant_fund.proofcore import ci

    sentinel = "deadbee" + "0" * 33
    monkeypatch.setattr(ci, "code_fingerprint", lambda: sentinel)
    assert replay._current_code_fingerprint() == sentinel


def test_env_fingerprint_dual_accept_dev_shim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compat shim (expires wave 3): bundles minted with the W6 dev literal
    ``...|quant_fund-dev`` still pass the env gate alongside the canonical
    §2.2 form."""
    from quant_fund.proofcore import ci

    _, bundle_dir, bundle_path, _, _ = mint_fixture(
        tmp_path,
        env_doc={
            "env_fingerprint": "devmachine|3.12.0|quant_fund-dev",
            "code_fingerprint": ci.code_fingerprint(),
        },
    )
    monkeypatch.setattr(
        replay, "_current_env_fingerprint", lambda: "devmachine|3.12.0|quant_fund-dev"
    )
    monkeypatch.setattr(replay, "_current_env_fingerprint_contracts", ci.env_fingerprint)
    ok, detail = replay_bundle(
        bundle_path, bundle_dir=bundle_dir, pit_root=None, executor=fake_executor
    )
    assert ok, detail


def test_default_executor_runner_import_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    import quant_fund.proof as proof_pkg

    monkeypatch.delattr(proof_pkg, "runner", raising=False)
    monkeypatch.setitem(sys.modules, "quant_fund.proof.runner", None)  # import fails
    _, bundle_dir, bundle_path, _, _ = mint_fixture(tmp_path)
    ok, detail = replay_bundle(bundle_path, bundle_dir=bundle_dir, pit_root=None)
    assert not ok
    assert detail.startswith("runner_unavailable:runner_import:")


def test_default_executor_replay_unavailable_passthrough(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quant_fund.proof.runner as runner_mod

    def raising(spec: RunSpec, *, vault: Any, bundle_dir: Path) -> tuple[bool, str]:
        raise replay.ReplayUnavailable("explicit-unavailable")

    monkeypatch.setattr(runner_mod, "run_proven", raising, raising=False)
    _, bundle_dir, bundle_path, _, _ = mint_fixture(tmp_path)
    ok, detail = replay_bundle(bundle_path, bundle_dir=bundle_dir, pit_root=None)
    assert not ok
    assert detail == "runner_unavailable:explicit-unavailable"


def test_default_executor_run_proven_reports_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quant_fund.proof.runner as runner_mod

    monkeypatch.setattr(
        runner_mod,
        "run_proven",
        lambda spec, *, vault, bundle_dir: (False, "mint blew up"),
        raising=False,
    )
    _, bundle_dir, bundle_path, _, _ = mint_fixture(tmp_path)
    ok, detail = replay_bundle(bundle_path, bundle_dir=bundle_dir, pit_root=None)
    assert not ok
    assert detail == "runner_unavailable:mint blew up"


def test_default_executor_run_proven_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """W6's runner fails closed by RAISING (§4.4); replay maps it to unavailable."""
    import quant_fund.proof.runner as runner_mod

    def raising_run_proven(spec: RunSpec, *, vault: Any, bundle_dir: Path) -> tuple[bool, str]:
        raise ProofError("estimator_not_allowlisted: 'nope'")

    monkeypatch.setattr(runner_mod, "run_proven", raising_run_proven, raising=False)
    _, bundle_dir, bundle_path, _, _ = mint_fixture(tmp_path)
    ok, detail = replay_bundle(bundle_path, bundle_dir=bundle_dir, pit_root=None)
    assert not ok
    assert detail == "runner_unavailable:estimator_not_allowlisted: 'nope'"


def test_default_executor_fresh_trace_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quant_fund.proof.runner as runner_mod

    monkeypatch.setattr(
        runner_mod,
        "run_proven",
        lambda spec, *, vault, bundle_dir: (True, "ab" * 32),
        raising=False,
    )
    _, bundle_dir, bundle_path, _, _ = mint_fixture(tmp_path)
    ok, detail = replay_bundle(bundle_path, bundle_dir=bundle_dir, pit_root=None)
    assert not ok
    assert detail.startswith("runner_unavailable:fresh_trace_unreadable:")


def test_default_executor_end_to_end_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default executor path with a run_proven that mints a fresh trace."""
    import quant_fund.proof.runner as runner_mod

    def fake_run_proven(
        spec: RunSpec, *, vault: Any, bundle_dir: Path, signing_key: Any = None
    ) -> tuple[bool, str]:
        trace = build_trace(spec)
        fresh_id = "cd" * 32
        (Path(bundle_dir) / f"{fresh_id}.trace.json").write_bytes(
            canonical_json_bytes(trace.model_dump(mode="json"))
        )
        _write_fills(spec, Path(bundle_dir))
        return True, fresh_id

    monkeypatch.setattr(runner_mod, "run_proven", fake_run_proven, raising=False)
    _, bundle_dir, bundle_path, stored, _ = mint_fixture(tmp_path)
    ok, detail = replay_bundle(bundle_path, bundle_dir=bundle_dir, pit_root=None)
    assert ok, detail
    verdict = _verdict(detail)
    assert verdict.status == "identical"
    assert verdict.compared_rows == len(stored.rows)
    assert set(verdict.recomputed_metrics) == set(METRIC_KEYS)
