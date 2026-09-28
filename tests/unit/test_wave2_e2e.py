"""Wave-2 integration end-to-end (WAVE2.md §9 done criteria).

Synthetic vault + RunSpec -> ``run_proven`` mints a signed bundle with
trace/env/seeds sidecars -> ``replay_bundle`` re-executes bit-exactly ->
``identical``. A late-arriving vault correction MUST flip replay to
``diverged`` with ``first_divergence`` at the first row whose watchdog-checked
reads changed; a raw byte-flip of a vault part is caught by the vault's own
manifest hash verification (fail closed). A sneaky raw ``pl.read_parquet``
inside a decision window raises ``LeakageError`` under the runner's enforce
IO guard. Finally, a RunSpec float param with >12 significant digits
round-trips mint -> replay identically (integration amendment I2: one
canonical ``spec_sha256``, no rounding anywhere).

All data is synthetic; correctness evidence only, never performance claims.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from quant_fund.leakage.watchdog import LeakageError
from quant_fund.pit.vault import PitVault
from quant_fund.proof.replay import replay_bundle
from quant_fund.proof.runner import run_proven
from quant_fund.proof.verify import verify_bundle
from quant_fund.proofcore.contracts import (
    DecisionGrid,
    FeatureDecl,
    ReplayVerdict,
    RunSpec,
    sha256_hex_json,
)

DATASET = "silver/bars"
T0 = datetime(2024, 1, 10, tzinfo=UTC)
DAY = timedelta(days=1)
SIGNING_KEY = b"wave2-e2e-key"

#: 17 significant digits — round(x, 12) would CHANGE it; the reconciled
#: spec_sha256 must keep it bit-exact through mint -> config -> replay.
LONG_FLOAT = 1.2345678901234567
assert round(LONG_FLOAT, 12) != LONG_FLOAT


def _bars_rows(n: int = 20) -> list[dict[str, object]]:
    return [
        {
            "event_time": datetime(2024, 1, 1, tzinfo=UTC) + i * DAY,
            "known_at": datetime(2024, 1, 1, tzinfo=UTC) + i * DAY,
            "close": 100.0 + i + (0.5 if i % 2 else -0.5),
        }
        for i in range(n)
    ]


def _build_vault(root: Path) -> PitVault:
    vault = PitVault(root)
    vault.create_dataset(DATASET, security_level=False)
    vault.append(DATASET, pl.DataFrame(_bars_rows()))
    return vault


def _spec(*, span: float = 3.0, count: int = 6, seed: int = 42) -> RunSpec:
    return RunSpec(
        name="wave2-e2e",
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
        ),
        estimator="ewma_signal",
        estimator_params={
            "span": span,
            "label": {"dataset": DATASET, "column": "close", "horizon": 1},
        },
        seed=seed,
    )


def _mint(tmp_path: Path, *, spec: RunSpec | None = None) -> tuple[PitVault, Path, str, Path]:
    vault = _build_vault(tmp_path / "pit")
    bundle_dir = tmp_path / "proofs"
    ok, bundle_id = run_proven(
        spec or _spec(), vault=vault, bundle_dir=bundle_dir, signing_key=SIGNING_KEY
    )
    assert ok
    return vault, bundle_dir, bundle_id, bundle_dir / "bundles" / f"{bundle_id}.json"


def _replay(bundle_dir: Path, bundle_id: str, vault_root: Path) -> ReplayVerdict:
    ok, detail = replay_bundle(
        bundle_dir / "bundles" / f"{bundle_id}.json",
        bundle_dir=bundle_dir,
        pit_root=vault_root,
        vault=PitVault(vault_root),
    )
    verdict = ReplayVerdict.model_validate(json.loads(detail))
    assert ok == (verdict.status == "identical"), detail
    return verdict


def test_mint_then_replay_is_identical(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PROOFCORE_SIGNING_KEY", SIGNING_KEY.decode())
    _, bundle_dir, bundle_id, bundle_path = _mint(tmp_path)

    # The wave-1 verifier accepts the signed bundle (strict signature)...
    result = verify_bundle(bundle_path, bundle_dir=bundle_dir, strict_signature=True)
    assert result.ok, result.reasons

    # ...and bit-exact replay of the same machine/vault is identical.
    verdict = _replay(bundle_dir, bundle_id, tmp_path / "pit")
    assert verdict.status == "identical"
    assert verdict.compared_rows == 6
    assert verdict.first_divergence is None


def test_long_float_spec_param_round_trips_bit_exact(tmp_path) -> None:
    """Amendment I2: spec_sha256 = sha256(canonical json dump), NO rounding.

    Under the pre-reconciliation mint (round_floats on the config sidecar)
    this spec's 17-significant-digit span would rebuild differently and replay
    would fail with spec_mismatch.
    """
    spec = _spec(span=LONG_FLOAT)
    assert sha256_hex_json(spec.model_dump(mode="json")) == sha256_hex_json(
        RunSpec.model_validate(spec.model_dump(mode="json")).model_dump(mode="json")
    )
    _, bundle_dir, bundle_id, _ = _mint(tmp_path, spec=spec)

    # The config sidecar carries the UNROUNDED spec dump.
    config = json.loads((bundle_dir / f"{bundle_id}.config.json").read_bytes())
    assert config["run_spec"]["estimator_params"]["span"] == LONG_FLOAT

    verdict = _replay(bundle_dir, bundle_id, tmp_path / "pit")
    assert verdict.status == "identical"


def test_vault_correction_diverges_at_first_affected_row(tmp_path) -> None:
    """A late-arriving correction changes reads from a specific window on.

    Correction: event_time 2024-01-09 (close 999.0) with known_at 2024-01-12.
    Windows 0-1 read only watermarks <= 2024-01-11 (invisible); window seq 2
    realizes the seq-1 label with a read at watermark 2024-01-12, where the
    correction first becomes visible -> data_manifest_sha256 diverges there.
    """
    vault, bundle_dir, bundle_id, _ = _mint(tmp_path)
    correction = pl.DataFrame(
        [
            {
                "event_time": datetime(2024, 1, 9, tzinfo=UTC),
                "known_at": datetime(2024, 1, 12, tzinfo=UTC),
                "close": 999.0,
            }
        ]
    )
    vault.append(DATASET, correction)

    verdict = _replay(bundle_dir, bundle_id, tmp_path / "pit")
    assert verdict.status == "diverged"
    assert verdict.reason == "trace_divergence"
    assert verdict.first_divergence is not None
    assert verdict.first_divergence.seq == 2
    assert verdict.first_divergence.field == "data_manifest_sha256"
    assert verdict.first_divergence.expected_sha256 != verdict.first_divergence.actual_sha256


def test_vault_part_byte_flip_fails_closed(tmp_path) -> None:
    """Flipping one byte in a committed vault part is caught by the vault's
    manifest hash verification during re-execution — replay must NOT report
    identical (vault integrity gate, defense in depth under the trace)."""
    _, bundle_dir, bundle_id, _ = _mint(tmp_path)
    part = next((tmp_path / "pit").rglob("parts/*.parquet"))
    blob = bytearray(part.read_bytes())
    blob[len(blob) // 2] ^= 0xFF
    part.write_bytes(bytes(blob))

    ok, detail = replay_bundle(
        bundle_dir / "bundles" / f"{bundle_id}.json",
        bundle_dir=bundle_dir,
        pit_root=tmp_path / "pit",
        vault=PitVault(tmp_path / "pit"),
    )
    # The vault's manifest hash check raises ManifestError (a ProofcoreError)
    # during re-execution; replay reports the runner unavailable, never a
    # verdict of identity.
    assert not ok
    assert detail.startswith("runner_unavailable:")


def test_sneaky_raw_parquet_read_inside_window_raises(tmp_path) -> None:
    """IO guard (WAVE2 §6, enforce mode): a raw pl.read_parquet of a path
    outside the vault inside a decision window raises LeakageError."""
    sneaky = tmp_path / "rogue.parquet"
    pl.DataFrame({"x": [1.0]}).write_parquet(sneaky)

    class SneakyVault(PitVault):
        def asof(self, name, t, **kwargs):
            pl.read_parquet(sneaky)  # rogue second source inside the window
            return super().asof(name, t, **kwargs)

    vault = SneakyVault(tmp_path / "pit")
    vault.create_dataset(DATASET, security_level=False)
    vault.append(DATASET, pl.DataFrame(_bars_rows()))
    with pytest.raises(LeakageError, match="io_guard"):
        run_proven(_spec(), vault=vault, bundle_dir=tmp_path / "proofs")
    assert not (tmp_path / "proofs").exists()


def test_runner_works_when_io_guard_unavailable(tmp_path, monkeypatch) -> None:
    """Defense-in-depth optionality: an unimportable guard logs once and the
    runner still mints (watchdog + declared-surface checks remain active)."""
    import quant_fund.proof.runner as runner_mod

    monkeypatch.setitem(sys.modules, "quant_fund.leakage.guard", None)  # import fails
    monkeypatch.setattr(runner_mod, "_GUARD_UNAVAILABLE_LOGGED", False)
    vault, bundle_dir = _build_vault(tmp_path / "pit"), tmp_path / "proofs"
    ok, bundle_id = run_proven(_spec(), vault=vault, bundle_dir=bundle_dir)
    assert ok
    assert (bundle_dir / f"{bundle_id}.trace.json").is_file()
    assert runner_mod._GUARD_UNAVAILABLE_LOGGED


def test_tampered_wave2_sidecars_fail_verify(tmp_path) -> None:
    """Task-2 contract: verify.py hash-checks trace/env/seeds via the config
    sidecar commitment (check 6b, mirroring the wave-1 config pattern)."""
    _, bundle_dir, bundle_id, bundle_path = _mint(tmp_path)

    assert verify_bundle(bundle_path, bundle_dir=bundle_dir, strict_signature=False).ok

    # Flip one byte in the trace sidecar.
    trace_path = bundle_dir / f"{bundle_id}.trace.json"
    blob = bytearray(trace_path.read_bytes())
    blob[-3] ^= 0x01
    trace_path.write_bytes(bytes(blob))
    result = verify_bundle(bundle_path, bundle_dir=bundle_dir, strict_signature=False)
    assert not result.ok
    assert "sidecar:trace_sha256_mismatch" in result.reasons

    # Fresh bundle; delete the seeds sidecar -> missing.
    _, bundle_dir2, bundle_id2, bundle_path2 = _mint(tmp_path / "second")
    (bundle_dir2 / f"{bundle_id2}.seeds.json").unlink()
    result2 = verify_bundle(bundle_path2, bundle_dir=bundle_dir2, strict_signature=False)
    assert not result2.ok
    assert "sidecar:seeds:missing" in result2.reasons


def test_hostile_commitment_key_rejected(tmp_path) -> None:
    """Check 6b sanitizes commitment keys: a ``..``-laden or non-hex
    commitment is an invalid_commitment failure, never a path traversal."""
    from quant_fund.proof.bundle import build_bundle
    from quant_fund.proof.recorder import InMemoryRecorder
    from tests.unit.proof_fake_vault import synthetic_signal_log, synthetic_trade_log

    bundle_dir = tmp_path / "proofs"
    bundle = build_bundle(
        run_kind="backtest",
        data_manifest=InMemoryRecorder().manifest_summary(),
        config_dump={
            "run_spec": {},
            "sidecars": {"../escape_sha256": "a" * 64, "trace_sha256": "b" * 64},
        },
        seed=1,
        signal_log=synthetic_signal_log(3),
        trade_log=synthetic_trade_log(3),
        engine_metrics={"n": 1.0},
        bundle_dir=bundle_dir,
    )
    result = verify_bundle(
        bundle_dir / "bundles" / f"{bundle.bundle_id}.json",
        bundle_dir=bundle_dir,
        strict_signature=False,
    )
    assert not result.ok
    assert "sidecar:../escape:invalid_commitment" in result.reasons
    assert "sidecar:trace:missing" in result.reasons
