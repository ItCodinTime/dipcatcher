"""Property tests: replay determinism and divergence localization (WAVE2.md §5).

Determinism environment (CI pins these; set here too for local runs):
TZ=UTC, LC_ALL=C, PYTHONHASHSEED=0 (hash seed must be fixed at interpreter
start — CI exports it; nothing here relies on hash() ordering anyway).

For random RunSpec grids (3-8 windows) the fake executor deterministically
derives a synthetic trace from the spec, so run-then-replay is ALWAYS
identical. Mutating one hash field of one re-executed row MUST flip the
verdict to ``diverged`` with ``first_divergence.seq`` at the mutated row.

All data is synthetic; no performance claims are made.
"""

from __future__ import annotations

import os

os.environ.setdefault("TZ", "UTC")
os.environ.setdefault("LC_ALL", "C")

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from quant_fund.proof.replay import ROW_HASH_FIELDS, replay_bundle
from quant_fund.proofcore.contracts import (
    DecisionGrid,
    FeatureDecl,
    ReplayVerdict,
    RunSpec,
    sha256_hex_json,
)
from tests.unit.test_replay_engine import fake_executor, mint_fixture

_SETTINGS = settings(
    max_examples=25,
    derandomize=True,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.too_slow],
)

T0 = datetime(2024, 1, 1, tzinfo=UTC)

_names = st.text(alphabet="abcdefgh", min_size=1, max_size=10)
_kinds = st.sampled_from(["vault_column_lag", "vault_window_agg", "prior_decision_state"])
_params = st.fixed_dictionaries(
    {
        "lag": st.integers(min_value=1, max_value=5),
        "scale": st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    }
)
_specs = st.builds(
    lambda name, step, count, feats, est, est_params, seed: RunSpec(
        name=name,
        vault_uri="vault://property-synthetic",
        decision_grid=DecisionGrid(start=T0, step=step, count=count),
        features=feats,
        estimator=est,
        estimator_params=est_params,
        seed=seed,
    ),
    name=_names,
    step=st.sampled_from(["1d", "2d", "6h", "30m", "90s"]),
    count=st.integers(min_value=3, max_value=8),
    feats=st.lists(
        st.builds(FeatureDecl, name=_names, kind=_kinds, params=_params),
        min_size=1,
        max_size=3,
    ).map(tuple),
    est=st.sampled_from(["ewma_signal", "linear_regression_np"]),
    est_params=st.fixed_dictionaries({"halflife": st.floats(min_value=0.5, max_value=9.0)}),
    seed=st.integers(min_value=0, max_value=10**6),
)


def _fresh_dir(tmp_path: Path) -> Path:
    """Hypothesis reuses the function-scoped tmp_path across examples."""
    return Path(tempfile.mkdtemp(dir=tmp_path))


@_SETTINGS
@given(spec=_specs)
def test_run_then_replay_is_always_identical(tmp_path: Path, spec: RunSpec) -> None:
    """Determinism: the same spec re-executes bit-exactly, every time."""
    _, bundle_dir, bundle_path, stored, _ = mint_fixture(_fresh_dir(tmp_path), spec=spec)
    ok, detail = replay_bundle(
        bundle_path, bundle_dir=bundle_dir, pit_root=None, executor=fake_executor
    )
    assert ok, detail
    verdict = ReplayVerdict.model_validate(json.loads(detail))
    assert verdict.status == "identical"
    assert verdict.compared_rows == len(stored.rows) == spec.decision_grid.count
    assert verdict.first_divergence is None


@_SETTINGS
@given(spec=_specs, data=st.data())
def test_single_row_mutation_must_diverge_at_that_row(
    tmp_path: Path, spec: RunSpec, data: st.DataObject
) -> None:
    """Any single-field mutation is caught, localized at the mutated row."""
    _, bundle_dir, bundle_path, stored, _ = mint_fixture(_fresh_dir(tmp_path), spec=spec)
    seq = data.draw(st.integers(min_value=0, max_value=spec.decision_grid.count - 1))
    field = data.draw(st.sampled_from(ROW_HASH_FIELDS))
    mutated = sha256_hex_json(["mutated", spec.name, spec.seed, seq, field])
    assert mutated != getattr(stored.rows[seq], field)

    def executor(s: RunSpec, vault: object, tmp: Path):  # noqa: ANN202
        return fake_executor(s, vault, tmp, mutate=(seq, field, mutated))

    ok, detail = replay_bundle(bundle_path, bundle_dir=bundle_dir, pit_root=None, executor=executor)
    assert not ok, "a mutated re-execution must never verify as identical"
    verdict = ReplayVerdict.model_validate(json.loads(detail))
    assert verdict.status == "diverged"
    assert verdict.first_divergence is not None
    assert verdict.first_divergence.seq == seq
    assert verdict.first_divergence.field == field
    assert verdict.first_divergence.expected_sha256 == getattr(stored.rows[seq], field)
    assert verdict.first_divergence.actual_sha256 == mutated
