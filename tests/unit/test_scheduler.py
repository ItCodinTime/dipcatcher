"""Scheduler unit tests (WAVE2.md §3). Synthetic specs only."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from quant_fund.proofcore.contracts import (
    DecisionGrid,
    FeatureDecl,
    ProofError,
    RunSpec,
)
from quant_fund.proofcore.scheduler import DecisionWindow, Scheduler, parse_step, window_seed

T0 = datetime(2024, 1, 10, 9, 30, tzinfo=UTC)


def _spec(step: str = "1d", count: int = 5, seed: int = 42, start: datetime = T0) -> RunSpec:
    return RunSpec(
        name="sched-test",
        vault_uri="vault://main",
        decision_grid=DecisionGrid(start=start, step=step, count=count),
        features=(
            FeatureDecl(
                name="lag1",
                kind="vault_column_lag",
                params={"dataset": "silver/bars", "column": "close", "lag": 1},
            ),
        ),
        estimator="ewma_signal",
        estimator_params={"label": {"dataset": "silver/bars", "column": "close", "horizon": 1}},
        seed=seed,
    )


# -- parse_step ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1d", timedelta(days=1)),
        ("7d", timedelta(days=7)),
        ("1h", timedelta(hours=1)),
        ("12h", timedelta(hours=12)),
        ("1m", timedelta(minutes=1)),
        ("30m", timedelta(minutes=30)),
        ("1s", timedelta(seconds=1)),
        ("90s", timedelta(seconds=90)),
    ],
)
def test_parse_step_valid(text: str, expected: timedelta) -> None:
    assert parse_step(text) == expected


@pytest.mark.parametrize(
    "text",
    ["0d", "0s", "-1d", "1.5d", "1w", "1D", "d", "", "1", "1dd", " 1d", "1d ", "01d"],
)
def test_parse_step_rejects_illegal(text: str) -> None:
    with pytest.raises(ProofError, match="step"):
        parse_step(text)


def test_parse_step_rejects_non_string() -> None:
    with pytest.raises(ProofError, match="string"):
        parse_step(None)  # type: ignore[arg-type]


# -- Scheduler ---------------------------------------------------------------


def test_scheduler_windows_are_strict_ordered_grid() -> None:
    scheduler = Scheduler(_spec(step="1d", count=5))
    windows = list(scheduler)
    assert len(scheduler) == 5
    assert len(windows) == 5
    for seq, window in enumerate(windows):
        assert isinstance(window, DecisionWindow)
        assert window.seq == seq
        assert window.decision_time == T0 + timedelta(days=seq)
        assert window.prior_times == tuple(T0 + timedelta(days=i) for i in range(seq))


def test_scheduler_frozen_windows() -> None:
    window = Scheduler(_spec()).window_at(0)
    with pytest.raises(AttributeError):
        window.seq = 99  # type: ignore[misc]


def test_scheduler_determinism_same_spec_same_windows() -> None:
    spec = _spec()
    first = list(Scheduler(spec))
    second = list(Scheduler(spec))
    assert first == second
    assert [w.seed for w in first] == [w.seed for w in second]


def test_window_seed_derivation_matches_spec_formula() -> None:
    import hashlib

    expected = int.from_bytes(hashlib.sha256(b"42|3").digest()[:8], "big")
    assert window_seed(42, 3) == expected
    assert Scheduler(_spec(seed=42)).window_at(3).seed == expected


def test_window_seed_depends_on_seed_and_seq() -> None:
    seeds_a = [w.seed for w in Scheduler(_spec(seed=1))]
    seeds_b = [w.seed for w in Scheduler(_spec(seed=2))]
    assert seeds_a != seeds_b
    assert len(set(seeds_a)) == len(seeds_a)


@pytest.mark.parametrize("bad_seq", [-1, 5, 100])
def test_window_at_bounds_proof(bad_seq: int) -> None:
    with pytest.raises(ProofError, match="out of bounds"):
        Scheduler(_spec(count=5)).window_at(bad_seq)


def test_window_at_rejects_non_int() -> None:
    with pytest.raises(ProofError, match="int"):
        Scheduler(_spec()).window_at(1.5)  # type: ignore[arg-type]
    with pytest.raises(ProofError, match="int"):
        Scheduler(_spec()).window_at(True)  # type: ignore[arg-type]


def test_scheduler_rejects_naive_grid_start() -> None:
    naive = datetime(2024, 1, 10, 9, 30)
    with pytest.raises(ValidationError):
        _spec(start=naive)


def test_scheduler_rejects_bad_step_at_construction() -> None:
    spec = _spec(step="1w")
    with pytest.raises(ProofError, match="step"):
        Scheduler(spec)


def test_scheduler_count_one_window_has_no_priors() -> None:
    scheduler = Scheduler(_spec(count=1))
    window = scheduler.window_at(0)
    assert window.prior_times == ()
    assert window.decision_time == T0


def test_scheduler_properties_and_naive_defense() -> None:
    scheduler = Scheduler(_spec(step="2h", count=3, seed=9))
    assert scheduler.spec.seed == 9
    assert scheduler.step == timedelta(hours=2)
    # Defense in depth: a spec that bypasses pydantic validation still fails.
    from quant_fund.proofcore.contracts import DecisionGrid as _Grid

    rogue_grid = _Grid.model_construct(start=datetime(2024, 1, 10), step="1d", count=2)
    rogue = RunSpec.model_construct(decision_grid=rogue_grid)
    with pytest.raises(ProofError, match="timezone-aware"):
        Scheduler(rogue)
