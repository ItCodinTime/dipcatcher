"""Wave 25: CPCV combinatorial_purged_cv extremes (complements test_validation / bench_cpcv_audit)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from math import comb

import pytest

from quant_fund.research.benches import bench_cpcv_audit
from quant_fund.validation.cpcv import combinatorial_purged_cv
from quant_fund.validation.walk_forward import assert_no_label_overlap, session_index


def _times(n: int, *, start: datetime | None = None) -> list[datetime]:
    t0 = start or datetime(2020, 1, 2, tzinfo=UTC)
    return [t0 + timedelta(days=i) for i in range(n)]


@pytest.mark.parametrize(
    "n_groups,n_test_groups",
    [
        (1, 1),
        (2, 0),
        (2, 2),
        (3, 3),
        (0, 1),
        (-1, 1),
        (4, -1),
    ],
)
def test_cpcv_invalid_group_counts_raise(n_groups: int, n_test_groups: int) -> None:
    with pytest.raises(ValueError, match="invalid CPCV group counts"):
        combinatorial_purged_cv(_times(60), n_groups, n_test_groups, horizon_bars=1, embargo_bars=0)


def test_cpcv_negative_horizon_or_embargo_raise() -> None:
    times = _times(40)
    with pytest.raises(ValueError, match="horizon_bars"):
        combinatorial_purged_cv(times, 4, 1, horizon_bars=-1, embargo_bars=0)
    with pytest.raises(ValueError, match="embargo_bars"):
        combinatorial_purged_cv(times, 4, 1, horizon_bars=0, embargo_bars=-2)


def test_cpcv_empty_dates_yield_no_folds() -> None:
    assert combinatorial_purged_cv([], 6, 2, horizon_bars=2, embargo_bars=2) == []


def test_cpcv_too_few_unique_dates_raise() -> None:
    # n_groups=6 with only 5 unique dates cannot form non-empty groups
    with pytest.raises(ValueError, match="at least n_groups"):
        combinatorial_purged_cv(_times(5), 6, 2, horizon_bars=1, embargo_bars=0)


def test_cpcv_fold_count_matches_combinations() -> None:
    times = _times(60)
    for n_groups, n_test in ((6, 2), (5, 2), (4, 1), (8, 3)):
        folds = combinatorial_purged_cv(times, n_groups, n_test, horizon_bars=2, embargo_bars=1)
        assert len(folds) == comb(n_groups, n_test)


def test_cpcv_train_test_disjoint_and_no_label_overlap() -> None:
    times = _times(120)
    horizon, embargo = 3, 2
    folds = combinatorial_purged_cv(
        times, n_groups=6, n_test_groups=2, horizon_bars=horizon, embargo_bars=embargo
    )
    idx = session_index(times)
    assert len(folds) == comb(6, 2)
    for f in folds:
        assert f.train_times, "purged train must be non-empty"
        assert f.test_times
        assert set(f.train_times).isdisjoint(set(f.test_times))
        assert_no_label_overlap(f, horizon, idx)


def test_cpcv_purge_embargo_geometry_adjacent_test_groups() -> None:
    """n=60, 6 groups of 10; test groups 0+1 → train starts at index 22 after h=3,e=2."""
    times = _times(60)
    folds = combinatorial_purged_cv(
        times, n_groups=6, n_test_groups=2, horizon_bars=3, embargo_bars=2
    )
    idx = session_index(times)
    target_test = set(times[0:20])
    matches = [f for f in folds if set(f.test_times) == target_test]
    assert len(matches) == 1
    train_idx = [idx[t] for t in matches[0].train_times]
    # Embargo after group1 (hi=19) drops 20,21; purge span does not drop later train.
    assert train_idx == list(range(22, 60))
    assert_no_label_overlap(matches[0], 3, idx)


def test_cpcv_embargo_clears_sessions_before_later_test_block() -> None:
    """Non-contiguous test groups: embargo before a later test group drops prior train."""
    times = _times(60)
    # groups of 10; pick groups 0 and 3 → test [0:10) U [30:40)
    folds = combinatorial_purged_cv(
        times, n_groups=6, n_test_groups=2, horizon_bars=0, embargo_bars=2
    )
    idx = session_index(times)
    target = set(times[0:10]) | set(times[30:40])
    matches = [f for f in folds if set(f.test_times) == target]
    assert len(matches) == 1
    train_idx = set(idx[t] for t in matches[0].train_times)
    # Embargo after g0 drops 10,11; before g3 drops 28,29; after g3 drops 40,41
    assert 10 not in train_idx and 11 not in train_idx
    assert 28 not in train_idx and 29 not in train_idx
    assert 40 not in train_idx and 41 not in train_idx
    # Complement groups 1,2,4,5 without embargo zones remain
    assert 12 in train_idx and 27 in train_idx and 42 in train_idx


def test_cpcv_dedupes_duplicate_timestamps() -> None:
    base = _times(60)
    duped = base + base[10:30]
    folds = combinatorial_purged_cv(
        duped, n_groups=6, n_test_groups=2, horizon_bars=1, embargo_bars=1
    )
    assert len(folds) == comb(6, 2)
    idx = session_index(base)
    for f in folds:
        assert len(f.train_times) == len(set(f.train_times))
        assert len(f.test_times) == len(set(f.test_times))
        assert_no_label_overlap(f, 1, idx)


def test_bench_cpcv_audit_integrity_flags() -> None:
    """Research audit is validation_integrity_only — no live P&L / Sharpe claim."""
    out = bench_cpcv_audit(n_dates=120, n_groups=6, n_test_groups=2, horizon_bars=2, embargo_bars=2)
    assert out["purge_embargo_valid"] is True
    assert out["n_folds"] == out["expected_folds"] == float(comb(6, 2))
    assert out["claim"] == "validation_integrity_only"
    assert out["date_level"] is True
    forbidden = {"sharpe", "sortino", "calmar", "pnl", "nav", "return"}
    assert forbidden.isdisjoint(out.keys())


def test_cpcv_aggressive_purge_may_yield_fewer_folds_than_combinations() -> None:
    """Huge horizon/embargo can wipe purged train → fold count < C(n_groups, n_test).

    Honest residual path: fewer (even zero) folds, never a silent claim of full
    combinatorial coverage. Complements empty-group / bad n_test ValueErrors.
    """
    times = _times(24)
    n_groups, n_test = 6, 2
    expected = comb(n_groups, n_test)
    full = combinatorial_purged_cv(times, n_groups, n_test, horizon_bars=0, embargo_bars=0)
    assert len(full) == expected

    reduced = combinatorial_purged_cv(times, n_groups, n_test, horizon_bars=10, embargo_bars=10)
    assert 0 < len(reduced) < expected

    wiped = combinatorial_purged_cv(times, n_groups, n_test, horizon_bars=20, embargo_bars=20)
    assert wiped == []
