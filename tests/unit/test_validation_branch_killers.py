"""Mutmut kill tests: boundary semantics of purge / embargo / CPCV / walk-forward.

Every assertion pins a fail-closed edge: the session-index arithmetic, the
inclusive holdout window, the ``(i, i+horizon]`` label interval, and the
delay-1 embargo tail.  A mutant that flips ``>=`` to ``>`` on an embargo or
purge boundary must die here.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from quant_fund.config.models import ValidationConfig
from quant_fund.validation.cpcv import (
    combinatorial_purged_cv,
    combinatorial_purged_indices,
    cpcv_n_paths,
    cpcv_n_splits,
    cpcv_path_assignments,
    stitch_group_paths,
)
from quant_fund.validation.embargo import embargo_mask
from quant_fund.validation.purging import overlaps, purge_mask
from quant_fund.validation.walk_forward import (
    Fold,
    assert_no_label_overlap,
    fold_ic_stability,
    row_mask_for_times,
    session_index,
    timestamp_ns,
    walk_forward,
)


def _times(n: int, *, start: datetime | None = None) -> list[datetime]:
    t0 = start or datetime(2021, 1, 4, tzinfo=UTC)
    return [t0 + timedelta(days=i) for i in range(n)]


# ------------------------------- purging.overlaps -------------------------------


def test_overlaps_label_end_on_test_start_boundary() -> None:
    """label_end == test_start intersects: ``test_start <= label_end`` is inclusive."""
    t = _times(4)
    delta = timedelta(days=1)
    # decision t0, horizon 1 → label window (t0, t1]; test at t1 → overlap.
    assert overlaps(t[0], t[1], 1, delta) is True
    # horizon 0 → label window (t0, t0] ends strictly before t1 → no overlap.
    assert overlaps(t[0], t[1], 0, delta) is False
    # decision strictly before test_start with label_end one tick short.
    assert overlaps(t[0], t[2], 1, delta) is False


def test_overlaps_decision_on_or_after_test_start() -> None:
    t = _times(4)
    delta = timedelta(days=1)
    assert overlaps(t[2], t[2], 0, delta) is True
    assert overlaps(t[3], t[2], 1, delta) is True


# ------------------------------- purging.purge_mask -------------------------------


def test_purge_mask_label_end_times_boundary_and_alignment() -> None:
    t = _times(6)
    # Explicit end exactly at test_start → purged (inclusive boundary).
    ends = [t[0], t[1], t[3], t[4], t[5], t[5]]
    keep = purge_mask(t, t[3], t[4], 0, label_end_times=ends)
    # i=2 has end_time t3 == test_start → unsafe; i=3,4 in holdout.
    assert keep == [True, True, False, False, False, True]
    # End one tick before test_start → i=2 stays.
    ends[2] = t[2]
    keep = purge_mask(t, t[3], t[4], 0, label_end_times=ends)
    assert keep == [True, True, True, False, False, True]


def test_purge_mask_label_end_times_misaligned_raises() -> None:
    t = _times(4)
    with pytest.raises(ValueError, match="align"):
        purge_mask(t, t[1], t[2], 0, label_end_times=t[:3])
    with pytest.raises(ValueError, match="not precede"):
        purge_mask(t, t[1], t[2], 0, label_end_times=[t[0], t[0], t[2], t[3]])


def test_purge_mask_unknown_decision_uses_insertion_point() -> None:
    """A decision time missing from session_index keeps its sorted position."""
    t = _times(8)
    idx = session_index(t)
    gap_time = t[4] + timedelta(hours=12)  # between sessions 4 and 5
    keep = purge_mask([gap_time], t[5], t[6], horizon_bars=0, session_index=idx)
    # Insertion index 5 sits inside holdout [5, 6] → dropped.
    assert keep == [False]
    keep = purge_mask([gap_time], t[6], t[7], horizon_bars=1, session_index=idx)
    # Insertion index 5, label_end 6 >= test_lo 6 → dropped.
    assert keep == [False]
    keep = purge_mask([gap_time], t[6], t[7], horizon_bars=0, session_index=idx)
    assert keep == [True]


def test_purge_mask_empty_session_index_keeps_all() -> None:
    t = _times(3)
    assert purge_mask(t, t[0], t[1], 2, session_index={}) == [True, True, True]


def test_purge_mask_test_start_between_sessions() -> None:
    """test_start not on a session: test_lo is the next session, conservatively."""
    t = _times(8)
    idx = session_index(t)
    mid = t[4] + timedelta(hours=12)
    keep = purge_mask(t, mid, t[6], horizon_bars=0, session_index=idx)
    # holdout = sessions 5..6; i=4 is before test_lo=5 and label 4 < 5 → kept.
    assert keep == [True, True, True, True, True, False, False, True]


def test_purge_mask_unsorted_decision_times_align_elementwise() -> None:
    t = _times(6)
    keep = purge_mask([t[5], t[0], t[3]], t[2], t[3], horizon_bars=0)
    assert keep == [True, True, False]
    # h=1 also purges t0: its label window (0, 1] reaches holdout index 1.
    keep = purge_mask([t[5], t[0], t[3]], t[2], t[3], horizon_bars=1)
    assert keep == [True, False, False]


# ------------------------------- embargo -------------------------------


def test_embargo_boundary_inclusive_window() -> None:
    """Sessions ``end_i < i <= end_i + embargo_bars`` drop; the block itself keeps."""
    t = _times(8)
    idx = session_index(t)
    mask = embargo_mask(t, t[2], 2, idx)
    assert mask == [True, True, True, False, False, True, True, True]
    # i == end_i + bars is the last dropped session.
    mask = embargo_mask(t, t[2], 1, idx)
    assert mask[3] is False and mask[4] is True


def test_embargo_covers_everything_after_block_end() -> None:
    t = _times(4)
    idx = session_index(t)
    assert embargo_mask(t, t[0], 99, idx) == [True, False, False, False]


def test_embargo_unknown_end_tie_picks_earlier_session() -> None:
    t = _times(4)
    idx = session_index(t)
    # Midpoint between t1 and t2 → min() over sorted keys returns the earlier.
    mid = t[1] + timedelta(hours=12)
    mask = embargo_mask(t, mid, 1, idx)
    # nearest is t1 (i=1) → drop i=2 only.
    assert mask == [True, True, False, True]


def test_embargo_block_end_after_all_sessions_keeps_all() -> None:
    t = _times(4)
    idx = session_index(t)
    assert embargo_mask(t, datetime(2030, 1, 1, tzinfo=UTC), 3, idx) == [True] * 4


def test_embargo_single_decision_time() -> None:
    """Mask length equals len(decision_times) — a lone decision is kept."""
    t = _times(1)
    idx = session_index(t)
    assert embargo_mask(t, t[0], 5, idx) == [True]


# ------------------------------- cpcv -------------------------------


def test_cpcv_exact_fold_geometry() -> None:
    """n=12, 4 groups of 3, k=1, h=1, e=1 — exact purge+embargo boundaries."""
    times = _times(12)
    folds = combinatorial_purged_cv(
        times, n_groups=4, n_test_groups=1, horizon_bars=1, embargo_bars=1
    )
    assert len(folds) == 4
    trains = [[times.index(x) for x in f.train_times] for f in folds]
    tests = [[times.index(x) for x in f.test_times] for f in folds]
    # test group 0: embargo drops train 3; horizon drops nothing (i<0 never).
    assert trains[0] == [4, 5, 6, 7, 8, 9, 10, 11]
    assert tests[0] == [0, 1, 2]
    # test group 1: horizon drops 2, embargo drops 6.
    assert trains[1] == [0, 1, 7, 8, 9, 10, 11]
    # test group 3: horizon drops 8, trailing embargo is empty.
    assert trains[3] == [0, 1, 2, 3, 4, 5, 6, 7]
    for f in folds:
        assert f.val_times == []


def test_cpcv_noncontiguous_test_groups_purge_per_block() -> None:
    """Test groups {0, 2}: group 1 remains trainable — no min/max span wipe."""
    times = _times(20)
    folds = combinatorial_purged_cv(
        times, n_groups=4, n_test_groups=2, horizon_bars=0, embargo_bars=0
    )
    assert len(folds) == 6
    for f in folds:
        test_idx = {times.index(x) for x in f.test_times}
        train_idx = {times.index(x) for x in f.train_times}
        assert test_idx.isdisjoint(train_idx)
        # horizon=0, embargo=0: every non-test group is fully in train.
        expected_train = set(range(20)) - test_idx
        assert train_idx == expected_train


def test_cpcv_counts_and_paths_exact() -> None:
    assert cpcv_n_splits(6, 2) == 15
    assert cpcv_n_paths(6, 2) == 5
    assert cpcv_n_splits(4, 3) == 4
    assert cpcv_n_paths(4, 3) == 3
    # n_groups == 2 is legal: weaker lower bounds (``<= 2``/``< 3``) must not
    # turn this into a raise.
    assert cpcv_n_splits(2, 1) == 2
    assert cpcv_n_paths(2, 1) == 1
    with pytest.raises(ValueError, match="^invalid CPCV group counts$"):
        cpcv_n_splits(2, 0)
    with pytest.raises(ValueError, match="^invalid CPCV group counts$"):
        cpcv_n_splits(2, 2)
    with pytest.raises(ValueError, match="^invalid CPCV group counts$"):
        cpcv_n_paths(4, 4)


def test_cpcv_path_assignments_exact_matrix() -> None:
    """(4, 2): lexicographic splits (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)."""
    a = cpcv_path_assignments(4, 2)
    assert a.shape == (3, 4)
    expected = np.array([[0, 0, 1, 2], [1, 3, 3, 4], [2, 4, 5, 5]], dtype=np.int64)
    np.testing.assert_array_equal(a, expected)


def test_stitch_group_paths_exact_values() -> None:
    a = cpcv_path_assignments(4, 2)
    scores = np.arange(24, dtype=float).reshape(6, 4)
    stitched = stitch_group_paths(scores, a)
    # stitched[p, g] = scores[assignments[p, g], g]
    np.testing.assert_array_equal(stitched[0], [0.0, 1.0, 6.0, 11.0])
    np.testing.assert_array_equal(stitched[2], [8.0, 17.0, 22.0, 23.0])
    assert stitched.shape == (3, 4)


def test_stitch_group_paths_contract_violations_raise() -> None:
    a = cpcv_path_assignments(4, 2)
    scores = np.zeros((6, 4))
    with pytest.raises(ValueError, match="2-d"):
        stitch_group_paths(np.zeros(6), a)
    with pytest.raises(ValueError, match="n_groups"):
        stitch_group_paths(np.zeros((6, 5)), a)
    with pytest.raises(ValueError, match="outside"):
        stitch_group_paths(scores, a + 10)


def test_cpcv_indices_exact_purge_embargo() -> None:
    folds = combinatorial_purged_indices(12, 4, 1, label_horizon=1, embargo=1)
    assert len(folds) == 4
    np.testing.assert_array_equal(folds[0][0], np.arange(4, 12))
    np.testing.assert_array_equal(folds[0][1], np.arange(0, 3))
    np.testing.assert_array_equal(folds[1][0], np.array([0, 1, 7, 8, 9, 10, 11]))
    np.testing.assert_array_equal(folds[3][0], np.arange(0, 8))


def test_cpcv_indices_guard_clauses() -> None:
    with pytest.raises(ValueError, match="label_horizon"):
        combinatorial_purged_indices(12, 4, 1, label_horizon=-1, embargo=0)
    with pytest.raises(ValueError, match="embargo"):
        combinatorial_purged_indices(12, 4, 1, label_horizon=0, embargo=-1)
    with pytest.raises(ValueError, match="at least n_groups"):
        combinatorial_purged_indices(3, 4, 1, label_horizon=0, embargo=0)


# ------------------------------- walk_forward -------------------------------


def test_session_index_collapses_duplicates_sorted() -> None:
    t = _times(4)
    idx = session_index([t[2], t[0], t[2], t[1]])
    assert idx == {t[0]: 0, t[1]: 1, t[2]: 2}


def test_timestamp_ns_conversions() -> None:
    t = _times(3)
    assert timestamp_ns([]).size == 0
    arr = np.array(t, dtype="datetime64[ns]")
    out = timestamp_ns(arr)
    assert out.dtype == np.int64
    assert int(out[0]) == int(arr[0].astype(np.int64))
    # aware datetime → UTC naive nanoseconds; naive → wall clock.
    aware = timestamp_ns(t)
    naive = timestamp_ns([x.replace(tzinfo=None) for x in t])
    np.testing.assert_array_equal(aware, naive)
    assert naive[0] == int(np.datetime64(t[0].replace(tzinfo=None), "ns").astype(np.int64))


def test_row_mask_for_times_edges() -> None:
    t = _times(5)
    assert row_mask_for_times([], t).size == 0
    assert not row_mask_for_times(t, []).any()
    mask = row_mask_for_times(t, [t[1], t[3]])
    np.testing.assert_array_equal(mask, [False, True, False, True, False])


def test_walk_forward_expanding_vs_rolling_train_window() -> None:
    times = _times(8)
    cfg = ValidationConfig(scheme="rolling", train_bars=2, val_bars=1, test_bars=1)
    folds = walk_forward(times, cfg, horizon_bars=0, embargo_bars=0)
    assert len(folds) == 3
    # rolling: fold at cursor=4 trains on sessions [2, 3] only.
    assert folds[1].train_times == [times[2], times[3]]
    assert folds[1].val_times == [times[4]]
    assert folds[1].test_times == [times[5]]
    cfg2 = ValidationConfig(scheme="expanding", train_bars=2, val_bars=1, test_bars=1)
    folds2 = walk_forward(times, cfg2, horizon_bars=0, embargo_bars=0)
    assert folds2[1].train_times == times[:4]


def test_walk_forward_embargo_tail_boundary() -> None:
    """Train keeps sessions with ``idx + embargo_bars < val_lo`` — strict inequality."""
    times = _times(10)
    cfg = ValidationConfig(scheme="expanding", train_bars=4, val_bars=2, test_bars=2)
    folds = walk_forward(times, cfg, horizon_bars=0, embargo_bars=2)
    # cursor=4, val=[4,5]: kept train sessions have i+2 < 4 → i in {0,1}.
    assert folds[0].train_times == times[:2]
    # embargo=1 keeps i <= 2.
    folds = walk_forward(times, cfg, horizon_bars=0, embargo_bars=1)
    assert folds[0].train_times == times[:3]


def test_walk_forward_label_end_times_purge() -> None:
    times = _times(10)
    cfg = ValidationConfig(scheme="expanding", train_bars=4, val_bars=2, test_bars=2)
    # label_end for session 2 reaches val_start=4 → purged even with horizon 0.
    ends = list(times)
    ends[2] = times[4]
    folds = walk_forward(times, cfg, horizon_bars=0, embargo_bars=0, label_end_times=ends)
    assert folds[0].train_times == [times[0], times[1], times[3]]
    with pytest.raises(ValueError, match="align"):
        walk_forward(times, cfg, horizon_bars=0, embargo_bars=0, label_end_times=times[:3])
    bad = list(times)
    bad[1] = times[0]
    with pytest.raises(ValueError, match="not precede"):
        walk_forward(times, cfg, horizon_bars=0, embargo_bars=0, label_end_times=bad)


def test_walk_forward_label_end_max_merge() -> None:
    """Duplicate decision times keep the latest label end (conservative)."""
    times = _times(7)
    dup = [times[2], times[2], times[0], times[1], times[3], times[4], times[5], times[6]]
    ends = [times[5], times[3], times[0], times[1], times[3], times[4], times[5], times[6]]
    cfg = ValidationConfig(scheme="expanding", train_bars=4, val_bars=1, test_bars=1)
    folds = walk_forward(dup, cfg, horizon_bars=0, embargo_bars=0, label_end_times=ends)
    # session 2's merged end is t5 — reaches val session 4 → purged.
    assert folds[0].train_times == [times[0], times[1], times[3]]


def test_assert_no_label_overlap_noncontiguous_blocks() -> None:
    times = _times(12)
    idx = session_index(times)
    # Holdout {3,4} ∪ {8,9}: gap sessions 5,6,7 are safe for train at h=1.
    fold = Fold(
        train_times=[times[5], times[6]],
        val_times=[times[3], times[4]],
        test_times=[times[8], times[9]],
    )
    assert_no_label_overlap(fold, 1, idx)
    # label_end == vmin boundary is an overlap.
    bad = Fold(
        train_times=[times[6]],
        val_times=[times[3], times[4]],
        test_times=[times[8], times[9]],
    )
    with pytest.raises(AssertionError, match="label overlap"):
        assert_no_label_overlap(bad, 2, idx)  # (6, 8] reaches block [8, 9]
    with pytest.raises(AssertionError, match="label overlap"):
        assert_no_label_overlap(
            Fold([times[2]], [times[3]], [times[4]]), 1, idx
        )  # (2, 3] reaches 3
    # horizon 0 → empty label window → never raises.
    assert_no_label_overlap(Fold([times[6]], [times[7]], [times[8]]), 0, idx)
    # empty holdout → no-op.
    assert_no_label_overlap(Fold(times[:3], [], []), 4, idx)


def test_fold_ic_stability_exact_values() -> None:
    out = fold_ic_stability([])
    assert out["n_folds"] == 0 and out["stability"] == 0.0
    assert np.isnan(out["mean_ic"]) and np.isnan(out["std_ic"])
    out = fold_ic_stability([0.1, -0.2, 0.05], min_ic=0.0)
    assert out["n_folds"] == 3
    assert out["stability"] == pytest.approx(2 / 3)
    assert out["mean_ic"] == pytest.approx(-0.05 / 3)
    var = ((0.1 + 0.05 / 3) ** 2 + (-0.2 + 0.05 / 3) ** 2 + (0.05 + 0.05 / 3) ** 2) / 2
    assert out["std_ic"] == pytest.approx(var**0.5)
    # Non-finite and None entries are filtered before scoring.
    out = fold_ic_stability([0.1, float("nan"), float("inf"), None], min_ic=0.05)
    assert out["n_folds"] == 1 and out["stability"] == 1.0
    assert out["std_ic"] == 0.0  # var / max(n-1, 1) guards single fold
    # boundary: v == min_ic does NOT count (strict >).
    out = fold_ic_stability([0.05, 0.06], min_ic=0.05)
    assert out["stability"] == pytest.approx(0.5)


# ------------------------------- second wave ---------------------------------


def test_purge_mask_arg_validation_messages() -> None:
    # ``match`` is re.search — anchor so ``XX…XX`` message mutants die.
    t = _times(4)
    with pytest.raises(ValueError, match="^horizon_bars must be non-negative$"):
        purge_mask(t, t[1], t[2], -1)
    with pytest.raises(ValueError, match="^label_end_times must align with decision_times$"):
        purge_mask(t, t[1], t[2], 0, label_end_times=t[:3])
    with pytest.raises(ValueError, match="^test_end must be on or after test_start$"):
        purge_mask(t, t[2], t[1], 0)
    with pytest.raises(ValueError, match="^label_end_times must not precede decision_times$"):
        purge_mask(
            t,
            t[3],
            t[3],
            0,
            label_end_times=[datetime(2020, 1, 1, tzinfo=UTC)] * 4,
        )
    with pytest.raises(ValueError, match="^horizon_bars must be non-negative$"):
        overlaps(t[0], t[1], -1, timedelta(days=1))


def test_cpcv_two_group_split_is_valid() -> None:
    # n_groups == 2 is legal (C(2,1) = 2 splits).
    folds = combinatorial_purged_cv(_times(6), 2, 1, horizon_bars=0, embargo_bars=0)
    assert len(folds) == 2


def test_cpcv_single_session_groups_allowed() -> None:
    # n == n_groups → every group is one session; still a valid CPCV.
    folds = combinatorial_purged_cv(_times(4), 4, 1, horizon_bars=0, embargo_bars=0)
    assert len(folds) == 4
    assert all(len(f.test_times) == 1 for f in folds)


def test_cpcv_datetime_arg_validation() -> None:
    with pytest.raises(ValueError, match="^invalid CPCV group counts$"):
        combinatorial_purged_cv(_times(8), 1, 1, horizon_bars=0, embargo_bars=0)
    with pytest.raises(ValueError, match="^invalid CPCV group counts$"):
        combinatorial_purged_cv(_times(8), 2, 0, horizon_bars=0, embargo_bars=0)
    with pytest.raises(ValueError, match="^invalid CPCV group counts$"):
        combinatorial_purged_cv(_times(8), 2, 2, horizon_bars=0, embargo_bars=0)
    with pytest.raises(ValueError, match="^horizon_bars must be non-negative$"):
        combinatorial_purged_cv(_times(8), 2, 1, horizon_bars=-1, embargo_bars=0)
    with pytest.raises(ValueError, match="^embargo_bars must be non-negative$"):
        combinatorial_purged_cv(_times(8), 2, 1, horizon_bars=0, embargo_bars=-1)
    with pytest.raises(ValueError, match="^need at least n_groups"):
        combinatorial_purged_cv(_times(4), 8, 1, horizon_bars=0, embargo_bars=0)


def test_cpcv_indices_and_stitch_arg_validation() -> None:
    with pytest.raises(ValueError, match="^label_horizon must be non-negative$"):
        combinatorial_purged_indices(8, 4, 1, label_horizon=-1, embargo=0)
    with pytest.raises(ValueError, match="^embargo must be non-negative$"):
        combinatorial_purged_indices(8, 4, 1, label_horizon=0, embargo=-1)
    with pytest.raises(ValueError, match="^need at least n_groups"):
        combinatorial_purged_indices(3, 4, 1, label_horizon=0, embargo=0)
    with pytest.raises(ValueError, match="^group_scores and assignments must be 2-d$"):
        stitch_group_paths(np.zeros(4), np.zeros((2, 4), dtype=int))
    with pytest.raises(ValueError, match="^group_scores and assignments must be 2-d$"):
        stitch_group_paths(np.zeros((3, 4)), np.zeros(4, dtype=int))
    with pytest.raises(ValueError, match="^assignments and group_scores disagree on n_groups$"):
        stitch_group_paths(np.zeros((3, 4)), np.zeros((2, 5), dtype=int))
    with pytest.raises(ValueError, match="^assignment split id is outside group_scores$"):
        stitch_group_paths(np.zeros((3, 4)), np.array([[3, 0, 0, 0]]))


def test_cpcv_indices_single_session_test_fold() -> None:
    # n == n_groups → every test group is a single index; ``test.size == 1``
    # must not be treated as empty.
    folds = combinatorial_purged_indices(4, 4, 1, label_horizon=0, embargo=0)
    assert len(folds) == 4
    for train, test in folds:
        assert test.size == 1
        assert train.size == 3


def test_cpcv_cv_interior_empty_train_does_not_stop_scan() -> None:
    # Datetime API: combo (1,3) fully purges its train set mid-iteration;
    # later combos must still produce folds (``continue``, not ``break``).
    folds = combinatorial_purged_cv(_times(10), 4, 2, horizon_bars=2, embargo_bars=0)
    assert len(folds) == 5
    base = _times(10)
    np.testing.assert_array_equal([t.day for t in folds[-1].test_times], [t.day for t in base[5:]])


def test_cpcv_indices_single_session_train_fold() -> None:
    # k = n_groups - 1 → train is one single-session group (kept.size == 1).
    folds = combinatorial_purged_indices(4, 4, 3, label_horizon=0, embargo=0)
    assert len(folds) == 4
    for train, test in folds:
        assert train.size == 1
        assert test.size == 3


def test_cpcv_indices_fold_dtypes() -> None:
    folds = combinatorial_purged_indices(8, 4, 1, label_horizon=0, embargo=0)
    for train, test in folds:
        assert train.dtype == np.int64
        assert test.dtype == np.int64


def test_walk_forward_label_end_times_validation() -> None:
    times = _times(6)
    cfg = ValidationConfig(scheme="expanding", train_bars=2, val_bars=1, test_bars=1)
    with pytest.raises(ValueError, match="^label_end_times must align with times$"):
        walk_forward(times, cfg, horizon_bars=0, embargo_bars=0, label_end_times=times[:5])
    with pytest.raises(ValueError, match="^label_end_times must not precede times$"):
        ends = list(times)
        ends[2] = times[0]
        walk_forward(times, cfg, horizon_bars=0, embargo_bars=0, label_end_times=ends)


def test_timestamp_ns_single_element_and_dtypes() -> None:
    t = _times(1)
    out = timestamp_ns(t)
    assert out.dtype == np.int64
    assert out.size == 1
    d64 = np.datetime64("2021-01-04T00:00:00", "ns")
    np.testing.assert_array_equal(
        timestamp_ns([d64]),
        timestamp_ns(np.array([d64], dtype="datetime64[ns]")),
    )


def test_one_timestamp_ns_object_paths() -> None:
    # ``object`` arrays force the per-element fallback (the vectorized path
    # only fires for datetime64 dtype).
    d64 = np.array([np.datetime64("2021-01-04T00:00:00", "ns")], dtype=object)
    expected = int(np.datetime64("2021-01-04T00:00:00", "ns").astype(np.int64))
    assert int(timestamp_ns(d64)[0]) == expected
    aware = np.array([datetime(2021, 1, 4, tzinfo=UTC)], dtype=object)
    assert int(timestamp_ns(aware)[0]) == expected
    naive = np.array([datetime(2021, 1, 4)], dtype=object)
    assert int(timestamp_ns(naive)[0]) == expected
    misc = np.array(["2021-01-04T00:00:00"], dtype=object)
    assert int(timestamp_ns(misc)[0]) == expected


def test_one_timestamp_ns_preserves_sub_microsecond_ns() -> None:
    # int64 nanoseconds exceed float53 precision: any ``astype(None)``
    # (float64) round-trip corrupts odd ns counts — the int64 chain is load-bearing.
    d64 = np.array([np.datetime64("2021-01-04T00:00:00.000000001", "ns")], dtype=object)
    assert int(timestamp_ns(d64)[0]) == 1609718400000000001
    us = np.array([datetime(2021, 1, 4, 0, 0, 0, 1, tzinfo=UTC)], dtype=object)
    assert int(timestamp_ns(us)[0]) == 1609718400000001000
    s = np.array(["2021-01-04T00:00:00.000000003"], dtype=object)
    assert int(timestamp_ns(s)[0]) == 1609718400000000003


def test_timestamp_ns_empty_is_int64() -> None:
    assert timestamp_ns([]).dtype == np.int64
    assert timestamp_ns(np.array([], dtype="datetime64[ns]")).dtype == np.int64


def test_timestamp_ns_string_input() -> None:
    out = timestamp_ns(["2021-01-04"])
    assert out.dtype == np.int64
    assert int(out[0]) == int(np.datetime64("2021-01-04", "ns").astype(np.int64))


def test_row_mask_bool_dtype() -> None:
    t = _times(3)
    mask = row_mask_for_times(t, [t[1]])
    assert mask.dtype == np.bool_
    np.testing.assert_array_equal(mask, [False, True, False])
    # Single-element input must still produce a one-element mask.
    np.testing.assert_array_equal(row_mask_for_times([t[1]], [t[1]]), [True])
    # Empty inputs keep dtype=bool.
    assert row_mask_for_times([], t).dtype == np.bool_
    assert row_mask_for_times(t, []).dtype == np.bool_


def test_walk_forward_expanding_train_starts_at_first_session() -> None:
    times = _times(8)
    cfg = ValidationConfig(scheme="expanding", train_bars=2, val_bars=1, test_bars=1)
    folds = walk_forward(times, cfg, horizon_bars=0, embargo_bars=0)
    assert len(folds) == 3
    for f in folds:
        assert f.train_times[0] == times[0]


def test_walk_forward_horizon_purge_exact() -> None:
    # horizon=2 drops train sessions with i+2 >= val_lo=6 → kept = i < 4.
    times = _times(12)
    cfg = ValidationConfig(scheme="expanding", train_bars=6, val_bars=2, test_bars=2)
    folds = walk_forward(times, cfg, horizon_bars=2, embargo_bars=0)
    assert folds[0].train_times == times[:4]


def test_assert_no_label_overlap_second_block_session() -> None:
    """The second holdout session must still bound the first block (ids[1:])."""
    times = _times(12)
    idx = session_index(times)
    # Train i=3 with h=1: window (3,4] intersects holdout block [3,4].
    fold = Fold(
        train_times=[times[3]],
        val_times=[times[3], times[4]],
        test_times=[times[8], times[9]],
    )
    with pytest.raises(AssertionError, match="label overlap"):
        assert_no_label_overlap(fold, 1, idx)


def test_assert_no_label_overlap_zero_horizon_is_vacuous() -> None:
    """h=0 → label_end == i → the scan skips: a train row inside the holdout
    must NOT raise (the label window is empty)."""
    times = _times(10)
    idx = session_index(times)
    fold = Fold(
        train_times=[times[8]],
        val_times=[times[8]],
        test_times=[times[9]],
    )
    assert_no_label_overlap(fold, 0, idx)


def test_assert_no_label_overlap_train_at_block_end() -> None:
    """i == vmax: the window (vmax, vmax+h] starts AFTER the block → no raise."""
    times = _times(12)
    idx = session_index(times)
    fold = Fold(
        train_times=[times[4]],
        val_times=[times[3], times[4]],
        test_times=[times[8], times[9]],
    )
    assert_no_label_overlap(fold, 1, idx)


def test_fold_ic_stability_default_threshold_and_n2() -> None:
    # Default min_ic=0.0: strictly-positive ICs count as stable.
    out = fold_ic_stability([0.5, -0.1])
    assert out["stability"] == pytest.approx(0.5)
    # n=2 → var = Σ(v-μ)² / max(n-1=1, 1) — the unbiased divisor.
    out = fold_ic_stability([0.0, 0.2])
    assert out["std_ic"] == pytest.approx(0.14142135623730953)


def test_cpcv_indices_interior_empty_train_does_not_stop_scan() -> None:
    # Combo (1,3) fully purges its train set and is skipped at position 4 of
    # 6 — later combos must still produce folds (``continue``, not ``break``).
    folds = combinatorial_purged_indices(10, 4, 2, label_horizon=2, embargo=0)
    assert len(folds) == 5
    tests = [te.tolist() for _, te in folds]
    assert [2, 3, 4, 7, 8, 9] not in tests  # (1,3) purged everything — omitted
    np.testing.assert_array_equal(folds[-1][1], [5, 6, 7, 8, 9])
    np.testing.assert_array_equal(folds[-1][0], [0, 1, 2])


def test_purge_mask_decision_at_test_lo_kept_when_holdout_empty() -> None:
    # A test block strictly between two sessions has an empty session holdout
    # (test_lo=1 > test_hi=0): a decision at test_lo is AFTER the test window
    # and must be kept — ``i < test_lo`` must not become ``i <= test_lo``.
    times = [
        datetime(2021, 1, 4, tzinfo=UTC),
        datetime(2021, 1, 5, tzinfo=UTC),
        datetime(2021, 1, 6, tzinfo=UTC),
    ]
    idx = {t: i for i, t in enumerate(times)}
    keep = purge_mask(
        times,
        times[0] + timedelta(hours=6),
        times[0] + timedelta(hours=12),
        1,
        session_index=idx,
    )
    # times[0] is purged (label reaches the test window); times[1] sits at
    # test_lo but beyond the holdout — it must stay kept.
    assert keep == [False, True, True]


def test_cpcv_embargo_before_boundary_index_dropped() -> None:
    # Group 1 test block is uniq[2:4]; embargo_bars=1 drops the session at
    # exactly idx == lo - embargo == 1.  A ``<`` mutant on the embargo start
    # boundary would keep it.
    times = _times(10)
    folds = combinatorial_purged_cv(times, 5, 1, horizon_bars=0, embargo_bars=1)
    fold = [f for f in folds if f.test_times == [times[2], times[3]]][0]
    assert times[1] not in fold.train_times
    assert times[4] not in fold.train_times
    assert times[0] in fold.train_times and times[5] in fold.train_times


def test_cpcv_empty_kept_fold_skips_not_stops() -> None:
    # test=(1,) empties the train set under embargo=2 and is skipped; the loop
    # must continue so test=(2,) still yields a fold.  ``break`` would end at 1.
    folds = combinatorial_purged_indices(6, 3, 1, label_horizon=0, embargo=2)
    assert len(folds) == 2
    np.testing.assert_array_equal(folds[0][1], [0, 1])
    np.testing.assert_array_equal(folds[1][1], [4, 5])


def test_assert_no_label_overlap_gap_in_holdout_not_merged() -> None:
    # Holdout {0,1,3} is two blocks: train index 1 with horizon 1 ends at 2,
    # inside the gap — no intersection.  A j == hi+2 merge would fabricate a
    # (0,3) block and raise spuriously.
    times = _times(4)
    idx = session_index(times)
    fold = Fold(train_times=[times[1]], val_times=[times[0], times[3]], test_times=[times[1]])
    assert_no_label_overlap(fold, 1, idx)
