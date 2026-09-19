"""Wave 24: walk_forward fold/purge/embargo extremes (complements test_validation / fold_stability)."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from quant_fund.config.models import ValidationConfig
from quant_fund.validation.walk_forward import (
    Fold,
    assert_no_label_overlap,
    fold_ic_stability,
    session_index,
    walk_forward,
)


def _times(n: int, *, start: datetime | None = None) -> list[datetime]:
    t0 = start or datetime(2020, 1, 2, tzinfo=UTC)
    return [t0 + timedelta(days=i) for i in range(n)]


def test_walk_forward_empty_and_too_short_yield_no_folds() -> None:
    cfg = ValidationConfig(train_bars=10, val_bars=5, test_bars=5)
    assert walk_forward([], cfg, horizon_bars=1, embargo_bars=0) == []
    short = _times(19)  # need 20 for one fold
    assert walk_forward(short, cfg, horizon_bars=1, embargo_bars=0) == []
    exact = _times(20)
    folds = walk_forward(exact, cfg, horizon_bars=1, embargo_bars=0)
    assert len(folds) == 1
    assert len(folds[0].val_times) == 5
    assert len(folds[0].test_times) == 5


def test_walk_forward_fold_count_expanding_vs_rolling() -> None:
    times = _times(40)
    cfg_e = ValidationConfig(train_bars=10, val_bars=5, test_bars=5, scheme="expanding")
    cfg_r = ValidationConfig(train_bars=10, val_bars=5, test_bars=5, scheme="rolling")
    folds_e = walk_forward(times, cfg_e, horizon_bars=3, embargo_bars=2)
    folds_r = walk_forward(times, cfg_r, horizon_bars=3, embargo_bars=2)
    # cursor steps by val+test=10; starts at 10 → ends 20,30,40 → 3 folds
    assert len(folds_e) == 3
    assert len(folds_r) == 3
    # Expanding train origin stays at t0; rolling train window slides forward
    assert folds_e[0].train_times[0] == times[0]
    assert folds_e[-1].train_times[0] == times[0]
    assert folds_r[0].train_times[0] == times[0]
    assert folds_r[1].train_times[0] > folds_r[0].train_times[0]
    assert folds_r[2].train_times[0] > folds_r[1].train_times[0]
    # Rolling train sizes stay capped; expanding grows
    assert len(folds_r[0].train_times) == len(folds_r[-1].train_times)
    assert len(folds_e[-1].train_times) > len(folds_e[0].train_times)


def test_walk_forward_purge_embargo_geometry() -> None:
    """horizon=3 + embargo=2 drops the last 3 of a 10-bar train block (indices 7–9)."""
    times = _times(40)
    cfg = ValidationConfig(train_bars=10, val_bars=5, test_bars=5, scheme="expanding")
    folds = walk_forward(times, cfg, horizon_bars=3, embargo_bars=2)
    f0 = folds[0]
    idx = session_index(times)
    va0 = idx[f0.val_times[0]]
    assert va0 == 10
    # Keep only sessions with idx + embargo < va0 AND purge-safe → indices 0..6
    assert [idx[t] for t in f0.train_times] == list(range(7))
    assert max(f0.train_times) < min(f0.val_times) <= max(f0.val_times) < min(f0.test_times)
    for f in folds:
        assert_no_label_overlap(f, 3, idx)


def test_walk_forward_dedupes_duplicate_timestamps() -> None:
    base = _times(25)
    duped = base + base[5:15]  # duplicates must not inflate fold geometry
    cfg = ValidationConfig(train_bars=10, val_bars=5, test_bars=5)
    folds = walk_forward(duped, cfg, horizon_bars=1, embargo_bars=0)
    uniq_n = len(set(duped))
    assert uniq_n == 25
    assert len(folds) == 1  # 10+5+5 = 20 ≤ 25; next would need 30
    assert len(folds[0].val_times) == 5
    assert len(folds[0].test_times) == 5


def test_assert_no_label_overlap_catches_forged_overlap() -> None:
    times = _times(30)
    idx = session_index(times)
    bad = Fold(
        train_times=times[8:12],
        val_times=times[10:15],
        test_times=times[15:20],
    )
    with pytest.raises(AssertionError, match="label overlap"):
        assert_no_label_overlap(bad, horizon_bars=5, idx=idx)


def test_assert_no_label_overlap_empty_holdout_and_safe_fold() -> None:
    times = _times(30)
    idx = session_index(times)
    empty_holdout = Fold(train_times=times[:8], val_times=[], test_times=[])
    assert_no_label_overlap(empty_holdout, horizon_bars=5, idx=idx)
    # Safe: train ends well before holdout with horizon=2
    safe = Fold(
        train_times=times[:5],
        val_times=times[10:15],
        test_times=times[15:20],
    )
    assert_no_label_overlap(safe, horizon_bars=2, idx=idx)


def test_fold_ic_stability_empty_nan_single_complements_happy_path() -> None:
    # Happy path lives in test_fold_stability.py — cover edges only here.
    empty = fold_ic_stability([])
    assert empty["n_folds"] == 0
    assert empty["stability"] == 0.0
    assert math.isnan(float(empty["mean_ic"]))
    assert math.isnan(float(empty["std_ic"]))

    all_bad = fold_ic_stability([float("nan"), None])  # type: ignore[list-item]
    assert all_bad["n_folds"] == 0
    assert all_bad["stability"] == 0.0

    single = fold_ic_stability([0.1], min_ic=0.0)
    assert single == {"n_folds": 1, "stability": 1.0, "mean_ic": 0.1, "std_ic": 0.0}

    below = fold_ic_stability([0.01, -0.02], min_ic=0.05)
    assert below["n_folds"] == 2
    assert below["stability"] == 0.0
