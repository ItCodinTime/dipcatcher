"""Wave 25: purging.overlaps / purge_mask empty and overlap edges."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from quant_fund.validation.purging import overlaps, purge_mask
from quant_fund.validation.walk_forward import session_index


def _times(n: int) -> list[datetime]:
    t0 = datetime(2018, 1, 2, tzinfo=UTC)
    return [t0 + timedelta(days=i) for i in range(n)]


def test_purge_mask_empty_decision_times() -> None:
    t = _times(3)
    assert purge_mask([], t[1], t[2], horizon_bars=1) == []


def test_purge_mask_negative_horizon_raises() -> None:
    t = _times(5)
    with pytest.raises(ValueError, match="horizon_bars"):
        purge_mask(t, t[1], t[2], horizon_bars=-1)


def test_purge_mask_test_end_before_start_raises() -> None:
    t = _times(5)
    with pytest.raises(ValueError, match="test_end"):
        purge_mask(t, t[3], t[1], horizon_bars=1)


def test_purge_mask_horizon_zero_drops_inside_test() -> None:
    """horizon=0: label interval (i, i] is empty → drop every i inside the inclusive holdout."""
    t = _times(6)
    # test [2,3]; in_holdout = 2 <= i <= 3 → i in {2, 3}
    keep = purge_mask(t, t[2], t[3], horizon_bars=0)
    assert keep == [True, True, False, False, True, True]


def test_purge_mask_drops_pre_test_label_overlap() -> None:
    """horizon=2 vs test starting at index 4: indices 2,3 overlap into test."""
    t = _times(8)
    keep = purge_mask(t, t[4], t[5], horizon_bars=2)
    # 2:6 → F,F,F,F (4,5 are the inclusive holdout, 2,3 label-overlap into it)
    assert keep == [True, True, False, False, False, False, True, True]


def test_purge_mask_post_test_train_kept() -> None:
    """Sessions strictly after test_end stay in train (purge is label-forward)."""
    t = _times(10)
    keep = purge_mask(t, t[3], t[5], horizon_bars=2)
    assert keep[6:] == [True, True, True, True]


def test_purge_mask_with_explicit_session_index() -> None:
    t = _times(8)
    idx = session_index(t)
    keep = purge_mask(t, t[4], t[5], horizon_bars=2, session_index=idx)
    assert keep[:2] == [True, True]
    assert keep[2:5] == [False, False, False]


def test_purge_mask_equal_test_start_end() -> None:
    t = _times(5)
    keep = purge_mask(t, t[2], t[2], horizon_bars=1)
    # test_lo=test_hi=2: in_holdout → i==2; pre-test label overlap → i==1
    assert keep == [True, False, False, True, True]


def test_overlaps_negative_horizon_raises() -> None:
    t0 = _times(1)[0]
    with pytest.raises(ValueError, match="horizon_bars"):
        overlaps(t0, t0 + timedelta(days=1), -1, timedelta(days=1))


def test_overlaps_label_reaches_test_start() -> None:
    t = _times(5)
    delta = timedelta(days=1)
    # decision at t[0], horizon 1 day → label_end=t[1]; test_start=t[2] → no overlap
    assert overlaps(t[0], t[2], 1, delta) is False
    # horizon 3 → label_end=t[3]; test_start=t[2] intersects
    assert overlaps(t[0], t[2], 3, delta) is True
    # decision on/after test start always overlaps
    assert overlaps(t[2], t[2], 0, delta) is True
    assert overlaps(t[3], t[2], 1, delta) is True
