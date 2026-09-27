from datetime import UTC, datetime, timedelta

from quant_fund.config.models import ValidationConfig
from quant_fund.validation.cpcv import combinatorial_purged_cv
from quant_fund.validation.purging import overlaps, purge_mask
from quant_fund.validation.walk_forward import assert_no_label_overlap, session_index, walk_forward


def _times(n: int) -> list[datetime]:
    t0 = datetime(2018, 1, 2, tzinfo=UTC)
    return [t0 + timedelta(days=i) for i in range(n)]


def test_overlaps_when_decision_is_inside_test_window() -> None:
    t0 = _times(1)[0]
    assert overlaps(t0 + timedelta(days=2), t0, 1, timedelta(days=1))


def test_purge_mask_keeps_only_non_overlapping_labels() -> None:
    times = _times(8)
    keep = purge_mask(times, times[4], times[5], horizon_bars=2)

    assert keep[:2] == [True, True]
    # inclusive holdout [4, 5] and the pre-test label overlap [2, 3] are all purged
    assert keep[2:6] == [False, False, False, False]
    assert keep[6:] == [True, True]


def test_purge_mask_uses_explicit_sparse_label_end_times() -> None:
    times = _times(5)
    keep = purge_mask(
        times,
        times[2],
        times[2],
        horizon_bars=1,
        label_end_times=[times[2], times[2], times[3], times[4], times[4]],
    )

    # The first decision's observed next bar is time[2], so it overlaps the
    # holdout even though a dense one-bar session horizon would end at time[1].
    assert keep[:3] == [False, False, False]
    assert keep[3:] == [True, True]


def test_walk_forward_uses_sparse_label_endpoints() -> None:
    times = _times(7)
    cfg = ValidationConfig(train_bars=3, val_bars=1, test_bars=2)
    folds = walk_forward(
        times,
        cfg,
        horizon_bars=1,
        embargo_bars=0,
        label_end_times=[times[3], times[2], times[5], times[4], times[5], times[6], times[6]],
    )

    assert len(folds) == 1
    # time[0]'s observed label reaches the validation boundary and must be purged;
    # a dense one-bar assumption would incorrectly retain it.
    assert folds[0].train_times == [times[1]]


def test_walk_forward_no_overlap() -> None:
    times = _times(400)
    cfg = ValidationConfig(train_bars=100, val_bars=20, test_bars=20)
    folds = walk_forward(times, cfg, horizon_bars=5, embargo_bars=5)
    assert folds
    idx = session_index(times)
    for f in folds:
        assert_no_label_overlap(f, 5, idx)
        assert f.train_times
        assert f.test_times
        assert max(f.train_times) < min(f.val_times)


def test_cpcv_produces_folds() -> None:
    times = _times(120)
    folds = combinatorial_purged_cv(
        times, n_groups=6, n_test_groups=2, horizon_bars=3, embargo_bars=3
    )
    assert len(folds) > 1
    idx = session_index(times)
    for f in folds:
        assert_no_label_overlap(f, 3, idx)
