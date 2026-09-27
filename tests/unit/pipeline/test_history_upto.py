"""Wave 9: history_upto day-index concat vs filter(event_time <= asof)."""

from datetime import UTC, datetime

import polars as pl
import pytest

from quant_fund.pipeline.forecast import (
    build_event_time_day_index,
    history_for_calibration,
    history_upto,
    sort_for_history,
    under_history_sort_contract,
)


def _sorted_panel(*, tz_aware: bool = True, n_days: int = 5, n_names: int = 3) -> pl.DataFrame:
    rows: list[dict] = []
    for d in range(n_days):
        t = datetime(2024, 1, 1 + d, tzinfo=UTC) if tz_aware else datetime(2024, 1, 1 + d)
        for i in range(n_names):
            rows.append(
                {
                    "event_time": t,
                    "security_id": f"S{i}",
                    "x": float(d * 10 + i),
                }
            )
    return pl.DataFrame(rows).sort(["event_time", "security_id"])


def _assert_identical(a: pl.DataFrame, b: pl.DataFrame) -> None:
    assert a.columns == b.columns
    assert a.height == b.height
    assert a["security_id"].to_list() == b["security_id"].to_list()
    assert a["x"].to_list() == pytest.approx(b["x"].to_list())
    # Exact event_time identity (order + values)
    assert a["event_time"].to_list() == b["event_time"].to_list()


@pytest.mark.parametrize("tz_aware", [True, False])
def test_history_upto_matches_filter_sorted(tz_aware: bool) -> None:
    frame = _sorted_panel(tz_aware=tz_aware)
    index = build_event_time_day_index(frame)
    times = frame["event_time"].unique().sort().to_list()
    for asof in times:
        via_filter = frame.filter(pl.col("event_time") <= asof)
        via_index = history_upto(frame, asof, day_index=index)
        _assert_identical(via_index, via_filter)


def test_history_upto_without_index_equals_filter() -> None:
    frame = _sorted_panel()
    asof = datetime(2024, 1, 3, tzinfo=UTC)
    a = history_upto(frame, asof)
    b = frame.filter(pl.col("event_time") <= asof)
    _assert_identical(a, b)


def test_history_upto_before_all_empty() -> None:
    frame = _sorted_panel()
    index = build_event_time_day_index(frame)
    miss = datetime(2020, 1, 1, tzinfo=UTC)
    empty = history_upto(frame, miss, day_index=index)
    assert empty.height == 0
    assert empty.columns == frame.columns


def test_history_upto_duplicate_exact_timestamps() -> None:
    t = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    t2 = datetime(2024, 1, 1, 18, 0, tzinfo=UTC)
    rows = [
        {"event_time": t, "security_id": "A", "x": 1.0},
        {"event_time": t, "security_id": "A", "x": 2.0},  # duplicate exact stamp
        {"event_time": t2, "security_id": "B", "x": 3.0},
        {"event_time": t2, "security_id": "C", "x": 4.0},
    ]
    frame = pl.DataFrame(rows).sort(["event_time", "security_id"])
    index = build_event_time_day_index(frame)
    via_filter = frame.filter(pl.col("event_time") <= t)
    via_index = history_upto(frame, t, day_index=index)
    _assert_identical(via_index, via_filter)
    assert via_index.height == 2
    via_all = history_upto(frame, t2, day_index=index)
    _assert_identical(via_all, frame)


def test_history_upto_unsorted_preserves_multiset_not_always_order() -> None:
    """Document order edge: unsorted frames may reorder under day-slice concat."""
    rows: list[dict] = []
    for d in [3, 1, 4, 2, 0]:
        t = datetime(2024, 1, 1 + d, tzinfo=UTC)
        for i in range(2):
            rows.append(
                {
                    "event_time": t,
                    "security_id": f"S{i}",
                    "x": float(d * 10 + i),
                }
            )
    frame = pl.DataFrame(rows)  # deliberately unsorted
    asof = datetime(2024, 1, 3, tzinfo=UTC)
    index = build_event_time_day_index(frame)
    via_filter = frame.filter(pl.col("event_time") <= asof)
    via_index = history_upto(frame, asof, day_index=index)
    assert sorted(via_index["x"].to_list()) == sorted(via_filter["x"].to_list())
    assert sorted(via_index["security_id"].to_list()) == sorted(via_filter["security_id"].to_list())
    # Order may differ — production wiring deferred (panels are sorted).
    assert via_index["x"].to_list() != via_filter["x"].to_list()


def test_mixed_naive_aware_rejected_by_polars() -> None:
    rows = [
        {"event_time": datetime(2024, 1, 1), "security_id": "A", "x": 1.0},
        {"event_time": datetime(2024, 1, 1, tzinfo=UTC), "security_id": "B", "x": 2.0},
    ]
    with pytest.raises(pl.exceptions.SchemaError):
        pl.DataFrame(rows)


def test_sort_for_history_makes_unsorted_match_filter_order() -> None:
    """Wave 10: after HISTORY_SORT_KEYS sort, day-index path matches filter order."""
    rows: list[dict] = []
    for d in [3, 1, 4, 2, 0]:
        t = datetime(2024, 1, 1 + d, tzinfo=UTC)
        for i in range(2):
            rows.append(
                {
                    "event_time": t,
                    "security_id": f"S{1 - i}",  # reverse within day
                    "x": float(d * 10 + i),
                }
            )
    unsorted = pl.DataFrame(rows)
    assert not under_history_sort_contract(unsorted)
    sorted_frame = sort_for_history(unsorted)
    assert under_history_sort_contract(sorted_frame)
    index = build_event_time_day_index(sorted_frame)
    asof = datetime(2024, 1, 3, tzinfo=UTC)
    via_filter = sorted_frame.filter(pl.col("event_time") <= asof)
    via_index = history_upto(sorted_frame, asof, day_index=index)
    _assert_identical(via_index, via_filter)


def test_history_for_calibration_uses_day_index_under_contract() -> None:
    frame = _sorted_panel(n_days=8, n_names=3)
    assert under_history_sort_contract(frame)
    index = build_event_time_day_index(frame)
    asof = frame["event_time"].unique().sort().to_list()[5]
    via_filter = history_for_calibration(frame, asof, horizon_bars=1)
    via_index = history_for_calibration(frame, asof, horizon_bars=1, day_index=index)
    _assert_identical(via_index, via_filter)
    assert via_index.height > 0


def test_history_for_calibration_skips_day_index_when_unsorted() -> None:
    """Unsorted + day_index must not reorder vs explicit filter path."""
    # Later day first so filter order (d=1 then d=0) differs from chrono concat.
    rows: list[dict] = []
    for d in [1, 0, 3, 2, 4]:
        t = datetime(2024, 1, 1 + d, tzinfo=UTC)
        for i in range(2):
            rows.append(
                {
                    "event_time": t,
                    "security_id": f"S{i}",
                    "x": float(d * 10 + i),
                }
            )
    unsorted = pl.DataFrame(rows)
    assert not under_history_sort_contract(unsorted)
    bad_index = build_event_time_day_index(unsorted)
    asof = datetime(2024, 1, 4, tzinfo=UTC)
    via_safe = history_for_calibration(unsorted, asof, horizon_bars=1, day_index=bad_index)
    via_filter = history_for_calibration(unsorted, asof, horizon_bars=1)
    _assert_identical(via_safe, via_filter)
    times = unsorted["event_time"].unique().sort().to_list()
    idx = next(i for i, t in enumerate(times) if t >= asof)
    cutoff = times[idx - 1 - 1]  # horizon_bars=1 → last = idx - 2
    blind = history_upto(unsorted, cutoff, day_index=bad_index)
    # Blind chrono-concat reorders vs filter-preserving fallback.
    assert blind["x"].to_list() != via_safe["x"].to_list()
    assert sorted(blind["x"].to_list()) == sorted(via_safe["x"].to_list())


def test_history_prefix_upto_matches_filter_sorted() -> None:
    """Wave 12: prefix slice ≡ filter under HISTORY_SORT_KEYS contract."""
    from quant_fund.pipeline.forecast import history_prefix_upto

    frame = _sorted_panel(n_days=6, n_names=4)
    assert under_history_sort_contract(frame)
    times = frame["event_time"].unique().sort().to_list()
    for asof in times:
        via_filter = frame.filter(pl.col("event_time") <= asof)
        via_prefix = history_prefix_upto(frame, asof)
        _assert_identical(via_prefix, via_filter)


def test_history_upto_sorted_uses_prefix_not_reorder() -> None:
    """Wave 12 fast path: sorted + day_index still matches filter order."""
    frame = _sorted_panel(n_days=5, n_names=3)
    index = build_event_time_day_index(frame)
    asof = frame["event_time"].unique().sort().to_list()[2]
    via_filter = frame.filter(pl.col("event_time") <= asof)
    via_index = history_upto(frame, asof, day_index=index)
    via_assume = history_upto(frame, asof, day_index=index, assume_sorted=True)
    _assert_identical(via_index, via_filter)
    _assert_identical(via_assume, via_filter)


def test_history_for_calibration_assume_sorted_matches_filter() -> None:
    """Wave 12/43: assume_sorted + event_times on sorted frame matches filter."""
    frame = _sorted_panel(n_days=8, n_names=3)
    times = frame["event_time"].unique().sort().to_list()
    asof = times[5]
    via_filter = history_for_calibration(frame, asof, horizon_bars=1)
    via_fast = history_for_calibration(
        frame, asof, horizon_bars=1, assume_sorted=True, event_times=times
    )
    _assert_identical(via_fast, via_filter)
    assert via_fast.height > 0


def test_history_for_calibration_rejects_negative_horizon() -> None:
    frame = _sorted_panel()
    asof = frame["event_time"].unique().sort().to_list()[2]
    with pytest.raises(ValueError, match="horizon_bars must be a non-negative integer"):
        history_for_calibration(frame, asof, horizon_bars=-1)


@pytest.mark.parametrize("horizon_bars", [0.5, float("nan"), float("inf")])
def test_history_for_calibration_rejects_non_integral_horizon(horizon_bars) -> None:
    frame = _sorted_panel()
    asof = frame["event_time"].unique().sort().to_list()[2]
    with pytest.raises(ValueError, match="horizon_bars must be a non-negative integer"):
        history_for_calibration(frame, asof, horizon_bars=horizon_bars)


@pytest.mark.parametrize(
    "event_times",
    [
        [datetime(2024, 1, 2, tzinfo=UTC), datetime(2024, 1, 1, tzinfo=UTC)],
        [datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 1, tzinfo=UTC)],
    ],
)
def test_history_for_calibration_rejects_non_strict_event_times(event_times) -> None:
    frame = _sorted_panel()
    asof = frame["event_time"].unique().sort().to_list()[2]
    with pytest.raises(ValueError, match="event_times must be strictly increasing"):
        history_for_calibration(frame, asof, horizon_bars=1, event_times=event_times)


def test_history_for_calibration_rejects_non_datetime_event_times() -> None:
    frame = _sorted_panel()
    asof = frame["event_time"].unique().sort().to_list()[2]
    with pytest.raises(ValueError, match="event_times must contain datetime values"):
        history_for_calibration(frame, asof, horizon_bars=1, event_times=["2024-01-01"])


def _unsorted_panel() -> pl.DataFrame:
    rows: list[dict] = []
    for d in [3, 1, 4, 2, 0]:
        t = datetime(2024, 1, 1 + d, tzinfo=UTC)
        for i in range(2):
            rows.append(
                {
                    "event_time": t,
                    "security_id": f"S{i}",
                    "x": float(d * 10 + i),
                }
            )
    return pl.DataFrame(rows)


def test_history_upto_assume_sorted_unsorted_raises() -> None:
    """Wave 43: assume_sorted=True + day_index on unsorted frame fails closed."""
    frame = _unsorted_panel()
    assert not under_history_sort_contract(frame)
    index = build_event_time_day_index(frame)
    asof = datetime(2024, 1, 3, tzinfo=UTC)
    with pytest.raises(ValueError, match="assume_sorted=True requires under_history_sort_contract"):
        history_upto(frame, asof, day_index=index, assume_sorted=True)


def test_history_upto_assume_sorted_sorted_matches_filter() -> None:
    """Wave 43: sorted + assume_sorted=True + day_index still matches filter."""
    frame = _sorted_panel(n_days=5, n_names=3)
    assert under_history_sort_contract(frame)
    index = build_event_time_day_index(frame)
    asof = frame["event_time"].unique().sort().to_list()[2]
    via_filter = frame.filter(pl.col("event_time") <= asof)
    via_assume = history_upto(frame, asof, day_index=index, assume_sorted=True)
    _assert_identical(via_assume, via_filter)


def test_history_upto_no_day_index_unsorted_unchanged() -> None:
    """Wave 43: no day_index path stays filter (assume_sorted ignored for prefix)."""
    frame = _unsorted_panel()
    asof = datetime(2024, 1, 3, tzinfo=UTC)
    via_plain = history_upto(frame, asof)
    via_assume = history_upto(frame, asof, assume_sorted=True)
    via_filter = frame.filter(pl.col("event_time") <= asof)
    _assert_identical(via_plain, via_filter)
    _assert_identical(via_assume, via_filter)


def test_history_for_calibration_assume_sorted_unsorted_raises() -> None:
    """Wave 43: assume_sorted=True on unsorted calibration history fails closed."""
    frame = _unsorted_panel()
    assert not under_history_sort_contract(frame)
    asof = datetime(2024, 1, 4, tzinfo=UTC)
    times = frame["event_time"].unique().sort().to_list()
    with pytest.raises(ValueError, match="assume_sorted=True requires under_history_sort_contract"):
        history_for_calibration(frame, asof, horizon_bars=1, assume_sorted=True, event_times=times)


def test_history_upto_assume_sorted_empty_ok() -> None:
    """Wave 43: empty frame + assume_sorted=True does not raise."""
    frame = _sorted_panel().head(0)
    index = build_event_time_day_index(_sorted_panel())  # nonempty index keys ok
    asof = datetime(2024, 1, 3, tzinfo=UTC)
    out = history_upto(frame, asof, day_index=index, assume_sorted=True)
    assert out.height == 0


def test_history_prefix_upto_rejects_non_datetime_asof() -> None:
    """Wave 54: non-datetime asof fail-closed."""
    from quant_fund.pipeline.forecast import history_prefix_upto

    frame = _sorted_panel()
    with pytest.raises(TypeError, match="asof must be a datetime"):
        history_prefix_upto(frame, "2020-01-01")  # type: ignore[arg-type]
