"""Wave 8: event_time day-index helper matches filter(event_time == asof)."""

from datetime import UTC, datetime

import polars as pl
import pytest

from quant_fund.pipeline.forecast import build_event_time_day_index, slice_day


def _panel(n_days: int = 5, n_names: int = 3) -> pl.DataFrame:
    rows: list[dict] = []
    for d in range(n_days):
        t = datetime(2024, 1, 1 + d, tzinfo=UTC)
        for i in range(n_names):
            rows.append(
                {
                    "event_time": t,
                    "security_id": f"S{i}",
                    "x": float(d * 10 + i),
                }
            )
    return pl.DataFrame(rows)


def test_day_index_matches_filter_equality() -> None:
    frame = _panel()
    index = build_event_time_day_index(frame)
    times = frame["event_time"].unique().sort().to_list()
    assert len(index) == len(times)
    for t in times:
        via_filter = frame.filter(pl.col("event_time") == t).sort("security_id")
        via_index = slice_day(frame, t, day_index=index).sort("security_id")
        assert via_index.height == via_filter.height
        assert via_index["security_id"].to_list() == via_filter["security_id"].to_list()
        assert via_index["x"].to_list() == pytest.approx(via_filter["x"].to_list())


def test_slice_day_without_index_equals_filter() -> None:
    frame = _panel()
    t = datetime(2024, 1, 3, tzinfo=UTC)
    a = slice_day(frame, t).sort("security_id")
    b = frame.filter(pl.col("event_time") == t).sort("security_id")
    assert a.height == b.height == 3
    assert a["x"].to_list() == b["x"].to_list()


def test_slice_day_miss_returns_empty_schema() -> None:
    frame = _panel()
    index = build_event_time_day_index(frame)
    miss = datetime(2099, 1, 1, tzinfo=UTC)
    empty = slice_day(frame, miss, day_index=index)
    assert empty.height == 0
    assert empty.columns == frame.columns


def test_empty_frame_index() -> None:
    frame = pl.DataFrame(
        schema={"event_time": pl.Datetime(time_zone="UTC"), "security_id": pl.Utf8, "x": pl.Float64}
    )
    assert build_event_time_day_index(frame) == {}
