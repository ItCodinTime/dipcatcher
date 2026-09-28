"""Wave 28: build_labels empty / missing OHLCV fail-closed."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from quant_fund.config.models import AppConfig, HorizonConfig
from quant_fund.data.point_in_time import PointInTimeError
from quant_fund.labels.engine import build_labels


def test_build_labels_empty_bars_raise() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        build_labels(pl.DataFrame(), AppConfig())


def test_build_labels_missing_ohlcv_raise() -> None:
    bars = pl.DataFrame(
        {
            "security_id": ["A"],
            "event_time": [datetime(2024, 1, 2)],
        }
    )
    with pytest.raises(ValueError, match="missing required OHLCV"):
        build_labels(bars, AppConfig())


def test_build_labels_minimal_row_runs() -> None:
    """One complete bar + benchmark calendar is accepted (forward labels may be null)."""
    times = [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)]
    bars = pl.DataFrame(
        {
            "security_id": ["A", "A", "A", "SEC_MKT", "SEC_MKT", "SEC_MKT"],
            "event_time": times + times,
            "close_total_return": [100.0, 101.0, 102.0, 100.0, 100.5, 101.0],
        }
    )
    cfg = AppConfig(horizons=HorizonConfig(bars=[1], names=["1d"]))
    out = build_labels(bars, cfg)
    assert out.height == 6
    assert "future_return_1" in out.columns
    assert "future_excess_return_1" in out.columns
    assert "label_end_time_1" in out.columns
    a = out.filter(pl.col("security_id") == "A").sort("event_time")
    assert a.get_column("label_end_time_1").to_list() == [times[1], times[2], None]


def test_build_labels_missing_benchmark_fails_closed() -> None:
    """A tape without the configured benchmark yields all-null
    future_excess_return_* labels; fail at label build, not downstream."""
    times = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(40)]
    bars = pl.DataFrame(
        {
            "security_id": ["A"] * 40,
            "event_time": times,
            "close_total_return": [100.0 + i for i in range(40)],
        }
    )
    with pytest.raises(PointInTimeError, match="benchmark_id='SEC_MKT' has no rows"):
        build_labels(bars, AppConfig(horizons=HorizonConfig(bars=[1], names=["1d"])))


def test_build_labels_thin_benchmark_fails_closed() -> None:
    """A benchmark with <= min-horizon bars makes every excess label degenerate."""
    times = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(40)]
    bars = pl.DataFrame(
        {
            "security_id": ["A"] * 40 + ["SEC_MKT"] * 3,
            "event_time": times + times[:3],
            "close_total_return": [100.0 + i for i in range(40)] + [100.0, 100.5, 101.0],
        }
    )
    cfg = AppConfig(horizons=HorizonConfig(bars=[5, 10], names=["5d", "10d"]))
    with pytest.raises(PointInTimeError, match="needs more than the min horizon"):
        build_labels(bars, cfg)


def test_build_labels_short_benchmark_keeps_shorter_horizons() -> None:
    """A benchmark that can serve h=5 but not h=20 still builds labels; the
    under-served horizon stays sparse rather than failing the whole frame."""
    times = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(12)]
    bars = pl.DataFrame(
        {
            "security_id": ["A"] * 12 + ["SEC_MKT"] * 12,
            "event_time": times * 2,
            "close_total_return": [100.0 + i for i in range(12)] * 2,
        }
    )
    cfg = AppConfig(horizons=HorizonConfig(bars=[5, 20], names=["5d", "20d"]))
    out = build_labels(bars, cfg)
    assert "future_excess_return_5" in out.columns
