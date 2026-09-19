"""Wave 28: build_labels empty / missing OHLCV fail-closed."""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from quant_fund.config.models import AppConfig, HorizonConfig
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
