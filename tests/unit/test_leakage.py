from datetime import UTC, datetime

import polars as pl
import pytest

from quant_fund.schemas.errors import PointInTimeError
from quant_fund.schemas.pit import assert_pit_safe


def test_future_fundamental_injection() -> None:
    decision = datetime(2020, 6, 1, tzinfo=UTC)
    filing_available = datetime(2020, 8, 1, tzinfo=UTC)
    with pytest.raises(PointInTimeError):
        assert_pit_safe(filing_available, decision)


def test_future_cs_normalization_would_use_full_sample() -> None:
    """Full-sample z-score differs from per-date z-score — detector for leakage pattern."""
    t0 = datetime(2020, 1, 2, tzinfo=UTC)
    t1 = datetime(2020, 1, 3, tzinfo=UTC)
    df = pl.DataFrame({"event_time": [t0, t0, t1, t1], "x": [0.0, 1.0, 100.0, 101.0]})
    global_mean = df["x"].mean()
    per_date = df.group_by("event_time").agg(pl.col("x").mean().alias("m"))
    t0_mean = per_date.filter(pl.col("event_time") == t0)["m"][0]
    assert global_mean != t0_mean
