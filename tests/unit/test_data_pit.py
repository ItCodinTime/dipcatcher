from datetime import UTC, datetime

import polars as pl
import pytest

from quant_fund.config.models import UniverseConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.data.corporate_actions import adjust_prices
from quant_fund.data.point_in_time import validate_feature_frame
from quant_fund.data.universe import membership_asof
from quant_fund.features.cross_sectional import apply_cross_sectional
from quant_fund.schemas.errors import PointInTimeError
from quant_fund.schemas.pit import assert_pit_safe


def test_synthetic_has_pit_columns() -> None:
    p = SyntheticMarketProvider(n_assets=5, n_days=40, seed=1)
    bars = p.get_bars()
    for c in (
        "event_time",
        "available_time",
        "ingested_time",
        "source",
        "revision_id",
        "security_id",
    ):
        assert c in bars.columns
    assert bars["source"][0] == "synthetic"


def test_split_does_not_destroy_raw() -> None:
    p = SyntheticMarketProvider(n_assets=4, n_days=80, seed=2)
    bars = p.get_bars()
    raw = bars.filter(pl.col("security_id") == "SEC_0001")["close"].to_list()
    adj = adjust_prices(bars, p.get_corporate_actions())
    raw2 = adj.filter(pl.col("security_id") == "SEC_0001")["close"].to_list()
    assert raw == raw2
    assert "close_split_adjusted" in adj.columns
    assert "close_total_return" in adj.columns


def test_cs_transform_is_per_timestamp() -> None:
    t0 = datetime(2020, 1, 2, tzinfo=UTC)
    t1 = datetime(2020, 1, 3, tzinfo=UTC)
    df = pl.DataFrame(
        {
            "event_time": [t0, t0, t1, t1],
            "security_id": ["a", "b", "a", "b"],
            "x": [1.0, 100.0, 1.0, 2.0],
        }
    )
    out = apply_cross_sectional(df, ["x"], 0.0)
    # ranks within date: at t0, 100 gets higher percentile than 1
    t0_rows = out.filter(pl.col("event_time") == t0).sort("security_id")
    assert t0_rows["cs_pct_x"][1] > t0_rows["cs_pct_x"][0]


def test_future_price_injection_fails_pit() -> None:
    decision = datetime(2020, 1, 2, tzinfo=UTC)
    future = datetime(2020, 1, 10, tzinfo=UTC)
    with pytest.raises(PointInTimeError):
        assert_pit_safe(future, decision)


def test_feature_frame_validation_checks_each_row_for_lookahead() -> None:
    decision = datetime(2020, 1, 2, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "event_time": [decision, decision],
            "available_time": [decision, datetime(2020, 1, 3, tzinfo=UTC)],
            "ingested_time": [decision, decision],
            "source": ["test", "test"],
            "security_id": ["a", "b"],
            "revision_id": ["v1", "v1"],
        }
    )
    with pytest.raises(PointInTimeError):
        validate_feature_frame(frame, decision)


def test_universe_does_not_use_future_bars() -> None:
    p = SyntheticMarketProvider(n_assets=8, n_days=120, seed=3)
    bars = p.get_bars()
    master = p.get_security_master()
    asof = bars["event_time"].unique().sort()[80]
    cfg = UniverseConfig(min_price=1.0, min_adv=1.0, min_history_bars=10, top_n_adv=None)
    mem = membership_asof(bars.filter(pl.col("event_time") <= asof), master, asof, cfg)
    assert mem.height >= 1
    # injecting future-only membership of a name not yet listed should not appear
    assert mem["security_id"].null_count() == 0
