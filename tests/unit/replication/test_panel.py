"""Panel construction and loading tests (synthetic known answers)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from quant_fund.research.replication.panel import (
    PANEL_COLUMNS,
    make_panel,
    market_returns,
    validate_panel,
)
from quant_fund.research.replication.synthetic import synthetic_panel, trading_days


def _toy_ohlc() -> pl.DataFrame:
    """Two securities x 3 bars with hand-computable returns."""
    return pl.DataFrame(
        {
            "security_id": ["A", "A", "A", "B", "B", "B"],
            "symbol": ["A", "A", "A", "B", "B", "B"],
            "event_time": [
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
            ],
            "open": [101.0, 105.0, 99.0, 50.0, 55.0, 44.0],
            "close": [105.0, 99.0, 110.0, 55.0, 44.0, 66.0],
        }
    ).with_columns(pl.col("event_time").str.to_datetime())


def test_make_panel_derived_returns() -> None:
    panel = make_panel(_toy_ohlc())
    assert panel.columns == list(PANEL_COLUMNS)
    a = panel.filter(pl.col("security_id") == "A")
    # ret_cc: 105/105-1 row0 null; 99/105-1; 110/99-1
    assert a["ret_cc"][0] is None
    assert a["ret_cc"][1] == pytest.approx(99.0 / 105.0 - 1.0)
    assert a["ret_cc"][2] == pytest.approx(110.0 / 99.0 - 1.0)
    # ret_on: open_t / close_{t-1} - 1
    assert a["ret_on"][0] is None
    assert a["ret_on"][1] == pytest.approx(105.0 / 105.0 - 1.0)
    assert a["ret_on"][2] == pytest.approx(99.0 / 99.0 - 1.0)
    # ret_id: close/open - 1
    assert a["ret_id"][0] == pytest.approx(105.0 / 101.0 - 1.0)


def test_validate_panel_rejects_duplicates() -> None:
    panel = make_panel(_toy_ohlc())
    dup = pl.concat([panel, panel.head(1)])
    with pytest.raises(ValueError, match="duplicate"):
        validate_panel(dup)


def test_validate_panel_rejects_nonpositive_price() -> None:
    bad = _toy_ohlc().with_columns(
        pl.when(pl.col("security_id") == "B").then(-1.0).otherwise(pl.col("open")).alias("open")
    )
    with pytest.raises(ValueError, match="non-positive"):
        make_panel(bad)


def test_market_returns_equal_weight() -> None:
    panel = make_panel(_toy_ohlc())
    mkt = market_returns(panel)
    assert mkt.height == 3
    day2 = mkt.filter(pl.col("event_time").dt.day() == 3)
    expected = ((99.0 / 105.0 - 1.0) + (44.0 / 55.0 - 1.0)) / 2.0
    assert day2["mkt_ret_cc"][0] == pytest.approx(expected)
    assert day2["n_assets"][0] == 2


def test_synthetic_panel_deterministic_and_labeled() -> None:
    a = synthetic_panel(n_assets=4, n_days=80, seed=7)
    b = synthetic_panel(n_assets=4, n_days=80, seed=7)
    c = synthetic_panel(n_assets=4, n_days=80, seed=8)
    assert a.equals(b)
    assert not a.equals(c)
    assert a.height == 4 * 80


def test_synthetic_panel_ohlc_identity() -> None:
    panel = synthetic_panel(n_assets=4, n_days=60, seed=3, overnight_drift=0.8)
    # log(1+ret_on) + log(1+ret_id) == log(1+ret_cc) up to fp noise
    sub = panel.drop_nulls()
    lhs = np.log1p(sub["ret_on"].to_numpy()) + np.log1p(sub["ret_id"].to_numpy())
    rhs = np.log1p(sub["ret_cc"].to_numpy())
    np.testing.assert_allclose(lhs, rhs, atol=1e-10)


def test_trading_days_are_weekdays() -> None:
    days = trading_days("2024-01-06", 7)  # starts Saturday
    import pandas as pd

    idx = pd.DatetimeIndex(days)
    assert (idx.dayofweek < 5).all()
