"""Session multi-snap book write helpers (CLI substrate)."""

from __future__ import annotations

from pathlib import Path

from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.book_panel import validate_book_panel
from quant_fund.microstructure.synthetic_lob import (
    aggregate_session_book_to_daily,
    synthesize_session_l2,
)
from quant_fund.northset.identities import session_candles_from_daily


def test_session_book_parquet_roundtrip(tmp_path: Path) -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=20, seed=4).get_bars()
    session = session_candles_from_daily(bars, n_candles=8, seed=4)
    book = synthesize_session_l2(session, depth=5, seed=4)
    path = tmp_path / "session_l2.parquet"
    book.write_parquet(path)
    import polars as pl

    loaded = pl.read_parquet(path)
    assert loaded.height == session.height
    assert "parent_event_time" in loaded.columns
    # required panel cols still validate when session keys dropped
    validate_book_panel(
        loaded.drop([c for c in ("parent_event_time", "session_index") if c in loaded.columns])
    )
    daily = aggregate_session_book_to_daily(loaded)
    daily_path = tmp_path / "session_daily.parquet"
    daily.write_parquet(daily_path)
    assert pl.read_parquet(daily_path).height == daily.height
