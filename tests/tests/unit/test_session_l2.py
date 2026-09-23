"""Multi-snapshot session L2 path for Northset."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.synthetic_lob import (
    aggregate_session_book_to_daily,
    synthesize_session_l2,
)
from quant_fund.northset.benches import bench_northset
from quant_fund.northset.identities import (
    session_candles_from_daily,
    validate_session_book_counts,
)
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent


def _bars(n_assets: int = 6, n_days: int = 24, seed: int = 21):
    return SyntheticMarketProvider(n_assets=n_assets, n_days=n_days, seed=seed).get_bars()


def test_session_l2_one_snap_per_session_candle() -> None:
    bars = _bars()
    session = session_candles_from_daily(bars, n_candles=8, seed=21)
    book = synthesize_session_l2(session, depth=5, seed=21)
    assert book.height == session.height
    assert "parent_event_time" in book.columns
    assert "session_index" in book.columns
    assert (book["best_bid"] < book["best_ask"]).all()


def test_aggregate_session_book_path_stats() -> None:
    bars = _bars(n_assets=4, n_days=20, seed=9)
    session = session_candles_from_daily(bars, n_candles=8, seed=9)
    book = synthesize_session_l2(session, depth=5, seed=9)
    daily = aggregate_session_book_to_daily(book)
    assert daily.height > 0
    assert int(daily["n_session_book_snaps"].min()) == 8
    assert "session_ofi_sum" in daily.columns
    assert "session_book_vpin" in daily.columns
    vpin = daily["session_book_vpin"].drop_nulls()
    assert len(vpin) > 0
    assert float(vpin.max()) <= 1.0 + 1e-9


def test_bench_northset_session_l2_ics() -> None:
    bars = _bars(n_assets=8, n_days=40, seed=11)
    cfg = AppConfig()
    cfg.northset.require_adjusted_ohlc = False
    cfg.data.source = "synthetic"
    cfg.data.synthetic_seed = 11
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = True
    cfg.northset.n_session_candles = 8
    receipt = bench_northset(bars, cfg)
    assert receipt["use_session_l2"] is True
    assert receipt["n_session_book_rows"] > 0
    assert receipt["mean_session_book_snaps"] == 8.0
    assert "session_ofi_sum_p_ic" in receipt
    assert "session_book_vpin_p_ic" in receipt
    assert family_blob_forbidden_metrics_absent(receipt) is True
    assert "live_pnl_claim" not in receipt


def test_bench_northset_can_disable_session_l2() -> None:
    bars = _bars(n_assets=6, n_days=30, seed=3)
    cfg = AppConfig()
    cfg.northset.require_adjusted_ohlc = False
    cfg.data.source = "synthetic"
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = False
    receipt = bench_northset(bars, cfg)
    assert receipt["use_session_l2"] is False
    assert receipt["n_session_book_rows"] == 0


def test_validate_session_book_counts_ok() -> None:
    bars = _bars(n_assets=3, n_days=20, seed=2)
    session = session_candles_from_daily(bars, n_candles=8, seed=2)
    book = synthesize_session_l2(session, depth=3, seed=2)
    out = validate_session_book_counts(book, n_session_candles=8)
    assert out.height == book.height


def test_validate_session_book_counts_fail_closed() -> None:
    import pytest

    bars = _bars(n_assets=2, n_days=20, seed=5)
    session = session_candles_from_daily(bars, n_candles=8, seed=5)
    book = synthesize_session_l2(session, depth=3, seed=5)
    # Drop one snap → mismatch
    trimmed = book.head(book.height - 1)
    with pytest.raises(ValueError, match="count mismatch"):
        validate_session_book_counts(trimmed, n_session_candles=8)
