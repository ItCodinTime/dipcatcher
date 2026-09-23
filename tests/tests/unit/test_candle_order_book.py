"""Candle + order-book research path — real fused features, not wave theater."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import polars as pl
import pytest

from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure import (
    attach_candle_book_features,
    bench_candle_order_book,
    book_metrics_from_snapshot,
    microprice,
    synthesize_l2_from_bars,
    synthesize_snapshots_from_bars,
)
from quant_fund.microstructure.book_metrics import DEPTH_SHAPE_FIELDS
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent
from quant_fund.schemas.order_book import BookLevel, OrderBookSnapshot


def _bars():
    provider = SyntheticMarketProvider(n_assets=6, n_days=40, seed=11)
    if hasattr(provider, "get_bars"):
        return provider.get_bars()
    return provider._bars


def test_order_book_snapshot_rejects_crossed_book() -> None:
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="crossed or locked"):
        OrderBookSnapshot(
            security_id="A",
            event_time=ts,
            available_time=ts,
            bids=[BookLevel(price=10.0, size=1.0)],
            asks=[BookLevel(price=9.5, size=1.0)],
            depth=1,
        )


def test_synthesize_l2_aligned_to_bars() -> None:
    bars = _bars()
    snaps = synthesize_snapshots_from_bars(bars, depth=5, seed=11)
    assert len(snaps) == bars.height
    book = synthesize_l2_from_bars(bars, depth=5, seed=11)
    assert book.height == bars.height
    assert "imbalance_top" in book.columns
    assert "spread_bps" in book.columns
    # no crossed books
    assert (book["best_bid"] < book["best_ask"]).all()


def test_microprice_between_bid_ask() -> None:
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    snap = OrderBookSnapshot(
        security_id="A",
        event_time=ts,
        available_time=ts,
        bids=[BookLevel(price=99.0, size=3.0), BookLevel(price=98.5, size=2.0)],
        asks=[BookLevel(price=101.0, size=1.0), BookLevel(price=101.5, size=2.0)],
        depth=2,
    )
    mp = microprice(snap)
    assert snap.best_bid < mp < snap.best_ask
    metrics = book_metrics_from_snapshot(snap)
    assert metrics["spread_bps"] > 0
    assert -1.0 <= metrics["imbalance_top"] <= 1.0


def test_attach_candle_book_features_interactions() -> None:
    bars = _bars()
    fused = attach_candle_book_features(bars, depth=5, seed=11)
    assert fused.height == bars.height
    for col in (
        "candle_body_ret",
        "imbalance_top",
        "spread_bps",
        "candle_dir_x_imbalance",
        "spread_x_range",
        "imbalance_x_body_frac",
        "signed_vol_x_imbalance",
    ):
        assert col in fused.columns


def test_bench_candle_order_book_honesty() -> None:
    bars = _bars()
    receipt = bench_candle_order_book(bars, depth=5, seed=11, label="SYNTHETIC")
    assert receipt["family"] == "candle_order_book"
    assert receipt["research_only"] is True
    assert "live_pnl_claim" not in receipt  # pnl token forbidden in research blobs
    assert receipt["n_scored"] > 0
    assert family_blob_forbidden_metrics_absent(receipt) is True
    assert "sharpe" not in {k.lower() for k in receipt}


def test_bench_candle_order_book_date_level_ic() -> None:
    """Institutional path: date-level IC + HAC keys, not pooled-only."""
    bars = _bars()
    receipt = bench_candle_order_book(bars, depth=5, seed=11, label="SYNTHETIC", min_names=3)
    assert receipt["ic_method"] == "date_level_spearman_hac"
    assert receipt["dgp"] == "synthetic_lob"
    assert "ic_imbalance_top_t" in receipt
    assert "ic_imbalance_top_p" in receipt
    assert "ic_imbalance_top_n_dates" in receipt
    assert float(receipt["ic_imbalance_top_n_dates"]) >= 1.0


def test_attach_preserves_book_source_honesty() -> None:

    bars = _bars()
    fused_syn = attach_candle_book_features(bars, depth=5, seed=11)
    assert fused_syn["book_source"][0] == "synthetic_lob"
    assert fused_syn["book_dgp"][0] == "synthetic_lob"

    book = synthesize_l2_from_bars(bars, depth=5, seed=11).with_columns(
        pl.lit("alpaca").alias("source")
    )
    fused_ext = attach_candle_book_features(bars, book=book)
    assert fused_ext["book_source"][0] == "alpaca"
    assert fused_ext["book_dgp"][0] == "external_panel"


def test_attach_rejects_mixed_book_sources() -> None:

    bars = _bars()
    book = synthesize_l2_from_bars(bars, depth=5, seed=11)
    # Force two sources
    n = book.height
    sources = ["alpaca"] * (n // 2) + ["polygon"] * (n - n // 2)
    book = book.with_columns(pl.Series("source", sources))
    with pytest.raises(ValueError, match="mixes sources"):
        attach_candle_book_features(bars, book=book)


def test_attach_rejects_external_book_without_source() -> None:
    bars = _bars()
    book = synthesize_l2_from_bars(bars, depth=5, seed=11).drop("source")
    with pytest.raises(ValueError, match="source"):
        attach_candle_book_features(bars, book=book)


def test_attach_join_coverage_synthetic_full() -> None:
    from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
    from quant_fund.microstructure.candle_book_features import attach_candle_book_features

    bars = SyntheticMarketProvider(n_assets=4, n_days=20, seed=3).get_bars()
    fused = attach_candle_book_features(bars, depth=3, seed=3)
    assert "join_coverage" in fused.columns
    assert abs(float(fused["join_coverage"][0]) - 1.0) < 1e-12


def test_attach_join_coverage_fail_closed_external() -> None:
    import pytest

    from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
    from quant_fund.microstructure.candle_book_features import attach_candle_book_features
    from quant_fund.microstructure.synthetic_lob import synthesize_l2_from_bars

    bars = SyntheticMarketProvider(n_assets=4, n_days=24, seed=8).get_bars()
    book = synthesize_l2_from_bars(bars, depth=3, seed=8).with_columns(
        pl.lit("alpaca").alias("source")
    )
    # Keep only first ~10% of book rows → coverage collapse
    slim = book.head(max(1, book.height // 10))
    with pytest.raises(ValueError, match="join coverage"):
        attach_candle_book_features(bars, book=slim, min_join_coverage=0.5)


def test_log_size_slopes_multi_level_vs_top_only() -> None:
    """SYNTHETIC depth>1 → finite slopes; top-only → NaN (distinct contract)."""
    import math

    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    top = OrderBookSnapshot(
        security_id="A",
        event_time=ts,
        available_time=ts,
        bids=[BookLevel(price=99.0, size=3.0)],
        asks=[BookLevel(price=101.0, size=1.0)],
        depth=1,
    )
    deep = OrderBookSnapshot(
        security_id="A",
        event_time=ts,
        available_time=ts,
        bids=[
            BookLevel(price=99.0, size=math.exp(1.0)),
            BookLevel(price=98.5, size=math.exp(0.7)),
            BookLevel(price=98.0, size=math.exp(0.4)),
        ],
        asks=[
            BookLevel(price=101.0, size=math.exp(0.5)),
            BookLevel(price=101.5, size=math.exp(0.3)),
            BookLevel(price=102.0, size=math.exp(0.1)),
        ],
        depth=3,
    )
    m_top = book_metrics_from_snapshot(top)
    m_deep = book_metrics_from_snapshot(deep)
    assert m_top["n_bid_levels"] == 1.0 and m_top["n_ask_levels"] == 1.0
    assert math.isnan(m_top["bid_log_size_slope"]) and math.isnan(m_top["ask_log_size_slope"])
    assert m_deep["n_bid_levels"] == 3.0 and m_deep["n_ask_levels"] == 3.0
    assert math.isfinite(m_deep["bid_log_size_slope"])
    assert math.isfinite(m_deep["ask_log_size_slope"])
    # Distinct from top-only NaN contract
    assert m_deep["bid_log_size_slope"] != m_top["bid_log_size_slope"]
    assert m_deep["ask_log_size_slope"] != m_top["ask_log_size_slope"]
    # Known OLS: log sizes 1.0, 0.7, 0.4 → slope -0.3
    assert abs(m_deep["bid_log_size_slope"] - (-0.3)) < 1e-9
    assert abs(m_deep["ask_log_size_slope"] - (-0.2)) < 1e-9


def test_synthetic_panel_depth_slopes_finite_and_not_top_only() -> None:
    """SyntheticOrderBookProvider depth>1 stamps real log-size slopes."""
    import math

    from quant_fund.data.adapters.order_book import SyntheticOrderBookProvider

    bars = SyntheticMarketProvider(n_assets=4, n_days=20, seed=11).get_bars()
    deep = SyntheticOrderBookProvider(depth=5, seed=11).get_book_panel(bars)
    top = SyntheticOrderBookProvider(depth=1, seed=11).get_book_panel(bars)
    assert float(deep["n_bid_levels"].min()) >= 2.0
    assert float(deep["n_ask_levels"].min()) >= 2.0
    bid_s = [float(x) for x in deep["bid_log_size_slope"]]
    ask_s = [float(x) for x in deep["ask_log_size_slope"]]
    assert all(math.isfinite(x) for x in bid_s)
    assert all(math.isfinite(x) for x in ask_s)
    top_bid = [float(x) for x in top["bid_log_size_slope"]]
    assert all(math.isnan(x) for x in top_bid)
    # Multi-level slopes are not the top-only NaN sentinel
    assert any(math.isfinite(x) and math.isnan(y) for x, y in zip(bid_s, top_bid, strict=True))


def test_order_book_snapshot_rejects_empty_side() -> None:
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="non-empty"):
        OrderBookSnapshot(
            security_id="A",
            event_time=ts,
            available_time=ts,
            bids=[],
            asks=[BookLevel(price=101.0, size=1.0)],
            depth=1,
        )


def test_order_book_snapshot_rejects_locked_book() -> None:
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="crossed or locked"):
        OrderBookSnapshot(
            security_id="A",
            event_time=ts,
            available_time=ts,
            bids=[BookLevel(price=100.0, size=1.0)],
            asks=[BookLevel(price=100.0, size=1.0)],
            depth=1,
        )


def test_attach_external_depth_honesty_fail() -> None:
    """External panel with deep n_* but NaN slopes must fail at fuse."""
    import polars as pl
    import pytest

    from quant_fund.data.adapters.order_book import SyntheticOrderBookProvider
    from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
    from quant_fund.microstructure.candle_book_features import attach_candle_book_features

    bars = SyntheticMarketProvider(n_assets=3, n_days=20, seed=9).get_bars()
    panel = SyntheticOrderBookProvider(depth=5, seed=9).get_book_panel(bars)
    lied = panel.with_columns(
        pl.lit(float("nan")).alias("bid_log_size_slope"),
        pl.lit(float("nan")).alias("ask_log_size_slope"),
    )
    with pytest.raises(ValueError, match="depth honesty"):
        attach_candle_book_features(bars, book=lied)


def test_log_price_geometry_multi_level_vs_top_only() -> None:
    """Price geometry: top-only NaN; depth≥2 finite log-price slope + log tick spacing."""
    import math

    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    top = OrderBookSnapshot(
        security_id="A",
        event_time=ts,
        available_time=ts,
        bids=[BookLevel(price=99.0, size=3.0)],
        asks=[BookLevel(price=101.0, size=1.0)],
        depth=1,
    )
    # Bids: prices e^1, e^0.7, e^0.4 → log-price OLS slope -0.3; gaps known
    bid_prices = [math.exp(1.0), math.exp(0.7), math.exp(0.4)]
    ask_prices = [math.exp(1.0), math.exp(1.2), math.exp(1.4)]  # ascending asks
    deep = OrderBookSnapshot(
        security_id="A",
        event_time=ts,
        available_time=ts,
        bids=[
            BookLevel(price=bid_prices[0], size=3.0),
            BookLevel(price=bid_prices[1], size=2.0),
            BookLevel(price=bid_prices[2], size=1.0),
        ],
        asks=[
            BookLevel(price=ask_prices[0] + 1.0, size=1.0),  # keep uncrossed vs bids
            BookLevel(price=ask_prices[1] + 1.0, size=1.0),
            BookLevel(price=ask_prices[2] + 1.0, size=1.0),
        ],
        depth=3,
    )
    # Recompute asks with clean log ladder above best bid
    deep = OrderBookSnapshot(
        security_id="A",
        event_time=ts,
        available_time=ts,
        bids=[
            BookLevel(price=math.exp(1.0), size=3.0),
            BookLevel(price=math.exp(0.7), size=2.0),
            BookLevel(price=math.exp(0.4), size=1.0),
        ],
        asks=[
            BookLevel(price=math.exp(1.5), size=1.0),
            BookLevel(price=math.exp(1.7), size=1.0),
            BookLevel(price=math.exp(1.9), size=1.0),
        ],
        depth=3,
    )
    m_top = book_metrics_from_snapshot(top)
    m_deep = book_metrics_from_snapshot(deep)
    for key in (
        "bid_log_price_slope",
        "ask_log_price_slope",
        "bid_mean_log_tick_spacing",
        "ask_mean_log_tick_spacing",
    ):
        assert math.isnan(m_top[key])
        assert math.isfinite(m_deep[key])
        assert m_deep[key] != m_top[key]
    assert abs(m_deep["bid_log_price_slope"] - (-0.3)) < 1e-9
    assert abs(m_deep["ask_log_price_slope"] - 0.2) < 1e-9
    # Adjacent gaps on bids: |e^0.7-e^1|, |e^0.4-e^0.7|
    bid_gaps = [
        abs(math.exp(0.7) - math.exp(1.0)),
        abs(math.exp(0.4) - math.exp(0.7)),
    ]
    expected_bid_spacing = sum(math.log(g) for g in bid_gaps) / 2.0
    assert abs(m_deep["bid_mean_log_tick_spacing"] - expected_bid_spacing) < 1e-9
    ask_gaps = [
        abs(math.exp(1.7) - math.exp(1.5)),
        abs(math.exp(1.9) - math.exp(1.7)),
    ]
    expected_ask_spacing = sum(math.log(g) for g in ask_gaps) / 2.0
    assert abs(m_deep["ask_mean_log_tick_spacing"] - expected_ask_spacing) < 1e-9


def test_snapshots_to_panel_carries_price_geometry() -> None:
    """snapshots_to_panel spreads new price-geometry fields from metrics."""
    import math

    from quant_fund.microstructure.book_panel import snapshots_to_panel

    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    deep = OrderBookSnapshot(
        security_id="A",
        event_time=ts,
        available_time=ts,
        bids=[
            BookLevel(price=math.exp(1.0), size=3.0),
            BookLevel(price=math.exp(0.7), size=2.0),
            BookLevel(price=math.exp(0.4), size=1.0),
        ],
        asks=[
            BookLevel(price=math.exp(1.5), size=1.0),
            BookLevel(price=math.exp(1.7), size=1.0),
            BookLevel(price=math.exp(1.9), size=1.0),
        ],
        depth=3,
        source="synthetic",
    )
    top = OrderBookSnapshot(
        security_id="B",
        event_time=ts,
        available_time=ts,
        bids=[BookLevel(price=99.0, size=1.0)],
        asks=[BookLevel(price=101.0, size=1.0)],
        depth=1,
        source="synthetic",
    )
    panel = snapshots_to_panel([deep, top])
    for col in (
        "bid_log_price_slope",
        "ask_log_price_slope",
        "bid_mean_log_tick_spacing",
        "ask_mean_log_tick_spacing",
    ):
        assert col in panel.columns
    deep_row = panel.filter(panel["security_id"] == "A")
    top_row = panel.filter(panel["security_id"] == "B")
    assert math.isfinite(float(deep_row["bid_log_price_slope"][0]))
    assert math.isnan(float(top_row["bid_log_price_slope"][0]))
    assert math.isfinite(float(deep_row["bid_mean_log_tick_spacing"][0]))
    assert math.isnan(float(top_row["bid_mean_log_tick_spacing"][0]))


def test_order_book_rejects_tied_adjacent_prices() -> None:
    """Strict ordering: equal adjacent prices fail closed at the schema."""
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="strictly descending"):
        OrderBookSnapshot(
            security_id="A",
            event_time=ts,
            available_time=ts,
            bids=[
                BookLevel(price=100.0, size=2.0),
                BookLevel(price=100.0, size=1.0),
                BookLevel(price=99.0, size=1.0),
            ],
            asks=[
                BookLevel(price=101.0, size=1.0),
                BookLevel(price=102.0, size=1.0),
            ],
            depth=3,
        )
    with pytest.raises(ValueError, match="strictly ascending"):
        OrderBookSnapshot(
            security_id="A",
            event_time=ts,
            available_time=ts,
            bids=[
                BookLevel(price=100.0, size=2.0),
                BookLevel(price=99.0, size=1.0),
            ],
            asks=[
                BookLevel(price=101.0, size=1.0),
                BookLevel(price=101.0, size=1.0),
            ],
            depth=2,
        )


def test_depth_shape_fields_contract_top_vs_deep() -> None:
    """All DEPTH_SHAPE_FIELDS NaN at depth=1 and finite at depth≥2 (positive gaps)."""
    import math

    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    top = OrderBookSnapshot(
        security_id="A",
        event_time=ts,
        available_time=ts,
        bids=[BookLevel(price=99.0, size=1.0)],
        asks=[BookLevel(price=101.0, size=1.0)],
        depth=1,
    )
    deep = OrderBookSnapshot(
        security_id="A",
        event_time=ts,
        available_time=ts,
        bids=[
            BookLevel(price=math.exp(1.0), size=math.exp(1.0)),
            BookLevel(price=math.exp(0.7), size=math.exp(0.7)),
            BookLevel(price=math.exp(0.4), size=math.exp(0.4)),
        ],
        asks=[
            BookLevel(price=math.exp(1.5), size=math.exp(0.5)),
            BookLevel(price=math.exp(1.7), size=math.exp(0.3)),
            BookLevel(price=math.exp(1.9), size=math.exp(0.1)),
        ],
        depth=3,
    )
    m_top = book_metrics_from_snapshot(top)
    m_deep = book_metrics_from_snapshot(deep)
    for key in DEPTH_SHAPE_FIELDS:
        assert key in m_top and key in m_deep
        assert math.isnan(m_top[key])
        assert math.isfinite(m_deep[key])


def test_bench_receipt_has_book_age_stats() -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=20, seed=12).get_bars()
    receipt = bench_candle_order_book(bars, depth=3, seed=12, label="SYNTHETIC", min_names=3)
    assert "mean_book_age_seconds" in receipt
    assert "max_book_age_seconds" in receipt
    assert receipt["mean_book_age_seconds"] == receipt["mean_book_age_seconds"]  # not NaN for syn


def test_bench_scores_depth_shape_price_fields() -> None:
    """SYNTHETIC depth>1 panels expose price-geometry ICs on the receipt."""
    bars = SyntheticMarketProvider(n_assets=6, n_days=40, seed=14).get_bars()
    receipt = bench_candle_order_book(bars, depth=5, seed=14, label="SYNTHETIC", min_names=3)
    for key in (
        "ic_bid_log_price_slope",
        "ic_ask_log_price_slope",
        "ic_bid_mean_log_tick_spacing",
        "ic_ask_mean_log_tick_spacing",
    ):
        assert key in receipt
        assert key + "_n_dates" in receipt


def test_bench_receipt_depth_shape_finite_rate_deep() -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=24, seed=15).get_bars()
    receipt = bench_candle_order_book(bars, depth=5, seed=15, label="SYNTHETIC", min_names=3)
    rate = float(receipt["depth_shape_finite_rate"])
    assert rate == rate  # finite
    assert 0.0 <= rate <= 1.0


def test_attach_stale_book_event_fail_closed() -> None:
    """Asof on available_time can match while book event is older than max age."""
    from datetime import timedelta

    import polars as pl
    import pytest

    from quant_fund.data.adapters.order_book import SyntheticOrderBookProvider
    from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
    from quant_fund.microstructure.candle_book_features import attach_candle_book_features

    bars = SyntheticMarketProvider(n_assets=3, n_days=20, seed=16).get_bars()
    panel = SyntheticOrderBookProvider(depth=3, seed=16).get_book_panel(bars)
    stale = panel.with_columns(
        (pl.col("available_time") - timedelta(days=10)).alias("event_time"),
    )
    with pytest.raises(ValueError, match="book age exceeds"):
        attach_candle_book_features(
            bars, book=stale, min_join_coverage=0.01, max_book_age_seconds=86_400
        )


def test_bench_scores_raw_microprice_minus_mid() -> None:
    """Raw microprice_minus_mid is in FEATURE_COLS and receipt mean is finite on synth."""
    from quant_fund.microstructure.bench import _FEATURE_COLS

    assert "microprice_minus_mid" in _FEATURE_COLS
    bars = SyntheticMarketProvider(n_assets=4, n_days=24, seed=17).get_bars()
    receipt = bench_candle_order_book(bars, depth=5, seed=17, label="SYNTHETIC", min_names=3)
    assert "ic_microprice_minus_mid" in receipt
    assert math.isfinite(float(receipt["mean_microprice_minus_mid"]))
