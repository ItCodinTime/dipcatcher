"""Vendor-shaped L2 book panel + VPIN / queue imbalance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.order_book import (
    ParquetOrderBookProvider,
    SyntheticOrderBookProvider,
)
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.book_panel import (
    BOOK_PANEL_REQUIRED,
    load_book_panel,
    validate_book_panel,
    write_book_panel,
)
from quant_fund.northset.benches import bench_northset
from quant_fund.northset.estimators import queue_imbalance, vpin_proxy
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent


def _bars(n_assets: int = 6, n_days: int = 35, seed: int = 19):
    return SyntheticMarketProvider(n_assets=n_assets, n_days=n_days, seed=seed).get_bars()


def test_synthetic_panel_validates_and_roundtrips(tmp_path: Path) -> None:
    bars = _bars()
    panel = SyntheticOrderBookProvider(depth=5, seed=19).get_book_panel(bars)
    for col in BOOK_PANEL_REQUIRED:
        assert col in panel.columns
    path = write_book_panel(panel, tmp_path / "l2.parquet")
    loaded = load_book_panel(path)
    assert loaded.height == panel.height
    again = ParquetOrderBookProvider(path).get_book_panel()
    assert again.height == panel.height


def test_validate_rejects_crossed_book() -> None:
    bars = _bars(n_assets=2, n_days=20, seed=2)
    panel = SyntheticOrderBookProvider(depth=3, seed=2).get_book_panel(bars)
    import polars as pl

    bad = panel.with_columns(pl.col("best_bid").alias("best_ask"))
    with pytest.raises(ValueError, match="crossed"):
        validate_book_panel(bad)


def test_validate_rejects_null_and_duplicate_pit_rows() -> None:
    import polars as pl

    panel = SyntheticOrderBookProvider(depth=3, seed=2).get_book_panel(
        _bars(n_assets=2, n_days=20, seed=2)
    )
    null_availability = panel.with_columns(pl.lit(None).alias("available_time"))
    with pytest.raises(ValueError, match="available_time"):
        validate_book_panel(null_availability)
    duplicate = pl.concat([panel, panel.slice(0, 1)])
    with pytest.raises(ValueError, match="duplicate"):
        validate_book_panel(duplicate)


def test_delayed_book_is_selected_only_after_it_becomes_available() -> None:
    import polars as pl

    from quant_fund.microstructure.candle_book_features import (
        attach_candle_book_features,
    )

    bars = _bars(n_assets=2, n_days=20, seed=4)
    panel = (
        SyntheticOrderBookProvider(depth=3, seed=4)
        .get_book_panel(bars)
        .with_columns((pl.col("available_time") + pl.duration(hours=1)).alias("available_time"))
    )
    fused = attach_candle_book_features(
        bars,
        book=panel,
        depth=3,
        min_join_coverage=0.9,
        max_book_age_seconds=4 * 86_400,
    )
    assert fused.height == bars.height - bars["security_id"].n_unique()
    assert (fused["book_available_time"] <= fused["decision_time"]).all()
    assert (fused["book_event_time"] < fused["event_time"]).all()


def test_queue_and_vpin_finite() -> None:
    bars = _bars()
    book = SyntheticOrderBookProvider(depth=5, seed=7).get_book_panel(bars)
    book = queue_imbalance(book)
    book = vpin_proxy(book, window=12)
    assert "queue_imbalance" in book.columns
    assert "vpin" in book.columns
    qi = book["queue_imbalance"].drop_nulls()
    assert len(qi) > 0
    assert float(qi.abs().max()) <= 1.0 + 1e-9


def test_vpin_volume_bucket_differs_from_count_window() -> None:
    bars = _bars()
    book = SyntheticOrderBookProvider(depth=5, seed=7).get_book_panel(bars)
    count = vpin_proxy(book, window=12)
    volume = vpin_proxy(book, window=12, bucket_volume=500.0)
    c = count["vpin"].drop_nulls().to_numpy().astype(float)
    v = volume["vpin"].drop_nulls().to_numpy().astype(float)
    assert c.size > 0 and v.size > 0
    assert float(np.nanmax(np.abs(v))) <= 1.0 + 1e-9
    assert not np.allclose(c[: min(c.size, v.size)], v[: min(c.size, v.size)])


def test_bench_northset_with_parquet_book(tmp_path: Path) -> None:
    bars = _bars(n_assets=8, n_days=40, seed=11)
    panel = SyntheticOrderBookProvider(depth=5, seed=11).get_book_panel(bars)
    path = write_book_panel(panel, tmp_path / "vendor_shaped.parquet")
    cfg = AppConfig()
    cfg.northset.require_adjusted_ohlc = False
    cfg.data.source = "synthetic"
    cfg.data.synthetic_seed = 11
    cfg.northset.min_names = 3
    cfg.northset.book_panel_path = str(path)
    receipt = bench_northset(bars, cfg)
    assert receipt["family"] == "northset"
    assert receipt["book_source"] == "synthetic"  # from panel source column
    assert "live_pnl_claim" not in receipt
    assert family_blob_forbidden_metrics_absent(receipt) is True
    assert receipt["vpin_mean"] == receipt["vpin_mean"]  # finite or nan ok if nan!=nan skip
    assert "vpin_p_ic" in receipt
    assert "queue_imbalance_p_ic" in receipt


def test_parquet_order_book_provider_missing_path_fail_closed(tmp_path) -> None:
    import pytest

    from quant_fund.data.adapters.order_book import ParquetOrderBookProvider

    missing = tmp_path / "nope.parquet"
    with pytest.raises(FileNotFoundError, match="not found"):
        ParquetOrderBookProvider(missing).get_book_panel()


def test_load_book_panel_missing_path_fail_closed(tmp_path) -> None:
    import pytest

    from quant_fund.microstructure.book_panel import load_book_panel

    missing = tmp_path / "absent_book.parquet"
    with pytest.raises(FileNotFoundError, match="not found"):
        load_book_panel(missing)


def test_validate_book_panel_depth_honesty_deep_ok() -> None:
    from quant_fund.data.adapters.order_book import SyntheticOrderBookProvider
    from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
    from quant_fund.microstructure.book_panel import validate_book_panel

    bars = SyntheticMarketProvider(n_assets=3, n_days=20, seed=4).get_bars()
    panel = SyntheticOrderBookProvider(depth=5, seed=4).get_book_panel(bars)
    out = validate_book_panel(panel)
    assert out.height == panel.height


def test_validate_book_panel_depth_honesty_fail_deep_nan() -> None:
    import polars as pl
    import pytest

    from quant_fund.data.adapters.order_book import SyntheticOrderBookProvider
    from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
    from quant_fund.microstructure.book_panel import validate_book_panel

    bars = SyntheticMarketProvider(n_assets=3, n_days=20, seed=5).get_bars()
    panel = SyntheticOrderBookProvider(depth=5, seed=5).get_book_panel(bars)
    # Lie: claim depth>=2 but stamp NaN slopes
    lied = panel.with_columns(
        pl.lit(float("nan")).alias("bid_log_size_slope"),
        pl.lit(float("nan")).alias("ask_log_size_slope"),
    )
    with pytest.raises(ValueError, match="depth honesty"):
        validate_book_panel(lied)


def test_validate_book_panel_depth_honesty_fail_thin_finite() -> None:
    import polars as pl
    import pytest

    from quant_fund.data.adapters.order_book import SyntheticOrderBookProvider
    from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
    from quant_fund.microstructure.book_panel import validate_book_panel

    bars = SyntheticMarketProvider(n_assets=3, n_days=20, seed=6).get_bars()
    panel = SyntheticOrderBookProvider(depth=1, seed=6).get_book_panel(bars)
    # Lie: top-only levels but finite slopes
    lied = panel.with_columns(
        pl.lit(-0.3).alias("bid_log_size_slope"),
        pl.lit(-0.2).alias("ask_log_size_slope"),
    )
    with pytest.raises(ValueError, match="depth honesty"):
        validate_book_panel(lied)


def test_book_panel_optional_includes_depth_shape_fields() -> None:
    from quant_fund.microstructure.book_metrics import DEPTH_SHAPE_FIELDS
    from quant_fund.microstructure.book_panel import BOOK_PANEL_OPTIONAL

    for col in DEPTH_SHAPE_FIELDS:
        assert col in BOOK_PANEL_OPTIONAL


def test_depth_honesty_price_slope_thin_finite_fails() -> None:
    import polars as pl
    import pytest

    from quant_fund.data.adapters.order_book import SyntheticOrderBookProvider
    from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
    from quant_fund.microstructure.book_panel import validate_book_panel

    bars = SyntheticMarketProvider(n_assets=3, n_days=20, seed=7).get_bars()
    panel = SyntheticOrderBookProvider(depth=1, seed=7).get_book_panel(bars)
    lied = panel.with_columns(pl.lit(-0.1).alias("bid_log_price_slope"))
    with pytest.raises(ValueError, match="depth honesty"):
        validate_book_panel(lied)


def test_depth_honesty_price_slope_deep_nan_fails() -> None:
    import polars as pl
    import pytest

    from quant_fund.data.adapters.order_book import SyntheticOrderBookProvider
    from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
    from quant_fund.microstructure.book_panel import validate_book_panel

    bars = SyntheticMarketProvider(n_assets=3, n_days=20, seed=8).get_bars()
    panel = SyntheticOrderBookProvider(depth=5, seed=8).get_book_panel(bars)
    lied = panel.with_columns(pl.lit(float("nan")).alias("ask_log_price_slope"))
    with pytest.raises(ValueError, match="depth honesty"):
        validate_book_panel(lied)
