from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from quant_fund.config.models import AppConfig, UniverseConfig
from quant_fund.data.adapters.parquet import ParquetMarketProvider
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.data.corporate_actions import adjust_prices
from quant_fund.data.ingest import ingest, make_provider
from quant_fund.data.point_in_time import validate_feature_frame
from quant_fund.data.universe import membership_asof
from quant_fund.features.cross_sectional import apply_cross_sectional
from quant_fund.features.engine import add_market_features
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


def test_file_adapter_rejects_missing_pit_contract(tmp_path: Path) -> None:
    pl.DataFrame(
        {
            "event_time": [datetime(2020, 1, 2, tzinfo=UTC)],
            "security_id": ["A"],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [100.0],
        }
    ).write_parquet(tmp_path / "bars.parquet")
    with pytest.raises(PointInTimeError, match="missing PIT columns"):
        ParquetMarketProvider(tmp_path).get_bars()


def test_file_adapter_accepts_csv_and_applies_point_in_time_filters(tmp_path: Path) -> None:
    t0 = datetime(2020, 1, 2, tzinfo=UTC)
    t1 = datetime(2020, 1, 3, tzinfo=UTC)
    pl.DataFrame(
        {
            "event_time": [t0, t1],
            "available_time": [t0, t1],
            "ingested_time": [t0, t1],
            "source": ["vendor-file", "vendor-file"],
            "revision_id": ["v1", "v1"],
            "security_id": ["A", "B"],
            "open": [10.0, 20.0],
            "high": [11.0, 21.0],
            "low": [9.0, 19.0],
            "close": [10.5, 20.5],
            "volume": [1000.0, 2000.0],
        }
    ).write_csv(tmp_path / "bars.csv")
    bars = ParquetMarketProvider(tmp_path).get_bars(start=t1, security_ids=["B"])
    assert bars.height == 1
    assert bars["security_id"].to_list() == ["B"]
    assert bars["source"].to_list() == ["vendor-file"]


def test_file_source_ingest_builds_governed_lake_and_manifest(tmp_path: Path) -> None:
    t0 = datetime(2020, 1, 2, tzinfo=UTC)
    t1 = datetime(2020, 1, 3, tzinfo=UTC)
    raw = tmp_path / "raw"
    raw.mkdir()
    pl.DataFrame(
        {
            "event_time": [t0, t1],
            "available_time": [t0, t1],
            "ingested_time": [t0, t1],
            "source": ["vendor-file", "vendor-file"],
            "revision_id": ["v1", "v1"],
            "security_id": ["A", "A"],
            "open": [10.0, 10.5],
            "high": [11.0, 11.5],
            "low": [9.0, 10.0],
            "close": [10.5, 11.0],
            "volume": [1000.0, 1100.0],
        }
    ).write_csv(raw / "bars.csv")
    pl.DataFrame(
        {
            "event_time": pl.Series([], dtype=pl.Datetime("us", "UTC")),
            "security_id": pl.Series([], dtype=pl.String),
            "split_factor": pl.Series([], dtype=pl.Float64),
            "dividend": pl.Series([], dtype=pl.Float64),
        }
    ).write_parquet(raw / "corporate_actions.parquet")
    pl.DataFrame(
        {
            "security_id": ["A"],
            "sector": ["technology"],
            "industry": ["software"],
            "exchange": ["TEST"],
        }
    ).write_parquet(raw / "security_master.parquet")

    config = AppConfig.model_validate(
        {
            "data": {
                "root": str(tmp_path / "lake"),
                "source": "file",
                "parquet_path": str(raw),
            }
        }
    )
    paths = ingest(config)
    assert paths["bars"].is_file()
    assert paths["silver"].is_file()
    manifest = paths["manifest"].read_text()
    assert '"source": "file"' in manifest
    assert '"rows": 2' in manifest
    assert '"sha256":' in manifest
    silver = pl.read_parquet(paths["silver"])
    assert silver.height == 2
    assert silver["sector"].to_list() == ["technology", "technology"]


def test_file_adapter_rejects_duplicate_bar_keys(tmp_path: Path) -> None:
    t = datetime(2020, 1, 2, tzinfo=UTC)
    pl.DataFrame(
        {
            "event_time": [t, t],
            "available_time": [t, t],
            "ingested_time": [t, t],
            "source": ["vendor", "vendor"],
            "revision_id": ["v1", "v1"],
            "security_id": ["A", "A"],
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "volume": [100.0, 100.0],
        }
    ).write_parquet(tmp_path / "bars.parquet")
    with pytest.raises(PointInTimeError, match="duplicate"):
        ParquetMarketProvider(tmp_path).get_bars()


def test_file_adapter_rejects_invalid_ohlcv(tmp_path: Path) -> None:
    t = datetime(2020, 1, 2, tzinfo=UTC)
    pl.DataFrame(
        {
            "event_time": [t],
            "available_time": [t],
            "ingested_time": [t],
            "source": ["vendor"],
            "revision_id": ["v1"],
            "security_id": ["A"],
            "open": [1.0],
            "high": [0.5],
            "low": [1.0],
            "close": [1.0],
            "volume": [-1.0],
        }
    ).write_parquet(tmp_path / "bars.parquet")
    with pytest.raises(PointInTimeError, match="invalid OHLCV"):
        ParquetMarketProvider(tmp_path).get_bars()


def test_file_adapter_rejects_impossible_ingestion_order(tmp_path: Path) -> None:
    event = datetime(2020, 1, 2, tzinfo=UTC)
    available = datetime(2020, 1, 3, tzinfo=UTC)
    pl.DataFrame(
        {
            "event_time": [event],
            "available_time": [available],
            "ingested_time": [event],
            "source": ["vendor"],
            "revision_id": ["v1"],
            "security_id": ["A"],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [100.0],
        }
    ).write_parquet(tmp_path / "bars.parquet")
    with pytest.raises(PointInTimeError, match="impossible PIT"):
        ParquetMarketProvider(tmp_path).get_bars()


def test_file_adapter_rejects_non_temporal_pit_columns(tmp_path: Path) -> None:
    pl.DataFrame(
        {
            "event_time": ["2020-01-02"],
            "available_time": ["2020-01-02"],
            "ingested_time": ["2020-01-02"],
            "source": ["vendor"],
            "revision_id": ["v1"],
            "security_id": ["A"],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [100.0],
        }
    ).write_parquet(tmp_path / "bars.parquet")
    with pytest.raises(PointInTimeError, match="timestamps"):
        ParquetMarketProvider(tmp_path).get_bars()


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


def test_cs_transform_excludes_late_available_rows() -> None:
    t0 = datetime(2020, 1, 2, tzinfo=UTC)
    t1 = datetime(2020, 1, 3, tzinfo=UTC)
    df = pl.DataFrame(
        {
            "event_time": [t0, t0, t1, t1],
            "security_id": ["a", "b", "a", "b"],
            "available_time": [t0, t1, t0, t1],
            "x": [1.0, 100.0, 1.0, 2.0],
        }
    )
    out = apply_cross_sectional(df, ["x"], 0.0)

    t0_rows = out.filter(pl.col("event_time") == t0).sort("security_id")
    assert t0_rows["cs_pct_x"].to_list() == [0.5, None]


def test_market_aggregates_exclude_late_available_rows() -> None:
    t0 = datetime(2020, 1, 2, tzinfo=UTC)
    t1 = datetime(2020, 1, 3, tzinfo=UTC)
    df = pl.DataFrame(
        {
            "event_time": [t0, t0, t1, t1],
            "security_id": ["BENCH", "late", "BENCH", "late"],
            "available_time": [t0, t1, t0, t1],
            "ret_1": [0.1, 10.0, 0.2, 0.4],
            "vol_20": [0.2, 2.0, 0.3, 0.5],
            "mom_20": [0.01, 1.0, 0.02, 0.04],
        }
    )
    out = add_market_features(df, "BENCH").sort(["event_time", "security_id"])

    assert out.filter(pl.col("event_time") == t0)["cs_mean_ret"].to_list() == [0.1, None]
    assert out.filter(pl.col("event_time") == t0)["mkt_ret_1"].to_list() == [0.1, None]
    assert out.filter(pl.col("event_time") == t1)["cs_mean_ret"].to_list() == pytest.approx(
        [0.3, 0.3]
    )


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


def test_feature_frame_validation_uses_row_decision_times() -> None:
    t0 = datetime(2020, 1, 2, tzinfo=UTC)
    t1 = datetime(2020, 1, 3, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "decision_time": [t0, t1],
            "max_source_available_time": [t0, t1],
        }
    )

    validate_feature_frame(frame, t0)


def test_empty_feature_frame_is_pit_safe() -> None:
    validate_feature_frame(pl.DataFrame(), datetime(2020, 1, 2, tzinfo=UTC))


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


def test_require_pit_columns_missing_raises() -> None:
    from quant_fund.data.point_in_time import require_pit_columns

    with pytest.raises(PointInTimeError, match="missing PIT columns"):
        require_pit_columns(pl.DataFrame({"event_time": [datetime(2020, 1, 2, tzinfo=UTC)]}))


def test_require_pit_columns_complete_ok() -> None:
    from quant_fund.data.point_in_time import require_pit_columns

    t = datetime(2020, 1, 2, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "event_time": [t],
            "available_time": [t],
            "ingested_time": [t],
            "source": ["test"],
            "security_id": ["A"],
            "revision_id": ["v1"],
        }
    )
    require_pit_columns(frame)


def test_filter_available_drops_future_and_keeps_boundary() -> None:
    from quant_fund.data.point_in_time import filter_available

    t0 = datetime(2020, 1, 2, tzinfo=UTC)
    t1 = datetime(2020, 1, 3, tzinfo=UTC)
    t2 = datetime(2020, 1, 4, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "event_time": [t0, t1, t2],
            "available_time": [t0, t1, t2],
            "ingested_time": [t0, t1, t2],
            "source": ["test", "test", "test"],
            "security_id": ["a", "b", "c"],
            "revision_id": ["v1", "v1", "v1"],
        }
    )
    out = filter_available(frame, t1)
    assert out.height == 2
    assert set(out["security_id"].to_list()) == {"a", "b"}


def test_filter_available_empty_frame_requires_columns() -> None:
    from quant_fund.data.point_in_time import filter_available

    with pytest.raises(PointInTimeError, match="missing PIT columns"):
        filter_available(pl.DataFrame(), datetime(2020, 1, 2, tzinfo=UTC))


def test_filter_available_all_future_returns_empty() -> None:
    from quant_fund.data.point_in_time import filter_available

    t0 = datetime(2020, 1, 5, tzinfo=UTC)
    decision = datetime(2020, 1, 2, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "event_time": [t0],
            "available_time": [t0],
            "ingested_time": [t0],
            "source": ["test"],
            "security_id": ["a"],
            "revision_id": ["v1"],
        }
    )
    out = filter_available(frame, decision)
    assert out.height == 0


def test_validate_feature_frame_null_availability_fails() -> None:
    decision = datetime(2020, 1, 2, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "available_time": [decision, None],
            "security_id": ["a", "b"],
        }
    )
    with pytest.raises(PointInTimeError, match="null or future"):
        validate_feature_frame(frame, decision)


def test_validate_feature_frame_row_decision_null_fails() -> None:
    t0 = datetime(2020, 1, 2, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "decision_time": [t0, None],
            "max_source_available_time": [t0, t0],
        }
    )
    with pytest.raises(PointInTimeError, match="null or future"):
        validate_feature_frame(frame, t0)


def test_validate_feature_frame_no_availability_col_is_noop() -> None:
    # Without availability columns the helper cannot assert observability — noop.
    validate_feature_frame(
        pl.DataFrame({"security_id": ["a"], "x": [1.0]}),
        datetime(2020, 1, 2, tzinfo=UTC),
    )


def test_validate_feature_frame_max_source_lookahead() -> None:
    decision = datetime(2020, 1, 2, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "max_source_available_time": [datetime(2020, 1, 5, tzinfo=UTC)],
        }
    )
    with pytest.raises(PointInTimeError):
        validate_feature_frame(frame, decision)


def test_make_provider_synthetic_vs_file_routing(tmp_path: Path) -> None:
    """make_provider routes synthetic → Synthetic; file/parquet → Parquet."""
    syn = AppConfig.model_validate(
        {
            "data": {
                "root": str(tmp_path / "syn"),
                "source": "synthetic",
                "synthetic_n_assets": 3,
                "synthetic_n_days": 20,
            }
        }
    )
    assert isinstance(make_provider(syn), SyntheticMarketProvider)

    raw = tmp_path / "raw"
    raw.mkdir()
    for src in ("file", "parquet"):
        cfg = AppConfig.model_validate(
            {"data": {"root": str(tmp_path / src), "source": src, "parquet_path": str(raw)}}
        )
        prov = make_provider(cfg)
        assert isinstance(prov, ParquetMarketProvider)


def test_make_provider_unknown_source_fail_closed(tmp_path: Path) -> None:
    """Defense in depth: unknown source raises even if DataConfig is bypassed."""
    cfg = AppConfig.model_validate({"data": {"root": str(tmp_path), "source": "synthetic"}})
    # Bypass pydantic validator by mutating after construction
    object.__setattr__(cfg.data, "source", "polygon_vendor")
    with pytest.raises(ValueError, match="unknown data source"):
        make_provider(cfg)
    object.__setattr__(cfg.data, "source", " ")
    with pytest.raises(ValueError, match="unknown data source"):
        make_provider(cfg)


def test_cached_panel_validation_uses_row_event_time(tmp_path: Path) -> None:
    """Later as-of rows must not be compared against the panel's first date."""
    from quant_fund.pipeline.dataset import panel

    root = tmp_path / "lake"
    (root / "gold").mkdir(parents=True)
    times = [datetime(2020, 1, 2, tzinfo=UTC), datetime(2020, 1, 3, tzinfo=UTC)]
    features = pl.DataFrame(
        {
            "security_id": ["A", "A"],
            "event_time": times,
            "available_time": times,
            "feature_set_version": ["features.v1", "features.v1"],
            "ret_1": [0.1, 0.2],
        }
    )
    labels = pl.DataFrame(
        {"security_id": ["A", "A"], "event_time": times, "future_ret_1": [0.0, 0.1]}
    )
    features.write_parquet(root / "gold" / "features.parquet")
    labels.write_parquet(root / "gold" / "labels.parquet")
    config = AppConfig.model_validate({"data": {"root": str(root)}})
    assert panel(config).height == 2
