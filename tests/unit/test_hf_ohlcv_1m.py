"""Offline tests for the Hugging Face OHLCV-1m adapter.

The checked-in fixture is a vendor-schema slice. Network tests are marked
``network`` and are excluded by CI's ``-m "not network"`` lane.
"""

from __future__ import annotations

from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError

import polars as pl
import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.hf_ohlcv_1m import (
    DATASET_REVISION,
    REVISION_ID,
    SOURCE_NAME,
    HfOhlcv1mProvider,
    MonthNotFound,
    OhlcvQualityError,
    empty_corporate_actions,
    minute_gap_report,
    month_parquet_url,
    normalize_vendor_frame,
    read_ohlcv_1m,
    resample_ohlcv,
    stream_https,
)
from quant_fund.data.corporate_actions import adjust_prices
from quant_fund.data.ingest import ingest, make_provider
from quant_fund.data.sources.base import SourceError
from quant_fund.data.sources.registry import get_source
from quant_fund.microstructure.bench import bench_candle_order_book
from quant_fund.microstructure.candle_book_features import candle_features_from_bars
from quant_fund.northset.data_view import canonical_northset_bars

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "hf_ohlcv_1m" / "ohlcv_sample.parquet"


def _clock() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


def _cache(tmp_path: Path, revision: str = "fixture") -> Path:
    dest = tmp_path / "cache" / revision
    dest.mkdir(parents=True)
    target = dest / "ohlcv_1992-01.parquet"
    target.write_bytes(FIXTURE.read_bytes())
    return tmp_path / "cache"


def _refuse_download(url: str, dest: Path) -> None:
    raise AssertionError(f"unexpected download {url} -> {dest}")


def test_fixture_month_maps_onto_canonical_bars(tmp_path: Path) -> None:
    provider = HfOhlcv1mProvider(
        _cache(tmp_path),
        symbols="AAPL,MSFT",
        start="1992-01-02",
        end="1992-01-02",
        revision="fixture",
        fetcher=_refuse_download,
        clock=_clock,
    )
    bars = provider.get_bars()
    assert bars.columns[:6] == [
        "security_id",
        "symbol",
        "event_time",
        "available_time",
        "ingested_time",
        "source",
    ]
    assert set(bars["security_id"].to_list()) == {"AAPL", "MSFT"}
    aapl = bars.filter(pl.col("security_id") == "AAPL").sort("event_time")
    assert aapl.height == 5
    assert aapl["event_time"][0] == datetime(1992, 1, 2, 14, 31, tzinfo=UTC)
    assert aapl["available_time"][0] == aapl["event_time"][0]
    assert aapl["open"][0] == 10.0
    assert aapl["session"][0] == "rth"
    assert aapl["session"][-1] == "ext"
    assert set(bars["source"].to_list()) == {SOURCE_NAME}
    assert set(bars["revision_id"].to_list()) == {REVISION_ID}
    assert set(bars["currency"].to_list()) == {"USD"}
    report = provider.quality_report.filter(pl.col("security_id") == "AAPL")
    assert report["n_rth"][0] == 4
    assert report["n_rth_missing"][0] == 386
    assert report["n_ext"][0] == 1
    assert report["max_rth_gap_minutes"][0] == 385
    assert report["n_missing_weekdays"][0] == 0
    assert minute_gap_report(aapl)["n_rth_missing"][0] == 386


def test_symbol_and_session_filters(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    day = read_ohlcv_1m(
        symbols=["MSFT"],
        cache_dir=cache,
        start="1992-01-02",
        end="1992-01-03",
        revision="fixture",
        fetcher=_refuse_download,
        clock=_clock,
    )
    assert day.bars["security_id"].unique().to_list() == ["MSFT"]
    rth = read_ohlcv_1m(
        symbols="AAPL",
        cache_dir=cache,
        start="1992-01-02",
        end="1992-01-02",
        revision="fixture",
        session="rth",
        fetcher=_refuse_download,
        clock=_clock,
    )
    assert rth.bars.height == 4
    assert set(rth.bars["session"].to_list()) == {"rth"}
    with pytest.raises(OhlcvQualityError, match="ZZZZ"):
        read_ohlcv_1m(
            symbols="ZZZZ",
            cache_dir=cache,
            start="1992-01-02",
            end="1992-01-02",
            revision="fixture",
            fetcher=_refuse_download,
            clock=_clock,
        )


def test_bad_bars_fail_closed_without_repair() -> None:
    base = pl.read_parquet(FIXTURE)
    twx = {
        "timestamp": datetime(1992, 1, 15, 21, 5, tzinfo=UTC),
        "open": 95.0,
        "high": 95.25,
        "low": 95.25,
        "close": 95.25,
        "volume": 10500.0,
        "ticker": "TWX",
    }
    broken = pl.concat([base.clear(), pl.DataFrame([twx])], how="vertical_relaxed")
    broken = broken.with_columns(pl.col("timestamp").cast(pl.Datetime("ns", "UTC")))
    with pytest.raises(OhlcvQualityError, match="not repaired"):
        normalize_vendor_frame(broken, clock=_clock)

    def _one(**overrides: object) -> None:
        row = {
            "timestamp": datetime(1992, 1, 2, 14, 30, tzinfo=UTC),
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.5,
            "volume": 1.0,
            "ticker": "AAPL",
        }
        row.update(overrides)
        frame = pl.DataFrame([row]).with_columns(pl.col("timestamp").cast(pl.Datetime("ns", "UTC")))
        normalize_vendor_frame(frame, clock=_clock)

    with pytest.raises(OhlcvQualityError, match="high < low"):
        _one(high=8.0, low=9.0)
    with pytest.raises(OhlcvQualityError, match="non-positive open"):
        _one(open=0.0, low=0.0)
    with pytest.raises(OhlcvQualityError, match="volume"):
        _one(volume=-1.0)
    dup = pl.DataFrame(
        [
            {
                "timestamp": datetime(1992, 1, 2, 14, 30, tzinfo=UTC),
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 1.0,
                "ticker": "AAPL",
            },
            {
                "timestamp": datetime(1992, 1, 2, 14, 30, tzinfo=UTC),
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 2.0,
                "ticker": "aapl",
            },
        ]
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("ns", "UTC")))
    with pytest.raises(OhlcvQualityError, match="duplicate"):
        normalize_vendor_frame(dup, clock=_clock)


def test_naive_and_unaligned_timestamps_raise() -> None:
    naive = pl.DataFrame(
        {
            "timestamp": [datetime(1992, 1, 2, 14, 30)],
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "volume": [1.0],
            "ticker": ["AAPL"],
        }
    )
    with pytest.raises(OhlcvQualityError, match="timezone-aware"):
        normalize_vendor_frame(naive, clock=_clock)
    skewed = pl.DataFrame(
        {
            "timestamp": [datetime(1992, 1, 2, 14, 30, 30, tzinfo=UTC)],
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "volume": [1.0],
            "ticker": ["AAPL"],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("ns", "UTC")))
    with pytest.raises(OhlcvQualityError, match="minute-aligned"):
        normalize_vendor_frame(skewed, clock=_clock)


def test_eastern_timestamp_converts_and_labels_rth() -> None:
    frame = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 7, 1, 9, 30)],
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "volume": [1.0],
            "ticker": ["AAPL"],
        }
    ).with_columns(
        pl.col("timestamp")
        .dt.replace_time_zone("America/New_York")
        .cast(pl.Datetime("ns", "America/New_York"))
    )
    bars = normalize_vendor_frame(frame, clock=_clock)
    assert bars["event_time"][0] == datetime(2024, 7, 1, 13, 31, tzinfo=UTC)
    assert bars["session"][0] == "rth"


def test_strict_gaps_and_off_session_flag_instead_of_filling(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    with pytest.raises(OhlcvQualityError, match="strict_gaps"):
        read_ohlcv_1m(
            symbols="AAPL",
            cache_dir=cache,
            start="1992-01-02",
            end="1992-01-02",
            revision="fixture",
            strict_gaps=True,
            fetcher=_refuse_download,
            clock=_clock,
        )
    weekend = pl.DataFrame(
        {
            "timestamp": [datetime(1992, 1, 4, 15, 0, tzinfo=UTC)],
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "volume": [1.0],
            "ticker": ["AAPL"],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("ns", "UTC")))
    kept = normalize_vendor_frame(weekend, clock=_clock)
    assert kept["session"][0] == "off"
    assert minute_gap_report(kept)["n_off"][0] == 1
    path = tmp_path / "cache" / "fixture" / "ohlcv_1992-01.parquet"
    pl.concat([pl.read_parquet(path), weekend], how="vertical_relaxed").write_parquet(path)
    with pytest.raises(OhlcvQualityError, match="strict_off"):
        read_ohlcv_1m(
            symbols="AAPL",
            cache_dir=cache,
            start="1992-01-04",
            end="1992-01-04",
            revision="fixture",
            strict_off=True,
            fetcher=_refuse_download,
            clock=_clock,
        )


def test_resample_does_not_invent_missing_minutes(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    minutes = read_ohlcv_1m(
        symbols="AAPL",
        cache_dir=cache,
        start="1992-01-02",
        end="1992-01-02",
        revision="fixture",
        fetcher=_refuse_download,
        clock=_clock,
    ).bars
    five = resample_ohlcv(minutes, "5m")
    opening = five.filter(pl.col("session") == "rth")
    assert opening.height == 1
    assert opening["n_source_minutes"][0] == 4
    assert opening["open"][0] == 10.0
    assert opening["high"][0] == 13.0
    assert opening["low"][0] == 9.0
    assert opening["close"][0] == 11.5
    assert opening["volume"][0] == 250.0
    assert five.height == 2
    daily = resample_ohlcv(minutes, "1d")
    assert daily.height == 1
    assert daily["session"][0] == "mixed"
    assert daily["volume"][0] == 260.0
    assert daily["n_source_minutes"][0] == 5
    rth_day = resample_ohlcv(minutes, "1d", session="rth")
    assert rth_day["close"][0] == 11.5
    assert rth_day["session"][0] == "rth"
    hourly = resample_ohlcv(minutes, "1h", session="rth")
    assert hourly.height == 1
    assert hourly["n_source_minutes"][0] == 4
    assert hourly["event_time"][0] == datetime(1992, 1, 2, 14, 35, tzinfo=UTC)
    with pytest.raises(ValueError, match="monthly"):
        resample_ohlcv(minutes, "1M")


def test_missing_cache_does_not_download(tmp_path: Path) -> None:
    with pytest.raises(OhlcvQualityError, match="downloads are off"):
        read_ohlcv_1m(
            symbols="AAPL",
            cache_dir=tmp_path / "empty",
            start="1992-01-02",
            end="1992-01-02",
            revision="fixture",
            allow_download=False,
            fetcher=_refuse_download,
            clock=_clock,
        )
    with pytest.raises(OhlcvQualityError, match="max_months"):
        read_ohlcv_1m(
            symbols="AAPL",
            cache_dir=tmp_path / "empty",
            start="1990-01-01",
            end="1992-01-02",
            revision="fixture",
            allow_download=True,
            max_months=2,
            fetcher=_refuse_download,
            clock=_clock,
        )


def test_download_writes_then_serves_from_cache(tmp_path: Path) -> None:
    calls: list[str] = []

    def _fetch(url: str, dest: Path) -> None:
        calls.append(url)
        dest.write_bytes(FIXTURE.read_bytes())

    first = read_ohlcv_1m(
        symbols="IBM",
        cache_dir=tmp_path / "cache",
        start="1992-01-03",
        end="1992-01-03",
        revision="fixture",
        allow_download=True,
        fetcher=_fetch,
        clock=_clock,
    )
    assert first.bars.height == 1
    assert first.bars["security_id"][0] == "IBM"
    assert calls == [month_parquet_url(1992, 1, "fixture")]
    read_ohlcv_1m(
        symbols="IBM",
        cache_dir=tmp_path / "cache",
        start="1992-01-03",
        end="1992-01-03",
        revision="fixture",
        allow_download=True,
        fetcher=_refuse_download,
        clock=_clock,
    )


def test_stream_https_caps_bytes_and_maps_404(tmp_path: Path) -> None:
    payload = b"abcdef"

    class _Body:
        def __init__(self, data: bytes) -> None:
            self._data = data
            self._off = 0

        def read(self, n: int) -> bytes:
            chunk = self._data[self._off : self._off + n]
            self._off += n
            return chunk

        def __enter__(self) -> _Body:
            return self

        def __exit__(self, *_: object) -> bool:
            return False

    dest = tmp_path / "month.parquet"
    stream_https(
        "https://example.invalid/month.parquet",
        dest,
        opener=lambda *_a, **_k: _Body(payload),
        max_bytes=100,
        timeout=1.0,
    )
    assert dest.read_bytes() == payload

    oversize = tmp_path / "big.parquet"
    with pytest.raises(SourceError, match="exceeded"):
        stream_https(
            "https://example.invalid/big.parquet",
            oversize,
            opener=lambda *_a, **_k: _Body(payload),
            max_bytes=3,
            timeout=1.0,
        )
    assert not oversize.exists()

    def _missing(*_a: object, **_k: object) -> object:
        raise HTTPError("https://example.invalid/missing", 404, "missing", hdrs=None, fp=BytesIO())  # type: ignore[arg-type]

    with pytest.raises(MonthNotFound):
        stream_https(
            "https://example.invalid/missing",
            tmp_path / "missing.parquet",
            opener=_missing,
            max_bytes=100,
            timeout=1.0,
        )
    with pytest.raises(Exception, match="non-HTTPS"):
        stream_https(
            "http://example.invalid/month.parquet",
            tmp_path / "plain.parquet",
            opener=lambda *_a, **_k: _Body(payload),
            max_bytes=100,
            timeout=1.0,
        )


def test_security_master_and_empty_actions(tmp_path: Path) -> None:
    provider = HfOhlcv1mProvider(
        _cache(tmp_path),
        symbols="AA.PR,AAPL",
        start="1992-01-02",
        end="1992-01-03",
        revision="fixture",
        fetcher=_refuse_download,
        clock=_clock,
    )
    # AA.PR is not in the fixture; the dotted form is accepted as a ticker
    # and then fails closed because the month has no such rows.
    with pytest.raises(OhlcvQualityError, match="AA.PR"):
        provider.get_bars()
    provider.symbols = ["AAPL", "IBM"]
    bars = provider.get_bars()
    master = provider.get_security_master()
    assert master["exchange"].unique().to_list() == ["UNKNOWN"]
    assert master["security_type"].unique().to_list() == ["unknown"]
    assert master["valid_to"].null_count() == master.height
    assert set(master["security_id"].to_list()) == set(bars["security_id"].to_list())
    assert provider.get_corporate_actions().is_empty()
    assert empty_corporate_actions().columns == provider.get_corporate_actions().columns


def test_downstream_bars_need_no_special_case(tmp_path: Path) -> None:
    provider = HfOhlcv1mProvider(
        _cache(tmp_path),
        symbols="AAPL,MSFT,IBM",
        start="1992-01-02",
        end="1992-01-03",
        revision="fixture",
        fetcher=_refuse_download,
        clock=_clock,
    )
    bars = provider.get_bars()
    candles = candle_features_from_bars(bars)
    assert candles.height == bars.height
    assert "candle_close" in candles.columns
    silver = adjust_prices(bars, provider.get_corporate_actions())
    assert silver["split_factor"].unique().to_list() == [1.0]
    view = canonical_northset_bars(silver)
    assert view.price_basis == "split_adjusted"
    assert bars["revision_id"][0] == REVISION_ID
    receipt = bench_candle_order_book(bars, label=SOURCE_NAME, min_names=1)
    assert receipt["family"] == "candle_order_book"
    assert receipt["data_source"] == SOURCE_NAME
    assert receipt["n_bars"] == bars.height
    assert receipt["book_source"] == "synthetic_lob"


def test_make_provider_and_ingest_stay_offline(tmp_path: Path) -> None:
    cache = _cache(tmp_path, revision=DATASET_REVISION)
    cfg = AppConfig.model_validate(
        {
            "data": {
                "root": str(tmp_path / "lake"),
                "source": "hf_ohlcv_1m",
                "source_symbol": "AAPL",
                "source_interval": "1m",
                "source_path": str(cache),
            }
        }
    )
    provider = make_provider(cfg)
    assert isinstance(provider, HfOhlcv1mProvider)
    assert provider.allow_download is False
    paths = ingest(cfg)
    stored = pl.read_parquet(paths["bars"])
    assert stored["security_id"].unique().to_list() == ["AAPL"]
    assert stored.height == 5
    actions = pl.read_parquet(paths["actions"])
    assert actions.is_empty()


def test_collect_cli_uses_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = _cache(tmp_path)
    monkeypatch.setattr(
        "quant_fund.data.adapters.hf_ohlcv_1m.http_download",
        _refuse_download,
    )
    config = tmp_path / "research.yaml"
    root = tmp_path / "lake"
    config.write_text(f"data:\n  root: {root}\n", encoding="utf-8")
    result = CliRunner().invoke(
        app,
        [
            "collect",
            "--config",
            str(config),
            "--source",
            "hf_ohlcv_1m",
            "--param",
            "symbols=AAPL",
            "--param",
            "start=1992-01-02",
            "--param",
            "end=1992-01-02",
            "--param",
            f"cache_dir={cache}",
            "--param",
            "revision=fixture",
            "--param",
            "interval=5m",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "rows=" in result.output
    written = pl.read_parquet(root / "raw" / "sources" / "hf_ohlcv_1m.parquet")
    assert written["n_source_minutes"].sum() == 5
    assert get_source("hf-ohlcv-1m").name == SOURCE_NAME


@pytest.mark.network
def test_live_january_1992_aapl_slice(tmp_path: Path) -> None:
    """Optional Hub read of the month inspected while building the adapter."""
    result = read_ohlcv_1m(
        symbols="AAPL",
        cache_dir=tmp_path / "cache",
        start="1992-01-02",
        end="1992-01-02",
        allow_download=True,
        clock=_clock,
    )
    assert result.bars["open"][0] == 55.75
    assert result.bars["event_time"][0] == datetime(1992, 1, 2, 14, 31, tzinfo=UTC)
    assert result.bars["session"][0] == "rth"
    assert result.bars.height > 300
    assert int(result.quality["n_rth_missing"].sum()) > 0
    cached = list((tmp_path / "cache" / DATASET_REVISION).glob("ohlcv_*.parquet"))
    assert [path.name for path in cached] == ["ohlcv_1992-01.parquet"]
