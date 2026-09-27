"""data/sources contract: HTTP primitives, PIT stamping, storage, collector.

All network access is stubbed — these tests must be runnable offline.
"""

from __future__ import annotations

import inspect
import json
from datetime import UTC, datetime
from typing import Any

import polars as pl
import pytest

from quant_fund.data.collector import collect_source
from quant_fund.data.protocols import MarketDataProvider, SecurityMaster
from quant_fund.data.sources.adapters import (
    BeaSource,
    BinanceMarketSource,
    BinancePublicDataSource,
    CcxtSource,
    Fi2010Source,
    FredSource,
    ItchSampleSource,
    OpenBBSource,
    TreasurySource,
    WorldBankSource,
    _paginated_klines,
)
from quant_fund.data.sources.base import (
    HttpClient,
    SourceAdapter,
    SourceError,
    parse_time,
    pit_frame,
    query_url,
    utc_now,
)
from quant_fund.data.sources.normalize import (
    csv_rows,
    normalize_observations,
    normalize_ohlcv,
)
from quant_fund.data.sources.storage import write_source_frame


def _stub_client(**methods: Any) -> Any:
    class _C:
        def __getattr__(self, name: str) -> Any:
            return methods[name]

    return _C()


def test_http_client_validates_config() -> None:
    with pytest.raises(ValueError):
        HttpClient(timeout=0)
    with pytest.raises(ValueError):
        HttpClient(retries=-1)
    with pytest.raises(ValueError):
        HttpClient(max_bytes=0)
    c = HttpClient(timeout=5, retries=0)
    assert c.timeout == 5.0 and c.retries == 0


def test_http_client_rejects_non_http_scheme() -> None:
    c = HttpClient(retries=0)
    with pytest.raises(SourceError, match="scheme"):
        c.get_bytes("file:///etc/passwd")
    with pytest.raises(SourceError, match="scheme"):
        c.get_text("ftp://x")


def test_parse_time_variants() -> None:
    assert parse_time(1_700_000_000).tzinfo is not None
    assert parse_time(1_700_000_000_000) == parse_time(1_700_000_000)  # ms
    a = parse_time("2024-01-02T03:04:05Z")
    assert a == datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)
    b = parse_time("2024-01-02 03:04:05")
    assert b.tzinfo == UTC


def test_pit_frame_stamps_and_rejects_impossible_chain() -> None:
    out = pit_frame(
        [{"event_time": "2024-01-02", "available_time": "2024-01-03", "v": 1}],
        source="s",
    )
    assert out.columns >= ["event_time", "available_time", "ingested_time", "source"]
    assert out["source"].unique().to_list() == ["s"]
    assert out["revision_id"].unique().to_list() == ["v1"]
    # available < event is impossible.
    with pytest.raises(SourceError, match="impossible"):
        pit_frame(
            [{"event_time": "2024-01-05", "available_time": "2024-01-03", "v": 1}],
            source="s",
        )
    assert pit_frame([], source="s").is_empty()


def test_query_url_drops_none_and_encodes() -> None:
    url = query_url("https://x/y", {"a": 1, "b": None, "c": "s t"})
    assert url == "https://x/y?a=1&c=s+t"
    assert query_url("https://x/y", {}) == "https://x/y"


def test_source_adapter_defaults() -> None:
    a = SourceAdapter()
    assert a.client is not None
    with pytest.raises(NotImplementedError):
        a.fetch()
    assert a.get_corporate_actions().is_empty()
    assert a.get_security_master().is_empty()


def test_normalize_ohlcv_contract() -> None:
    rows = [
        {
            "symbol": "AAA",
            "timestamp": 1_704_067_200_000,  # 2024-01-01T00:00Z
            "open": 10,
            "high": 11,
            "low": 9,
            "close": 10.5,
            "volume": 100,
            "available_time": "2024-01-01T01:00:00Z",
        }
    ]
    out = normalize_ohlcv(rows, source="test")
    assert out["security_id"].to_list() == ["AAA"]
    assert out["close"].to_list() == [10.5]
    assert out["source"].to_list() == ["test"]


def test_normalize_ohlcv_fail_closed() -> None:
    base = {
        "symbol": "AAA",
        "timestamp": "2024-01-01",
        "open": 10,
        "high": 11,
        "low": 9,
        "close": 10.5,
        "volume": 5,
    }
    with pytest.raises(SourceError, match="no security_id"):
        normalize_ohlcv([{**base, "symbol": ""}], source="s")
    with pytest.raises(SourceError, match="no event_time"):
        normalize_ohlcv([{k: v for k, v in base.items() if k != "timestamp"}], source="s")
    with pytest.raises(SourceError, match="not numeric"):
        normalize_ohlcv([{**base, "open": "x"}], source="s")
    with pytest.raises(SourceError, match="positive"):
        normalize_ohlcv([{**base, "close": -1}], source="s")
    with pytest.raises(SourceError, match="envelope"):
        normalize_ohlcv([{**base, "volume": -1}], source="s")
    with pytest.raises(SourceError, match="low/high"):
        normalize_ohlcv([{**base, "close": 20}], source="s")
    # Duplicate (event, symbol) rejected.
    with pytest.raises(SourceError, match="duplicate"):
        normalize_ohlcv([base, dict(base)], source="s")


def test_normalize_observations_contract() -> None:
    rows = [
        {
            "series_id": "GDP",
            "date": "2024-01-01",
            "available_time": "2024-02-01",
            "value": "1.5",
        },
        {
            "series_id": "GDP",
            "date": "2024-02-01",
            "available_time": "2024-03-01",
            "value": ".",
        },  # skipped (missing marker)
    ]
    out = normalize_observations(rows, source="fred")
    assert out.height == 1
    assert out["value"].to_list() == ["1.5"]
    with pytest.raises(SourceError, match="no date"):
        normalize_observations([{"value": "1", "available_time": "2024-01-01"}], source="s")
    with pytest.raises(SourceError, match="available_time"):
        normalize_observations([{"date": "2024-01-01", "value": "1"}], source="s")


def test_csv_rows_requires_header() -> None:
    rows = csv_rows("a,b\n1,2\n")
    assert rows == [{"a": "1", "b": "2"}]
    with pytest.raises(SourceError, match="header"):
        csv_rows("")


def test_write_source_frame_receipt(tmp_path) -> None:
    frame = pl.DataFrame(
        {
            "event_time": [datetime(2024, 1, 1, tzinfo=UTC)],
            "available_time": [datetime(2024, 1, 2, tzinfo=UTC)],
            "ingested_time": [datetime(2024, 1, 3, tzinfo=UTC)],
            "security_id": ["A"],
            "source": ["fred"],
            "value": [1.5],
        }
    )
    paths = write_source_frame(frame, tmp_path, "fred", provenance={"k": "v"})
    receipt = json.loads(paths["receipt"].read_text())
    assert receipt["schema_version"] == 2
    assert receipt["rows"] == 1
    assert receipt["columns"] == sorted(frame.columns)
    assert receipt["pit_ranges"]["event_time"]["min"].startswith("2024-01-01")
    assert receipt["provenance"] == {"k": "v"}
    assert len(receipt["sha256"]) == 64
    assert pl.read_parquet(paths["data"]).equals(frame)


def test_write_source_frame_path_guards(tmp_path) -> None:
    frame = pl.DataFrame({"x": [1]})
    for bad_source in ("", ".", "..", "a/b", "a\\b"):
        with pytest.raises(SourceError):
            write_source_frame(frame, tmp_path, bad_source)
    with pytest.raises(SourceError, match="relative"):
        write_source_frame(frame, tmp_path, "fred", filename="/etc/x.parquet")
    with pytest.raises(SourceError, match="escapes"):
        write_source_frame(frame, tmp_path, "fred", filename="../x.parquet")
    with pytest.raises(SourceError, match=".parquet"):
        write_source_frame(frame, tmp_path, "fred", filename="x.csv")


def test_write_source_frame_label_mismatch(tmp_path) -> None:
    frame = pl.DataFrame({"source": ["fred", "other"], "x": [1, 2]})
    with pytest.raises(SourceError, match="do not match"):
        write_source_frame(frame, tmp_path, "fred")


def test_collect_source_end_to_end(tmp_path) -> None:
    frame = pit_frame(
        [{"event_time": "2024-01-02", "available_time": "2024-01-03", "v": 1}],
        source="stub",
    )

    class Stub(SourceAdapter):
        name = "stub"

        def fetch(self, **kwargs: Any) -> pl.DataFrame:
            return frame

    res = collect_source("stub", tmp_path, adapter=Stub())
    assert res.data.suffix == ".parquet"
    assert res.receipt.suffix == ".json"
    receipt = json.loads(res.receipt.read_text())
    assert receipt["source"] == "stub"
    with pytest.raises(ValueError, match="non-empty"):
        collect_source("  ", tmp_path)


def test_collect_source_validates_pit_contract(tmp_path) -> None:
    # Missing PIT columns.
    class BadCols(SourceAdapter):
        name = "bad"

        def fetch(self, **kwargs: Any) -> pl.DataFrame:
            return pl.DataFrame({"x": [1]})

    with pytest.raises(SourceError, match="PIT columns"):
        collect_source("bad", tmp_path, adapter=BadCols())

    # Source label mismatch.
    class BadLabel(SourceAdapter):
        name = "bad2"

        def fetch(self, **kwargs: Any) -> pl.DataFrame:
            return pit_frame(
                [{"event_time": "2024-01-01", "available_time": "2024-01-02"}],
                source="fred",
            )

    with pytest.raises(SourceError, match="does not match"):
        collect_source("bad2", tmp_path, adapter=BadLabel())

    # event_time > available_time.
    class BadChain(SourceAdapter):
        name = "bad3"

        def fetch(self, **kwargs: Any) -> pl.DataFrame:
            return pl.DataFrame(
                {
                    "event_time": [datetime(2024, 1, 5, tzinfo=UTC)],
                    "available_time": [datetime(2024, 1, 2, tzinfo=UTC)],
                    "ingested_time": [utc_now()],
                    "source": ["bad3"],
                    "revision_id": ["v1"],
                }
            )

    with pytest.raises(SourceError, match="event_time"):
        collect_source("bad3", tmp_path, adapter=BadChain())


def test_protocols_are_structural() -> None:
    # Not runtime_checkable -> isinstance raises TypeError; verify shape.
    class Impl:
        def get_bars(self, start=None, end=None, security_ids=None) -> pl.DataFrame:
            return pl.DataFrame()

        def get_corporate_actions(self, start=None, end=None) -> pl.DataFrame:
            return pl.DataFrame()

        def get_security_master(self) -> pl.DataFrame:
            return pl.DataFrame()

    impl = Impl()
    sig = inspect.signature(MarketDataProvider.get_bars)
    assert set(sig.parameters) - {"self"} == {"start", "end", "security_ids"}
    assert set(inspect.signature(SecurityMaster.asof).parameters) - {"self"} == {"when", "ticker"}
    assert impl.get_bars().is_empty()


def test_paginated_klines_walks_pages() -> None:
    calls: list[str] = []

    # Pages keyed by the requested startTime cursor.
    def get_json(url: str, **kw: Any) -> Any:
        calls.append(url)
        if "startTime=0" in url:
            return [
                [i * 100, "1", "2", "0.5", "1.5", "10", i * 100 + 99] for i in range(2)
            ]  # short page -> stop
        if "startTime=1000" in url:
            return [
                [1000 + i * 100, "1", "2", "0.5", "1.5", "10", 1000 + i * 100 + 99]
                for i in range(2)
            ]  # full page
        return []

    rows = _paginated_klines(
        _stub_client(get_json=get_json),
        "https://x",
        symbol="btcusdt",
        interval="1d",
        start_time=0,
        end_time=None,
        limit=3,
        max_pages=5,
        pause_seconds=0.0,
    )
    assert len(rows) == 2  # page had fewer than limit -> stops
    assert len(calls) == 1

    # Full pages paginate forward by last open time + 1.
    calls.clear()
    rows = _paginated_klines(
        _stub_client(get_json=get_json),
        "https://x",
        symbol="BTCUSDT",
        interval="1d",
        start_time=1000,
        end_time=None,
        limit=2,
        max_pages=5,
        pause_seconds=0.0,
    )
    assert len(rows) == 2
    assert len(calls) == 2  # second call at cursor 1201 -> empty

    with pytest.raises(ValueError, match="interval"):
        _paginated_klines(
            _stub_client(),
            "e",
            symbol="x",
            interval="5s",
            start_time=None,
            end_time=None,
            limit=10,
            max_pages=1,
            pause_seconds=0.0,
        )
    with pytest.raises(ValueError, match="limit"):
        _paginated_klines(
            _stub_client(),
            "e",
            symbol="x",
            interval="1d",
            start_time=None,
            end_time=None,
            limit=0,
            max_pages=1,
            pause_seconds=0.0,
        )
    with pytest.raises(SourceError, match="list"):
        _paginated_klines(
            _stub_client(get_json=lambda *a, **k: {"not": "list"}),
            "e",
            symbol="x",
            interval="1d",
            start_time=0,
            end_time=None,
            limit=10,
            max_pages=1,
            pause_seconds=0.0,
        )


def test_binance_public_fetch_normalizes() -> None:
    # Kline: [open_time, o, h, l, c, vol, close_time, ...]
    row = [1_700_000_000_000, "100", "110", "95", "105", "12", 1_700_086_399_999]
    src = BinancePublicDataSource(client=_stub_client(get_json=lambda *a, **k: [row]))
    frame = src.fetch(symbol="btcusdt")
    assert frame["security_id"].to_list() == ["BTCUSDT"]
    assert frame["close"].to_list() == [105.0]
    # Malformed row fails closed.
    src2 = BinancePublicDataSource(client=_stub_client(get_json=lambda *a, **k: [["x"]]))
    with pytest.raises(SourceError, match="malformed"):
        src2.fetch()
    with pytest.raises(ValueError, match="limit"):
        src.fetch(limit=0)


def test_binance_market_normalize_trade() -> None:
    msg = {
        "e": "trade",
        "s": "BTCUSDT",
        "t": 1,
        "p": "42000.5",
        "q": "0.01",
        "T": 1_700_000_000_000,
        "E": 1_700_000_000_500,
        "a": 7,
    }
    frame = BinanceMarketSource.normalize_trade({"data": msg})
    assert frame["security_id"].to_list() == ["BTCUSDT"]
    assert frame["close"].to_list() == [42000.5]
    with pytest.raises(SourceError, match="unsupported"):
        BinanceMarketSource.normalize_trade({"data": {"e": "bookTicker"}})


def test_fred_csv_path_normalizes(monkeypatch) -> None:
    csv = "observation_date,GDP\n2024-01-01,27.5\n2024-04-01,28.0\n"
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    src = FredSource(client=_stub_client(get_text=lambda *a, **k: csv))
    frame = src.fetch(series_id="GDP")
    assert frame.height == 2
    assert frame["security_id"].unique().to_list() == ["GDP"]


def test_treasury_source_normalizes() -> None:
    payload = {
        "data": [
            {
                "record_date": "2024-01-31",
                "security_type": "Treasury Bills",
                "avg_interest_rate_amt": "5.2",
            }
        ]
    }
    src = TreasurySource(client=_stub_client(get_json=lambda *a, **k: payload))
    frame = src.fetch()
    assert frame.height == 1
    assert frame["security_id"].to_list() == ["Treasury Bills"]
    # Unrecognized shape fails closed.
    bad = TreasurySource(client=_stub_client(get_json=lambda *a, **k: {"data": [{"x": 1}]}))
    with pytest.raises(SourceError, match="recognized"):
        bad.fetch()


def test_worldbank_source_normalizes_and_shape_guard() -> None:
    payload = [
        {"page": 1},
        [{"date": "2023", "value": 2.5e13}, {"date": "2022", "value": None}],
    ]
    src = WorldBankSource(client=_stub_client(get_json=lambda *a, **k: payload))
    frame = src.fetch(country="US", indicator="NY.GDP.MKTP.CD")
    assert frame.height == 1  # null values are skipped
    assert frame["security_id"].to_list() == ["US:NY.GDP.MKTP.CD"]
    bad = WorldBankSource(client=_stub_client(get_json=lambda *a, **k: {"x": 1}))
    with pytest.raises(SourceError, match="shape"):
        bad.fetch()


def test_bea_requires_key(monkeypatch) -> None:
    monkeypatch.delenv("BEA_API_KEY", raising=False)
    with pytest.raises(SourceError, match="api_key"):
        BeaSource(client=_stub_client()).fetch()


def test_optional_library_sources_fail_closed() -> None:
    for cls in (CcxtSource, OpenBBSource):
        with pytest.raises(SourceError, match="payload"):
            cls().fetch()
        # Explicit payloads pass through.
        out = cls().fetch(payload=[{"a": 1}])
        assert out.height == 1
        out2 = cls().fetch(payload=pl.DataFrame({"a": [1, 2]}))
        assert out2.height == 2


def test_itch_sample_reads_csv(tmp_path) -> None:
    f = tmp_path / "itch.csv"
    f.write_text(
        "security_id,timestamp,open,high,low,close,volume,available_time\n"
        "AAA,2024-01-01T00:00:00Z,10,11,9,10.5,5,2024-01-01T01:00:00Z\n"
    )
    out = ItchSampleSource().fetch(path=f)
    assert out["security_id"].to_list() == ["AAA"]
    assert out["source"].to_list() == ["nasdaq_itch"]


def test_fi2010_requires_event_time(tmp_path) -> None:
    f = tmp_path / "fi.csv"
    f.write_text("x,y\n1,2\n")
    with pytest.raises(SourceError, match="event_time"):
        Fi2010Source().fetch(path=f)
    f.write_text("event_time,x\n2024-01-01,2\n")
    out = Fi2010Source().fetch(path=f)
    assert out["source"].to_list() == ["fi_2010"]


def test_source_registry_lookup() -> None:
    from quant_fund.data.sources.registry import (
        ALIASES,
        SOURCE_REGISTRY,
        get_source,
        source_names,
    )

    names = source_names()
    assert "fred" in names and "hf_ohlcv_1m" in names  # lazy HF adapter installed
    assert isinstance(get_source("fred"), FredSource)
    # Aliases resolve and normalize case/whitespace.
    assert type(get_source(" binance ")).name == "binance_public_data"
    assert type(get_source("treasury")).name == "us_treasury"
    # Every alias points at a registered name.
    assert set(ALIASES.values()) <= set(SOURCE_REGISTRY)
    with pytest.raises(ValueError, match="unknown public data source"):
        get_source("nope")
    # Canonical names and aliases are unique keys.
    assert len(SOURCE_REGISTRY) == len(set(SOURCE_REGISTRY))
