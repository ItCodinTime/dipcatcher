"""Vendor adapter contract tests against recorded response fixtures.

Every ``get_daily_bars`` call replays a recorded vendor response body through
``httpx.MockTransport`` (``CassetteTransport``) — the HTTP layer, request
shape, auth wiring, and normalization are all exercised; the network is not.
No test sets a real API key.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import polars as pl
import pytest

from quant_fund.data.vendors import (
    VENDOR_ADAPTERS,
    AlphaVantageDailyAdapter,
    BinanceKlinesAdapter,
    PolygonDailyAdapter,
    StooqDailyAdapter,
    TiingoDailyAdapter,
    get_vendor_adapter,
    vendor_adapter_names,
)
from quant_fund.data.vendors.errors import (
    VendorAuthError,
    VendorRateLimitError,
    VendorResponseError,
)

from .helpers import CassetteTransport, FakeClock, cassette, fixture_body

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "vendors"
BAR_COLUMNS = {
    "security_id",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "event_time",
    "available_time",
    "ingested_time",
    "source",
    "revision_id",
}
EXPECTED_CLOSES = [470.05, 466.20, 471.09, 469.47]
EXPECTED_EVENT_TIMES = [
    datetime(2024, 1, 2, 21, 0, tzinfo=UTC),
    datetime(2024, 1, 3, 21, 0, tzinfo=UTC),
    datetime(2024, 1, 4, 21, 0, tzinfo=UTC),
    datetime(2024, 1, 5, 21, 0, tzinfo=UTC),
]


def _assert_bar_contract(frame: pl.DataFrame, *, source: str) -> None:
    assert set(frame.columns) == BAR_COLUMNS
    assert frame.height == 4
    assert frame["source"].unique().to_list() == [source]
    assert frame["security_id"].unique().to_list() == ["SPY"]
    # sorted, unique timestamps — the normalizer invariant
    assert frame["event_time"].is_sorted()
    assert frame["event_time"].n_unique() == 4
    assert frame["event_time"].to_list() == EXPECTED_EVENT_TIMES
    assert str(frame["event_time"].dtype.time_zone) == "UTC"
    # PIT chain: event <= available <= ingested
    assert (frame["event_time"] <= frame["available_time"]).all()
    assert (frame["available_time"] <= frame["ingested_time"]).all()
    for col in ("open", "high", "low", "close"):
        assert frame[col].is_finite().all()
        assert (frame[col] > 0).all()
    assert (frame["volume"] >= 0).all()
    assert (frame["low"] <= frame["high"]).all()


# -- registry -----------------------------------------------------------------


def test_registry_lists_all_five_vendors() -> None:
    assert vendor_adapter_names() == (
        "alpha_vantage",
        "binance_klines",
        "polygon",
        "stooq",
        "tiingo",
    )
    for name in vendor_adapter_names():
        assert isinstance(get_vendor_adapter(name), VENDOR_ADAPTERS[name])
    with pytest.raises(ValueError, match="unknown vendor adapter"):
        get_vendor_adapter("bloomberg")


# -- fixture parse contract ---------------------------------------------------


def test_tiingo_parses_recorded_bars() -> None:
    adapter = TiingoDailyAdapter()
    frame = adapter.parse_body(fixture_body("tiingo/spy_daily.json"), symbol="SPY")
    _assert_bar_contract(frame, source="tiingo")
    assert frame["close"].to_list() == EXPECTED_CLOSES
    assert frame["volume"].to_list() == [61234567.0, 72415987.0, 58902345.0, 66780456.0]
    assert frame["revision_id"].unique().to_list() == ["TIINGO_VENDOR_ADJ"]


def test_polygon_parses_recorded_bars_including_divergent() -> None:
    adapter = PolygonDailyAdapter()
    frame = adapter.parse_body(fixture_body("polygon/spy_aggs.json"), symbol="SPY")
    _assert_bar_contract(frame, source="polygon")
    # The fixture intentionally diverges on 2024-01-04 (reconcile exercise).
    assert frame["close"].to_list() == [470.05, 466.20, 467.00, 469.47]
    assert frame["revision_id"].unique().to_list() == ["POLYGON_VENDOR_ADJ"]


def test_alpha_vantage_parses_reverse_chronological_fixture() -> None:
    adapter = AlphaVantageDailyAdapter()
    frame = adapter.parse_body(fixture_body("alphavantage/spy_daily.json"), symbol="SPY")
    _assert_bar_contract(frame, source="alpha_vantage")
    assert frame["close"].to_list() == EXPECTED_CLOSES


def test_stooq_parses_recorded_csv() -> None:
    adapter = StooqDailyAdapter()
    frame = adapter.parse_body(fixture_body("stooq/spy_daily.csv"), symbol="SPY")
    _assert_bar_contract(frame, source="stooq")
    assert frame["close"].to_list() == EXPECTED_CLOSES
    assert frame["revision_id"].unique().to_list() == ["STOOQ_VENDOR_ADJ"]


def test_binance_parses_recorded_klines() -> None:
    adapter = BinanceKlinesAdapter()
    frame = adapter.parse_body(fixture_body("binance/btcusdt_1d.json"), symbol="BTCUSDT")
    assert frame.height == 4
    assert frame["security_id"].unique().to_list() == ["BTCUSDT"]
    # Crypto bars stamp event_time at open (00:00Z), available at close.
    assert frame["event_time"].to_list() == [
        datetime(2024, 1, 2, tzinfo=UTC),
        datetime(2024, 1, 3, tzinfo=UTC),
        datetime(2024, 1, 4, tzinfo=UTC),
        datetime(2024, 1, 5, tzinfo=UTC),
    ]
    assert (frame["event_time"] < frame["available_time"]).all()
    assert frame["close"].to_list() == [44890.0, 44150.75, 44210.25, 45030.5]
    assert frame["revision_id"].unique().to_list() == ["BINANCE_RAW.1d"]


def test_parse_body_window_filter() -> None:
    adapter = TiingoDailyAdapter()
    frame = adapter.parse_body(
        fixture_body("tiingo/spy_daily.json"),
        symbol="SPY",
        start=datetime(2024, 1, 4, tzinfo=UTC),
    )
    assert frame["event_time"].to_list() == EXPECTED_EVENT_TIMES[2:]


# -- live request path through cassette transport ------------------------------


def test_tiingo_request_shape_and_auth_header() -> None:
    transport = cassette(fixture_body("tiingo/spy_daily.json"))
    adapter = TiingoDailyAdapter(
        api_key="fixture-key",
        client=None,
        transport=transport,
        rate_per_second=1000.0,
    )
    frame = adapter.get_daily_bars(
        "spy", start=datetime(2024, 1, 2, tzinfo=UTC), end=datetime(2024, 1, 5, tzinfo=UTC)
    )
    assert frame.height == 4
    request = transport.requests[0]
    assert str(request.url).startswith("https://api.tiingo.com/tiingo/daily/SPY/prices")
    assert request.url.params["startDate"] == "2024-01-02"
    assert request.url.params["endDate"] == "2024-01-05"
    assert request.headers["Authorization"] == "Token fixture-key"


def test_polygon_request_shape_and_bearer_auth() -> None:
    transport = cassette(fixture_body("polygon/spy_aggs.json"))
    adapter = PolygonDailyAdapter(
        api_key="fixture-key", transport=transport, rate_per_second=1000.0
    )
    adapter.get_daily_bars(
        "SPY", start=datetime(2024, 1, 2, tzinfo=UTC), end=datetime(2024, 1, 5, tzinfo=UTC)
    )
    request = transport.requests[0]
    assert "/v2/aggs/ticker/SPY/range/1/day/2024-01-02/2024-01-05" in str(request.url)
    assert request.url.params["adjusted"] == "true"
    assert request.headers["Authorization"] == "Bearer fixture-key"
    assert "apikey" not in str(request.url).lower()  # key stays out of the URL


def test_alpha_vantage_apikey_param_and_request_shape() -> None:
    transport = cassette(fixture_body("alphavantage/spy_daily.json"))
    adapter = AlphaVantageDailyAdapter(
        api_key="fixture-key", transport=transport, rate_per_second=1000.0
    )
    adapter.get_daily_bars("SPY")
    request = transport.requests[0]
    assert request.url.host == "www.alphavantage.co"
    assert request.url.params["function"] == "TIME_SERIES_DAILY"
    assert request.url.params["symbol"] == "SPY"
    assert request.url.params["apikey"] == "fixture-key"  # only auth channel AV supports


def test_stooq_request_shape_keyless() -> None:
    transport = cassette(fixture_body("stooq/spy_daily.csv"))
    adapter = StooqDailyAdapter(transport=transport, rate_per_second=1000.0)
    adapter.get_daily_bars("SPY")
    request = transport.requests[0]
    assert str(request.url).startswith("https://stooq.com/q/d/l/")
    assert request.url.params["s"] == "spy.us"
    assert request.url.params["i"] == "d"
    assert "authorization" not in {k.lower() for k in request.headers}


def test_binance_request_shape_keyless() -> None:
    transport = cassette(fixture_body("binance/btcusdt_1d.json"))
    adapter = BinanceKlinesAdapter(transport=transport, rate_per_second=1000.0)
    adapter.get_daily_bars(
        "BTCUSDT",
        start=datetime(2024, 1, 2, tzinfo=UTC),
        end=datetime(2024, 1, 6, tzinfo=UTC),
    )
    request = transport.requests[0]
    assert str(request.url).startswith("https://api.binance.com/api/v3/klines")
    assert request.url.params["symbol"] == "BTCUSDT"
    assert request.url.params["interval"] == "1d"
    assert request.url.params["startTime"] == "1704153600000"
    assert "authorization" not in {k.lower() for k in request.headers}


def test_binance_paginates_forward_by_last_open_time() -> None:
    day1 = [
        1704153600000,
        "44150.00",
        "44980.00",
        "43900.00",
        "44890.00",
        "100.0",
        1704239999999,
    ]
    day2 = [
        1704240000000,
        "44890.00",
        "45200.00",
        "44100.50",
        "44150.75",
        "110.0",
        1704326399999,
    ]
    transport = CassetteTransport(
        [httpx.Response(200, content=json.dumps([day1, day2]).encode()), httpx.Response(200, content=b"[]")]
    )
    adapter = BinanceKlinesAdapter(transport=transport, rate_per_second=1000.0)
    frame = adapter.get_klines(
        "BTCUSDT", interval="1d", start=datetime(2024, 1, 2, tzinfo=UTC), limit=2
    )
    assert frame.height == 2
    assert len(transport.requests) == 2
    # Second page cursor = last open time + 1.
    assert transport.requests[1].url.params["startTime"] == str(1704240000000 + 1)


def test_binance_drops_in_progress_bar() -> None:
    open_bar = json.loads(fixture_body("binance/btcusdt_1d.json"))
    in_progress = [
        1704499200000,
        "45000.00",
        "45100.00",
        "44900.00",
        "45050.00",
        "10.0",
        4_000_000_000_000,  # close time far in the future
    ]
    payload = open_bar + [in_progress]
    adapter = BinanceKlinesAdapter()
    frame = adapter.parse_body(payload, symbol="BTCUSDT")
    assert frame.height == 4  # in-progress kline dropped
    assert frame["event_time"].max() == datetime(2024, 1, 5, tzinfo=UTC)


def test_binance_rejects_bad_interval_and_limit() -> None:
    adapter = BinanceKlinesAdapter()
    with pytest.raises(ValueError, match="interval"):
        adapter.build_request("BTCUSDT", None, None, interval="7h")
    with pytest.raises(ValueError, match="limit"):
        adapter.build_request("BTCUSDT", None, None, limit=0)


# -- auth and payload failure paths --------------------------------------------


def test_missing_key_is_lazy_then_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    adapter = TiingoDailyAdapter(environ=None)
    # Construction succeeds — adapters parse fixtures with zero keys.
    assert adapter.parse_body(fixture_body("tiingo/spy_daily.json"), symbol="SPY").height == 4
    with pytest.raises(VendorAuthError, match="TIINGO_API_KEY"):
        adapter.get_daily_bars("SPY")


def test_key_resolution_explicit_beats_environ() -> None:
    transport = cassette(fixture_body("tiingo/spy_daily.json"))
    adapter = TiingoDailyAdapter(
        api_key="explicit-wins",
        environ={"TIINGO_API_KEY": "env-loses"},
        transport=transport,
        rate_per_second=1000.0,
    )
    adapter.get_daily_bars("SPY")
    assert transport.requests[0].headers["Authorization"] == "Token explicit-wins"


def test_alpha_vantage_throttle_note_is_rate_limit_error() -> None:
    adapter = AlphaVantageDailyAdapter()
    payload = {"Note": "Thank you for using Alpha Vantage! Our standard API rate limit is 25 requests per day."}
    with pytest.raises(VendorRateLimitError, match="throttled"):
        adapter.parse_body(payload, symbol="SPY")
    info = {"Information": "We have detected your API key..."}
    with pytest.raises(VendorRateLimitError):
        adapter.parse_body(info, symbol="SPY")


def test_alpha_vantage_error_message_and_missing_series() -> None:
    adapter = AlphaVantageDailyAdapter()
    with pytest.raises(VendorResponseError, match="API error"):
        adapter.parse_body({"Error Message": "Invalid API call"}, symbol="SPY")
    with pytest.raises(VendorResponseError, match="Time Series"):
        adapter.parse_body({"Meta Data": {}}, symbol="SPY")


def test_polygon_error_payload_and_empty_results() -> None:
    adapter = PolygonDailyAdapter()
    with pytest.raises(VendorResponseError, match="API error"):
        adapter.parse_body({"status": "ERROR", "error": "boom"}, symbol="SPY")
    empty = adapter.parse_body(
        {"status": "OK", "resultsCount": 0, "request_id": "x"}, symbol="SPY"
    )
    assert empty.is_empty()
    assert set(empty.columns) == BAR_COLUMNS


def test_tiingo_non_list_payload_fails_closed() -> None:
    adapter = TiingoDailyAdapter()
    with pytest.raises(VendorResponseError, match="JSON array"):
        adapter.parse_body({"detail": "Not found"}, symbol="SPY")
    with pytest.raises(VendorResponseError, match="missing fields"):
        adapter.parse_body([{"date": "2024-01-02T00:00:00.000Z"}], symbol="SPY")


def test_stooq_challenge_page_fails_closed() -> None:
    adapter = StooqDailyAdapter()
    with pytest.raises(VendorResponseError, match="throttle|challenge|not CSV"):
        adapter.parse_body("<html><body>Exceeded the daily hits limit</body></html>", symbol="SPY")


def test_binance_malformed_row_fails_closed() -> None:
    adapter = BinanceKlinesAdapter()
    with pytest.raises(VendorResponseError, match="malformed"):
        adapter.parse_body([["x"]], symbol="BTCUSDT")
    with pytest.raises(VendorResponseError, match="JSON array"):
        adapter.parse_body({"code": -1121, "msg": "Invalid symbol"}, symbol="BTCUSDT")


def test_adapters_never_touch_real_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any accidental real-network path would raise here."""

    def _boom(self: object, request: httpx.Request) -> httpx.Response:
        raise AssertionError("real network was touched")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _boom)
    adapter = TiingoDailyAdapter(
        api_key="k",
        transport=cassette(fixture_body("tiingo/spy_daily.json")),
        rate_per_second=1000.0,
    )
    assert adapter.get_daily_bars("SPY").height == 4


@pytest.mark.network
@pytest.mark.skipif(
    __import__("os").environ.get("VENDOR_LIVE_TESTS") != "1",
    reason="live vendor test — set VENDOR_LIVE_TESTS=1 to run (not in CI)",
)
def test_stooq_live_smoke() -> None:
    frame = StooqDailyAdapter().get_daily_bars(
        "SPY", start=datetime(2024, 1, 2, tzinfo=UTC), end=datetime(2024, 1, 5, tzinfo=UTC)
    )
    assert frame.height > 0
