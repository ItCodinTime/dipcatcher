from datetime import UTC, datetime

import pytest

from quant_fund.config.models import DataConfig
from quant_fund.data.sources.adapters import (
    BinanceFundingRateSource,
    BinancePerpUniverseSource,
    BinancePublicDataSource,
    BinanceUsdtmPerpSource,
)
from quant_fund.data.sources.base import SourceError
from quant_fund.data.sources.normalize import normalize_observations
from quant_fund.data.sources.registry import get_source


class FakeClient:
    def __init__(self, payload=None, text=""):
        self.payload = payload
        self.text = text
        self.urls = []

    def get_json(self, url, **kwargs):
        self.urls.append(url)
        return self.payload

    def get_text(self, url, **kwargs):
        self.urls.append(url)
        return self.text


def test_observations_require_explicit_release_time():
    with pytest.raises(SourceError, match="available_time"):
        normalize_observations(
            [{"security_id": "GDP", "event_time": "2020-01-01", "value": "1"}],
            source="fred",
        )


def test_binance_normalizes_fixture_with_close_availability():
    client = FakeClient(
        [
            [1577836800000, "10", "12", "9", "11", "100"]
            + [1577923199999, None, None, None, None, None]
        ]
    )
    frame = BinancePublicDataSource(client=client).fetch(symbol="BTCUSDT", interval="1d", limit=1)
    assert frame.select("symbol").to_series().to_list() == ["BTCUSDT"]
    assert frame["event_time"][0] == datetime(2020, 1, 1, tzinfo=UTC)
    assert frame["available_time"][0] >= frame["event_time"][0]


def test_public_source_names_are_configurable_without_optional_imports():
    assert DataConfig.model_validate({"source": "fred"}).source == "fred"
    assert get_source("binance").name == "binance_public_data"
    assert get_source("binance_perp").name == "binance_usdtm_perp"
    assert get_source("binance_funding").name == "binance_funding_rate"
    assert get_source("perp_universe").name == "binance_perp_universe"
    for name in ("binance_usdtm_perp", "binance_funding_rate", "binance_perp_universe"):
        assert DataConfig.model_validate({"source": name}).source == name


class FakeSeqClient:
    """Returns queued payloads in order; records every URL for cursor checks."""

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.urls = []

    def get_json(self, url, **kwargs):
        self.urls.append(url)
        if not self.payloads:
            raise AssertionError("unexpected extra request")
        return self.payloads.pop(0)

    def get_text(self, url, **kwargs):
        self.urls.append(url)
        return ""


def _kline(open_ms, close_ms, close="11"):
    return [open_ms, "10", "12", "9", close, "100", close_ms, None, None, None, None, None]


def test_perp_klines_paginate_forward_and_drop_in_progress():
    now_ms = int(datetime.now(tz=UTC).timestamp() * 1000)
    pages = [
        [
            _kline(1_577_836_800_000, 1_577_923_199_999),
            _kline(1_577_923_200_000, 1_578_009_599_999),
        ],
        [_kline(1_578_009_600_000, 1_578_096_000_000), _kline(1_578_096_000_000, now_ms + 60_000)],
        [],  # full page was returned above, so one more request confirms exhaustion
    ]
    client = FakeSeqClient(pages)
    frame = BinanceUsdtmPerpSource(client=client).fetch(
        symbol="BTCUSDT", interval="1d", limit=2, pause_seconds=0.0
    )
    assert frame.height == 3  # in-progress bar dropped
    assert len(client.urls) == 3
    assert "startTime=1577923200001" in client.urls[1]
    assert "startTime=1578096000001" in client.urls[2]
    assert frame["event_time"].is_sorted()
    assert set(frame["source"].to_list()) == {"binance_usdtm_perp"}


def test_perp_klines_reject_bad_interval_and_limit():
    src = BinanceUsdtmPerpSource(client=FakeSeqClient([]))
    with pytest.raises(ValueError, match="interval"):
        src.fetch(symbol="BTCUSDT", interval="7h")
    with pytest.raises(ValueError, match="limit"):
        src.fetch(symbol="BTCUSDT", interval="1h", limit=0)


def test_funding_rate_paginates_and_normalizes():
    pages = [
        [
            {"symbol": "BTCUSDT", "fundingTime": 1_577_836_800_000, "fundingRate": "0.0001"},
            {"symbol": "BTCUSDT", "fundingTime": 1_577_865_600_000, "fundingRate": "-0.0002"},
        ],
        [],
    ]
    client = FakeSeqClient(pages)
    frame = BinanceFundingRateSource(client=client).fetch(
        symbol="BTCUSDT", limit=2, pause_seconds=0.0
    )
    assert frame.height == 2
    assert frame["value"].to_list() == [0.0001, -0.0002]
    assert "startTime=1577865600001" in client.urls[1]
    assert (frame["event_time"] == frame["available_time"]).all()


def test_funding_rate_rejects_non_finite():
    client = FakeSeqClient(
        [[{"symbol": "BTCUSDT", "fundingTime": 1_577_836_800_000, "fundingRate": "nan"}]]
    )
    with pytest.raises(SourceError):
        BinanceFundingRateSource(client=client).fetch(symbol="BTCUSDT", pause_seconds=0.0)


def test_perp_universe_filters_and_ranks_by_quote_volume():
    info = {
        "symbols": [
            {
                "symbol": "AAAUSDT",
                "contractType": "PERPETUAL",
                "quoteAsset": "USDT",
                "status": "TRADING",
                "onboardDate": 1_600_000_000_000,
                "baseAsset": "AAA",
            },
            {
                "symbol": "BBBUSDT",
                "contractType": "PERPETUAL",
                "quoteAsset": "USDT",
                "status": "TRADING",
                "onboardDate": 1_600_000_100_000,
                "baseAsset": "BBB",
            },
            {
                "symbol": "CCCUSDT",
                "contractType": "CURRENT_QUARTER",
                "quoteAsset": "USDT",
                "status": "TRADING",
                "onboardDate": 1_600_000_000_000,
                "baseAsset": "CCC",
            },
            {
                "symbol": "DDDUSDT",
                "contractType": "PERPETUAL",
                "quoteAsset": "USDT",
                "status": "SETTLING",
                "onboardDate": 1_600_000_000_000,
                "baseAsset": "DDD",
            },
            {
                "symbol": "EEEUSDC",
                "contractType": "PERPETUAL",
                "quoteAsset": "USDC",
                "status": "TRADING",
                "onboardDate": 1_600_000_000_000,
                "baseAsset": "EEE",
            },
        ]
    }
    tickers = [
        {"symbol": "AAAUSDT", "quoteVolume": "500.0"},
        {"symbol": "BBBUSDT", "quoteVolume": "900.0"},
    ]
    client = FakeSeqClient([info, tickers])
    frame = BinancePerpUniverseSource(client=client).fetch()
    assert frame["security_id"].to_list() == ["BBBUSDT", "AAAUSDT"]
    assert frame["rank"].to_list() == [1, 2]
    assert frame["value"].to_list() == [900.0, 500.0]
    assert frame["event_time"][0] == datetime.fromtimestamp(1_600_000_100.0, tz=UTC)


def test_perp_universe_empty_is_fail_closed():
    client = FakeSeqClient([{"symbols": []}, []])
    with pytest.raises(SourceError, match="zero symbols"):
        BinancePerpUniverseSource(client=client).fetch()
