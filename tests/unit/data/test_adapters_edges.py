"""data/sources/adapters edge paths: HTTP-source parsing bodies and fail-closed
branches for every public collector — all behind a stubbed transport client."""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from quant_fund.data.sources.adapters import (
    AlfredSource,
    BeaSource,
    BinanceFundingRateSource,
    BinancePerpUniverseSource,
    BinanceUsdtmPerpSource,
    CcxtSource,
    CftcSource,
    CryptofeedSource,
    Fi2010Source,
    FinaSource,
    FredSource,
    GdeltSource,
    ItchSampleSource,
    OpenBBSource,
    SecEdgarSource,
    TreasurySource,
    WorldBankSource,
    _require_kline_rows,
)
from quant_fund.data.sources.base import SourceError

pytestmark = pytest.mark.synthetic


class _Client:
    def __init__(self, json_payload: Any = None, text_payload: str = "") -> None:
        self._json = json_payload
        self._text = text_payload
        self.json_urls: list[str] = []

    def get_json(self, url: str, headers: dict | None = None) -> Any:  # noqa: ARG002
        self.json_urls.append(url)
        return self._json

    def get_text(self, url: str) -> str:
        self.json_urls.append(url)
        return self._text


def _src(cls, json_payload: Any = None, text: str = ""):
    return cls(client=_Client(json_payload, text))


class TestHelpers:
    @pytest.mark.parametrize(
        "bad",
        [{}, ["x"], [[1]], None],
    )
    def test_require_kline_rows_rejects(self, bad) -> None:
        with pytest.raises(SourceError):
            _require_kline_rows(bad)

    def test_require_kline_rows_accepts(self) -> None:
        row = [1, 2, 3, 4, 5, 6, 7]
        assert _require_kline_rows([row]) == [row]


class TestGdelt:
    def test_empty_and_bad_shapes(self) -> None:
        assert _src(GdeltSource, {}).fetch(query="x").is_empty()
        assert _src(GdeltSource, {"timeline": []}).fetch(query="x").is_empty()
        with pytest.raises(SourceError):
            _src(GdeltSource, {"timeline": [{"data": "notalist"}]}).fetch(query="x")

    def test_happy(self) -> None:
        payload = {
            "timeline": [
                {"data": [{"date": "2024-01-01", "value": 0.5}]},
            ]
        }
        out = _src(GdeltSource, payload).fetch(query="fed")
        assert out.height == 1
        assert out["value"][0] == 0.5


class TestSecEdgar:
    def test_cik_guard(self) -> None:
        with pytest.raises(ValueError, match="digits"):
            _src(SecEdgarSource).fetch(cik="CIK-ABC")

    def test_ragged_columns(self) -> None:
        payload = {
            "filings": {
                "recent": {
                    "filingDate": ["2024-01-01", "2024-02-01"],
                    "form": ["8-K"],
                    "accessionNumber": ["a", "b"],
                }
            }
        }
        with pytest.raises(SourceError, match="ragged"):
            _src(SecEdgarSource, payload).fetch(cik="0000123")

    def test_happy(self) -> None:
        payload = {
            "filings": {
                "recent": {
                    "filingDate": ["2024-01-01"],
                    "form": ["8-K"],
                    "accessionNumber": ["0001"],
                }
            }
        }
        out = _src(SecEdgarSource, payload).fetch(cik="123")
        assert out.height == 1


class TestFred:
    def test_api_key_path(self, monkeypatch) -> None:
        payload = {"observations": [{"date": "2024-01-01", "value": "3.2"}]}
        out = _src(FredSource, payload).fetch(series_id="GDP", api_key="k")
        assert out.height == 1

    def test_csv_path_suffixed_column(self) -> None:
        text = "observation_date,GDP_20250101\n2024-01-01,1.5\n"
        out = _src(FredSource, text=text).fetch(series_id="GDP")
        assert out.height == 1

    def test_csv_ambiguous_column(self) -> None:
        text = "observation_date,A,B\n2024-01-01,1,2\n"
        with pytest.raises(SourceError, match="unambiguous"):
            _src(FredSource, text=text).fetch(series_id="GDP")

    def test_alfred_name(self) -> None:
        assert AlfredSource(client=_Client()).name == "alfred"


class TestTreasury:
    def test_empty_and_unrecognized(self) -> None:
        assert _src(TreasurySource, {}).fetch().is_empty()
        with pytest.raises(SourceError, match="recognized"):
            _src(TreasurySource, {"data": [{"weird": 1}]}).fetch()

    def test_happy(self) -> None:
        payload = {
            "data": [
                {
                    "record_date": "2024-01-01",
                    "avg_interest_rate_amt": "4.1",
                    "security_type": "bill",
                }
            ]
        }
        assert _src(TreasurySource, payload).fetch().height == 1


class TestCftc:
    def test_non_list_and_empty(self) -> None:
        with pytest.raises(SourceError, match="list"):
            _src(CftcSource, {"data": []}).fetch()
        assert _src(CftcSource, [{"x": 1}]).fetch().is_empty()

    def test_happy(self) -> None:
        payload = [
            {
                "report_date_as_yyyymmdd": "20240101",
                "open_interest_all": "10",
                "market_and_exchange_names": "WHEAT",
            }
        ]
        assert _src(CftcSource, payload).fetch().height == 1


class TestFina:
    def test_non_list_and_empty(self) -> None:
        with pytest.raises(SourceError, match="list"):
            _src(FinaSource, {"data": {}}).fetch()
        assert _src(FinaSource, [{"x": 1}]).fetch().is_empty()

    def test_happy(self) -> None:
        payload = {"data": [{"tradeDate": "2024-01-01", "shortVolume": 5}]}
        assert _src(FinaSource, payload).fetch().height == 1


class TestWorldBank:
    def test_bad_shape(self) -> None:
        with pytest.raises(SourceError, match="shape"):
            _src(WorldBankSource, {}).fetch()
        with pytest.raises(SourceError, match="shape"):
            _src(WorldBankSource, [1, "x"]).fetch()

    def test_happy_and_null_values(self) -> None:
        payload = [
            {"page": 1},
            [{"date": "2020", "value": 1.0}, {"date": "2019", "value": None}],
        ]
        out = _src(WorldBankSource, payload).fetch()
        assert out.height == 1


class TestBea:
    def test_requires_key(self, monkeypatch) -> None:
        monkeypatch.delenv("BEA_API_KEY", raising=False)
        with pytest.raises(SourceError, match="api_key"):
            _src(BeaSource).fetch()

    def test_happy_and_filters(self) -> None:
        payload = {
            "BEAAPI": {
                "Results": {
                    "Data": [
                        {"TimePeriod": "2024Q1", "DataValue": "1.2"},
                        {"TimePeriod": "", "DataValue": "9"},
                        {"TimePeriod": "2024Q2", "DataValue": ""},
                    ]
                }
            }
        }
        out = _src(BeaSource, payload).fetch(api_key="k")
        assert out.height == 1
        assert out["value"][0] == "1.2"

    @pytest.mark.parametrize(
        ("period", "expected"),
        [
            ("2024Q1", "2024-03-31"),
            ("2024M02", "2024-02-29"),
            ("2024", "2024-12-31"),
            ("weird", "weird"),
        ],
    )
    def test_period_mapping(self, period, expected) -> None:
        from quant_fund.data.sources.adapters import _bea_period_date

        assert _bea_period_date(period) == expected


class TestFileSources:
    def test_itch_sample(self, tmp_path) -> None:
        p = tmp_path / "itch.csv"
        p.write_text(
            "security_id,timestamp,open,high,low,close,volume\n"
            "AAPL,2024-01-02T09:30:00,100,101,99,100.5,1000\n"
        )
        assert _src(ItchSampleSource).fetch(path=p).height == 1

    def test_fi2010(self, tmp_path) -> None:
        p = tmp_path / "fi.csv"
        p.write_text("event_time,x\n2024-01-01,1\n")
        out = _src(Fi2010Source).fetch(path=p)
        assert out["source"][0] == "fi_2010"
        p2 = tmp_path / "bad.csv"
        p2.write_text("x,y\n1,2\n")
        with pytest.raises(SourceError, match="event_time"):
            _src(Fi2010Source).fetch(path=p2)


class TestOptionalLibraries:
    @pytest.mark.parametrize("cls", [CryptofeedSource, CcxtSource, OpenBBSource])
    def test_payload_paths(self, cls) -> None:
        src = cls(client=_Client())
        frame = pl.DataFrame({"a": [1]})
        assert src.fetch(payload=frame).equals(frame)
        assert src.fetch(payload=[{"a": 1}]).height == 1
        with pytest.raises(SourceError, match=cls.library_name):
            src.fetch()


class TestBinanceEdges:
    def test_usdtm_perp_bad_payload(self) -> None:
        src = _src(BinanceUsdtmPerpSource, {"x": 1})
        with pytest.raises(SourceError):
            src.fetch(symbol="BTCUSDT")

    def test_funding_guards(self) -> None:
        src = _src(BinanceFundingRateSource)
        with pytest.raises(ValueError, match="limit"):
            src.fetch(symbol="BTCUSDT", limit=0)
        with pytest.raises(ValueError, match="max_pages"):
            src.fetch(symbol="BTCUSDT", max_pages=0)
        src2 = _src(BinanceFundingRateSource, {"x": 1})
        with pytest.raises(SourceError, match="list"):
            src2.fetch(symbol="BTCUSDT")

    def test_funding_malformed_row(self) -> None:
        src = _src(BinanceFundingRateSource, [{"bad": "row"}])
        with pytest.raises(SourceError):
            src.fetch(symbol="BTCUSDT")

    def test_perp_universe_guards(self) -> None:
        src = _src(BinancePerpUniverseSource)
        with pytest.raises(ValueError, match="top_n"):
            src.fetch(top_n=0)
        src2 = _src(BinancePerpUniverseSource, {"x": 1})
        with pytest.raises(SourceError, match="symbols"):
            src2.fetch()
        src3 = _src(BinancePerpUniverseSource, {"symbols": []})
        with pytest.raises(SourceError, match="list"):
            src3.fetch()
