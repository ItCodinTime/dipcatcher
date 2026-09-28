"""data/adapters/stooq coverage: CSV parse matrix, session-close stamps,
terminal-vs-retry HTTP logic, file-lake writer, universe download wiring."""

from __future__ import annotations

import urllib.error
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from quant_fund.data.adapters import stooq as S

CSV = "Date,Open,High,Low,Close,Volume\n2024-01-02,100,102,99,101,1000000\n2024-01-03,101,103,100,102,2000000\n"


class TestSessionClose:
    def test_us_close_1600_et(self) -> None:
        ts = S.session_close(date(2024, 6, 3), "spy.us")
        assert ts.tzinfo is UTC
        assert ts.astimezone(S.US_CLOSE).hour == 16
        assert ts.astimezone(S.US_CLOSE).minute == 0

    def test_uk_close_1630_london(self) -> None:
        ts = S.session_close(date(2024, 6, 3), "vod.uk")
        assert ts.astimezone(S.UK_CLOSE).hour == 16
        assert ts.astimezone(S.UK_CLOSE).minute == 30


class TestParseCsv:
    def test_valid_rows_pit_columns(self) -> None:
        out = S.parse_stooq_csv(CSV, security_id="SPY", stooq_symbol="spy.us")
        assert out.height == 2
        row = out.row(0, named=True)
        assert row["security_id"] == "SPY"
        assert row["event_time"] == row["available_time"]
        assert row["event_time"] == S.session_close(date(2024, 1, 2), "spy.us")
        assert row["revision_id"] == "STOOQ_VENDOR_ADJ"
        assert row["currency"] == "USD"
        assert row["session"] == "rth"
        assert row["source"] == "stooq"

    def test_envelope_repair_min_max(self) -> None:
        # inverted high/low gets repaired (documented behavior)
        text = "Date,Open,High,Low,Close,Volume\n2024-01-02,100,99,102,101,10\n"
        out = S.parse_stooq_csv(text, security_id="X", stooq_symbol="x.us")
        assert out.height == 1
        assert out["high"][0] == 102.0
        assert out["low"][0] == 99.0

    def test_bad_rows_skipped(self) -> None:
        text = (
            "Date,Open,High,Low,Close,Volume\n"
            "bad-date,1,2,3,4,5\n"
            "2024-01-02,0,1,1,1,5\n"  # non-positive open
            "2024-01-03,1,2,0.5,1,-10\n"  # negative volume
            "2024-01-04,abc,2,1,1,5\n"  # unparseable
            ",1,2,3,4,5\n"  # blank date
            "2024-01-05,1,2,0.9,1.5,5\n"
        )
        out = S.parse_stooq_csv(text, security_id="X", stooq_symbol="x.us")
        assert out.height == 1

    def test_uk_currency(self) -> None:
        out = S.parse_stooq_csv(CSV, security_id="VOD", stooq_symbol="vod.uk")
        assert out["currency"][0] == "GBP"

    def test_empty_input(self) -> None:
        assert S.parse_stooq_csv("", security_id="X", stooq_symbol="x.us").is_empty()
        assert S.parse_stooq_csv(
            "Date,Open,High,Low,Close,Volume\n", security_id="X", stooq_symbol="x.us"
        ).is_empty()

    def test_np_finite(self) -> None:
        assert S.np_finite(1.0)
        assert not S.np_finite(float("nan"))
        assert not S.np_finite(float("inf"))
        assert not S.np_finite(float("-inf"))


class TestFetch:
    def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake(method: str, url: str, **kw: object) -> tuple[int, bytes]:
            return 200, b"Date,Open\n1,2\n"

        monkeypatch.setattr(S, "pooled_request", fake)
        assert S.fetch_stooq_csv("spy.us") == "Date,Open\n1,2\n"

    def test_404_is_terminal_no_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = []

        def fake(method: str, url: str, **kw: object) -> tuple[int, bytes]:
            calls.append(1)
            return 404, b""

        monkeypatch.setattr(S, "pooled_request", fake)
        with pytest.raises(urllib.error.HTTPError):
            S.fetch_stooq_csv("spy.us", retries=5)
        assert len(calls) == 1  # 4xx != 429 is terminal

    def test_500_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = []

        def fake(method: str, url: str, **kw: object) -> tuple[int, bytes]:
            calls.append(1)
            return 500, b""

        monkeypatch.setattr(S, "pooled_request", fake)
        with pytest.raises(urllib.error.HTTPError):
            S.fetch_stooq_csv("spy.us", retries=2)
        assert len(calls) == 3

    def test_ioerror_wraps_to_urlerror(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from quant_fund.data.concurrent_io import IoError

        def fake(method: str, url: str, **kw: object) -> tuple[int, bytes]:
            raise IoError("socket reset")

        monkeypatch.setattr(S, "pooled_request", fake)
        with pytest.raises(urllib.error.URLError):
            S.fetch_stooq_csv("spy.us", retries=1)


def _bars(currency: str = "USD") -> pl.DataFrame:
    ts = S.session_close(date(2024, 1, 2), "spy.us")
    return pl.DataFrame(
        {
            "security_id": ["AAA", "BBB"],
            "event_time": [ts, ts],
            "currency": [currency, currency],
            "open": [1.0, 2.0],
            "high": [1.1, 2.1],
            "low": [0.9, 1.9],
            "close": [1.05, 2.05],
            "volume": [10.0, 20.0],
        }
    )


class TestFileLake:
    def test_empty_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="empty"):
            S.write_file_lake(pl.DataFrame(), tmp_path)

    def test_writes_three_parquets(self, tmp_path: Path) -> None:
        out = S.write_file_lake(_bars(), tmp_path, sectors={"AAA": "Tech"})
        assert set(out) == {"bars", "master", "actions"}
        master = pl.read_parquet(out["master"])
        assert master.height == 2
        aaa = master.filter(pl.col("security_id") == "AAA").row(0, named=True)
        assert aaa["sector"] == "Tech"
        assert aaa["exchange"] == "XNAS"
        actions = pl.read_parquet(out["actions"])
        assert actions.is_empty()

    def test_gbp_routes_to_xlon(self, tmp_path: Path) -> None:
        out = S.write_file_lake(_bars("GBP"), tmp_path)
        master = pl.read_parquet(out["master"])
        assert set(master["exchange"]) == {"XLON"}


class TestDownloadUniverse:
    def test_ok_path(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(S, "fetch_stooq_csv", lambda sym, **kw: CSV)
        out = S.download_stooq_universe(
            tmp_path, names=(("AAA", "aaa.us"), ("BBB", "bbb.us")), pause_s=0.0
        )
        assert out["status"] == "ok"
        assert out["n_names"] == 2
        assert out["research_only"] is True
        paths = out["paths"]
        assert isinstance(paths, dict)
        assert Path(str(paths["bars"])).exists()

    def test_error_path_collected(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        def fake(sym: str, **kw: object) -> str:
            if "bad" in sym:
                raise urllib.error.URLError("down")
            return CSV

        monkeypatch.setattr(S, "fetch_stooq_csv", fake)
        out = S.download_stooq_universe(
            tmp_path, names=(("OK", "ok.us"), ("BAD", "bad.us")), pause_s=0.0
        )
        assert out["status"] == "ok"
        assert out["n_names"] == 1
        errors = out["errors"]
        assert isinstance(errors, dict)
        assert "BAD" in errors

    def test_all_fail_empty_status(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        def fake(sym: str, **kw: object) -> str:
            raise urllib.error.URLError("down")

        monkeypatch.setattr(S, "fetch_stooq_csv", fake)
        out = S.download_stooq_universe(tmp_path, names=(("A", "a.us"),), pause_s=0.0)
        assert out["status"] == "empty"
        assert out["n_names"] == 0

    def test_start_end_filters(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(S, "fetch_stooq_csv", lambda sym, **kw: CSV)
        out = S.download_stooq_universe(
            tmp_path,
            names=(("AAA", "aaa.us"),),
            start=datetime(2024, 1, 3, tzinfo=UTC),
            pause_s=0.0,
        )
        bars = pl.read_parquet(tmp_path / "bars.parquet")
        assert bars.height == 1  # only the Jan-3 bar survives
        assert out["n_bars"] == 1
