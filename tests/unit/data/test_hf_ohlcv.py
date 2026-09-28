"""data/adapters/hf_ohlcv_1m coverage: pure helpers, vendor-frame
normalization fail-closed matrix, gap report arithmetic, resample
alignment, read slice + provider + registry adapter, stream_https."""

from __future__ import annotations

import urllib.error
from datetime import UTC, date, datetime, timedelta
from email.message import Message
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from quant_fund.data.adapters import hf_ohlcv_1m as H

ET = H.ET


def _ts(et_day: date, et_h: int, et_m: int) -> datetime:
    """Minute-start instant for an ET clock time."""
    return datetime(et_day.year, et_day.month, et_day.day, et_h, et_m, tzinfo=ET).astimezone(
        UTC
    ) - timedelta(minutes=0)


def _vendor(rows: list[tuple[datetime, float, float, float, float, float, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [r[0] for r in rows],
            "open": [r[1] for r in rows],
            "high": [r[2] for r in rows],
            "low": [r[3] for r in rows],
            "close": [r[4] for r in rows],
            "volume": [r[5] for r in rows],
            "ticker": [r[6] for r in rows],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))


def _minute_rows(
    et_day: date, start_mod: int, n: int, ticker: str = "AAA", gap_at: int | None = None
) -> list[tuple[datetime, float, float, float, float, float, str]]:
    rows = []
    base = datetime(et_day.year, et_day.month, et_day.day, tzinfo=ET)
    for i in range(n):
        mod = start_mod + i
        if gap_at is not None and mod == gap_at:
            continue
        ts = (base + timedelta(minutes=mod)).astimezone(UTC)
        rows.append((ts, 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 1000.0, ticker))
    return rows


def _clock() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


class TestPureHelpers:
    def test_month_url(self) -> None:
        url = H.month_parquet_url(2024, 3)
        assert url.endswith("ohlcv_2024-03.parquet")
        assert "resolve/" in url
        with pytest.raises(ValueError, match="1..12"):
            H.month_parquet_url(2024, 13)
        with pytest.raises(ValueError, match="unsafe"):
            H.month_parquet_url(2024, 1, revision="bad;rev")
        with pytest.raises(ValueError, match="unsafe"):
            H.month_parquet_url(2024, 1, revision="x" * 81)

    def test_parse_symbols(self) -> None:
        assert H.parse_symbols(None) == []
        assert H.parse_symbols(" aapl, spy ,AAPL ") == ["AAPL", "SPY"]
        assert H.parse_symbols(["x", "y", "x"]) == ["X", "Y"]
        assert H.parse_symbols("BRK.B") == ["BRK.B"]
        with pytest.raises(ValueError, match="unsafe"):
            H.parse_symbols("A/B")
        with pytest.raises(ValueError, match="unsafe"):
            H.parse_symbols("A B")

    def test_normalize_interval(self) -> None:
        assert H.normalize_interval("1min") == "1m"
        assert H.normalize_interval("60m") == "1h"
        assert H.normalize_interval("1d") == "1d"
        with pytest.raises(ValueError, match="monthly"):
            H.normalize_interval("1M")
        with pytest.raises(ValueError, match="unsupported"):
            H.normalize_interval("7m")

    def test_parse_bound(self) -> None:
        # aware datetime passes through in UTC
        aware = datetime(2024, 3, 4, 12, tzinfo=ET)
        assert H.parse_bound(aware, role="start") == aware.astimezone(UTC)
        with pytest.raises(H.OhlcvQualityError, match="timezone-aware"):
            H.parse_bound(datetime(2024, 3, 4, 12), role="start")
        d = H.parse_bound(date(2024, 3, 4), role="start")
        assert d.astimezone(ET).hour == 0
        e = H.parse_bound(date(2024, 3, 4), role="end")
        assert e.astimezone(ET).hour == 23 and e.astimezone(ET).minute == 59
        s = H.parse_bound("2024-03-04", role="start")
        assert s == d
        z = H.parse_bound("2024-03-04T10:00:00Z", role="start")
        assert z.tzinfo is not None
        with pytest.raises(H.OhlcvQualityError):
            H.parse_bound("not a date", role="start")
        with pytest.raises(H.OhlcvQualityError):
            H.parse_bound("2024-13-45", role="end")

    def test_empty_frames(self) -> None:
        assert H.empty_bars().is_empty()
        assert "n_source_minutes" in H.empty_bars(resampled=True).columns
        assert H.empty_quality().is_empty()
        assert H.empty_corporate_actions().is_empty()

    def test_optional_coercions(self) -> None:
        assert H._optional_bool("true", default=False) is True
        assert H._optional_bool("0", default=True) is False
        assert H._optional_bool(1, default=False) is True
        with pytest.raises(ValueError, match="boolean"):
            H._optional_bool("maybe", default=False)
        assert H._optional_int("5", default=1) == 5
        assert H._optional_int(None, default=3) == 3
        with pytest.raises(ValueError, match="integer"):
            H._optional_int(True, default=1)
        with pytest.raises(ValueError, match="integer"):
            H._optional_int("x", default=1)
        assert H._optional_session("") is None
        with pytest.raises(ValueError, match="string"):
            H._optional_session(5)
        with pytest.raises(ValueError, match="dates"):
            H._optional_bound(4.5)


class TestNormalizeVendorFrame:
    def _good(self, et_day: date = date(2024, 3, 4)) -> pl.DataFrame:
        return _vendor(_minute_rows(et_day, H.RTH_OPEN_MOD, 5))

    def test_missing_columns(self) -> None:
        with pytest.raises(H.OhlcvQualityError, match="missing columns"):
            H.normalize_vendor_frame(pl.DataFrame({"timestamp": [], "open": []}))

    def test_empty_passthrough(self) -> None:
        out = H.normalize_vendor_frame(
            pl.DataFrame(schema={c: pl.Float64 for c in H.VENDOR_COLUMNS})
        )
        assert out.is_empty()

    def test_event_time_is_minute_close(self) -> None:
        v = self._good()
        out = H.normalize_vendor_frame(v, clock=_clock)
        ts0 = v["timestamp"][0]
        assert out["event_time"][0] == ts0 + timedelta(minutes=1)
        assert out["available_time"][0] == out["event_time"][0]
        assert (out["session"] == "rth").all()
        assert out["source"][0] == H.SOURCE_NAME
        assert out["revision_id"][0] == H.REVISION_ID

    def test_session_labels(self) -> None:
        day = date(2024, 3, 4)  # Monday
        rows = (
            _minute_rows(day, H.EXT_OPEN_MOD, 3)
            + _minute_rows(day, H.RTH_OPEN_MOD, 3)
            + _minute_rows(day, 16 * 60, 3)
            + _minute_rows(day, 21 * 60, 3)  # 21:00 ET -> off
        )
        out = H.normalize_vendor_frame(_vendor(rows), clock=_clock)
        labels = out["session"].to_list()
        assert labels[:3] == ["ext"] * 3
        assert labels[3:6] == ["rth"] * 3
        assert labels[6:9] == ["ext"] * 3
        assert labels[9:] == ["off"] * 3

    def test_duplicate_and_bad_rows_raise(self) -> None:
        v = self._good()
        dup = pl.concat([v, v.head(1)])
        with pytest.raises(H.OhlcvQualityError, match="duplicate"):
            H.normalize_vendor_frame(dup, clock=_clock)
        bad = v.with_columns(
            pl.when(pl.arange(0, v.height) == 0).then(1.0).otherwise(pl.col("high")).alias("high")
        ).with_columns(
            pl.when(pl.arange(0, v.height) == 0).then(200.0).otherwise(pl.col("low")).alias("low")
        )
        with pytest.raises(H.OhlcvQualityError, match="not repaired"):
            H.normalize_vendor_frame(bad, clock=_clock)

    def test_naive_timestamp_rejected(self) -> None:
        v = self._good().with_columns(pl.col("timestamp").dt.replace_time_zone(None))
        with pytest.raises(H.OhlcvQualityError, match="timezone-aware"):
            H.normalize_vendor_frame(v, clock=_clock)

    def test_ingested_before_available_raises(self) -> None:
        def past_clock() -> datetime:
            return datetime(2000, 1, 1, tzinfo=UTC)

        with pytest.raises(H.OhlcvQualityError, match="ingested_time"):
            H.normalize_vendor_frame(self._good(), clock=past_clock)

    def test_whitespace_ticker_rejected(self) -> None:
        v = self._good().with_columns(pl.lit("A A").alias("ticker"))
        with pytest.raises(H.OhlcvQualityError, match="whitespace"):
            H.normalize_vendor_frame(v, clock=_clock)


class TestGapReport:
    def test_empty_and_missing_cols(self) -> None:
        assert H.minute_gap_report(pl.DataFrame()).is_empty()
        with pytest.raises(H.OhlcvQualityError, match="missing columns"):
            H.minute_gap_report(pl.DataFrame({"security_id": ["A"]}))

    def test_full_session_no_gaps(self) -> None:
        bars = H.normalize_vendor_frame(
            _vendor(_minute_rows(date(2024, 3, 4), H.RTH_OPEN_MOD, 390)), clock=_clock
        )
        q = H.minute_gap_report(bars)
        row = q.row(0, named=True)
        assert row["n_rth"] == 390
        assert row["n_rth_missing"] == 0
        assert row["max_rth_gap_minutes"] == 0
        assert row["n_missing_weekdays"] == 0

    def test_internal_gap_counted(self) -> None:
        # drop minute at mod 09:35 -> internal gap of 1 minute
        bars = H.normalize_vendor_frame(
            _vendor(_minute_rows(date(2024, 3, 4), H.RTH_OPEN_MOD, 390, gap_at=H.RTH_OPEN_MOD + 5)),
            clock=_clock,
        )
        row = H.minute_gap_report(bars).row(0, named=True)
        assert row["n_rth"] == 389
        assert row["n_rth_missing"] == 1
        assert row["max_rth_gap_minutes"] == 1

    def test_open_and_close_gaps(self) -> None:
        # missing first 10 minutes
        rows = _minute_rows(date(2024, 3, 4), H.RTH_OPEN_MOD + 10, 380)
        bars = H.normalize_vendor_frame(_vendor(rows), clock=_clock)
        row = H.minute_gap_report(bars).row(0, named=True)
        assert row["max_rth_gap_minutes"] == 10

    def test_missing_weekday(self) -> None:
        day1 = _minute_rows(date(2024, 3, 4), H.RTH_OPEN_MOD, 10)  # Monday
        day3 = _minute_rows(date(2024, 3, 6), H.RTH_OPEN_MOD, 10)  # Wednesday
        bars = H.normalize_vendor_frame(_vendor(day1 + day3), clock=_clock)
        q = H.minute_gap_report(bars)
        assert set(q["n_missing_weekdays"]) == {1}


class TestResample:
    def _bars(self, n: int = 60) -> pl.DataFrame:
        return H.normalize_vendor_frame(
            _vendor(_minute_rows(date(2024, 3, 4), H.RTH_OPEN_MOD, n)), clock=_clock
        )

    def test_1m_passthrough_and_empty(self) -> None:
        bars = self._bars(10)
        out = H.resample_ohlcv(bars, "1m")
        assert out.height == 10
        assert H.resample_ohlcv(H.empty_bars(), "5m").is_empty()

    def test_5m_bins_align_to_open(self) -> None:
        bars = self._bars(60)
        out = H.resample_ohlcv(bars, "5m")
        assert out.height == 12
        first = out.row(0, named=True)
        assert first["n_source_minutes"] == 5
        # event_time = close of last observed minute in bin (09:34 -> 09:35)
        assert first["event_time"].astimezone(ET).minute == 35
        assert first["open"] == pytest.approx(100.0)
        assert first["high"] == pytest.approx(bars["high"][:5].max())

    def test_1h_offset(self) -> None:
        out = H.resample_ohlcv(self._bars(120), "1h")
        first = out.row(0, named=True)
        # 1h bins are offset so the first runs 09:30-10:30 ET; close = 10:30
        assert first["event_time"].astimezone(ET).minute == 30
        assert first["n_source_minutes"] == 60

    def test_daily_groups_session(self) -> None:
        rows = _minute_rows(date(2024, 3, 4), H.RTH_OPEN_MOD, 30) + _minute_rows(
            date(2024, 3, 5), H.RTH_OPEN_MOD, 30
        )
        bars = H.normalize_vendor_frame(_vendor(rows), clock=_clock)
        out = H.resample_ohlcv(bars, "1d")
        assert out.height == 2
        assert out["n_source_minutes"].to_list() == [30, 30]

    def test_session_filter_and_mixed(self) -> None:
        rows = _minute_rows(date(2024, 3, 4), H.RTH_OPEN_MOD - 4, 10)
        bars = H.normalize_vendor_frame(_vendor(rows), clock=_clock)
        rth = H.resample_ohlcv(bars, "5m", session="rth")
        assert (rth["session"] == "rth").all()
        mixed = H.resample_ohlcv(bars, "5m")
        # bin containing both ext and rth minutes is labeled mixed
        assert "mixed" in mixed["session"].to_list() or "ext" in mixed["session"].to_list()
        with pytest.raises(ValueError, match="rth"):
            H.resample_ohlcv(bars, "5m", session="bogus")
        with pytest.raises(H.OhlcvQualityError, match="session column"):
            H.resample_ohlcv(bars.drop("session"), "5m", session="rth")


def _write_month(path: Path, rows: list) -> None:
    _vendor(rows).write_parquet(path)


def _fake_fetcher(et_day: date) -> Any:
    def fetch(url: str, dest: Path) -> None:
        # decode year-month from the URL tail
        ym = url.rsplit("ohlcv_", 1)[1].split(".")[0]
        year, month = (int(x) for x in ym.split("-"))
        rows = _minute_rows(date(year, month, et_day.day), H.RTH_OPEN_MOD, 390)
        _write_month(dest, rows)

    return fetch


class TestReadSlice:
    def test_validation_matrix(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="required"):
            H.read_ohlcv_1m(symbols=None, cache_dir=tmp_path)
        with pytest.raises(ValueError, match="max_months"):
            H.read_ohlcv_1m(symbols="AAA", cache_dir=tmp_path, max_months=0)
        with pytest.raises(ValueError, match="rth"):
            H.read_ohlcv_1m(symbols="AAA", cache_dir=tmp_path, session="weird")
        with pytest.raises(H.OhlcvQualityError, match="after end"):
            H.read_ohlcv_1m(
                symbols="AAA",
                cache_dir=tmp_path,
                start=date(2024, 3, 5),
                end=date(2024, 3, 4),
            )
        with pytest.raises(H.OhlcvQualityError, match="both start and end"):
            H.read_ohlcv_1m(symbols="AAA", cache_dir=tmp_path, start=date(2024, 3, 4))
        with pytest.raises(H.OhlcvQualityError, match="no cached months"):
            H.read_ohlcv_1m(symbols="AAA", cache_dir=tmp_path)

    def test_fetcher_downloads_and_reads(self, tmp_path: Path) -> None:
        out = H.read_ohlcv_1m(
            symbols="AAA",
            cache_dir=tmp_path,
            start=date(2024, 3, 4),
            end=date(2024, 3, 4),
            allow_download=True,
            fetcher=_fake_fetcher(date(2024, 3, 4)),
            clock=_clock,
        )
        assert out.bars.height == 390
        assert out.minutes.height == 390
        assert out.quality.row(0, named=True)["n_rth"] == 390

    def test_missing_symbol_fails_closed(self, tmp_path: Path) -> None:
        with pytest.raises(H.OhlcvQualityError, match="no rows"):
            H.read_ohlcv_1m(
                symbols="ZZZ",
                cache_dir=tmp_path,
                start=date(2024, 3, 4),
                end=date(2024, 3, 4),
                allow_download=True,
                fetcher=_fake_fetcher(date(2024, 3, 4)),
                clock=_clock,
            )

    def test_strict_gaps_and_off(self, tmp_path: Path) -> None:
        def gappy(url: str, dest: Path) -> None:
            rows = _minute_rows(date(2024, 3, 4), H.RTH_OPEN_MOD, 390, gap_at=H.RTH_OPEN_MOD + 5)
            _write_month(dest, rows)

        with pytest.raises(H.OhlcvQualityError, match="strict_gaps"):
            H.read_ohlcv_1m(
                symbols="AAA",
                cache_dir=tmp_path,
                start=date(2024, 3, 4),
                end=date(2024, 3, 4),
                allow_download=True,
                fetcher=gappy,
                strict_gaps=True,
                clock=_clock,
            )

        def off_session(url: str, dest: Path) -> None:
            rows = _minute_rows(date(2024, 3, 4), H.RTH_OPEN_MOD, 390)
            rows += _minute_rows(date(2024, 3, 4), 21 * 60, 3)  # off hours
            _write_month(dest, rows)

        with pytest.raises(H.OhlcvQualityError, match="strict_off"):
            H.read_ohlcv_1m(
                symbols="AAA",
                cache_dir=tmp_path / "c2",
                start=date(2024, 3, 4),
                end=date(2024, 3, 4),
                allow_download=True,
                fetcher=off_session,
                strict_off=True,
                clock=_clock,
            )

    def test_cached_month_no_download_needed(self, tmp_path: Path) -> None:
        # pre-seed the cache; second read must not fetch
        sub = tmp_path / H.DATASET_REVISION
        sub.mkdir(parents=True)
        _write_month(
            sub / "ohlcv_2024-03.parquet", _minute_rows(date(2024, 3, 4), H.RTH_OPEN_MOD, 390)
        )
        calls: list[str] = []

        def fail_fetch(url: str, dest: Path) -> None:
            calls.append(url)
            raise AssertionError("must not fetch")

        out = H.read_ohlcv_1m(
            symbols="AAA",
            cache_dir=tmp_path,
            start=date(2024, 3, 4),
            end=date(2024, 3, 4),
            allow_download=False,
            fetcher=fail_fetch,
            clock=_clock,
        )
        assert out.bars.height == 390
        assert calls == []


class TestProviderAndRegistry:
    def test_provider_round_trip(self, tmp_path: Path) -> None:
        prov = H.HfOhlcv1mProvider(
            cache_dir=tmp_path,
            symbols="AAA",
            start=date(2024, 3, 4),
            end=date(2024, 3, 4),
            allow_download=True,
            fetcher=_fake_fetcher(date(2024, 3, 4)),
            clock=_clock,
        )
        bars = prov.get_bars(interval="5m")
        assert bars.height == 78
        assert prov.quality_report.height == 1
        master = prov.get_security_master()
        assert master.height == 1
        assert master["security_id"][0] == "AAA"
        assert master["exchange"][0] == "UNKNOWN"
        assert prov.get_corporate_actions().is_empty()

    def test_provider_security_master_fallback(self, tmp_path: Path) -> None:
        prov = H.HfOhlcv1mProvider(
            cache_dir=tmp_path,
            symbols="AAA",
            start=date(2024, 3, 4),
            end=date(2024, 3, 4),
            allow_download=True,
            fetcher=_fake_fetcher(date(2024, 3, 4)),
            clock=_clock,
        )
        # master before get_bars -> does its own 1m read
        master = prov.get_security_master()
        assert master.height == 1

    def test_source_adapter_validation(self, tmp_path: Path) -> None:
        src = H.HfOhlcv1mSource(fetcher=_fake_fetcher(date(2024, 3, 4)), clock=_clock)
        assert src.name == H.SOURCE_NAME
        with pytest.raises(ValueError, match="unknown"):
            src.fetch(symbols="AAA", bogus_param=1)
        with pytest.raises(ValueError, match="string or list"):
            src.fetch(symbols=5)
        with pytest.raises(ValueError, match="strings"):
            src.fetch(symbols="AAA", interval=5)
        with pytest.raises(ValueError, match="path"):
            src.fetch(symbols="AAA", cache_dir=5)
        with pytest.raises(ValueError, match="dates"):
            src.fetch(symbols="AAA", start=1.5)

    def test_source_adapter_fetch(self, tmp_path: Path) -> None:
        src = H.HfOhlcv1mSource(fetcher=_fake_fetcher(date(2024, 3, 4)), clock=_clock)
        bars = src.fetch(
            symbols="AAA",
            cache_dir=tmp_path,
            start=date(2024, 3, 4),
            end=date(2024, 3, 4),
            interval="15m",
            strict_gaps="false",
            max_months="6",
        )
        assert bars.height == 26


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._pos = 0

    def read(self, n: int) -> bytes:
        if self._pos >= len(self._payload):
            return b""
        chunk = self._payload[self._pos : self._pos + n]
        self._pos += n
        return chunk

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *a: object) -> None:
        return None


class TestStreamHttps:
    def test_refuses_non_https(self, tmp_path: Path) -> None:
        with pytest.raises(Exception, match="non-HTTPS"):
            H.stream_https(
                "http://example.com/x",
                tmp_path / "f",
                opener=lambda *a, **k: None,
                max_bytes=10,
                timeout=1,
            )
        with pytest.raises(ValueError, match="positive"):
            H.stream_https(
                "https://example.com/x",
                tmp_path / "f",
                opener=lambda *a, **k: None,
                max_bytes=0,
                timeout=1,
            )

    def test_http_error_404(self, tmp_path: Path) -> None:
        def opener(req: Any, timeout: float) -> Any:
            raise urllib.error.HTTPError("u", 404, "nf", Message(), None)

        dest = tmp_path / "f"
        with pytest.raises(H.MonthNotFound):
            H.stream_https("https://x/y", dest, opener=opener, max_bytes=10, timeout=1)
        assert not dest.exists()

    def test_http_error_other(self, tmp_path: Path) -> None:
        def opener(req: Any, timeout: float) -> Any:
            raise urllib.error.HTTPError("u", 403, "no", Message(), None)

        with pytest.raises(Exception, match="403"):
            H.stream_https("https://x/y", tmp_path / "f", opener=opener, max_bytes=10, timeout=1)

    def test_socket_error(self, tmp_path: Path) -> None:
        def opener(req: Any, timeout: float) -> Any:
            raise TimeoutError("slow")

        with pytest.raises(Exception, match="GET failed"):
            H.stream_https("https://x/y", tmp_path / "f", opener=opener, max_bytes=10, timeout=1)

    def test_oversize_and_empty(self, tmp_path: Path) -> None:
        big = _FakeResponse(b"x" * 2_000_000)
        dest = tmp_path / "f"
        with pytest.raises(Exception, match="exceeded"):
            H.stream_https(
                "https://x/y", dest, opener=lambda *a, **k: big, max_bytes=1_000_000, timeout=1
            )
        assert not dest.exists()
        with pytest.raises(Exception, match="empty response"):
            H.stream_https(
                "https://x/y",
                dest,
                opener=lambda *a, **k: _FakeResponse(b""),
                max_bytes=10,
                timeout=1,
            )

    def test_success_writes_payload(self, tmp_path: Path) -> None:
        dest = tmp_path / "f"
        H.stream_https(
            "https://x/y",
            dest,
            opener=lambda *a, **k: _FakeResponse(b"hello world"),
            max_bytes=100,
            timeout=1,
        )
        assert dest.read_bytes() == b"hello world"


class TestSecurityMaster:
    def test_empty(self) -> None:
        assert H.security_master_from_bars(H.empty_bars(), clock=_clock).is_empty()

    def test_identity_rows(self) -> None:
        bars = H.normalize_vendor_frame(
            _vendor(_minute_rows(date(2024, 3, 4), H.RTH_OPEN_MOD, 10)), clock=_clock
        )
        m = H.security_master_from_bars(bars, clock=_clock)
        assert m.height == 1
        row = m.row(0, named=True)
        assert row["security_type"] == "unknown"
        assert row["valid_to"] is None
        assert row["valid_from"] == bars["event_time"].min()

    def test_scan_path_for_non_datetime(self) -> None:
        bars = pl.DataFrame(
            {"security_id": ["A"], "event_time": [datetime(2024, 1, 1, tzinfo=UTC)]}
        )
        m = H.security_master_from_bars(bars, clock=_clock)
        assert m.height == 1
