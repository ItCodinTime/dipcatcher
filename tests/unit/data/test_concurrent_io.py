"""Ordered concurrent I/O: retries, rate limits, pooling, cancellation."""

from __future__ import annotations

import asyncio
import random
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import polars as pl
import pytest

from quant_fund.data.concurrent_io import (
    AsyncRateLimiter,
    ConnectionPool,
    IoCancelled,
    IoError,
    RateLimiter,
    amap_ordered,
    call_with_retry,
    default_pool,
    map_ordered,
    pooled_request,
    pooled_stream,
)
from quant_fund.data.sources.base import HttpClient, SourceError

_START = datetime(2024, 1, 2, tzinfo=UTC)
_END = datetime(2024, 1, 4, tzinfo=UTC)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        server = self.server
        assert isinstance(server, _Server)
        if path == "/redirect":
            self._send(302, b"", location="/ok")
            return
        if path == "/loop":
            self._send(302, b"", location="/loop")
            return
        if path == "/big":
            self._send(200, b"x" * 64)
            return
        if path == "/flaky":
            server.flaky_hits += 1
            if server.flaky_hits == 1:
                self._send(503, b"busy")
                return
            self._send(200, b"ok")
            return
        if path == "/missing":
            self._send(404, b"no")
            return
        if path == "/hang":
            time.sleep(1.0)
            self._send(200, b"late")
            return
        self._send(200, b"ok")

    def _send(self, status: int, body: bytes, *, location: str | None = None) -> None:
        self.send_response(status)
        if location is not None:
            self.send_header("Location", location)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


class _Server(ThreadingHTTPServer):
    flaky_hits: int
    accepts: int

    def server_bind(self) -> None:
        super().server_bind()
        self.flaky_hits = 0
        self.accepts = 0

    def get_request(self) -> tuple[object, object]:
        conn, addr = super().get_request()
        self.accepts += 1
        return conn, addr


@pytest.fixture
def http_server() -> Iterator[_Server]:
    server = _Server(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _url(server: _Server, path: str) -> str:
    host, port = server.server_address[:2]
    return f"http://{host}:{port}{path}"


def test_call_with_retry_exponential_without_jitter() -> None:
    sleeps: list[float] = []
    calls = {"n": 0}

    def fn() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("transient")
        return "ok"

    assert (
        call_with_retry(
            fn,
            retries=2,
            backoff_s=0.2,
            max_backoff_s=8.0,
            jitter=False,
            sleeper=sleeps.append,
        )
        == "ok"
    )
    assert sleeps == [0.2, 0.4]
    assert calls["n"] == 3


def test_call_with_retry_jitter_is_bounded_and_terminal_errors_do_not_retry() -> None:
    rng = random.Random(0)
    sleeps: list[float] = []

    def fail() -> None:
        raise TimeoutError("slow")

    with pytest.raises(TimeoutError):
        call_with_retry(
            fail,
            retries=3,
            backoff_s=0.5,
            max_backoff_s=1.0,
            jitter=True,
            rng=rng,
            sleeper=sleeps.append,
        )
    assert len(sleeps) == 3
    assert sleeps[0] <= 0.5
    assert sleeps[1] <= 1.0
    assert all(delay >= 0 for delay in sleeps)

    terminal = {"n": 0}

    def bad() -> None:
        terminal["n"] += 1
        raise ValueError("no")

    with pytest.raises(ValueError):
        call_with_retry(bad, retries=4, retry_on=(OSError,), sleeper=sleeps.append)
    assert terminal["n"] == 1


def test_call_with_retry_cancel_during_backoff() -> None:
    cancel = threading.Event()

    def sleeper(_delay: float) -> None:
        cancel.set()

    def fail() -> None:
        raise OSError("again")

    with pytest.raises(IoCancelled):
        call_with_retry(
            fail,
            retries=2,
            backoff_s=1.0,
            jitter=False,
            cancel=cancel,
            sleeper=sleeper,
        )


def test_rate_limiter_spaces_starts_and_honours_cancel() -> None:
    now = {"t": 0.0}

    def clock() -> float:
        return now["t"]

    def sleeper(delay: float) -> None:
        now["t"] += delay

    limiter = RateLimiter(0.2, clock=clock, sleeper=sleeper)
    limiter.acquire()
    limiter.acquire()
    limiter.acquire()
    assert now["t"] == pytest.approx(0.4)

    idle = RateLimiter(0.0, clock=clock, sleeper=sleeper)
    idle.acquire()
    assert now["t"] == pytest.approx(0.4)

    with pytest.raises(ValueError):
        RateLimiter(-1)

    cancel = threading.Event()
    cancel.set()
    with pytest.raises(IoCancelled):
        RateLimiter(1.0).acquire(cancel)


def test_map_ordered_preserves_order_and_first_error() -> None:
    barrier = threading.Barrier(2)

    def work(item: int) -> int:
        barrier.wait(timeout=3)
        return item * 10

    assert map_ordered(work, [1, 2], max_workers=2) == [10, 20]
    assert map_ordered(lambda item: item, []) == []
    assert map_ordered(lambda item: item, [], max_workers=0) == []
    with pytest.raises(ValueError):
        map_ordered(lambda item: item, [1], max_workers=0)

    def boom(item: int) -> int:
        if item == 0:
            raise ValueError("earlier")
        time.sleep(0.05)
        raise RuntimeError("later")

    with pytest.raises(ValueError, match="earlier"):
        map_ordered(boom, [0, 1], max_workers=2)

    cancel = threading.Event()
    cancel.set()
    with pytest.raises(IoCancelled):
        map_ordered(lambda item: item, [1, 2], max_workers=2, cancel=cancel)


def test_amap_ordered_preserves_order_and_cancels() -> None:
    async def scenario() -> None:
        async def work(item: int) -> int:
            await asyncio.sleep(0.01 * (3 - item))
            return item

        assert await amap_ordered(work, [1, 2, 3]) == [1, 2, 3]
        assert await amap_ordered(lambda item: item + 1, [1, 2]) == [2, 3]

        started = asyncio.Event()

        async def slow(item: int) -> int:
            started.set()
            await asyncio.sleep(30)
            return item

        task = asyncio.create_task(amap_ordered(slow, [1, 2], max_in_flight=2))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        async def flaky(item: int) -> int:
            if item == 0:
                raise ValueError("first")
            return item

        with pytest.raises(ValueError, match="first"):
            await amap_ordered(flaky, [0, 1])

    asyncio.run(scenario())
    assert asyncio.run(amap_ordered(lambda item: item, [])) == []
    with pytest.raises(ValueError):
        asyncio.run(amap_ordered(lambda item: item, [1], max_in_flight=0))


def test_async_rate_limiter_spaces_acquires() -> None:
    async def scenario() -> float:
        limiter = AsyncRateLimiter(0.05)
        loop = asyncio.get_running_loop()
        started = loop.time()
        await limiter.acquire()
        await limiter.acquire()
        return loop.time() - started

    assert asyncio.run(scenario()) >= 0.04
    with pytest.raises(ValueError):
        AsyncRateLimiter(-0.1)

    async def zero() -> None:
        limiter = AsyncRateLimiter(0.0)
        await limiter.acquire()
        await limiter.acquire()

    asyncio.run(zero())


def test_connection_pool_reuses_and_caps_idle() -> None:
    pool = ConnectionPool(max_per_host=1)
    first = pool.checkout("http", "127.0.0.1", 9, timeout=1.0)
    second = pool.checkout("http", "127.0.0.1", 9, timeout=1.0)
    pool.release("http", "127.0.0.1", 9, first)
    pool.release("http", "127.0.0.1", 9, second)
    again = pool.checkout("http", "127.0.0.1", 9, timeout=1.0)
    assert again is first
    pool.discard(again)
    with pytest.raises(IoError):
        pool.checkout("ftp", "127.0.0.1", 9, timeout=1.0)
    with pytest.raises(ValueError):
        ConnectionPool(0)
    assert default_pool() is default_pool()


def test_pooled_request_reuses_redirects_and_limits(
    http_server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = ConnectionPool()
    status, body = pooled_request(
        "GET", _url(http_server, "/ok"), timeout=2, max_bytes=100, pool=pool
    )
    status2, body2 = pooled_request(
        "GET", _url(http_server, "/ok"), timeout=2, max_bytes=100, pool=pool
    )
    assert (status, body, status2, body2) == (200, b"ok", 200, b"ok")
    assert http_server.accepts == 1

    status, body = pooled_request(
        "GET", _url(http_server, "/redirect"), timeout=2, max_bytes=100, pool=pool
    )
    assert status == 200 and body == b"ok"
    with pytest.raises(IoError, match="redirect"):
        pooled_request(
            "GET",
            _url(http_server, "/loop"),
            timeout=2,
            max_bytes=100,
            pool=pool,
            max_redirects=1,
        )
    with pytest.raises(IoError, match="exceeded"):
        pooled_request("GET", _url(http_server, "/big"), timeout=2, max_bytes=8, pool=pool)
    with pytest.raises(TimeoutError):
        pooled_request("GET", _url(http_server, "/hang"), timeout=0.2, max_bytes=100, pool=pool)
    with pytest.raises(IoError):
        pooled_request("GET", "file:///etc/passwd", timeout=1, max_bytes=10)
    with pytest.raises(ValueError):
        pooled_request("GET", _url(http_server, "/ok"), timeout=0, max_bytes=10)

    dest = tmp_path / "body.bin"
    assert pooled_stream(_url(http_server, "/ok"), dest, timeout=2, max_bytes=100, pool=pool) == 200
    assert dest.read_bytes() == b"ok"
    redirected = tmp_path / "redirected.bin"
    assert (
        pooled_stream(
            _url(http_server, "/redirect"), redirected, timeout=2, max_bytes=100, pool=pool
        )
        == 200
    )
    assert redirected.read_bytes() == b"ok"
    missing = tmp_path / "missing.bin"
    assert (
        pooled_stream(_url(http_server, "/missing"), missing, timeout=2, max_bytes=100, pool=pool)
        == 404
    )
    assert not missing.exists()
    oversize = tmp_path / "big.bin"
    with pytest.raises(IoError, match="exceeded"):
        pooled_stream(_url(http_server, "/big"), oversize, timeout=2, max_bytes=8, pool=pool)
    assert not oversize.exists()

    monkeypatch.setattr("quant_fund.data.concurrent_io._default_sleep", lambda _delay: None)
    client = HttpClient(timeout=2, retries=1, max_bytes=100)
    assert client.get_bytes(_url(http_server, "/flaky")) == b"ok"
    assert http_server.flaky_hits == 2
    with pytest.raises(SourceError, match="failed after retries"):
        client.get_bytes(_url(http_server, "/missing"))


def test_yahoo_and_stooq_downloads_keep_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from quant_fund.data.adapters import stooq as stooq_mod
    from quant_fund.data.adapters import yahoo_eod as yahoo_mod

    barrier = threading.Barrier(2)

    def fetch_chart(symbol: str, **_kwargs: object) -> dict[str, object]:
        barrier.wait(timeout=3)
        return {"symbol": symbol}

    def parse_chart(
        _payload: dict[str, object], *, security_id: str, yahoo_symbol: str
    ) -> pl.DataFrame:
        return _bar(security_id, yahoo_symbol)

    monkeypatch.setattr(yahoo_mod, "fetch_yahoo_chart", fetch_chart)
    monkeypatch.setattr(yahoo_mod, "parse_yahoo_chart", parse_chart)
    yahoo = yahoo_mod.download_yahoo_universe(
        tmp_path / "yahoo",
        names=(("BBB", "BBB"), ("AAA", "AAA")),
        pause_s=0,
        max_workers=2,
        start=_START,
        end=_END,
    )
    assert yahoo["status"] == "ok"
    saved = pl.read_parquet(tmp_path / "yahoo" / "bars.parquet")
    assert saved["security_id"].unique().sort().to_list() == ["AAA", "BBB"]

    def fail_chart(symbol: str, **_kwargs: object) -> dict[str, object]:
        raise yahoo_mod.urllib.error.URLError(symbol)

    monkeypatch.setattr(yahoo_mod, "fetch_yahoo_chart", fail_chart)
    empty = yahoo_mod.download_yahoo_universe(
        tmp_path / "yahoo-empty",
        names=(("ZZ", "ZZ"), ("AA", "AA")),
        pause_s=0,
        max_workers=2,
    )
    assert empty["status"] == "empty"
    assert list(empty["errors"]) == ["ZZ", "AA"]

    def fetch_csv(symbol: str, **_kwargs: object) -> str:
        barrier.wait(timeout=3)
        return (
            "Date,Open,High,Low,Close,Volume\n"
            "2024-01-02,10,11,9,10.5,100\n"
            "2024-01-03,10.5,12,10,11,110\n"
        )

    monkeypatch.setattr(stooq_mod, "fetch_stooq_csv", fetch_csv)
    stooq = stooq_mod.download_stooq_universe(
        tmp_path / "stooq",
        names=(("MSFT", "msft.us"), ("AAPL", "aapl.us")),
        pause_s=0,
        max_workers=2,
        start=_START,
        end=_END,
    )
    assert stooq["status"] == "ok"
    tape = pl.read_parquet(tmp_path / "stooq" / "bars.parquet")
    assert tape["security_id"].unique().sort().to_list() == ["AAPL", "MSFT"]


def _bar(security_id: str, _symbol: str) -> pl.DataFrame:
    ts = datetime(2024, 1, 2, 21, tzinfo=UTC)
    return pl.DataFrame(
        {
            "security_id": [security_id],
            "symbol": [security_id],
            "event_time": [ts],
            "available_time": [ts],
            "ingested_time": [ts],
            "source": ["yahoo"],
            "revision_id": ["YAHOO_VENDOR_ADJ"],
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "volume": [100.0],
            "currency": ["USD"],
            "session": ["rth"],
        }
    )


def test_hf_months_and_deep_bars_fetch_concurrently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from quant_fund.data.adapters.hf_ohlcv_1m import _month_paths
    from quant_fund.paper.quantile_signals import load_deep_bars

    barrier = threading.Barrier(2)

    def fetcher(url: str, dest: Path) -> None:
        barrier.wait(timeout=3)
        month = url.rsplit("_", 1)[-1][:7]
        year_s, month_s = month.split("-")
        stamp = datetime(int(year_s), int(month_s), 2, 15, tzinfo=UTC)
        pl.DataFrame(
            {
                "timestamp": [stamp],
                "open": [1.0],
                "high": [1.1],
                "low": [0.9],
                "close": [1.0],
                "volume": [5.0],
                "ticker": ["AAA"],
            }
        ).write_parquet(dest)

    paths = _month_paths(
        cache=tmp_path / "hf",
        revision="rev",
        start=datetime(2024, 1, 15, tzinfo=UTC),
        end=datetime(2024, 2, 2, tzinfo=UTC),
        max_months=6,
        allow_download=True,
        fetcher=fetcher,
    )
    assert [path.name for path in paths] == ["ohlcv_2024-01.parquet", "ohlcv_2024-02.parquet"]

    stamps = [
        datetime(2024, 1, 2, tzinfo=UTC),
        datetime(2024, 1, 3, tzinfo=UTC),
    ]
    for symbol in ("aaa", "bbb"):
        pl.DataFrame(
            {"event_time": stamps, "close": [1.0, 2.0], "volume": [3.0, 4.0]}
        ).write_parquet(tmp_path / f"{symbol}_1d_deep.parquet")
    real = pl.read_parquet

    def wrapped(path: Path, *args: object, **kwargs: object) -> pl.DataFrame:
        barrier.wait(timeout=3)
        return real(path, *args, **kwargs)

    monkeypatch.setattr("quant_fund.paper.quantile_signals.pl.read_parquet", wrapped)
    panel = load_deep_bars(tmp_path, ["AAA", "BBB"], "1d")
    assert panel["security_id"].unique().sort().to_list() == ["AAA", "BBB"]


def test_yahoo_retries_throttle_but_not_terminal_status(monkeypatch: pytest.MonkeyPatch) -> None:
    from quant_fund.data.adapters.yahoo_eod import fetch_yahoo_chart

    monkeypatch.setattr("quant_fund.data.concurrent_io._default_sleep", lambda _delay: None)
    calls = {"n": 0}

    def throttled(*_args: object, **_kwargs: object) -> tuple[int, bytes]:
        calls["n"] += 1
        if calls["n"] < 3:
            return 429, b""
        return 200, b'{"chart": {"result": []}}'

    monkeypatch.setattr("quant_fund.data.adapters.yahoo_eod.pooled_request", throttled)
    payload = fetch_yahoo_chart("SPY", start=_START, end=_END, retries=3, backoff_s=0.01)
    assert payload["chart"]["result"] == []
    assert calls["n"] == 3

    calls["n"] = 0

    def missing(*_args: object, **_kwargs: object) -> tuple[int, bytes]:
        calls["n"] += 1
        return 404, b""

    monkeypatch.setattr("quant_fund.data.adapters.yahoo_eod.pooled_request", missing)
    with pytest.raises(yahoo_http_error()):
        fetch_yahoo_chart("SPY", start=_START, end=_END, retries=3, backoff_s=0.01)
    assert calls["n"] == 1


def yahoo_http_error() -> type[BaseException]:
    import urllib.error

    return urllib.error.HTTPError


def test_probe_names_overlap_and_keep_input_order(monkeypatch: pytest.MonkeyPatch) -> None:
    from fx1.data.sources.router import probe_names

    barrier = threading.Barrier(2)

    def get_spec(name: str) -> object:
        return name

    def build_adapter(name: str) -> object:
        class _Adapter:
            def probe(self) -> str:
                if name != "only":
                    barrier.wait(timeout=3)
                return f"probe:{name}"

        return _Adapter()

    monkeypatch.setattr("fx1.data.sources.router.get_spec", get_spec)
    monkeypatch.setattr("fx1.data.sources.router.build_adapter", build_adapter)
    assert probe_names(["wind", "ifind"]) == ["probe:wind", "probe:ifind"]
    assert probe_names(["only"]) == ["probe:only"]
