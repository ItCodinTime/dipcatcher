"""Bounded concurrent I/O: ordered results, retries, rate limits, pooling.

Market-data downloads and multi-file reads share this helper so bursts stay
capped, transient failures back off, and callers still see input order.
In-flight socket reads run until their timeout; queued work stops when the
batch is cancelled.
"""

from __future__ import annotations

import asyncio
import http.client
import inspect
import random
import threading
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import cast
from urllib.parse import urljoin, urlsplit

_CHUNK = 1024 * 1024


class IoError(Exception):
    """Non-retryable transport failure (scheme, size cap, redirect loop)."""


class IoCancelled(BaseException):
    """The caller cancelled the batch before this item finished."""


def _default_sleep(delay: float) -> None:
    time.sleep(delay)


def _raise_if_cancelled(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise IoCancelled("I/O batch cancelled")


def _sleep(delay: float, cancel: threading.Event | None, sleeper: Callable[[float], None]) -> None:
    if delay <= 0:
        _raise_if_cancelled(cancel)
        return
    if cancel is None:
        sleeper(delay)
        return
    deadline = time.monotonic() + delay
    while True:
        _raise_if_cancelled(cancel)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        sleeper(min(0.05, remaining))


class RateLimiter:
    """Minimum spacing between acquisitions, shared across threads."""

    def __init__(
        self,
        min_interval_s: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if min_interval_s < 0:
            raise ValueError("min_interval_s must be non-negative")
        self.min_interval_s = float(min_interval_s)
        self._clock = clock
        self._sleeper = sleeper
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self, cancel: threading.Event | None = None) -> None:
        _raise_if_cancelled(cancel)
        if self.min_interval_s <= 0:
            return
        while True:
            _raise_if_cancelled(cancel)
            with self._lock:
                now = self._clock()
                if now >= self._next:
                    self._next = now + self.min_interval_s
                    return
                wait = self._next - now
            # Re-check cancellation between short sleeps instead of one blocking wait.
            self._sleeper(wait if cancel is None else min(0.05, wait))


class AsyncRateLimiter:
    """Async counterpart of :class:`RateLimiter`. The lock is not held while sleeping."""

    def __init__(self, min_interval_s: float) -> None:
        if min_interval_s < 0:
            raise ValueError("min_interval_s must be non-negative")
        self.min_interval_s = float(min_interval_s)
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        if self.min_interval_s <= 0:
            return
        loop = asyncio.get_running_loop()
        async with self._lock:
            now = loop.time()
            scheduled = now if now >= self._next else self._next
            self._next = scheduled + self.min_interval_s
            delay = scheduled - now
        if delay > 0:
            await asyncio.sleep(delay)


def call_with_retry[R](
    fn: Callable[[], R],
    *,
    retries: int = 2,
    backoff_s: float = 0.25,
    max_backoff_s: float = 8.0,
    jitter: bool = True,
    retry_on: tuple[type[BaseException], ...] = (TimeoutError, OSError),
    cancel: threading.Event | None = None,
    sleeper: Callable[[float], None] | None = None,
    rng: random.Random | None = None,
) -> R:
    """Call ``fn`` until it returns or retries are exhausted.

    Backoff is exponential. With ``jitter``, each wait is uniform in
    ``[0, min(max_backoff_s, backoff_s * 2**attempt)]``.
    """
    if retries < 0:
        raise ValueError("retries must be non-negative")
    if backoff_s < 0 or max_backoff_s < 0:
        raise ValueError("backoff must be non-negative")
    sleep = _default_sleep if sleeper is None else sleeper
    uniform = rng.uniform if rng is not None else random.uniform
    attempt = 0
    while True:
        _raise_if_cancelled(cancel)
        try:
            return fn()
        except retry_on as exc:
            if attempt >= retries:
                raise
            delay = min(max_backoff_s, backoff_s * (2.0**attempt))
            attempt += 1
            if jitter and delay > 0:
                delay = uniform(0.0, delay)
            _sleep(delay, cancel, sleep)
            del exc


def map_ordered[T, R](
    fn: Callable[[T], R],
    items: Sequence[T],
    *,
    max_workers: int = 8,
    min_interval_s: float = 0.0,
    retries: int = 0,
    backoff_s: float = 0.25,
    max_backoff_s: float = 8.0,
    jitter: bool = True,
    retry_on: tuple[type[BaseException], ...] = (TimeoutError, OSError),
    cancel: threading.Event | None = None,
) -> list[R]:
    """Run ``fn`` over ``items`` and return results in input order.

    ``min_interval_s`` spaces *starts* (a rate limit), independent of the
    worker cap. The first real failure in input order is re-raised after
    in-flight work finishes. Setting ``cancel`` skips work that has not
    started and stops further retries.
    """
    seq = list(items)
    if not seq:
        return []
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    cancel_event = cancel if cancel is not None else threading.Event()
    limiter = RateLimiter(min_interval_s)

    def run_one(item: T) -> R:
        limiter.acquire(cancel_event)

        def attempt() -> R:
            _raise_if_cancelled(cancel_event)
            return fn(item)

        return call_with_retry(
            attempt,
            retries=retries,
            backoff_s=backoff_s,
            max_backoff_s=max_backoff_s,
            jitter=jitter,
            retry_on=retry_on,
            cancel=cancel_event,
        )

    if len(seq) == 1 or max_workers == 1:
        return [run_one(item) for item in seq]

    results: list[R | None] = [None] * len(seq)
    errors: list[BaseException | None] = [None] * len(seq)
    pool = ThreadPoolExecutor(
        max_workers=min(max_workers, len(seq)),
        thread_name_prefix="qf-io",
    )
    try:
        futures = {pool.submit(run_one, item): index for index, item in enumerate(seq)}
        try:
            for fut in as_completed(futures):
                index = futures[fut]
                try:
                    results[index] = fut.result()
                except IoCancelled as exc:
                    errors[index] = exc
                except BaseException as exc:
                    errors[index] = exc
        except BaseException:
            cancel_event.set()
            raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    for found in errors:
        if found is not None and not isinstance(found, IoCancelled):
            raise found
    for found in errors:
        if isinstance(found, IoCancelled):
            raise found
    return cast(list[R], results)


async def _call_async[T, R](
    fn: Callable[[T], R | Awaitable[R]],
    item: T,
    *,
    retries: int,
    backoff_s: float,
    max_backoff_s: float,
    jitter: bool,
    retry_on: tuple[type[BaseException], ...],
) -> R:
    attempt = 0
    while True:
        try:
            if inspect.iscoroutinefunction(fn):
                value = await fn(item)  # type: ignore[misc]
            else:
                value = await asyncio.to_thread(fn, item)
            return cast(R, value)
        except retry_on as exc:
            if attempt >= retries:
                raise
            delay = min(max_backoff_s, backoff_s * (2.0**attempt))
            attempt += 1
            if jitter and delay > 0:
                delay = random.uniform(0.0, delay)
            if delay > 0:
                await asyncio.sleep(delay)
            del exc


async def amap_ordered[T, R](
    fn: Callable[[T], R | Awaitable[R]],
    items: Sequence[T],
    *,
    max_in_flight: int = 8,
    min_interval_s: float = 0.0,
    retries: int = 0,
    backoff_s: float = 0.25,
    max_backoff_s: float = 8.0,
    jitter: bool = True,
    retry_on: tuple[type[BaseException], ...] = (TimeoutError, OSError),
) -> list[R]:
    """Async ordered map. Cancelling the task cancels queued children.

    Sync callables run in threads, still bounded by ``max_in_flight``.
    """
    seq = list(items)
    if not seq:
        return []
    if max_in_flight < 1:
        raise ValueError("max_in_flight must be >= 1")
    if retries < 0:
        raise ValueError("retries must be non-negative")
    sem = asyncio.Semaphore(max_in_flight)
    limiter = AsyncRateLimiter(min_interval_s)

    async def one(item: T) -> R:
        async with sem:
            await limiter.acquire()
            return await _call_async(
                fn,
                item,
                retries=retries,
                backoff_s=backoff_s,
                max_backoff_s=max_backoff_s,
                jitter=jitter,
                retry_on=retry_on,
            )

    tasks = [asyncio.create_task(one(item)) for item in seq]
    try:
        raw = await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    for item in raw:
        if isinstance(item, asyncio.CancelledError):
            raise item
    for item in raw:
        if isinstance(item, BaseException):
            raise item
    return cast(list[R], raw)


class ConnectionPool:
    """Small keep-alive pool keyed by scheme, host, and port.

    Idle connections are capped per host. In-use connections are owned by
    the caller until :meth:`release` or :meth:`discard`.
    """

    def __init__(self, max_per_host: int = 8) -> None:
        if max_per_host < 1:
            raise ValueError("max_per_host must be >= 1")
        self.max_per_host = int(max_per_host)
        self._idle: dict[tuple[str, str, int], list[http.client.HTTPConnection]] = {}
        self._lock = threading.Lock()

    def checkout(
        self, scheme: str, host: str, port: int, timeout: float
    ) -> http.client.HTTPConnection:
        key = (scheme, host, port)
        with self._lock:
            bucket = self._idle.get(key)
            while bucket:
                conn = bucket.pop()
                conn.timeout = timeout
                return conn
        if scheme == "https":
            conn = http.client.HTTPSConnection(host, port, timeout=timeout)
        elif scheme == "http":
            conn = http.client.HTTPConnection(host, port, timeout=timeout)
        else:
            raise IoError(f"unsupported URL scheme: {scheme}")
        return conn

    def release(self, scheme: str, host: str, port: int, conn: http.client.HTTPConnection) -> None:
        key = (scheme, host, port)
        with self._lock:
            bucket = self._idle.setdefault(key, [])
            if len(bucket) < self.max_per_host:
                bucket.append(conn)
                return
        conn.close()

    def discard(self, conn: http.client.HTTPConnection) -> None:
        conn.close()


_POOL_LOCK = threading.Lock()
_DEFAULT_POOL: ConnectionPool | None = None


def default_pool() -> ConnectionPool:
    global _DEFAULT_POOL
    with _POOL_LOCK:
        if _DEFAULT_POOL is None:
            _DEFAULT_POOL = ConnectionPool()
        return _DEFAULT_POOL


def _split(url: str) -> tuple[str, str, int, str]:
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"} or not parts.hostname:
        raise IoError(f"unsupported URL: {url}")
    port = parts.port or (443 if scheme == "https" else 80)
    path = urljoin("/", parts.path or "/")
    if parts.query:
        path = f"{path}?{parts.query}"
    return scheme, parts.hostname, port, path


def _read_capped(response: http.client.HTTPResponse, max_bytes: int) -> tuple[bytes, bool]:
    chunks: list[bytes] = []
    total = 0
    while True:
        block = response.read(min(_CHUNK, max_bytes + 1 - total))
        if not block:
            return b"".join(chunks), False
        total += len(block)
        if total > max_bytes:
            return b"", True
        chunks.append(block)


def pooled_request(
    method: str,
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float,
    max_bytes: int,
    pool: ConnectionPool | None = None,
    max_redirects: int = 10,
) -> tuple[int, bytes]:
    """GET/HEAD (or other methods without a body) through the connection pool.

    Redirects are followed for GET and HEAD. Responses over ``max_bytes``
    raise :class:`IoError` and the connection is not reused.
    """
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    if max_redirects < 0:
        raise ValueError("max_redirects must be non-negative")
    pool = pool or default_pool()
    current = url
    redirects = 0
    while True:
        scheme, host, port, path = _split(current)
        conn = pool.checkout(scheme, host, port, timeout)
        try:
            conn.request(method, path, headers=dict(headers or {}))
            response = conn.getresponse()
            status = int(response.status)
            location = response.getheader("Location")
            body, too_big = _read_capped(response, max_bytes)
            reusable = (not too_big) and (not response.will_close)
            if too_big:
                pool.discard(conn)
                raise IoError(f"response exceeded {max_bytes} bytes: {current}")
            if reusable:
                pool.release(scheme, host, port, conn)
            else:
                pool.discard(conn)
        except IoError:
            raise
        except Exception:
            pool.discard(conn)
            raise
        if status in {301, 302, 303, 307, 308} and method.upper() in {"GET", "HEAD"} and location:
            if redirects >= max_redirects:
                raise IoError(f"too many redirects: {url}")
            redirects += 1
            current = urljoin(current, location)
            if status == 303:
                method = "GET"
            continue
        return status, body


def _release_response(
    pool: ConnectionPool,
    scheme: str,
    host: str,
    port: int,
    conn: http.client.HTTPConnection,
    response: http.client.HTTPResponse,
) -> None:
    if response.will_close:
        pool.discard(conn)
    else:
        pool.release(scheme, host, port, conn)


def pooled_stream(
    url: str,
    dest: Path,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float,
    max_bytes: int,
    pool: ConnectionPool | None = None,
    max_redirects: int = 10,
) -> int:
    """Stream one GET to ``dest``. Oversize and transport errors delete ``dest``.

    Redirects are followed, same cap as :func:`pooled_request`. Only the final
    response is written.
    """
    path = Path(dest)
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    if max_redirects < 0:
        raise ValueError("max_redirects must be non-negative")
    pool = pool or default_pool()
    path.parent.mkdir(parents=True, exist_ok=True)
    current = url
    redirects = 0
    while True:
        scheme, host, port, request_path = _split(current)
        conn = pool.checkout(scheme, host, port, timeout)
        handed_off = False
        try:
            conn.request("GET", request_path, headers=dict(headers or {}))
            response = conn.getresponse()
            status = int(response.status)
            location = response.getheader("Location")
            if status in {301, 302, 303, 307, 308} and location:
                too_big = _read_capped(response, max_bytes)[1]
                if too_big or response.will_close:
                    pool.discard(conn)
                else:
                    pool.release(scheme, host, port, conn)
                handed_off = True
                if too_big:
                    raise IoError(f"response exceeded {max_bytes} bytes: {current}")
                if redirects >= max_redirects:
                    raise IoError(f"too many redirects: {url}")
                redirects += 1
                current = urljoin(current, location)
                continue
            total = 0
            with path.open("wb") as handle:
                while True:
                    block = response.read(_CHUNK)
                    if not block:
                        break
                    total += len(block)
                    if total > max_bytes:
                        pool.discard(conn)
                        handed_off = True
                        raise IoError(f"response exceeded {max_bytes} bytes: {current}")
                    handle.write(block)
            _release_response(pool, scheme, host, port, conn, response)
            handed_off = True
            if status >= 400 or total < 1:
                path.unlink(missing_ok=True)
            return status
        except Exception:
            if not handed_off:
                pool.discard(conn)
            path.unlink(missing_ok=True)
            raise
