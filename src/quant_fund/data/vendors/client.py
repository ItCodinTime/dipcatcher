"""HTTP client for public market-data vendors.

One client shape for every vendor: token-bucket pacing before each request,
bounded retries with exponential backoff plus jitter, and explicit
``Retry-After`` honoring (seconds or HTTP-date). The transport layer is
``httpx`` — inject ``httpx.MockTransport`` (or a full ``httpx.Client``) so
tests replay recorded fixtures with zero network. Sleeper, monotonic clock,
wall clock, and RNG are all injectable for deterministic tests.
"""

from __future__ import annotations

import email.utils
import json
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx

from quant_fund.data.vendors.errors import (
    VendorHTTPError,
    VendorRateLimitError,
    VendorResponseError,
)
from quant_fund.data.vendors.ratelimit import TokenBucket

USER_AGENT = "dipcatcher-vendors/1.0 (research; public market-data endpoints)"
DEFAULT_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_BODY_SNIPPET = 300


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded retry with exponential backoff + jitter, honoring Retry-After.

    ``backoff_for(attempt)`` yields ``base * factor**attempt`` clamped to
    ``backoff_max_seconds``; when the server sends ``Retry-After`` the delay
    is at least that value (never less — the server's hint wins). A uniform
    ``[0, jitter_seconds]`` additive jitter decorrelates concurrent clients.
    """

    max_attempts: int = 4
    backoff_base_seconds: float = 0.5
    backoff_factor: float = 2.0
    backoff_max_seconds: float = 30.0
    jitter_seconds: float = 0.25
    retry_statuses: frozenset[int] = field(default_factory=lambda: DEFAULT_RETRY_STATUSES)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.backoff_base_seconds < 0:
            raise ValueError("backoff_base_seconds must be non-negative")
        if self.backoff_factor < 1:
            raise ValueError("backoff_factor must be >= 1")
        if self.backoff_max_seconds < 0:
            raise ValueError("backoff_max_seconds must be non-negative")
        if self.jitter_seconds < 0:
            raise ValueError("jitter_seconds must be non-negative")

    def backoff_for(self, attempt: int) -> float:
        """Exponential backoff for the 0-based ``attempt`` retry index."""
        raw = self.backoff_base_seconds * (self.backoff_factor**attempt)
        return min(raw, self.backoff_max_seconds)


def parse_retry_after(value: str | None, *, now: datetime | None = None) -> float | None:
    """Parse a ``Retry-After`` header to seconds; ``None`` when absent/unparseable.

    Supports both delta-seconds (``"120"``) and HTTP-date
    (``"Wed, 21 Oct 2015 07:28:00 GMT"``) forms. Negative deltas clamp to 0.
    """
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        parsed = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    reference = (now or datetime.now(tz=UTC)).astimezone(UTC)
    return max(0.0, (parsed - reference).total_seconds())


def _snippet(response: httpx.Response) -> str:
    try:
        return response.text[:_BODY_SNIPPET]
    except Exception:  # noqa: BLE001 — error path must never raise again
        return "<undecodable body>"


class VendorClient:
    """Paced, retrying GET client shared by every vendor adapter.

    No request is ever issued at construction: ``httpx.Client`` init is inert
    and the token bucket only gates inside :meth:`get`. Pass ``transport=``
    (e.g. ``httpx.MockTransport``) or ``http_client=`` to replay fixtures.
    """

    def __init__(
        self,
        base_url: str = "",
        *,
        rate_per_second: float = 1.0,
        capacity: float | None = None,
        retry: RetryPolicy | None = None,
        transport: httpx.BaseTransport | None = None,
        http_client: httpx.Client | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
        timeout: float = 30.0,
        max_response_bytes: int = 25_000_000,
        default_headers: Mapping[str, str] | None = None,
        user_agent: str = USER_AGENT,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_response_bytes < 1:
            raise ValueError("max_response_bytes must be positive")
        self.retry = retry or RetryPolicy()
        self._bucket = TokenBucket(
            rate_per_second, capacity=capacity, clock=clock, sleeper=sleeper
        )
        self._sleeper = sleeper
        self._rng = rng if rng is not None else random.Random()
        self._now = now or (lambda: datetime.now(tz=UTC))
        self.max_response_bytes = int(max_response_bytes)
        if http_client is not None:
            self._http = http_client
            self._owns_http = False
        else:
            headers = {"User-Agent": user_agent, "Accept": "*/*"}
            headers.update(default_headers or {})
            self._http = httpx.Client(
                base_url=base_url,
                transport=transport,
                timeout=timeout,
                follow_redirects=True,
                headers=headers,
            )
            self._owns_http = True

    # -- internals --

    def _sleep_retry(self, attempt: int, retry_after: float | None) -> float:
        """Sleep for the retry delay; returns the delay for observability."""
        base = self.retry.backoff_for(attempt)
        delay = base if retry_after is None else max(base, retry_after)
        delay += self._rng.uniform(0.0, self.retry.jitter_seconds)
        self._sleeper(delay)
        return delay

    def _check_size(self, response: httpx.Response, url: str) -> None:
        if len(response.content) > self.max_response_bytes:
            raise VendorResponseError(
                f"response exceeded {self.max_response_bytes} bytes: {url}"
            )

    # -- public API --

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        """GET ``url`` (absolute, or relative to ``base_url``) with pacing+retry.

        Retryable outcomes: transport errors (connect/read timeouts, resets)
        and statuses in ``retry.retry_statuses`` (default 429 + 5xx). All other
        4xx fail closed immediately — a 401/403/404 will never fix itself.
        """
        attempts = self.retry.max_attempts
        last_error: Exception | None = None
        last_status: int | None = None
        for attempt in range(attempts):
            self._bucket.acquire()
            try:
                response = self._http.get(url, params=params, headers=headers)
            except httpx.TransportError as exc:
                last_error = exc
                last_status = None
                if attempt < attempts - 1:
                    self._sleep_retry(attempt, retry_after=None)
                    continue
                break
            status = response.status_code
            if status in self.retry.retry_statuses:
                last_status = status
                last_error = VendorHTTPError(
                    f"GET {url} returned HTTP {status}: {_snippet(response)}",
                    status_code=status,
                )
                if attempt < attempts - 1:
                    retry_after = parse_retry_after(
                        response.headers.get("retry-after"), now=self._now()
                    )
                    self._sleep_retry(attempt, retry_after=retry_after)
                    continue
                break
            if status >= 400:
                raise VendorHTTPError(
                    f"GET {url} failed with HTTP {status}: {_snippet(response)}",
                    status_code=status,
                )
            self._check_size(response, url)
            return response
        if last_status == 429:
            raise VendorRateLimitError(
                f"GET {url} throttled after {attempts} attempts", status_code=429
            ) from last_error
        raise VendorHTTPError(
            f"GET {url} failed after {attempts} attempts", status_code=last_status
        ) from last_error

    def get_json(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        response = self.get(url, params=params, headers=headers)
        try:
            return json.loads(response.content.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            raise VendorResponseError(f"invalid JSON response: {url}") from exc

    def get_text(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> str:
        return self.get(url, params=params, headers=headers).content.decode(
            "utf-8-sig", errors="replace"
        )

    def get_bytes(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> bytes:
        return self.get(url, params=params, headers=headers).content

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> VendorClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
