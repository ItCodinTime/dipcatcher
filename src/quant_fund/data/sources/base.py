"""Shared primitives for explicit, point-in-time public-data fetches."""

from __future__ import annotations

import http.client
import json
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlencode, urlsplit

import polars as pl

from quant_fund.data.concurrent_io import IoError, call_with_retry, pooled_request


class SourceError(RuntimeError):
    """Raised when a source cannot be fetched or normalized safely."""


class HttpClient:
    """Small standard-library HTTP client with bounded retries and response size."""

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        retries: int = 2,
        max_bytes: int = 25_000_000,
        user_agent: str = "dipcatcher/1.0 (research; contact=research@example.invalid)",
    ) -> None:
        if timeout <= 0 or retries < 0 or max_bytes < 1:
            raise ValueError("timeout must be positive; retries non-negative; max_bytes positive")
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.max_bytes = int(max_bytes)
        self.user_agent = user_agent

    def _request(self, url: str, *, headers: dict[str, str] | None = None) -> bytes:
        if urlsplit(url).scheme.lower() not in {"http", "https"}:
            raise SourceError(f"unsupported URL scheme: {url}")
        request_headers = {"User-Agent": self.user_agent, "Accept": "*/*"}
        request_headers.update(headers or {})

        def once() -> bytes:
            try:
                status, body = pooled_request(
                    "GET",
                    url,
                    headers=request_headers,
                    timeout=self.timeout,
                    max_bytes=self.max_bytes,
                )
            except IoError as exc:
                raise SourceError(str(exc)) from exc
            if status < 400:
                return body
            # 429 and 5xx are transient. Other 4xx will not succeed on retry.
            if status == 429 or status >= 500:
                raise OSError(status, f"HTTP {status}")
            raise SourceError(f"GET failed after retries: {url}")

        try:
            return call_with_retry(
                once,
                retries=self.retries,
                backoff_s=1.0,
                max_backoff_s=4.0,
                jitter=True,
                # http.client transport failures (IncompleteRead, RemoteDisconnected,
                # BadStatusLine) are not OSError subclasses and must retry too.
                retry_on=(OSError, http.client.HTTPException),
            )
        except (OSError, http.client.HTTPException) as exc:
            raise SourceError(f"GET failed after retries: {url}") from exc

    def get_json(self, url: str, *, headers: dict[str, str] | None = None) -> Any:
        try:
            return json.loads(
                self._request(url, headers={"Accept": "application/json", **(headers or {})})
            )
        except json.JSONDecodeError as exc:
            raise SourceError(f"invalid JSON response: {url}") from exc

    def get_text(self, url: str, *, headers: dict[str, str] | None = None) -> str:
        return self._request(url, headers=headers).decode("utf-8-sig", errors="replace")

    def get_bytes(self, url: str, *, headers: dict[str, str] | None = None) -> bytes:
        return self._request(url, headers=headers)


def utc_now() -> datetime:
    return datetime.now(UTC)


def parse_time(value: Any) -> datetime:
    """Parse common vendor timestamps and return an aware UTC datetime."""
    if isinstance(value, (int, float)):
        number = float(value)
        if number > 10_000_000_000:
            number /= 1000.0
        return datetime.fromtimestamp(number, tz=UTC)
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    return (parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed).astimezone(UTC)


def pit_frame(rows: list[dict[str, Any]], *, source: str, revision_id: str = "v1") -> pl.DataFrame:
    """Construct a normalized frame and stamp the PIT contract on every row."""
    ingested = utc_now()
    stamped: list[dict[str, Any]] = []
    for original in rows:
        row = dict(original)
        if "event_time" not in row:
            raise SourceError("source row is missing event_time")
        try:
            event = parse_time(row.pop("event_time"))
            available_raw = row.pop("available_time", None)
            available = event if available_raw is None else parse_time(available_raw)
        except (TypeError, ValueError, OverflowError, OSError) as exc:
            raise SourceError(f"source row has an unparseable timestamp: {exc}") from exc
        if event > available or available > ingested:
            raise SourceError("impossible event_time/available_time/ingested_time chain")
        stamped.append(
            {
                **row,
                "event_time": event,
                "available_time": available,
                "ingested_time": ingested,
                "source": source,
                "revision_id": revision_id,
            }
        )
    return pl.DataFrame(stamped) if stamped else pl.DataFrame()


def query_url(base: str, params: dict[str, Any]) -> str:
    encoded = urlencode({key: value for key, value in params.items() if value is not None})
    return f"{base}?{encoded}" if encoded else base


class SourceAdapter:
    """Nominal base class used by the registry and CLI."""

    name = "source"

    def __init__(self, client: HttpClient | None = None) -> None:
        self.client = client or HttpClient()

    def fetch(self, **kwargs: Any) -> pl.DataFrame:
        raise NotImplementedError

    def get_bars(self, **kwargs: Any) -> pl.DataFrame:
        return self.fetch(**kwargs)

    def get_corporate_actions(self, **_: Any) -> pl.DataFrame:
        return pl.DataFrame()

    def get_security_master(self) -> pl.DataFrame:
        return pl.DataFrame()
