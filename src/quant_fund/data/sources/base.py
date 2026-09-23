"""Shared primitives for explicit, point-in-time public-data fetches."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

import polars as pl


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
        request = Request(url, headers=request_headers, method="GET")
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:  # noqa: S310  # nosec B310
                    body = response.read(self.max_bytes + 1)
                if len(body) > self.max_bytes:
                    raise SourceError(f"response exceeded {self.max_bytes} bytes: {url}")
                return body
            except HTTPError as exc:
                last_error = exc
                if exc.code < 500 and exc.code != 429:
                    break
            except (OSError, URLError, TimeoutError) as exc:
                last_error = exc
            if attempt < self.retries:
                time.sleep(min(2.0**attempt, 4.0))
        raise SourceError(f"GET failed after retries: {url}") from last_error

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
        event = parse_time(row.pop("event_time"))
        available_raw = row.pop("available_time", None)
        available = event if available_raw is None else parse_time(available_raw)
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
