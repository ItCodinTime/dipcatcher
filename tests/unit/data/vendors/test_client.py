"""VendorClient: pacing, retry, Retry-After honoring — all on cassettes.

Every request is served by ``httpx.MockTransport``; no socket is opened.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from quant_fund.data.vendors.client import (
    RetryPolicy,
    VendorClient,
    parse_retry_after,
)
from quant_fund.data.vendors.errors import (
    VendorHTTPError,
    VendorRateLimitError,
    VendorResponseError,
)

from .helpers import (
    CassetteTransport,
    FakeClock,
    cassette,
    json_response,
    make_client,
)


def test_get_success_records_request_and_paces() -> None:
    transport = cassette(b"hello", b"world")
    client, clock = make_client(transport, rate_per_second=100.0)
    response = client.get("/bars", params={"symbol": "SPY"}, headers={"X-Test": "1"})
    assert response.status_code == 200
    assert response.content == b"hello"
    request = transport.requests[0]
    assert request.url.path == "/bars"
    assert request.url.params["symbol"] == "SPY"
    assert request.headers["X-Test"] == "1"
    # First request was free (bucket starts full); second is paced.
    client.get("/bars2")
    assert len(clock.sleeps) == 1
    assert clock.sleeps[0] == pytest.approx(0.01)


def test_get_passes_absolute_urls_through() -> None:
    transport = cassette(b"ok")
    client, _ = make_client(transport)
    client.get("https://elsewhere.example/v1/x", params={"a": "b c"})
    assert str(transport.requests[0].url) == "https://elsewhere.example/v1/x?a=b+c"


def test_429_honors_retry_after_seconds() -> None:
    transport = CassetteTransport(
        [
            httpx.Response(429, content=b"slow down", headers={"Retry-After": "2"}),
            httpx.Response(200, content=b"fine"),
        ]
    )
    client, clock = make_client(transport)
    response = client.get("/x")
    assert response.status_code == 200
    # Delay = max(backoff, Retry-After) + jitter(0) -> exactly 2.0s slept.
    assert clock.sleeps == [pytest.approx(2.0)]
    assert len(transport.requests) == 2


def test_429_retry_after_http_date() -> None:
    now = datetime(2024, 1, 5, 12, 0, 0, tzinfo=UTC)
    retry_at = (now + timedelta(seconds=7)).strftime("%a, %d %b %Y %H:%M:%S GMT")
    transport = CassetteTransport(
        [
            httpx.Response(429, content=b"", headers={"Retry-After": retry_at}),
            httpx.Response(200, content=b"ok"),
        ]
    )
    client, clock = make_client(transport, now=lambda: now)
    client.get("/x")
    assert clock.sleeps == [pytest.approx(7.0)]


def test_5xx_retries_with_backoff() -> None:
    transport = CassetteTransport(
        [httpx.Response(503, content=b""), httpx.Response(200, content=b"ok")]
    )
    client, clock = make_client(
        transport, retry=RetryPolicy(backoff_base_seconds=0.25, jitter_seconds=0.0)
    )
    assert client.get("/x").status_code == 200
    assert clock.sleeps == [pytest.approx(0.25)]


def test_4xx_fails_closed_without_retry() -> None:
    transport = CassetteTransport([httpx.Response(404, content=b"nope")])
    client, _ = make_client(transport)
    with pytest.raises(VendorHTTPError, match="HTTP 404"):
        client.get("/x")
    assert len(transport.requests) == 1  # never retried


def test_429_exhaustion_raises_rate_limit_error() -> None:
    transport = CassetteTransport([httpx.Response(429) for _ in range(3)])
    client, _ = make_client(
        transport, retry=RetryPolicy(max_attempts=3, backoff_base_seconds=0.0)
    )
    with pytest.raises(VendorRateLimitError, match="throttled"):
        client.get("/x")
    assert len(transport.requests) == 3


def test_transport_error_retries_then_exhausts() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    transport = httpx.MockTransport(boom)
    client, _ = make_client(
        transport, retry=RetryPolicy(max_attempts=2, backoff_base_seconds=0.0)
    )
    with pytest.raises(VendorHTTPError, match="after 2 attempts"):
        client.get("/x")


def test_retry_after_never_shortens_backoff() -> None:
    # Retry-After of 0.001 < backoff 0.5 -> sleep stays >= backoff.
    transport = CassetteTransport(
        [
            httpx.Response(500, headers={"Retry-After": "0.001"}),
            httpx.Response(200, content=b"ok"),
        ]
    )
    client, clock = make_client(
        transport, retry=RetryPolicy(backoff_base_seconds=0.5, jitter_seconds=0.0)
    )
    client.get("/x")
    assert clock.sleeps[0] == pytest.approx(0.5)


def test_jitter_is_additive_and_bounded() -> None:
    transport = CassetteTransport(
        [httpx.Response(500), httpx.Response(500), httpx.Response(200)]
    )
    client, clock = make_client(
        transport,
        retry=RetryPolicy(
            max_attempts=3, backoff_base_seconds=1.0, jitter_seconds=0.5
        ),
        rng=random.Random(7),
    )
    client.get("/x")
    assert len(clock.sleeps) == 2
    # attempt 0 backoff = 1.0, attempt 1 = 2.0; each + [0, 0.5) jitter.
    assert 1.0 <= clock.sleeps[0] < 1.5
    assert 2.0 <= clock.sleeps[1] < 2.5


def test_get_json_and_bad_json() -> None:
    client, _ = make_client(cassette(json_response({"a": 1}).content))
    assert client.get_json("/x") == {"a": 1}
    client2, _ = make_client(cassette(b"not json{{{"))
    with pytest.raises(VendorResponseError, match="invalid JSON"):
        client2.get_json("/x")


def test_response_size_cap() -> None:
    client, _ = make_client(cassette(b"x" * 64), max_response_bytes=10)
    with pytest.raises(VendorResponseError, match="exceeded"):
        client.get("/x")


def test_invalid_client_config() -> None:
    with pytest.raises(ValueError, match="timeout"):
        VendorClient("https://x", timeout=0)
    with pytest.raises(ValueError, match="max_attempts"):
        RetryPolicy(max_attempts=0)
    with pytest.raises(ValueError, match="jitter"):
        RetryPolicy(jitter_seconds=-1)


def test_client_context_manager_closes() -> None:
    with make_client(cassette(b"ok"))[0] as client:
        assert client.get("/x").status_code == 200


def test_parse_retry_after_variants() -> None:
    assert parse_retry_after(None) is None
    assert parse_retry_after("") is None
    assert parse_retry_after(" 12 ") == 12.0
    assert parse_retry_after("1.5") == 1.5
    assert parse_retry_after("-3") == 0.0
    assert parse_retry_after("garbage") is None
    now = datetime(2024, 1, 1, tzinfo=UTC)
    future = (now + timedelta(seconds=42)).strftime("%a, %d %b %Y %H:%M:%S GMT")
    assert parse_retry_after(future, now=now) == pytest.approx(42.0)
    past = (now - timedelta(seconds=5)).strftime("%a, %d %b %Y %H:%M:%S GMT")
    assert parse_retry_after(past, now=now) == 0.0
