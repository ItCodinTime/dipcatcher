"""Shared helpers: fixture files, fake clock, httpx cassette transports.

Every test replays recorded response bodies (tests/fixtures/vendors/) through
``httpx.MockTransport`` — no test touches the network, and no test needs an
API key (env vars are exercised only via injected ``environ`` mappings).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import httpx

from quant_fund.data.vendors.client import RetryPolicy, VendorClient

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "vendors"


def fixture_body(relative: str) -> bytes:
    return (FIXTURES / relative).read_bytes()


class FakeClock:
    """Monotonic clock + sleeper pair: sleep() advances the recorded time."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        assert seconds >= 0
        self.sleeps.append(seconds)
        self.now += seconds


class CassetteTransport(httpx.MockTransport):
    """MockTransport that serves queued responses and records requests."""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self._queue = list(responses)
        self.requests: list[httpx.Request] = []

        def _handle(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            if not self._queue:
                raise AssertionError(f"unexpected request: {request.url}")
            return self._queue.pop(0)

        super().__init__(_handle)


def cassette(
    *bodies: bytes | str,
    status: int = 200,
    headers: dict[str, str] | None = None,
) -> CassetteTransport:
    """Queued cassette: every body yields one response."""
    responses = []
    for body in bodies:
        payload = body.encode() if isinstance(body, str) else body
        responses.append(httpx.Response(status, content=payload, headers=headers or {}))
    return CassetteTransport(responses)


def json_response(
    payload: object, *, status: int = 200, headers: dict[str, str] | None = None
) -> httpx.Response:
    return httpx.Response(
        status,
        content=json.dumps(payload).encode(),
        headers={"content-type": "application/json", **(headers or {})},
    )


def make_client(
    transport: httpx.MockTransport,
    *,
    base_url: str = "https://fixture.invalid",
    retry: RetryPolicy | None = None,
    fake: FakeClock | None = None,
    **kwargs: object,
) -> tuple[VendorClient, FakeClock]:
    """VendorClient bound to a cassette transport with a fake clock."""
    clock = fake or FakeClock()
    client = VendorClient(
        base_url,
        transport=transport,
        retry=retry or RetryPolicy(jitter_seconds=0.0, backoff_base_seconds=0.01),
        clock=clock.monotonic,
        sleeper=clock.sleep,
        rng=kwargs.pop("rng", random.Random(0)),
        **kwargs,
    )
    return client, clock


__all__ = [
    "FIXTURES",
    "fixture_body",
    "FakeClock",
    "CassetteTransport",
    "cassette",
    "json_response",
    "make_client",
]
