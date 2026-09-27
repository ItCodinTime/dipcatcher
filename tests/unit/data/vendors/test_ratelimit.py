"""TokenBucket pacing math under a fake clock — no real sleeping."""

from __future__ import annotations

import pytest

from quant_fund.data.vendors.ratelimit import TokenBucket

from .helpers import FakeClock


def test_bucket_starts_full_and_first_acquire_is_free() -> None:
    clock = FakeClock()
    bucket = TokenBucket(10.0, capacity=3.0, clock=clock.monotonic, sleeper=clock.sleep)
    assert bucket.tokens == 3.0
    assert bucket.acquire() == 0.0
    assert clock.sleeps == []


def test_bucket_drains_at_rate_and_reports_wait() -> None:
    clock = FakeClock()
    bucket = TokenBucket(2.0, capacity=2.0, clock=clock.monotonic, sleeper=clock.sleep)
    assert bucket.acquire() == 0.0
    assert bucket.acquire() == 0.0
    # Third acquire needs one token at 2/s -> 0.5s wait.
    waited = bucket.acquire()
    assert waited == pytest.approx(0.5)
    assert clock.sleeps == [pytest.approx(0.5)]


def test_bucket_refills_proportionally_over_idle_time() -> None:
    clock = FakeClock()
    bucket = TokenBucket(4.0, capacity=4.0, clock=clock.monotonic, sleeper=clock.sleep)
    bucket.acquire(4.0)
    clock.now += 0.5  # idle half a second -> +2 tokens at 4/s
    assert bucket.tokens == pytest.approx(2.0)
    clock.now += 10.0  # long idle clamps at capacity
    assert bucket.tokens == 4.0


def test_bucket_never_beats_the_rate_over_a_run() -> None:
    clock = FakeClock()
    rate, capacity = 5.0, 3.0
    bucket = TokenBucket(rate, capacity=capacity, clock=clock.monotonic, sleeper=clock.sleep)
    consumed = 0
    for _ in range(12):
        bucket.acquire()
        consumed += 1
    # Elapsed wall time must be >= (consumed - initial capacity) / rate.
    assert clock.now >= (consumed - capacity) / rate - 1e-9
    assert clock.sleeps and all(s > 0 for s in clock.sleeps)


def test_bucket_validates_config() -> None:
    with pytest.raises(ValueError, match="rate_per_second"):
        TokenBucket(0.0)
    with pytest.raises(ValueError, match="capacity"):
        TokenBucket(1.0, capacity=0)
    bucket = TokenBucket(1.0, capacity=2.0, clock=lambda: 0.0, sleeper=lambda s: None)
    with pytest.raises(ValueError, match="capacity"):
        bucket.acquire(3.0)
    with pytest.raises(ValueError, match="positive"):
        bucket.acquire(0.0)
