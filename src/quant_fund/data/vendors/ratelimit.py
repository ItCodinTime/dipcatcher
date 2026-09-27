"""Token-bucket rate limiter with injectable clock/sleeper.

The limiter is deliberately synchronous and dependency-free: ``acquire``
blocks by calling the injected ``sleeper`` (``time.sleep`` by default) and
reads time from the injected ``clock`` (``time.monotonic`` by default).
Tests inject fakes so no test ever sleeps or touches a wall clock.
"""

from __future__ import annotations

import time
from collections.abc import Callable


class TokenBucket:
    """Classic token bucket: steady drain at ``rate_per_second``, finite burst.

    The bucket starts full. ``acquire`` consumes ``tokens`` and returns the
    number of seconds the caller had to wait (0.0 when tokens were already
    available). The injected ``sleeper`` must advance the injected ``clock``
    by the slept duration for the refill math to progress — the real
    ``time.sleep``/``time.monotonic`` pair satisfies this trivially.
    """

    def __init__(
        self,
        rate_per_second: float,
        *,
        capacity: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if rate_per_second <= 0:
            raise ValueError("rate_per_second must be positive")
        cap = float(capacity) if capacity is not None else max(1.0, rate_per_second)
        if cap <= 0:
            raise ValueError("capacity must be positive")
        self.rate_per_second = float(rate_per_second)
        self.capacity = cap
        self._tokens = cap
        self._clock = clock
        self._sleeper = sleeper
        self._updated = float(clock())

    @property
    def tokens(self) -> float:
        """Current token balance after lazy refill (read-only)."""
        self._refill()
        return self._tokens

    def _refill(self) -> None:
        now = float(self._clock())
        elapsed = now - self._updated
        if elapsed > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate_per_second)
            self._updated = now

    def acquire(self, tokens: float = 1.0) -> float:
        """Block until ``tokens`` are available; return seconds waited."""
        if tokens <= 0:
            raise ValueError("tokens must be positive")
        if tokens > self.capacity:
            raise ValueError(
                f"requested {tokens} tokens exceeds bucket capacity {self.capacity}"
            )
        waited = 0.0
        while True:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return waited
            deficit = tokens - self._tokens
            wait = deficit / self.rate_per_second
            self._sleeper(wait)
            waited += wait
