"""Deterministic runtime: seeded randomness and a replayable effect trace.

Recording draws from a private PCG64 stream and appends every external
effect. Replay serves those effects back and raises
:class:`~quant_fund.simtest.eventlog.ReplayDivergence` when the call
sequence or a recomputed response differs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.simtest.eventlog import EventLog, ReplayDivergence
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

__all__ = ["DeterministicRuntime", "ReplayDivergence"]


class DeterministicRuntime:
    """Process-local source of randomness, ids, and recorded effects."""

    def __init__(self, seed: int, *, replay_log: bytes | None = None) -> None:
        self.seed = int(seed)
        self.replaying = replay_log is not None
        self.log = EventLog.from_bytes(replay_log) if replay_log is not None else EventLog()
        self._rng: np.random.Generator | None
        if self.replaying:
            self._rng = None
        else:
            self._rng = np.random.Generator(np.random.PCG64(self.seed))
        self._counter = 0
        self.hashes: list[str] = []

    def uniform(self) -> float:
        """Uniform draw in ``[0, 1)``. Replay returns the recorded draw."""
        if self.replaying:
            return float(self.log.pop("uniform", {}))
        rng = self._rng
        if rng is None:
            raise RuntimeError("recording runtime is missing its generator")
        value = float(rng.random())
        self.log.append("uniform", {}, value)
        return value

    def mint(self, kind: str) -> str:
        """Deterministic id. The factory signature matches ``SimulatedBroker``."""
        value = f"sim-{kind}-{self._counter:08d}"
        self._counter += 1
        recorded = self.effect("id", {"kind": kind}, value)
        return str(recorded)

    def effect(self, kind: str, request: Any, response: Any) -> Any:
        """Record ``response`` or, on replay, require it to match the trace."""
        if self.replaying:
            recorded = self.log.pop(kind, request)
            if _frozen_equal(recorded, response):
                return recorded
            raise ReplayDivergence(f"{kind} response diverged")
        self.log.append(kind, request, response)
        return response

    def commit_state(self, label: str, payload: dict[str, Any]) -> str:
        """Hash one intermediate state and require replay to reproduce it."""
        digest = hash_bytes(canonical_json_bytes(payload))
        if self.replaying:
            recorded = str(self.log.pop("state", {"label": label}))
            if recorded != digest:
                raise ReplayDivergence(f"state {label}: {digest} != {recorded}")
        else:
            self.log.append("state", {"label": label}, digest)
        self.hashes.append(digest)
        return digest

    def to_bytes(self) -> bytes:
        return self.log.to_bytes()


def _frozen_equal(recorded: Any, response: Any) -> bool:
    from quant_fund.simtest.eventlog import freeze

    return bool(freeze(recorded) == freeze(response))
