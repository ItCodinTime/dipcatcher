"""Property tests: asof() can NEVER return rows with known_at > t (§12 W1).

Stateful Hypothesis over arbitrary append/restate sequences, including
adversarial far-future known_at injections, checked against a pure-Python
bitemporal model. Plus a restatement-inertness property: any restatement
published after t leaves asof(t) byte-identical.
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, rule

from quant_fund.pit import PitVault, RestatementPolicy, VaultUnavailableError

T0 = datetime(2024, 1, 1, tzinfo=UTC)
SIDS = ("A", "B", "C")
MAX_EVENT_OFFSET = 5
MAX_KNOWN_OFFSET = 12


def _expected_latest(
    state: dict[tuple[str, int], list[tuple[int, float]]], t_offset: int
) -> dict[tuple[str, int], float]:
    """Pure-Python model: per key, value of greatest known_at <= t."""
    out: dict[tuple[str, int], float] = {}
    for key, versions in state.items():
        visible = [(ka, v) for ka, v in versions if ka <= t_offset]
        if visible:
            # Greatest known_at; ties resolve to the LATEST arrival, matching
            # the vault's stable sort + group_by(last).
            best = max(range(len(visible)), key=lambda i: (visible[i][0], i))
            out[key] = visible[best][1]
    return out


def _batch(
    sids: list[str], event_offsets: list[int], known_offsets: list[int], values: list[float]
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "security_id": sids,
            "event_time": [T0 + timedelta(days=o) for o in event_offsets],
            "known_at": [T0 + timedelta(days=o) for o in known_offsets],
            "close": values,
        }
    )


@settings(
    max_examples=15,
    derandomize=True,
    deadline=None,
    suppress_health_check=list(HealthCheck),
    stateful_step_count=20,
)
class PitVaultMachine(RuleBasedStateMachine):
    """Arbitrary append/restate/read sequences against the model (§12 W1)."""

    @initialize(data=st.data())
    def seed(self, data) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="pit-machine-")
        self.vault = PitVault(Path(self._tmp.name))
        self.vault.create_dataset("silver/bars")
        # (sid, event_offset) -> [(known_offset, value)] in arrival order
        self.state: dict[tuple[str, int], list[tuple[int, float]]] = {}
        n = data.draw(st.integers(min_value=1, max_value=4))
        self._ingest(data, n, allow_future=False)

    def teardown(self) -> None:
        self._tmp.cleanup()

    def _ingest(self, data, n: int, *, allow_future: bool) -> None:
        sids = [data.draw(st.sampled_from(SIDS)) for _ in range(n)]
        events = [data.draw(st.integers(min_value=0, max_value=MAX_EVENT_OFFSET)) for _ in range(n)]
        hi = 10_000 if allow_future else MAX_KNOWN_OFFSET
        knowns = [data.draw(st.integers(min_value=0, max_value=hi)) for _ in range(n)]
        values = [
            data.draw(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False)) for _ in range(n)
        ]
        self.vault.append("silver/bars", _batch(sids, events, knowns, values))
        for sid, ev, ka, val in zip(sids, events, knowns, values, strict=True):
            self.state.setdefault((sid, ev), []).append((ka, val))

    @rule(data=st.data())
    def append_batch(self, data) -> None:
        n = data.draw(st.integers(min_value=1, max_value=6))
        self._ingest(data, n, allow_future=True)  # adversarial far-future known_at

    @rule(data=st.data())
    def restate_batch(self, data) -> None:
        if not self.state:
            return
        n = data.draw(st.integers(min_value=1, max_value=4))
        keys = [data.draw(st.sampled_from(sorted(self.state))) for _ in range(n)]
        hi = data.draw(st.integers(min_value=0, max_value=10_000))
        values = [
            data.draw(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False)) for _ in range(n)
        ]
        frame = pl.DataFrame(
            {
                "security_id": [k[0] for k in keys],
                "event_time": [T0 + timedelta(days=k[1]) for k in keys],
                "close": values,
            }
        )
        self.vault.restate("silver/bars", frame, known_at=T0 + timedelta(days=hi))
        for (sid, ev), val in zip(keys, values, strict=True):
            self.state.setdefault((sid, ev), []).append((hi, val))

    @rule(
        data=st.data(),
        policy=st.sampled_from(list(RestatementPolicy)),
    )
    def asof_never_future(self, data, policy: RestatementPolicy) -> None:
        t_offset = data.draw(st.integers(min_value=0, max_value=MAX_KNOWN_OFFSET + 2))
        t = T0 + timedelta(days=t_offset)
        try:
            out = self.vault.asof("silver/bars", t, policy=policy)
        except VaultUnavailableError:
            # Fail-closed empty is only legal when NOTHING is visible at t.
            assert not _expected_latest(self.state, t_offset)
            return
        # THE INVARIANT: no returned row may carry known_at > t.
        assert out.rows == out.frame.height
        if out.rows:
            assert out.max_known_at <= t
            assert out.frame["known_at"].max() <= t
        # Cross-check against the model.
        got = {
            (sid, (et - T0).days): val
            for sid, et, val in zip(
                out.frame["security_id"].to_list(),
                out.frame["event_time"].to_list(),
                out.frame["close"].to_list(),
                strict=True,
            )
        }
        if policy is RestatementPolicy.LATEST_KNOWN:
            assert got == _expected_latest(self.state, t_offset)
        else:  # STRICT_FIRST: earliest known_at version per key
            expected = {}
            for key, versions in self.state.items():
                visible = [(ka, v) for ka, v in versions if ka <= t_offset]
                if visible:
                    # Smallest known_at; ties resolve to the FIRST arrival.
                    first = min(range(len(visible)), key=lambda i: (visible[i][0], i))
                    expected[key] = visible[first][1]
            assert got == expected


TestPitVaultStateful = PitVaultMachine.TestCase


@given(
    data=st.data(),
    t_offset=st.integers(min_value=0, max_value=MAX_KNOWN_OFFSET),
    future_offset=st.integers(min_value=MAX_KNOWN_OFFSET + 1, max_value=100_000),
)
@settings(max_examples=30, derandomize=True, deadline=None)
def test_adversarial_future_known_at_never_visible(data, t_offset: int, future_offset: int) -> None:
    """Inject rows with known_at FAR beyond t; asof(t) must not see them."""
    with tempfile.TemporaryDirectory(prefix="pit-adv-") as root:
        vault = PitVault(root)
        vault.create_dataset("silver/bars")
        n = data.draw(st.integers(min_value=1, max_value=6))
        sids = [data.draw(st.sampled_from(SIDS)) for _ in range(n)]
        events = [data.draw(st.integers(min_value=0, max_value=MAX_EVENT_OFFSET)) for _ in range(n)]
        knowns = [data.draw(st.integers(min_value=0, max_value=t_offset)) for _ in range(n)]
        values = [
            data.draw(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False)) for _ in range(n)
        ]
        vault.append("silver/bars", _batch(sids, events, knowns, values))
        # Adversarial injection: poison values published far in the future.
        poison = _batch(sids, events, [future_offset] * n, [-9e99] * n)
        vault.append("silver/bars", poison)

        t = T0 + timedelta(days=t_offset)
        out = vault.asof("silver/bars", t)
        assert out.frame["known_at"].max() <= t
        assert all(v != -9e99 for v in out.frame["close"].to_list())
        # STRICT_FIRST too: future restatements cannot poison either policy.
        strict = vault.asof("silver/bars", t, policy=RestatementPolicy.STRICT_FIRST)
        assert strict.frame["known_at"].max() <= t


@given(
    data=st.data(),
    t_offset=st.integers(min_value=0, max_value=MAX_KNOWN_OFFSET),
    publish_offset=st.integers(min_value=1, max_value=10_000),
)
@settings(max_examples=30, derandomize=True, deadline=None)
def test_restatement_inert_before_publication(data, t_offset: int, publish_offset: int) -> None:
    """asof(t) is byte-identical before/after a restatement published > t."""
    with tempfile.TemporaryDirectory(prefix="pit-inert-") as root:
        vault = PitVault(root)
        vault.create_dataset("silver/bars")
        n = data.draw(st.integers(min_value=1, max_value=6))
        sids = [data.draw(st.sampled_from(SIDS)) for _ in range(n)]
        events = [data.draw(st.integers(min_value=0, max_value=MAX_EVENT_OFFSET)) for _ in range(n)]
        knowns = [data.draw(st.integers(min_value=0, max_value=t_offset)) for _ in range(n)]
        values = [
            data.draw(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False)) for _ in range(n)
        ]
        vault.append("silver/bars", _batch(sids, events, knowns, values))

        t = T0 + timedelta(days=t_offset)
        before = vault.asof("silver/bars", t)
        publish = T0 + timedelta(days=t_offset + publish_offset)  # strictly > t
        vault.restate(
            "silver/bars",
            pl.DataFrame(
                {
                    "security_id": list(sids),
                    "event_time": [T0 + timedelta(days=e) for e in events],
                    "close": [-1.0] * n,
                }
            ),
            known_at=publish,
        )
        after = vault.asof("silver/bars", t)
        assert after.content_sha256 == before.content_sha256
        assert after.frame.equals(before.frame)


def test_model_sanity() -> None:
    """The pure-Python model itself: max picks greatest known_at <= t."""
    state = {("A", 0): [(0, 1.0), (5, 2.0), (9, 3.0)]}
    assert _expected_latest(state, 4) == {("A", 0): 1.0}
    assert _expected_latest(state, 5) == {("A", 0): 2.0}
    assert _expected_latest(state, 100) == {("A", 0): 3.0}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
