"""P4.2 — ``run_backtest_fast`` must emit *byte*-identical output.

The vectorized replay exists for incumbent-benchmark latency; using it in a
receipt is only honest if it is the same engine. These properties assert the
``equity`` and ``fills`` frames serialize to identical Arrow IPC bytes — not
merely tolerance-equal floats — and that metrics match exactly, over
generated workloads spanning cost surfaces, risk gates, fill conventions,
shorts, missing bars, sparse rebalance grids, delisted names, and ghost
weight rows. The interpreted fallback is asserted byte-identical to the
numba kernel too, so the claim holds with or without the accelerator.
"""

from __future__ import annotations

import hashlib
import math
import struct
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from quant_fund.backtest import fast_replay
from quant_fund.backtest.engine import _run_backtest_event_loop, run_backtest
from quant_fund.backtest.fast_replay import run_backtest_fast
from quant_fund.config.models import (
    AppConfig,
    CostConfig,
    ExecutionConfig,
    FillConvention,
    KillSwitchConfig,
    RiskGateConfig,
)
from tests.property._profiles import adversarial_settings

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _gen_workload(seed: int) -> tuple[pl.DataFrame, pl.DataFrame, AppConfig]:
    """Seeded matched-class workload: target-percent panel, bars, config."""
    rng = np.random.default_rng(seed)
    n_assets = int(rng.integers(1, 9))
    n_days = int(rng.integers(10, 120))
    sids = [f"S{k:02d}" for k in range(n_assets)]
    missing = float(rng.uniform(0.0, 0.15))
    delist = bool(rng.random() < 0.15) and n_days > 30
    bar_rows = []
    for k, sid in enumerate(sids):
        px = 50.0 + 100.0 * float(rng.random())
        for i in range(n_days):
            if rng.random() < missing and i > 2:
                continue
            if delist and sid == sids[0] and i > n_days // 2:
                continue  # name stops printing entirely — staleness/fail-closed lane
            drift = 0.0004 * ((i + 3 * k) % 7 - 3)
            noise = float(rng.normal(0, 0.02))
            close = px * (1 + drift + noise)
            bar_rows.append(
                {
                    "security_id": sid,
                    "event_time": T0 + timedelta(days=i),
                    "open": None if rng.random() < 0.03 else px * (1 + noise / 2),
                    "close": close if rng.random() > 0.02 else None,
                    "close_total_return": close if rng.random() > 0.05 else None,
                    "volume": 1_000_000.0 + 10_000 * i,
                    "adv": close * (1_000_000.0 + 10_000 * i),
                    "vol_20": 0.02 + 0.001 * k,
                    "source": "file",
                }
            )
            px = close
    bars = pl.DataFrame(
        bar_rows,
        schema={
            "security_id": pl.String,
            "event_time": pl.Datetime("us", "UTC"),
            "open": pl.Float64,
            "close": pl.Float64,
            "close_total_return": pl.Float64,
            "volume": pl.Float64,
            "adv": pl.Float64,
            "vol_20": pl.Float64,
            "source": pl.String,
        },
    )

    lo = float(rng.uniform(-0.5, 0.0))
    hi = float(rng.uniform(0.5, 1.5))
    sparse = float(rng.uniform(0.0, 0.4))
    w_rows = []
    for i in range(n_days):
        for sid in sids:
            if rng.random() < sparse:
                continue
            w_rows.append(
                {
                    "event_time": T0 + timedelta(days=i),
                    "security_id": sid,
                    "target_weight": float(rng.uniform(lo, hi)),
                }
            )
    if rng.random() < 0.2:
        # Weight rows on a name with no bars — tradable nowhere; both engines
        # must ignore it identically.
        for i in range(0, n_days, 5):
            w_rows.append(
                {
                    "event_time": T0 + timedelta(days=i),
                    "security_id": "GHOST",
                    "target_weight": 0.05,
                }
            )
    weights = pl.DataFrame(
        w_rows,
        schema={
            "event_time": pl.Datetime("us", "UTC"),
            "security_id": pl.String,
            "target_weight": pl.Float64,
        },
    )

    cfg = AppConfig(
        costs=CostConfig(
            commission_bps=float(rng.uniform(0, 20)),
            half_spread_bps=float(rng.uniform(0, 10)),
            impact_y=float(rng.uniform(0, 0.5)),
            bps_per_turnover=float(rng.uniform(0, 5)),
            borrow_bps_per_year=float(rng.uniform(0, 300)),
            frictionless=bool(rng.random() < 0.15),
            participation_limit=float(rng.uniform(0.05, 1.0)),
        ),
        risk_gate=RiskGateConfig(
            max_order_notional=float(rng.choice([1e5, 1e6, 1e9, 1e12])),
            max_gross=float(rng.choice([0.5, 1.0, 2.0, 100.0])),
            max_net=float(rng.choice([0.1, 0.5, 1.0, 100.0])),
            max_name=float(rng.choice([0.05, 0.2, 0.5, 1.0])),
            max_participation=float(rng.uniform(0.05, 1.0)),
            max_predicted_vol=float(rng.choice([0.01, 0.05, 0.4, 100.0])),
            stale_price_bars=int(rng.integers(0, 7)),
            stale_model_hours=1e9,
        ),
        execution=ExecutionConfig(
            fill=(FillConvention.NEXT_OPEN if rng.random() < 0.5 else FillConvention.CLOSE_AUCTION),
            allow_close_auction=False,
        ),
        kill_switch=KillSwitchConfig(state="ENABLED" if rng.random() < 0.8 else "HALT_NEW_ORDERS"),
    )
    return bars, weights, cfg


def _value_bytes(v: object) -> bytes:
    """Canonical byte encoding: floats by IEEE-754 bits, NaN canonical."""
    if isinstance(v, bool):
        return b"b" + bytes([v])
    if isinstance(v, int):
        return b"i" + str(v).encode()
    if isinstance(v, float):
        return b"nan" if math.isnan(v) else b"f" + struct.pack(">d", v)
    if isinstance(v, str):
        return b"s" + v.encode()
    if isinstance(v, dict):
        return b"d" + _metrics_bytes(v)  # type: ignore[arg-type]
    if isinstance(v, (list, tuple)):
        return b"l" + b"".join(_value_bytes(x) for x in v)
    if v is None:
        return b"n"
    return b"?" + repr(v).encode()


def _metrics_bytes(metrics: dict) -> bytes:
    h = hashlib.sha256()
    for k in sorted(metrics):
        h.update(k.encode())
        h.update(b"\x00")
        h.update(_value_bytes(metrics[k]))
    return h.digest()


def _normalized_metrics(metrics: dict) -> dict:
    """Drop the overlay date counters the dispatcher stamps post-hoc, and
    canonicalize group-aggregate list order in ``implementation_shortfall``.

    The event loop counts GARCH overlay engagements inline; the fast path
    never runs when an overlay artifact exists, so any present counter must
    be 0 — assert that, then exclude them from the byte comparison.

    ``aggregate_shortfall`` orders ``top_cost_names``/``by_side`` by a
    ``group_by`` + ``sort`` on the aggregated value alone; exact ties keep
    polars' hash-partition order, which is nondeterministic across process
    executions — a shared-tail artifact outside the fast path's control.
    Element order in those lists is therefore canonicalized (by content
    bytes) before hashing; element *content* still must match byte-exactly.
    """
    out = dict(metrics)
    for key in ("garch_risk_overlay_dates", "realized_garch_risk_overlay_dates"):
        if key in out:
            assert out[key] == 0, f"{key}={out[key]} implies an overlay artifact was engaged"
            del out[key]
    shortfall = out.get("implementation_shortfall")
    if isinstance(shortfall, dict):
        shortfall = dict(shortfall)
        for list_key in ("top_cost_names", "by_side"):
            rows = shortfall.get(list_key)
            if isinstance(rows, list):
                shortfall[list_key] = sorted(rows, key=_value_bytes)
        out["implementation_shortfall"] = shortfall
    return out


def _frame_ipc_bytes(df: pl.DataFrame) -> bytes:
    """Serialize to Arrow IPC bytes — catches dtype/order gaps array-equality misses."""
    buf = df.write_ipc(None)
    return buf.getvalue() if hasattr(buf, "getvalue") else bytes(buf)


def _assert_byte_identical(ref, fast) -> None:
    assert ref.equity.schema == fast.equity.schema
    assert _frame_ipc_bytes(ref.equity) == _frame_ipc_bytes(fast.equity)
    assert ref.fills.schema == fast.fills.schema
    assert _frame_ipc_bytes(ref.fills) == _frame_ipc_bytes(fast.fills)
    assert _metrics_bytes(_normalized_metrics(ref.metrics)) == _metrics_bytes(
        _normalized_metrics(fast.metrics)
    )
    assert ref.frictionless == fast.frictionless
    assert ref.source_note == fast.source_note


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@adversarial_settings()
def test_fast_replay_byte_identical_on_generated_workloads(seed: int) -> None:
    bars, weights, cfg = _gen_workload(seed)
    try:
        ref = _run_backtest_event_loop(bars, weights, cfg)
    except Exception as e:  # noqa: BLE001 - whatever ref does, fast must do
        with pytest.raises(type(e)):
            run_backtest_fast(bars, weights, cfg)
        return
    fast = run_backtest_fast(bars, weights, cfg)
    _assert_byte_identical(ref, fast)
    # The public entry point auto-dispatches to the same bytes.
    _assert_byte_identical(ref, run_backtest(bars, weights, cfg))


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@adversarial_settings()
def test_interpreted_fallback_byte_identical(seed: int) -> None:
    """Without numba the interpreted loop must produce the same bytes."""
    original = fast_replay.HAVE_NUMBA
    fast_replay.HAVE_NUMBA = False
    try:
        bars, weights, cfg = _gen_workload(seed)
        try:
            ref = _run_backtest_event_loop(bars, weights, cfg)
        except Exception as e:  # noqa: BLE001
            with pytest.raises(type(e)):
                run_backtest_fast(bars, weights, cfg)
            return
        _assert_byte_identical(ref, run_backtest_fast(bars, weights, cfg))
    finally:
        fast_replay.HAVE_NUMBA = original


def test_fast_flag_true_refuses_unsupported_workloads() -> None:
    """fast=True must fail closed, not degrade to the event loop."""
    bars, weights, cfg = _gen_workload(4)
    mixed = bars.with_columns(pl.col("event_time").cast(pl.Datetime("ns", "UTC")))
    with pytest.raises(ValueError, match="fast replay"):
        run_backtest(mixed, weights, cfg, fast=True)
    dup = pl.concat([bars, bars.head(1)])
    with pytest.raises(ValueError, match="fast replay"):
        run_backtest(dup, weights, cfg, fast=True)
    empty = pl.DataFrame(schema=bars.schema)
    with pytest.raises(ValueError, match="fast replay"):
        run_backtest(empty, weights, cfg, fast=True)
    cfg_auction = cfg.model_copy(deep=True)
    cfg_auction.execution.allow_close_auction = True
    with pytest.raises(ValueError, match="allow_close_auction"):
        run_backtest(bars, weights, cfg_auction, fast=True)


def test_fast_flag_false_and_auto_paths() -> None:
    bars, weights, cfg = _gen_workload(9)
    ref = _run_backtest_event_loop(bars, weights, cfg)
    _assert_byte_identical(ref, run_backtest(bars, weights, cfg, fast=False))
    _assert_byte_identical(ref, run_backtest(bars, weights, cfg, fast=True))
