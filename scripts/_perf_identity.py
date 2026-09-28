"""P4.1/P6.8 byte-identity + before/after timing for the event-loop sweep.

Loads the pre-change ``engine.py`` straight from a git ref (default
``origin/main`` — the same frozen-module idiom as ``_bench_before_after.py``)
and replays a battery of SYNTHETIC workloads through the old and new
``_run_backtest_event_loop``. For each workload it asserts identical Arrow
IPC bytes for ``equity`` and ``fills`` and an identical metrics digest —
the methodology of ``tests/property/test_fast_replay_byte_identity.py``.

Usage:
    uv run python scripts/_perf_identity.py [--base-ref REF] [--reps N]
"""

import argparse
import hashlib
import importlib.util
import math
import struct
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts._perf_sweep import INIT_NAV, build_workload, make_cfg  # noqa: E402

from quant_fund.config.models import (  # noqa: E402
    AppConfig,
    CostConfig,
    ExecutionConfig,
    FillConvention,
    KillSwitchConfig,
    RiskGateConfig,
)

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _load_frozen_engine(ref: str):
    src = subprocess.run(
        ["git", "show", f"{ref}:src/quant_fund/backtest/engine.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    tmp = Path(tempfile.mkdtemp(prefix="engine_before_")) / "_engine_before.py"
    tmp.write_text(src)
    spec = importlib.util.spec_from_file_location("_engine_before", tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_engine_before"] = mod
    spec.loader.exec_module(mod)
    return mod


def _value_bytes(v: object) -> bytes:
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
    """Canonicalize metrics whose values are not bit-stable.

    ``aggregate_shortfall`` computes ``by_side``/``top_cost_names`` via polars
    ``group_by`` float sums, which are nondeterministic at the ulp level even
    on an identical input frame (parallel partitioning order). Those lists are
    replaced by sorted ``(key, n_fills)`` membership pairs — integer counts are
    order-stable, so a genuine divergence in group membership still fails.
    """
    out = dict(metrics)
    for key in ("garch_risk_overlay_dates", "realized_garch_risk_overlay_dates"):
        out.pop(key, None)
    shortfall = out.get("implementation_shortfall")
    if isinstance(shortfall, dict):
        shortfall = dict(shortfall)
        for list_key, key_field in (("top_cost_names", "security_id"), ("by_side", "side_sign")):
            rows = shortfall.pop(list_key, None)
            if isinstance(rows, list):
                shortfall[list_key + "_members"] = sorted(
                    (str(r.get(key_field)), r.get("n_fills")) for r in rows
                )
        out["implementation_shortfall"] = shortfall
    return out


def _frame_ipc_bytes(df: pl.DataFrame) -> bytes:
    buf = df.write_ipc(None)
    return buf.getvalue() if hasattr(buf, "getvalue") else bytes(buf)


def _assert_byte_identical(ref, new, label: str) -> None:
    assert ref.equity.schema == new.equity.schema, label
    assert _frame_ipc_bytes(ref.equity) == _frame_ipc_bytes(new.equity), label
    assert ref.fills.schema == new.fills.schema, label
    assert _frame_ipc_bytes(ref.fills) == _frame_ipc_bytes(new.fills), label
    assert _metrics_bytes(_normalized_metrics(ref.metrics)) == _metrics_bytes(
        _normalized_metrics(new.metrics)
    ), label
    assert ref.frictionless == new.frictionless, label
    assert ref.source_note == new.source_note, label
    print(f"  {label}: equity/fills/metrics byte-identical")


def _workload_sparse_delist() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Missing opens, null closes/CTR, a delisted name, sparse weights."""
    rng = np.random.default_rng(7)
    n_days = 400
    sids = [f"S{k:02d}" for k in range(11)]
    rows = []
    for k, sid in enumerate(sids):
        px = 40.0 + 60.0 * float(rng.random())
        for i in range(n_days):
            if rng.random() < 0.10 and i > 2:
                continue
            if sid == sids[0] and i > n_days // 2:
                continue  # delisted mid-panel
            noise = float(rng.normal(0, 0.02))
            close = px * (1 + noise)
            rows.append(
                {
                    "security_id": sid,
                    "event_time": T0 + timedelta(days=i),
                    "open": None if rng.random() < 0.05 else px * (1 + noise / 3),
                    "close": None if rng.random() < 0.03 else close,
                    "close_total_return": None if rng.random() < 0.06 else close,
                    "volume": 1_000_000.0 + 5_000 * i,
                    "adv": close * (1_000_000.0 + 5_000 * i),
                    "vol_20": 0.02 + 0.0005 * k,
                    "source": "synthetic",
                }
            )
            px = close
    bars = pl.DataFrame(rows)
    w_rows = []
    for i in range(0, n_days, 3):
        for sid in sids:
            if rng.random() < 0.25:
                continue
            w_rows.append(
                {
                    "event_time": T0 + timedelta(days=i),
                    "security_id": sid,
                    "target_weight": float(rng.uniform(-0.15, 0.12)),
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
    return bars, weights


def _workload_int_close() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Integer-typed price column exercises the scalar ``_valid_price`` path."""
    bars = pl.DataFrame(
        {
            "security_id": ["A", "A", "A", "B", "B", "B"],
            "event_time": [T0 + timedelta(days=i) for i in (0, 1, 2)] * 2,
            "open": [10, 11, 12, 5, 5, 6],
            "close": [10, 11, 12, 5, 5, 6],
            "close_total_return": [10.0, 11.0, 12.0, 5.0, 5.0, 6.0],
            "volume": [1e6] * 6,
            "adv": [1e9] * 6,
            "vol_20": [0.02] * 6,
            "source": ["synthetic"] * 6,
        }
    ).with_columns(pl.col("open").cast(pl.Int64), pl.col("close").cast(pl.Int64))
    weights = pl.DataFrame(
        {
            "event_time": [T0, T0 + timedelta(days=1)],
            "security_id": ["A", "B"],
            "target_weight": [0.4, -0.1],
        }
    )
    return bars, weights


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ref", default="origin/main")
    ap.add_argument("--reps", type=int, default=7)
    args = ap.parse_args()

    before = _load_frozen_engine(args.base_ref)
    from quant_fund.backtest.engine import _run_backtest_event_loop

    bars, weights = build_workload()
    cfg = make_cfg()

    cases: list[tuple[str, pl.DataFrame, pl.DataFrame, AppConfig]] = []
    cases.append(("sma11x1000", bars, weights, cfg))

    sparse_bars, sparse_weights = _workload_sparse_delist()
    cfg_sparse = cfg.model_copy(deep=True)
    cfg_sparse.risk_gate.stale_price_bars = 1_000_000
    cases.append(("sparse_delist", sparse_bars, sparse_weights, cfg_sparse))
    cfg_auction = cfg.model_copy(deep=True)
    cfg_auction.execution = ExecutionConfig(
        fill=FillConvention.CLOSE_AUCTION, allow_close_auction=False
    )
    cases.append(("close_auction_fill", bars, weights, cfg_auction))
    cfg_halt = cfg.model_copy(deep=True)
    cfg_halt.kill_switch = KillSwitchConfig(state="HALT_NEW_ORDERS")
    cases.append(("kill_halt", bars, weights, cfg_halt))
    cfg_free = cfg.model_copy(deep=True)
    cfg_free.costs = CostConfig(frictionless=True)
    cases.append(("frictionless", bars, weights, cfg_free))
    cfg_tight = cfg.model_copy(deep=True)
    cfg_tight.risk_gate = RiskGateConfig(
        max_order_notional=2e4,
        max_gross=0.5,
        max_net=0.4,
        max_name=0.05,
        max_participation=0.02,
        max_predicted_vol=0.05,
        stale_price_bars=3,
        stale_model_hours=1e9,
    )
    cases.append(("tight_gate_rejects", sparse_bars, sparse_weights, cfg_tight))
    int_bars, int_weights = _workload_int_close()
    cases.append(("int_price_fallback", int_bars, int_weights, cfg))

    for label, b, w, c in cases:
        try:
            ref = before._run_backtest_event_loop(b, w, c, initial_nav=INIT_NAV)
        except Exception as e:  # noqa: BLE001 - ref behavior must reproduce
            try:
                _run_backtest_event_loop(b, w, c, initial_nav=INIT_NAV)
            except Exception as e2:  # noqa: BLE001
                # Frozen module defines its own exception classes — compare
                # name and payload, not identity.
                assert type(e2).__name__ == type(e).__name__, (label, e, e2)
                assert e2.args == e.args, (label, e, e2)
                print(f"  {label}: identical exception {type(e).__name__}")
                continue
            raise AssertionError(f"{label}: ref raised {e!r} but new returned") from e
        new = _run_backtest_event_loop(b, w, c, initial_nav=INIT_NAV)
        _assert_byte_identical(ref, new, label)

    # Before/after timings on the headline workload.
    def _med(fn) -> float:
        fn(bars, weights, cfg, initial_nav=INIT_NAV)
        ts = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            fn(bars, weights, cfg, initial_nav=INIT_NAV)
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts)) * 1e3

    t_before = _med(before._run_backtest_event_loop)
    t_after = _med(_run_backtest_event_loop)
    print(f"event_loop before: {t_before:.1f} ms median")
    print(f"event_loop after:  {t_after:.1f} ms median")
    print(f"speedup: {t_before / t_after:.2f}x")
    digest = hashlib.sha256(
        _frame_ipc_bytes(_run_backtest_event_loop(bars, weights, cfg, initial_nav=INIT_NAV).equity)
    ).hexdigest()
    print(f"sweep workload equity sha256 (both engines): {digest}")


if __name__ == "__main__":
    main()
