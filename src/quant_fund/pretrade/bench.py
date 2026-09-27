"""Honest hot-path latency distribution for one pre-trade check-set.

The timed region is ``PretradeEngine.check`` on a reused order view after
warmup. GC is disabled during the sample loop. The gate uses the median
trial so a single scheduling spike does not define the result, and every
trial is reported. A sample that is not an allow fails the run: the gate
measures the full allow path, not an early deny.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from quant_fund.pretrade.codes import KIND_ORDER, reason_names
from quant_fund.pretrade.config import LimitConfig, PretradeConfig, SessionConfig
from quant_fund.pretrade.engine import OrderView, PretradeEngine

P50_LIMIT_NS = 5_000
P99_LIMIT_NS = 20_000

_HMAC = b"pretrade-bench-hmac-key!!"


def _pin_affinity() -> str:
    if not hasattr(os, "sched_setaffinity"):
        return "unavailable"
    try:
        os.sched_setaffinity(0, {0})
    except OSError:
        return "unavailable"
    return "0"


def benchmark_config() -> PretradeConfig:
    """Wide limits so the measured order is allowed. Every check still runs."""
    return PretradeConfig(
        schema_version=1,
        limits=LimitConfig(
            max_order_notional=1e12,
            max_order_quantity=1e12,
            price_collar_bps=50.0,
            max_position_notional=1e12,
            max_position_quantity=1e12,
            max_gross_notional=1e12,
            max_net_notional=1e12,
            max_name_concentration=1.0,
            max_daily_loss_fraction=0.5,
            max_trailing_drawdown_fraction=0.5,
            max_orders_per_window=200,
            max_messages_per_window=200,
            rate_window_ns=50_000,
            duplicate_window_ns=50_000,
            max_reference_age_ns=86_400_000_000_000,
            max_mark_age_ns=86_400_000_000_000,
            pdt_equity_threshold=25_000.0,
            pdt_window_sessions=5,
            pdt_max_day_trades=3,
            settlement_ns=86_400_000_000_000,
            qty_tick=1e-4,
            price_tick=1e-4,
            cash_account=True,
            restrict_to_settled_cash=True,
            ring_capacity=256,
            symbol_capacity=16,
        ),
        session=SessionConfig(
            timezone="America/New_York",
            open_minute=9 * 60 + 30,
            close_minute=16 * 60,
        ),
    )


def _session_open_ns() -> int:
    when = datetime(2024, 1, 3, 15, 0, tzinfo=ZoneInfo("America/New_York"))
    return int(when.timestamp() * 1_000_000_000)


def _percentile(sorted_samples: list[int], p: float) -> int:
    if not sorted_samples:
        raise ValueError("no samples")
    index = int(round(p * (len(sorted_samples) - 1)))
    if index < 0:
        index = 0
    if index >= len(sorted_samples):
        index = len(sorted_samples) - 1
    return sorted_samples[index]


def _summarize(samples: list[int]) -> dict[str, float | int]:
    ordered = sorted(samples)
    n = len(ordered)
    return {
        "n": n,
        "min_ns": ordered[0],
        "p50_ns": _percentile(ordered, 0.50),
        "p90_ns": _percentile(ordered, 0.90),
        "p95_ns": _percentile(ordered, 0.95),
        "p99_ns": _percentile(ordered, 0.99),
        "p999_ns": _percentile(ordered, 0.999),
        "max_ns": ordered[-1],
        "mean_ns": sum(ordered) / n,
    }


def _prepare(engine: PretradeEngine, ts0: int) -> OrderView:
    sid = engine.ensure_symbol("BENCH")
    engine.set_symbol(sid, pos=0.0, ref_px=100.0, ref_ts_ns=ts0, locate_ok=True, bid=100.0)
    engine.update_account(
        nav=1_000_000.0,
        mark_ts_ns=ts0,
        session_start_nav=1_000_000.0,
        peak_nav=1_000_000.0,
    )
    engine.book.settled = 1e12
    return OrderView(
        symbol_id=sid,
        side=1,
        qty=10.0,
        px=100.0,
        ts_ns=ts0,
        session_id=1,
        is_limit=1,
        kind=KIND_ORDER,
    )


def _run_trial(engine: PretradeEngine, view: OrderView, samples: int, step_ns: int) -> list[int]:
    out = [0] * samples
    ts = view.ts_ns
    qty_tick = engine.book.qty_tick
    gc.disable()
    try:
        i = 0
        while i < samples:
            view.qty = 10.0 + (i + 1) * qty_tick
            view.ts_ns = ts
            t0 = time.perf_counter_ns()
            bits = engine.check(view, apply=False)
            out[i] = time.perf_counter_ns() - t0
            if bits != 0:
                raise RuntimeError(
                    f"benchmark expected an allow, got {reason_names(bits)} bits={bits}"
                )
            ts += step_ns
            i += 1
    finally:
        gc.enable()
    view.ts_ns = ts
    return out


def run_benchmark(
    *,
    trials: int = 5,
    samples: int = 8000,
    warmup: int = 2000,
    gate: bool = False,
) -> dict[str, Any]:
    if trials < 1 or samples < 2:
        raise ValueError("trials and samples must be positive")
    affinity = _pin_affinity()
    ts0 = _session_open_ns()
    engine = PretradeEngine(
        benchmark_config(),
        hmac_key=_HMAC,
        initial_nav=1_000_000.0,
        initial_cash=1_000_000.0,
        ts_ns=ts0,
    )
    view = _prepare(engine, ts0)
    _run_trial(engine, view, warmup, 1_000)
    trial_rows: list[dict[str, float | int]] = []
    for _ in range(trials):
        trial_rows.append(_summarize(_run_trial(engine, view, samples, 1_000)))
    order = sorted(range(trials), key=lambda i: (trial_rows[i]["p99_ns"], trial_rows[i]["p50_ns"]))
    median_i = order[len(order) // 2]
    chosen = trial_rows[median_i]
    p50 = int(chosen["p50_ns"])
    p99 = int(chosen["p99_ns"])
    passed = p50 < P50_LIMIT_NS and p99 < P99_LIMIT_NS
    report: dict[str, Any] = {
        "schema": 1,
        "implementation": "cpython-slots",
        "numba": False,
        "rust": False,
        "clock": "time.perf_counter_ns",
        "gc_disabled_during_samples": True,
        "affinity": affinity,
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "path": "allow",
        "apply": False,
        "samples_per_trial": samples,
        "warmup": warmup,
        "trials": trial_rows,
        "median_trial_index": median_i,
        "p50_ns": p50,
        "p90_ns": chosen["p90_ns"],
        "p95_ns": chosen["p95_ns"],
        "p99_ns": p99,
        "p999_ns": chosen["p999_ns"],
        "min_ns": chosen["min_ns"],
        "max_ns": chosen["max_ns"],
        "mean_ns": chosen["mean_ns"],
        "gate_p50_ns": P50_LIMIT_NS,
        "gate_p99_ns": P99_LIMIT_NS,
        "gate_requested": gate,
        "gate_pass": passed,
        "percentile": "nearest-rank round(p * (n - 1))",
        "gate_rule": "median trial by p99 must have p50 < 5000ns and p99 < 20000ns",
        "checks": [
            "kill_switch",
            "stale_reference_and_mark",
            "non_finite",
            "market_hours",
            "halt",
            "max_order_quantity",
            "max_order_notional",
            "price_collar",
            "position_quantity",
            "position_notional",
            "gross_exposure",
            "net_exposure",
            "concentration",
            "duplicate_order",
            "order_rate",
            "message_rate",
            "reg_sho",
            "pattern_day_trader",
            "good_faith_violation",
            "buying_power",
            "daily_loss",
            "trailing_drawdown",
        ],
        "scenario": {
            "side": "buy",
            "is_limit": True,
            "nav": 1_000_000.0,
            "pdt_binding": False,
            "rate_window_ns": 50_000,
            "step_ns": 1_000,
            "note": (
                "Qty changes by one tick per sample so fingerprints stay unique "
                "inside the duplicate window. Rate and duplicate rings hold about "
                "window/step live entries and expire from the tail each call."
            ),
        },
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pre-trade hot-path latency distribution")
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--samples", type=int, default=8000)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args(argv)
    report = run_benchmark(
        trials=args.trials,
        samples=args.samples,
        warmup=args.warmup,
        gate=args.gate,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.write("\n")
    if args.gate and not report["gate_pass"]:
        print(
            "pretrade latency gate failed: "
            f"p50={report['p50_ns']}ns (limit {P50_LIMIT_NS}), "
            f"p99={report['p99_ns']}ns (limit {P99_LIMIT_NS})",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
