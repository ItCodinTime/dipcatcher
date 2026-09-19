#!/usr/bin/env python3
"""Deterministic synthetic paper-loop benchmark; never certifies live performance or SOTA.

The report measures this repository's simulated ledger loop and durable artifact
publication under a fixed synthetic workload. It deliberately reports incumbent
package availability separately; it does not claim a head-to-head comparison.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from quant_fund.config.loader import load_config
from quant_fund.paper.ledger import load_broker_state, validate_ledger_schema
from quant_fund.paper.loop import run_paper_loop

INCUMBENT_URLS = {
    "qlib": "https://github.com/microsoft/qlib",
    "vectorbt": "https://vectorbt.dev/",
    "zipline-reloaded": "https://zipline.ml4trading.io/",
}


def _bars(n_days: int, n_assets: int) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    start = datetime(2024, 1, 1, tzinfo=UTC)
    for day in range(n_days):
        event_time = start + timedelta(days=day)
        for asset in range(n_assets):
            security_id = f"S{asset:03d}"
            close = 100.0 + asset + 0.05 * day + 0.1 * ((day + asset) % 7)
            rows.append(
                {
                    "security_id": security_id,
                    "event_time": event_time,
                    "open": close * (1.0 + 0.0005),
                    "close": close,
                    "close_total_return": close,
                    "volume": 1_000_000.0,
                    "adv": 100_000_000.0,
                    "vol_20": 0.02,
                    "source": "synthetic",
                }
            )
    return pl.DataFrame(rows)


def _weights(n_days: int, n_assets: int) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    start = datetime(2024, 1, 1, tzinfo=UTC)
    for day in range(n_days):
        event_time = start + timedelta(days=day)
        for asset in range(n_assets):
            rows.append(
                {
                    "event_time": event_time,
                    "security_id": f"S{asset:03d}",
                    "target_weight": (0.02 if asset % 2 == 0 else -0.015),
                }
            )
    return pl.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--assets", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 2 or args.assets < 1:
        raise ValueError("--steps must be >= 2 and --assets must be >= 1")
    report_path = args.output.resolve()
    if report_path.exists():
        raise ValueError(f"refusing to overwrite {report_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    bars = _bars(args.steps + 2, args.assets)
    weights = _weights(args.steps + 2, args.assets)
    data_root = report_path.parent / "paper_data"
    config = load_config("configs/paper.yaml")
    config.data.root = data_root
    config.costs.frictionless = True
    config.paper.promote_min_steps = args.steps + 1
    run_id = "benchmark-paper-loop"

    started = time.perf_counter()
    result = run_paper_loop(
        bars,
        config,
        champion_weights=weights,
        shadow_weights=weights,
        initial_nav=1_000_000.0,
        max_steps=args.steps,
        run_id=run_id,
        prefer_latest=False,
    )
    elapsed = time.perf_counter() - started

    root = data_root / "metadata" / "paper" / run_id
    artifact_rows: dict[str, int] = {}
    for name in ("orders", "equity", "shadow_equity", "positions", "cash_ledger"):
        path = root / f"{name}.parquet"
        artifact_rows[name] = int(pl.read_parquet(path).height) if path.is_file() else 0
    state = load_broker_state(data_root, run_id)
    schema = validate_ledger_schema(root)
    incumbent_availability = {
        name: importlib.util.find_spec(name.replace("-", "_")) is not None
        for name in ("qlib", "vectorbt", "zipline")
    }
    report = {
        "protocol": {
            "source": "synthetic",
            "seed": 0,
            "steps": args.steps,
            "assets": args.assets,
            "fill": "next_open",
            "capital": "simulated_only",
            "live_pnl_claim": False,
            "research_only": True,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "polars": pl.__version__,
        },
        "elapsed_seconds": elapsed,
        "steps_completed": int(result.metrics["n_steps"]),
        "steps_per_second": float(result.metrics["n_steps"] / elapsed),
        "artifact_rows": artifact_rows,
        "broker_state_step": None if state is None else int(state.get("step", -1)),
        "ledger_schema_ok": bool(schema["ok"]),
        "ledger_schema_errors": schema["errors"],
        "incumbent_package_availability": incumbent_availability,
        "incumbent_reference_urls": INCUMBENT_URLS,
        "limitations": [
            "synthetic bars and deterministic weights",
            "simulated fills and no broker connection",
            "no matched incumbent workload was executed",
            "timing is a local lab measurement, not production latency",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
