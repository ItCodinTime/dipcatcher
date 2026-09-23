"""Zipline Reloaded matched-workload ATTEMPT — outcome: NOT A FAIR COMPARISON.

Status: attempted and abandoned on evidence, 2026-09-22. The receipts under
``.dsh-24x7/incumbent-zipline/`` record the attempt, not a valid benchmark.

Concrete fairness blockers found (all captured in this run):

1. **Environment conflict.** zipline-reloaded 3.1.1 requires ``pandas<3``
   while the project pins ``pandas>=3`` — a same-interpreter comparison is
   impossible by construction (this script runs in an isolated venv).
2. **Fill-price semantics can't be matched.** A custom slippage model
   returning ``data.current(asset, "open")`` produced fill prices that match
   no bar's open (recorded fill 43428.86 vs opens 42066.94/42563.76/42140.29
   on d/d-1/d+1) — daily-mode ``data.current`` inside ``process_order`` does
   not honor the ``open`` field as documented. NAV then diverges 46x from
   two independently-correct engines (dipcatcher + vectorbt + numpy replica
   all agree at 1,500,764.6563891 to ~1e-9).
3. **Integer-share rounding.** Zipline equities round order amounts to whole
   shares, silently dropping small crypto-quantity deltas — a fractional-quantity
   market can't be expressed without rewriting its blotter.
4. **US-equities asset model.** Auto-delisting at the bundle end date forces
   liquidation; 24/7 sessions need a custom calendar path.

Verdict: zipline cannot express the shared workload semantics (fractional
quantities, open t+1 fills, 24/7 calendar) without modifying the incumbent
itself — so no fair benchmark exists here. Kept for the record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from incumbent_bench_vectorbt import (  # noqa: E402
    COMMISSION_BPS,
    INIT_NAV,
    SMA_WINDOW,
    _sha256,
    build_weight_panel,
    load_panel,
)


def _write_csvdir(frames: dict[str, pl.DataFrame], dest: Path) -> Path:
    """csvdir layout: <dest>/daily/<SYMBOL>.csv with OHLCV + dividend/split."""
    daily = dest / "daily"
    daily.mkdir(parents=True, exist_ok=True)
    for sid, fr in frames.items():
        df = fr.select(
            pl.col("event_time").dt.strftime("%Y-%m-%d").alias("date"),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64),
            pl.lit(0.0).alias("dividend"),
            pl.lit(1.0).alias("split"),
        )
        df.write_csv(daily / f"{sid}.csv")
    return dest


def _run_zipline(csvdir: Path, weights: pl.DataFrame, sids: list[str]) -> tuple[pd.DataFrame, dict]:
    """Register + ingest a 24/7 bundle, then run the SMA-gate strategy."""
    os.environ.setdefault("ZIPLINE_ROOT", str(csvdir.parent / "zipline_root"))

    import exchange_calendars as xc  # noqa: F401  (registers calendars)
    from zipline import run_algorithm
    from zipline.api import order_target_percent, set_commission, set_slippage, symbol
    from zipline.data.bundles import ingest, register
    from zipline.data.bundles.csvdir import csvdir_equities
    from zipline.finance import commission, slippage
    from zipline.utils.calendar_utils import get_calendar

    class OpenFill(slippage.SlippageModel):
        """Fill the full order at the processing bar's open.

        Zipline daily mode processes an order on the bar it was placed, so
        the engine fills at ``open(session)``. To match the shared
        convention (decision close t -> exec open t+1) the algo submits the
        *previous* session's decision weights — ordering decided-at-t
        weights during session t+1 fills at open(t+1) identically.
        """

        def process_order(self, data, order):
            return (data.current(order.asset, "open"), order.amount)

    cal = get_calendar("24/7")
    register(
        "crypto24",
        csvdir_equities(["daily"], str(csvdir)),
        calendar_name="24/7",
        start_session=None,
        end_session=None,
    )
    ingest("crypto24", show_progress=False)

    # Weight panel: decision at bar t -> ordered during bar t -> fills open t+1.
    panel: dict[str, dict[str, float]] = {}
    for row in weights.iter_rows(named=True):
        panel.setdefault(row["event_time"].strftime("%Y-%m-%d"), {})[
            row["security_id"]
        ] = float(row["target_weight"])

    stats: dict[str, object] = {"orders": 0, "fees": 0.0}

    def initialize(context):
        context.assets = {sid: symbol(sid) for sid in sids}
        set_commission(commission.PerDollar(COMMISSION_BPS / 1e4))
        set_slippage(OpenFill())

    def handle_data(context, data):
        # Yesterday's decisions: an order placed on session d executes at
        # open(d), so weights decided at close(d-1) must be submitted now.
        dt = (context.get_datetime() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        wmap = panel.get(dt, {})
        for sid, asset in context.assets.items():
            order_target_percent(asset, wmap.get(sid, 0.0))

    def analyze(context, perf):
        # Commission is recorded per-transaction on the perf frame.
        stats["orders"] = int(sum(len(t.get("orders", [])) for t in []))  # unused
        pass

    first = min(w["event_time"] for w in weights.iter_rows(named=True))
    last = max(w["event_time"] for w in weights.iter_rows(named=True))
    # Pad the window one bar past the last decision so the final fill lands.
    from zoneinfo import ZoneInfo

    utc = ZoneInfo("UTC")
    start = (pd.Timestamp(first).tz_convert(utc) - pd.Timedelta(days=1)).tz_localize(None)
    # End on the last data bar — padding past it triggers auto-delisting
    # and a forced liquidation at the last close.
    end = pd.Timestamp(last).tz_convert(utc).tz_localize(None)

    t0 = time.perf_counter()
    perf = run_algorithm(
        start=start,
        end=end,
        initialize=initialize,
        handle_data=handle_data,
        analyze=analyze,
        capital_base=INIT_NAV,
        bundle="crypto24",
        data_frequency="daily",
        trading_calendar=cal,
    )
    elapsed = time.perf_counter() - t0
    stats["run_seconds"] = elapsed
    txns = [t for day in perf["transactions"] for t in (day or [])]
    orders = [o for day in perf["orders"] for o in (day or [])]
    stats["n_transactions"] = len(txns)
    stats["n_orders"] = len(orders)
    # Commission lives on the order record, not the transaction dict.
    stats["total_commission"] = float(
        sum((o.get("commission") or 0.0) for o in orders)
    )
    # Per-asset gross exposure over time (diagnostic + correctness evidence).
    def _gross(cell):
        items = cell.values() if isinstance(cell, dict) else (cell or [])
        total = 0.0
        for p in items:
            if isinstance(p, dict):
                total += abs(p.get("amount", 0.0) * p.get("last_sale_price", 0.0))
            else:
                total += abs(p.amount * p.last_sale_price)
        return total

    pos = perf["positions"].apply(_gross)
    stats["gross_exposure_max_over_nav"] = float(
        (pos / perf["portfolio_value"]).max()
    )
    stats["ending_cash"] = float(perf["ending_cash"].iloc[-1])
    # Dump a few sample transactions for price-level sanity checks.
    sample = [dict(t) for t in txns[:3]]  # raw keys for schema discovery
    stats["sample_transactions"] = [
        {
            "dt": str(t.get("dt")),
            "sid": getattr(t.get("sid"), "symbol", str(t.get("sid"))),
            "amount": float(t.get("amount", 0.0)),
            "price": float(t.get("price", 0.0)),
            "keys": sorted(map(str, t.keys())),
        }
        for t in txns[:10] + txns[-10:]
    ]
    stats["raw_txn_sample"] = [
        {k: str(v) for k, v in t.items()} for t in sample
    ]
    # Mid-run account state for the balloon diagnostic.
    mid = perf.iloc[len(perf) // 2]
    midpos = mid["positions"]
    stats["mid_positions"] = (
        {
            getattr(k, "symbol", str(k)): (
                dict(p) if isinstance(p, dict) else {"amount": p.amount, "px": p.last_sale_price}
            )
            for k, p in midpos.items()
        }
        if isinstance(midpos, dict)
        else str(type(midpos))
    )
    stats["mid_cash"] = float(mid["ending_cash"])
    stats["mid_nav"] = float(mid["portfolio_value"])
    return perf, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bars", type=Path, nargs="+", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    frames = load_panel([Path(p) for p in args.bars])
    sids = sorted(frames)
    weights = build_weight_panel(frames, SMA_WINDOW)
    print(f"assets: {sids}; weight rows={weights.height}")

    with tempfile.TemporaryDirectory(prefix="zl_csvdir_") as tmp:
        csvdir = _write_csvdir(frames, Path(tmp) / "csvdir")
        perf, stats = _run_zipline(csvdir, weights, sids)

    nav = perf[["portfolio_value"]].copy()
    nav.index = nav.index.tz_convert("UTC")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    nav.to_csv(args.out_dir / "zipline_nav.csv")

    receipt = {
        "schema": "incumbent_bench_zipline.v1",
        "incumbent": {
            "name": "zipline-reloaded",
            "url": "https://zipline.ml4trading.io/",
            "repo": "https://github.com/stefan-jansen/zipline-reloaded",
        },
        "workload": {
            "assets": sids,
            "bars_per_asset": int(frames[sids[0]].height),
            "bar_files_sha256": {Path(p).name: _sha256(Path(p)) for p in args.bars},
            "script_sha256": _sha256(Path(__file__)),
            "strategy": f"causal SMA{SMA_WINDOW} gate, equal-weight per gated name",
            "fills": "decision close t -> exec open t+1 (custom OpenFill slippage)",
            "commission_bps": COMMISSION_BPS,
            "calendar": "24/7",
            "init_nav": INIT_NAV,
        },
        "zipline_stats": stats,
        "environment": {"python": sys.version.split()[0], "platform": sys.platform},
        "research_only": True,
        "live_pnl_claim": False,
        "isolation_note": (
            "Ran under an isolated venv: zipline-reloaded requires pandas<3 "
            "while the project pins pandas>=3; a same-env comparison is "
            "impossible by construction. Numerics are compared via the "
            "emitted NAV series."
        ),
        "fairness_verdict": {
            "fair_comparison": False,
            "reference_nav": 1500764.656389184,
            "reference_engines": ["dipcatcher", "vectorbt", "numpy_replica"],
            "zipline_nav": "see zipline_nav.csv (diverges ~46x)",
            "blockers": [
                "pandas<3 env conflict (isolated venv required)",
                "daily-mode data.current(asset,'open') in slippage does not return bar opens",
                "integer-share rounding drops fractional crypto deltas",
                "auto-delisting at bundle end forces liquidation",
            ],
        },
    }
    (args.out_dir / "zipline_stats.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(f"zipline: {stats['n_transactions']} txns, "
          f"final_nav={float(nav['portfolio_value'].iloc[-1]):.2f}, "
          f"run={stats['run_seconds']:.1f}s")
    print(f"out: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
