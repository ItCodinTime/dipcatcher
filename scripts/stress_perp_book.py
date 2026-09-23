"""Stress harness for the perp book — reruns one weight panel under adverse
mutations: cost multipliers, fill latency, funding spikes, bar outages.

Each scenario is a full ``run_perp_backtest`` pass; the receipt reports whether
the headline metrics survive. Research diagnostic only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from quant_fund.backtest.perp_engine import run_perp_backtest  # noqa: E402
from quant_fund.config.loader import load_config  # noqa: E402


def inject_outages(bars: pl.DataFrame, drop_frac: float, seed: int = 7) -> pl.DataFrame:
    """Drop a random fraction of bars — simulates API outages / missing data.
    Held positions then age their marks toward the stale-limit failure path."""
    rng = np.random.default_rng(seed)
    keep = rng.random(bars.height) >= drop_frac
    return bars.filter(pl.Series(keep))


def inject_price_shock(
    bars: pl.DataFrame, at_frac: float = 0.5, shock: float = -0.25
) -> pl.DataFrame:
    """Force a one-bar adverse move at ``at_frac`` of the timeline — a
    liquidation-cascade probe."""
    times = sorted(bars["event_time"].unique().to_list())
    t0 = times[int(len(times) * at_frac)]
    t1 = times[int(len(times) * at_frac) + 1] if int(len(times) * at_frac) + 1 < len(times) else t0
    return bars.with_columns(
        pl.when(pl.col("event_time") == pl.lit(t1))
        .then(pl.col("close") * (1.0 + shock))
        .otherwise(pl.col("close"))
        .alias("close"),
        pl.when(pl.col("event_time") == pl.lit(t1))
        .then(pl.col("low") * (1.0 + shock * 1.5))
        .otherwise(pl.col("low"))
        .alias("low"),
        pl.when(pl.col("event_time") == pl.lit(t1))
        .then(pl.col("high") * (1.0 + min(shock, 0.0)))
        .otherwise(pl.col("high"))
        .alias("high"),
    )


def _summary(m: dict) -> dict:
    keys = (
        "total_return",
        "sharpe",
        "max_drawdown",
        "liquidation_count",
        "funding_paid_total",
        "ruined",
        "n",
    )
    return {k: m.get(k) for k in keys if k in m}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bars", type=Path, required=True, help="perp panel parquet")
    p.add_argument("--funding", type=Path, default=None)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--config", default="configs/research.yaml")
    p.add_argument("--initial-nav", type=float, default=1_000_000.0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    bars = pl.read_parquet(args.bars)
    funding = pl.read_parquet(args.funding) if args.funding else None
    weights = pl.read_parquet(args.weights)

    base = load_config(args.config)
    base.risk_gate.max_name = 1.0
    base.risk_gate.max_gross = 4.0
    base.risk_gate.max_net = 4.0
    base.risk_gate.max_order_notional = 1e12

    scenarios: dict[str, tuple[pl.DataFrame, object]] = {}

    def cfg_with(**perp_kw):
        c = base.model_copy(deep=True)
        for k, v in perp_kw.items():
            setattr(c.perp, k, v)
        return c

    scenarios["baseline"] = (bars, base.model_copy(deep=True))

    c = base.model_copy(deep=True)
    c.costs.impact_y *= 2.0
    c.costs.half_spread_bps *= 3.0
    c.costs.commission_bps *= 2.0
    scenarios["costs_2x_impact_3x_spread"] = (bars, c)

    scenarios["latency_plus1bar"] = (bars, cfg_with(fill_delay_bars=1))
    scenarios["latency_plus2bar"] = (bars, cfg_with(fill_delay_bars=2))
    scenarios["funding_spike_3x"] = (bars, cfg_with(funding_spike_multiplier=3.0))
    scenarios["funding_spike_5x"] = (bars, cfg_with(funding_spike_multiplier=5.0))
    scenarios["outage_5pct_bars"] = (inject_outages(bars, 0.05), base.model_copy(deep=True))
    scenarios["outage_15pct_bars"] = (inject_outages(bars, 0.15), base.model_copy(deep=True))
    scenarios["price_shock_-25pct"] = (
        inject_price_shock(bars, 0.5, -0.25),
        base.model_copy(deep=True),
    )
    scenarios["no_wick_liquidation"] = (
        bars,
        cfg_with(liquidation_on_wick=False),
    )

    receipt: dict[str, Any] = {
        "schema": "perp_stress.v1",
        "live_pnl_claim": False,
        "scenarios": {},
    }
    for name, (b, c) in scenarios.items():
        try:
            res = run_perp_backtest(b, funding, weights, c, initial_nav=args.initial_nav)
            receipt["scenarios"][name] = _summary(res.metrics)
        except Exception as exc:  # fail-closed surfaces as data, not a crash
            receipt["scenarios"][name] = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"[{name}] done")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, default=str))
    print(f"receipt -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
