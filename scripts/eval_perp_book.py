"""Evaluate perp signal sleeves on collected Binance data — dev + locked holdout.

Loads ``{sym}_{interval}.perp.parquet`` / ``{sym}.funding.parquet`` /
``{sym}_{interval}.spot.parquet`` from ``--data-root/raw/sources``, builds each
sleeve's weight panel, runs the perp margin engine and the spot engine, and
writes a JSON receipt with per-segment metrics. The holdout tail is locked by
convention: it is reported, never tuned against.

Research diagnostic only — ``live_pnl_claim`` is always false.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

import polars as pl  # noqa: E402

from quant_fund.backtest.engine import run_backtest  # noqa: E402
from quant_fund.backtest.overlay import (  # noqa: E402
    CompositeScaler,
    DrawdownGovernor,
    VolTargetScaler,
)
from quant_fund.backtest.perp_engine import run_perp_backtest  # noqa: E402
from quant_fund.backtest.sleeves import (  # noqa: E402
    blend_weights,
    cross_sectional_momentum_weights,
    funding_carry_weights,
    funding_spike_fade_weights,
    slow_trend_weights,
    sweep_reclaim_weights,
)
from quant_fund.config.loader import load_config  # noqa: E402


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_panel(
    sources: Path, suffix: str, interval: str, symbols: list[str] | None = None
) -> pl.DataFrame:
    """Concatenate per-symbol parquet files into one panel."""
    frames = []
    for path in sorted(sources.glob(f"*_{interval}.{suffix}.parquet")):
        sym = path.name.split("_")[0].upper()
        if symbols and sym not in symbols:
            continue
        frames.append(pl.read_parquet(path))
    if not frames:
        raise SystemExit(f"no {suffix} {interval} files under {sources}")
    return pl.concat(frames)


def load_funding(sources: Path, symbols: list[str] | None = None) -> pl.DataFrame:
    frames = []
    for path in sorted(sources.glob("*.funding.parquet")):
        sym = path.name.split(".")[0].upper()
        if symbols and sym not in symbols:
            continue
        frames.append(pl.read_parquet(path))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames)


def split_dev_holdout(
    bars: pl.DataFrame, holdout_frac: float
) -> tuple[pl.DataFrame, pl.DataFrame, object]:
    """Time-split: the last ``holdout_frac`` of the union calendar is locked."""
    times = sorted(bars["event_time"].unique().to_list())
    cut = times[int(len(times) * (1.0 - holdout_frac))]
    dev = bars.filter(pl.col("event_time") < cut)
    hold = bars.filter(pl.col("event_time") >= cut)
    return dev, hold, cut


def _metrics_summary(m: dict) -> dict:
    keys = (
        "total_return",
        "cagr",
        "sharpe",
        "max_drawdown",
        "n",
        "periods_per_year",
        "mean_turnover",
        "funding_paid_total",
        "liquidation_count",
        "margin_rejects",
        "ruined",
        "flag_high_sharpe",
        "risk_gate_rejects",
    )
    return {k: m.get(k) for k in keys if k in m}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--interval", default="1h")
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--initial-nav", type=float, default=1_000_000.0)
    p.add_argument("--symbols", nargs="*", default=None)
    p.add_argument("--max-symbols", type=int, default=None)
    p.add_argument("--config", default="configs/research.yaml")
    p.add_argument("--target-vol", type=float, default=0.20)
    p.add_argument("--overlay", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    sources = args.data_root / "raw" / "sources"
    bars = load_panel(sources, "perp", args.interval, args.symbols)
    if args.max_symbols:
        keep = sorted(bars["security_id"].unique().to_list())[: args.max_symbols]
        bars = bars.filter(pl.col("security_id").is_in(keep))
    funding = load_funding(sources, args.symbols)
    funding = funding.filter(pl.col("security_id").is_in(bars["security_id"].unique().to_list()))

    cfg = load_config(args.config)
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_gross = 4.0
    cfg.risk_gate.max_net = 4.0
    cfg.risk_gate.max_order_notional = 1e12

    dev_bars, hold_bars, cut = split_dev_holdout(bars, args.holdout_frac)
    dev_fund = funding.filter(pl.col("event_time") < pl.lit(cut)) if funding.height else funding
    hold_fund = funding.filter(pl.col("event_time") >= pl.lit(cut)) if funding.height else funding

    sleeves = {
        "funding_carry": funding_carry_weights(bars, funding, vol_window=48, gross_scale=1.5),
        "funding_spike_fade": funding_spike_fade_weights(
            bars, funding, lookback_events=30, z_threshold=2.0, gross_scale=1.0
        ),
        "momentum": cross_sectional_momentum_weights(
            bars, lookback_bars=168, skip_bars=4, gross_scale=1.5
        ),
        "sweep_reclaim": sweep_reclaim_weights(bars, lookback=24, hold_bars=8, gross_scale=0.5),
        "slow_trend": slow_trend_weights(bars, fast_bars=168, slow_bars=720, gross_scale=1.0),
    }
    sleeves["blend_equal"] = blend_weights(
        {k: v for k, v in sleeves.items() if k != "blend_equal"},
        {
            "funding_carry": 0.35,
            "funding_spike_fade": 0.10,
            "momentum": 0.30,
            "sweep_reclaim": 0.10,
            "slow_trend": 0.15,
        },
    )

    def fresh_scaler() -> CompositeScaler | None:
        if not args.overlay:
            return None
        return CompositeScaler(
            [
                VolTargetScaler(target_ann_vol=args.target_vol, window=168),
                DrawdownGovernor(dd_soft=0.05, dd_hard=0.10, floor=0.25),
            ]
        )

    receipt: dict = {
        "schema": "perp_eval.v1",
        "created_at": datetime.now(tz=UTC).isoformat(),
        "interval": args.interval,
        "holdout_cut": str(cut),
        "n_symbols": bars["security_id"].n_unique(),
        "bar_span": [
            str(bars["event_time"].min()),
            str(bars["event_time"].max()),
        ],
        "n_bars": bars.height,
        "n_funding": funding.height,
        "live_pnl_claim": False,
        "sleeves": {},
    }

    for name, w in sleeves.items():
        seg: dict = {}
        for label, b, f in (("dev", dev_bars, dev_fund), ("holdout", hold_bars, hold_fund)):
            wseg = (
                w.filter(
                    pl.col("event_time").is_between(
                        pl.lit(b["event_time"].min()), pl.lit(b["event_time"].max())
                    )
                )
                if w.height
                else w
            )
            if b.height < 10 or wseg.height == 0:
                seg[label] = {"skipped": True}
                continue
            res = run_perp_backtest(
                b, f, wseg, cfg, initial_nav=args.initial_nav, scaler=fresh_scaler()
            )
            seg[label] = _metrics_summary(res.metrics)
        receipt["sleeves"][name] = seg
        print(f"[{name}] done")

    # Spot-only comparison book: long-only weights through the daily spot engine
    # are meaningless on hourly bars; run the same sleeves on spot bars but with
    # shorts clipped (spot can't short) — reported as a separate book.
    try:
        spot_bars = load_panel(sources, "spot", args.interval, args.symbols)
        if args.max_symbols:
            spot_bars = spot_bars.filter(
                pl.col("security_id").is_in(bars["security_id"].unique().to_list())
            )
        spot_bars = spot_bars.with_columns(pl.col("close").alias("close_total_return"))
        spot_dev, spot_hold, _ = split_dev_holdout(spot_bars, args.holdout_frac)
        spot_w = sleeves["blend_equal"].with_columns(pl.col("target_weight").clip(lower_bound=0.0))
        receipt["spot_blend_longonly"] = {}
        for label, b in (("dev", spot_dev), ("holdout", spot_hold)):
            wseg = spot_w.filter(
                pl.col("event_time").is_between(
                    pl.lit(b["event_time"].min()), pl.lit(b["event_time"].max())
                )
            )
            if b.height < 10 or wseg.height == 0:
                receipt["spot_blend_longonly"][label] = {"skipped": True}
                continue
            res = run_backtest(b, wseg, cfg, initial_nav=args.initial_nav)
            receipt["spot_blend_longonly"][label] = _metrics_summary(res.metrics)
        print("[spot blend] done")
    except SystemExit:
        receipt["spot_blend_longonly"] = {"skipped": "no spot files"}

    receipt["input_hashes"] = {p.name: _sha256(p) for p in sorted(sources.glob("*.parquet"))[:400]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, default=str))
    print(f"receipt -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
