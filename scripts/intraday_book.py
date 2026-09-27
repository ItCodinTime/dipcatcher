"""Intraday (hourly) multi-asset book — public Yahoo 1h data, honest costs.

Why hourly: ~7 bars/day -> ~1750 independent decision points/year vs 252
daily. Sharpe scales ~sqrt(bets) if a real edge exists, so intraday is the
credible route toward high-Sharpe on public data. Cost model is the same
honest one (commission + half-spread + borrow), next-bar fills.

Sleeves (all causal, z-scored cross-sectionally per bar):
  xrev : fade last-bar return z-score (long losers / short winners),
         inverse-vol sized — classic intraday reversal.
  xgap : fade overnight gap (first bar of day: open vs prior close),
         z-gated — overreaction fade.
  tmom : per-name trailing intraday momentum (5-bar sign), inv-vol.
  daymom : ride today's cumulative intraday return into the close
           (entered at bar k>=4 of day, flattened at session end).

Dev / locked-holdout split via --holdout-frac / --holdout. Metrics
annualized from measured bars-per-year. Receipt stamped research-only.
"""

from __future__ import annotations  # noqa: E402

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from quant_fund.backtest.overlay import (  # noqa: E402
    CompositeScaler,
    DrawdownGovernor,
    VolTargetScaler,
)
from quant_fund.backtest.perp_engine import run_perp_backtest  # noqa: E402
from quant_fund.config import load_config  # noqa: E402


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _load(src: Path) -> pl.DataFrame:
    parts = []
    for p in sorted(src.glob("*_1h.parquet")):
        df = pl.read_parquet(p)
        if df.height < 500:
            continue
        parts.append(
            df.with_columns(
                pl.lit("yahoo_1h").alias("source"),
                pl.col("close").alias("close_total_return"),
            ).select(
                "security_id",
                "event_time",
                "open",
                "high",
                "low",
                "close",
                "close_total_return",
                "volume",
                "source",
            )
        )
    if not parts:
        raise SystemExit(f"no hourly bars under {src}")
    return pl.concat(parts)


def _matrices(bars: pl.DataFrame):
    piv = bars.pivot(
        on="security_id", index="event_time", values="close", aggregate_function="last"
    ).sort("event_time")
    times = piv["event_time"].to_list()
    sids = [c for c in piv.columns if c != "event_time"]
    close = piv.select(sids).to_numpy()
    piv_o = bars.pivot(
        on="security_id", index="event_time", values="open", aggregate_function="last"
    ).sort("event_time")
    opens = piv_o.select(sids).to_numpy()
    return times, sids, close, opens


def _vol_matrix(close: np.ndarray, win: int) -> np.ndarray:
    """Per-bar realized vol of log returns, trailing `win` bars (causal)."""
    n, m = close.shape
    out = np.full((n, m), np.nan)
    logc = np.where(close > 0, np.log(np.where(close > 0, close, np.nan)), np.nan)
    r = np.full((n, m), np.nan)
    r[1:] = logc[1:] - logc[:-1]
    for i in range(win, n):
        w = r[i - win : i]
        out[i] = np.nanstd(w, axis=0)
    return out


def _zscore(x: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    if not np.isfinite(mad) or mad <= 1e-9:
        return np.full_like(x, np.nan)
    return np.clip((x - med) / mad, -3.0, 3.0)


def build_weights(times, sids, close, opens, args) -> dict:
    n, m = close.shape
    logc = np.where(close > 0, np.log(np.where(close > 0, close, np.nan)), np.nan)
    ret1 = np.full((n, m), np.nan)
    ret1[1:] = logc[1:] - logc[:-1]
    vol = _vol_matrix(close, args.vol_bars)
    sleeves = set(args.sleeves.split(","))
    out: dict = {}
    # day index per bar (date changes) + bar-of-day
    dates = np.array([t.date() for t in times])
    day_id = np.zeros(n, dtype=np.int64)
    day_id[1:] = np.cumsum(dates[1:] != dates[:-1])
    bar_of_day = np.zeros(n, dtype=np.int64)
    cur = 0
    for i in range(1, n):
        cur = 0 if day_id[i] != day_id[i - 1] else cur + 1
        bar_of_day[i] = cur
    for i in range(1, n):
        w: dict[str, float] = {}
        bod = bar_of_day[i]
        if "xrev" in sleeves:
            z = _zscore(ret1[i])
            for j in range(m):
                if not np.isfinite(z[j]) or abs(z[j]) < args.xrev_gate:
                    continue
                v = vol[i, j]
                if not np.isfinite(v) or v <= 0:
                    continue
                w[sids[j]] = w.get(sids[j], 0.0) - z[j] * args.xrev_risk / (
                    v * np.sqrt(args.bars_per_year)
                )
        if "xgap" in sleeves and bod == 0:
            gap = np.where(
                np.isfinite(opens[i]) & np.isfinite(close[i - 1]) & (close[i - 1] > 0),
                np.log(np.where(opens[i] > 0, opens[i], np.nan) / close[i - 1]),
                np.nan,
            )
            z = _zscore(gap)
            for j in range(m):
                if not np.isfinite(z[j]) or abs(z[j]) < args.xgap_gate:
                    continue
                v = vol[i, j]
                if not np.isfinite(v) or v <= 0:
                    continue
                w[sids[j]] = w.get(sids[j], 0.0) - z[j] * args.xgap_risk / (
                    v * np.sqrt(args.bars_per_year)
                )
        if "tmom" in sleeves and i >= args.tmom_bars + 1:
            mom = logc[i] - logc[i - args.tmom_bars]
            z = _zscore(mom)
            for j in range(m):
                if not np.isfinite(z[j]) or abs(z[j]) < args.tmom_gate:
                    continue
                v = vol[i, j]
                if not np.isfinite(v) or v <= 0:
                    continue
                w[sids[j]] = w.get(sids[j], 0.0) + z[j] * args.tmom_risk / (
                    v * np.sqrt(args.bars_per_year)
                )
        if "daymom" in sleeves and bod >= args.daymom_entry and bod < 6:
            first = i - bod
            dayret = logc[i] - logc[first]
            z = _zscore(dayret)
            for j in range(m):
                if not np.isfinite(z[j]) or abs(z[j]) < args.daymom_gate:
                    continue
                v = vol[i, j]
                if not np.isfinite(v) or v <= 0:
                    continue
                w[sids[j]] = w.get(sids[j], 0.0) + z[j] * args.daymom_risk / (
                    v * np.sqrt(args.bars_per_year)
                )
        # flatten at last bar of day: next bar opens a new session; weight map
        # already zeroes ungated names, but positions held to session end are
        # fine — engine carries weights until next explicit row anyway.
        gross = sum(abs(x) for x in w.values())
        if gross > args.gross_cap and gross > 0:
            f = args.gross_cap / gross
            w = {k: x * f for k, x in w.items()}
        out[times[i]] = w
    return out


def _weights_frame(wmap, times, sids) -> pl.DataFrame:
    rows = []
    for dt in times:
        w = wmap.get(dt)
        if not w:
            continue
        for sid, wt in w.items():
            rows.append((dt, sid, float(wt)))
    if not rows:
        return pl.DataFrame(
            schema={"event_time": pl.Datetime, "security_id": pl.Utf8, "weight": pl.Float64}
        )
    return pl.DataFrame(
        {
            "event_time": [r[0] for r in rows],
            "security_id": [r[1] for r in rows],
            "weight": [r[2] for r in rows],
        }
    )


def _metrics(m) -> dict:
    return {
        k: getattr(m, k, None)
        for k in (
            "total_return",
            "cagr",
            "sharpe",
            "max_drawdown",
            "n",
            "margin_rejects",
            "ruined",
            "turnover",
        )
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--src-dir", type=str, default="sources_yahoo_1h")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sleeves", type=str, default="xrev")
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--gross-cap", type=float, default=2.0)
    ap.add_argument("--name-cap", type=float, default=0.30)
    ap.add_argument("--target-vol", type=float, default=0.10)
    ap.add_argument("--vol-window", type=int, default=60)
    ap.add_argument("--max-scale", type=float, default=4.0)
    ap.add_argument("--dd-soft", type=float, default=0.03)
    ap.add_argument("--dd-hard", type=float, default=0.10)
    ap.add_argument("--dd-floor", type=float, default=0.3)
    ap.add_argument("--vol-bars", type=int, default=35)
    ap.add_argument("--bars-per-year", type=float, default=1750.0)
    ap.add_argument("--xrev-risk", type=float, default=0.02)
    ap.add_argument("--xrev-gate", type=float, default=1.0)
    ap.add_argument("--xgap-risk", type=float, default=0.03)
    ap.add_argument("--xgap-gate", type=float, default=1.5)
    ap.add_argument("--tmom-bars", type=int, default=5)
    ap.add_argument("--tmom-risk", type=float, default=0.02)
    ap.add_argument("--tmom-gate", type=float, default=1.0)
    ap.add_argument("--daymom-entry", type=int, default=4)
    ap.add_argument("--daymom-risk", type=float, default=0.03)
    ap.add_argument("--daymom-gate", type=float, default=0.75)
    ap.add_argument("--commission-bps", type=float, default=0.5)
    ap.add_argument("--half-spread-bps", type=float, default=2.0)
    ap.add_argument("--borrow-bps", type=float, default=50.0)
    ap.add_argument("--initial-nav", type=float, default=1.0)
    args = ap.parse_args()

    bars = _load(args.data_root / "raw" / args.src_dir)
    all_times = sorted(set(bars["event_time"].to_list()))
    cut = all_times[int(len(all_times) * (1.0 - args.holdout_frac))]
    seg0, seg1 = (all_times[0], cut) if not args.holdout else (cut, all_times[-1])
    bars_seg = bars.filter(
        (pl.col("event_time") >= seg0) & (pl.col("event_time") <= seg1)
        if args.holdout
        else pl.col("event_time") < seg1
    )
    seg_times = sorted(set(bars_seg["event_time"].to_list()))
    sids = sorted(set(bars_seg["security_id"].to_list()))

    times_full, sids_full, close_full, opens_full = _matrices(bars)
    wmap_full = build_weights(times_full, sids_full, close_full, opens_full, args)
    wmap = {dt: wmap_full.get(dt, {}) for dt in seg_times}
    weights = _weights_frame(wmap, seg_times, sids)

    # measured bars/year from the segment (annualization honesty)
    if len(seg_times) > 1:
        span_days = (seg_times[-1] - seg_times[0]).total_seconds() / 86400.0
        bpy = len(seg_times) / max(span_days, 1.0) * 365.25
    else:
        bpy = args.bars_per_year

    cfg = load_config(args.config)
    cfg.perp.funding_enabled = False
    cfg.perp.periods_per_year_override = bpy
    cfg.perp.bar_seconds_hint = 3600.0
    cfg.costs.commission_bps = args.commission_bps
    cfg.costs.half_spread_bps = args.half_spread_bps
    cfg.costs.borrow_bps_per_year = args.borrow_bps
    cfg.costs.frictionless = False
    cfg.constraints.gross_leverage = max(args.gross_cap * args.max_scale, 1.0)
    cfg.constraints.name_max = args.name_cap * args.max_scale
    cfg.constraints.net_exposure = 0.95
    cfg.risk_gate.max_gross = max(args.gross_cap * args.max_scale, 1.0)
    cfg.risk_gate.max_net = 2.0 * args.max_scale
    cfg.risk_gate.max_name = args.name_cap * args.max_scale
    cfg.risk_gate.max_predicted_vol = 10.0
    cfg.risk_gate.max_participation = 0.25

    scaler = CompositeScaler(
        [
            VolTargetScaler(
                target_ann_vol=args.target_vol,
                window=args.vol_window,
                periods_per_year=bpy,
                min_scale=0.0,
                max_scale=args.max_scale,
            ),
            DrawdownGovernor(dd_soft=args.dd_soft, dd_hard=args.dd_hard, floor=args.dd_floor),
        ]
    )

    t0 = datetime.now(UTC)
    res = run_perp_backtest(
        bars_seg, None, weights, cfg, initial_nav=args.initial_nav, scaler=scaler
    )
    elapsed = (datetime.now(UTC) - t0).total_seconds()
    m = res.metrics
    receipt = {
        "schema": "intraday_book.v1",
        "live_pnl_claim": False,
        "segment": "holdout" if args.holdout else "dev",
        "seg0": str(seg0),
        "seg1": str(seg1),
        "bars_per_year_measured": bpy,
        "sleeves": args.sleeves,
        "params": {
            "xrev_risk": args.xrev_risk,
            "xrev_gate": args.xrev_gate,
            "xgap_risk": args.xgap_risk,
            "xgap_gate": args.xgap_gate,
            "tmom_bars": args.tmom_bars,
            "tmom_risk": args.tmom_risk,
            "tmom_gate": args.tmom_gate,
            "daymom_entry": args.daymom_entry,
            "daymom_risk": args.daymom_risk,
            "daymom_gate": args.daymom_gate,
            "gross_cap": args.gross_cap,
            "name_cap": args.name_cap,
            "target_vol": args.target_vol,
            "max_scale": args.max_scale,
            "dd_soft": args.dd_soft,
            "dd_hard": args.dd_hard,
            "dd_floor": args.dd_floor,
            "commission_bps": args.commission_bps,
            "half_spread_bps": args.half_spread_bps,
            "borrow_bps_per_year": args.borrow_bps,
        },
        "metrics": _metrics(m),
        "n_weight_rows": weights.height,
        "n_bars": bars_seg.height,
        "elapsed_s": round(elapsed, 2),
        "data_sha256": {
            p.name: _sha256(p)
            for p in sorted((args.data_root / "raw" / args.src_dir).glob("*_1h.parquet"))
        },
        "caveats": [
            "730d Yahoo cap — short history, single regime (2023-26 bull).",
            "Dividend gaps inside window not adjusted (minor at hourly scale).",
            "half_spread 2bps is optimistic for small names, honest for ETFs.",
            "Survivorship: universe is current constituents, disclosed limitation.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, default=str) + "\n")
    print(
        json.dumps(
            {
                k: receipt["metrics"][k]
                for k in ("sharpe", "max_drawdown", "cagr", "total_return", "turnover")
            },
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
