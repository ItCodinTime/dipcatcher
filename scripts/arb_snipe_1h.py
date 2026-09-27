"""Epoch-snipe variant for the 1h cross-venue arb book (megaplan closer lane).

The frozen 1h champion holds continuously between funding epochs and pays
for it in basis-wick exposure (the −254% liquidation tail). This variant
holds only around each funding payment: a synthetic funding frame emits
the REAL rate at `t - lead` bars (enter before the epoch, guaranteed on
for the payment at t) and a forced-exit event (value=-1) at `t + wh`
bars. P&L settles on the real 1h spread frame; the question is whether
concentrating holding time on payment bars improves the Sharpe enough to
approach the gate. Dev-half grid over (lead, wh, enter, lb) freezes on
min(h1,h2) score, then ONE locked holdout eval. Failure is a result.

Usage: uv run python scripts/arb_snipe_1h.py --data data/arb_carry_book_1h
Writes artifacts/arb_snipe_1h.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
from datetime import UTC, datetime, timedelta

import polars as pl
from carry_research import eligible_coins, load_carry, make_cfg

from quant_fund.backtest.carry_engine import run_carry_backtest
from quant_fund.backtest.sleeves import basis_carry_hysteresis_weights


def _slice(p, s, f, lo, hi):
    lo = datetime.fromisoformat(lo) if isinstance(lo, str) else lo
    hi = datetime.fromisoformat(hi) if isinstance(hi, str) else hi

    def f_(df):
        return df.filter((pl.col("event_time") >= lo) & (pl.col("event_time") < hi))

    return f_(p), f_(s), f_(f)


def _score(m: dict) -> float:
    if not m or m.get("n", 0) < 30:
        return -999.0
    pen = 0.5 * (m.get("liquidation_count") or 0) + 0.001 * (m.get("margin_rejects") or 0)
    return (m.get("sharpe") or -999.0) - pen


def _snipe_funding(fund: pl.DataFrame, lead_bars: int, hold_bars: int) -> pl.DataFrame:
    """Per real event (sid, t, rate>0): enter-signal at t - lead_bars h,
    forced exit (value=-1) at t + hold_bars h."""
    enter = fund.filter(pl.col("value") > 0).select(
        (pl.col("event_time") - timedelta(hours=lead_bars)).alias("event_time"),
        pl.col("security_id"),
        pl.col("value"),
    )
    exit_ = fund.filter(pl.col("value") > 0).select(
        (pl.col("event_time") + timedelta(hours=hold_bars)).alias("event_time"),
        pl.col("security_id"),
        pl.lit(-1.0).alias("value"),
    )
    return pl.concat([enter, exit_]).sort(["security_id", "event_time"])


def _run(perp, spot, fund_real, synth_fund, cfg_kw) -> dict:
    w = basis_carry_hysteresis_weights(perp, synth_fund, **cfg_kw)
    res = run_carry_backtest(perp, spot, fund_real, w, make_cfg(), initial_nav=1e6)
    m = res.metrics
    keep = (
        "total_return",
        "cagr",
        "sharpe",
        "max_drawdown",
        "mean_turnover",
        "funding_net",
        "funding_events_dropped",
        "liquidation_count",
        "margin_rejects",
        "risk_gate_rejects",
        "n",
    )
    return {k: m.get(k) for k in keep}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/arb_carry_book_1h")
    ap.add_argument("--split", default="2026-06-01T00:00:00")
    ap.add_argument("--mid", default="2026-04-15T00:00:00")
    ap.add_argument("--out", default="artifacts/arb_snipe_1h.json")
    args = ap.parse_args()
    split = datetime.fromisoformat(args.split).replace(tzinfo=UTC)
    mid = datetime.fromisoformat(args.mid).replace(tzinfo=UTC)

    perp, spot, fund = load_carry(data_dir=pathlib.Path(args.data))
    print(
        "sids:",
        perp["security_id"].n_unique(),
        "| fund:",
        fund.height,
        "| span:",
        fund["event_time"].min(),
        "->",
        fund["event_time"].max(),
        flush=True,
    )

    leads = (0, 1)
    whs = (1, 2, 4)
    s_frames = {(ld, wh): _snipe_funding(fund, ld, wh) for ld, wh in itertools.product(leads, whs)}

    dev_p, dev_s, dev_f = _slice(perp, spot, fund, datetime(1970, 1, 1, tzinfo=UTC), split)
    keep_dev = eligible_coins(dev_p, dev_s)
    dev_p = dev_p.filter(pl.col("security_id").is_in(keep_dev))
    dev_s = dev_s.filter(pl.col("security_id").is_in(keep_dev))
    dev_f = dev_f.filter(pl.col("security_id").is_in(keep_dev))
    dev_sf = {
        k: f_.filter(pl.col("security_id").is_in(keep_dev) & (pl.col("event_time") < split))
        for k, f_ in s_frames.items()
    }

    GRID = []
    for ld, wh, enter, lb in itertools.product(leads, whs, (1.7e-4, 5e-4, 1e-3), (1, 3)):
        GRID.append(
            dict(
                lead=ld,
                wh=wh,
                enter_rate=enter,
                exit_rate=0.0,
                lookback_events=lb,
                name_weight=0.08,
                max_names=30,
                rebalance_band=1.3,
                rate_scale_ref=2e-3,
                rate_scale_cap=1.5,
                rate_scale_floor=1.0,
            )
        )

    stage = {}
    for i, g in enumerate(GRID):
        cfg_kw = {k: v for k, v in g.items() if k not in ("lead", "wh")}
        sdev = dev_sf[(g["lead"], g["wh"])]
        rec = {"cfg": g}
        try:
            h1p, h1s, h1f = _slice(dev_p, dev_s, dev_f, datetime(1970, 1, 1, tzinfo=UTC), mid)
            h1y = sdev.filter(pl.col("event_time") < mid)
            m1 = _run(h1p, h1s, h1f, h1y, cfg_kw)
            h2p, h2s, h2f = _slice(dev_p, dev_s, dev_f, mid, split)
            h2y = sdev.filter(pl.col("event_time") >= mid)
            m2 = _run(h2p, h2s, h2f, h2y, cfg_kw)
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
            stage[f"g{i}"] = rec
            continue
        rec["h1_sharpe"], rec["h2_sharpe"] = m1.get("sharpe"), m2.get("sharpe")
        rec["score"] = min(_score(m1), _score(m2))
        stage[f"g{i}"] = rec
        print(
            f"g{i} h1={rec['h1_sharpe']} h2={rec['h2_sharpe']} score={rec['score']:.3f}",
            flush=True,
        )

    done = {k: v for k, v in stage.items() if "score" in v}
    if not done:
        print("no config completed the dev grid — aborting before holdout")
        return 1
    best_key = max(done, key=lambda k: done[k]["score"])
    champ = done[best_key]["cfg"]
    print("frozen champion:", best_key, champ, flush=True)

    out = {
        "stage": stage,
        "frozen": {
            "id": best_key,
            **champ,
            "qualified_on_dev": done[best_key]["score"] > -999,
        },
    }
    champ_kw = {k: v for k, v in champ.items() if k not in ("lead", "wh")}
    sfull = s_frames[(champ["lead"], champ["wh"])]
    for label, (p, s, f) in {
        "dev": (dev_p, dev_s, dev_f),
        "holdout": _slice(perp, spot, fund, split, datetime(2100, 1, 1, tzinfo=UTC)),
    }.items():
        keep_l = eligible_coins(p, s)
        pp = p.filter(pl.col("security_id").is_in(keep_l))
        ss = s.filter(pl.col("security_id").is_in(keep_l))
        ff = f.filter(pl.col("security_id").is_in(keep_l))
        yy = sfull.filter(pl.col("security_id").is_in(keep_l))
        m = _run(pp, ss, ff, yy, champ_kw)
        out[label] = m
        print(label, json.dumps(m, default=str), flush=True)

    h_sharpe = out["holdout"].get("sharpe")
    h_mdd = out["holdout"].get("max_drawdown")
    out["gate"] = {
        "PROVEN": bool(
            out["frozen"]["qualified_on_dev"]
            and h_sharpe
            and h_sharpe > 5.0
            and h_mdd
            and abs(h_mdd) < 0.05
        )
    }
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print("wrote", args.out, "| gate:", out["gate"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
