"""Basis-divergence fade for the multi-venue arb book (megaplan closer lane).

Every lane so far sized positions by the FUNDING spread. This one ignores
funding for membership: each pair-sid's mark basis
(log(perp_close/spot_close)) is z-scored per sid over a trailing lookback,
and the sizing signal is −z — enter long the pair when the basis is
depressed vs its own history (fade the venue-divergence wicks that caused
the liquidation tails) and ride the reversion. Emitted as a synthetic
funding frame consumed by ``basis_carry_hysteresis_weights``; real funding
still settles on top. Dev-half grid over (z lookback, z enter, z exit,
hysteresis lb) freezes on min(h1,h2) score, then ONE locked holdout eval.
Failure is a result.

Usage: uv run python scripts/arb_basis_fade.py --data data/arb3_book
Writes artifacts/arb_basis_fade.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
from datetime import UTC, datetime

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


def _basis_z_funding(perp: pl.DataFrame, spot: pl.DataFrame, lookback: int) -> pl.DataFrame:
    """value = -z(log(perp_close/spot_close)) per sid — negative basis z
    (depressed basis) maps to positive weight (long the pair)."""
    b = perp.select("event_time", "security_id", pl.col("close").alias("pc")).join(
        spot.select("event_time", "security_id", pl.col("close").alias("sc")),
        on=["event_time", "security_id"],
        how="inner",
    )
    return (
        b.sort(["security_id", "event_time"])
        .with_columns((pl.col("pc") / pl.col("sc")).log().alias("basis"))
        .with_columns(
            pl.col("basis")
            .rolling_mean(lookback, min_samples=max(10, lookback // 3))
            .over("security_id")
            .alias("mu"),
            pl.col("basis")
            .rolling_std(lookback, min_samples=max(10, lookback // 3))
            .over("security_id")
            .alias("sd"),
        )
        .with_columns((-(pl.col("basis") - pl.col("mu")) / (pl.col("sd") + 1e-9)).alias("value"))
        .select("event_time", "security_id", "value")
        .drop_nulls("value")
        .filter(pl.col("value").is_finite())
    )


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
    ap.add_argument("--data", default="data/arb3_book")
    ap.add_argument("--split", default="2025-01-01T00:00:00")
    ap.add_argument("--out", default="artifacts/arb_basis_fade.json")
    args = ap.parse_args()
    split = datetime.fromisoformat(args.split).replace(tzinfo=UTC)

    perp, spot, fund = load_carry(data_dir=pathlib.Path(args.data))
    print("sids:", perp["security_id"].n_unique(), flush=True)

    z_lookbacks = (20, 60, 120)
    z_frames = {zl: _basis_z_funding(perp, spot, zl) for zl in z_lookbacks}
    for zl, zf in z_frames.items():
        print(f"zl={zl} z-events={zf.height}", flush=True)

    dev_p, dev_s, dev_f = _slice(perp, spot, fund, datetime(1970, 1, 1, tzinfo=UTC), split)
    keep_dev = eligible_coins(dev_p, dev_s)
    dev_p = dev_p.filter(pl.col("security_id").is_in(keep_dev))
    dev_s = dev_s.filter(pl.col("security_id").is_in(keep_dev))
    dev_f = dev_f.filter(pl.col("security_id").is_in(keep_dev))
    dev_z = {
        zl: zf.filter(pl.col("security_id").is_in(keep_dev) & (pl.col("event_time") < split))
        for zl, zf in z_frames.items()
    }
    mid = datetime(2024, 1, 1, tzinfo=UTC)

    GRID = []
    for zl, ze, zx, lb, rsr in itertools.product(
        z_lookbacks,
        (0.5, 1.0, 1.5, 2.0),
        (-0.5, 0.0, 0.5),
        (3, 9),
        (2.0,),
    ):
        GRID.append(
            dict(
                z_lookback=zl,
                enter_rate=ze,
                exit_rate=zx,
                lookback_events=lb,
                name_weight=0.08,
                max_names=30,
                rebalance_band=1.3,
                rate_scale_ref=rsr,
                rate_scale_cap=1.5,
                rate_scale_floor=1.0,
            )
        )

    stage = {}
    for i, g in enumerate(GRID):
        cfg_kw = {k: v for k, v in g.items() if k != "z_lookback"}
        zdev = dev_z[g["z_lookback"]]
        rec = {"cfg": g}
        try:
            h1p, h1s, h1f = _slice(dev_p, dev_s, dev_f, datetime(1970, 1, 1, tzinfo=UTC), mid)
            h1y = zdev.filter(pl.col("event_time") < mid)
            m1 = _run(h1p, h1s, h1f, h1y, cfg_kw)
            h2p, h2s, h2f = _slice(dev_p, dev_s, dev_f, mid, split)
            h2y = zdev.filter(pl.col("event_time") >= mid)
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
    champ_kw = {k: v for k, v in champ.items() if k != "z_lookback"}
    zfull = z_frames[champ["z_lookback"]]
    for label, (p, s, f) in {
        "dev": (dev_p, dev_s, dev_f),
        "holdout": _slice(perp, spot, fund, split, datetime(2100, 1, 1, tzinfo=UTC)),
    }.items():
        keep_l = eligible_coins(p, s)
        pp = p.filter(pl.col("security_id").is_in(keep_l))
        ss = s.filter(pl.col("security_id").is_in(keep_l))
        ff = f.filter(pl.col("security_id").is_in(keep_l))
        yy = zfull.filter(pl.col("security_id").is_in(keep_l))
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
