"""Adaptive z-score entry for the multi-venue arb book (megaplan closer lane).

The frozen champion keys entries on the ABSOLUTE funding spread
(|spread| > enter_rate ≈ 1e-3). In the compressed 2025-26 regime that
threshold goes deaf — spreads still fluctuate but rarely reach the old
absolute level. This variant replaces the absolute threshold with a
per-sid trailing z-score: each pair-sid's daily funding spread is
z-scored against its own rolling mean/std over a lookback window, and
the z series is emitted as a synthetic funding frame consumed by
``basis_carry_hysteresis_weights``. Membership/sizing keys off the
z-score (sign preserved), while P&L settles on the REAL spread frame.
Grid over (z lookback, z enter, z exit, hysteresis lb, scale ref)
freezes on dev halves (min(h1,h2) score), then ONE locked holdout eval.
Failure is a result.

Usage: uv run python scripts/arb_zscore_spread.py --data data/arb3_book
Writes artifacts/arb_zscore_spread.json.
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


def _z_funding(fund: pl.DataFrame, lookback: int, sign: float = 1.0) -> pl.DataFrame:
    """Same event times as the real funding frame; value = trailing z of
    the realized spread (current observation included — causal at event
    time). z = (v - mean_lb(v)) / sd_lb(v). sign=-1 fades the spike."""
    return (
        fund.sort(["security_id", "event_time"])
        .with_columns(
            pl.col("value")
            .rolling_mean(lookback, min_samples=max(10, lookback // 3))
            .over("security_id")
            .alias("mu"),
            pl.col("value")
            .rolling_std(lookback, min_samples=max(10, lookback // 3))
            .over("security_id")
            .alias("sd"),
        )
        .with_columns(
            (sign * (pl.col("value") - pl.col("mu")) / (pl.col("sd") + 1e-9)).alias("value")
        )
        .select("event_time", "security_id", "value")
        .drop_nulls("value")
        .filter(pl.col("value").is_finite())
    )


def _z_funding_hybrid(fund: pl.DataFrame, lookback: int, ze: float) -> pl.DataFrame:
    """Absolute spread value, zeroed unless |z| >= ze — entry requires the
    spread to be BOTH large in absolute terms AND unusual vs the sid's own
    history."""
    return (
        fund.sort(["security_id", "event_time"])
        .with_columns(
            pl.col("value")
            .rolling_mean(lookback, min_samples=max(10, lookback // 3))
            .over("security_id")
            .alias("mu"),
            pl.col("value")
            .rolling_std(lookback, min_samples=max(10, lookback // 3))
            .over("security_id")
            .alias("sd"),
        )
        .with_columns(
            pl.when((pl.col("value") - pl.col("mu")).abs() / (pl.col("sd") + 1e-9) >= ze)
            .then(pl.col("value"))
            .otherwise(0.0)
            .alias("value")
        )
        .select("event_time", "security_id", "value")
        .drop_nulls("value")
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
    ap.add_argument("--out", default="artifacts/arb_zscore_spread.json")
    ap.add_argument(
        "--sign",
        type=float,
        default=1.0,
        help="1.0 = spread-momentum entry (ride the z spike); -1.0 = fade entry (bet on z mean-reversion)",
    )
    ap.add_argument(
        "--mode",
        choices=("z", "hybrid"),
        default="z",
        help="z: hysteresis on the z-score series itself (thresholds in z units); "
        "hybrid: absolute-spread hysteresis gated by |z|>=ze (thresholds in spread units)",
    )
    args = ap.parse_args()
    split = datetime.fromisoformat(args.split).replace(tzinfo=UTC)

    perp, spot, fund = load_carry(data_dir=pathlib.Path(args.data))
    print("sids:", perp["security_id"].n_unique(), "| fund:", fund.height, flush=True)

    z_lookbacks = (20, 60, 120)
    if args.mode == "hybrid":
        ze_levels = (1.0, 1.5, 2.0)
        z_frames = {
            (zl, ze): _z_funding_hybrid(fund, zl, ze)
            for zl, ze in itertools.product(z_lookbacks, ze_levels)
        }
    else:
        z_frames = {zl: _z_funding(fund, zl, sign=args.sign) for zl in z_lookbacks}
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
    mid = datetime(2024, 1, 1, tzinfo=UTC)  # same fixed dev-halves boundary

    GRID = []
    if args.mode == "hybrid":
        for zl, ze, enter, zx, lb in itertools.product(
            z_lookbacks,
            (1.0, 1.5, 2.0),
            (5e-4, 1e-3, 2e-3),
            (-1.25e-4, 0.0),
            (9,),
        ):
            GRID.append(
                dict(
                    z_lookback=zl,
                    z_gate=ze,
                    enter_rate=enter,
                    exit_rate=zx,
                    lookback_events=lb,
                    name_weight=0.08,
                    max_names=30,
                    rebalance_band=1.3,
                    rate_scale_ref=2e-3,
                    rate_scale_cap=1.5,
                    rate_scale_floor=1.0,
                )
            )
    else:
        for zl, ze, zx, lb, rsr in itertools.product(
            z_lookbacks,
            (1.0, 1.5, 2.0, 2.5),
            (-0.5, 0.0, 0.5),
            (3, 9),
            (2.0, 3.0),
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

    def _zkey(g):
        return (g["z_lookback"], g["z_gate"]) if args.mode == "hybrid" else g["z_lookback"]

    stage = {}
    for i, g in enumerate(GRID):
        cfg_kw = {k: v for k, v in g.items() if k not in ("z_lookback", "z_gate")}
        zdev = dev_z[_zkey(g)]
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
    champ_kw = {k: v for k, v in champ.items() if k not in ("z_lookback", "z_gate")}
    zfull = z_frames[_zkey(champ)]
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
