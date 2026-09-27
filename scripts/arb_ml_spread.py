"""ML spread-entry variant for the multi-venue arb book (megaplan closer lane).

Same honesty protocol as arb_sharpe5_grid: a GradientBoostingRegressor trained
on DEV events only predicts each pair-sid's NEXT-day funding spread from
strictly-past features (spread lags/rolling stats + perp-leg momentum/vol).
Predictions become a synthetic funding frame — one event per sid per day at
00:00 of the predicted day — consumed by ``basis_carry_hysteresis_weights``:
membership/sizing keys off PREDICTED spreads, while P&L is settled on the REAL
spread frame. Grid over predicted-enter thresholds freezes on dev halves
(min(h1,h2) score), then ONE locked holdout eval. Failure is a result.

Usage: uv run python scripts/arb_ml_spread.py --data data/arb3_book
Writes artifacts/arb_ml_spread.json.
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

FEATURES = ("s_lag1", "s_ma3", "s_ma9", "s_sd9", "s_hi9", "ret24", "vol24")


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


def _features(perp: pl.DataFrame, fund: pl.DataFrame) -> pl.DataFrame:
    """Per spread-event row: spread history + perp-leg mark momentum."""
    f = fund.sort(["security_id", "event_time"]).with_columns(
        pl.col("value").shift(1).over("security_id").alias("s_lag1"),
        pl.col("value").rolling_mean(3, min_samples=1).over("security_id").alias("s_ma3"),
        pl.col("value").rolling_mean(9, min_samples=1).over("security_id").alias("s_ma9"),
        pl.col("value").rolling_std(9, min_samples=2).over("security_id").alias("s_sd9"),
        pl.col("value").rolling_max(9, min_samples=1).over("security_id").alias("s_hi9"),
        pl.col("value").shift(-1).over("security_id").alias("target"),
    )
    px = (
        perp.sort(["security_id", "event_time"])
        .with_columns(
            pl.col("close").log().diff(1).over("security_id").alias("ret24"),
            pl.col("close")
            .pct_change()
            .rolling_std(24, min_samples=8)
            .over("security_id")
            .alias("vol24"),
        )
        .select("security_id", "event_time", "ret24", "vol24")
    )
    return (
        f.sort("security_id", "event_time")
        .join_asof(px, on="event_time", by="security_id", strategy="backward")
        .drop_nulls("s_lag1")
    )


def _synth_funding(scored: pl.DataFrame) -> pl.DataFrame:
    """Predicted spread for day d+1 lands at 00:00 of day d+1 (load_carry
    normalizes realized events to 00:00)."""
    return scored.select(
        (pl.col("event_time") + timedelta(days=1)).alias("event_time"),
        pl.col("security_id"),
        pl.col("pred").alias("value"),
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
    ap.add_argument("--out", default="artifacts/arb_ml_spread.json")
    args = ap.parse_args()
    split = datetime.fromisoformat(args.split).replace(tzinfo=UTC)

    perp, spot, fund = load_carry(data_dir=pathlib.Path(args.data))
    print("sids:", perp["security_id"].n_unique(), "| fund:", fund.height, flush=True)

    feats = _features(perp, fund)
    train = feats.filter(pl.col("event_time") < split).drop_nulls(
        subset=list(FEATURES) + ["target"]
    )
    print("feature rows:", feats.height, "| train:", train.height, flush=True)

    from sklearn.ensemble import GradientBoostingRegressor

    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=7,
    )
    model.fit(train.select(*FEATURES).to_numpy(), train["target"].to_numpy())
    ok = feats.drop_nulls(subset=list(FEATURES))
    scored = ok.with_columns(pl.lit(model.predict(ok.select(*FEATURES).to_numpy())).alias("pred"))
    print(
        "pred std dev:",
        float(scored.filter(pl.col("event_time") < split)["pred"].std()),
        "| holdout:",
        float(scored.filter(pl.col("event_time") >= split)["pred"].std()),
        flush=True,
    )
    syn = _synth_funding(scored)

    dev_p, dev_s, dev_f = _slice(perp, spot, fund, datetime(1970, 1, 1, tzinfo=UTC), split)
    keep_dev = eligible_coins(dev_p, dev_s)
    dev_p = dev_p.filter(pl.col("security_id").is_in(keep_dev))
    dev_s = dev_s.filter(pl.col("security_id").is_in(keep_dev))
    dev_f = dev_f.filter(pl.col("security_id").is_in(keep_dev))
    dev_syn = syn.filter(pl.col("security_id").is_in(keep_dev) & (pl.col("event_time") < split))
    mid = datetime(2024, 1, 1, tzinfo=UTC)  # same fixed dev-halves boundary as arb_sharpe5_grid

    GRID = [
        dict(
            enter_rate=e,
            exit_rate=x,
            lookback_events=lb,
            name_weight=0.08,
            max_names=mx,
            rebalance_band=1.3,
            rate_scale_ref=rsr,
            rate_scale_cap=1.5,
            rate_scale_floor=1.0,
        )
        for e, x, lb, mx, rsr in itertools.product(
            (5e-4, 1e-3, 2e-3),
            (-1.25e-4, 0.0),
            (3, 9),
            (30,),
            (2e-3,),
        )
    ]
    stage = {}
    for i, g in enumerate(GRID):
        rec = {"cfg": g}
        try:
            h1p, h1s, h1f = _slice(dev_p, dev_s, dev_f, datetime(1970, 1, 1, tzinfo=UTC), mid)
            h1y = dev_syn.filter(pl.col("event_time") < mid)
            m1 = _run(h1p, h1s, h1f, h1y, g)
            h2p, h2s, h2f = _slice(dev_p, dev_s, dev_f, mid, split)
            h2y = dev_syn.filter(pl.col("event_time") >= mid)
            m2 = _run(h2p, h2s, h2f, h2y, g)
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
            stage[f"g{i}"] = rec
            continue
        rec["h1_sharpe"], rec["h2_sharpe"] = m1.get("sharpe"), m2.get("sharpe")
        rec["score"] = min(_score(m1), _score(m2))
        stage[f"g{i}"] = rec
        print(
            f"g{i} h1={rec['h1_sharpe']} h2={rec['h2_sharpe']} score={rec['score']:.3f}", flush=True
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
        "frozen": {"id": best_key, **champ, "qualified_on_dev": done[best_key]["score"] > -999},
    }
    for label, (p, s, f) in {
        "dev": (dev_p, dev_s, dev_f),
        "holdout": _slice(perp, spot, fund, split, datetime(2100, 1, 1, tzinfo=UTC)),
    }.items():
        keep_l = eligible_coins(p, s)
        pp = p.filter(pl.col("security_id").is_in(keep_l))
        ss = s.filter(pl.col("security_id").is_in(keep_l))
        ff = f.filter(pl.col("security_id").is_in(keep_l))
        yy = syn.filter(pl.col("security_id").is_in(keep_l))
        m = _run(pp, ss, ff, yy, champ)
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
    pathlib.Path("artifacts").mkdir(exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print("GATE:", out["gate"], "| receipt ->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
