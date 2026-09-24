"""C5 optional ML sleeve: dev-trained funding predictor -> frozen holdout eval.

Megaplan Phase C5. A GradientBoostingRegressor predicts each symbol's NEXT
funding event value from strictly-past features (funding lags/rolling stats,
perp momentum and realized vol as-of the event time). Trained ONLY on dev
events (< split); weights emit `-pred_rate / vol` demeaned cross-sectionally
through the same `_cap_and_emit` shape as `funding_carry_weights`, then the
megaplan Evaluator runs the identical dev-half scoring -> freeze -> one
locked holdout eval. Failure is a result.

Usage: uv run python scripts/c5_ml_carry.py --data-dir data/binance_carry_1h
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from megaplan_eval import Evaluator, _score

from quant_fund.backtest.sleeves import _cap_and_emit, _per_symbol_vol
from quant_fund.config.models import AppConfig

SPLIT = datetime(2025, 1, 1, tzinfo=UTC)

FEATURES = ("f_lag1", "f_ma3", "f_ma9", "f_sd9", "ret24", "ret168", "vol168")


def _features(bars: pl.DataFrame, funding: pl.DataFrame) -> pl.DataFrame:
    """Per funding event: funding history + as-of perp momentum/vol."""
    f = funding.sort(["security_id", "event_time"]).with_columns(
        pl.col("value").shift(1).over("security_id").alias("f_lag1"),
        pl.col("value").rolling_mean(3, min_samples=1).over("security_id").alias("f_ma3"),
        pl.col("value").rolling_mean(9, min_samples=1).over("security_id").alias("f_ma9"),
        pl.col("value").rolling_std(9, min_samples=2).over("security_id").alias("f_sd9"),
        pl.col("value").shift(-1).over("security_id").alias("target"),
    )
    px = (
        bars.sort(["security_id", "event_time"])
        .with_columns(
            pl.col("close").log().diff(24).over("security_id").alias("ret24"),
            pl.col("close").log().diff(168).over("security_id").alias("ret168"),
            pl.col("close")
            .pct_change()
            .rolling_std(168, min_samples=24)
            .over("security_id")
            .alias("vol168"),
        )
        .select("security_id", "event_time", "ret24", "ret168", "vol168")
        .sort("security_id", "event_time")
    )
    return (
        f.sort("security_id", "event_time")
        .join_asof(px, on="event_time", by="security_id", strategy="backward")
        .drop_nulls("f_lag1")
    )


def _ml_weights(
    bars: pl.DataFrame,
    preds: pl.DataFrame,
    *,
    max_name: float,
    gross_scale: float,
    vol_window: int = 168,
) -> pl.DataFrame:
    """Same shape as funding_carry_weights with rate_ma <- predicted rate."""
    grid = bars.select("security_id", "event_time", "close").sort(["security_id", "event_time"])
    vols = _per_symbol_vol(bars, vol_window)
    joined = (
        grid.join_asof(
            preds.select("security_id", "event_time", "pred").sort("security_id", "event_time"),
            on="event_time",
            by="security_id",
            strategy="backward",
        )
        .join(vols, on=["security_id", "event_time"], how="left")
        .with_columns(
            pl.when(pl.col("_vol").is_not_null() & (pl.col("_vol") > 0))
            .then(-pl.col("pred") / pl.col("_vol"))
            .otherwise(None)
            .alias("_raw")
        )
        .with_columns((pl.col("_raw") - pl.col("_raw").median().over("event_time")).alias("_raw"))
    )
    return _cap_and_emit(joined, "_raw", max_name=max_name, gross_scale=gross_scale)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/binance_carry_1h"))
    ap.add_argument("--split", default=None)
    ap.add_argument("--out", default="artifacts/c5_ml_carry.json")
    args = ap.parse_args()
    split = datetime.fromisoformat(args.split).replace(tzinfo=UTC) if args.split else SPLIT

    from sklearn.ensemble import GradientBoostingRegressor

    bars = pl.read_parquet(args.data_dir / "perp_bars.parquet")
    funding = pl.read_parquet(args.data_dir / "funding.parquet")
    cfg = AppConfig()
    ev = Evaluator(bars, funding, cfg, 1e6, vol_window=168, ppy=8766.0)

    feats = _features(bars, funding)
    train = feats.filter(pl.col("event_time") < split).drop_nulls(
        subset=list(FEATURES) + ["target"]
    )
    print(f"feature rows: {feats.height} | train rows: {train.height}", flush=True)

    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=7
    )
    model.fit(train.select(*FEATURES).to_numpy(), train["target"].to_numpy())

    # Predict on every event with full features (dev + holdout; features are
    # strictly-past so holdout predictions are causal).
    scored = feats.drop_nulls(subset=list(FEATURES)).with_columns(
        pl.lit(
            model.predict(feats.drop_nulls(subset=list(FEATURES)).select(*FEATURES).to_numpy())
        ).alias("pred")
    )
    print(
        "pred std dev:",
        float(scored.filter(pl.col("event_time") < split)["pred"].std()),
        "| holdout:",
        float(scored.filter(pl.col("event_time") >= split)["pred"].std()),
    )

    times = ev.bar_times
    t0, t1 = times[0], times[-1]
    dev_times = sorted(bars.filter(pl.col("event_time") < split)["event_time"].unique().to_list())
    mid = dev_times[len(dev_times) // 2]

    grid = {
        f"mn{mn}_gs{gs}": (mn, gs)
        for mn, gs in ((0.03, 1.0), (0.05, 2.0), (0.08, 2.0), (0.05, 4.0))
    }
    stage = {}
    for name, (mn, gs) in grid.items():
        w = _ml_weights(bars, scored, max_name=mn, gross_scale=gs)
        h1 = ev.run(w, t0, mid, None)
        h2 = ev.run(w, mid, split, None)
        stage[name] = {
            "score": _score(h1, h2, mdd_gate=False),
            "h1": h1,
            "h2": h2,
            "cfg": {"max_name": mn, "gross_scale": gs},
        }
        print(f"{name}: score={stage[name]['score']:.3f}", flush=True)

    best = max(stage, key=lambda k: stage[k]["score"])
    champ = stage[best]["cfg"]
    w = _ml_weights(bars, scored, max_name=champ["max_name"], gross_scale=champ["gross_scale"])
    qualified = stage[best]["score"] > float("-inf")
    out = {"stage": stage, "frozen": {"id": best, **champ, "qualified_on_dev": qualified}}
    from datetime import timedelta

    for label, lo, hi in (
        ("dev", t0, split),
        ("holdout", split, t1 + timedelta(seconds=ev.step_s)),
    ):
        m = ev.run(w, lo, hi, 0.1)
        out[label] = m
        print(label, m, flush=True)
    h_sharpe = out["holdout"].get("sharpe")
    h_mdd = out["holdout"].get("max_drawdown")
    out["gate"] = {
        "PROVEN": bool(qualified and h_sharpe and h_sharpe > 5.0 and h_mdd and abs(h_mdd) < 0.05)
    }
    Path("artifacts").mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print("GATE:", out["gate"], "| receipt ->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
