"""Delta-neutral funding-carry experiments on Binance archive data.

Dev/holdout split at 2025-01-01 UTC (same convention as equity research).
Funding events are aggregated to the daily bar timestamp (sum of the day's
rates) because the engine applies funding only at bar boundaries.

Screening runs use the real ``run_carry_backtest`` engine — costs on both
legs, wick-paranoid liquidation, leverage cap — so results are engine-grade.
"""

from __future__ import annotations

import itertools
import json
import pathlib
from datetime import datetime

import polars as pl

from quant_fund.backtest.carry_engine import run_carry_backtest
from quant_fund.backtest.sleeves import basis_carry_hysteresis_weights
from quant_fund.config.models import AppConfig
from quant_fund.risk.overlay import BookRiskOverlay

DATA = pathlib.Path("data/binance_carry")
SPLIT = datetime(2025, 1, 1, tzinfo=__import__("datetime").timezone.utc)


class OverlayAdapter:
    """BookRiskOverlay -> perp_engine.OverlayScaler protocol."""

    def __init__(self, **kw):
        self.ov = BookRiskOverlay(**kw)

    def observe(self, dt, nav):
        self.ov.observe(nav)

    def scale(self, dt, targets):
        s = self.ov.preview_scale()
        if s <= 0:
            return {}
        return {k: v * s for k, v in targets.items()}


def load_carry(extra_dir: pathlib.Path | str | None = None):
    if extra_dir is not None:
        extra_dir = pathlib.Path(extra_dir)

    def _bars(name):
        frames = [pl.read_parquet(DATA / name)]
        if extra_dir is not None and (extra_dir / name).exists():
            frames.append(pl.read_parquet(extra_dir / name))
        return (
            pl.concat(frames)
            .unique(["event_time", "security_id"])
            .sort(["event_time", "security_id"])
        )

    perp = _bars("perp_bars.parquet")
    spot = _bars("spot_bars.parquet")
    fund = _bars("funding.parquet")
    # aggregate the day's funding events into the daily bar timestamp
    fund = (
        fund.with_columns(pl.col("event_time").dt.truncate("1d").alias("event_time"))
        .group_by(["event_time", "security_id"])
        .agg(pl.col("value").sum())
        .sort(["event_time", "security_id"])
    )
    return perp, spot, fund


def make_cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.risk_gate.max_name = 0.5  # pair book: weight is pair fraction, not directional
    cfg.risk_gate.max_predicted_vol = 3.0  # crypto daily vol >> equity 0.4 gate
    cfg.risk_gate.max_order_notional = 1e9
    cfg.risk_gate.max_gross = 10.0
    cfg.risk_gate.max_net = 10.0
    cfg.kill_switch.allow_auto_flatten = False
    return cfg


def run_one(
    perp,
    spot,
    fund,
    *,
    enter,
    exit_,
    lb,
    nw,
    mx,
    band,
    scaler_kw=None,
    rsr=None,
    rsc=2.0,
    rsf=0.3,
    vlb=None,
    vr=0.04,
):
    w = basis_carry_hysteresis_weights(
        perp,
        fund,
        enter_rate=enter,
        exit_rate=exit_,
        lookback_events=lb,
        name_weight=nw,
        max_names=mx,
        rebalance_band=band,
        rate_scale_ref=rsr,
        rate_scale_cap=rsc,
        rate_scale_floor=rsf,
        vol_lookback=vlb,
        vol_ref=vr,
    )
    scaler = OverlayAdapter(**scaler_kw) if scaler_kw else None
    res = run_carry_backtest(perp, spot, fund, w, make_cfg(), initial_nav=1e6, scaler=scaler)
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
    return {k: m.get(k) for k in keep}, w, res


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="grid", choices=["grid", "champion"])
    ap.add_argument(
        "--extra-dir",
        default=None,
        help="second parquet dir merged into the universe (e.g. data/binance_carry_extra)",
    )
    args = ap.parse_args()

    perp, spot, fund = load_carry(args.extra_dir)
    print("coins:", fund["security_id"].n_unique(), "fund rows:", fund.height)
    dev_end = SPLIT
    dev_p = perp.filter(pl.col("event_time") < dev_end)
    dev_s = spot.filter(pl.col("event_time") < dev_end)
    dev_f = fund.filter(pl.col("event_time") < dev_end)
    print("dev bars:", dev_p.height, "dev fund:", dev_f.height)

    if args.mode == "champion":
        tag = "_expanded" if args.extra_dir else ""
        return run_champion(perp, spot, fund, dev_p, dev_s, dev_f, dev_end, tag=tag)
    return run_grid(dev_p, dev_s, dev_f)


def eligible_coins(perp: pl.DataFrame, spot: pl.DataFrame, max_gap: int = 3) -> set:
    """Coins whose paired (perp ∩ spot) marks have no run of >max_gap missing bars.

    A held pair whose marks go stale beyond ``stale_price_bars`` aborts the
    engine, so the universe is restricted to continuously-listed names.
    """
    import numpy as np

    j = perp.select("event_time", "security_id").join(
        spot.select("event_time", "security_id"), on=["event_time", "security_id"]
    )
    ts = sorted(j["event_time"].unique().to_list())
    idx = {t: i for i, t in enumerate(ts)}
    last_i = len(ts) - 1
    keep: set[str] = set()
    for sid, sub in j.group_by("security_id"):
        have = np.zeros(len(ts), dtype=bool)
        for t in sub["event_time"].to_list():
            have[idx[t]] = True
        nz = np.nonzero(have)[0]
        # must still be listed at window end; gaps only inside its own span
        if nz.size == 0 or nz[-1] < last_i:
            continue
        span = have[nz[0] : nz[-1] + 1]
        mx = cur = 0
        for v in span:
            cur = cur + 1 if not v else 0
            mx = max(mx, cur)
        if mx <= max_gap:
            keep.add(str(sid[0] if isinstance(sid, tuple) else sid))
    return keep


def run_champion(perp, spot, fund, dev_p, dev_s, dev_f, dev_end, tag="") -> int:
    """Evaluate the dev-selected config on dev, holdout, and full windows."""
    champ = dict(enter=0.00015, exit_=0.0, lb=9, nw=0.12, mx=15, band=1.3)
    if tag:
        # expanded 355-coin universe pick (docs/carry_expansion_2026_09.md)
        champ.update(enter=0.0002, nw=0.12, mx=60, rsr=0.002, rsc=1.5, rsf=1.0)
    out = {}
    for label, (p, s, f) in {
        "dev": (dev_p, dev_s, dev_f),
        "holdout": (
            perp.filter(pl.col("event_time") >= dev_end),
            spot.filter(pl.col("event_time") >= dev_end),
            fund.filter(pl.col("event_time") >= dev_end),
        ),
        "full": (perp, spot, fund),
    }.items():
        if p.height == 0:
            out[label] = {"skipped": "no bars"}
            continue
        keep = eligible_coins(p, s)
        p = p.filter(pl.col("security_id").is_in(keep))
        s = s.filter(pl.col("security_id").is_in(keep))
        f = f.filter(pl.col("security_id").is_in(keep))
        print(label, "eligible coins:", len(keep))
        m, w, r = run_one(p, s, f, **champ)
        out[label] = m
        print(label, json.dumps(m, default=str), flush=True)
        if label == "full":
            r.equity.write_parquet(f"artifacts/carry_equity{tag}_full.parquet")
            w.write_parquet(f"artifacts/carry_weights{tag}_full.parquet")
    pathlib.Path("artifacts").mkdir(exist_ok=True)
    pathlib.Path(f"artifacts/carry_champion{tag}.json").write_text(
        json.dumps(out, indent=2, default=str)
    )
    print(f"wrote artifacts/carry_champion{tag}.json + equity/weights parquet")
    return 0


def run_grid(dev_p, dev_s, dev_f) -> int:

    grid = []
    for enter, lb, nw, mx, band in itertools.product(
        (0.0002, 0.0003, 0.0005), (3, 9), (0.05, 0.08, 0.10), (10, 15), (1.3, 1.5)
    ):
        grid.append(dict(enter=enter, exit_=0.0, lb=lb, nw=nw, mx=mx, band=band))
    out = {}
    for g in grid:
        key = f"e{g['enter']}_lb{g['lb']}_nw{g['nw']}_mx{g['mx']}_b{g['band']}"
        try:
            m, w, r = run_one(dev_p, dev_s, dev_f, **g)
            out[key] = m
            print(key, json.dumps(m, default=str), flush=True)
        except Exception as e:  # noqa: BLE001
            print(key, "ERR", type(e).__name__, e, flush=True)
    pathlib.Path("artifacts").mkdir(exist_ok=True)
    pathlib.Path("artifacts/carry_dev_grid.json").write_text(json.dumps(out, indent=2, default=str))
    print("wrote artifacts/carry_dev_grid.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
