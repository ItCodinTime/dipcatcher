"""Static mass-proportional weight tilt on the frozen top-120 universe.

Champion so far: topk_mass:120 frozen universe + base config → holdout 4.02.
Unexplored: tilt emitted weights toward historically-heavier payers —
w_i *= clip((mass_i/median_mass)^p, 0.5, cap). Unlike rate_exponent (which
tilts by the CURRENT rate), this tilts by each name's dev-window funding
mass — a constant, frozen on dev.

Honest protocol as before: dev-only scoring on h2 + dev, freeze, one
holdout eval. Baseline (p=None, no tilt) is a candidate.
"""

from __future__ import annotations

import itertools
import json
import pathlib
from datetime import UTC, datetime

import polars as pl
from carry_research import eligible_coins, load_carry, make_cfg

from quant_fund.backtest.carry_engine import run_carry_backtest
from quant_fund.backtest.sleeves import basis_carry_hysteresis_weights

DATA = pathlib.Path("data/arb3_book")
DEV_END = datetime(2025, 1, 1, tzinfo=UTC)
DEV_MID = datetime(2024, 1, 1, tzinfo=UTC)

SIZER = dict(
    enter_rate=1e-3,
    exit_rate=-1.25e-4,
    lookback_events=9,
    name_weight=0.08,
    max_names=30,
    rebalance_band=1.3,
    rate_scale_ref=2e-3,
    rate_scale_cap=1.5,
    rate_scale_floor=1.0,
)


def _slice(p, s, f, lo, hi):
    q = lambda d: d.filter((pl.col("event_time") >= lo) & (pl.col("event_time") < hi))  # noqa: E731
    return q(p), q(s), q(f)


def _score(m: dict) -> float:
    if not m or m.get("n", 0) < 30:
        return -999.0
    pen = 0.5 * (m.get("liquidation_count") or 0) + 0.001 * (m.get("margin_rejects") or 0)
    return (m.get("sharpe") or -999.0) - pen


def main() -> int:
    perp, spot, fund = load_carry(data_dir=DATA)
    dev_p, dev_s, dev_f = _slice(perp, spot, fund, datetime(1970, 1, 1, tzinfo=UTC), DEV_END)
    h2 = _slice(dev_p, dev_s, dev_f, DEV_MID, DEV_END)

    g = dev_f.group_by("security_id").agg(tot=pl.col("value").sum())
    names = set(g.sort("tot", descending=True).head(120)["security_id"].to_list())
    mass = dict(zip(g["security_id"].to_list(), g["tot"].to_list(), strict=False))
    med = sorted(mass[n] for n in names)[len(names) // 2]

    def ev(p, s, f, nameset, p_, cap):
        keep = nameset & eligible_coins(p, s)
        pp = p.filter(pl.col("security_id").is_in(keep))
        ss = s.filter(pl.col("security_id").is_in(keep))
        ff = f.filter(pl.col("security_id").is_in(keep))
        w = basis_carry_hysteresis_weights(pp, ff, **SIZER)
        if p_ is not None:
            scale = {k: max(0.5, min(cap, (max(v, 0.0) / med) ** p_)) for k, v in mass.items()}
            w = w.with_columns(
                pl.when(pl.col("target_weight") > 0)
                .then(
                    pl.col("target_weight")
                    * pl.col("security_id").replace_strict(scale, default=1.0)
                )
                .otherwise(pl.col("target_weight"))
                .alias("target_weight")
            )
        cfg = make_cfg()
        res = run_carry_backtest(pp, ss, ff, w, cfg, initial_nav=1e6)
        return res.metrics

    res = {}
    grid = [(None, 1.0)] + list(itertools.product((0.5, 1.0, 1.5), (1.5, 2.0, 3.0)))
    for p_, cap in grid:
        md = ev(dev_p, dev_s, dev_f, names, p_, cap)
        m2 = ev(*h2, names, p_, cap)
        sc = min(_score(m2), _score(md) * 1.01)
        res[f"p{p_}:c{cap}"] = {
            "dev": md.get("sharpe"),
            "h2": m2.get("sharpe"),
            "score": sc,
        }
        print(
            f"p={p_} cap={cap} dev={md.get('sharpe'):.2f} h2={m2.get('sharpe'):.2f} sc={sc:.3f}",
            flush=True,
        )

    pathlib.Path("artifacts").mkdir(exist_ok=True)
    pathlib.Path("artifacts/arb_mass_tilt.json").write_text(json.dumps(res, indent=2, default=str))
    base = res["pNone:c1.0"]["score"]
    best = max(res, key=lambda k: res[k]["score"])
    if res[best]["score"] <= base:
        print("no tilt beats baseline on dev — holdout pointless")
        return 0
    p_, cap = float(best.split("p")[1].split(":")[0]), float(best.split("c")[1])
    mh = ev(*_slice(perp, spot, fund, DEV_END, datetime(2100, 1, 1, tzinfo=UTC)), names, p_, cap)
    print("FROZEN holdout:", json.dumps(mh, default=str))
    pathlib.Path("artifacts/arb_mass_tilt_champion.json").write_text(
        json.dumps({"frozen": best, "holdout": mh}, indent=2, default=str)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
