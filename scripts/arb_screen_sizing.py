"""Sizing re-tune on the frozen top-120 funding-mass universe.

topk_mass:120 froze the universe (dev screen) and lifted holdout to 4.02.
On the same frozen universe, re-sweep the sizing axes (nw, mx, enter, lb,
band, vr) — dev-only selection (min(h1,h2)+0.01*dev), freeze, one holdout
eval. Also tries a per-name dev-Sharpe screen (topk by funding mass / spread
vol) as an alternative frozen universe.

Writes artifacts/arb_screen_sizing_{grid,champion}.json.
"""

from __future__ import annotations

import itertools
import json
import pathlib
from datetime import UTC, datetime

import polars as pl
from carry_research import eligible_coins, load_carry, run_one

DATA = pathlib.Path("data/arb3_book")
DEV_END = datetime(2025, 1, 1, tzinfo=UTC)
DEV_MID = datetime(2024, 1, 1, tzinfo=UTC)

BASE = dict(
    enter=1e-3,
    exit_=-1.25e-4,
    lb=9,
    nw=0.08,
    mx=30,
    band=1.3,
    rsr=2e-3,
    rsc=1.5,
    rsf=1.0,
    rexp=0.0,
    vlb=None,
    vr=0.04,
)

GRID = [
    dict(nw=nw, mx=mx, enter=1e-3, lb=9, band=band, vr=0.04, vlb=None, rexp=rexp)
    for nw, mx, band, rexp in itertools.product(
        (0.08, 0.12, 0.2, 0.3),
        (20, 30, 45),
        (1.3, 1.6),
        (0.0, 1.0),
    )
]
# vol_ref axis on the frozen universe (never swept with vlb on)
GRID += [
    dict(nw=0.08, mx=30, enter=1e-3, lb=9, band=1.3, vr=vr, vlb=30, rexp=0.0)
    for vr in (0.02, 0.04, 0.08, 0.16)
]


def _topk_mass(fund_dev: pl.DataFrame, k: int) -> set[str]:
    g = fund_dev.group_by("security_id").agg(tot=pl.col("value").sum())
    return set(g.sort("tot", descending=True).head(k)["security_id"].to_list())


def _topk_sharpe(fund_dev: pl.DataFrame, k: int) -> set[str]:
    g = fund_dev.group_by("security_id").agg(mu=pl.col("value").mean(), sd=pl.col("value").std())
    g = g.with_columns(sr=pl.col("mu") / (pl.col("sd") + 1e-12))
    return set(g.sort("sr", descending=True).head(k)["security_id"].to_list())


UNIVERSES = {
    "mass120": lambda f: _topk_mass(f, 120),
    "sharpe120": lambda f: _topk_sharpe(f, 120),
}


def _slice(p, s, f, lo, hi):
    def f_(df):
        return df.filter((pl.col("event_time") >= lo) & (pl.col("event_time") < hi))

    return f_(p), f_(s), f_(f)


def _score(m: dict) -> float:
    if not m or m.get("n", 0) < 30:
        return -999.0
    pen = 0.5 * (m.get("liquidation_count") or 0) + 0.001 * (m.get("margin_rejects") or 0)
    return (m.get("sharpe") or -999.0) - pen


def _eval_on(p, s, f, names, cfg):
    keep = names & eligible_coins(p, s)
    return run_one(
        p.filter(pl.col("security_id").is_in(keep)),
        s.filter(pl.col("security_id").is_in(keep)),
        f.filter(pl.col("security_id").is_in(keep)),
        **cfg,
    )[0]


def main() -> int:
    perp, spot, fund = load_carry(data_dir=DATA)
    dev_p, dev_s, dev_f = _slice(perp, spot, fund, datetime(1970, 1, 1, tzinfo=UTC), DEV_END)
    h2 = _slice(dev_p, dev_s, dev_f, DEV_MID, DEV_END)

    results = {}
    for uname, fn in UNIVERSES.items():
        # universe screened on each window's own data for dev-internal scoring
        names_h2 = fn(h2[2])
        names_dev = fn(dev_f)
        for i, g in enumerate(GRID):
            key = f"{uname}:g{i}"
            cfg = BASE | g
            try:
                # h2 eval uses the h2-screened universe (out-of-screen-train)
                m2 = _eval_on(*h2, names_h2, cfg)
                md = _eval_on(dev_p, dev_s, dev_f, names_dev, cfg)
            except Exception as e:  # noqa: BLE001
                results[key] = {"error": f"{type(e).__name__}: {e}"}
                continue
            sc = min(_score(m2), _score(md) * 1.01)
            results[key] = {
                "cfg": cfg,
                "h2": m2.get("sharpe"),
                "dev": md.get("sharpe"),
                "score": sc,
            }
            if i % 20 == 0:
                print(
                    f"{key} h2={m2.get('sharpe')} dev={md.get('sharpe')} sc={sc:.3f}",
                    flush=True,
                )

    pathlib.Path("artifacts").mkdir(exist_ok=True)
    pathlib.Path("artifacts/arb_screen_sizing_grid.json").write_text(
        json.dumps(results, indent=2, default=str)
    )

    best_key = max(
        (k for k in results if "score" in results[k]),
        key=lambda k: results[k]["score"],
        default=None,
    )
    if best_key is None:
        print("no config qualified — stop")
        return 1
    uname = best_key.split(":")[0]
    champ = results[best_key]["cfg"]
    names_dev = UNIVERSES[uname](dev_f)
    print("frozen:", best_key, "names:", len(names_dev), champ)

    out = {
        "frozen": best_key,
        "frozen_config": champ,
        "universe": uname,
        "n_names": len(names_dev),
    }
    for label, (p, s, f) in {
        "dev": (dev_p, dev_s, dev_f),
        "holdout": _slice(perp, spot, fund, DEV_END, datetime(2100, 1, 1, tzinfo=UTC)),
        "full": (perp, spot, fund),
    }.items():
        m = _eval_on(p, s, f, names_dev, champ)
        out[label] = m
        print(label, json.dumps(m, default=str), flush=True)

    pathlib.Path("artifacts/arb_screen_sizing_champion.json").write_text(
        json.dumps(out, indent=2, default=str)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
