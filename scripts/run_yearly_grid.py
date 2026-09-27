import itertools
import json
import pathlib
import sys

sys.path.insert(0, ".")
from datetime import UTC, datetime

import polars as pl
from carry_research import eligible_coins, load_carry, run_one

perp, spot, fund = load_carry(data_dir="../data/binance_carry")
E = datetime(1970, 1, 1, tzinfo=UTC)
DEV_END = datetime(2025, 1, 1, tzinfo=UTC)
dev_p = perp.filter(pl.col("event_time") < DEV_END)
dev_s = spot.filter(pl.col("event_time") < DEV_END)
dev_f = fund.filter(pl.col("event_time") < DEV_END)
keep = eligible_coins(dev_p, dev_s)


def fl(d):
    return d.filter(pl.col("security_id").is_in(keep))


dev_p, dev_s, dev_f = fl(dev_p), fl(dev_s), fl(dev_f)
years = [
    (datetime(y, 1, 1, tzinfo=UTC), datetime(y + 1, 1, 1, tzinfo=UTC))
    for y in (2021, 2022, 2023, 2024)
]
yr = [
    (
        dev_p.filter(pl.col("event_time").is_between(a, b)),
        dev_s.filter(pl.col("event_time").is_between(a, b)),
        dev_f.filter(pl.col("event_time").is_between(a, b)),
    )
    for a, b in years
]


def sc(m):
    if not m or m.get("n", 0) < 15:
        return -999.0
    return (m.get("sharpe") or -999.0) - 0.5 * (m.get("liquidation_count") or 0)


GRID = [
    dict(enter=e, exit_=x, lb=lb, nw=nw, mx=mx, band=1.3, rsr=2e-3, rsc=1.5, rsf=1.0)
    for e, x, lb, nw, mx in itertools.product(
        (2e-4, 3e-4, 5e-4), (-1.25e-4, 0.0), (3, 9), (0.08, 0.12), (15, 30, 60)
    )
]
res = {}
for i, g in enumerate(GRID):
    ys = []
    for p, s, f in yr:
        if p.height == 0:
            ys.append(None)
            continue
        try:
            m, _, _ = run_one(p, s, f, **g)
            ys.append(round(sc(m), 2))
        except Exception:
            ys.append(None)
    md, _, _ = run_one(dev_p, dev_s, dev_f, **g)
    real = [y for y in ys if y is not None]
    score = min(real) if real else -999
    res[f"g{i}"] = {"cfg": g, "yr": ys, "dev": md.get("sharpe"), "minyr": score}
    print(f"g{i} yr={ys} dev={md.get('sharpe'):.2f} min={score:.2f}", flush=True)
pathlib.Path("../artifacts").mkdir(exist_ok=True)
pathlib.Path("../artifacts/binance144_yrgrid.json").write_text(
    json.dumps(res, indent=2, default=str)
)
top = sorted((k for k in res if res[k]["minyr"] > -900), key=lambda k: -res[k]["minyr"])[:8]
print("TOP worst-year:")
for k in top:
    print(k, res[k])
# frozen eval of the top worst-year config on holdout
g = res[top[0]]["cfg"]
hp = perp.filter(pl.col("event_time") >= DEV_END)
hs = spot.filter(pl.col("event_time") >= DEV_END)
hf = fund.filter(pl.col("event_time") >= DEV_END)
kh = eligible_coins(hp, hs)


def fl2(d):
    return d.filter(pl.col("security_id").is_in(kh))


mh, _, _ = run_one(fl2(hp), fl2(hs), fl2(hf), **g)
print("WORST-YEAR champ", g)
print("holdout", json.dumps(mh, default=str))
