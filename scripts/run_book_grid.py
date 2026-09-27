import itertools
import json
import pathlib
import sys

sys.path.insert(0, ".")
from datetime import UTC, datetime

import polars as pl
from carry_research import eligible_coins, load_carry, run_one

BOOK = sys.argv[1] if len(sys.argv) > 1 else "data/binance_carry"
TAG = sys.argv[2] if len(sys.argv) > 2 else "binance144"

perp, spot, fund = load_carry(data_dir="../" + BOOK)
DEV_END = datetime(2025, 1, 1, tzinfo=UTC)
DEV_MID = datetime(2024, 1, 1, tzinfo=UTC)
E = datetime(1970, 1, 1, tzinfo=UTC)


def sl(p, s, f, lo, hi):
    def g(d):
        return d.filter((pl.col("event_time") >= lo) & (pl.col("event_time") < hi))

    return g(p), g(s), g(f)


dev_p, dev_s, dev_f = sl(perp, spot, fund, E, DEV_END)
keep = eligible_coins(dev_p, dev_s)


def fil(d):
    return d.filter(pl.col("security_id").is_in(keep))


dev_p, dev_s, dev_f = fil(dev_p), fil(dev_s), fil(dev_f)
h1 = sl(dev_p, dev_s, dev_f, E, DEV_MID)
h2 = sl(dev_p, dev_s, dev_f, DEV_MID, DEV_END)
print("eligible", len(keep), flush=True)

GRID = [
    dict(enter=e, exit_=x, lb=lb, nw=nw, mx=mx, band=1.3, rsr=2e-3, rsc=1.5, rsf=1.0)
    for e, x, lb, nw, mx in itertools.product(
        (2e-4, 3e-4, 5e-4), (-1.25e-4, 0.0), (3, 9), (0.08, 0.12), (15, 30, 60)
    )
]


def score(m):
    if not m or m.get("n", 0) < 30:
        return -999.0
    return (
        (m.get("sharpe") or -999.0)
        - 0.5 * (m.get("liquidation_count") or 0)
        - 0.001 * (m.get("margin_rejects") or 0)
    )


res = {}
for i, g in enumerate(GRID):
    try:
        m1, _, _ = run_one(*h1, **g)
        m2, _, _ = run_one(*h2, **g)
        md, _, _ = run_one(dev_p, dev_s, dev_f, **g)
    except Exception as e:
        res[f"g{i}"] = {"cfg": g, "error": str(e)[:80]}
        continue
    sc = min(score(m1), score(m2)) + 0.01 * (md.get("sharpe") or 0)
    res[f"g{i}"] = {
        "cfg": g,
        "h1": m1.get("sharpe"),
        "h2": m2.get("sharpe"),
        "dev": md.get("sharpe"),
        "score": sc,
    }
    print(
        f"g{i} h1={res[f'g{i}']['h1']:.2f} h2={res[f'g{i}']['h2']:.2f} dev={res[f'g{i}']['dev']:.2f} sc={sc:.2f}",
        flush=True,
    )

pathlib.Path("artifacts").mkdir(exist_ok=True)
pathlib.Path(f"artifacts/{TAG}_grid.json").write_text(json.dumps(res, indent=2, default=str))
bk = max((k for k in res if "score" in res[k]), key=lambda k: res[k]["score"], default=None)
print("champion", bk, res[bk])
out = {}
for label, (p, s, f) in {
    "dev": (dev_p, dev_s, dev_f),
    "holdout": sl(perp, spot, fund, DEV_END, datetime(2100, 1, 1, tzinfo=UTC)),
    "full": (perp, spot, fund),
}.items():
    kl = eligible_coins(p, s)

    def fl(d, kl=kl):
        return d.filter(pl.col("security_id").is_in(kl))

    m, w, r = run_one(fl(p), fl(s), fl(f), **res[bk]["cfg"])
    out[label] = m
    out[label]["eligible"] = len(kl)
    print(label, json.dumps(m, default=str), flush=True)
out["frozen_config"] = res[bk]["cfg"]
out["book"] = BOOK
pathlib.Path(f"artifacts/{TAG}_champion.json").write_text(json.dumps(out, indent=2, default=str))
hs = out["holdout"].get("sharpe") or 0
dd = out["holdout"].get("max_drawdown") or -1
print(
    "GATE holdout>5 & DD<5%:",
    "PASS" if (hs > 5 and dd > -0.05) else "NOT PROVEN",
    f"(sharpe={hs:.2f}, dd={dd:.3f})",
)
