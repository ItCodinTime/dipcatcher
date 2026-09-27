"""Direction-asymmetric entry + direction-restricted universes on the arb book.

The h↔b pair is antisymmetric: h>b sids pay (dev mean +1.97e-4, 11.7% of
events above enter=1e-3) while b>h sids mostly lose (−1.97e-4, 4.7%). Two
questions on the frozen top-120 funding-mass universe:
  1. Does a higher enter bar on the losing direction (enter_rate_by_prefix)
     lift the book?
  2. Does restricting the universe to the paying direction beat the
     symmetric book?

Frozen universe: topk dev funding-mass names (the 4.02 lane). Dev-only
selection (min(h2, dev*1.01) as in arb_elig_screen), freeze, one holdout
eval. Baseline (symmetric enter=1e-3) is a candidate — asymmetry is adopted
only if it beats the status quo on dev.

Writes artifacts/arb_asym_enter_{grid,champion}.json.
"""

from __future__ import annotations

import json
import pathlib
from datetime import UTC, datetime

import polars as pl
from carry_research import eligible_coins, load_carry, run_one

DATA = pathlib.Path("data/arb3_book")
DEV_END = datetime(2025, 1, 1, tzinfo=UTC)
DEV_MID = datetime(2024, 1, 1, tzinfo=UTC)

BASE = dict(enter=1e-3, exit_=-1.25e-4, lb=9, nw=0.08, mx=30, band=1.3, rsr=2e-3, rsc=1.5, rsf=1.0)


def _topk_mass(fund_dev: pl.DataFrame, k: int, dirs: tuple[str, ...]) -> set[str]:
    f = fund_dev
    if dirs:
        pat = "^AX:(?:" + "|".join(dirs) + "):"
        f = f.filter(pl.col("security_id").str.contains(pat))
    g = f.group_by("security_id").agg(tot=pl.col("value").sum())
    return set(g.sort("tot", descending=True).head(k)["security_id"].to_list())


UNIVERSES = {
    "mass120": lambda f: _topk_mass(f, 120, ("b>h", "h>b")),
    "hb120": lambda f: _topk_mass(f, 120, ("h>b",)),
    "hb60": lambda f: _topk_mass(f, 60, ("h>b",)),
    "hb30": lambda f: _topk_mass(f, 30, ("h>b",)),
}

EBPS = [None]
EBPS += [{"AX:b>h:": eb, "AX:h>b:": 1e-3} for eb in (2e-3, 4e-3, 8e-3)]
EBPS += [{"AX:b>h:": 2e-3, "AX:h>b:": eh} for eh in (5e-4, 2e-3)]


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
        names_h2 = fn(h2[2])
        names_dev = fn(dev_f)
        if len(names_dev) < 10:
            continue
        for i, ebp in enumerate(EBPS):
            key = f"{uname}:e{i}"
            cfg = BASE | {"ebp": ebp}
            try:
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
                "n_dev_names": len(names_dev),
            }
            print(
                f"{key} h2={m2.get('sharpe')} dev={md.get('sharpe')} sc={sc:.3f}",
                flush=True,
            )

    pathlib.Path("artifacts").mkdir(exist_ok=True)
    pathlib.Path("artifacts/arb_asym_enter_grid.json").write_text(
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

    pathlib.Path("artifacts/arb_asym_enter_champion.json").write_text(
        json.dumps(out, indent=2, default=str)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
