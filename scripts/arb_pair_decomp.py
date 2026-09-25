"""Venue-pair decomposition on the 3-venue arb book.

Question: does one venue pair (h↔b, h↔o, b↔o) carry the book while others
dilute? The frozen champion (enter=1e-3, exit=-1.25e-4, lb=9, nw=0.08,
mx=30, band=1.3, rsr=2e-3, rsc=1.5, rsf=1.0) is evaluated on each venue-pair
sub-book — dev-half score only. The best sub-book is frozen and evaluated
once on holdout.

Selection is dev-only; the holdout eval is a single locked run — failure is
a result. Writes artifacts/arb3_pairdecomp_{grid,champion}.json.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from datetime import UTC, datetime

import polars as pl
from carry_research import eligible_coins, load_carry, run_one

DATA = pathlib.Path("data/arb3_book")
DEV_END = datetime(2025, 1, 1, tzinfo=UTC)
DEV_MID = datetime(2024, 1, 1, tzinfo=UTC)

CHAMP = dict(
    enter=1e-3,
    exit_=-1.25e-4,
    lb=9,
    nw=0.08,
    mx=30,
    band=1.3,
    rsr=2e-3,
    rsc=1.5,
    rsf=1.0,
)

SUB_BOOKS = {
    "hb": ("b>h", "h>b"),
    "ho": ("h>o", "o>h"),
    "bo": ("b>o", "o>b"),
    "hb+ho": ("b>h", "h>b", "h>o", "o>h"),
    "hb+bo": ("b>h", "h>b", "b>o", "o>b"),
    "ho+bo": ("h>o", "o>h", "b>o", "o>b"),
}


def _pair_filter(dirs: tuple[str, ...]) -> pl.Expr:
    pat = "^AX:(?:" + "|".join(re.escape(d) for d in dirs) + "):"
    return pl.col("security_id").str.contains(pat)


def _slice(p, s, f, lo, hi):
    def f_(df):
        return df.filter((pl.col("event_time") >= lo) & (pl.col("event_time") < hi))

    return f_(p), f_(s), f_(f)


def _score(m: dict) -> float:
    if not m or m.get("n", 0) < 30:
        return -999.0
    pen = 0.5 * (m.get("liquidation_count") or 0) + 0.001 * (m.get("margin_rejects") or 0)
    return (m.get("sharpe") or -999.0) - pen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    args = ap.parse_args()
    perp, spot, fund = load_carry(data_dir=pathlib.Path(args.data))

    results = {}
    for name, dirs in SUB_BOOKS.items():
        pf = _pair_filter(dirs)
        p = perp.filter(pf)
        s = spot.filter(pf)
        f = fund.filter(pf)
        dev_p, dev_s, dev_f = _slice(p, s, f, datetime(1970, 1, 1, tzinfo=UTC), DEV_END)
        keep = eligible_coins(dev_p, dev_s)
        dev_p = dev_p.filter(pl.col("security_id").is_in(keep))
        dev_s = dev_s.filter(pl.col("security_id").is_in(keep))
        dev_f = dev_f.filter(pl.col("security_id").is_in(keep))
        if not len(keep):
            results[name] = {"error": "no dev-eligible names"}
            continue
        h1 = _slice(dev_p, dev_s, dev_f, datetime(1970, 1, 1, tzinfo=UTC), DEV_MID)
        h2 = _slice(dev_p, dev_s, dev_f, DEV_MID, DEV_END)
        rec = {"dirs": list(dirs), "eligible": len(keep)}
        try:
            m1, _, _ = run_one(*h1, **CHAMP)
            m2, _, _ = run_one(*h2, **CHAMP)
            md, _, _ = run_one(dev_p, dev_s, dev_f, **CHAMP)
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
            results[name] = rec
            continue
        rec.update(
            h1_sharpe=m1.get("sharpe"),
            h2_sharpe=m2.get("sharpe"),
            dev_sharpe=md.get("sharpe"),
            h1=m1,
            h2=m2,
            dev=md,
        )
        rec["score"] = min(_score(m1), _score(m2)) + 0.01 * (md.get("sharpe") or 0)
        results[name] = rec
        print(
            f"{name} elig={len(keep)} h1={m1.get('sharpe')} h2={m2.get('sharpe')} "
            f"dev={md.get('sharpe')} score={rec['score']:.3f}",
            flush=True,
        )

    pathlib.Path("artifacts").mkdir(exist_ok=True)
    pathlib.Path("artifacts/arb3_pairdecomp_grid.json").write_text(
        json.dumps(results, indent=2, default=str)
    )

    best = max(
        (k for k in results if "score" in results[k]),
        key=lambda k: results[k]["score"],
        default=None,
    )
    if best is None:
        print("no sub-book qualified — aborting before holdout")
        return 1
    print("frozen sub-book:", best)

    pf = _pair_filter(tuple(results[best]["dirs"]))
    p = perp.filter(pf)
    s = spot.filter(pf)
    f = fund.filter(pf)
    out = {
        "frozen_sub_book": best,
        "frozen_config": CHAMP,
        "grid_scores": {k: v.get("score") for k, v in results.items()},
    }
    for label, (pp, ss, ff) in {
        "dev": _slice(p, s, f, datetime(1970, 1, 1, tzinfo=UTC), DEV_END),
        "holdout": _slice(p, s, f, DEV_END, datetime(2100, 1, 1, tzinfo=UTC)),
        "full": (p, s, f),
    }.items():
        keep_l = eligible_coins(pp, ss)
        m, _, _ = run_one(
            pp.filter(pl.col("security_id").is_in(keep_l)),
            ss.filter(pl.col("security_id").is_in(keep_l)),
            ff.filter(pl.col("security_id").is_in(keep_l)),
            **CHAMP,
        )
        out[label] = m
        out[label]["eligible"] = len(keep_l)
        print(label, json.dumps(m, default=str), flush=True)

    pathlib.Path("artifacts/arb3_pairdecomp_champion.json").write_text(
        json.dumps(out, indent=2, default=str)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
