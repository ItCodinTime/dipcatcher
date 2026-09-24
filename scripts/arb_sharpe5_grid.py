"""Sharpe>5 attack lane: dev-only selection on the cross-venue arb book.

Protocol (honest, no holdout peeking at selection time):
1. Split the dev window (pre-2025-01-01) into two internal halves.
2. Score each grid config by min(Sharpe_h1, Sharpe_h2) — a config must work
   in BOTH dev halves, not just the regime that flatters it — tie-broken by
   full-dev Sharpe, then higher max_names (diversification).
3. Freeze the winner; evaluate once on dev, once on holdout, once on full.
4. Report whatever the numbers are — failure is a result (per .dsh-24x7).

Requires data/arb_carry_book/{perp_bars,spot_bars,funding}.parquet from
build_arb_book.py. Writes artifacts/arb_sharpe5_{grid,champion}.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
from datetime import UTC, datetime

import polars as pl
from carry_research import eligible_coins, load_carry, run_one

DATA = pathlib.Path("data/arb_carry_book")
DEV_END = datetime(2025, 1, 1, tzinfo=UTC)
DEV_MID = datetime(2024, 1, 1, tzinfo=UTC)

GRID = [
    dict(enter=e, exit_=x, lb=lb, nw=nw, mx=mx, band=1.3, rsr=rsr, rsc=1.5, rsf=1.0)
    for e, x, lb, nw, mx, rsr in itertools.product(
        (2e-4, 3e-4, 5e-4),  # enter: funding spread entry (daily)
        (-1.25e-4, 0.0),  # exit: hysteresis floor
        (3, 9),  # lookback events
        (0.08, 0.12),  # name weight
        (15, 30, 60),  # max names
        (2e-3,),  # rate scale reference
    )
]
# --refine: fine-grain around the extended-grid champion region (dev-only).
REFINE_GRID = [
    dict(enter=e, exit_=x, lb=lb, nw=nw, mx=mx, band=1.3, rsr=rsr, rsc=1.5, rsf=1.0)
    for e, x, lb, nw, mx, rsr in itertools.product(
        (7e-4, 1e-3, 1.4e-3),
        (-1.25e-4, 0.0),
        (3, 5, 9),
        (0.08, 0.12),
        (30,),
        (2e-3,),
    )
]
# --extend: widen past the observed champion's grid edges (dev-only selection).
EXT_GRID = [
    dict(enter=e, exit_=x, lb=lb, nw=nw, mx=mx, band=1.3, rsr=rsr, rsc=1.5, rsf=1.0)
    for e, x, lb, nw, mx, rsr in itertools.product(
        (5e-4, 1e-3, 2e-3, 4e-3),
        (-1.25e-4, 0.0, 1e-4),
        (9, 21, 45),
        (0.08, 0.2),
        (15, 30),
        (2e-3,),
    )
]


def _slice(p, s, f, lo, hi):
    lo = datetime.fromisoformat(lo) if isinstance(lo, str) else lo
    hi = datetime.fromisoformat(hi) if isinstance(hi, str) else hi

    def f_(df):
        return df.filter((pl.col("event_time") >= lo) & (pl.col("event_time") < hi))

    return f_(p), f_(s), f_(f)


def _score(m: dict) -> float:
    """Selection score: penalize missing/Sharpe and risky books."""
    if not m or m.get("n", 0) < 30:
        return -999.0
    pen = 0.0
    pen += 0.5 * (m.get("liquidation_count") or 0)
    pen += 0.001 * (m.get("margin_rejects") or 0)
    return (m.get("sharpe") or -999.0) - pen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--out-prefix", default="arb_sharpe5")
    ap.add_argument("--extend", action="store_true")
    ap.add_argument("--refine", action="store_true")
    args = ap.parse_args()
    grid = REFINE_GRID if args.refine else (EXT_GRID if args.extend else GRID)
    perp, spot, fund = load_carry(data_dir=pathlib.Path(args.data))
    print("arb book sids:", perp["security_id"].n_unique(), "fund:", fund.height)

    dev_p, dev_s, dev_f = _slice(perp, spot, fund, datetime(1970, 1, 1, tzinfo=UTC), DEV_END)
    keep = eligible_coins(dev_p, dev_s)
    dev_p = dev_p.filter(pl.col("security_id").is_in(keep))
    dev_s = dev_s.filter(pl.col("security_id").is_in(keep))
    dev_f = dev_f.filter(pl.col("security_id").is_in(keep))
    print("dev eligible pairs:", len(keep))

    h1 = _slice(dev_p, dev_s, dev_f, datetime(1970, 1, 1, tzinfo=UTC), DEV_MID)
    h2 = _slice(dev_p, dev_s, dev_f, DEV_MID, DEV_END)

    results = {}
    for i, g in enumerate(grid):
        rec = {"cfg": g}
        try:
            m1, _, _ = run_one(*h1, **g)
            m2, _, _ = run_one(*h2, **g)
            md, _, _ = run_one(dev_p, dev_s, dev_f, **g)
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
            results[f"g{i}"] = rec
            continue
        rec["h1_sharpe"], rec["h2_sharpe"], rec["dev_sharpe"] = (
            m1.get("sharpe"),
            m2.get("sharpe"),
            md.get("sharpe"),
        )
        rec["score"] = min(_score(m1), _score(m2)) + 0.01 * (md.get("sharpe") or 0)
        results[f"g{i}"] = rec
        print(
            f"g{i} h1={rec['h1_sharpe']} h2={rec['h2_sharpe']} dev={rec['dev_sharpe']} score={rec['score']:.3f}",
            flush=True,
        )

    pathlib.Path("artifacts").mkdir(exist_ok=True)
    pathlib.Path(f"artifacts/{args.out_prefix}_grid.json").write_text(
        json.dumps(results, indent=2, default=str)
    )

    best_key = max(
        (k for k in results if "score" in results[k]),
        key=lambda k: results[k]["score"],
        default=None,
    )
    if best_key is None:
        print("no config completed the dev grid — aborting before holdout")
        return 1
    champ = results[best_key]["cfg"]
    print("frozen champion:", best_key, champ)

    out = {}
    for label, (p, s, f) in {
        "dev": (dev_p, dev_s, dev_f),
        "holdout": _slice(perp, spot, fund, DEV_END, datetime(2100, 1, 1, tzinfo=UTC)),
        "full": (perp, spot, fund),
    }.items():
        keep_l = eligible_coins(p, s)
        m, w, r = run_one(
            p.filter(pl.col("security_id").is_in(keep_l)),
            s.filter(pl.col("security_id").is_in(keep_l)),
            f.filter(pl.col("security_id").is_in(keep_l)),
            **champ,
        )
        out[label] = m
        out[label]["eligible"] = len(keep_l)
        print(label, json.dumps(m, default=str), flush=True)

    out["frozen_config"] = champ
    out["selection"] = "argmax(min(sharpe_h1,sharpe_h2)+0.01*dev_sharpe) — dev only"
    pathlib.Path(f"artifacts/{args.out_prefix}_champion.json").write_text(
        json.dumps(out, indent=2, default=str)
    )
    gate = (out["holdout"].get("sharpe") or 0) > 5 and (
        out["holdout"].get("max_drawdown") or -1
    ) > -0.05
    print("MEGAPLAN GATE (holdout Sharpe>5 & DD<5%):", "PASS" if gate else "NOT PROVEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
