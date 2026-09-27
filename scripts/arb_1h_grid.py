"""Frozen-config grid eval on the 1h cross-venue arb book.

Same protocol as ``arb_sharpe5_grid.py`` (dev-half grid -> champion on
min(h1,h2) -> one locked holdout eval, failure is a result) but pointed at
``data/arb_carry_book_1h`` — 1h price grain with funding spread emitted once
per Binance 8h epoch (Σ HL hourly − r_Binance). Loads parquets directly:
``carry_research.load_carry`` truncates funding to daily timestamps, which
would destroy the epoch grain here.

Writes artifacts/arb_1h_{grid,champion}.json.
"""

from __future__ import annotations

import itertools
import json
import pathlib
from datetime import UTC, datetime

import polars as pl
from carry_research import eligible_coins, run_one

DATA = pathlib.Path("data/arb_carry_book_1h")
SPLIT_DEFAULT = datetime(2025, 1, 1, tzinfo=UTC)

# Entry/exit thresholds are per-8h-epoch spreads (~1/3 of the daily scale).
GRID = [
    dict(enter=e, exit_=x, lb=lb, nw=nw, mx=mx, band=1.3, rsr=rsr, rsc=1.5, rsf=1.0)
    for e, x, lb, nw, mx, rsr in itertools.product(
        (6e-5, 1e-4, 1.7e-4),  # enter: funding spread per epoch
        (-4e-5, 0.0),  # exit: hysteresis floor per epoch
        (3, 9),  # lookback epochs
        (0.08, 0.12),  # name weight
        (15, 30, 60),  # max names
        (6e-4,),  # rate scale reference (epoch scale)
    )
]


def _slice(p, s, f, lo, hi):
    lo = datetime.fromisoformat(lo) if isinstance(lo, str) else lo
    hi = datetime.fromisoformat(hi) if isinstance(hi, str) else hi

    def f_(df):
        return df.filter((pl.col("event_time") >= lo) & (pl.col("event_time") < hi))

    return f_(p), f_(s), f_(f)


def _score(m: dict) -> float:
    if not m or m.get("n", 0) < 30:
        return -999.0
    pen = 0.0
    pen += 0.5 * (m.get("liquidation_count") or 0)
    pen += 0.001 * (m.get("margin_rejects") or 0)
    return (m.get("sharpe") or -999.0) - pen


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--split", default=None, help="ISO dev/holdout boundary")
    ap.add_argument("--out-prefix", default="arb_1h")
    args = ap.parse_args()
    dev_end = (
        datetime.fromisoformat(args.split).replace(tzinfo=UTC) if args.split else SPLIT_DEFAULT
    )
    data = pathlib.Path(args.data)
    p = pl.read_parquet(data / "perp_bars.parquet")
    s = pl.read_parquet(data / "spot_bars.parquet")
    f = pl.read_parquet(data / "funding.parquet")
    t0, t1 = p["event_time"].min(), p["event_time"].max()
    dev_mid = t0 + (dev_end - t0) / 2
    print(f"bars {t0} .. {t1} | dev halves split at {dev_mid} | holdout {dev_end} .. {t1}")

    kl = eligible_coins(p, s)
    p = p.filter(pl.col("security_id").is_in(kl))
    s = s.filter(pl.col("security_id").is_in(kl))
    print(f"eligible: {len(kl)} sids")

    results = {}
    for i, cfg in enumerate(GRID):
        rec = {"cfg": cfg}
        for tag, lo, hi in (
            ("h1", t0, dev_mid),
            ("h2", dev_mid, dev_end),
            ("dev", t0, dev_end),
        ):
            pp, ss, ff = _slice(p, s, f, lo, hi)
            m, _, _ = run_one(pp, ss, ff, **cfg)
            rec[f"{tag}_sharpe"] = m.get("sharpe")
            if tag == "dev":
                rec["dev_metrics"] = m
        rec["score"] = min(
            _score(
                {
                    "n": 999,
                    "sharpe": rec["h1_sharpe"],
                    "liquidation_count": rec["dev_metrics"].get("liquidation_count"),
                }
            ),
            _score(
                {
                    "n": 999,
                    "sharpe": rec["h2_sharpe"],
                    "liquidation_count": rec["dev_metrics"].get("liquidation_count"),
                }
            ),
        ) + 0.01 * (rec["dev_sharpe"] or 0)
        results[f"g{i}"] = rec
        print(
            f"g{i} h1={rec['h1_sharpe']} h2={rec['h2_sharpe']} dev={rec['dev_sharpe']} score={rec['score']:.3f}",
            flush=True,
        )

    pathlib.Path("artifacts").mkdir(exist_ok=True)
    pathlib.Path("artifacts/arb_1h_grid.json").write_text(
        json.dumps(results, indent=2, default=str)
    )

    bk = max((k for k in results if "score" in results[k]), key=lambda k: results[k]["score"])
    champ = results[bk]["cfg"]
    print("frozen champion:", bk, champ)

    out = {"champion_id": bk, "grid": results}
    for label, lo, hi in (("dev", t0, dev_end), ("holdout", dev_end, t1), ("full", t0, t1)):
        pp, ss, ff = _slice(p, s, f, lo, hi)
        m, _, _ = run_one(pp, ss, ff, **champ)
        out[label] = m
        out[label]["eligible"] = len(kl)
        print(label, json.dumps(m, default=str), flush=True)
    out["frozen_config"] = champ
    out["selection"] = "argmax(min(sharpe_h1,sharpe_h2)+0.01*dev_sharpe) — dev only"
    pathlib.Path("artifacts/arb_1h_champion.json").write_text(
        json.dumps(out, indent=2, default=str)
    )
    gate = (out["holdout"].get("sharpe") or 0) > 5 and (
        (out["holdout"].get("max_drawdown") or -1) > -0.05
    )
    print("MEGAPLAN GATE (holdout Sharpe>5 & DD<5%):", "PASS" if gate else "NOT PROVEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
