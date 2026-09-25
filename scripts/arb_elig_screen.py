"""Funding-level eligibility screen on the h↔b arb book.

Pair decomposition showed the 3-venue book is effectively h↔b (OKX sids have
no dev coverage and are never picked). New lane: freeze a name set by the
names' own funding-spread quality computed on dev — median spread, fraction
of events above the entry bar, or top-K by dev funding mass — then evaluate
the frozen set once on holdout.

Honest selection: each candidate screen is applied to h1 and evaluated on
h2 (strictly out-of-screen) plus applied to dev and evaluated on dev;
score = min of the two Sharpes. The winning screen is frozen as
screen(dev) → one locked holdout eval. Baseline (no screen) is a candidate,
so a screen is only adopted if it beats the status quo on dev.

Writes artifacts/arb_elig_screen_{grid,champion}.json.
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


def _screen_names(fund_dev: pl.DataFrame, kind: str, param: float) -> set[str]:
    """Names passing the dev-computed funding screen."""
    g = fund_dev.group_by("security_id").agg(
        med=pl.col("value").median(),
        frac=pl.col("value").gt(1e-3).mean(),
        tot=pl.col("value").sum(),
        mu=pl.col("value").mean(),
        cnt=pl.col("value").gt(1e-3).sum(),
    )
    if kind == "median_q":
        q = g.select(pl.col("med").quantile(param)).item()
        return set(g.filter(pl.col("med") >= q)["security_id"].to_list())
    if kind == "frac":
        return set(g.filter(pl.col("frac") >= param)["security_id"].to_list())
    if kind == "topk_mass":
        return set(g.sort("tot", descending=True).head(int(param))["security_id"].to_list())
    if kind == "topk_mean":
        return set(g.sort("mu", descending=True).head(int(param))["security_id"].to_list())
    if kind == "topk_cnt":
        return set(g.sort("cnt", descending=True).head(int(param))["security_id"].to_list())
    raise ValueError(kind)


SCREENS = [("none", 0.0)]
SCREENS += [("median_q", q) for q in (0.5, 0.6, 0.7, 0.8)]
SCREENS += [("frac", f) for f in (0.2, 0.3, 0.4, 0.5)]
SCREENS += [("topk_mass", k) for k in (20, 40, 60, 80, 100, 120, 150)]
SCREENS += [("topk_mean", k) for k in (60, 90, 120, 150)]
SCREENS += [("topk_cnt", k) for k in (60, 70, 80, 90, 100, 110, 120, 150)]


def _slice(p, s, f, lo, hi):
    def f_(df):
        return df.filter((pl.col("event_time") >= lo) & (pl.col("event_time") < hi))

    return f_(p), f_(s), f_(f)


def _score(m: dict) -> float:
    if not m or m.get("n", 0) < 30:
        return -999.0
    pen = 0.5 * (m.get("liquidation_count") or 0) + 0.001 * (m.get("margin_rejects") or 0)
    return (m.get("sharpe") or -999.0) - pen


def _eval_on(p, s, f, names):
    keep = names & eligible_coins(p, s)
    return run_one(
        p.filter(pl.col("security_id").is_in(keep)),
        s.filter(pl.col("security_id").is_in(keep)),
        f.filter(pl.col("security_id").is_in(keep)),
        **CHAMP,
    )[0]


def main() -> int:
    perp, spot, fund = load_carry(data_dir=DATA)
    dev_p, dev_s, dev_f = _slice(perp, spot, fund, datetime(1970, 1, 1, tzinfo=UTC), DEV_END)
    h1 = _slice(dev_p, dev_s, dev_f, datetime(1970, 1, 1, tzinfo=UTC), DEV_MID)
    h2 = _slice(dev_p, dev_s, dev_f, DEV_MID, DEV_END)

    results = {}
    for kind, param in SCREENS:
        try:
            if kind == "none":
                m2, _, _ = run_one(*h2, **CHAMP)
                md = _eval_on(dev_p, dev_s, dev_f, eligible_coins(dev_p, dev_s))
            else:
                names_h1 = _screen_names(h1[2], kind, param)
                names_dev = _screen_names(dev_f, kind, param)
                if len(names_h1) < 10 or len(names_dev) < 10:
                    results[f"{kind}:{param}"] = {"error": "screen too tight"}
                    continue
                m2 = _eval_on(*h2, names_h1)
                md = _eval_on(dev_p, dev_s, dev_f, names_dev)
            sc = min(_score(m2), _score(md) * 1.01)
            results[f"{kind}:{param}"] = {
                "h2": m2,
                "dev": md,
                "score": sc,
            }
            print(
                f"{kind}:{param} h2={m2.get('sharpe')} dev={md.get('sharpe')} score={sc:.3f}",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            results[f"{kind}:{param}"] = {"error": f"{type(e).__name__}: {e}"}

    pathlib.Path("artifacts").mkdir(exist_ok=True)
    pathlib.Path("artifacts/arb_elig_screen_grid.json").write_text(
        json.dumps(results, indent=2, default=str)
    )

    best_key = max(
        (k for k in results if "score" in results[k]),
        key=lambda k: results[k]["score"],
        default=None,
    )
    if best_key is None or best_key == "none:0.0":
        print("no screen beats baseline on dev — holding out is pointless; stop")
        return 0
    kind, param = best_key.split(":")
    names_dev = _screen_names(dev_f, kind, float(param))
    print("frozen screen:", best_key, "names:", len(names_dev))

    out = {"frozen_screen": best_key, "frozen_config": CHAMP, "n_names": len(names_dev)}
    for label, (p, s, f) in {
        "dev": (dev_p, dev_s, dev_f),
        "holdout": _slice(perp, spot, fund, DEV_END, datetime(2100, 1, 1, tzinfo=UTC)),
        "full": (perp, spot, fund),
    }.items():
        m = _eval_on(p, s, f, names_dev)
        out[label] = m
        print(label, json.dumps(m, default=str), flush=True)

    pathlib.Path("artifacts/arb_elig_screen_champion.json").write_text(
        json.dumps(out, indent=2, default=str)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
