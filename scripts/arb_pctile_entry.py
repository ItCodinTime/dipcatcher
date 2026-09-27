"""Cross-sectional percentile entry on the h↔b arb book.

The absolute entry bar (spread > 1e-3) goes deaf as cross-venue funding
spreads compress — in the 2026 holdout regime few names clear it. A
regime-adaptive gate: rank each name's spread within the day's cross-section
and emit its percentile rank as a synthetic "rate" — the same hysteresis
sizer then operates in percentile space: enter when a name's trailing
percentile stays above enter_p, exit when it decays below exit_p. The book
stays populated in every regime (always a top-X% set), yet quality ranking
is preserved.

Runs on the full book and on the cnt:k90 screened universe (4.18 champion).
Dev-only selection (min(h2, dev*1.01)), freeze, one holdout eval. Baseline
(absolute enter) is a candidate — percentile entry is adopted only if it
beats the status quo on dev.

Writes artifacts/arb_pctile_{grid,champion}.json.
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

BASE = dict(enter=1e-3, exit_=-1.25e-4, lb=9, nw=0.08, mx=30, band=1.3, rsr=2e-3, rsc=1.5, rsf=1.0)

GRID = [
    dict(enter_p=e, exit_p=x, lb=lb)
    for e, x, lb in itertools.product((0.5, 0.6, 0.7, 0.8, 0.9), (0.2, 0.3, 0.5), (3, 9))
]


def _pctile_funding(fund: pl.DataFrame) -> pl.DataFrame:
    """Emit each event's within-day cross-sectional percentile rank."""
    return fund.with_columns(
        (pl.col("value").rank("ordinal") / pl.col("value").count())
        .over("event_time")
        .alias("value")
    ).sort("event_time", "security_id")


def _topk_cnt(fund_dev: pl.DataFrame, k: int) -> set[str]:
    g = fund_dev.group_by("security_id").agg(cnt=pl.col("value").gt(1e-3).sum())
    return set(g.sort("cnt", descending=True).head(k)["security_id"].to_list())


def _slice(p, s, f, lo, hi):
    q = lambda d: d.filter((pl.col("event_time") >= lo) & (pl.col("event_time") < hi))  # noqa: E731
    return q(p), q(s), q(f)


def _score(m: dict) -> float:
    if not m or m.get("n", 0) < 30:
        return -999.0
    pen = 0.5 * (m.get("liquidation_count") or 0) + 0.001 * (m.get("margin_rejects") or 0)
    return (m.get("sharpe") or -999.0) - pen


def _eval_on(p, s, f, names, cfg_pctile):
    keep = names & eligible_coins(p, s)
    ff = f.filter(pl.col("security_id").is_in(keep))
    cfg = BASE | {
        "enter": cfg_pctile["enter_p"],
        "exit_": cfg_pctile["exit_p"],
        "lb": cfg_pctile["lb"],
        "rsr": None,  # percentile space already normalizes; no rate scaling
    }
    return run_one(
        p.filter(pl.col("security_id").is_in(keep)),
        s.filter(pl.col("security_id").is_in(keep)),
        _pctile_funding(ff),
        **cfg,
    )[0]


def _eval_abs(p, s, f, names):
    """Baseline: absolute-spread hysteresis (the 4.18 champion config)."""
    keep = names & eligible_coins(p, s)
    return run_one(
        p.filter(pl.col("security_id").is_in(keep)),
        s.filter(pl.col("security_id").is_in(keep)),
        f.filter(pl.col("security_id").is_in(keep)),
        **BASE,
    )[0]


def main() -> int:
    perp, spot, fund = load_carry(data_dir=DATA)
    dev_p, dev_s, dev_f = _slice(perp, spot, fund, datetime(1970, 1, 1, tzinfo=UTC), DEV_END)
    h2 = _slice(dev_p, dev_s, dev_f, DEV_MID, DEV_END)

    universes = {
        "full": eligible_coins(dev_p, dev_s),
        "cnt90": _topk_cnt(dev_f, 90) & eligible_coins(dev_p, dev_s),
    }

    results = {}
    for uname, names_dev in universes.items():
        # baseline absolute champion on this universe
        md = _eval_abs(dev_p, dev_s, dev_f, names_dev)
        m2 = _eval_abs(*h2, names_dev)
        sc = min(_score(m2), _score(md) * 1.01)
        results[f"{uname}:abs"] = {"dev": md.get("sharpe"), "h2": m2.get("sharpe"), "score": sc}
        print(
            f"{uname}:abs dev={md.get('sharpe'):.2f} h2={m2.get('sharpe'):.2f} sc={sc:.3f}",
            flush=True,
        )

        for i, g in enumerate(GRID):
            key = f"{uname}:p{i}"
            try:
                m2 = _eval_on(*h2, names_dev, g)
                md = _eval_on(dev_p, dev_s, dev_f, names_dev, g)
            except Exception as e:  # noqa: BLE001
                results[key] = {"error": f"{type(e).__name__}: {e}"}
                continue
            sc = min(_score(m2), _score(md) * 1.01)
            results[key] = {"cfg": g, "dev": md.get("sharpe"), "h2": m2.get("sharpe"), "score": sc}
            if i % 10 == 0:
                print(
                    f"{key} dev={md.get('sharpe'):.2f} h2={m2.get('sharpe'):.2f} sc={sc:.3f}",
                    flush=True,
                )

    pathlib.Path("artifacts").mkdir(exist_ok=True)
    pathlib.Path("artifacts/arb_pctile_grid.json").write_text(
        json.dumps(results, indent=2, default=str)
    )

    best_key = max(
        (k for k in results if "score" in results[k]),
        key=lambda k: results[k]["score"],
        default=None,
    )
    if best_key is None or best_key.endswith(":abs"):
        print("no percentile config beats absolute baseline on dev — stop")
        return 0
    uname = best_key.split(":")[0]
    champ = results[best_key]["cfg"]
    names_dev = universes[uname]
    print("frozen:", best_key, champ, "names:", len(names_dev))

    out = {"frozen": best_key, "frozen_cfg": champ, "universe": uname}
    for label, (p, s, f) in {
        "dev": (dev_p, dev_s, dev_f),
        "holdout": _slice(perp, spot, fund, DEV_END, datetime(2100, 1, 1, tzinfo=UTC)),
        "full": (perp, spot, fund),
    }.items():
        m = _eval_on(p, s, f, names_dev, champ)
        out[label] = m
        print(label, json.dumps(m, default=str), flush=True)

    pathlib.Path("artifacts/arb_pctile_champion.json").write_text(
        json.dumps(out, indent=2, default=str)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
