"""Post-hoc leverage profile of the FROZEN 3-venue arb champion.

Not a grid and not a new selection: the champion config is already locked
(enter=1e-3, exit=-1.25e-4, lb=9, nw=0.08, mx=30, band=1.3, rsr=2e-3,
rsc=1.5, rsf=1.0). This script only scales name_weight by k and raises
the perp leverage cap so sizing does not bind, characterizing the
return/MDD scaling of the identical strategy. Recorded honestly:
Sharpe is approximately scale-invariant; CAGR and MDD scale ~linearly.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from datetime import UTC, datetime

import polars as pl
from carry_research import eligible_coins, load_carry, make_cfg

from quant_fund.backtest.carry_engine import run_carry_backtest
from quant_fund.backtest.sleeves import basis_carry_hysteresis_weights

CHAMP = dict(
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
    lo = datetime.fromisoformat(lo) if isinstance(lo, str) else lo
    hi = datetime.fromisoformat(hi) if isinstance(hi, str) else hi

    def f_(df):
        return df.filter((pl.col("event_time") >= lo) & (pl.col("event_time") < hi))

    return f_(p), f_(s), f_(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/arb3_book")
    ap.add_argument("--split", default="2025-01-01T00:00:00")
    ap.add_argument("--out", default="artifacts/arb_lev_profile.json")
    args = ap.parse_args()
    split = datetime.fromisoformat(args.split).replace(tzinfo=UTC)

    perp, spot, fund = load_carry(data_dir=pathlib.Path(args.data))
    p, s, f = _slice(perp, spot, fund, split, datetime(2100, 1, 1, tzinfo=UTC))
    keep = eligible_coins(p, s)
    pp = p.filter(pl.col("security_id").is_in(keep))
    ss = s.filter(pl.col("security_id").is_in(keep))
    ff = f.filter(pl.col("security_id").is_in(keep))
    print("holdout sids:", len(keep), "| n:", pp["event_time"].n_unique(), flush=True)

    out = {"frozen_champion": CHAMP, "rows": {}}
    for k in (1.0, 2.0, 3.0, 4.0, 5.0):
        cfg_kw = dict(CHAMP)
        cfg_kw["name_weight"] = CHAMP["name_weight"] * k
        # keep margin cap comfortably above the resulting perp gross
        cfg = make_cfg()
        cfg.perp.max_leverage = math.ceil(0.08 * k * 30) + 2
        w = basis_carry_hysteresis_weights(pp, ff, **cfg_kw)
        res = run_carry_backtest(pp, ss, ff, w, cfg, initial_nav=1e6)
        m = res.metrics
        row = {
            "nw_mult": k,
            "name_weight": cfg_kw["name_weight"],
            "max_leverage": cfg.perp.max_leverage,
            "total_return": m.get("total_return"),
            "cagr": m.get("cagr"),
            "sharpe": m.get("sharpe"),
            "max_drawdown": m.get("max_drawdown"),
            "funding_net": m.get("funding_net"),
            "liquidation_count": m.get("liquidation_count"),
            "margin_rejects": m.get("margin_rejects"),
            "n": m.get("n"),
        }
        out["rows"][f"k{k:g}"] = row
        print(
            f"k={k:g} lev={cfg.perp.max_leverage} ret={row['total_return']:.4f} "
            f"cagr={row['cagr']} sharpe={row['sharpe']:.3f} mdd={row['max_drawdown']:.4f} "
            f"liq={row['liquidation_count']} mj={row['margin_rejects']}",
            flush=True,
        )

    pathlib.Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print("wrote", args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
