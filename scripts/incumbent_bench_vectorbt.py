"""Matched-workload benchmark: dipcatcher `run_backtest` vs vectorbt `Portfolio.from_orders`.

Same real Binance daily bars, same causal target-weight panel, same costs,
same fill convention (decision at close t -> fill at open t+1). vectorbt sees
the weight panel shifted one bar forward with ``price=open`` and
``size_type='targetpercent'``, which is the identical economic semantics.

Dimensions scored:
  correctness  - per-bar NAV parity between the two engines + fills/costs
  latency      - median wall time per backtest run over R repetitions
  reliability  - fault-injection: duplicate weight rows, stale marks, kill
                 switch (dipcatcher must fail closed; incumbent behaviour is
                 recorded, not asserted)

Research-only: synthetic-order-free, real bars, no live-P&L claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

COMMISSION_BPS = 10.0  # realistic taker fee, identical on both sides
SMA_WINDOW = 20
INIT_NAV = 1_000_000.0


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_panel(paths: list[Path]) -> dict[str, pl.DataFrame]:
    """Real collected bars; close_total_return = close (spot, no dividends)."""
    out = {}
    for p in paths:
        fr = pl.read_parquet(p).sort("event_time")
        if "close_total_return" not in fr.columns:
            fr = fr.with_columns(pl.col("close").alias("close_total_return"))
        out[fr["security_id"][0]] = fr
    return out


def build_weight_panel(frames: dict[str, pl.DataFrame], sma_window: int) -> pl.DataFrame:
    """Causal momentum gate: w_t = 1/N for assets whose close_t > SMA_t.

    The panel is an input to BOTH engines, so the rule only needs to be
    deterministic and causal — equal weight over gated names.
    """
    rows = []
    n = len(frames)
    closes = {}
    times = None
    for sid, fr in frames.items():
        c = fr["close"].to_numpy()
        closes[sid] = c
        t = fr["event_time"].to_list()
        times = t if times is None else times
        assert t == times, "assets share a calendar in this workload"
    n_bars = len(times)
    for i in range(n_bars):
        gated = []
        for sid, c in closes.items():
            if i + 1 >= sma_window:
                sma = float(np.mean(c[i + 1 - sma_window : i + 1]))
                if c[i] > sma:
                    gated.append(sid)
        for sid in closes:
            # Emit every (date, sid) cell explicitly: absent cells mean
            # "target 0" to vectorbt (shift+fillna(0)) but "carry the last
            # target" to engines with sparse-rebalance-grid semantics — an
            # implicit row would make the two sides run different books.
            # 0.9/n keeps a cash buffer: the parity workload tests execution
            # accounting, not the engines' differing cash-constraint policies
            # (dipcatcher rejects whole orders, vectorbt clips — recorded
            # separately under fault_injection.cash_constraint_note).
            rows.append(
                {
                    "event_time": times[i],
                    "security_id": sid,
                    "target_weight": 0.9 / n if sid in gated else 0.0,
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "event_time": pl.Datetime("us", "UTC"),
            "security_id": pl.String,
            "target_weight": pl.Float64,
        },
    )


def run_dipcatcher(bars: pl.DataFrame, weights: pl.DataFrame, cost_bps: float, engine: str = "ref"):
    from quant_fund.config.models import AppConfig, CostConfig, RiskGateConfig

    if engine == "fast":
        from quant_fund.backtest.fast_replay import run_backtest_fast as _run
    else:
        from quant_fund.backtest.engine import run_backtest as _run

    cfg = AppConfig(
        costs=CostConfig(
            commission_bps=cost_bps,
            half_spread_bps=0.0,
            impact_y=0.0,
            bps_per_turnover=0.0,
            borrow_bps_per_year=0.0,
            financing_bps_per_year=0.0,
            frictionless=False,
            participation_limit=1.0,
        ),
        risk_gate=RiskGateConfig(
            max_order_notional=1e12,
            max_gross=100.0,
            max_net=100.0,
            max_name=1.0,
            max_participation=1.0,
            max_predicted_vol=100.0,
            stale_price_bars=3,
            stale_model_hours=1e9,
        ),
    )
    res = _run(bars, weights, cfg, initial_nav=INIT_NAV)
    return res.equity, res.metrics, res.fills


def run_vectorbt(
    open_px: pd.DataFrame, close_px: pd.DataFrame, weights_dec: pd.DataFrame, cost_bps: float
):
    """Identical economics: decision weights shifted +1 bar, filled at open."""
    import vectorbt as vbt

    size = weights_dec.shift(1).fillna(0.0)  # weight decided at t executes at t+1 open
    pf = vbt.Portfolio.from_orders(
        close=close_px,
        size=size,
        size_type="targetpercent",
        price=open_px,
        fees=cost_bps / 1e4,
        init_cash=INIT_NAV,
        cash_sharing=True,
        direction="longonly",
        freq="1D",
        group_by=True,
    )
    equity = pf.value()
    stats = {
        "final_value": float(equity.iloc[-1]),
        "n_orders": int(pf.orders.count()) if hasattr(pf.orders, "count") else None,
        "total_fees": float(pf.orders.records_readable["Fees"].sum())
        if pf.orders.count() > 0
        else 0.0,
    }
    return equity, stats, pf


def fault_injection(bars: pl.DataFrame, weights: pl.DataFrame) -> dict:
    """Behavioural checks: dipcatcher must fail closed; incumbent recorded."""
    from quant_fund.backtest.engine import StaleValuationError, run_backtest
    from quant_fund.config.models import AppConfig, CostConfig, KillSwitchConfig, RiskGateConfig
    from quant_fund.schemas.errors import KillSwitchActive  # noqa: F401

    def cfg(kill: str = "ENABLED") -> AppConfig:
        return AppConfig(
            costs=CostConfig(
                commission_bps=COMMISSION_BPS,
                half_spread_bps=0.0,
                impact_y=0.0,
                borrow_bps_per_year=0.0,
                participation_limit=1.0,
            ),
            risk_gate=RiskGateConfig(
                max_order_notional=1e12,
                max_gross=100.0,
                max_net=100.0,
                max_name=1.0,
                max_participation=1.0,
                max_predicted_vol=100.0,
                stale_price_bars=3,
                stale_model_hours=1e9,
            ),
            kill_switch=KillSwitchConfig(state=kill),
        )

    out: dict[str, object] = {}

    # 1. duplicate (event_time, sid) weight rows -> must raise
    dup = pl.concat([weights, weights.head(1)])
    try:
        run_backtest(bars, dup, cfg())
        out["duplicate_weight_rows"] = "ACCEPTED (unexpected)"
    except ValueError as e:
        out["duplicate_weight_rows"] = f"rejected: {e}"

    # 2. stale marks: held position whose close+close_total_return stop printing.
    # Weight rows run through the end so the engine keeps the position on.
    sid_hold = sorted(bars["security_id"].unique().to_list())[0]
    all_dates = sorted(bars["event_time"].unique().to_list())
    tail_start = len(all_dates) - 8  # > stale_price_bars=3
    stale = bars.with_columns(
        pl.when(
            (pl.col("security_id") == sid_hold) & pl.col("event_time").is_in(all_dates[tail_start:])
        )
        .then(None)
        .otherwise(pl.col("close"))
        .alias("close")
    ).with_columns(
        pl.when(
            (pl.col("security_id") == sid_hold) & pl.col("event_time").is_in(all_dates[tail_start:])
        )
        .then(None)
        .otherwise(pl.col("close_total_return"))
        .alias("close_total_return")
    )
    w_hold = pl.DataFrame(
        {
            "event_time": all_dates[:-1],
            "security_id": [sid_hold] * (len(all_dates) - 1),
            "target_weight": [0.5] * (len(all_dates) - 1),
        }
    ).with_columns(pl.col("event_time").cast(pl.Datetime("us", "UTC")))
    try:
        run_backtest(stale, w_hold, cfg())
        out["stale_held_marks"] = "ACCEPTED (unexpected)"
    except StaleValuationError as e:
        out["stale_held_marks"] = f"rejected: {e}"
    except Exception as e:  # noqa: BLE001
        out["stale_held_marks"] = f"other: {type(e).__name__}: {e}"

    # 3. kill switch: HALT_NEW_ORDERS -> every attempted order blocked+counted
    res = run_backtest(bars, weights, cfg(kill="HALT_NEW_ORDERS"))
    out["kill_switch_halts"] = {
        "nav_start": float(res.equity["nav"][0]) if res.equity.height else None,
        "nav_end": float(res.equity["nav"][-1]) if res.equity.height else None,
        "fills": res.fills.height,
        "halt_rejects": res.metrics.get("kill_switch_halts"),
        "flat_equity": bool(
            res.equity.height and (res.equity["nav"].max() == res.equity["nav"].min())
        ),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bars",
        type=Path,
        nargs="+",
        default=[
            ROOT / "data/raw/sources/btcusdt_1d.parquet",
            ROOT / "data/raw/sources/ethusdt_1d.parquet",
            ROOT / "data/raw/sources/solusdt_1d.parquet",
        ],
    )
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument(
        "--engine",
        choices=["ref", "fast"],
        default="ref",
        help="dipcatcher engine: reference loop or bit-identical fast replay",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import vectorbt as vbt

    frames = load_panel(args.bars)
    sids = sorted(frames)
    print(f"assets: {sids}")

    weights = build_weight_panel(frames, SMA_WINDOW)
    print(f"weight panel: {weights.height} rows, {weights['event_time'].n_unique()} decision dates")

    bars = pl.concat(list(frames.values()))
    idx = pd.DatetimeIndex(pd.to_datetime(frames[sids[0]]["event_time"].to_list()))
    open_px = pd.DataFrame({sid: frames[sid]["open"].to_numpy() for sid in sids}, index=idx)
    close_px = pd.DataFrame({sid: frames[sid]["close"].to_numpy() for sid in sids}, index=idx)
    wmat = (
        weights.to_pandas()
        .assign(event_time=lambda d: pd.to_datetime(d["event_time"]))
        .pivot(index="event_time", columns="security_id", values="target_weight")
        .reindex(index=idx, columns=sids)
        .fillna(0.0)
    )

    # --- correctness -------------------------------------------------------
    t0 = time.perf_counter()
    eq_dc, met_dc, fills_dc = run_dipcatcher(bars, weights, COMMISSION_BPS, engine=args.engine)
    t_dc = time.perf_counter() - t0
    eq_vb, st_vb, pf_vb = run_vectorbt(open_px, close_px, wmat, COMMISSION_BPS)
    print(f"dipcatcher: {eq_dc.height} nav rows, {t_dc * 1e3:.0f} ms")

    nav_dc = dict(zip(eq_dc["event_time"].to_list(), eq_dc["nav"].to_list(), strict=True))
    nav_vb = {t.to_pydatetime(): float(v) for t, v in eq_vb.items()}
    common = sorted(set(nav_dc) & set(nav_vb))
    diffs = np.array([nav_dc[t] - nav_vb[t] for t in common])
    rel = np.abs(diffs) / np.array([nav_vb[t] for t in common])
    correctness = {
        "common_dates": len(common),
        "nav_max_abs_diff": float(np.abs(diffs).max()),
        "nav_max_rel_diff": float(rel.max()),
        "nav_final_dipcatcher": float(eq_dc["nav"][-1]),
        "nav_final_vectorbt": float(eq_vb.iloc[-1]),
        "fills_dipcatcher": int(fills_dc.height),
        "dipcatcher_total_fees": float(fills_dc["fee"].sum()) if fills_dc.height else 0.0,
        "cash_rejects_dipcatcher": int(met_dc.get("cash_rejects", 0) or 0),
        "risk_gate_rejects_dipcatcher": int(met_dc.get("risk_gate_rejects", 0) or 0),
        "vectorbt_orders": st_vb["n_orders"],
        "vectorbt_total_fees": st_vb["total_fees"],
        "dipcatcher_metrics_keys": sorted(met_dc.keys())[:40],
    }
    print(
        json.dumps(
            {k: v for k, v in correctness.items() if k != "dipcatcher_metrics_keys"}, indent=2
        )
    )

    # --- latency -----------------------------------------------------------
    reps = args.reps
    ts_dc, ts_vb = [], []
    # Interleave reps: under drifting load (shared host), sequential blocks
    # would attribute load windows to whichever engine ran during them.
    for _ in range(reps):
        t0 = time.perf_counter()
        run_dipcatcher(bars, weights, COMMISSION_BPS, engine=args.engine)
        ts_dc.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        run_vectorbt(open_px, close_px, wmat, COMMISSION_BPS)
        ts_vb.append(time.perf_counter() - t0)
    latency = {
        "reps": reps,
        "dipcatcher_ms_median": float(np.median(ts_dc) * 1e3),
        "vectorbt_ms_median": float(np.median(ts_vb) * 1e3),
        "dipcatcher_ms_all": [round(t * 1e3, 2) for t in ts_dc],
        "vectorbt_ms_all": [round(t * 1e3, 2) for t in ts_vb],
    }
    print(
        f"latency: dipcatcher {latency['dipcatcher_ms_median']:.0f} ms | "
        f"vectorbt {latency['vectorbt_ms_median']:.0f} ms"
    )

    # --- reliability / operability ----------------------------------------
    faults = fault_injection(bars, weights)
    print(json.dumps(faults, indent=2, default=str))

    # vectorbt counterpart observations (recorded, not asserted)
    vbt_faults = {}
    try:
        dup_w = wmat.copy()
        dup_w.iloc[5] = 0.99  # abrupt conflicting target - vbt accepts silently
        pf2 = vbt.Portfolio.from_orders(
            close=close_px,
            size=dup_w.shift(1).fillna(0.0),
            size_type="targetpercent",
            price=open_px,
            fees=COMMISSION_BPS / 1e4,
            init_cash=INIT_NAV,
            cash_sharing=True,
            direction="longonly",
            freq="1D",
            group_by=True,
        )
        vbt_faults["conflicting_target"] = f"accepted, final={float(pf2.value().iloc[-1]):.2f}"
    except Exception as e:  # noqa: BLE001
        vbt_faults["conflicting_target"] = f"raised {type(e).__name__}"
    stale_close = close_px.copy()
    stale_close.iloc[-30:, 0] = np.nan
    try:
        pf3 = vbt.Portfolio.from_orders(
            close=stale_close,
            size=wmat.shift(1).fillna(0.0),
            size_type="targetpercent",
            price=open_px,
            fees=COMMISSION_BPS / 1e4,
            init_cash=INIT_NAV,
            cash_sharing=True,
            direction="longonly",
            freq="1D",
            group_by=True,
        )
        v = pf3.value()
        vbt_faults["stale_marks"] = (
            f"accepted; equity NaN rows={int(v.isna().sum())}, "
            f"final={float(v.iloc[-1]) if not np.isnan(v.iloc[-1]) else 'NaN'}"
        )
    except Exception as e:  # noqa: BLE001
        vbt_faults["stale_marks"] = f"raised {type(e).__name__}: {e}"
    faults["vectorbt_counterpart"] = vbt_faults

    receipt = {
        "schema": "incumbent_bench.v1",
        "dipcatcher_engine": args.engine,
        "incumbent": {
            "name": "vectorbt",
            "version": vbt.__version__,
            "url": "https://vectorbt.dev/",
            "repo": "https://github.com/polakowo/vectorbt",
        },
        "workload": {
            "assets": sids,
            "bars_per_asset": int(frames[sids[0]].height),
            "bar_files_sha256": {p.name: _sha256(p) for p in args.bars},
            "script_sha256": _sha256(Path(__file__)),
            "strategy": f"causal SMA{SMA_WINDOW} gate, equal-weight 1/{len(sids)} per gated name",
            "fills": "decision close t -> exec open t+1 (both engines)",
            "commission_bps": COMMISSION_BPS,
            "spread_impact_borrow": 0.0,
            "init_nav": INIT_NAV,
        },
        "correctness": correctness,
        "latency": latency,
        "fault_injection": faults,
        "environment": {
            "python": sys.version.split()[0],
            "platform": sys.platform,
        },
        "known_semantic_differences": [
            "cash constraint: with weights summing to exactly 1.0, dipcatcher "
            "rejects whole orders that would overdraw cash (329 rejects on the "
            "3-asset workload), while vectorbt clips/fills; workloads use a 0.9 "
            "target-sum buffer so the parity test isolates accounting.",
        ],
        "research_only": True,
        "live_pnl_claim": False,
        "disclaimer": (
            "Single matched workload vs vectorbt on real Binance daily bars. "
            "Latency is single-process wall time for identical economics; "
            "correctness is NAV parity; reliability demos are behavioural. "
            "Not a claim of superiority across all product dimensions."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n")
    print(f"receipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
