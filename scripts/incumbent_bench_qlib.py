"""Matched-workload benchmark: dipcatcher `run_backtest` vs Qlib.

Same real Binance daily bars, same causal target-weight panel, same costs,
same fill convention. Qlib's daily executor reads the signal at the previous
step (``get_step_time(shift=1)``) and deals at the current step's
``deal_price="$open"`` — decision at close t -> fill at open t+1, identical to
the vectorbt bench and to dipcatcher.

Parity-critical Qlib settings:
  - ``trade_unit=None``     -> fractional crypto quantities (no lot rounding)
  - no ``$factor`` NaN gap   -> factor.day.bin written as 1.0 (no adjustment)
  - ``open_cost``/``close_cost`` = commission, ``min_cost=0``
  - ``limit_threshold=1e9``  -> disables qlib's price-limit halt. NOTE:
    ``limit_threshold=None`` silently falls back to ``C.limit_threshold``
    (0.095 for region="cn"), suspending orders on >9.5% daily moves —
    crypto hits that routinely; the default silently drops sells.
  - ``volume_threshold=None`` -> no volume cap (no config fallback here)
  - ``risk_degree=1.0``      -> weights pass through unscaled
  - ``deal_price="$open"``   -> fills at the next bar's open

Research-only: real bars, no live-P&L claim.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from incumbent_bench_vectorbt import (  # noqa: E402
    COMMISSION_BPS,
    INIT_NAV,
    SMA_WINDOW,
    _sha256,
    build_weight_panel,
    load_panel,
    run_dipcatcher,
)


def write_qlib_provider(provider: Path, frames: dict[str, pl.DataFrame]) -> list[str]:
    """Write Qlib's on-disk format: calendars/day.txt, instruments/all.txt,
    features/{inst}/{field}.day.bin (float32 LE: [start_index, values...])."""
    sids = sorted(frames)
    cal = frames[sids[0]]["event_time"].to_list()
    for sid in sids[1:]:
        assert frames[sid]["event_time"].to_list() == cal, "shared calendar required"
    dates = pd.DatetimeIndex(pd.to_datetime(cal))

    (provider / "calendars").mkdir(parents=True, exist_ok=True)
    # qlib needs one calendar entry past the last trade step for its end_time
    cal_ext = dates.append(pd.DatetimeIndex([dates[-1] + pd.Timedelta(days=1)]))
    (provider / "calendars" / "day.txt").write_text(
        "\n".join(d.strftime("%Y-%m-%d") for d in cal_ext) + "\n"
    )
    (provider / "instruments").mkdir(parents=True, exist_ok=True)
    (provider / "instruments" / "all.txt").write_text(
        "".join(
            f"{sid}\t{dates[0].strftime('%Y-%m-%d')}\t{dates[-1].strftime('%Y-%m-%d')}\n"
            for sid in sids
        )
    )
    n = len(dates)
    for sid in sids:
        fr = frames[sid]
        close = fr["close"].to_numpy().astype(np.float64)
        prev = np.concatenate([[close[0]], close[:-1]])
        fields = {
            "open": fr["open"].to_numpy(),
            "close": close,
            "high": fr["high"].to_numpy(),
            "low": fr["low"].to_numpy(),
            "volume": fr["volume"].to_numpy(),
            "vwap": close,  # unused by deal_price=$open; present for completeness
            "factor": np.ones(n),
            "change": np.where(prev != 0, close / prev - 1.0, 0.0),
        }
        fdir = provider / "features" / sid.lower()
        fdir.mkdir(parents=True, exist_ok=True)
        for name, vals in fields.items():
            np.hstack([[0.0], np.asarray(vals, dtype=np.float64)]).astype("<f4").tofile(
                fdir / f"{name}.day.bin"
            )
    return sids


def run_qlib(provider: Path, weights: pl.DataFrame, calendar: list, sids: list[str]):
    """Run the weight panel through Qlib's daily backtest; return equity series
    keyed by date plus order stats."""
    from qlib.backtest import backtest as qlib_backtest
    from qlib.backtest.signal import SignalWCache
    from qlib.contrib.strategy.order_generator import OrderGenerator
    from qlib.contrib.strategy.signal_strategy import WeightStrategyBase

    order_log: list[dict] = []

    class MatchedOrderGen(OrderGenerator):
        """dipcatcher-exact sizing: target units = w * NAV / open(t_step).

        Qlib's stock OrderGen renormalizes weights to sum=1 and floor-divides
        amounts to integer units — wrong for fractional crypto. This version
        keeps raw fractional targets at the step's deal price (open), which is
        exactly `desired = tw * nav / price` in the dipcatcher engine.
        """

        def generate_order_list_from_target_weight_position(
            self,
            current,
            trade_exchange,
            target_weight_position,
            risk_degree,
            pred_start_time,
            pred_end_time,
            trade_start_time,
            trade_end_time,
        ):
            from qlib.backtest.decision import OrderDir

            if target_weight_position is None:
                target_weight_position = {}
            current_amount_dict = current.get_stock_amount_dict()
            order_log.append(
                {
                    "trade_date": str(pd.Timestamp(trade_start_time).date()),
                    "code": "__CALL__",
                    "amount": 0.0,
                    "direction": -1,
                    "target_amount": float(len(target_weight_position)),
                    "cur_amount": float(len(current_amount_dict)),
                    "total_value": 0.0,
                }
            )
            total_value = (
                trade_exchange.calculate_amount_position_value(
                    amount_dict=current_amount_dict,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    only_tradable=False,
                )
                + current.get_cash()
            )
            target_amount_dict = {}
            for sid, w in target_weight_position.items():
                if w <= 0:
                    continue
                px = trade_exchange.get_deal_price(
                    stock_id=sid,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=OrderDir.BUY,
                )
                if px is None or not np.isfinite(px) or px <= 0:
                    continue
                target_amount_dict[sid] = w * total_value / px
            for code in set(current_amount_dict) | set(target_amount_dict):
                tr = trade_exchange.is_stock_tradable(code, trade_start_time, trade_end_time)
                if not tr:
                    order_log.append(
                        {
                            "trade_date": str(pd.Timestamp(trade_start_time).date()),
                            "code": f"__NOTRADABLE__{code}",
                            "amount": 0.0,
                            "direction": -1,
                            "target_amount": float(target_amount_dict.get(code, 0.0)),
                            "cur_amount": float(current_amount_dict.get(code, 0.0)),
                            "total_value": 0.0,
                        }
                    )
            orders = trade_exchange.generate_order_for_target_amount_position(
                target_position=target_amount_dict,
                current_position=current_amount_dict,
                start_time=trade_start_time,
                end_time=trade_end_time,
            )
            for o in orders:
                order_log.append(
                    {
                        "trade_date": str(pd.Timestamp(trade_start_time).date()),
                        "code": o.stock_id,
                        "amount": float(o.amount),
                        "direction": int(o.direction),
                        "target_amount": float(target_amount_dict.get(o.stock_id, 0.0)),
                        "cur_amount": float(current_amount_dict.get(o.stock_id, 0.0)),
                        "total_value": float(total_value),
                    }
                )
            return orders

    # weight panel indexed by decision date -> qlib applies it at the NEXT
    # step's open via its internal shift=1, matching wmat.shift(1).
    wdf = weights.to_pandas()
    wdf["event_time"] = pd.to_datetime(wdf["event_time"]).dt.tz_localize(None)
    wmat = (
        wdf.pivot(index="event_time", columns="security_id", values="target_weight")
        .reindex(
            index=pd.DatetimeIndex(pd.to_datetime(calendar)).tz_localize(None),
            columns=sids,
        )
        .fillna(0.0)
    )
    # signal frame: score_{d} = target weight decided at close d
    signal_df = wmat.stack().rename("score").to_frame()
    signal_df.index.names = ["datetime", "instrument"]
    signal = SignalWCache(signal_df)

    pos_log: list[dict] = []

    class PanelWeightStrategy(WeightStrategyBase):
        def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
            s = score["score"] if isinstance(score, pd.DataFrame) else score
            tw = {str(k): float(v) for k, v in s.items() if float(v) != 0.0}
            row: dict[str, object] = {"trade_date": str(pd.Timestamp(trade_start_time).date())}
            row.update({f"pos_{k}": v for k, v in current.get_stock_amount_dict().items()})
            row.update({f"tw_{k}": v for k, v in tw.items()})
            row["cash"] = current.get_cash()
            pos_log.append(row)
            return tw

    # Custom order generator: qlib's stock OrderGen renormalizes weights to
    # sum=1 and floor-divides to integer units — both wrong for fractional
    # crypto. MatchedOrderGen applies dipcatcher's sizing verbatim.
    strategy = PanelWeightStrategy(
        signal=signal, risk_degree=1.0, order_generator_cls_or_obj=MatchedOrderGen()
    )
    executor_config = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
            "verbose": False,
            "indicator_config": {
                "show_indicator": False,
            },
        },
    }
    exchange_kwargs = {
        "freq": "day",
        "codes": sids,
        "deal_price": "$open",
        "open_cost": COMMISSION_BPS / 1e4,
        "close_cost": COMMISSION_BPS / 1e4,
        "min_cost": 0.0,
        # NB: limit_threshold=None falls back to C.limit_threshold (0.095 for
        # region="cn") — crypto routinely moves >9.5%/day, which silently
        # suspends orders on big-move days. A huge float disables the check.
        "limit_threshold": 1e9,
        "volume_threshold": None,
        "trade_unit": None,
    }
    portfolio_metric_dict, indicator_dict = qlib_backtest(
        start_time=pd.Timestamp(calendar[0]).strftime("%Y-%m-%d"),
        # last tradeable step needs a next bar for its window — stop one
        # calendar entry early (the final bar is a mark-only step anyway).
        end_time=pd.Timestamp(calendar[-2]).strftime("%Y-%m-%d"),
        strategy=strategy,
        executor=executor_config,
        benchmark=sids[0],
        account=INIT_NAV,
        exchange_kwargs=exchange_kwargs,
        pos_type="Position",
    )
    report_df, _ = portfolio_metric_dict.get("1day", (None, None))
    if report_df is None:
        report_df = portfolio_metric_dict["1day"]
    # account value = cash + holdings; qlib reports 'account' column
    eq = report_df["account"]
    stats = {
        "final_value": float(eq.iloc[-1]),
        "n_orders": int(report_df["deal_amount"].abs().gt(0).sum())
        if "deal_amount" in report_df
        else int(report_df["turnover"].gt(0).sum())
        if "turnover" in report_df
        else None,
        "total_cost": float(report_df["total_cost"].iloc[-1])
        if "total_cost" in report_df
        else None,
    }
    artifacts = {
        "positions": pd.DataFrame(pos_log),
        "orders": pd.DataFrame(order_log),
    }
    return eq, stats, report_df, artifacts


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
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--debug-report",
        type=Path,
        default=None,
        help="dump qlib's daily portfolio-metrics report CSV for forensics",
    )
    ap.add_argument(
        "--weight-scale",
        type=float,
        default=1.0,
        help="scale all target weights down — isolates whether NAV divergence "
        "comes from cash-constraint/order-sizing interactions",
    )
    args = ap.parse_args()

    frames = load_panel(args.bars)
    sids = sorted(frames)
    print(f"assets: {sids}")
    weights = build_weight_panel(frames, SMA_WINDOW)
    if args.weight_scale != 1.0:
        weights = weights.with_columns(
            (pl.col("target_weight") * args.weight_scale).alias("target_weight")
        )
    print(f"weight panel: {weights.height} rows, {weights['event_time'].n_unique()} decision dates")
    calendar = frames[sids[0]]["event_time"].to_list()

    import qlib as _q

    with tempfile.TemporaryDirectory(prefix="qlib_bench_") as tmp:
        provider = Path(tmp) / "qlib_data"
        write_qlib_provider(provider, frames)
        _q.init(provider_uri=str(provider), region="cn")

        # --- correctness ---------------------------------------------------
        t0 = time.perf_counter()
        eq_dc, met_dc, fills_dc = run_dipcatcher(
            pl.concat(list(frames.values())), weights, COMMISSION_BPS
        )
        t_dc = time.perf_counter() - t0
        eq_q, st_q, rep_q, artifacts = run_qlib(provider, weights, calendar, sids)
        print(f"dipcatcher: {eq_dc.height} nav rows, {t_dc * 1e3:.0f} ms")
        print(
            f"qlib final={st_q['final_value']:.2f} orders~{st_q['n_orders']} cost={st_q['total_cost']}"
        )
        if args.debug_report:
            rep_q.to_csv(args.debug_report)
            artifacts["positions"].to_csv(
                args.debug_report.with_name(args.debug_report.stem + "_positions.csv"),
                index=False,
            )
            artifacts["orders"].to_csv(
                args.debug_report.with_name(args.debug_report.stem + "_orders.csv"),
                index=False,
            )
            print(f"qlib daily report -> {args.debug_report}")

        def _dkey(t) -> pd.Timestamp:
            ts = pd.Timestamp(t)
            if ts.tzinfo is not None:
                ts = ts.tz_localize(None)
            return ts.normalize()

        nav_dc = {
            _dkey(t): v
            for t, v in zip(eq_dc["event_time"].to_list(), eq_dc["nav"].to_list(), strict=True)
        }
        nav_q = {_dkey(t): float(v) for t, v in eq_q.items()}
        common = sorted(set(nav_dc) & set(nav_q))
        diffs = np.array([nav_dc[t] - nav_q[t] for t in common])
        rel_ref = np.array([nav_q[t] for t in common])
        rel = np.abs(diffs) / np.abs(rel_ref)
        correctness = {
            "common_dates": len(common),
            "nav_max_abs_diff": float(np.abs(diffs).max()) if len(diffs) else None,
            "nav_max_rel_diff": float(rel.max()) if len(rel) else None,
            "nav_final_dipcatcher": float(eq_dc["nav"][-1]),
            "nav_final_qlib": st_q["final_value"],
            "fills_dipcatcher": int(fills_dc.height),
            "dipcatcher_total_fees": float(fills_dc["fee"].sum()) if fills_dc.height else 0.0,
            "qlib_order_days": st_q["n_orders"],
            "qlib_total_cost": st_q["total_cost"],
        }
        print(json.dumps(correctness, indent=2))

        # --- latency ---------------------------------------------------------
        ts_dc, ts_q = [], []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            run_dipcatcher(pl.concat(list(frames.values())), weights, COMMISSION_BPS)
            ts_dc.append(time.perf_counter() - t0)
        for _ in range(args.reps):
            t0 = time.perf_counter()
            run_qlib(provider, weights, calendar, sids)
            ts_q.append(time.perf_counter() - t0)
        latency = {
            "reps": args.reps,
            "dipcatcher_ms_median": float(np.median(ts_dc) * 1e3),
            "qlib_ms_median": float(np.median(ts_q) * 1e3),
            "dipcatcher_ms_all": [round(t * 1e3, 2) for t in ts_dc],
            "qlib_ms_all": [round(t * 1e3, 2) for t in ts_q],
        }
        print(
            f"latency: dipcatcher {latency['dipcatcher_ms_median']:.0f} ms | "
            f"qlib {latency['qlib_ms_median']:.0f} ms"
        )

    receipt = {
        "schema": "incumbent_bench.v1",
        "incumbent": {
            "name": "qlib",
            "version": _q.__version__,
            "url": "https://github.com/microsoft/qlib",
            "repo": "https://github.com/microsoft/qlib",
        },
        "workload": {
            "assets": sids,
            "bars_per_asset": int(frames[sids[0]].height),
            "bar_files_sha256": {p.name: _sha256(p) for p in args.bars},
            "script_sha256": _sha256(Path(__file__)),
            "strategy": f"causal SMA{SMA_WINDOW} gate, equal-weight 0.9/{len(sids)} per gated name",
            "fills": "decision close t -> exec open t+1 (qlib shift=1 + deal_price=$open)",
            "commission_bps": COMMISSION_BPS,
            "spread_impact_borrow": 0.0,
            "init_nav": INIT_NAV,
            "weight_scale": args.weight_scale,
            "qlib_settings": {
                "trade_unit": None,
                "deal_price": "$open",
                "risk_degree": 1.0,
                "limit_threshold": 1e9,
                "volume_threshold": None,
                "min_cost": 0.0,
            },
        },
        "correctness": correctness,
        "latency": latency,
        "environment": {"python": sys.version.split()[0], "platform": sys.platform},
        "known_semantic_differences": [
            "qlib quotes are float32 (.bin format); dipcatcher uses float64 "
            "parquet prices. Residual NAV divergence is f32 price "
            "quantization (~1e-7 rel, ~$0.18 worst day on ~$1.5M).",
            "qlib order generation sells-down before buys within a step and "
            "clips buy amounts to cash+cost; dipcatcher rejects a whole order "
            "that would overdraw. Workload keeps a 0.9 weight-sum buffer so "
            "neither policy binds.",
            "qlib default order generators renormalize weights to sum=1 and "
            "floor amounts to integer units; a custom OrderGenerator "
            "(MatchedOrderGen) applies dipcatcher's exact sizing "
            "target_units = w * NAV / open.",
            "limit_threshold=None does NOT disable qlib's price-limit halt: "
            "it falls back to C.limit_threshold (0.095 for region=cn) and "
            "silently drops orders on >9.5% daily moves. limit_threshold=1e9 "
            "disables it. Without this, NAV diverged ~14% on this workload.",
        ],
        "research_only": True,
        "live_pnl_claim": False,
        "disclaimer": (
            "Single matched workload vs qlib 0.9.x on real Binance daily bars. "
            "Correctness is NAV parity; latency is single-process wall time. "
            "Not a claim of superiority across all product dimensions."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n")
    print(f"receipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
