"""Parity harness CLI.

    uv run --no-sync python -m quant_fund.research.parity run \
        --strategy buy_hold_equal --out artifacts/parity
    uv run --no-sync python -m quant_fund.research.parity shootout \
        --symbols 50 --days 1260 --reps 3

All runs use the labeled SYNTHETIC generator; outputs carry
``research_only``/``live_pnl_claim`` flags downstream consumers must not strip.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from quant_fund.research.parity.canonical import EngineRun
from quant_fund.research.parity.engines import KNOWN_ENGINES, available_engines, run_engine
from quant_fund.research.parity.reconcile import Tolerance, reconcile_all
from quant_fund.research.parity.report import write_report
from quant_fund.research.parity.shootout import time_engines
from quant_fund.research.parity.strategies import STRATEGIES, build_plan
from quant_fund.research.parity.synthetic import make_synthetic_panel


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="parity", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "shootout"):
        sp = sub.add_parser(name)
        sp.add_argument("--symbols", type=int, default=5)
        sp.add_argument("--days", type=int, default=252)
        sp.add_argument("--seed", type=int, default=7)
        sp.add_argument("--initial-nav", type=float, default=1_000_000.0)
        sp.add_argument("--commission-bps", type=float, default=0.0)
        sp.add_argument("--strategy", action="append", default=None)
        sp.add_argument("--engine", action="append", default=None)
        sp.add_argument("--out", type=Path, default=Path("artifacts/parity"))
        sp.add_argument("--reps", type=int, default=3)
        sp.add_argument("--all", action="store_true", help="all strategies+engines")
    return p


def _select(values: list[str] | None, known: tuple[str, ...] | dict[str, object]) -> list[str]:
    if not values:
        return list(known)
    for v in values:
        if v not in known:
            raise SystemExit(f"unknown {v!r}; have {sorted(known)}")
    return values


def cmd_run(args: argparse.Namespace) -> int:
    panel = make_synthetic_panel(
        n_symbols=args.symbols, n_days=args.days, seed=args.seed
    )
    engines = _select(args.engine, KNOWN_ENGINES)
    strategies = _select(args.strategy, STRATEGIES)
    runs: dict[str, EngineRun] = {}
    for sname in strategies:
        plan = build_plan(sname, panel)
        for engine in engines:
            runs[f"{engine}:{sname}"] = run_engine(
                engine,
                panel,
                plan,
                initial_nav=args.initial_nav,
                commission_bps=args.commission_bps,
            )
    reports = reconcile_all(runs, reference="native", tol=Tolerance())
    out = write_report(
        args.out / f"parity_s{args.symbols}_d{args.days}_seed{args.seed}.md",
        runs=runs,
        reports=reports,
        scenario=f"commission_bps={args.commission_bps}",
        panel_desc=(
            f"{args.symbols} symbols x {args.days} daily bars, seed={args.seed}, "
            f"initial_nav={args.initial_nav:g}"
        ),
    )
    print(f"wrote {out}")
    for rep in reports:
        print(json.dumps(rep.summary(), default=str))
    bad = [r for r in reports if r.verdict == "UNEXPLAINED"]
    return 1 if bad else 0


def cmd_shootout(args: argparse.Namespace) -> int:
    panel = make_synthetic_panel(
        n_symbols=args.symbols, n_days=args.days, seed=args.seed
    )
    engines = _select(args.engine, KNOWN_ENGINES)
    strategies = _select(args.strategy, STRATEGIES)
    plans = [build_plan(s, panel) for s in strategies]
    result = time_engines(
        panel,
        plans,
        engines,
        initial_nav=args.initial_nav,
        commission_bps=args.commission_bps,
        reps=args.reps,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / f"shootout_s{args.symbols}_d{args.days}_seed{args.seed}.json"
    result["label"] = "SYNTHETIC"
    result["research_only"] = True
    result["live_pnl_claim"] = False
    result["engines_available"] = available_engines()
    result["panel"] = f"{args.symbols}x{args.days} seed={args.seed}"
    dest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {dest}")
    for row in result["rows"]:
        print(
            f"{row['engine']:>10} {row['strategy']:>16} {row['status']:>5} "
            f"first={row['first']:.4f}s min={row['min']:.4f}s median={row['median']:.4f}s"
            if row["status"] == "ok"
            else f"{row['engine']:>10} {row['strategy']:>16} {row['status']:>5} {row['notes']}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "shootout":
        return cmd_shootout(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
