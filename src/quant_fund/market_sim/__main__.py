"""Command line for the agent-market simulator.

``python -m quant_fund.market_sim report`` writes the measurement JSON used
by ``docs/MARKET_SIM.md``. Output is a simulation diagnostic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_fund.market_sim.harness import (
    lightspeed_momentum_weight,
    mean_reversion_weight,
    stress_strategy,
)
from quant_fund.market_sim.native import matching_benchmark
from quant_fund.market_sim.scenarios import SCENARIOS, run_scenario, scenario_spread
from quant_fund.market_sim.stylized import run_stylized_validation
from quant_fund.utils.atomicio import atomic_write_text


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _scenario_blob(name: str) -> dict[str, object]:
    result = run_scenario(name)
    blob: dict[str, object] = dict(result.summary())
    blob["hook"] = result.hook
    return blob


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m quant_fund.market_sim")
    sub = parser.add_subparsers(dest="cmd", required=True)

    bench = sub.add_parser("bench", help="time the matching core")
    bench.add_argument("--events", type=int, default=1_000_000)
    bench.add_argument("--seed", type=int, default=1)

    stylized = sub.add_parser("stylized", help="run the pre-registered fact tests")
    stylized.add_argument("--skip-impact", action="store_true")

    sub.add_parser("scenarios", help="run the five scenarios at the default length")
    sub.add_parser("stress", help="stress momentum and mean-reversion weights")

    report = sub.add_parser("report", help="bench, stylized facts, scenarios, and stress")
    report.add_argument("--events", type=int, default=1_000_000)
    report.add_argument("--out", type=Path, default=None)

    args = parser.parse_args(argv)
    if args.cmd == "bench":
        _print(matching_benchmark(args.events, args.seed))
        return
    if args.cmd == "stylized":
        _print(run_stylized_validation(measure=not args.skip_impact))
        return
    if args.cmd == "scenarios":
        payload: dict[str, object] = {name: _scenario_blob(name) for name in SCENARIOS}
        payload["spread"] = scenario_spread()
        _print(payload)
        return
    if args.cmd == "stress":
        _print(
            {
                "momentum": stress_strategy(lightspeed_momentum_weight, name="lightspeed_momentum"),
                "mean_reversion": stress_strategy(mean_reversion_weight, name="mean_reversion"),
            }
        )
        return
    if args.cmd == "report":
        payload = {
            "benchmark": matching_benchmark(args.events, 1),
            "stylized": run_stylized_validation(),
            "scenarios": {name: _scenario_blob(name) for name in SCENARIOS},
            "spread": scenario_spread(),
            "stress": {
                "momentum": stress_strategy(lightspeed_momentum_weight, name="lightspeed_momentum"),
                "mean_reversion": stress_strategy(mean_reversion_weight, name="mean_reversion"),
            },
        }
        text = json.dumps(payload, indent=2, sort_keys=True, default=str)
        if args.out is not None:
            atomic_write_text(args.out, text)
        print(text)
        return
    raise SystemExit(f"unknown command {args.cmd}")


if __name__ == "__main__":
    main()
