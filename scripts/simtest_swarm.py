"""Run a deterministic-simulation swarm.

The schedule is synthetic. The report counts invariant failures and trace
sizes. It does not emit return, risk, or live-P&L headlines.

Examples:
  uv run python scripts/simtest_swarm.py --seeds 48 --days 5
  uv run python scripts/simtest_swarm.py --seeds 4000 --days 8
"""

from __future__ import annotations

import argparse
import json
import sys

from quant_fund.simtest.swarm import run_swarm


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seeded fault swarm for the paper session")
    parser.add_argument("--seeds", type=int, default=48)
    parser.add_argument("--days", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--max-faults", type=int, default=3)
    parser.add_argument("--no-shrink", action="store_true")
    args = parser.parse_args(argv)
    report = run_swarm(
        args.seeds,
        n_days=args.days,
        base_seed=args.base_seed,
        max_faults=args.max_faults,
        shrink=not args.no_shrink,
    )
    payload = {
        "n_seeds": report.n_seeds,
        "n_days": report.n_days,
        "max_faults": report.max_faults,
        "failures": len(report.failures),
        "elapsed_seconds": report.elapsed_seconds,
        "sessions_per_second": report.sessions_per_second,
        "outcome_counts": report.outcome_counts,
        "log_bytes_min": report.log_bytes_min,
        "log_bytes_max": report.log_bytes_max,
        "research_only": True,
        "live_pnl_claim": False,
        "data_source": "SYNTHETIC",
        "failure_details": [
            {
                "seed": item.seed,
                "reason": item.reason,
                "schedule": list(item.schedule),
                "minimized": list(item.minimized),
            }
            for item in report.failures
        ],
    }
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
