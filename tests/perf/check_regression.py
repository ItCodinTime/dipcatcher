"""Fail when a benchmark's calibration-normalized time regresses.

Absolute milliseconds are not comparable across GitHub-hosted runners and a
developer machine. Each benchmark is scored as::

    ratio = median_seconds(benchmark) / median_seconds(test_calibration_matmul)

and compared with the same ratio in the committed baseline. A benchmark fails
when ``ratio > baseline_ratio * (1 + threshold)``.

The baseline JSON is whatever ``pytest-benchmark --benchmark-json`` wrote,
including ``machine_info``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _benchmarks(path: Path) -> tuple[dict[str, Any], dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    machine = payload.get("machine_info", {})
    medians: dict[str, float] = {}
    for row in payload.get("benchmarks", []):
        name = str(row.get("fullname") or row.get("name"))
        stats = row.get("stats") or {}
        median = stats.get("median")
        if median is None:
            raise SystemExit(f"{path}: {name} has no stats.median")
        medians[name] = float(median)
    return machine, medians


# Unmarked tests/perf cases. The CI job passes --ci so dropping one fails the gate.
_CI_SUFFIXES = (
    "test_calibration_matmul",
    "test_cross_section[24sym-252d]",
    "test_build_features[24sym-252d]",
    "test_resample[1sym-1y-5m]",
    "test_resample[1sym-1y-1d]",
    "test_security_master[8sym-20d]",
    "test_read_ohlcv_1m[8sym-20d]",
    "test_backtest[8sym-126d]",
    "test_dip_detect[40sym-1000d]",
    "test_dip_bench[20sym-800d]",
)


def _calibration_key(medians: dict[str, float]) -> str:
    matches = [name for name in medians if name.endswith("test_calibration_matmul")]
    if len(matches) != 1:
        raise SystemExit(f"expected one calibration benchmark, found {matches}")
    return matches[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("current", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument(
        "--ci",
        action="store_true",
        help="require the fast-subset benchmark ids (see _CI_SUFFIXES)",
    )
    args = parser.parse_args(argv)
    if args.threshold < 0:
        raise SystemExit("threshold must be >= 0")
    current_machine, current = _benchmarks(args.current)
    baseline_machine, baseline = _benchmarks(args.baseline)
    current_key = _calibration_key(current)
    baseline_key = _calibration_key(baseline)
    current_cal = current[current_key]
    baseline_cal = baseline[baseline_key]
    if current_cal <= 0 or baseline_cal <= 0:
        raise SystemExit("calibration median must be positive")

    print(f"threshold: {args.threshold:.0%} above the calibration-normalized baseline")
    print(f"current machine: {_machine_line(current_machine)}")
    print(f"baseline machine: {_machine_line(baseline_machine)}")
    print(
        f"calibration median: current {current_cal * 1e3:.3f} ms, baseline {baseline_cal * 1e3:.3f} ms"
    )
    print()
    header = f"{'benchmark':<72} {'cur_ms':>10} {'base_ms':>10} {'ratio':>8} {'limit':>8}"
    print(header)
    failures: list[str] = []
    if args.ci:
        for suffix in _CI_SUFFIXES:
            if not any(suffix in name for name in current):
                failures.append(f"CI benchmark missing from this run: {suffix}")
    shared = sorted(
        name for name in current if name in baseline and name not in {current_key, baseline_key}
    )
    for name in sorted(current):
        if name in baseline or name.endswith("test_calibration_matmul"):
            continue
        failures.append(f"no baseline entry for {name}")
    for name in shared:
        if name.endswith("test_calibration_matmul"):
            continue
        cur_ratio = current[name] / current_cal
        base_ratio = baseline[name] / baseline_cal
        limit = base_ratio * (1.0 + args.threshold)
        flag = ""
        if cur_ratio > limit:
            flag = " FAIL"
            failures.append(
                f"{name}: ratio {cur_ratio:.3f} > limit {limit:.3f} "
                f"(current {current[name] * 1e3:.3f} ms, baseline {baseline[name] * 1e3:.3f} ms)"
            )
        short = name.split("::")[-1]
        print(
            f"{short:<72} {current[name] * 1e3:10.3f} {baseline[name] * 1e3:10.3f} "
            f"{cur_ratio / base_ratio:8.3f} {1.0 + args.threshold:8.3f}{flag}"
        )
    print()
    if failures:
        print("regressions:")
        for line in failures:
            print(f"  {line}")
        return 1
    print("no calibration-normalized regressions")
    return 0


def _machine_line(info: dict[str, Any]) -> str:
    cpu = info.get("cpu") or {}
    brand = cpu.get("brand_raw") or cpu.get("arch") or "unknown cpu"
    python = info.get("python_version") or "unknown python"
    system = info.get("system") or info.get("platform") or ""
    return f"{brand}; python {python}; {system}"


if __name__ == "__main__":
    sys.exit(main())
