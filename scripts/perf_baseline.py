#!/usr/bin/env python3
"""Performance-baseline regression harness for quant_fund hot paths.

Records median-of-5 wall timings for pure-CPU benchmarks over the repo's
scoring / inference / multiple-testing kernels, then compares a new run
against a stored baseline. A bench regresses when it is slower than
``threshold`` x baseline (default 1.25, in the style of pybroker's CI
gate); >threshold speedups are reported as informational ratchet
opportunities, not failures. Benches missing from either file fail the
comparison closed. Timing harness only — research diagnostic, not market
evidence; no live-PnL claim.

Usage:
    python scripts/perf_baseline.py record --output baseline.json
    python scripts/perf_baseline.py compare --baseline baseline.json --current run.json
    python scripts/perf_baseline.py compare --baseline a.json --current b.json --json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from quant_fund.metrics.anytime_fdr import e_bh  # noqa: E402
from quant_fund.metrics.conformal import conformal_quantile  # noqa: E402
from quant_fund.metrics.energy_score import energy_score  # noqa: E402
from quant_fund.metrics.inference import stationary_bootstrap_indices  # noqa: E402
from quant_fund.metrics.scoring import crps_from_quantiles  # noqa: E402
from quant_fund.metrics.snooping import reality_check  # noqa: E402
from quant_fund.utils.reproducibility import git_revision  # noqa: E402

REPEATS = 5
DEFAULT_THRESHOLD = 1.25
WATERMARK = "research-only; no live-PnL claim"

BenchFn = Callable[[np.random.Generator, int], object]


def _bench_crps_quantiles(rng: np.random.Generator, n: int) -> float:
    """CRPS-from-quantiles over a 19-point equispaced quantile grid."""
    taus = np.linspace(0.05, 0.95, 19)
    y = rng.normal(size=n)
    quantiles = y[:, None] + rng.normal(0.0, 0.5, size=(n, taus.size)) + (taus[None, :] - 0.5)
    return float(crps_from_quantiles(y, quantiles, taus))


def _bench_stationary_bootstrap(rng: np.random.Generator, n: int) -> np.ndarray:
    """Stationary bootstrap index matrix, B=1000 draws on ``n`` dates."""
    mean_block = float(min(12.0, max(1.0, n / 4.0)))
    out: NDArray[np.intp] = stationary_bootstrap_indices(n, 1000, mean_block, rng)
    return out


def _bench_reality_check_battery(rng: np.random.Generator, n: int) -> object:
    """White reality check on 6 rankers x ``n`` dates, B=1000 (INFLIGHT hot path)."""
    f = rng.normal(0.0, 1.0, size=(n, 6))
    f += np.linspace(-0.05, 0.15, 6)[None, :]
    return reality_check(f, n_boot=1000)


def _bench_conformal_quantile(rng: np.random.Generator, n: int) -> float:
    """Split-conformal quantile of ``n`` absolute residual scores."""
    scores = np.abs(rng.normal(size=n))
    return float(conformal_quantile(scores, 0.1))


def _bench_e_bh(rng: np.random.Generator, n: int) -> object:
    """e-BH at alpha=0.05 on ``n`` e-values (Exponential(1) under the null)."""
    out = None
    for _ in range(200):
        out = e_bh(rng.exponential(1.0, size=n), 0.05)
    return out


def _bench_energy_score(rng: np.random.Generator, n: int) -> float:
    """Energy score of an (n, 8) Gaussian ensemble against one observation."""
    ensemble = rng.normal(0.0, 1.0, size=(n, 8))
    observation = rng.normal(0.0, 1.0, size=8)
    return float(energy_score(ensemble, observation))


BENCHES: dict[str, BenchFn] = {
    "crps_quantiles": _bench_crps_quantiles,
    "stationary_bootstrap": _bench_stationary_bootstrap,
    "reality_check_battery": _bench_reality_check_battery,
    "conformal_quantile": _bench_conformal_quantile,
    "e_bh": _bench_e_bh,
    "energy_score": _bench_energy_score,
}

# Fixed input sizes; tests monkeypatch this dict with tiny sizes so record()
# stays fast. Keys must stay exactly in sync with BENCHES (validated at record
# and compare time — a drifted registry fails closed rather than silently
# dropping a bench).
BENCH_SIZES: dict[str, int] = {
    "crps_quantiles": 20_000,
    "stationary_bootstrap": 600,
    "reality_check_battery": 240,
    "conformal_quantile": 200_000,
    "e_bh": 2_000,
    "energy_score": 2_000,
}

# Per-bench fixed seeds so timings are reproducible run to run.
BENCH_SEEDS: dict[str, int] = {name: 20_260_927 + 1_000 * i for i, name in enumerate(BENCHES)}


class BaselineError(ValueError):
    """A baseline file or comparison input is malformed or inconsistent."""


def _check_registry() -> None:
    if set(BENCHES) != set(BENCH_SIZES) or set(BENCHES) != set(BENCH_SEEDS):
        raise BaselineError(
            f"BENCHES / BENCH_SIZES / BENCH_SEEDS out of sync: "
            f"{sorted(set(BENCHES) ^ set(BENCH_SIZES))}, "
            f"{sorted(set(BENCHES) ^ set(BENCH_SEEDS))}"
        )


def _time_bench(fn: BenchFn, seed: int, size: int) -> float:
    rng = np.random.default_rng(seed)
    timings = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        fn(rng, size)
        timings.append(time.perf_counter() - start)
    median = float(statistics.median(timings))
    if not math.isfinite(median) or median <= 0.0:
        raise RuntimeError(f"non-finite or non-positive median timing: {median!r}")
    return median


def record(output_path: Path | str) -> dict[str, Any]:
    """Run all benches and write a baseline JSON to ``output_path``."""
    _check_registry()
    benches = {}
    for name, fn in BENCHES.items():
        benches[name] = {
            "median_seconds": _time_bench(fn, BENCH_SEEDS[name], BENCH_SIZES[name]),
            "repeats": REPEATS,
        }
    payload = {
        "created_utc": datetime.now(UTC).isoformat(),
        "git_revision": git_revision(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "watermark": WATERMARK,
        "benches": benches,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _load_baseline(path: Path | str) -> dict[str, dict[str, float]]:
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BaselineError(f"cannot read {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise BaselineError(f"malformed JSON in {source}: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("benches"), dict):
        raise BaselineError(f"{source}: expected an object with a 'benches' mapping")
    out: dict[str, dict[str, float]] = {}
    for name, entry in raw["benches"].items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise BaselineError(f"{source}: bench entries must map names to objects")
        median = entry.get("median_seconds")
        if not isinstance(median, (int, float)) or isinstance(median, bool):
            raise BaselineError(f"{source}: {name}.median_seconds must be a number")
        if not math.isfinite(float(median)) or float(median) <= 0.0:
            raise BaselineError(f"{source}: {name}.median_seconds must be finite and > 0")
        repeats = entry.get("repeats", REPEATS)
        if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
            raise BaselineError(f"{source}: {name}.repeats must be a positive integer")
        out[name] = {"median_seconds": float(median), "repeats": repeats}
    return out


@dataclass(frozen=True)
class CompareResult:
    """Outcome of comparing a current run against a stored baseline.

    ``regressions`` and ``speedups`` are tuples of
    (name, baseline_seconds, current_seconds, ratio). ``missing`` holds bench
    names present in exactly one file. ``passed`` is False when any
    regression or any missing bench exists (fail closed); informational
    speedups never fail the comparison.
    """

    regressions: tuple[tuple[str, float, float, float], ...]
    speedups: tuple[tuple[str, float, float, float], ...]
    missing: tuple[str, ...]
    passed: bool


def compare(
    baseline_path: Path | str,
    current_path: Path | str,
    threshold: float = DEFAULT_THRESHOLD,
) -> CompareResult:
    """Compare two baseline JSONs; a bench regresses iff current > threshold * baseline."""
    _check_registry()
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError(f"threshold must be a positive finite number, got {threshold!r}")
    baseline = _load_baseline(baseline_path)
    current = _load_baseline(current_path)
    names = set(baseline) | set(current)
    unknown = sorted(names - set(BENCHES))
    if unknown:
        raise BaselineError(f"unknown bench names (registry drift?): {unknown}")
    missing = tuple(sorted(set(baseline) ^ set(current)))
    regressions: list[tuple[str, float, float, float]] = []
    speedups: list[tuple[str, float, float, float]] = []
    for name in sorted(set(baseline) & set(current)):
        base = baseline[name]["median_seconds"]
        cur = current[name]["median_seconds"]
        ratio = cur / base
        row = (name, base, cur, ratio)
        if ratio > threshold:
            regressions.append(row)
        elif ratio < 1.0 / threshold:
            speedups.append(row)
    return CompareResult(
        regressions=tuple(regressions),
        speedups=tuple(speedups),
        missing=missing,
        passed=not regressions and not missing,
    )


def _format_compare(result: CompareResult) -> str:
    lines = []
    for name, base, cur, ratio in result.regressions:
        lines.append(f"REGRESSION {name}: {base:.6f}s -> {cur:.6f}s ({ratio:.2f}x)")
    for name, base, cur, ratio in result.speedups:
        lines.append(f"speedup    {name}: {base:.6f}s -> {cur:.6f}s ({ratio:.2f}x)")
    for name in result.missing:
        lines.append(f"missing    {name}")
    lines.append("PASS" if result.passed else "FAIL")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record", help="run all benches and write a baseline JSON")
    rec.add_argument("--output", type=Path, required=True)
    cmp_parser = sub.add_parser("compare", help="compare a current run against a baseline")
    cmp_parser.add_argument("--baseline", type=Path, required=True)
    cmp_parser.add_argument("--current", type=Path, required=True)
    cmp_parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    cmp_parser.add_argument("--json", action="store_true", help="print CompareResult as JSON")
    args = parser.parse_args(argv)

    try:
        if args.command == "record":
            record(args.output)
            print(f"wrote {args.output}")
            return 0
        result = compare(args.baseline, args.current, threshold=args.threshold)
        if args.json:
            print(json.dumps(asdict(result), indent=2))
        else:
            print(_format_compare(result))
        return 0 if result.passed else 1
    except (BaselineError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
