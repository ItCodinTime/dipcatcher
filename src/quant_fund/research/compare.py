"""Paired comparison of two research runs (receipt JSONs or result dirs).

Researchers accumulate sealed receipts and result directories whose scores are
per-fold or per-observation series (e.g. ``four_chronological_fold_means`` in
``basis_reversion_screen`` receipts, ``latency.*_ms_all`` in incumbent benches,
``metrics.*`` scalars in dip-bench receipts). Comparing two runs by hand is
error-prone; this module:

1. loads a run from a receipt JSON or a directory of JSONs,
2. diffs their config-like subtrees and scalar fields,
3. aligns same-named numeric series and runs paired inference on the
   per-observation delta (Diebold–Mariano with the existing Newey–West HAC
   implementation, plus a stationary-bootstrap CI on the mean delta),
4. emits a Markdown or JSON report, or a ``run_compare.v1`` receipt blob.

Honesty contract: headline verdicts use proper score deltas only. Scalar fields
whose key tokens collide with ``FORBIDDEN_RESEARCH_METRIC_KEYS`` (sharpe,
sortino, calmar, pnl, nav) are reported under ``excluded_diagnostics`` and are
never part of a verdict. Runs with too few paired observations get an explicit
``insufficient paired observations`` verdict rather than a silent pass.

Usage::

    python -m quant_fund.research.compare run_a.json run_b.json
    python -m quant_fund.research.compare a_dir/ b_dir/ --format json --out c.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.inference import (
    diebold_mariano,
    optimal_block_length,
    stationary_bootstrap_indices,
)
from quant_fund.research.catalog import FORBIDDEN_RESEARCH_METRIC_KEYS

Array = NDArray[np.float64]

COMPARE_RECEIPT_SCHEMA = "run_compare.v1"

# Top-level receipt keys treated as configuration/context rather than results.
_CONFIG_ROOT_KEYS = (
    "params",
    "config",
    "allocator",
    "workload",
    "environment",
    "inputs",
    "inputs_sha256",
    "input_hashes",
    "data",
    "data_source",
)

_VERDICT_INSUFFICIENT = "insufficient paired observations"
_VERDICT_UNALIGNED = "unaligned series lengths"
_VERDICT_NO_EFFECT = "no_effect"
_VERDICT_NO_DIFF = "no significant difference"
_VERDICT_A_BETTER = "a_better"
_VERDICT_B_BETTER = "b_better"
_VERDICT_AMBIGUOUS = "ambiguous"
_VERDICT_INCONCLUSIVE = "inconclusive"


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _leaf_tokens(key: str) -> set[str]:
    return {tok for tok in str(key).lower().replace("-", "_").split("_") if tok}


def _is_forbidden_key(path: str) -> bool:
    """True when any dot-path segment carries a forbidden headline token."""
    for segment in path.split("."):
        if _leaf_tokens(segment) & FORBIDDEN_RESEARCH_METRIC_KEYS:
            return True
    return False


def _flatten_scalars(obj: Any, prefix: str = "") -> dict[str, float]:
    """Flatten every finite numeric leaf into ``dotted.path -> value``."""
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for key in sorted(obj, key=str):
            out.update(_flatten_scalars(obj[key], f"{prefix}{key}."))
    elif isinstance(obj, (list, tuple)):
        # Numeric lists are series, not scalars; walk only non-numeric items.
        if not any(_is_number(item) for item in obj):
            for index, item in enumerate(obj):
                out.update(_flatten_scalars(item, f"{prefix}{index}."))
    elif _is_number(obj):
        out[prefix.rstrip(".")] = float(obj)
    return out


def _flatten_series(obj: Any, prefix: str = "") -> dict[str, list[float]]:
    """Collect flat numeric lists (len >= 2) as per-observation series."""
    out: dict[str, list[float]] = {}
    if isinstance(obj, dict):
        for key in sorted(obj, key=str):
            out.update(_flatten_series(obj[key], f"{prefix}{key}."))
    elif isinstance(obj, (list, tuple)):
        if len(obj) >= 2 and all(_is_number(item) for item in obj):
            out[prefix.rstrip(".")] = [float(item) for item in obj]
        else:
            for index, item in enumerate(obj):
                out.update(_flatten_series(item, f"{prefix}{index}."))
    return out


def _flatten_leaves(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten config subtrees into ``dotted.path -> json-able leaf``."""
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key in sorted(obj, key=str):
            out.update(_flatten_leaves(obj[key], f"{prefix}{key}."))
    elif isinstance(obj, (list, tuple)):
        if all(not isinstance(item, (dict, list, tuple)) for item in obj):
            out[prefix.rstrip(".")] = list(obj)
        else:
            for index, item in enumerate(obj):
                out.update(_flatten_leaves(item, f"{prefix}{index}."))
    else:
        out[prefix.rstrip(".")] = obj
    return out


@dataclass(frozen=True)
class RunData:
    """One loaded research run: scalar fields, series, and config roots."""

    label: str
    path: str
    payload: dict[str, Any]
    scalars: dict[str, float]
    excluded_scalars: dict[str, float]
    series: dict[str, list[float]]
    config: dict[str, Any]

    @property
    def schema(self) -> Any:
        return self.payload.get("schema") or self.payload.get("schema_version")


def _merge_dir_payload(path: Path) -> dict[str, Any]:
    """Load a result dir: the single JSON inside, or all JSONs keyed by stem."""
    files = sorted(p for p in path.glob("*.json") if p.is_file())
    if not files:
        raise ValueError(f"no *.json results under {path}")
    if len(files) == 1:
        payload = json.loads(files[0].read_text())
    else:
        payload = {f.stem: json.loads(f.read_text()) for f in files}
    if not isinstance(payload, dict):
        raise TypeError(f"result payload at {path} must be a JSON object")
    return payload


def load_run(path: str | Path) -> RunData:
    """Load one run from a receipt JSON file or a directory of JSON files."""
    p = Path(path)
    if p.is_dir():
        payload = _merge_dir_payload(p)
    elif p.is_file():
        payload = json.loads(p.read_text())
        if not isinstance(payload, dict):
            raise TypeError(f"receipt at {p} must be a JSON object")
    else:
        raise FileNotFoundError(f"run path does not exist: {p}")

    scalars: dict[str, float] = {}
    excluded: dict[str, float] = {}
    for key, value in _flatten_scalars(payload).items():
        if _is_forbidden_key(key):
            excluded[key] = value
        elif key.split(".", 1)[0] not in _CONFIG_ROOT_KEYS:
            # Config-root scalars are reported in the config diff, not as
            # metric deltas.
            scalars[key] = value
    series = {
        key: values
        for key, values in _flatten_series(payload).items()
        if not _is_forbidden_key(key)
    }
    config = {key: payload[key] for key in _CONFIG_ROOT_KEYS if key in payload}
    return RunData(
        label=p.stem,
        path=str(p),
        payload=payload,
        scalars=scalars,
        excluded_scalars=excluded,
        series=series,
        config=config,
    )


def _delta_bootstrap_ci(
    delta: Array,
    *,
    n_boot: int,
    alpha: float,
    seed: int,
) -> tuple[float, float, float]:
    """Stationary-bootstrap percentile CI for the mean of ``delta``.

    Block length comes from the Politis–White automatic selector; the seeded
    index matrix keeps the interval deterministic and sign-symmetric under
    run-order swaps. Returns ``(lo, hi, mean_block)``; NaN bounds when n < 5.
    """
    d = np.asarray(delta, dtype=float)
    d = d[np.isfinite(d)]
    n = int(d.size)
    if n < 5:
        return float("nan"), float("nan"), float("nan")
    mean_block = optimal_block_length(d)
    if not math.isfinite(mean_block) or mean_block < 1.0:
        mean_block = float(max(1.0, round(n ** (1.0 / 3.0))))
    mean_block = float(min(max(mean_block, 1.0), float(n)))
    rng = np.random.default_rng(int(seed))
    idx = stationary_bootstrap_indices(n, int(n_boot), mean_block, rng)
    stats_boot = d[idx].mean(axis=1)
    lo = float(np.quantile(stats_boot, alpha / 2.0))
    hi = float(np.quantile(stats_boot, 1.0 - alpha / 2.0))
    return lo, hi, mean_block


@dataclass(frozen=True)
class SeriesComparison:
    """Paired inference on one aligned score series (delta = a - b)."""

    key: str
    n_a: int
    n_b: int
    n_paired: int
    mean_a: float
    mean_b: float
    mean_delta: float
    dm_stat: float
    dm_p: float
    dm_lags: int
    bootstrap_lo: float
    bootstrap_hi: float
    bootstrap_block: float
    verdict: str
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "n_a": self.n_a,
            "n_b": self.n_b,
            "n_paired": self.n_paired,
            "mean_a": _finite_or_none(self.mean_a),
            "mean_b": _finite_or_none(self.mean_b),
            "mean_delta": _finite_or_none(self.mean_delta),
            "dm_stat": _finite_or_none(self.dm_stat),
            "dm_p": _finite_or_none(self.dm_p),
            "dm_lags": self.dm_lags,
            "bootstrap_lo": _finite_or_none(self.bootstrap_lo),
            "bootstrap_hi": _finite_or_none(self.bootstrap_hi),
            "bootstrap_block": _finite_or_none(self.bootstrap_block),
            "verdict": self.verdict,
            "note": self.note,
        }


def compare_series(
    key: str,
    a: list[float],
    b: list[float],
    *,
    higher_is_better: bool = False,
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int = 7,
    min_paired: int = 10,
    name_a: str = "a",
    name_b: str = "b",
) -> SeriesComparison:
    """Paired comparison of one score series.

    ``delta = a - b``. With ``higher_is_better=False`` (loss convention) a
    negative delta favors ``a``; with ``higher_is_better=True`` a positive
    delta favors ``a``. n < ``min_paired`` is reported honestly as
    ``insufficient paired observations`` — inference is not run.
    """
    n_a, n_b = len(a), len(b)
    mean_a = float(np.mean(a)) if n_a else float("nan")
    mean_b = float(np.mean(b)) if n_b else float("nan")
    mean_delta = mean_a - mean_b

    def _base(verdict: str, note: str) -> SeriesComparison:
        return SeriesComparison(
            key=key,
            n_a=n_a,
            n_b=n_b,
            n_paired=0,
            mean_a=mean_a,
            mean_b=mean_b,
            mean_delta=mean_delta,
            dm_stat=float("nan"),
            dm_p=float("nan"),
            dm_lags=0,
            bootstrap_lo=float("nan"),
            bootstrap_hi=float("nan"),
            bootstrap_block=float("nan"),
            verdict=verdict,
            note=note,
        )

    if n_a != n_b or n_a < 2:
        return _base(
            _VERDICT_UNALIGNED,
            f"cannot pair: len(a)={n_a}, len(b)={n_b}",
        )
    n = n_a
    if n < min_paired:
        return SeriesComparison(
            key=key,
            n_a=n_a,
            n_b=n_b,
            n_paired=n,
            mean_a=mean_a,
            mean_b=mean_b,
            mean_delta=mean_delta,
            dm_stat=float("nan"),
            dm_p=float("nan"),
            dm_lags=0,
            bootstrap_lo=float("nan"),
            bootstrap_hi=float("nan"),
            bootstrap_block=float("nan"),
            verdict=_VERDICT_INSUFFICIENT,
            note=f"n={n} < min_paired={min_paired}; no inference run",
        )
    if not (math.isfinite(mean_a) and math.isfinite(mean_b)):
        return _base(_VERDICT_INCONCLUSIVE, "non-finite values after flattening")

    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    dm = diebold_mariano(a_arr, b_arr, name_a=name_a, name_b=name_b)
    delta = a_arr - b_arr
    lo, hi, block = _delta_bootstrap_ci(delta, n_boot=n_boot, alpha=alpha, seed=seed)

    if bool(np.all(delta == 0.0)):
        verdict = _VERDICT_NO_EFFECT
        note = "identical paired series"
    elif not math.isfinite(dm.statistic) or not math.isfinite(dm.p_value):
        verdict = _VERDICT_INCONCLUSIVE
        note = "degenerate delta variance; HAC t undefined"
    else:
        significant_dm = dm.p_value < alpha
        ci_excludes_zero = math.isfinite(lo) and math.isfinite(hi) and (lo > 0.0 or hi < 0.0)
        favors_a = mean_delta > 0.0 if higher_is_better else mean_delta < 0.0
        if significant_dm and ci_excludes_zero:
            verdict = _VERDICT_A_BETTER if favors_a else _VERDICT_B_BETTER
            note = f"DM p={dm.p_value:.4g} and bootstrap CI excludes 0"
        elif not significant_dm and not ci_excludes_zero:
            verdict = _VERDICT_NO_DIFF
            note = f"DM p={dm.p_value:.4g} >= {alpha}; CI covers 0"
        else:
            verdict = _VERDICT_AMBIGUOUS
            note = "DM and bootstrap CI disagree; treat as inconclusive"

    return SeriesComparison(
        key=key,
        n_a=n_a,
        n_b=n_b,
        n_paired=n,
        mean_a=mean_a,
        mean_b=mean_b,
        mean_delta=mean_delta,
        dm_stat=dm.statistic,
        dm_p=dm.p_value,
        dm_lags=dm.lags,
        bootstrap_lo=lo,
        bootstrap_hi=hi,
        bootstrap_block=block,
        verdict=verdict,
        note=note,
    )


def _diff_config(a: RunData, b: RunData) -> list[dict[str, Any]]:
    leaves_a = _flatten_leaves(a.config)
    leaves_b = _flatten_leaves(b.config)
    diffs: list[dict[str, Any]] = []
    for key in sorted(set(leaves_a) | set(leaves_b)):
        va = leaves_a.get(key, "<absent>")
        vb = leaves_b.get(key, "<absent>")
        if va != vb:
            diffs.append({"path": key, "a": va, "b": vb})
    return diffs


def _diff_scalars(a: RunData, b: RunData) -> list[dict[str, Any]]:
    diffs: list[dict[str, Any]] = []
    for key in sorted(set(a.scalars) | set(b.scalars)):
        va = a.scalars.get(key)
        vb = b.scalars.get(key)
        if va is None or vb is None:
            diffs.append({"path": key, "a": va, "b": vb, "delta": None})
        elif not math.isclose(va, vb, rel_tol=0.0, abs_tol=0.0):
            diffs.append({"path": key, "a": va, "b": vb, "delta": va - vb})
    return diffs


@dataclass
class RunComparison:
    """Full comparison of two runs."""

    label_a: str
    label_b: str
    schema_a: Any
    schema_b: Any
    higher_is_better: bool
    alpha: float
    min_paired: int
    n_boot: int
    seed: int
    config_diff: list[dict[str, Any]] = field(default_factory=list)
    metric_deltas: list[dict[str, Any]] = field(default_factory=list)
    excluded_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    series: list[SeriesComparison] = field(default_factory=list)
    series_only_in_a: list[str] = field(default_factory=list)
    series_only_in_b: list[str] = field(default_factory=list)

    @property
    def verdict_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for comp in self.series:
            counts[comp.verdict] = counts.get(comp.verdict, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_a": self.label_a,
            "run_b": self.label_b,
            "schema_a": self.schema_a,
            "schema_b": self.schema_b,
            "higher_is_better": self.higher_is_better,
            "alpha": self.alpha,
            "min_paired": self.min_paired,
            "n_boot": self.n_boot,
            "seed": self.seed,
            "convention": (
                "delta = a - b; positive delta favors a"
                if self.higher_is_better
                else "delta = a - b; negative delta favors a (loss convention)"
            ),
            "config_diff": self.config_diff,
            "metric_deltas": self.metric_deltas,
            "excluded_diagnostics": self.excluded_diagnostics,
            "series": [comp.to_dict() for comp in self.series],
            "series_only_in_a": self.series_only_in_a,
            "series_only_in_b": self.series_only_in_b,
            "verdict_counts": self.verdict_counts,
        }

    def to_markdown(self) -> str:
        lines = [
            "# Run comparison",
            "",
            f"- **A**: `{self.label_a}` (schema: {self.schema_a})",
            f"- **B**: `{self.label_b}` (schema: {self.schema_b})",
            f"- Convention: {self.to_dict()['convention']}",
            f"- alpha={self.alpha}, min_paired={self.min_paired}, "
            f"n_boot={self.n_boot}, seed={self.seed}",
            "",
            "## Verdicts",
            "",
        ]
        counts = self.verdict_counts
        if counts:
            for verdict in sorted(counts):
                lines.append(f"- `{verdict}`: {counts[verdict]}")
        else:
            lines.append("- no aligned series found")
        lines += ["", "## Paired series", ""]
        if self.series:
            lines.append(
                "| series | n | mean_a | mean_b | delta | DM t | DM p | bootstrap CI | verdict |"
            )
            lines.append("|---|---|---|---|---|---|---|---|---|")
            for comp in self.series:
                lines.append(
                    f"| `{comp.key}` | {comp.n_paired} | {_fmt(comp.mean_a)} "
                    f"| {_fmt(comp.mean_b)} | {_fmt(comp.mean_delta)} "
                    f"| {_fmt(comp.dm_stat)} | {_fmt(comp.dm_p)} "
                    f"| [{_fmt(comp.bootstrap_lo)}, {_fmt(comp.bootstrap_hi)}] "
                    f"| {comp.verdict} |"
                )
        else:
            lines.append("_none_")
        lines += ["", "## Metric deltas (scalar fields)", ""]
        if self.metric_deltas:
            lines.append("| field | a | b | delta |")
            lines.append("|---|---|---|---|")
            for row in self.metric_deltas:
                delta = row["delta"]
                delta_text = f"{delta:g}" if isinstance(delta, float) else "n/a"
                lines.append(
                    f"| `{row['path']}` | {_fmtv(row['a'])} | {_fmtv(row['b'])} | {delta_text} |"
                )
        else:
            lines.append("_none_")
        if self.excluded_diagnostics:
            lines += [
                "",
                "## Excluded diagnostics (not research evidence)",
                "",
                "Fields matching forbidden headline tokens (sharpe/sortino/"
                "calmar/pnl/nav) are listed for context only and never enter "
                "a verdict.",
                "",
            ]
            for row in self.excluded_diagnostics:
                lines.append(f"- `{row['path']}`: a={_fmtv(row['a'])}, b={_fmtv(row['b'])}")
        lines += ["", "## Config diff", ""]
        if self.config_diff:
            lines.append("| path | a | b |")
            lines.append("|---|---|---|")
            for row in self.config_diff:
                lines.append(f"| `{row['path']}` | {_fmtv(row['a'])} | {_fmtv(row['b'])} |")
        else:
            lines.append("_identical config roots_")
        if self.series_only_in_a or self.series_only_in_b:
            lines += ["", "## Unpaired series", ""]
            for key in self.series_only_in_a:
                lines.append(f"- `{key}`: only in A")
            for key in self.series_only_in_b:
                lines.append(f"- `{key}`: only in B")
        lines += [
            "",
            "_Research diagnostic only. Not live-trading evidence; proper scores only._",
            "",
        ]
        return "\n".join(lines)


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _fmt(value: float) -> str:
    return f"{value:.6g}" if math.isfinite(value) else "nan"


def _fmtv(value: Any) -> str:
    if isinstance(value, float):
        return _fmt(value)
    if isinstance(value, list):
        return "[" + ", ".join(_fmtv(v) for v in value) + "]"
    return str(value)


def compare_runs(
    path_a: str | Path,
    path_b: str | Path,
    *,
    higher_is_better: bool = False,
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int = 7,
    min_paired: int = 10,
    series_keys: list[str] | None = None,
) -> RunComparison:
    """Load two runs and produce a paired comparison report."""
    run_a = load_run(path_a)
    run_b = load_run(path_b)

    keys_a = set(run_a.series)
    keys_b = set(run_b.series)
    if series_keys is not None:
        wanted = set(series_keys)
        common = sorted(keys_a & keys_b & wanted)
    else:
        common = sorted(keys_a & keys_b)

    comparisons = [
        compare_series(
            key,
            run_a.series[key],
            run_b.series[key],
            higher_is_better=higher_is_better,
            alpha=alpha,
            n_boot=n_boot,
            seed=seed,
            min_paired=min_paired,
            name_a=run_a.label,
            name_b=run_b.label,
        )
        for key in common
    ]

    excluded: list[dict[str, Any]] = []
    for key in sorted(set(run_a.excluded_scalars) | set(run_b.excluded_scalars)):
        excluded.append(
            {
                "path": key,
                "a": run_a.excluded_scalars.get(key),
                "b": run_b.excluded_scalars.get(key),
            }
        )

    return RunComparison(
        label_a=run_a.label,
        label_b=run_b.label,
        schema_a=run_a.schema,
        schema_b=run_b.schema,
        higher_is_better=higher_is_better,
        alpha=alpha,
        min_paired=min_paired,
        n_boot=n_boot,
        seed=seed,
        config_diff=_diff_config(run_a, run_b),
        metric_deltas=_diff_scalars(run_a, run_b),
        excluded_diagnostics=excluded,
        series=comparisons,
        series_only_in_a=sorted(keys_a - keys_b),
        series_only_in_b=sorted(keys_b - keys_a),
    )


def receipt_payload(comparison: RunComparison) -> dict[str, Any]:
    """Wrap a comparison in the repo's receipt conventions (research-only)."""
    return {
        "schema": COMPARE_RECEIPT_SCHEMA,
        "generated_at": datetime.now(UTC).isoformat(),
        "research_only": True,
        "live_pnl_claim": False,
        "disclaimer": (
            "Paired run comparison on stored research receipts. Diagnostic "
            "only — proper scores, no live-trading or P&L claims."
        ),
        "report": comparison.to_dict(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("run_a", type=Path, help="receipt JSON or result dir for run A")
    parser.add_argument("run_b", type=Path, help="receipt JSON or result dir for run B")
    parser.add_argument(
        "--higher-is-better",
        action="store_true",
        help="score direction: positive delta favors A (default: loss convention)",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--min-paired",
        type=int,
        default=10,
        help="below this paired n the verdict is 'insufficient paired observations'",
    )
    parser.add_argument(
        "--series",
        dest="series_keys",
        nargs="*",
        default=None,
        help="restrict to these series keys",
    )
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--out", type=Path, default=None, help="write the report here")
    parser.add_argument(
        "--receipt-out",
        type=Path,
        default=None,
        help="also write a run_compare.v1 receipt blob here",
    )
    args = parser.parse_args(argv)
    try:
        comparison = compare_runs(
            args.run_a,
            args.run_b,
            higher_is_better=args.higher_is_better,
            alpha=args.alpha,
            n_boot=args.n_boot,
            seed=args.seed,
            min_paired=args.min_paired,
            series_keys=args.series_keys,
        )
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
        return
    text = (
        json.dumps(comparison.to_dict(), indent=2, allow_nan=False)
        if args.format == "json"
        else comparison.to_markdown()
    )
    if args.out is not None:
        args.out.write_text(text + "\n")
    else:
        print(text)
    if args.receipt_out is not None:
        args.receipt_out.write_text(
            json.dumps(receipt_payload(comparison), indent=2, allow_nan=False) + "\n"
        )


if __name__ == "__main__":
    main()
