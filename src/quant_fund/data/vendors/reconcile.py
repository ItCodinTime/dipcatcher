"""Cross-vendor OHLCV reconciliation report.

Compares normalized bar frames from ≥2 vendors on the shared
``(security_id, event_time)`` keys and flags every field whose absolute and
relative difference both exceed tolerance. Emits a JSON-serializable report
plus a console rendering; the CLI parses *recorded vendor response bodies*
through each adapter's own ``parse_body`` — the same code path a live fetch
uses — so reconciliation exercises real normalization, never a mock.

Usage (offline, fixture files)::

    python -m quant_fund.data.vendors.reconcile \
        --vendor tiingo=tests/fixtures/vendors/tiingo/spy_daily.json \
        --vendor polygon=tests/fixtures/vendors/polygon/spy_aggs.json \
        --symbol SPY --json report.json

Exit code: ``0`` clean, ``1`` divergent, ``2`` usage error.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from quant_fund.data.vendors import get_vendor_adapter, vendor_adapter_names

BAR_FIELDS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
JOIN_KEYS: tuple[str, ...] = ("security_id", "event_time")
_EPS = 1e-30


@dataclass(frozen=True)
class ReconcileTolerances:
    """Per-field tolerance policy.

    A difference is flagged only when ``abs_diff > abs_tol`` *and*
    ``rel_diff > rel_tol`` — the absolute floor keeps the relative test from
    exploding on near-zero values. ``volume_rel`` is looser because vendor
    volume prints legitimately diverge (consolidated vs primary tape).
    """

    rel: float = 1e-4
    abs: float = 1e-9
    volume_rel: float = 5e-2

    def __post_init__(self) -> None:
        if self.rel < 0 or self.abs < 0 or self.volume_rel < 0:
            raise ValueError("tolerances must be non-negative")

    def for_field(self, name: str) -> float:
        return self.volume_rel if name == "volume" else self.rel


@dataclass
class PairReport:
    left: str
    right: str
    shared_bars: int = 0
    matched_bars: int = 0
    divergent_bars: int = 0
    left_only: list[str] = field(default_factory=list)
    right_only: list[str] = field(default_factory=list)
    mismatches: list[dict[str, Any]] = field(default_factory=list)


def _key_label(security_id: Any, event_time: Any) -> str:
    ts = event_time.isoformat() if hasattr(event_time, "isoformat") else str(event_time)
    return f"{security_id}@{ts}"


def _diff_field(left: float, right: float) -> tuple[float, float]:
    abs_diff = abs(left - right)
    rel_diff = abs_diff / max(abs(left), abs(right), _EPS)
    return abs_diff, rel_diff


def reconcile_pair(
    left_name: str,
    left: pl.DataFrame,
    right_name: str,
    right: pl.DataFrame,
    *,
    tolerances: ReconcileTolerances | None = None,
    fields: Sequence[str] = BAR_FIELDS,
) -> PairReport:
    """Diff two normalized vendor frames on shared (security_id, event_time)."""
    tol = tolerances or ReconcileTolerances()
    report = PairReport(left=left_name, right=right_name)
    for name, frame in ((left_name, left), (right_name, right)):
        missing = [key for key in JOIN_KEYS if key not in frame.columns]
        if missing:
            raise ValueError(f"{name} frame missing join keys {missing}")
        for fname in fields:
            if fname not in frame.columns:
                raise ValueError(f"{name} frame missing field {fname!r}")
    joined = left.join(right, on=list(JOIN_KEYS), how="inner", suffix="__right")
    report.shared_bars = joined.height
    divergent_keys: set[tuple[Any, Any]] = set()
    for row in joined.iter_rows(named=True):
        for fname in fields:
            l_val, r_val = row[fname], row[f"{fname}__right"]
            if l_val is None or r_val is None:
                continue
            if not (math.isfinite(l_val) and math.isfinite(r_val)):
                continue
            abs_diff, rel_diff = _diff_field(float(l_val), float(r_val))
            if abs_diff > tol.abs and rel_diff > tol.for_field(fname):
                key = (row["security_id"], row["event_time"])
                divergent_keys.add(key)
                ts = row["event_time"]
                report.mismatches.append(
                    {
                        "security_id": str(row["security_id"]),
                        "event_time": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                        "field": fname,
                        "left": float(l_val),
                        "right": float(r_val),
                        "abs_diff": abs_diff,
                        "rel_diff": rel_diff,
                    }
                )
    report.divergent_bars = len(divergent_keys)
    report.matched_bars = report.shared_bars - report.divergent_bars
    report.mismatches.sort(key=lambda m: (m["event_time"], m["security_id"], m["field"]))
    left_only = left.join(right, on=list(JOIN_KEYS), how="anti")
    right_only = right.join(left, on=list(JOIN_KEYS), how="anti")
    report.left_only = sorted(
        _key_label(r["security_id"], r["event_time"])
        for r in left_only.iter_rows(named=True)
    )
    report.right_only = sorted(
        _key_label(r["security_id"], r["event_time"])
        for r in right_only.iter_rows(named=True)
    )
    return report


def reconcile_frames(
    frames: Mapping[str, pl.DataFrame],
    *,
    tolerances: ReconcileTolerances | None = None,
    fields: Sequence[str] = BAR_FIELDS,
    now: datetime | None = None,
) -> dict[str, Any]:
    """All unordered vendor pairs → one JSON-serializable report."""
    if len(frames) < 2:
        raise ValueError("reconcile_frames needs at least two vendor frames")
    tol = tolerances or ReconcileTolerances()
    names = sorted(frames)
    pairs = [
        reconcile_pair(a, frames[a], b, frames[b], tolerances=tol, fields=fields)
        for a, b in itertools.combinations(names, 2)
    ]
    mismatch_count = sum(len(p.mismatches) for p in pairs)
    bars_only = sum(len(p.left_only) + len(p.right_only) for p in pairs)
    generated = (now or datetime.now(tz=UTC)).isoformat()
    return {
        "tool": "quant_fund.data.vendors.reconcile",
        "report_version": 1,
        "generated_at": generated,
        "vendors": names,
        "fields": list(fields),
        "tolerances": {"rel": tol.rel, "abs": tol.abs, "volume_rel": tol.volume_rel},
        "pairs": [asdict(p) for p in pairs],
        "status": "clean" if mismatch_count == 0 and bars_only == 0 else "divergent",
        "mismatch_count": mismatch_count,
        "unshared_bar_count": bars_only,
    }


def render_console(report: Mapping[str, Any]) -> str:
    """Human-readable rendering of a report dict."""
    lines = [
        f"vendor reconciliation — {', '.join(report['vendors'])}",
        f"tolerances: rel={report['tolerances']['rel']:g} "
        f"abs={report['tolerances']['abs']:g} "
        f"volume_rel={report['tolerances']['volume_rel']:g}",
    ]
    for pair in report["pairs"]:
        lines.append(
            f"  {pair['left']} vs {pair['right']}: "
            f"{pair['shared_bars']} shared bars, "
            f"{pair['matched_bars']} matched, "
            f"{pair['divergent_bars']} divergent"
        )
        for label, rows in (("only in " + pair["left"], pair["left_only"]),
                            ("only in " + pair["right"], pair["right_only"])):
            for item in rows:
                lines.append(f"    {label}: {item}")
        for m in pair["mismatches"]:
            lines.append(
                f"    {m['security_id']} {m['event_time']} {m['field']}: "
                f"{m['left']:g} vs {m['right']:g} "
                f"(abs {m['abs_diff']:.6g}, rel {m['rel_diff']:.6%})"
            )
    lines.append(
        f"status: {report['status']} "
        f"(mismatches={report['mismatch_count']}, "
        f"unshared_bars={report['unshared_bar_count']})"
    )
    return "\n".join(lines)


def _parse_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="quant_fund.data.vendors.reconcile",
        description="Cross-vendor OHLCV reconciliation on recorded response bodies.",
    )
    parser.add_argument(
        "--vendor",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help=f"vendor adapter name and response-body file; repeatable (>=2). "
        f"Names: {', '.join(vendor_adapter_names())}",
    )
    parser.add_argument("--symbol", required=True, help="security symbol stamped on bars")
    parser.add_argument("--start", default=None, help="ISO datetime lower bound (inclusive)")
    parser.add_argument("--end", default=None, help="ISO datetime upper bound (inclusive)")
    parser.add_argument("--rel-tol", type=float, default=ReconcileTolerances.rel)
    parser.add_argument("--abs-tol", type=float, default=ReconcileTolerances.abs)
    parser.add_argument("--volume-rel-tol", type=float, default=ReconcileTolerances.volume_rel)
    parser.add_argument(
        "--json",
        default=None,
        metavar="PATH",
        help="write the JSON report to PATH ('-' for stdout)",
    )
    args = parser.parse_args(argv)

    start = _parse_dt(args.start)
    end = _parse_dt(args.end)
    frames: dict[str, pl.DataFrame] = {}
    for spec in args.vendor:
        name, sep, path_s = spec.partition("=")
        if not sep or not name.strip() or not path_s.strip():
            parser.error(f"--vendor must be NAME=PATH, got {spec!r}")
        name = name.strip().lower()
        try:
            adapter = get_vendor_adapter(name)
        except ValueError as exc:
            parser.error(str(exc))
        path = Path(path_s.strip())
        if not path.is_file():
            parser.error(f"fixture file not found: {path}")
        try:
            frames[name] = adapter.parse_body(
                path.read_bytes(), symbol=args.symbol, start=start, end=end
            )
        except Exception as exc:  # noqa: BLE001 — CLI surfaces any parse failure
            parser.error(f"{name}: failed to parse {path}: {exc}")
    if len(frames) < 2:
        parser.error("reconciliation needs at least two --vendor NAME=PATH pairs")

    report = reconcile_frames(
        frames,
        tolerances=ReconcileTolerances(
            rel=args.rel_tol, abs=args.abs_tol, volume_rel=args.volume_rel_tol
        ),
    )
    print(render_console(report))
    if args.json is not None:
        payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.json == "-":
            sys.stdout.write(payload)
        else:
            Path(args.json).write_text(payload, encoding="utf-8")
    return 0 if report["status"] == "clean" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
