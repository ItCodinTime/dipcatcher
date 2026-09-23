"""Fast arena leaderboard from a merged SOTA loss shard or merge receipt.

Parallel arena lanes each add a challenger column; this renders a uniform
"where does my model rank" markdown table without re-running the DM/SPA/MCS
battery in ``scripts/sota_eval_kronos.py``. Pooled per-model finite means are
descriptive only — paired inference lives in the full merge receipt.

research-only; no live-PnL claim

Usage:
    python scripts/arena_table.py --losses merge_d1_v2aug.losses.npz
    python scripts/arena_table.py --losses shard.losses.npz --baseline dip_garch_t --out report.md
    python scripts/arena_table.py --losses merge_d1_v2aug.json   # receipt-only
    python scripts/arena_table.py --compare a.losses.npz b.losses.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

WATERMARK = "research-only; no live-PnL claim"
DEFAULT_BASELINE = "dip_garch_t"
DEFAULT_TAUS = (0.05, 0.5, 0.95)
# Name hints for legacy shards whose meta_json is absent (targets were just the
# leading columns). Meta ``targets`` is authoritative whenever present.
PUBLISHED_NAME_HINTS = frozenset(
    {
        "timesfm",
        "timesfm2",
        "chronos",
        "chronos2",
        "bolt",
        "bolt_small",
        "bolt_base",
        "kronos",
        "kronos_small",
        "kronos_base",
        "kronos_mini",
    }
)


def _finite_mean(values: np.ndarray) -> float | None:
    """Mean over finite entries only; None when empty (merge-receipt convention)."""
    arr = np.asarray(values, dtype=float)
    valid = arr[np.isfinite(arr)]
    return float(np.mean(valid)) if valid.size else None


def _finite_count(values: np.ndarray) -> int:
    return int(np.isfinite(np.asarray(values, dtype=float)).sum())


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


@dataclass
class Losses:
    """Row-level per-origin loss cube from a single-asset shard or merged npz."""

    path: Path
    model_names: list[str]
    crps: np.ndarray  # [R, M]
    pinball: np.ndarray | None  # [R, M, T]
    asset_ids: np.ndarray | None
    asset_names: list[str]
    targets: list[str]
    tau_labels: list[str]
    meta: dict[str, Any]

    @property
    def n_rows(self) -> int:
        return int(self.crps.shape[0])


def _tau_labels(meta: dict[str, Any], n_tau: int | None) -> list[str]:
    cfg_taus = meta.get("config", {}).get("taus") if isinstance(meta.get("config"), dict) else None
    if n_tau is None:
        return [f"{float(t):g}" for t in (cfg_taus or DEFAULT_TAUS)]
    if cfg_taus and len(cfg_taus) == n_tau:
        return [f"{float(t):g}" for t in cfg_taus]
    if n_tau == len(DEFAULT_TAUS):
        return [f"{t:g}" for t in DEFAULT_TAUS]
    return [f"tau{k}" for k in range(n_tau)]


def load_losses(path: Path | str) -> Losses:
    """Load a shard/merged losses npz (``crps_matrix`` or legacy ``loss_matrix``)."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if "crps_matrix" in data:
        crps = np.asarray(data["crps_matrix"], dtype=float)
    elif "loss_matrix" in data:
        # Legacy v1 shards carried only the CRPS matrix (no pinball/meta).
        crps = np.asarray(data["loss_matrix"], dtype=float)
    else:
        raise ValueError(f"{path}: no crps_matrix array (keys: {sorted(data)})")
    if crps.ndim != 2:
        raise ValueError(f"{path}: crps_matrix must be [rows, models], got {crps.shape}")
    if "model_names" not in data:
        raise ValueError(f"{path}: missing model_names")
    model_names = [str(x) for x in data["model_names"]]
    if len(model_names) != crps.shape[1]:
        raise ValueError(
            f"{path}: model_names ({len(model_names)}) != crps columns ({crps.shape[1]})"
        )
    pinball = None
    if "pinball_cube" in data:
        pinball = np.asarray(data["pinball_cube"], dtype=float)
        if pinball.ndim != 3 or pinball.shape[:2] != crps.shape:
            raise ValueError(f"{path}: pinball_cube {pinball.shape} mismatches crps {crps.shape}")
    meta: dict[str, Any] = {}
    if "meta_json" in data:
        meta = json.loads(str(data["meta_json"]))
    asset_ids = np.asarray(data["asset_ids"]) if "asset_ids" in data else None
    asset_names = [str(x) for x in meta.get("asset_names", [])]
    targets = [str(t) for t in (meta.get("targets") or meta.get("config", {}).get("targets") or [])]
    return Losses(
        path=path,
        model_names=model_names,
        crps=crps,
        pinball=pinball,
        asset_ids=asset_ids,
        asset_names=asset_names,
        targets=targets,
        tau_labels=_tau_labels(meta, pinball.shape[2] if pinball is not None else None),
        meta=meta,
    )


def load_receipt(path: Path | str) -> dict[str, Any]:
    """Load a merge receipt json (``sota_eval.v4`` schema)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def companion_receipt_path(losses_path: Path | str) -> Path:
    """Sibling receipt for a losses file: ``x.losses.npz`` -> ``x.json``."""
    losses_path = Path(losses_path)
    name = losses_path.name
    for suffix in (".losses.npz", ".partial.npz", ".npz"):
        if name.endswith(suffix):
            return losses_path.with_name(name[: -len(suffix)] + ".json")
    return losses_path.with_suffix(".json")


def companion_losses_path(receipt_path: Path | str, receipt: dict[str, Any]) -> Path | None:
    """Sibling losses npz named by the receipt's ``losses_file`` key, if present."""
    receipt_path = Path(receipt_path)
    recorded = receipt.get("losses_file")
    if recorded and Path(str(recorded)).name == str(recorded):
        candidate = receipt_path.parent / str(recorded)
        if candidate.is_file():
            return candidate
    candidate = receipt_path.with_name(receipt_path.stem + ".losses.npz")
    return candidate if candidate.is_file() else None


# --------------------------------------------------------------------------
# Leaderboard rows
# --------------------------------------------------------------------------


def model_type(name: str, targets: Any = ()) -> str:
    """`published` for the foundation targets, `dip` for lab challengers."""
    if name in targets or name in PUBLISHED_NAME_HINTS:
        return "published"
    if name.startswith("dip_"):
        return "dip"
    return "other"


@dataclass
class ModelRow:
    model: str
    type: str
    mean_crps: float | None
    n_finite: int | None
    pinball: list[float | None]
    mcs: bool | None = None
    mcs_p: float | None = None
    delta: float | None = None
    delta_pct: float | None = None
    rank: int | None = None


def _mcs_fields(receipt: dict[str, Any] | None, model: str) -> tuple[bool | None, float | None]:
    if not receipt:
        return None, None
    included = receipt.get("mcs_included") or {}
    p_values = receipt.get("mcs_p_values") or {}
    member = bool(included[model]) if model in included else None
    p_raw = p_values.get(model)
    return member, (float(p_raw) if p_raw is not None else None)


def rows_from_losses(losses: Losses, receipt: dict[str, Any] | None = None) -> list[ModelRow]:
    """Per-model pooled stats from row-level losses (authoritative n_finite)."""
    rows = []
    for i, m in enumerate(losses.model_names):
        pin = (
            [_finite_mean(losses.pinball[:, i, k]) for k in range(losses.pinball.shape[2])]
            if losses.pinball is not None
            else []
        )
        member, p_value = _mcs_fields(receipt, m)
        rows.append(
            ModelRow(
                model=m,
                type=model_type(m, losses.targets),
                mean_crps=_finite_mean(losses.crps[:, i]),
                n_finite=_finite_count(losses.crps[:, i]),
                pinball=pin,
                mcs=member,
                mcs_p=p_value,
            )
        )
    return rows


def rows_from_receipt(receipt: dict[str, Any]) -> tuple[list[ModelRow], list[str]]:
    """Receipt-only fallback: pooled means + coverage counts, no row-level data."""
    means = receipt.get("mean_crps_pooled") or {}
    pins = receipt.get("mean_pinball_pooled") or {}
    coverage = receipt.get("coverage") or {}
    cfg = receipt.get("config") or {}
    targets = [str(t) for t in (cfg.get("targets") or [])]
    challengers = [str(c) for c in (receipt.get("challengers") or [])]
    ordered = list(dict.fromkeys([*targets, *challengers, *means]))
    taus = [float(t) for t in (cfg.get("taus") or DEFAULT_TAUS)]
    rows = []
    for m in ordered:
        if m not in means:
            continue
        pin_map = pins.get(m) or {}
        pin = [pin_map.get(str(t)) for t in taus]
        finite = (coverage.get(m) or {}).get("finite_crps")
        member, p_value = _mcs_fields(receipt, m)
        rows.append(
            ModelRow(
                model=m,
                type=model_type(m, targets),
                mean_crps=float(means[m]) if means.get(m) is not None else None,
                n_finite=int(finite) if finite is not None else None,
                pinball=[float(v) if v is not None else None for v in pin],
                mcs=member,
                mcs_p=p_value,
            )
        )
    return rows, [f"{t:g}" for t in taus]


def rank_rows(rows: list[ModelRow], baseline: str = DEFAULT_BASELINE) -> float | None:
    """Rank by mean_crps ascending (lower is better); fill delta vs baseline."""
    scored = sorted(
        (r for r in rows if r.mean_crps is not None),
        key=lambda r: (r.mean_crps, r.model),
    )
    for i, r in enumerate(scored):
        r.rank = i + 1
    base = next((r for r in rows if r.model == baseline), None)
    base_mean = base.mean_crps if base is not None else None
    if base_mean is not None:
        for r in rows:
            if r.mean_crps is not None:
                r.delta = r.mean_crps - base_mean
                r.delta_pct = r.delta / base_mean * 100.0 if base_mean != 0.0 else None
    return base_mean


def ordered_rows(rows: list[ModelRow]) -> list[ModelRow]:
    """Ranked models first (rank order); unranked models follow in file order."""
    ranked = sorted((r for r in rows if r.rank is not None), key=lambda r: r.rank or 0)
    return ranked + [r for r in rows if r.rank is None]


# --------------------------------------------------------------------------
# Markdown rendering
# --------------------------------------------------------------------------


def _num(v: float | None, nd: int = 6) -> str:
    if v is None:
        return "—"
    if v != 0.0 and abs(v) < 10 ** (-nd):
        return f"{v:.3e}"
    return f"{v:.{nd}f}"


def _int(v: int | None) -> str:
    return str(v) if v is not None else "—"


def render_leaderboard(
    rows: list[ModelRow],
    *,
    baseline: str = DEFAULT_BASELINE,
    tau_labels: list[str] | None = None,
    show_mcs: bool = False,
) -> str:
    """Markdown leaderboard table: rank, mean CRPS, pinball per tau, Δ baseline, MCS."""
    tau_labels = tau_labels or []
    has_pin = any(r.pinball for r in rows)
    headers = ["rank", "model", "type", "mean_crps", "n_finite"]
    if has_pin:
        headers += [f"pin@{t}" for t in tau_labels]
    headers += [f"dcrps_vs_{baseline}", "d%"]
    if show_mcs:
        headers.append("mcs@0.10")
    align = ["---:"] + ["---", "---"] + ["---:"] * (len(headers) - 3)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(align) + "|"]
    for r in ordered_rows(rows):
        cells = [
            str(r.rank) if r.rank is not None else "—",
            f"`{r.model}`",
            r.type,
            _num(r.mean_crps),
            _int(r.n_finite),
        ]
        if has_pin:
            pins = [_num(v) for v in r.pinball]
            cells += pins + ["—"] * (len(tau_labels) - len(pins))
        cells.append("—" if r.delta is None else f"{r.delta:+.6f}")
        cells.append("—" if r.delta_pct is None else f"{r.delta_pct:+.1f}%")
        if show_mcs:
            if r.mcs is None:
                cells.append("n/a")
            else:
                tag = "in" if r.mcs else "out"
                cells.append(f"{tag} (p={_num(r.mcs_p, 3)})" if r.mcs_p is not None else tag)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_report(
    losses_path: Path | str,
    *,
    receipt_path: Path | str | None = None,
    baseline: str = DEFAULT_BASELINE,
) -> str:
    """Full markdown report for one losses npz or merge receipt json."""
    losses_path = Path(losses_path)
    receipt: dict[str, Any] | None = None
    receipt_name: str | None = None
    losses: Losses | None = None
    if losses_path.suffix == ".json":
        receipt = load_receipt(losses_path)
        receipt_name = losses_path.name
        companion = companion_losses_path(losses_path, receipt)
        if companion is not None:
            losses = load_losses(companion)
    else:
        losses = load_losses(losses_path)
        candidate = Path(receipt_path) if receipt_path else companion_receipt_path(losses_path)
        if candidate.is_file():
            receipt = load_receipt(candidate)
            receipt_name = candidate.name
    if losses is not None:
        rows = rows_from_losses(losses, receipt)
        tau_labels = losses.tau_labels
        n_rows: int | None = losses.n_rows
        asset_names = losses.asset_names
        targets = losses.targets
        partial = bool(losses.meta.get("partial"))
        source = losses.path.name
    else:
        assert receipt is not None  # losses_path was .json
        rows, tau_labels = rows_from_receipt(receipt)
        n_rows = receipt.get("n_rows")
        asset_names = sorted((receipt.get("mean_crps_per_asset") or {}).keys())
        targets = [str(t) for t in (receipt.get("config") or {}).get("targets") or []]
        partial = False
        source = losses_path.name
    base_mean = rank_rows(rows, baseline)
    header = [
        WATERMARK,
        "",
        f"## Arena leaderboard — `{source}`",
        "",
        f"- rows: {_int(n_rows)} | models: {len(rows)}"
        + (f" | assets: {', '.join(asset_names)}" if asset_names else ""),
        f"- baseline: `{baseline}`"
        + (
            f" (mean_crps {_num(base_mean)}); delta = model - baseline, negative is better"
            if base_mean is not None
            else " — NOT PRESENT in model columns; delta columns are n/a"
        ),
        "- mean_crps / pinball means: pooled over rows finite for that model"
        " (n_finite disclosed); paired tests use complete rows only — see the full receipt",
    ]
    if receipt is not None:
        header.append(
            f"- mcs membership/p-values from receipt `{receipt_name}` (last full battery)"
        )
    else:
        header.append("- no companion receipt found; MCS column omitted")
    if partial:
        header.append("- WARNING: partial checkpoint shard — coverage incomplete")
    if losses is None:
        header.append("- receipt-only view: n_finite from the receipt coverage block")
    elif any(r.n_finite is not None and r.n_finite < losses.n_rows for r in rows):
        header.append("- NaN losses present — n_finite shows honest per-model coverage")
    if targets:
        header.append(f"- published targets: {', '.join(targets)}")
    table = render_leaderboard(
        rows,
        baseline=baseline,
        tau_labels=tau_labels,
        show_mcs=receipt is not None,
    )
    return "\n".join(header) + "\n\n" + table


# --------------------------------------------------------------------------
# Compare mode
# --------------------------------------------------------------------------


@dataclass
class CompareRow:
    model: str
    type: str
    sets: str  # "both" | "a only" | "b only"
    n_pair: int | None
    mean_a: float | None  # paired mean for shared models, own-file mean otherwise
    mean_b: float | None
    delta: float | None  # mean_B - mean_A on rows finite in both
    delta_pct: float | None


def compare_losses(a: Losses, b: Losses) -> list[CompareRow]:
    """Align two loss cubes by model name; paired delta on common finite rows."""
    idx_a = {m: i for i, m in enumerate(a.model_names)}
    idx_b = {m: i for i, m in enumerate(b.model_names)}
    union = list(dict.fromkeys([*a.model_names, *b.model_names]))
    targets = set(a.targets) | set(b.targets)
    paired = a.n_rows == b.n_rows
    rows: list[CompareRow] = []
    for m in union:
        ia, ib = idx_a.get(m), idx_b.get(m)
        if ia is not None and ib is not None:
            if paired:
                mask = np.isfinite(a.crps[:, ia]) & np.isfinite(b.crps[:, ib])
                n_pair: int | None = int(mask.sum())
                mean_a = _finite_mean(a.crps[mask, ia])
                mean_b = _finite_mean(b.crps[mask, ib])
                sets = "both"
            else:
                # Row grids differ (different panels): pairing is meaningless,
                # so fall back to own-file pooled means, disclosed as unpaired.
                n_pair = None
                mean_a = _finite_mean(a.crps[:, ia])
                mean_b = _finite_mean(b.crps[:, ib])
                sets = "both*"
            delta = mean_b - mean_a if (mean_a is not None and mean_b is not None) else None
        elif ia is not None:
            n_pair, mean_a, mean_b, delta, sets = (
                None,
                _finite_mean(a.crps[:, ia]),
                None,
                None,
                "a only",
            )
        else:
            n_pair, mean_a, mean_b, delta, sets = (
                None,
                None,
                _finite_mean(b.crps[:, ib]),
                None,
                "b only",
            )
        pct = delta / mean_a * 100.0 if (delta is not None and mean_a) else None
        rows.append(CompareRow(m, model_type(m, targets), sets, n_pair, mean_a, mean_b, delta, pct))

    def _key(r: CompareRow) -> tuple[int, bool, float, str]:
        group = {"both": 0, "both*": 0, "a only": 1, "b only": 2}[r.sets]
        return (group, r.delta is None, r.delta if r.delta is not None else 0.0, r.model)

    return sorted(rows, key=_key)


def render_compare(rows: list[CompareRow]) -> str:
    headers = ["model", "type", "sets", "n_pair", "crps_A", "crps_B", "d_B-A", "d%"]
    align = ["---", "---", "---"] + ["---:"] * 5
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(align) + "|"]
    for r in rows:
        cells = [
            f"`{r.model}`",
            r.type,
            r.sets,
            _int(r.n_pair),
            _num(r.mean_a),
            _num(r.mean_b),
            "—" if r.delta is None else f"{r.delta:+.6f}",
            "—" if r.delta_pct is None else f"{r.delta_pct:+.1f}%",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def compare_report(a: Losses, b: Losses) -> str:
    """Markdown report for ``--compare a.npz b.npz``."""
    rows = compare_losses(a, b)
    n_both = sum(1 for r in rows if r.sets.startswith("both"))
    header = [
        WATERMARK,
        "",
        f"## Arena compare — `{a.path.name}` (A) vs `{b.path.name}` (B)",
        "",
        f"- rows: A={a.n_rows}, B={b.n_rows} | models: A={len(a.model_names)}, "
        f"B={len(b.model_names)}, shared={n_both}",
        "- d_B-A = mean_B - mean_A over rows finite in BOTH (n_pair disclosed; "
        "negative = B better); single-sided models show own-file finite mean",
    ]
    if a.n_rows != b.n_rows:
        header.append(
            "- WARNING: row counts differ — shared models marked `both*` show "
            "unpaired pooled-mean deltas, NOT common-row deltas"
        )
    header.append("")
    return "\n".join(header) + "\n" + render_compare(rows)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Arena leaderboard table — research-only; no live-PnL claim.",
    )
    parser.add_argument(
        "--losses",
        type=Path,
        help="merged/shard losses .npz or merge receipt .json",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=None,
        help="merge receipt json for the MCS column (auto-discovered when omitted)",
    )
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--out", type=Path, help="write markdown here instead of stdout")
    parser.add_argument(
        "--compare",
        type=Path,
        nargs=2,
        metavar=("A_NPZ", "B_NPZ"),
        help="aligned two-file comparison on common finite rows",
    )
    args = parser.parse_args(argv)

    if args.compare:
        a_path, b_path = args.compare
        if a_path.suffix == ".json" or b_path.suffix == ".json":
            parser.error("--compare needs row-level .npz loss files, not receipts")
        report = compare_report(load_losses(a_path), load_losses(b_path))
    elif args.losses:
        report = build_report(args.losses, receipt_path=args.receipt, baseline=args.baseline)
    else:
        parser.error("pass --losses or --compare")
        return 2

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
