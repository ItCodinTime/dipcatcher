"""Phase B4 — effect sizes with CIs for the SOTA evaluation.

Reads the stored per-asset ``*.losses.npz`` matrices (no model needed) and
reports per-origin CRPS deltas (target − challenger; positive = challenger
better) with stationary-bootstrap confidence intervals — Politis–Romano
resampling with the Politis–White automatic block length, the same scheme the
MCS/SPA battery uses.

An adjudicator wants magnitudes, not only p-values: this emits, per asset and
pooled, every target×challenger pair's mean delta, relative improvement vs the
target's mean CRPS, and a 95% percentile CI. Research-only; no live-P&L claim.

Usage::

    python scripts/sota_effect_sizes.py --eval-dir .dsh-24x7/eval-full \
        --horizon d1 --out .dsh-24x7/evidence-sota-effect-sizes-d1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from quant_fund.metrics.inference import (  # noqa: E402
    optimal_block_length,
    stationary_bootstrap_indices,
)

TARGETS = ("kronos_small", "chronos2", "bolt_small", "timesfm")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _load_assets(eval_dir: Path, horizon: str) -> dict[str, dict[str, Any]]:
    """Return {asset: {matrix, models, sha}} for one horizon prefix."""
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(eval_dir.glob(f"{horizon}_*.losses.npz")):
        if "merged" in path.name:
            continue  # pooled matrix, not a per-asset shard
        d = np.load(path, allow_pickle=False)
        models = [str(m) for m in d["model_names"]]
        meta = json.loads(str(d["meta_json"]))
        names = meta.get("asset_names") or []
        if len(names) != 1:
            continue  # not a single-asset shard
        asset = str(names[0])
        out[asset] = {
            "matrix": np.asarray(d["crps_matrix"], dtype=float),
            "models": models,
            "sha": _sha256(path),
            "n_origins": int(d["crps_matrix"].shape[0]),
        }
    return out


def _pair_deltas(matrix: np.ndarray, models: list[str]) -> tuple[list[str], np.ndarray]:
    """Per-origin deltas target−challenger for every target×challenger pair."""
    t_idx = [models.index(t) for t in TARGETS if t in models]
    c_idx = [i for i, m in enumerate(models) if m not in TARGETS]
    labels = [f"{models[t]}_minus_{models[c]}" for t in t_idx for c in c_idx]
    deltas = np.stack([matrix[:, t] - matrix[:, c] for t in t_idx for c in c_idx], axis=1)
    return labels, deltas


def _effect_rows(
    label_prefix: str,
    labels: list[str],
    deltas: np.ndarray,
    target_means: dict[str, float],
    *,
    n_boot: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Stationary-bootstrap CI rows for a (n_origins × n_pairs) delta frame."""
    rng = np.random.default_rng(seed)
    rows = []
    for j, lab in enumerate(labels):
        col = deltas[:, j]
        col = col[np.isfinite(col)]  # per-pair complete-case
        n_complete = col.shape[0]
        if n_complete < 10:
            rows.append(
                {
                    "cell": label_prefix,
                    "pair": lab,
                    "target": lab.partition("_minus_")[0],
                    "challenger": lab.partition("_minus_")[2],
                    "n_origins": n_complete,
                    "status": "insufficient_data",
                }
            )
            continue
        block = max(1.0, float(optimal_block_length(col)))
        idx = stationary_bootstrap_indices(n_complete, n_boot, block, rng)
        boot_means = col[idx].mean(axis=1)
        lo, hi = np.percentile(boot_means, [2.5, 97.5])
        mean = float(col.mean())
        target, _, chal = lab.partition("_minus_")
        t_mean = target_means.get(target)
        rows.append(
            {
                "cell": label_prefix,
                "pair": lab,
                "target": target,
                "challenger": chal,
                "n_origins": int(n_complete),
                "mean_delta": mean,
                "ci95_lo": float(lo),
                "ci95_hi": float(hi),
                "pct_vs_target_mean": (
                    float(mean / t_mean) if t_mean and np.isfinite(t_mean) else None
                ),
                "significant_95": bool(lo > 0),
                "block_len": float(block),
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, default=Path(".dsh-24x7/eval-full"))
    ap.add_argument("--horizon", choices=["d1", "h4"], required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    assets = _load_assets(args.eval_dir, args.horizon)
    if not assets:
        print(f"no {args.horizon}_*.losses.npz under {args.eval_dir}", file=sys.stderr)
        return 1

    rows: list[dict[str, Any]] = []
    pooled_deltas: list[np.ndarray] = []
    pooled_labels: list[str] | None = None
    pooled_target_vals: dict[str, list[np.ndarray]] = {t: [] for t in TARGETS}
    asset_summaries: dict[str, Any] = {}

    for asset, pack in sorted(assets.items()):
        matrix, models = pack["matrix"], pack["models"]
        labels, deltas = _pair_deltas(matrix, models)
        t_means = {t: float(np.nanmean(matrix[:, models.index(t)])) for t in TARGETS if t in models}
        rows.extend(
            _effect_rows(asset, labels, deltas, t_means, n_boot=args.n_boot, seed=args.seed)
        )
        best_chal = min(
            (m for m in models if m not in TARGETS),
            key=lambda m: float(np.nanmean(matrix[:, models.index(m)])),
        )
        best_tgt = min(TARGETS, key=lambda t: t_means.get(t, np.inf))
        asset_summaries[asset] = {
            "n_origins": pack["n_origins"],
            "best_challenger": best_chal,
            "best_target": best_tgt,
            "challenger_mean_crps": float(np.nanmean(matrix[:, models.index(best_chal)])),
            "target_mean_crps": float(t_means[best_tgt]),
        }
        if pooled_labels is None:
            pooled_labels = labels
        elif labels != pooled_labels:
            print(f"model set differs for {asset}; skipping pooled", file=sys.stderr)
            continue
        pooled_deltas.append(deltas)
        for t in TARGETS:
            if t in models:
                pooled_target_vals[t].append(matrix[:, models.index(t)])

    if pooled_deltas:
        pooled = np.concatenate(pooled_deltas, axis=0)
        p_t_means = {
            t: float(np.nanmean(np.concatenate(v))) for t, v in pooled_target_vals.items() if v
        }
        rows.extend(
            _effect_rows(
                "POOLED",
                pooled_labels or [],
                pooled,
                p_t_means,
                n_boot=args.n_boot,
                seed=args.seed,
            )
        )

    pooled_rows = [r for r in rows if r["cell"] == "POOLED"]
    scored = [r for r in pooled_rows if "significant_95" in r]
    n_sig = sum(1 for r in scored if r["significant_95"])
    min_ci = min((r["ci95_lo"] for r in scored), default=None)

    receipt = {
        "schema": "sota_effect_sizes.v1",
        "horizon": args.horizon,
        "eval_dir": str(args.eval_dir),
        "n_boot": args.n_boot,
        "seed": args.seed,
        "input_sha256": {a: p["sha"] for a, p in assets.items()},
        "asset_summaries": asset_summaries,
        "pairs": rows,
        "pooled_summary": {
            "n_pairs": len(pooled_rows),
            "n_significant_95": n_sig,
            "min_ci95_lo": min_ci,
            "claim_check": ("all_pairs_ci_above_zero" if n_sig == len(scored) else "mixed"),
        },
        "research_only": True,
        "live_pnl_claim": False,
    }
    args.out.write_text(json.dumps(receipt, indent=2))
    print(
        f"{args.horizon}: {len(assets)} assets, {len(pooled_rows)} pooled pairs, "
        f"{n_sig} significant@95, min_ci_lo={min_ci:.5f} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
