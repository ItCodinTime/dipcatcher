"""Selection-concordance lane — do the multiple-comparison selectors agree?
(P3.9).

Every inferential primitive in ``metrics/snooping.py`` answers a different
question about the same loss tensor:

- **MCS** (Hansen–Lunde–Nason): which heads form a model confidence set at
  ``alpha`` — the eliminated set is ``~included``.
- **StepM** (Romano–Wolf): per-head FWER-controlled rejection of
  ``E[l_h - l_best] <= 0`` — the rejected set is the eliminated set.
- **DM matrix** (Diebold–Mariano + Andrews–Monahan LRV): head ``h`` is
  eliminated when *some* competitor ``j`` beats it at level ``alpha``
  (``t(l_h - l_j) > 0`` with ``p < alpha``).
- **SPA / Reality Check** (global): on the differentials-vs-best matrix they
  test "is any head *significantly worse* than the observed winner" — a
  decisiveness check, not a per-head set.

When these agree, "head X is best" is robust to the choice of multiple-
comparison correction. When they disagree — MCS keeps a head StepM rejects,
or the DM matrix eliminates nothing while SPA fires — the result is
dependence-fragile: the selection is an artifact of the bootstrap scheme, not
the data. This lane measures that disagreement directly instead of trusting
one selector's answer.

Reported per shard:

- ``eliminated_<selector>`` sets + the eliminated-set *intersection*;
- per-head ``elimination_confidence`` per selector (MCS: ``1 - p_h``; StepM:
  ``1 - p_adj_h``; DM: ``max_j t(l_h - l_j)``) and **Kendall's τ** between
  every selector pair over those confidence vectors;
- **Jaccard** between every eliminated-set pair and a strict ``concordant``
  flag (identical eliminated sets AND identical leader);
- global decisiveness: ``spa_p_consistent``, ``rc_p``, both on the
  differentials matrix, plus ``decisive = min(p) < alpha``.

Receipt.v2 ``kind = "selection_concordance"``; verdict ``pass`` when no head
fails. ``verify-receipt`` re-derives the embedded agreement stats
(intersection, ``concordant``, Jaccard pairs, ``decisive``) via
``concordance_consistency_errors``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy import stats as sstats

from quant_fund.metrics.hac import dm_hac_tstat
from quant_fund.metrics.scoring import pinball_loss
from quant_fund.metrics.snooping import (
    MIN_OBS,
    model_confidence_set,
    reality_check,
    spa_test,
    stepm,
)
from quant_fund.research.fleet_eval import (
    DEFAULT_TAUS,
    HeadFactory,
    ShardGenerator,
    SyntheticShard,
    _atomic_write_text,
    fleet_head_factories,
    resolve_shard_generators,
)
from quant_fund.research.receipt_v2 import build_receipt_v2, seal_receipt

CONCORDANCE_SCHEMA = "selection_concordance.v1"
CONCORDANCE_KIND = "selection_concordance"


def _predict_grid(
    shard: SyntheticShard,
    factory: HeadFactory,
    n_train: int,
    n_eval: int,
    taus: np.ndarray,
) -> np.ndarray:
    """Fit one head on the leading slice; return its ``(n_eval, n_taus)`` grid.

    Mirrors ``fleet_eval._score_row`` / ``expert_mixture._predict_grid``
    exactly, including the lagged predict path: ``fleet_lagged_predict``
    heads consume ``y[t-1]`` — already observed at each forecast origin — so
    no lookahead enters. (Local copy while ``expert_mixture`` is in-flight;
    unify once it lands.)
    """
    model = factory()
    model.fit(shard.x[:n_train], shard.y[:n_train])
    if getattr(model, "fleet_lagged_predict", False):
        lag_x = shard.y[n_train - 1 : n_train + n_eval - 1].reshape(-1, 1)
        q = np.asarray(model.predict(lag_x), dtype=float)
    else:
        q = np.asarray(model.predict(shard.x[n_train : n_train + n_eval]), dtype=float)
    if q.ndim != 2 or q.shape[0] != n_eval or q.shape[1] != taus.size:
        raise ValueError(f"predict returned shape {q.shape}; expected ({n_eval}, {taus.size})")
    if not np.isfinite(q).all():
        raise ValueError("predict returned non-finite quantiles")
    if np.any(np.diff(q, axis=1) < 0.0):
        raise ValueError("predict returned crossing quantiles")
    return q


def _loss_tensor(
    grids: NDArray[np.float64],
    y_eval: NDArray[np.float64],
    taus: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Per-(head, row) mean-pinball loss — the selector input tensor."""
    k, t, _ = grids.shape
    losses = np.zeros((k, t), dtype=float)
    for h in range(k):
        for j, tau in enumerate(taus):
            losses[h] += np.asarray(pinball_loss(y_eval, grids[h, :, j], float(tau)))
    return losses / taus.size


def _dm_eliminated(
    losses: Mapping[str, NDArray[np.float64]], alpha: float
) -> dict[str, dict[str, Any]]:
    """Per-head DM elimination: ``h`` eliminated iff some ``j`` beats it.

    Mirrors the ordered pairwise convention of ``fleet_significance``:
    ``t(l_h - l_j) > 0`` and ``p < alpha`` means ``h`` is significantly worse
    than ``j``. Confidence is ``max_j t(l_h - l_j)`` — larger = more clearly
    dominated. (Local copy of the matrix logic; unify with
    ``fleet_significance._dm_matrix`` once that module lands on main.)
    """
    names = sorted(losses)
    out: dict[str, dict[str, Any]] = {}
    for h in names:
        t_max = float("-inf")
        beaten_by: list[str] = []
        for j in names:
            if h == j:
                continue
            if np.array_equal(losses[h], losses[j]):
                continue  # identical series cannot separate the heads
            res = dm_hac_tstat(losses[h] - losses[j])
            t = float(res["t"])
            p = float(res["pvalue"])
            if np.isfinite(t):
                t_max = max(t_max, t)
                if t > 0.0 and np.isfinite(p) and p < alpha:
                    beaten_by.append(j)
        out[h] = {
            "eliminated": bool(beaten_by),
            "confidence": None if t_max == float("-inf") else t_max,
            "beaten_by": beaten_by,
        }
    return out


def _kendall(a: Sequence[float | None], b: Sequence[float | None]) -> float | None:
    pairs = [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    x = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    if np.all(x == x[0]) or np.all(y == y[0]):
        return None
    return float(sstats.kendalltau(x, y).statistic)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return float(len(a & b) / len(union)) if union else 1.0


def run_concordance(
    factories: Mapping[str, Any],
    shards: Iterable[str] | Mapping[str, ShardGenerator] | None = None,
    n_train: int = 512,
    n_eval: int = 256,
    *,
    seed: int = 0,
    taus: Sequence[float] = DEFAULT_TAUS,
    alpha: float = 0.10,
    n_boot: int = 500,
    block: float | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Run every selector on each shard's loss tensor; measure agreement."""
    if not isinstance(factories, Mapping) or not factories:
        raise ValueError("concordance requires a nonempty factory mapping")
    if not np.isfinite(alpha) or not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be finite and in (0, 1)")
    for label, v in (("n_train", n_train), ("n_eval", n_eval), ("n_boot", n_boot)):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise ValueError(f"{label} must be an int")
    if n_train < 2 or n_eval < MIN_OBS or n_boot < 2:
        raise ValueError(f"n_train >= 2, n_eval >= {MIN_OBS}, n_boot >= 2 required")
    tau_arr = np.asarray(list(taus), dtype=float)
    if tau_arr.ndim != 1 or tau_arr.size < 2 or np.any(np.diff(tau_arr) <= 0):
        raise ValueError("taus must be a strictly increasing sequence")
    if np.any((tau_arr <= 0.0) | (tau_arr >= 1.0)):
        raise ValueError("taus must lie in (0, 1)")
    if block is not None and (not np.isfinite(block) or block <= 0):
        raise ValueError("block must be a positive finite float or None")
    if shards is None:
        gen = resolve_shard_generators(None)
    elif isinstance(shards, Mapping):
        gen = dict(shards)
    else:
        gen = resolve_shard_generators(shards)
    if not gen:
        raise ValueError("concordance requires at least one shard")

    n_shard = n_train + n_eval
    rows: list[dict[str, Any]] = []
    shard_reports: list[dict[str, Any]] = []
    for shard_index, (shard_name, make_shard) in enumerate(gen.items()):
        shard_seed = int(seed) + shard_index
        shard = make_shard(n_shard, shard_seed)
        if not isinstance(shard, SyntheticShard):
            raise ValueError(f"shard {shard_name!r} did not return a SyntheticShard")
        y = np.asarray(shard.y, dtype=float).reshape(-1)
        x = np.asarray(shard.x, dtype=float)
        if shard.name != shard_name or shard.config.get("data_label") != "SYNTHETIC":
            raise ValueError(f"shard {shard_name!r} must match its name and SYNTHETIC label")
        if y.size != n_shard or x.ndim != 2 or x.shape[0] != n_shard:
            raise ValueError(
                f"shard {shard_name!r} produced {y.size} rows; needs exactly {n_shard}"
            )
        if not np.isfinite(y).all() or not np.isfinite(x).all():
            raise ValueError(f"shard {shard_name!r} produced non-finite data")
        shard = SyntheticShard(shard.name, x, y, dict(shard.config))
        y_eval = shard.y[n_train : n_train + n_eval]

        grids: dict[str, NDArray[np.float64]] = {}
        for name, factory in factories.items():
            try:
                grids[name] = _predict_grid(shard, factory, n_train, n_eval, tau_arr)
            except Exception as exc:
                rows.append(
                    {
                        "shard": shard_name,
                        "head": name,
                        "status": "error",
                        "error": str(exc),
                    }
                )
        if len(grids) < 2:
            rows.append(
                {
                    "shard": shard_name,
                    "head": "__all__",
                    "status": "error",
                    "error": "fewer than two heads produced valid grids",
                }
            )
            shard_reports.append({"shard": shard_name, "n_heads_ok": len(grids)})
            continue

        names = sorted(grids)
        k = len(names)
        g = np.stack([grids[n] for n in names])
        losses = {n: _loss_tensor(g[i : i + 1], y_eval, tau_arr)[0] for i, n in enumerate(names)}
        loss_mat = np.column_stack([losses[n] for n in names])
        perf = -loss_mat  # selectors take higher-is-better performance

        best_i = int(np.argmin(loss_mat.mean(axis=0)))
        d = loss_mat - loss_mat[:, best_i : best_i + 1]  # >=0; positive = worse

        # Selector 1 — MCS on negative losses.
        mcs = model_confidence_set(
            perf, n_boot=int(n_boot), block=block, alpha=alpha, seed=int(seed) + shard_index
        )
        elim_mcs = {names[i] for i, inc in enumerate(mcs.included) if not inc}
        conf_mcs = {
            names[i]: None if not np.isfinite(mcs.p_values[i]) else 1.0 - float(mcs.p_values[i])
            for i in range(k)
        }

        # Selector 2 — Romano–Wolf StepM on differentials vs the observed best.
        # stepm drops near-constant columns internally and does not report
        # which, so replicate _prepare's drop rule here and map results back:
        # a head indistinguishable from the winner is never rejected.
        spread = np.ptp(d, axis=0)
        scale_ref = np.maximum(np.abs(d).max(axis=0), 1e-12)
        keep = spread > 1e-10 * scale_ref
        keep_idx = np.flatnonzero(keep)
        sm = stepm(
            d[:, keep],
            n_boot=int(n_boot),
            block=block,
            alpha=alpha,
            seed=int(seed) + shard_index,
        )
        rejected_full = np.zeros(k, dtype=bool)
        adj_full = np.full(k, np.nan)
        for j, gi in enumerate(keep_idx):
            rejected_full[gi] = sm.rejected[j]
            adj_full[gi] = sm.adjusted_p[j]
        elim_stepm = {names[i] for i in range(k) if rejected_full[i]}
        conf_stepm = {
            names[i]: None if not np.isfinite(adj_full[i]) else 1.0 - float(adj_full[i])
            for i in range(k)
        }

        # Selector 3 — DM pairwise elimination.
        dm = _dm_eliminated(losses, alpha)
        elim_dm = {h for h, v in dm.items() if v["eliminated"]}
        conf_dm = {h: v["confidence"] for h, v in dm.items()}

        # Global decisiveness — SPA and Reality Check on the differentials.
        spa = spa_test(d, n_boot=int(n_boot), block=block, seed=int(seed) + shard_index)
        rc = reality_check(d, n_boot=int(n_boot), block=block, seed=int(seed) + shard_index)
        decisive = (
            bool(min(spa.p_consistent, rc.p_value) < alpha)
            if np.isfinite(min(spa.p_consistent, rc.p_value))
            else False
        )

        eliminated_sets = {"mcs": elim_mcs, "stepm": elim_stepm, "dm": elim_dm}
        intersection = set.intersection(*eliminated_sets.values()) if eliminated_sets else set()
        concordant = len({frozenset(s) for s in eliminated_sets.values()}) == 1

        conf_matrix = {
            "mcs": conf_mcs,
            "stepm": conf_stepm,
            "dm": conf_dm,
        }
        pair_stats: dict[str, dict[str, float | None]] = {}
        for a, b in combinations(("mcs", "stepm", "dm"), 2):
            pair_stats[f"{a}_vs_{b}"] = {
                "jaccard": _jaccard(eliminated_sets[a], eliminated_sets[b]),
                "kendall_tau": _kendall(
                    [conf_matrix[a][h] for h in names], [conf_matrix[b][h] for h in names]
                ),
            }

        for i, h in enumerate(names):
            rows.append(
                {
                    "shard": shard_name,
                    "head": h,
                    "status": "ok",
                    "error": None,
                    "mean_loss": float(loss_mat[:, i].mean()),
                    "is_best": i == best_i,
                    "eliminated_mcs": h in elim_mcs,
                    "eliminated_stepm": h in elim_stepm,
                    "eliminated_dm": h in elim_dm,
                    "eliminated_all": h in intersection,
                    "conf_mcs": conf_mcs[h],
                    "conf_stepm": conf_stepm[h],
                    "conf_dm": conf_dm[h],
                }
            )
        shard_reports.append(
            {
                "shard": shard_name,
                "n_heads_ok": k,
                "alpha": float(alpha),
                "n_boot": int(n_boot),
                "block": block if block is not None else mcs.block,
                "best": names[best_i],
                "eliminated": {s: sorted(e) for s, e in eliminated_sets.items()},
                "eliminated_intersection": sorted(intersection),
                "concordant": bool(concordant),
                "pair_stats": pair_stats,
                "spa_p_consistent": None
                if not np.isfinite(spa.p_consistent)
                else float(spa.p_consistent),
                "rc_p": None if not np.isfinite(rc.p_value) else float(rc.p_value),
                "decisive": decisive,
            }
        )

    frame = pl.DataFrame(rows)
    payload = {
        "alpha": float(alpha),
        "n_train": n_train,
        "n_eval": n_eval,
        "n_boot": int(n_boot),
        "block": None if block is None else float(block),
        "n_taus": int(tau_arr.size),
        "seed": seed,
        "shards": shard_reports,
        "n_rows": frame.height,
    }
    return frame, payload


def run_concordance_eval(
    *,
    seed: int = 0,
    n_train: int = 512,
    n_eval: int = 256,
    alpha: float = 0.10,
    n_boot: int = 500,
    block: float | None = None,
    head_names: Iterable[str] | None = None,
    shard_names: Iterable[str] | None = None,
    taus: Sequence[float] = DEFAULT_TAUS,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Dev-lane entrypoint: resolve the SYNTHETIC fleet + shards, run, seal."""
    factories = fleet_head_factories(list(taus), int(seed), head_names)
    frame, payload = run_concordance(
        factories,
        shard_names,
        n_train,
        n_eval,
        seed=seed,
        taus=taus,
        alpha=alpha,
        n_boot=n_boot,
        block=block,
    )
    n_errors = int(
        frame.filter(pl.col("status") == "error").height if "status" in frame.columns else 0
    )
    envelope = build_receipt_v2(
        kind=CONCORDANCE_KIND,
        data_label="SYNTHETIC",
        dataset={
            "shards": [
                {"name": s["shard"], "generator": "resolve_shard_generators"}
                for s in payload["shards"]
            ],
            "heads": sorted(factories),
        },
        params={
            "alpha": alpha,
            "n_train": n_train,
            "n_eval": n_eval,
            "n_boot": int(n_boot),
            "block": block,
            "seed": seed,
            "taus": [float(t) for t in taus],
        },
        code_files=(Path(__file__),),
        verdict="pass" if n_errors == 0 else "fail",
        payload={"schema": CONCORDANCE_SCHEMA, **payload},
    )
    return frame, seal_receipt(envelope)


def concordance_contract_errors(receipt: Mapping[str, Any]) -> list[str]:
    """Contract check on the sealed receipt.v2 envelope's payload."""
    errors: list[str] = []
    if not isinstance(receipt, Mapping):
        return ["receipt_not_mapping"]
    if receipt.get("schema") != "receipt.v2":
        errors.append("envelope_schema")
    if receipt.get("kind") != CONCORDANCE_KIND:
        errors.append("kind")
    if receipt.get("data_label") != "SYNTHETIC":
        errors.append("data_label")
    payload = receipt.get("payload")
    if not isinstance(payload, Mapping):
        errors.append("payload_not_object")
        return errors
    if payload.get("schema") != CONCORDANCE_SCHEMA:
        errors.append("payload_schema")
    if payload.get("live_pnl_claim") not in (None, False):
        errors.append("payload_live_pnl_claim")
    if not isinstance(payload.get("shards"), list):
        errors.append("payload_shards_missing")
    return errors


def concordance_consistency_errors(body: Mapping[str, Any]) -> list[str]:
    """Re-derive the embedded agreement stats from the sealed elimination
    sets — catches a tampered ``concordant``/``pair_stats``/intersection."""
    payload = body.get("payload")
    if not isinstance(payload, Mapping):
        return []
    shards = payload.get("shards")
    if not isinstance(shards, list):
        return []
    errors: list[str] = []
    for report in shards:
        if not isinstance(report, Mapping):
            continue
        tag = report.get("shard", "?")
        eliminated = report.get("eliminated")
        if not isinstance(eliminated, Mapping):
            continue
        sets = {s: set(v) for s, v in eliminated.items() if isinstance(v, list)}
        intersection = sorted(set.intersection(*sets.values())) if sets else []
        if report.get("eliminated_intersection") != intersection:
            errors.append(f"intersection_mismatch:{tag}")
        expected_concordant = len({frozenset(s) for s in sets.values()}) == 1
        if bool(report.get("concordant")) != expected_concordant:
            errors.append(f"concordant_mismatch:{tag}")
        pair_stats = report.get("pair_stats")
        if isinstance(pair_stats, Mapping):
            for pair_key, stats in pair_stats.items():
                parts = str(pair_key).split("_vs_")
                if len(parts) != 2 or not isinstance(stats, Mapping):
                    continue
                a, b = parts
                if a in sets and b in sets and "jaccard" in stats:
                    want = _jaccard(sets[a], sets[b])
                    got = stats.get("jaccard")
                    if got is None or not math.isclose(
                        float(got), want, rel_tol=0.0, abs_tol=1e-12
                    ):
                        errors.append(f"jaccard_mismatch:{tag}:{pair_key}")
        alpha = payload.get("alpha")
        spa_p = report.get("spa_p_consistent")
        rc_p = report.get("rc_p")
        if isinstance(alpha, (int, float)) and spa_p is not None and rc_p is not None:
            want_decisive = min(float(spa_p), float(rc_p)) < float(alpha)
            if bool(report.get("decisive")) != want_decisive:
                errors.append(f"decisive_mismatch:{tag}")
    return errors


def write_concordance_receipt(receipt: Mapping[str, Any], out_dir: Path) -> Path:
    """Write the sealed receipt under ``receipts/`` keyed by content digest."""
    errors = concordance_contract_errors(receipt)
    if errors:
        raise ValueError(f"refusing to write invalid receipt: {errors}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    digest = str(receipt.get("receipt_sha256", ""))[:16]
    path = out_dir / f"concordance_{digest}.json"
    _atomic_write_text(path, json.dumps(receipt, indent=2, sort_keys=True))
    return path


def format_concordance_table(frame: pl.DataFrame) -> str:
    """Render per-head elimination agreement as text."""
    if frame.height == 0:
        return "selection concordance: no rows"
    ok = frame.filter(pl.col("status") == "ok")
    lines = ["shard | head | loss | mcs | stepm | dm | all"]
    for r in ok.iter_rows(named=True):
        lines.append(
            f"{r['shard']} | {r['head']} | {r['mean_loss']:.5f} | "
            f"{'X' if r['eliminated_mcs'] else '-'} | "
            f"{'X' if r['eliminated_stepm'] else '-'} | "
            f"{'X' if r['eliminated_dm'] else '-'} | "
            f"{'X' if r['eliminated_all'] else '-'}"
        )
    return "\n".join(lines)


__all__ = [
    "CONCORDANCE_KIND",
    "CONCORDANCE_SCHEMA",
    "concordance_consistency_errors",
    "concordance_contract_errors",
    "format_concordance_table",
    "run_concordance",
    "run_concordance_eval",
    "write_concordance_receipt",
]
