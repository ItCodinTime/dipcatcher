"""Causal online expert-mixture over fleet heads — prediction with expert advice.

Every fleet head emits a conditional quantile grid on the eval slice. This
module then mixes the heads *row by row* with weights driven only by losses
realized strictly before the forecast row — the classic prediction-with-
expert-advice problem, here with proper-scored quantile experts:

- ``uniform``     : static equal-weight mixture (the naive ensemble baseline).
- ``ewa``         : exponentially weighted average forecaster,
                    ``w_h(t) ∝ exp(-η_t · Σ_{k<t} l_h(k))`` where ``l_h(k)`` is
                    the mean pinball of head ``h`` at row ``k``. The step size
                    ``η_t`` is scale-free and causal: ``1 / σ`` of the losses
                    observed so far (uniform weights until ≥2 losses seen).
- ``fixed_share`` : EWA followed by a share update ``w ← (1-α)w + α/K``
                    (Herbster–Warmuth), so the mixture can snap to a new expert
                    after a regime break instead of converging permanently.

Two properties are asserted, not assumed:

1. Convexity bound (theorem, checked per row in tests): pinball is convex in
   the quantile, so ``pinball(y, Σ w_h q_h, τ) ≤ Σ w_h pinball(y, q_h, τ)`` —
   the mixture cannot lose more than the weight-averaged expert loss. The same
   inequality lifts to quantile-CRPS.
2. Causality: weights at row ``t`` depend only on ``l_h(k<t)``. Tests verify
   that perturbing future shard data leaves every earlier mixed row identical.

Honest scope: mixtures trade a bounded regret against the best *fixed* expert
for robustness; they are not claimed to beat the best head on homogeneous
shards — on regime/vol-break shards the fixed-share variant's tracking is the
interesting object. All books are SYNTHETIC; metrics are proper scores only
(CRPS, per-τ pinball, PIT-KS, coverage) — never P&L/Sharpe.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_fund.metrics.probability import pit_ks
from quant_fund.metrics.scoring import (
    coverage,
    crps_from_quantiles,
    mean_pinball,
    pinball_loss,
    pit_values,
    rearrange_quantiles,
)
from quant_fund.research.fleet_eval import (
    COVERAGE_LEVELS,
    DEFAULT_TAUS,
    ShardGenerator,
    SyntheticShard,
    _atomic_write_text,
    fleet_head_factories,
    resolve_shard_generators,
)
from quant_fund.research.receipt_v2 import build_receipt_v2, seal_receipt

EXPERT_MIXTURE_SCHEMA = "expert_mixture_eval.v1"
MIXERS = ("uniform", "ewa", "fixed_share")


def _predict_grid(
    factory: Any,
    shard: SyntheticShard,
    n_train: int,
    n_eval: int,
    taus: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Fit a head on the train slice and emit its ``(n_eval, n_taus)`` grid.

    Mirrors ``fleet_eval._score_row``'s predict path: one-step series heads
    consume the lag-1 observed return at each forecast origin, so no future
    data enters the prediction.
    """
    model = factory()
    model.fit(shard.x[:n_train], shard.y[:n_train])
    if getattr(model, "fleet_lagged_predict", False):
        lag_x = shard.y[n_train - 1 : n_train + n_eval - 1].reshape(-1, 1)
        q = np.asarray(model.predict(lag_x), dtype=float)
    else:
        q = np.asarray(model.predict(shard.x[n_train : n_train + n_eval]), dtype=float)
    if q.ndim != 2 or q.shape != (n_eval, taus.size):
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
    """Per-(head, row) mean-pinball loss — the expert loss sequence."""
    k, t, _ = grids.shape
    losses = np.zeros((k, t), dtype=float)
    for h in range(k):
        for j, tau in enumerate(taus):
            losses[h] += np.asarray(pinball_loss(y_eval, grids[h, :, j], float(tau)))
    return losses / taus.size


def uniform_weights(losses: NDArray[np.float64]) -> NDArray[np.float64]:
    """Static equal weights for every row — the naive ensemble baseline."""
    k, t = losses.shape
    return np.full((k, t), 1.0 / k)


def ewa_weights(losses: NDArray[np.float64]) -> NDArray[np.float64]:
    """Exponentially weighted average forecaster with a causal scale-free η.

    Multiplicative update ``w_i(t) ∝ w_i(t-1) · exp(-η_t · l_i(t-1))`` where
    ``η_t = 1 / σ_t`` and ``σ_t`` is the standard deviation of the expert
    losses realized strictly before row ``t`` (η dimensioned in inverse-loss
    units). Uniform until two distinct losses have been observed.
    """
    k, t = losses.shape
    w = np.full((k, t), 1.0 / k)
    prev = np.full(k, 1.0 / k)
    for i in range(1, t):
        sigma = float(np.std(losses[:, :i]))
        eta = 1.0 / sigma if sigma > 1e-12 else 0.0
        prev = prev * np.exp(-eta * losses[:, i - 1])
        prev = prev / prev.sum()
        w[:, i] = prev
    return w


def fixed_share_weights(losses: NDArray[np.float64], alpha: float = 0.05) -> NDArray[np.float64]:
    """Herbster–Warmuth fixed-share: EWA on per-round losses, then a share
    update ``w ← (1-α)·w + α/K``.

    Unlike cumulative-loss EWA — which is the same multiplicative recursion
    and therefore cannot forgive an expert's early catastrophes — the share
    term keeps every expert's weight ≥ α/K, so a head that becomes best after
    a regime break regains dominance in O(log(1/α)/η) rows.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    k, t = losses.shape
    w = np.full((k, t), 1.0 / k)
    prev = np.full(k, 1.0 / k)
    for i in range(1, t):
        sigma = float(np.std(losses[:, :i]))
        eta = 1.0 / sigma if sigma > 1e-12 else 0.0
        prev = prev * np.exp(-eta * losses[:, i - 1])
        prev = prev / prev.sum()
        prev = (1.0 - alpha) * prev + alpha / k
        w[:, i] = prev
    return w


def _mix_grid(grids: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
    """Weighted quantile mixture per row, rearranged to restore monotonicity."""
    mixed = np.einsum("kt,ktj->tj", weights, grids)
    return np.asarray(rearrange_quantiles(mixed), dtype=float)


def _grid_metrics(
    y_eval: NDArray[np.float64],
    q: NDArray[np.float64],
    taus: NDArray[np.float64],
    shard: SyntheticShard,
) -> dict[str, float | None]:
    """Proper-score row for one quantile grid (mirrors the fleet row shape)."""
    ks, ks_p = pit_ks(pit_values(y_eval, q, taus))
    row: dict[str, float | None] = {
        "crps": crps_from_quantiles(y_eval, q, taus),
        "pit_ks": ks,
        "pit_ks_p": None if shard.config.get("serial_dependence") else ks_p,
    }
    for level in COVERAGE_LEVELS:
        lo = np.flatnonzero(np.isclose(taus, (1.0 - level) / 2.0, atol=1e-9))
        hi = np.flatnonzero(np.isclose(taus, (1.0 + level) / 2.0, atol=1e-9))
        key = f"coverage_{int(round(level * 100))}"
        row[key] = coverage(y_eval, q[:, lo[0]], q[:, hi[0]]) if lo.size and hi.size else None
    for j, tau in enumerate(taus):
        row[f"pinball_{tau:g}"] = mean_pinball(y_eval, q[:, j], float(tau))
    return row


def run_expert_mixture(
    factories: Mapping[str, Any],
    shards: Iterable[str] | Mapping[str, ShardGenerator] | None = None,
    n_train: int = 512,
    n_eval: int = 256,
    *,
    seed: int = 0,
    taus: Sequence[float] = DEFAULT_TAUS,
    alpha: float = 0.05,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Mix fleet heads online on each shard; score the mixtures causally.

    Returns a results frame (one row per shard × {each expert + each mixer})
    plus the unsealed receipt payload. Experts that fail to fit/predict are
    excluded from the mixture for that shard and recorded as error rows —
    visible, never silent. Best-expert regret is reported per mixer:
    ``crps_mixer - min_h crps_head``.
    """
    if not isinstance(factories, Mapping) or not factories:
        raise ValueError("expert mixture requires a nonempty factory mapping")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    for label, v in (("n_train", n_train), ("n_eval", n_eval)):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 2:
            raise ValueError(f"{label} must be an int >= 2")
    tau_arr = np.asarray(list(taus), dtype=float)
    if tau_arr.ndim != 1 or tau_arr.size < 2 or np.any(np.diff(tau_arr) <= 0):
        raise ValueError("taus must be a strictly increasing sequence")
    if np.any((tau_arr <= 0.0) | (tau_arr >= 1.0)):
        raise ValueError("taus must lie in (0, 1)")
    gen = resolve_shard_generators(shards)

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
        head_crps: dict[str, float] = {}
        for name, factory in factories.items():
            try:
                grid = _predict_grid(factory, shard, n_train, n_eval, tau_arr)
            except Exception as exc:
                rows.append(
                    {
                        "shard": shard_name,
                        "member": name,
                        "role": "expert",
                        "status": "error",
                        "error": str(exc),
                    }
                )
                continue
            metrics = _grid_metrics(y_eval, grid, tau_arr, shard)
            head_crps[name] = float(metrics["crps"])  # type: ignore[arg-type]
            grids[name] = grid
            rows.append(
                {
                    "shard": shard_name,
                    "member": name,
                    "role": "expert",
                    "status": "ok",
                    "error": None,
                    **metrics,
                }
            )

        if len(grids) < 2:
            rows.append(
                {
                    "shard": shard_name,
                    "member": "__all__",
                    "role": "mixer",
                    "status": "error",
                    "error": "fewer than two experts produced valid grids",
                }
            )
            shard_reports.append({"shard": shard_name, "n_experts_ok": len(grids), "mixers": {}})
            continue

        names = sorted(grids)
        g = np.stack([grids[n] for n in names])  # (K, T, J)
        losses = _loss_tensor(g, y_eval, tau_arr)  # (K, T)
        best_head = min(head_crps.values())
        worst_head = max(head_crps.values())

        mixes: dict[str, NDArray[np.float64]] = {
            "uniform": uniform_weights(losses),
            "ewa": ewa_weights(losses),
            "fixed_share": fixed_share_weights(losses, alpha=alpha),
        }
        mixer_report: dict[str, Any] = {}
        for mixer_name, w in mixes.items():
            q_mix = _mix_grid(g, w)
            metrics = _grid_metrics(y_eval, q_mix, tau_arr, shard)
            hhi = np.sum(w * w, axis=0)
            eff_k = np.where(hhi > 0, 1.0 / np.maximum(hhi, 1e-12), np.nan)
            crps_mix = float(metrics["crps"])  # type: ignore[arg-type]
            rows.append(
                {
                    "shard": shard_name,
                    "member": mixer_name,
                    "role": "mixer",
                    "status": "ok",
                    "error": None,
                    "regret_vs_best": crps_mix - best_head,
                    "eff_experts_min": float(np.nanmin(eff_k)),
                    "eff_experts_mean": float(np.nanmean(eff_k)),
                    **metrics,
                }
            )
            mixer_report[mixer_name] = {
                "crps": crps_mix,
                "regret_vs_best_expert": crps_mix - best_head,
                "excess_over_worst_expert": crps_mix - worst_head,
                "eff_experts_min": float(np.nanmin(eff_k)),
                "eff_experts_mean": float(np.nanmean(eff_k)),
            }
        shard_reports.append(
            {
                "shard": shard_name,
                "n_experts_ok": len(grids),
                "best_expert_crps": best_head,
                "worst_expert_crps": worst_head,
                "mixers": mixer_report,
            }
        )

    frame = pl.DataFrame(rows)
    payload = {
        "alpha": alpha,
        "n_train": n_train,
        "n_eval": n_eval,
        "n_taus": int(tau_arr.size),
        "seed": seed,
        "shards": shard_reports,
        "n_rows": frame.height,
    }
    return frame, payload


def run_expert_mixture_eval(
    *,
    seed: int = 0,
    n_train: int = 512,
    n_eval: int = 256,
    alpha: float = 0.05,
    head_names: Iterable[str] | None = None,
    shard_names: Iterable[str] | None = None,
    taus: Sequence[float] = DEFAULT_TAUS,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Dev-lane entrypoint: resolve the SYNTHETIC fleet + shards, run, receipt."""
    tau_arr = list(taus)
    factories = fleet_head_factories(tau_arr, seed, names=head_names)
    frame, payload = run_expert_mixture(
        factories,
        shard_names,
        n_train=n_train,
        n_eval=n_eval,
        seed=seed,
        taus=tau_arr,
        alpha=alpha,
    )
    params = {
        "heads": sorted(factories.keys()),
        "alpha": alpha,
        "eta": "causal 1/sigma of realized losses (self-normalizing)",
        "scope_note": (
            "prediction-with-expert-advice over the SYNTHETIC fleet; mixtures "
            "are compared on best-fixed-expert regret, never claimed dominant"
        ),
    }
    payload["heads"] = sorted(factories.keys())
    receipt = build_receipt_v2(
        kind="expert_mixture_eval",
        data_label="SYNTHETIC",
        dataset={
            "name": "synthetic_shard_fleet",
            "source": "synthetic",
            "rows": payload["n_rows"],
        },
        params=params,
        code_files=(Path(__file__),),
        verdict="pass",
        payload=payload,
    )
    return frame, receipt


def write_expert_mixture_receipt(receipt: Mapping[str, Any], out_dir: Path) -> Path:
    """Seal and atomically persist the mixture receipt."""
    if receipt.get("kind") != "expert_mixture_eval" or receipt.get("data_label") != "SYNTHETIC":
        raise ValueError(
            "expert-mixture receipts must be kind=expert_mixture_eval with "
            "data_label=SYNTHETIC (honesty contract)"
        )
    sealed = seal_receipt(dict(receipt))
    digest = sealed["receipt_sha256"]
    path = out_dir / f"expert_mixture_{digest[:16]}.json"
    _atomic_write_text(path, json.dumps(sealed, indent=2, sort_keys=True) + "\n")
    return path


def format_expert_mixture_table(frame: pl.DataFrame) -> str:
    """ASCII digest: per shard, expert range and each mixer's CRPS + regret."""
    if frame.height == 0:
        return "expert_mixture: no rows"
    cols = ["shard", "member", "role", "status", "crps", "regret_vs_best", "eff_experts_mean"]
    view = frame.select([c for c in cols if c in frame.columns])
    lines = [f"expert_mixture_eval — {frame.height} rows (SYNTHETIC)"]
    for row in view.iter_rows(named=True):
        crps = row.get("crps")
        regret = row.get("regret_vs_best")
        eff = row.get("eff_experts_mean")
        lines.append(
            "  {shard:<14} {member:<12} {role:<6} {status:<5} crps={crps} "
            "regret={regret} effK={eff}".format(
                shard=row["shard"],
                member=row["member"],
                role=row["role"],
                status=row["status"],
                crps=f"{crps:.5f}" if isinstance(crps, (int, float)) else "-",
                regret=f"{regret:+.5f}" if isinstance(regret, (int, float)) else "-",
                eff=f"{eff:.2f}" if isinstance(eff, (int, float)) else "-",
            )
        )
    return "\n".join(lines)
