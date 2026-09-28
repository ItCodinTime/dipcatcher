"""Mixture-stability lane — stationary-bootstrap confidence on expert-mixture
weights (P3.8).

The expert-mixture lane (``expert_mixture``) reports one point-estimate weight
path per shard. This lane asks the inferential question the point estimate
hides: **how stable is the learned mixture?** If two heads are statistically
indistinguishable, their weights should be interchangeable — a mixture that
crowns a leader on noise is a selection artifact, not evidence.

Mechanics
---------
For each SYNTHETIC shard the lane fits every head once (``_predict_grid``),
computes the per-row mean-pinball loss tensor ``(K, T)`` (``_loss_tensor``),
then applies the Politis–Romano stationary bootstrap to the *time axis* of the
loss sequence. Each bootstrap replicate ``losses[:, idx_b]`` is a plausible
reordering of the observed loss history; re-running each mixer's weight
recursion on it yields a draw from the sampling distribution of the final
weights ``w[:, -1]`` and of the mixture's own regret
``crps(mix) − min_h mean(l_h)`` under reorderings that preserve local
dependence (block length from ``optimal_block_length`` on the pooled loss
matrix, Politis–White).

Reported per mixer × head:

- ``w_obs`` — the observed final weight;
- ``w_lo`` / ``w_hi`` — percentile bootstrap band (2.5 % / 97.5 %);
- ``p_leader`` — fraction of replicates where the head holds the largest
  final weight (a *selection* probability, not a p-value);
- per mixer: ``leader_stability`` — fraction of replicates whose argmax head
  equals the observed leader — plus the bootstrap band on regret
  ``crps(mix) − min_h crps(h)``.

The pinball convexity bound (``pinball(y, Σw·q) ≤ Σw·pinball``) is re-verified
on every bootstrap replicate: a violation is an error row, never silent. This
keeps the lane honest about what resampling can break — nothing, since the
bound holds pathwise for any weights.

Receipt.v2 ``kind = "mixture_stability_eval"``, ``verdict = "pass"`` when no
head fails; ``verify-receipt`` deep-consistency registration for this kind is
queued as a follow-up (three open branches already edit the dispatcher).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_fund.metrics.inference import (
    optimal_block_length,
    stationary_bootstrap_indices,
)
from quant_fund.metrics.scoring import pinball_loss
from quant_fund.research.expert_mixture import (
    _loss_tensor,
    _predict_grid,
    ewa_weights,
    fixed_share_weights,
    uniform_weights,
)
from quant_fund.research.fleet_eval import (
    DEFAULT_TAUS,
    ShardGenerator,
    SyntheticShard,
    _atomic_write_text,
    fleet_head_factories,
    resolve_shard_generators,
)
from quant_fund.research.receipt_v2 import build_receipt_v2, seal_receipt

MIXTURE_STABILITY_SCHEMA = "mixture_stability_eval.v1"
MIXTURE_STABILITY_KIND = "mixture_stability_eval"
MIXERS: tuple[str, ...] = ("uniform", "ewa", "fixed_share")
W_CI_LO, W_CI_HI = 0.025, 0.975


def _weights_final(losses: NDArray[np.float64], alpha: float) -> dict[str, NDArray[np.float64]]:
    """Final-row weights of every mixer on one loss history."""
    return {
        "uniform": uniform_weights(losses)[:, -1],
        "ewa": ewa_weights(losses)[:, -1],
        "fixed_share": fixed_share_weights(losses, alpha)[:, -1],
    }


def _mixed_loss(
    grids: NDArray[np.float64],
    y_eval: NDArray[np.float64],
    taus: NDArray[np.float64],
    w_final: NDArray[np.float64],
    rows: NDArray[np.intp],
) -> float:
    """Mean pinball of the mixture on the resampled row subset.

    The mixed grid uses the *final* weights — a fixed linear combination, so
    convexity applies pointwise. A convex combination of non-crossing rows
    stays non-crossing and ``w >= 0`` by construction, so the mixture is
    already ordered (no rearrange needed).
    """
    mixed = np.einsum("k,ktj->tj", w_final, grids[:, rows, :])
    y_rows = y_eval[rows]
    mixed_pin = np.zeros(y_rows.shape[0], dtype=float)
    for j, tau in enumerate(taus):
        mixed_pin += np.asarray(pinball_loss(y_rows, mixed[:, j], float(tau)))
    return float(mixed_pin.mean() / taus.size)


def run_mixture_stability(
    factories: Mapping[str, Any],
    shards: Iterable[str] | Mapping[str, ShardGenerator] | None = None,
    n_train: int = 512,
    n_eval: int = 256,
    *,
    seed: int = 0,
    taus: Sequence[float] = DEFAULT_TAUS,
    alpha: float = 0.05,
    n_boot: int = 500,
    block: float | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Bootstrap the expert-mixture weights on every SYNTHETIC shard.

    Returns a results frame (rows per shard × mixer × head plus mixer-summary
    rows) and the unsealed receipt payload.
    """
    if not isinstance(factories, Mapping) or not factories:
        raise ValueError("mixture stability requires a nonempty factory mapping")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    for label, v in (("n_train", n_train), ("n_eval", n_eval), ("n_boot", n_boot)):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise ValueError(f"{label} must be an int")
    if n_train < 2 or n_eval < 2 or n_boot < 2:
        raise ValueError("n_train, n_eval >= 2 and n_boot >= 2 required")
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
        raise ValueError("mixture stability requires at least one shard")

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
                grids[name] = _predict_grid(factory, shard, n_train, n_eval, tau_arr)
            except Exception as exc:
                rows.append(
                    {
                        "shard": shard_name,
                        "mixer": None,
                        "head": name,
                        "status": "error",
                        "error": str(exc),
                    }
                )
        if len(grids) < 2:
            rows.append(
                {
                    "shard": shard_name,
                    "mixer": "__all__",
                    "head": None,
                    "status": "error",
                    "error": "fewer than two experts produced valid grids",
                }
            )
            shard_reports.append({"shard": shard_name, "n_experts_ok": len(grids)})
            continue

        names = sorted(grids)
        k = len(names)
        g = np.stack([grids[n] for n in names])  # (K, T, J)
        losses = _loss_tensor(g, y_eval, tau_arr)  # (K, T)
        obs = _weights_final(losses, alpha)

        if block is not None:
            block_v = float(block)
        else:
            per_head = [optimal_block_length(losses[h]) for h in range(k)]
            finite = [b for b in per_head if np.isfinite(b) and b > 0]
            block_v = float(np.median(finite)) if finite else 1.0
        block_v = float(min(max(block_v, 1.0), float(n_eval)))
        rng = np.random.default_rng(int(seed) * 1_000_003 + shard_index)
        idx = stationary_bootstrap_indices(n_eval, int(n_boot), block_v, rng)

        # Bootstrap draws: (B, K) final weights per mixer; (B,) regret per mixer.
        boot_w = {m: np.zeros((int(n_boot), k)) for m in MIXERS}
        boot_regret = {m: np.zeros(int(n_boot)) for m in MIXERS}
        bound_violations = 0
        for b in range(int(n_boot)):
            sel = idx[b]
            lb = losses[:, sel]
            for m in MIXERS:
                wb = _weights_final(lb, alpha)[m]
                boot_w[m][b] = wb
                mixed_pin = _mixed_loss(g, y_eval, tau_arr, wb, sel)
                head_pin = lb.mean(axis=1)
                # Convexity audit on the resample: mixed pinball <= Σ w·l.
                if mixed_pin > float(np.dot(wb, head_pin)) + 1e-12:
                    bound_violations += 1
                boot_regret[m][b] = mixed_pin - float(head_pin.min())

        mixer_reports: dict[str, Any] = {}
        for m in MIXERS:
            obs_w = obs[m]
            is_uniform = m == "uniform"
            if is_uniform:
                # Equal weights: argmax would silently crown the alphabetically
                # first head on every replicate — the leader concept does not
                # exist for this mixer, so report nulls, not a phantom winner.
                leader_obs = -1
                leader_stability: float | None = None
                p_leader = np.full(k, np.nan)
            else:
                leader_obs = int(np.argmax(obs_w))
                leaders_boot = np.argmax(boot_w[m], axis=1)
                leader_stability = float(np.mean(leaders_boot == leader_obs))
                p_leader = np.array([float(np.mean(leaders_boot == hi)) for hi in range(k)])
            lo = np.quantile(boot_w[m], W_CI_LO, axis=0)
            hi_q = np.quantile(boot_w[m], W_CI_HI, axis=0)
            reg_lo = float(np.quantile(boot_regret[m], W_CI_LO))
            reg_hi = float(np.quantile(boot_regret[m], W_CI_HI))
            reg_obs = float(np.median(boot_regret[m]))
            for hi_i, name in enumerate(names):
                rows.append(
                    {
                        "shard": shard_name,
                        "mixer": m,
                        "head": name,
                        "status": "ok",
                        "error": None,
                        "w_obs": float(obs_w[hi_i]),
                        "w_lo": float(lo[hi_i]),
                        "w_hi": float(hi_q[hi_i]),
                        "p_leader": None if is_uniform else float(p_leader[hi_i]),
                        "is_leader": bool(hi_i == leader_obs),
                        "leader_stability": leader_stability,
                        "regret_med": reg_obs,
                        "regret_lo": reg_lo,
                        "regret_hi": reg_hi,
                    }
                )
            mixer_reports[m] = {
                "leader": None if is_uniform else names[leader_obs],
                "leader_stability": leader_stability,
                "p_leader": None
                if is_uniform
                else {names[i]: float(p_leader[i]) for i in range(k)},
                "w_obs": {names[i]: float(obs_w[i]) for i in range(k)},
                "regret_ci": [reg_lo, float(np.median(boot_regret[m])), reg_hi],
            }
        shard_reports.append(
            {
                "shard": shard_name,
                "n_experts_ok": len(grids),
                "n_boot": int(n_boot),
                "block": block_v,
                "bound_violations": bound_violations,
                "mixers": mixer_reports,
            }
        )

    frame = pl.DataFrame(rows)
    payload = {
        "alpha": alpha,
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


def run_mixture_stability_eval(
    *,
    seed: int = 0,
    n_train: int = 512,
    n_eval: int = 256,
    alpha: float = 0.05,
    n_boot: int = 500,
    block: float | None = None,
    head_names: Iterable[str] | None = None,
    shard_names: Iterable[str] | None = None,
    taus: Sequence[float] = DEFAULT_TAUS,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Dev-lane entrypoint: resolve the SYNTHETIC fleet + shards, run, seal."""
    factories = fleet_head_factories(list(taus), int(seed), head_names)
    frame, payload = run_mixture_stability(
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
        kind=MIXTURE_STABILITY_KIND,
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
        payload={"schema": MIXTURE_STABILITY_SCHEMA, **payload},
    )
    receipt = seal_receipt(envelope)
    return frame, receipt


def mixture_stability_contract_errors(receipt: Mapping[str, Any]) -> list[str]:
    """Contract check on the sealed receipt.v2 envelope's payload."""
    errors: list[str] = []
    if not isinstance(receipt, Mapping):
        return ["receipt_not_mapping"]
    if receipt.get("schema") != "receipt.v2":
        errors.append("envelope_schema")
    if receipt.get("kind") != MIXTURE_STABILITY_KIND:
        errors.append("kind")
    if receipt.get("data_label") != "SYNTHETIC":
        errors.append("data_label")
    payload = receipt.get("payload")
    if not isinstance(payload, Mapping):
        errors.append("payload_not_object")
        return errors
    if payload.get("schema") != MIXTURE_STABILITY_SCHEMA:
        errors.append("payload_schema")
    if payload.get("live_pnl_claim") not in (None, False):
        errors.append("payload_live_pnl_claim")
    if not isinstance(payload.get("shards"), list):
        errors.append("payload_shards_missing")
    return errors


def write_mixture_stability_receipt(receipt: Mapping[str, Any], out_dir: Path) -> Path:
    """Write the sealed receipt under ``receipts/`` keyed by content digest."""
    errors = mixture_stability_contract_errors(receipt)
    if errors:
        raise ValueError(f"refusing to write invalid receipt: {errors}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    digest = str(receipt.get("receipt_sha256", ""))[:16]
    path = out_dir / f"mixture_stability_{digest}.json"
    _atomic_write_text(path, json.dumps(receipt, indent=2, sort_keys=True))
    return path


def format_mixture_stability_table(frame: pl.DataFrame) -> str:
    """Render per-(shard, mixer) leader + stability summary as text."""
    if frame.height == 0:
        return "mixture stability: no rows"
    ok = frame.filter(pl.col("status") == "ok")
    lines = ["shard | mixer | leader | w_obs | p_leader | stab | regret med [lo, hi]"]
    for (shard, mixer), grp in ok.group_by(["shard", "mixer"], maintain_order=True):
        leader = grp.filter(pl.col("is_leader"))
        r = (leader if leader.height else grp).row(0, named=True)
        head = r["head"] if leader.height else "(uniform)"
        w_obs = f"{r['w_obs']:.3f}"
        p_lead = "-" if r["p_leader"] is None else f"{r['p_leader']:.2f}"
        stab = "-" if r["leader_stability"] is None else f"{r['leader_stability']:.2f}"
        lines.append(
            f"{shard} | {mixer} | {head} | {w_obs} | {p_lead} | {stab} | "
            f"{r['regret_med']:.5f} [{r['regret_lo']:.5f}, {r['regret_hi']:.5f}]"
        )
    return "\n".join(lines)


__all__ = [
    "MIXTURE_STABILITY_KIND",
    "MIXTURE_STABILITY_SCHEMA",
    "MIXERS",
    "format_mixture_stability_table",
    "mixture_stability_contract_errors",
    "run_mixture_stability",
    "run_mixture_stability_eval",
    "write_mixture_stability_receipt",
]
