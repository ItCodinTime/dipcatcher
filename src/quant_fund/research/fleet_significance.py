"""Fleet significance: predictive-ability tests on the fleet's proper scores.

``run_distribution_fleet`` ranks challenger heads by mean pinball/CRPS but a
mean ranking says nothing about whether the gaps are statistically real. This
module is the inferential layer: for every shard it scores each head's
*per-row* loss series (mean pinball over the tau grid, or the unreduced
Gneiting–Raftery CRPS Riemann sum), then runs

- a pairwise Diebold–Mariano matrix — ``dm_hac_tstat`` on each ordered loss
  differential ``d_ij = l_i - l_j`` with the Andrews–Monahan prewhitened
  long-run variance (or classic Newey–West when ``prewhiten=False``); the sign
  convention is ``t > 0`` ⇒ head ``i`` is worse than head ``j``, and
- Hansen–Lunde–Nason (2011) model confidence sets — ``model_confidence_set``
  on negative losses with the stationary bootstrap, yielding the set of heads
  not significantly dominated.

A pooled lane standardizes each shard's losses by the shard's cross-head
standard deviation and concatenates, giving one matrix/set across regimes.

Honesty contract: losses are proper scores only; every shard is labeled
SYNTHETIC and results are correctness/calibration evidence, never market or
live-P&L claims. ``write_fleet_significance_receipt`` persists the sealed
receipt.v2 envelope under ``receipts/``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from quant_fund.metrics.hac import dm_hac_tstat
from quant_fund.metrics.scoring import pinball_loss
from quant_fund.metrics.snooping import model_confidence_set
from quant_fund.research.fleet_eval import (
    DEFAULT_TAUS,
    SHARD_GENERATORS,
    HeadFactory,
    ShardGenerator,
    SyntheticShard,
    _atomic_write_text,
    fleet_head_factories,
    resolve_shard_generators,
)
from quant_fund.research.receipt_v2 import build_receipt_v2, seal_receipt
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

FLEET_SIG_SCHEMA = "fleet_significance_eval.v1"
LOSS_FAMILIES: tuple[str, ...] = ("pinball", "crps")
_MCS_MIN_OBS = 10


def _predict_grid(
    shard: SyntheticShard,
    factory: HeadFactory,
    n_train: int,
    n_eval: int,
    taus: np.ndarray,
) -> np.ndarray:
    """Fit one head on the leading slice; return its ``(n_eval, n_taus)`` grid.

    Mirrors ``_score_row`` in ``fleet_eval`` exactly, including the lagged
    predict path: ``fleet_lagged_predict`` heads consume ``y[t-1]`` — already
    observed at each forecast origin — so no lookahead enters.
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


def _loss_series(q: np.ndarray, y_eval: np.ndarray, taus: np.ndarray, loss: str) -> np.ndarray:
    """Per-row proper-score loss ``(n_eval,)``. Smaller is better."""
    if loss == "pinball":
        # Mean pinball over the tau grid, per eval row.
        out = np.zeros(y_eval.shape[0], dtype=float)
        for j, tau in enumerate(taus):
            out += pinball_loss(y_eval, q[:, j], float(tau))
        return out / float(taus.size)
    if loss == "crps":
        # Unreduced Gneiting–Raftery Riemann sum: 2 * ∫ pinball_tau dτ.
        dt = np.diff(np.concatenate([[0.0], taus]))
        out = np.zeros(y_eval.shape[0], dtype=float)
        for j, tau in enumerate(taus):
            out += 2.0 * pinball_loss(y_eval, q[:, j], float(tau)) * float(dt[j])
        return out
    raise ValueError(f"loss must be one of {LOSS_FAMILIES}, got {loss!r}")


def _dm_matrix(
    losses: Mapping[str, np.ndarray],
    *,
    prewhiten: bool,
) -> dict[str, dict[str, dict[str, float | None]]]:
    """Ordered pairwise DM matrix. ``out[i][j]`` tests ``E[l_i - l_j] = 0``."""
    names = sorted(losses)
    out: dict[str, dict[str, dict[str, float | None]]] = {}
    for i in names:
        out[i] = {}
        for j in names:
            if i == j or np.array_equal(losses[i], losses[j]):
                # Identical loss series: the differential is exactly zero.
                out[i][j] = {"t": 0.0, "p": 1.0}
                continue
            res = dm_hac_tstat(losses[i] - losses[j], prewhiten=prewhiten)
            t = float(res["t"])
            p = float(res["pvalue"])
            out[i][j] = {
                "t": t if np.isfinite(t) else None,
                "p": p if np.isfinite(p) else None,
            }
    return out


def _mcs(
    losses: Mapping[str, np.ndarray],
    *,
    n_boot: int,
    block: float | None,
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    """Hansen MCS on negative losses (the API takes higher-is-better)."""
    names = sorted(losses)
    perf = -np.column_stack([losses[n] for n in names])
    result = model_confidence_set(
        perf, n_boot=int(n_boot), block=block, alpha=float(alpha), seed=int(seed)
    )
    return {
        "models": names,
        "alpha": float(alpha),
        "n_boot": int(n_boot),
        "block": float(result.block),
        "n_obs": int(result.n_obs),
        "included": {n: bool(i) for n, i in zip(names, result.included, strict=True)},
        "p_values": {
            n: (float(p) if np.isfinite(p) else None)
            for n, p in zip(names, result.p_values, strict=True)
        },
        "elimination_order": [names[i] for i in result.elimination_order],
    }


def _wins(
    dm: Mapping[str, Mapping[str, Mapping[str, float | None]]], alpha: float
) -> dict[str, int]:
    """Per-head count of opponents it significantly beats (t<0, p<alpha)."""
    return {
        i: sum(
            1
            for j, cell in row.items()
            if i != j
            and cell["t"] is not None
            and cell["p"] is not None
            and cell["t"] < 0.0
            and cell["p"] < alpha
        )
        for i, row in dm.items()
    }


def run_fleet_significance(
    factories: Mapping[str, HeadFactory],
    shards: Iterable[str] | Mapping[str, ShardGenerator] | None = None,
    n_train: int = 192,
    n_eval: int = 96,
    *,
    seed: int = 0,
    taus: Sequence[float] = DEFAULT_TAUS,
    loss: str = "pinball",
    n_boot: int = 500,
    alpha: float = 0.10,
    block: float | None = None,
    prewhiten: bool = True,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Score every head per row on each shard; test pairwise significance.

    Per-shard output is a DM matrix plus an MCS over ok heads; a pooled lane
    standardizes per-shard losses (cross-head std) and concatenates. A head
    that fails to fit or predict is recorded as an ``error`` row and excluded
    from that shard's tests. Fails closed when fewer than two heads score on
    any shard. Returns the results frame plus the unsealed receipt.v2 payload.
    """
    if not isinstance(factories, Mapping) or not factories:
        raise ValueError("significance requires a nonempty mapping of head factories")
    if (
        isinstance(n_train, bool)
        or not isinstance(n_train, (int, np.integer))
        or isinstance(n_eval, bool)
        or not isinstance(n_eval, (int, np.integer))
        or n_train < 1
        or n_eval < _MCS_MIN_OBS
    ):
        raise ValueError(f"n_train must be >=1 and n_eval >= {_MCS_MIN_OBS}")
    n_train = int(n_train)
    n_eval = int(n_eval)
    tau_arr = np.asarray(list(taus), dtype=float)
    if (
        tau_arr.size == 0
        or not np.isfinite(tau_arr).all()
        or np.any((tau_arr <= 0.0) | (tau_arr >= 1.0))
        or np.any(np.diff(tau_arr) <= 0.0)
    ):
        raise ValueError("taus must be a nonempty strictly increasing grid inside (0, 1)")
    if loss not in LOSS_FAMILIES:
        raise ValueError(f"loss must be one of {LOSS_FAMILIES}, got {loss!r}")
    if int(n_boot) < 1:
        raise ValueError("n_boot must be >= 1")
    if not np.isfinite(alpha) or not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be finite and in (0, 1)")
    resolved: Mapping[str, ShardGenerator]
    if shards is None:
        resolved = SHARD_GENERATORS
    elif isinstance(shards, Mapping):
        resolved = shards
    else:
        resolved = resolve_shard_generators(shards)
    if not resolved:
        raise ValueError("significance requires at least one shard")
    for name in resolved:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("shard names must be nonempty strings")

    n_shard = n_train + n_eval
    rows: list[dict[str, Any]] = []
    shard_meta: dict[str, Any] = {}
    shard_results: list[dict[str, Any]] = []
    pooled_losses: dict[str, list[np.ndarray]] = {name: [] for name in factories}
    n_error_rows = 0
    for shard_index, (shard_name, generator) in enumerate(resolved.items()):
        shard_seed = int(seed) + shard_index
        shard = generator(n_shard, shard_seed)
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
        shard_meta[shard_name] = {
            "n": int(y.size),
            "seed": shard_seed,
            "x_sha256": hash_bytes(x.tobytes()),
            "y_sha256": hash_bytes(y.tobytes()),
            "config": shard.config,
        }

        y_eval = y[n_train : n_train + n_eval]
        losses: dict[str, np.ndarray] = {}
        errors: dict[str, str] = {}
        for model_name in sorted(factories):
            try:
                q = _predict_grid(shard, factories[model_name], n_train, n_eval, tau_arr)
                losses[model_name] = _loss_series(q, y_eval, tau_arr, loss)
            except Exception as exc:  # recorded, never silent
                errors[model_name] = str(exc)
                n_error_rows += 1
        if len(losses) < 2:
            raise ValueError(f"shard {shard_name!r} has {len(losses)} scorable heads; need >= 2")
        dm = _dm_matrix(losses, prewhiten=prewhiten)
        mcs = _mcs(
            losses,
            n_boot=int(n_boot),
            block=block,
            alpha=float(alpha),
            seed=int(seed) + 10_000 + shard_index,
        )
        wins = _wins(dm, float(alpha))
        shard_results.append(
            {
                "shard": shard_name,
                "n_eval": n_eval,
                "loss": loss,
                "models": sorted(losses),
                "excluded": sorted(errors),
                "mean_loss": {n: float(np.mean(v)) for n, v in sorted(losses.items())},
                "dm": dm,
                "mcs": mcs,
            }
        )
        mean_losses = {n: float(np.mean(v)) for n, v in losses.items()}
        ranks = {n: r + 1 for r, n in enumerate(sorted(mean_losses, key=lambda k: mean_losses[k]))}
        # Per-shard standardization for the pooled lane: divide by the
        # cross-head pooled std so regimes with different loss scales pool.
        pooled_all = np.concatenate(list(losses.values()))
        scale = float(np.std(pooled_all))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        for model_name in factories:
            if model_name in losses:
                pooled_losses[model_name].append(losses[model_name] / scale)
                rows.append(
                    {
                        "shard": shard_name,
                        "model": model_name,
                        "status": "ok",
                        "error": None,
                        "loss": loss,
                        "mean_loss": mean_losses[model_name],
                        "rank": ranks[model_name],
                        "dm_wins": wins[model_name],
                        "mcs_included": mcs["included"][model_name],
                        "mcs_p": mcs["p_values"][model_name],
                    }
                )
            else:
                rows.append(
                    {
                        "shard": shard_name,
                        "model": model_name,
                        "status": "error",
                        "error": errors[model_name],
                        "loss": loss,
                        "mean_loss": None,
                        "rank": None,
                        "dm_wins": None,
                        "mcs_included": None,
                        "mcs_p": None,
                    }
                )

    pooled_ok = {n: v for n, v in pooled_losses.items() if v}
    pooled: dict[str, Any] | None = None
    if len(pooled_ok) >= 2:
        cat = {n: np.concatenate(v) for n, v in pooled_ok.items()}
        pooled = {
            "n_obs": int(next(iter(cat.values())).shape[0]),
            "loss": loss,
            "standardization": "per-shard cross-head std",
            "dm": _dm_matrix(cat, prewhiten=prewhiten),
            "mcs": _mcs(
                cat,
                n_boot=int(n_boot),
                block=block,
                alpha=float(alpha),
                seed=int(seed) + 20_000,
            ),
        }

    frame = pl.DataFrame(rows)
    payload: dict[str, Any] = {
        "schema": FLEET_SIG_SCHEMA,
        "kind": "fleet_significance_eval",
        "data_label": "SYNTHETIC",
        "live_pnl_claim": False,
        "n_train": n_train,
        "n_eval": n_eval,
        "seed": int(seed),
        "taus": [float(t) for t in tau_arr],
        "loss": loss,
        "n_boot": int(n_boot),
        "alpha": float(alpha),
        "block": None if block is None else float(block),
        "prewhiten": bool(prewhiten),
        "models": sorted(str(k) for k in factories),
        "shards": shard_meta,
        "shard_results": shard_results,
        "pooled": pooled,
        "n_error_rows": n_error_rows,
        "scope_note": (
            "Significance over SYNTHETIC per-row proper-score losses "
            "(DM/Andrews–Monahan pairwise matrix + Hansen–Lunde–Nason MCS). "
            "Correctness/calibration evidence only."
        ),
    }
    envelope = build_receipt_v2(
        kind="fleet_significance_eval",
        data_label="SYNTHETIC",
        dataset={
            name: {"x_sha256": m["x_sha256"], "y_sha256": m["y_sha256"]}
            for name, m in shard_meta.items()
        },
        params={
            "models": sorted(str(k) for k in factories),
            "taus": [float(t) for t in tau_arr],
            "n_train": n_train,
            "n_eval": n_eval,
            "seed": int(seed),
            "shards": sorted(shard_meta),
            "loss": loss,
            "n_boot": int(n_boot),
            "alpha": float(alpha),
            "block": None if block is None else float(block),
            "prewhiten": bool(prewhiten),
        },
        code_files=(Path(__file__),),
        verdict="pass" if n_error_rows == 0 else "fail",
        payload=payload,
    )
    return frame, envelope


def run_fleet_significance_eval(
    models: Iterable[str] | None = None,
    shards: Iterable[str] | None = None,
    *,
    seed: int,
    taus: Sequence[float] = DEFAULT_TAUS,
    n_train: int = 192,
    n_eval: int = 96,
    loss: str = "pinball",
    n_boot: int = 500,
    alpha: float = 0.10,
    block: float | None = None,
    prewhiten: bool = True,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Resolve the fleet registries, then run the significance harness."""
    factories = fleet_head_factories(taus, seed, models)
    resolved = resolve_shard_generators(shards)
    return run_fleet_significance(
        factories,
        resolved,
        n_train,
        n_eval,
        seed=seed,
        taus=taus,
        loss=loss,
        n_boot=n_boot,
        alpha=alpha,
        block=block,
        prewhiten=prewhiten,
    )


def fleet_significance_dataset_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The dataset identity bound by the envelope's ``dataset_hash``."""
    shards = payload.get("shards")
    if not isinstance(shards, Mapping):
        raise ValueError("significance receipt has no shards block")
    out: dict[str, Any] = {}
    for name, meta in shards.items():
        if not isinstance(meta, Mapping):
            raise ValueError(f"significance shard {name!r} metadata is not an object")
        out[str(name)] = {
            "x_sha256": meta.get("x_sha256"),
            "y_sha256": meta.get("y_sha256"),
        }
    return out


def fleet_significance_params(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The run parameters bound by the envelope's ``params_hash``."""
    shards = payload.get("shards")
    return {
        "models": payload.get("models"),
        "taus": payload.get("taus"),
        "n_train": payload.get("n_train"),
        "n_eval": payload.get("n_eval"),
        "seed": payload.get("seed"),
        "shards": sorted(str(name) for name in shards) if isinstance(shards, Mapping) else None,
        "loss": payload.get("loss"),
        "n_boot": payload.get("n_boot"),
        "alpha": payload.get("alpha"),
        "block": payload.get("block"),
        "prewhiten": payload.get("prewhiten"),
    }


def fleet_significance_v2_consistency_errors(
    envelope: Mapping[str, Any],
) -> list[str]:
    """Re-derive a significance receipt.v2 envelope's bound digests."""
    payload = envelope.get("payload")
    if not isinstance(payload, Mapping):
        return ["payload_not_object"]
    errors = fleet_significance_contract_errors(envelope)
    if errors:
        return errors
    try:
        dataset = fleet_significance_dataset_identity(payload)
    except ValueError as exc:
        return [*errors, f"payload_{exc}"]
    if hash_bytes(canonical_json_bytes(dataset)) != envelope.get("dataset_hash"):
        errors.append("dataset_hash_mismatch")
    if hash_bytes(canonical_json_bytes(fleet_significance_params(payload))) != envelope.get(
        "params_hash"
    ):
        errors.append("params_hash_mismatch")
    return errors


def fleet_significance_contract_errors(receipt: Mapping[str, Any]) -> list[str]:
    """Contract check on the sealed receipt.v2 envelope's payload."""
    errors: list[str] = []
    payload = receipt.get("payload")
    if not isinstance(payload, Mapping):
        return ["payload_not_object"]
    if receipt.get("schema") != "receipt.v2":
        errors.append("schema_not_receipt_v2")
    if payload.get("schema") != FLEET_SIG_SCHEMA:
        errors.append("payload_schema_mismatch")
    if payload.get("kind") != "fleet_significance_eval":
        errors.append("kind_mismatch")
    if payload.get("data_label") != "SYNTHETIC":
        errors.append("data_label_mismatch")
    if payload.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim")
    if not isinstance(payload.get("shard_results"), list):
        errors.append("shard_results_missing")
    return errors


def write_fleet_significance_receipt(
    receipt: Mapping[str, Any],
    receipts_dir: Path | str = Path("receipts"),
) -> Path:
    """Seal the envelope and write ``receipts/fleet_significance_<hash>.json``."""
    if fleet_significance_contract_errors(receipt):
        raise ValueError("fleet significance receipt violates its contract")
    payload = seal_receipt(receipt)
    path = Path(receipts_dir) / f"fleet_significance_{payload['receipt_sha256'][:16]}.json"
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def format_fleet_significance_table(frame: pl.DataFrame) -> str:
    """Compact per-(shard, model) summary for CLI output."""
    cols = (
        "shard",
        "model",
        "status",
        "mean_loss",
        "rank",
        "dm_wins",
        "mcs_included",
        "mcs_p",
    )
    return frame.select([c for c in cols if c in frame.columns]).__str__()
