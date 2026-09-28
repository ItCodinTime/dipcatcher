"""Multi-horizon distribution bench on seeded SYNTHETIC shards (P3.2-lite).

``fleet_eval`` scores only one-step forecasts: each eval row's quantile
vector is scored against the next single return. That contract cannot score
horizon-aligned heads — ``HStepScaledDistribution`` emits per-horizon blocks
whose ``h>1`` cells were disclosed in ULTRAPLAN as "fleet scoring remains
open pending horizon-aligned targets". This lane closes that gap.

Protocol: every model maps ``(y[:n_train], horizons, taus)`` to a quantile
matrix of shape ``(n_horizons, n_taus)`` — a distribution for the forward
h-step *sum* ``y[t:t+h].sum()``. Targets are non-overlapping forward sums
starting at ``n_train`` (stride ``h`` per horizon keeps consecutive targets
disjoint, so no engineered serial dependence inside a scored column).
Scores are proper rules only: per-τ pinball, quantile CRPS, PIT-KS, and
central 80/90% coverage. All output is SYNTHETIC correctness evidence.

Models:
    hstep_t / hstep_emp — the two native constructions of
        ``HStepScaledDistribution`` (Student-t iid-sum scaling, overlapping
        empirical bootstrap).
    gaussian_iid — the naive null: ``h*mu + sqrt(h)*sigma*z_tau``.
    ewma_iid — RiskMetrics λ=0.94 trailing vol with iid-sum scaling; the
        vol-timing baseline the challenger must beat on clustered shards.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
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
    pit_values,
    rearrange_quantiles,
)
from quant_fund.models.hstep import HStepScaledDistribution
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent
from quant_fund.research.fleet_eval import (
    COVERAGE_LEVELS,
    DEFAULT_TAUS,
    SHARD_GENERATORS,
    SyntheticShard,
    _atomic_write_text,
    resolve_shard_generators,
)
from quant_fund.research.receipt_v2 import build_receipt_v2, seal_receipt
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

Array = NDArray[np.float64]

HSTEP_BENCH_SCHEMA = "hstep_bench.v1"
HSTEP_BENCH_KIND = "hstep_bench"
DEFAULT_HORIZONS: tuple[int, ...] = (1, 5, 20)
EWMA_LAMBDA = 0.94

# model name -> (y_window, horizons, taus) -> (n_horizons, n_taus) quantiles
HstepModel = Callable[[Array, tuple[int, ...], Array], Array]


def _check_window(y: Array, max_h: int) -> Array:
    yy = np.asarray(y, dtype=float).reshape(-1)
    yy = yy[np.isfinite(yy)]
    if yy.size < 30 + max_h:
        raise ValueError(f"hstep window needs >= {30 + max_h} finite observations")
    return yy


def _gaussian_iid(y: Array, horizons: tuple[int, ...], taus: Array) -> Array:
    """Naive null: h*mu + sqrt(h)*sigma*z_tau."""
    from scipy.stats import norm

    yy = _check_window(y, horizons[-1])
    mu = float(yy.mean())
    sig = float(yy.std(ddof=1))
    if not np.isfinite(sig) or sig <= 0.0:
        raise ValueError("gaussian_iid requires positive window variance")
    z = np.asarray(norm.ppf(np.asarray(taus, dtype=float)), dtype=float)
    q = np.empty((len(horizons), taus.size))
    for b, h in enumerate(horizons):
        q[b] = h * mu + np.sqrt(h) * sig * z
    return np.asarray(rearrange_quantiles(q), dtype=float)


def _ewma_sigma(yy: Array, lam: float = EWMA_LAMBDA) -> float:
    """RiskMetrics trailing sigma: recursion s_t^2 = lam*s_{t-1}^2 + (1-lam)*r_t^2."""
    r2 = (yy - yy.mean()) ** 2
    s2 = float(r2[0])
    for v in r2[1:]:
        s2 = lam * s2 + (1.0 - lam) * float(v)
    return float(np.sqrt(max(s2, 0.0)))


def _ewma_iid(y: Array, horizons: tuple[int, ...], taus: Array) -> Array:
    """Vol-timing baseline: h*mu + sqrt(h)*sigma_ewma*z_tau."""
    from scipy.stats import norm

    yy = _check_window(y, horizons[-1])
    sig = _ewma_sigma(yy)
    if not np.isfinite(sig) or sig <= 0.0:
        raise ValueError("ewma_iid requires positive window variance")
    mu = float(yy.mean())
    z = np.asarray(norm.ppf(np.asarray(taus, dtype=float)), dtype=float)
    q = np.empty((len(horizons), taus.size))
    for b, h in enumerate(horizons):
        q[b] = h * mu + np.sqrt(h) * sig * z
    return np.asarray(rearrange_quantiles(q), dtype=float)


def _hstep(block: str) -> HstepModel:
    def _quantiles(y: Array, horizons: tuple[int, ...], taus: Array) -> Array:
        model = HStepScaledDistribution(list(map(float, taus)), horizons=horizons)
        model.fit(np.ones((y.size, 1)), y)
        if block == "student_t":
            assert model.q_student_ is not None
            return np.asarray(model.q_student_, dtype=float)
        assert model.q_emp_ is not None
        return np.asarray(model.q_emp_, dtype=float)

    return _quantiles


HSTEP_MODELS: dict[str, HstepModel] = {
    "gaussian_iid": _gaussian_iid,
    "ewma_iid": _ewma_iid,
    "hstep_t": _hstep("student_t"),
    "hstep_emp": _hstep("empirical"),
}


def resolve_hstep_models(names: Iterable[str] | None = None) -> dict[str, HstepModel]:
    chosen = list(HSTEP_MODELS) if names is None else [str(n).strip() for n in names]
    unknown = sorted(set(chosen) - set(HSTEP_MODELS))
    if unknown:
        raise ValueError(f"unknown hstep model(s): {', '.join(unknown)}")
    return {name: HSTEP_MODELS[name] for name in chosen}


def _pinball_key(tau: float) -> str:
    return f"pinball_{tau:g}"


def _coverage_key(level: float) -> str:
    return f"coverage_{int(round(level * 100))}"


def _central_interval_index(taus: Array, level: float) -> tuple[int, int] | None:
    lo, hi = (1.0 - level) / 2.0, (1.0 + level) / 2.0
    i = np.flatnonzero(np.isclose(taus, lo, atol=1e-9))
    j = np.flatnonzero(np.isclose(taus, hi, atol=1e-9))
    if i.size == 0 or j.size == 0:
        return None
    return int(i[0]), int(j[0])


def _forward_sums(y: Array, start: int, end: int, h: int) -> Array:
    """Non-overlapping forward h-sums y[t:t+h].sum() for t in [start, end), stride h."""
    c = np.empty(y.size + 1, dtype=float)
    c[0] = 0.0
    np.cumsum(y, out=c[1:])
    starts = np.arange(start, end - h + 1, h, dtype=int)
    return np.asarray(c[starts + h] - c[starts], dtype=float)


def _score_cell(
    shard: SyntheticShard,
    shard_seed: int,
    name: str,
    model: HstepModel,
    n_train: int,
    n_eval: int,
    horizons: tuple[int, ...],
    taus: Array,
    coverage_index: dict[float, tuple[int, int] | None],
) -> list[dict[str, Any]]:
    base: dict[str, Any] = {
        "shard": shard.name,
        "model": name,
        "status": "ok",
        "error": None,
        "n_train": n_train,
        "seed": shard_seed,
    }
    try:
        q_all = np.asarray(model(shard.y[:n_train], horizons, taus), dtype=float)
        if q_all.shape != (len(horizons), taus.size):
            raise ValueError(
                f"model returned shape {q_all.shape}; expected ({len(horizons)}, {taus.size})"
            )
        if not np.isfinite(q_all).all():
            raise ValueError("model returned non-finite quantiles")
        if np.any(np.diff(q_all, axis=1) < 0.0):
            raise ValueError("model returned crossing quantiles")
    except Exception as exc:
        return [
            {
                **base,
                "horizon": h,
                "n_targets": None,
                "status": "error",
                "error": str(exc),
                "crps": None,
                "pit_ks": None,
                "pit_ks_p": None,
                **{_coverage_key(level): None for level in COVERAGE_LEVELS},
                **{_pinball_key(float(t)): None for t in taus},
            }
            for h in horizons
        ]
    rows: list[dict[str, Any]] = []
    for b, h in enumerate(horizons):
        row: dict[str, Any] = {**base, "horizon": h}
        targets = _forward_sums(shard.y, n_train, n_train + n_eval, h)
        row["n_targets"] = int(targets.size)
        if targets.size == 0:
            row.update(
                {
                    "status": "error",
                    "error": "no horizon-aligned targets in eval slice",
                    "crps": None,
                    "pit_ks": None,
                    "pit_ks_p": None,
                    **{_coverage_key(level): None for level in COVERAGE_LEVELS},
                    **{_pinball_key(float(t)): None for t in taus},
                }
            )
            rows.append(row)
            continue
        q = q_all[b]
        row["crps"] = crps_from_quantiles(targets, np.tile(q, (targets.size, 1)), taus)
        ks, ks_p = pit_ks(pit_values(targets, np.tile(q, (targets.size, 1)), taus))
        row["pit_ks"] = ks
        # Targets are disjoint blocks per h, but shard generators may still
        # plant serial dependence inside the block — suppress p there.
        row["pit_ks_p"] = None if shard.config.get("serial_dependence") else ks_p
        for level, pair in coverage_index.items():
            if pair is not None:
                row[_coverage_key(level)] = coverage(
                    targets, np.full(targets.size, q[pair[0]]), np.full(targets.size, q[pair[1]])
                )
        for j, tau in enumerate(taus):
            row[_pinball_key(float(tau))] = mean_pinball(
                targets, np.full(targets.size, q[j]), float(tau)
            )
        rows.append(row)
    return rows


def run_hstep_bench(
    models: Mapping[str, HstepModel] | None = None,
    shards: Iterable[str] | Mapping[str, Any] | None = None,
    n_train: int = 512,
    n_eval: int = 256,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    seed: int = 0,
    taus: Sequence[float] = DEFAULT_TAUS,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Score multi-horizon constructions against forward h-step sums.

    Returns the results frame plus the unsealed receipt payload. A model that
    fails emits ``error`` rows (visible, never silent); the harness fails
    closed on degenerate arguments, degenerate shard output, malformed model
    output, or forbidden headline metrics in the payload.
    """
    resolved_models = dict(HSTEP_MODELS) if models is None else dict(models)
    if not resolved_models:
        raise ValueError("hstep bench requires a nonempty model mapping")
    hs: list[int] = []
    for h in horizons:
        if isinstance(h, (bool, np.bool_)) or not isinstance(h, (int, np.integer)) or int(h) < 1:
            raise ValueError("horizons must be positive integers")
        hs.append(int(h))
    if not hs or len(set(hs)) != len(hs):
        raise ValueError("horizons must be distinct positive integers")
    hs_t = tuple(sorted(hs))
    if (
        isinstance(n_train, bool)
        or not isinstance(n_train, (int, np.integer))
        or n_train < 1
        or isinstance(n_eval, bool)
        or not isinstance(n_eval, (int, np.integer))
        or n_eval < 1
    ):
        raise ValueError("n_train and n_eval must be positive")
    n_train, n_eval = int(n_train), int(n_eval)
    if n_eval < hs_t[-1]:
        raise ValueError("n_eval must cover the largest horizon")
    tau_arr = np.asarray(list(taus), dtype=float)
    if (
        tau_arr.size == 0
        or not np.isfinite(tau_arr).all()
        or np.any((tau_arr <= 0.0) | (tau_arr >= 1.0))
        or np.any(np.diff(tau_arr) <= 0.0)
    ):
        raise ValueError("taus must be a nonempty strictly increasing grid inside (0, 1)")
    resolved: Mapping[str, Any]
    if shards is None:
        resolved = SHARD_GENERATORS
    elif isinstance(shards, Mapping):
        resolved = shards
    else:
        resolved = resolve_shard_generators(shards)
    if not resolved:
        raise ValueError("hstep bench requires at least one shard")

    coverage_index = {level: _central_interval_index(tau_arr, level) for level in COVERAGE_LEVELS}
    n_shard = n_train + n_eval
    rows: list[dict[str, Any]] = []
    shard_meta: dict[str, Any] = {}
    for shard_index, (shard_name, generator) in enumerate(resolved.items()):
        shard_seed = int(seed) + shard_index
        shard = generator(n_shard, shard_seed)
        if not isinstance(shard, SyntheticShard):
            raise ValueError(f"shard {shard_name!r} did not return a SyntheticShard")
        y = np.asarray(shard.y, dtype=float).reshape(-1)
        if shard.name != shard_name or shard.config.get("data_label") != "SYNTHETIC":
            raise ValueError(f"shard {shard_name!r} must match its name and SYNTHETIC label")
        if y.size != n_shard or not np.isfinite(y).all():
            raise ValueError(f"shard {shard_name!r} produced {y.size} rows; needs {n_shard}")
        shard = SyntheticShard(shard.name, shard.x, y, dict(shard.config))
        shard_meta[shard_name] = {
            "n": int(y.size),
            "seed": shard_seed,
            "x_sha256": hash_bytes(np.asarray(shard.x, dtype=float).tobytes()),
            "y_sha256": hash_bytes(y.tobytes()),
            "config": shard.config,
        }
        for model_name, model in resolved_models.items():
            rows.extend(
                _score_cell(
                    shard,
                    shard_seed,
                    model_name,
                    model,
                    n_train,
                    n_eval,
                    hs_t,
                    tau_arr,
                    coverage_index,
                )
            )

    columns = [
        "shard",
        "model",
        "horizon",
        "status",
        "error",
        "n_train",
        "n_targets",
        "seed",
        "crps",
        "pit_ks",
        "pit_ks_p",
        *[_coverage_key(level) for level in COVERAGE_LEVELS],
        *[_pinball_key(float(t)) for t in tau_arr],
    ]
    schema = {
        "shard": pl.String,
        "model": pl.String,
        "status": pl.String,
        "error": pl.String,
        "horizon": pl.Int64,
        "n_train": pl.Int64,
        "n_targets": pl.Int64,
        "seed": pl.Int64,
        **{
            k: pl.Float64
            for k in columns
            if k
            not in {"shard", "model", "status", "error", "horizon", "n_train", "n_targets", "seed"}
        },
    }
    frame = pl.DataFrame(rows, schema=schema, orient="row").select(columns)

    inputs_sha256 = hash_bytes(
        canonical_json_bytes(
            {
                "shards": {
                    name: {"x_sha256": meta["x_sha256"], "y_sha256": meta["y_sha256"]}
                    for name, meta in shard_meta.items()
                },
                "models": sorted(str(k) for k in resolved_models),
                "horizons": list(hs_t),
                "taus": [float(t) for t in tau_arr],
                "n_train": n_train,
                "n_eval": n_eval,
                "seed": int(seed),
            }
        )
    )
    receipt: dict[str, Any] = {
        "schema": HSTEP_BENCH_SCHEMA,
        "kind": HSTEP_BENCH_KIND,
        "data_label": "SYNTHETIC",
        "live_pnl_claim": False,
        "dev_only": True,
        "generated_at": datetime.now(UTC).isoformat(),
        "git_revision": git_revision(),
        "seed": int(seed),
        "n_train": n_train,
        "n_eval": n_eval,
        "horizons": list(hs_t),
        "taus": [float(t) for t in tau_arr],
        "models": sorted(str(k) for k in resolved_models),
        "shards": shard_meta,
        "inputs_sha256": inputs_sha256,
        "n_rows": len(rows),
        "n_error_rows": sum(1 for row in rows if row["status"] != "ok"),
        "results": frame.to_dicts(),
    }
    return frame, receipt


def hstep_bench_v1_contract_errors(receipt: Mapping[str, Any]) -> list[str]:
    """Fail-closed contract for a ``hstep_bench.v1`` payload (writer + verifier)."""
    research_blob = {key: value for key, value in receipt.items() if key != "live_pnl_claim"}
    errors: list[str] = []
    if receipt.get("schema") != HSTEP_BENCH_SCHEMA:
        errors.append("schema_not_hstep_bench_v1")
    if receipt.get("kind") != HSTEP_BENCH_KIND:
        errors.append("kind_not_hstep_bench")
    if receipt.get("data_label") != "SYNTHETIC":
        errors.append("data_label_not_synthetic")
    if receipt.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim_not_false")
    if not isinstance(receipt.get("results"), list) or not receipt["results"]:
        errors.append("results_missing_or_empty")
    if not family_blob_forbidden_metrics_absent(research_blob):
        errors.append("forbidden_metric_keys")
    return errors


def hstep_bench_dataset_identity(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """The dataset identity bound by a v2 ``dataset_hash``: shard content hashes."""
    shards = receipt.get("shards")
    if not isinstance(shards, Mapping):
        raise ValueError("hstep_bench receipt has no shards block")
    dataset: dict[str, Any] = {}
    for name, meta in shards.items():
        if not isinstance(meta, Mapping):
            raise ValueError(f"hstep_bench shard {name!r} metadata is not an object")
        dataset[str(name)] = {
            "x_sha256": meta.get("x_sha256"),
            "y_sha256": meta.get("y_sha256"),
        }
    return dataset


def hstep_bench_params(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """The run parameters bound by a v2 ``params_hash``."""
    shards = receipt.get("shards")
    return {
        "models": receipt.get("models"),
        "horizons": receipt.get("horizons"),
        "taus": receipt.get("taus"),
        "n_train": receipt.get("n_train"),
        "n_eval": receipt.get("n_eval"),
        "seed": receipt.get("seed"),
        "shards": sorted(str(name) for name in shards) if isinstance(shards, Mapping) else None,
    }


def hstep_bench_v1_verdict(receipt: Mapping[str, Any]) -> str:
    """pass iff every scored cell is ok; a recorded error is a fail."""
    n_error_rows = receipt.get("n_error_rows")
    return "pass" if n_error_rows == 0 else "fail"


def hstep_bench_receipt_v2(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Wrap a ``hstep_bench.v1`` payload in the unified ``receipt.v2`` envelope.

    Validates the v1 contract first — a malformed v1 receipt is never wrapped.
    """
    if hstep_bench_v1_contract_errors(receipt):
        raise ValueError("hstep_bench receipt violates its synthetic research contract")
    return build_receipt_v2(
        kind=str(receipt["kind"]),
        data_label=str(receipt["data_label"]),
        dataset=hstep_bench_dataset_identity(receipt),
        params=hstep_bench_params(receipt),
        code_files=(Path(__file__),),
        verdict=hstep_bench_v1_verdict(receipt),
        payload=dict(receipt),
        generated_at=str(receipt["generated_at"]),
        revision=str(receipt["git_revision"]),
    )


def hstep_bench_v2_consistency_errors(envelope: Mapping[str, Any]) -> list[str]:
    """Re-derive a hstep_bench receipt.v2 envelope's bound digests from its payload."""
    errors: list[str] = []
    payload = envelope.get("payload")
    if not isinstance(payload, Mapping):
        return ["payload_not_object"]
    contract_errors = hstep_bench_v1_contract_errors(payload)
    errors.extend(f"payload_{name}" for name in contract_errors)
    if contract_errors:
        return errors
    try:
        dataset = hstep_bench_dataset_identity(payload)
    except ValueError as exc:
        return [*errors, f"payload_{exc}"]
    if hash_bytes(canonical_json_bytes(dataset)) != envelope.get("dataset_hash"):
        errors.append("dataset_hash_mismatch")
    if hash_bytes(canonical_json_bytes(hstep_bench_params(payload))) != envelope.get("params_hash"):
        errors.append("params_hash_mismatch")
    if hstep_bench_v1_verdict(payload) != envelope.get("verdict"):
        errors.append("verdict_mismatch")
    return errors


def write_hstep_bench_receipt(
    receipt: Mapping[str, Any],
    out_dir: Path | str = Path("receipts"),
    *,
    receipt_version: int = 1,
) -> Path:
    """Seal a hstep_bench receipt and write ``receipts/hstep_bench_<hash>.json``.

    The filename hash is the sha256 of the canonical sealed receipt; the same
    digest is embedded as ``receipt_sha256``. The write is atomic and refuses
    to replace an existing different receipt. ``receipt_version=2`` wraps the
    v1 payload in the unified ``receipt.v2`` envelope before sealing.
    """
    if receipt_version == 1:
        if hstep_bench_v1_contract_errors(receipt):
            raise ValueError("hstep_bench receipt violates its synthetic research contract")
        body: Mapping[str, Any] = receipt
    elif receipt_version == 2:
        body = hstep_bench_receipt_v2(receipt)
    else:
        raise ValueError(f"receipt_version must be 1 or 2, got {receipt_version!r}")
    sealed = seal_receipt(body)
    digest = str(sealed["receipt_sha256"])
    import json

    path = Path(out_dir) / f"hstep_bench_{digest[:16]}.json"
    _atomic_write_text(path, json.dumps(sealed, indent=2, sort_keys=True) + "\n")
    return path


def hstep_bench_report(frame: pl.DataFrame) -> str:
    """Compact per-(shard, model, horizon) CRPS/coverage table for stdout."""
    lines = ["shard                 model          h    crps      pit_ks   cov80   cov90"]
    for row in frame.sort(["shard", "model", "horizon"]).iter_rows(named=True):
        if row["status"] != "ok":
            lines.append(
                f"{row['shard']:<21} {row['model']:<14} {row['horizon']:>2}  ERROR: {row['error']}"
            )
            continue
        lines.append(
            f"{row['shard']:<21} {row['model']:<14} {row['horizon']:>2}  "
            f"{row['crps']:>8.4f}  {row['pit_ks']:>7.3f}  "
            f"{row['coverage_80']:>6.3f}  {row['coverage_90']:>6.3f}"
        )
    return "\n".join(lines)
