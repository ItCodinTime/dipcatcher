"""Calibration evaluation of the distribution fleet on seeded SYNTHETIC shards.

Runs the identical (head, shard, origin) cells as ``fleet_eval``: each cell
fits on the leading ``n_train`` slice and forecasts the trailing ``n_eval``
rows at one walk-forward origin — ``fleet_lagged_predict`` heads consume the
observed lag-1 column (frozen coefficients, no refit, no lookahead). Where
``fleet_eval`` reports proper *scores*, this lane reports *calibration*
diagnostics per cell:

- ``pit_*``: the Diebold–Gunther–Tay (1998) PIT histogram — K equal-width
  bins, their shares, the chi-square flatness statistic, and its p-value
  (``metrics.density_forecast.pit_histogram``; the repo's existing stat).
- ``coverage_<L>``: empirical coverage of the symmetric central intervals
  the tau grid supports (fleet_eval's ``COVERAGE_LEVELS`` pairing).
- ``calibration_slope`` / ``calibration_intercept``: the quantile
  reliability regression — empirical hit rate ``h_j = P(y <= q_j)``
  OLS-regressed on the nominal coordinate ``tau_j`` across the grid. A
  calibrated forecaster has ``h(tau) == tau``: slope 1, intercept 0. This
  is the quantile analog of Cox's (1958) calibration slope; see Gneiting,
  Balabdaoui & Raftery (2007) for the reliability-diagram construction and
  Koenker & Machado (1999) §2 for the quantile goodness-of-fit frame.
- ``mz_slope`` / ``mz_intercept``: a Mincer–Zarnowitz (1969) OLS of the
  realized outcome on the predicted median quantile — the literal "regress
  realized outcome on predicted quantile" diagnostic. ``None`` when the
  predicted coordinate is (near-)constant across origins: a constant
  regressor carries no calibration information, which is the usual case
  for the fleet's unconditional heads on unconditional shards.

The harness fails closed on degenerate arguments or degenerate shard
output (raise); per-cell failures — insufficient PIT support, non-finite
coverage, or a diagnostic that cannot be computed honestly — are recorded
as ``status="error"`` rows (visible, never silent) and flip the receipt
verdict to ``fail``. ``write_calibration_receipt`` seals JSON evidence
under ``receipts/`` in both the lane ``calibration_eval.v1`` schema and the
unified ``receipt.v2`` envelope (``receipt_version=2``). All output is
SYNTHETIC correctness evidence, never market data.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_fund.metrics.density_forecast import pit_histogram
from quant_fund.metrics.scoring import coverage, pit_values
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent
from quant_fund.research.fleet_eval import (
    COVERAGE_LEVELS,
    DEFAULT_TAUS,
    SHARD_GENERATORS,
    HeadFactory,
    ShardGenerator,
    SyntheticShard,
    _atomic_write_text,
    _central_interval_index,
    resolve_shard_generators,
)
from quant_fund.research.receipt_v2 import build_receipt_v2, seal_receipt
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

Array = NDArray[np.float64]

CALIBRATION_EVAL_SCHEMA = "calibration_eval.v1"
CALIBRATION_EVAL_KIND = "calibration_eval"
DEFAULT_PIT_BINS = 10

# PIT transforms are clamped into the open unit interval the histogram
# requires; ``pit_values`` already bounds extrapolated transforms to 0/1.
_PIT_EPS = 1e-9
# A regressor with less variance than this cannot anchor a slope honestly.
_MZ_VAR_FLOOR = 1e-24


def _as_1d(name: str, x: Array) -> Array:
    out = np.asarray(x, dtype=float).reshape(-1)
    if out.size == 0 or not np.isfinite(out).all():
        raise ValueError(f"{name} must be nonempty and finite")
    return out


def _finite_scalar(name: str, value: float) -> float:
    """Fail-closed scalar: a diagnostic that is not finite is an error."""
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} is not finite")
    return out


def pit_histogram_block(pits: Array, n_bins: int = DEFAULT_PIT_BINS) -> dict[str, Any]:
    """K-bin PIT histogram: shares, chi-square flatness, p-value.

    Boundary transforms (exactly 0 or 1 — realized outcomes outside the
    predicted quantile support) are clipped into ``(eps, 1 - eps)`` and
    their mass reported as ``pit_boundary_frac``. ``pit_histogram`` fails
    closed on fewer than 50 finite transforms or ``bins`` outside [5, 50].
    """
    p = _as_1d("pits", pits)
    boundary_frac = float(np.mean((p <= 0.0) | (p >= 1.0)))
    hist = pit_histogram(np.clip(p, _PIT_EPS, 1.0 - _PIT_EPS), bins=n_bins)
    shares = np.asarray(hist["shares"], dtype=float)
    return {
        "pit_bins": int(n_bins),
        "pit_shares": [float(s) for s in shares],
        "pit_chi2": _finite_scalar("pit_chi2", float(np.asarray(hist["chi2"]))),
        "pit_chi2_p": _finite_scalar("pit_chi2_p", float(np.asarray(hist["pvalue"]))),
        "pit_boundary_frac": boundary_frac,
    }


def quantile_hit_rates(y: Array, quantiles: Array, taus: Array) -> Array:
    """Empirical hit rate ``h_j = P(y <= q_j)`` per grid level."""
    yy = _as_1d("y", y)
    q = np.asarray(quantiles, dtype=float)
    t = _as_1d("taus", taus)
    if q.ndim != 2 or q.shape[0] != yy.size or q.shape[1] != t.size:
        raise ValueError("quantiles must be (n_y, n_taus) matching y and taus")
    if not np.isfinite(q).all():
        raise ValueError("quantiles must be finite")
    return np.asarray((yy[:, None] <= q).mean(axis=0), dtype=np.float64)


def reliability_regression(hit_rates: Array, taus: Array) -> tuple[float, float]:
    """OLS of empirical hit rates on nominal quantile coordinates.

    Returns ``(slope, intercept)``; a calibrated forecaster gives
    ``(1.0, 0.0)``. The quantile analog of Cox's (1958) calibration slope —
    the reliability-diagram regression of Gneiting, Balabdaoui & Raftery
    (2007) in the Koenker & Machado (1999) goodness-of-fit frame. Needs at
    least two distinct grid levels to anchor a slope.
    """
    h = _as_1d("hit_rates", hit_rates)
    t = _as_1d("taus", taus)
    if h.size != t.size:
        raise ValueError("hit_rates and taus must have the same length")
    if t.size < 2:
        raise ValueError("reliability regression needs at least two tau levels")
    if np.any((t <= 0.0) | (t >= 1.0)) or np.any(np.diff(t) <= 0.0):
        raise ValueError("taus must be strictly increasing inside (0, 1)")
    if np.any((h < 0.0) | (h > 1.0)):
        raise ValueError("hit_rates must lie in [0, 1]")
    slope, intercept = np.polyfit(t, h, 1)
    return _finite_scalar("calibration_slope", float(slope)), _finite_scalar(
        "calibration_intercept", float(intercept)
    )


def median_mz_regression(y: Array, q_median: Array) -> tuple[float, float] | None:
    """Mincer–Zarnowitz (1969) OLS of realized outcome on predicted median.

    Slope 1 under a correctly located conditional-median forecast. Returns
    ``None`` when the predicted coordinate has essentially no variation —
    a constant regressor carries no calibration information (the usual
    case for unconditional heads on unconditional shards).
    """
    yy = _as_1d("y", y)
    q = _as_1d("q_median", q_median)
    if yy.size != q.size:
        raise ValueError("y and q_median must have the same length")
    if q.size < 2 or float(np.var(q)) <= _MZ_VAR_FLOOR:
        return None
    slope, intercept = np.polyfit(q, yy, 1)
    return _finite_scalar("mz_slope", float(slope)), _finite_scalar(
        "mz_intercept", float(intercept)
    )


def _hit_rate_key(tau: float) -> str:
    return f"hit_rate_{tau:g}"


def _coverage_key(level: float) -> str:
    return f"coverage_{int(round(level * 100))}"


def _median_tau_index(taus: Array) -> int:
    return int(np.argmin(np.abs(taus - 0.5)))


def _score_cell(
    shard: SyntheticShard,
    shard_seed: int,
    name: str,
    factory: HeadFactory,
    n_train: int,
    n_eval: int,
    taus: Array,
    coverage_index: dict[float, tuple[int, int] | None],
    pit_bins: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "shard": shard.name,
        "model": name,
        "family": "distribution",
        "status": "ok",
        "error": None,
        "n_train": n_train,
        "n_eval": n_eval,
        "seed": shard_seed,
        "pit_bins": int(pit_bins),
        "pit_shares": None,
        "pit_chi2": None,
        "pit_chi2_p": None,
        "pit_boundary_frac": None,
        **{_coverage_key(level): None for level in COVERAGE_LEVELS},
        "calibration_slope": None,
        "calibration_intercept": None,
        "mz_slope": None,
        "mz_intercept": None,
        **{_hit_rate_key(float(t)): None for t in taus},
    }
    try:
        model = factory()
        model.fit(shard.x[:n_train], shard.y[:n_train])
        meta = model.metadata()
        row["family"] = str(meta.family)
        row["model_version"] = str(getattr(meta, "version", "unknown") or "unknown")
        row["model_head"] = str(getattr(meta, "name", "") or "")
        if getattr(model, "fleet_lagged_predict", False):
            # Same origin protocol as fleet_eval: row i's features are the
            # observed lag-1 value y[n_train+i-1] — no lookahead.
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
        y_eval = shard.y[n_train : n_train + n_eval]
        pits = pit_values(y_eval, q, taus)
        if not np.isfinite(pits).all():
            raise ValueError("missing PIT support for one or more eval rows")
        block = pit_histogram_block(pits, pit_bins)
        row.update(block)
        # The chi-square p-value assumes independent PIT draws; serially
        # dependent shards keep the statistic but suppress the p-value
        # (the fleet_eval pit_ks_p convention).
        if shard.config.get("serial_dependence"):
            row["pit_chi2_p"] = None
        for level, pair in coverage_index.items():
            if pair is not None:
                cov = coverage(y_eval, q[:, pair[0]], q[:, pair[1]])
                row[_coverage_key(level)] = _finite_scalar("coverage", cov)
        hits = quantile_hit_rates(y_eval, q, taus)
        slope, intercept = reliability_regression(hits, taus)
        row["calibration_slope"] = slope
        row["calibration_intercept"] = intercept
        mz = median_mz_regression(y_eval, q[:, _median_tau_index(taus)])
        if mz is not None:
            row["mz_slope"], row["mz_intercept"] = mz
        for j, tau in enumerate(taus):
            row[_hit_rate_key(float(tau))] = float(hits[j])
    except Exception as exc:
        row["status"] = "error"
        row["error"] = str(exc)
    return row


def run_calibration_eval(
    factories: Mapping[str, HeadFactory],
    shards: Iterable[str] | Mapping[str, ShardGenerator] | None = None,
    n_train: int = 512,
    n_eval: int = 256,
    *,
    seed: int = 0,
    taus: Sequence[float] = DEFAULT_TAUS,
    pit_bins: int = DEFAULT_PIT_BINS,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Score every (head, shard) cell's forecast calibration, not its accuracy.

    Mirrors ``run_distribution_fleet``: identical seeded shards, identical
    leading/trailing origin split, identical lagged-predict convention.
    Diagnostics are calibration-only (PIT histogram, central-interval
    coverage, reliability and Mincer–Zarnowitz slopes) — never headline
    P&L metrics. Returns the results frame plus the unsealed
    ``calibration_eval.v1`` receipt payload.
    """
    if not isinstance(factories, Mapping) or not factories:
        raise ValueError("calibration eval requires a nonempty mapping of head factories")
    if (
        isinstance(n_train, bool)
        or not isinstance(n_train, (int, np.integer))
        or n_train < 1
        or isinstance(n_eval, bool)
        or not isinstance(n_eval, (int, np.integer))
        or n_eval < 1
    ):
        raise ValueError("n_train and n_eval must be positive")
    n_train = int(n_train)
    n_eval = int(n_eval)
    if isinstance(pit_bins, bool) or not isinstance(pit_bins, (int, np.integer)):
        raise ValueError("pit_bins must be an integer")
    pit_bins = int(pit_bins)
    tau_arr = np.asarray(list(taus), dtype=float)
    if (
        tau_arr.size < 2
        or not np.isfinite(tau_arr).all()
        or np.any((tau_arr <= 0.0) | (tau_arr >= 1.0))
        or np.any(np.diff(tau_arr) <= 0.0)
    ):
        raise ValueError("taus must be at least two strictly increasing levels inside (0, 1)")
    resolved: Mapping[str, ShardGenerator]
    if shards is None:
        resolved = SHARD_GENERATORS
    elif isinstance(shards, Mapping):
        resolved = shards
    else:
        resolved = resolve_shard_generators(shards)
    if not resolved:
        raise ValueError("calibration eval requires at least one shard")
    for name in resolved:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("shard names must be nonempty strings")

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
        for model_name, factory in factories.items():
            rows.append(
                _score_cell(
                    shard,
                    shard_seed,
                    model_name,
                    factory,
                    n_train,
                    n_eval,
                    tau_arr,
                    coverage_index,
                    pit_bins,
                )
            )

    model_versions: dict[str, dict[str, str]] = {}
    for row in rows:
        version = row.pop("model_version", None)
        head = row.pop("model_head", None)
        model_versions.setdefault(
            str(row["model"]),
            {
                "head": str(head) if head else str(row["model"]),
                "version": str(version) if version else "unknown",
            },
        )
    for model_name, blob in model_versions.items():
        if blob["version"] == "unknown":
            try:
                meta = factories[model_name]().metadata()
                head = getattr(meta, "name", None)
                version = getattr(meta, "version", None)
                if head:
                    blob["head"] = str(head)
                if version:
                    blob["version"] = str(version)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass

    columns = [
        "shard",
        "model",
        "family",
        "status",
        "error",
        "n_train",
        "n_eval",
        "seed",
        "pit_bins",
        "pit_shares",
        "pit_chi2",
        "pit_chi2_p",
        "pit_boundary_frac",
        *[_coverage_key(level) for level in COVERAGE_LEVELS],
        "calibration_slope",
        "calibration_intercept",
        "mz_slope",
        "mz_intercept",
        *[_hit_rate_key(float(t)) for t in tau_arr],
    ]
    schema: dict[str, Any] = {
        **{key: pl.String for key in ("shard", "model", "family", "status", "error")},
        **{key: pl.Int64 for key in ("n_train", "n_eval", "seed", "pit_bins")},
        "pit_shares": pl.List(pl.Float64),
        **{
            key: pl.Float64
            for key in columns
            if key
            not in {
                "shard",
                "model",
                "family",
                "status",
                "error",
                "n_train",
                "n_eval",
                "seed",
                "pit_bins",
                "pit_shares",
            }
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
                "models": sorted(str(k) for k in factories),
                "model_versions": model_versions,
                "taus": [float(t) for t in tau_arr],
                "coverage_levels": [float(level) for level in COVERAGE_LEVELS],
                "pit_bins": int(pit_bins),
                "n_train": n_train,
                "n_eval": n_eval,
                "seed": int(seed),
            }
        )
    )
    receipt: dict[str, Any] = {
        "schema": CALIBRATION_EVAL_SCHEMA,
        "kind": CALIBRATION_EVAL_KIND,
        "data_label": "SYNTHETIC",
        "live_pnl_claim": False,
        "generated_at": datetime.now(UTC).isoformat(),
        "git_revision": git_revision(),
        "seed": int(seed),
        "n_train": n_train,
        "n_eval": n_eval,
        "taus": [float(t) for t in tau_arr],
        "coverage_levels": [float(level) for level in COVERAGE_LEVELS],
        "pit_bins": int(pit_bins),
        "models": sorted(str(k) for k in factories),
        "model_versions": model_versions,
        "shards": shard_meta,
        "inputs_sha256": inputs_sha256,
        "n_rows": len(rows),
        "n_error_rows": sum(1 for row in rows if row["status"] != "ok"),
        "results": rows,
    }
    return frame, receipt


def calibration_contract_errors(receipt: Mapping[str, Any]) -> list[str]:
    """Fail-closed contract for a ``calibration_eval.v1`` payload (writer + verifier)."""
    research_blob = {key: value for key, value in receipt.items() if key != "live_pnl_claim"}
    errors: list[str] = []
    if receipt.get("schema") != CALIBRATION_EVAL_SCHEMA:
        errors.append("schema_not_calibration_eval_v1")
    if receipt.get("kind") != CALIBRATION_EVAL_KIND:
        errors.append("kind_not_calibration_eval")
    if receipt.get("data_label") != "SYNTHETIC":
        errors.append("data_label_not_synthetic")
    if receipt.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim_not_false")
    if not isinstance(receipt.get("results"), list) or not receipt["results"]:
        errors.append("results_missing_or_empty")
    if not family_blob_forbidden_metrics_absent(research_blob):
        errors.append("forbidden_metric_keys")
    results = receipt.get("results")
    if isinstance(results, list):
        if receipt.get("n_rows") is not None and receipt.get("n_rows") != len(results):
            errors.append("n_rows_mismatch")
        error_rows = sum(
            1 for row in results if not isinstance(row, Mapping) or row.get("status") != "ok"
        )
        if receipt.get("n_error_rows") is not None and receipt.get("n_error_rows") != error_rows:
            errors.append("n_error_rows_mismatch")
    return errors


def calibration_dataset_identity(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """The dataset identity bound by a v2 ``dataset_hash``: shard content hashes."""
    shards = receipt.get("shards")
    if not isinstance(shards, Mapping):
        raise ValueError("calibration receipt has no shards block")
    dataset: dict[str, Any] = {}
    for name, meta in shards.items():
        if not isinstance(meta, Mapping):
            raise ValueError(f"calibration shard {name!r} metadata is not an object")
        dataset[str(name)] = {
            "x_sha256": meta.get("x_sha256"),
            "y_sha256": meta.get("y_sha256"),
        }
    return dataset


def calibration_params(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """The run parameters bound by a v2 ``params_hash``."""
    shards = receipt.get("shards")
    return {
        "models": receipt.get("models"),
        "taus": receipt.get("taus"),
        "coverage_levels": receipt.get("coverage_levels"),
        "pit_bins": receipt.get("pit_bins"),
        "n_train": receipt.get("n_train"),
        "n_eval": receipt.get("n_eval"),
        "seed": receipt.get("seed"),
        "shards": sorted(str(name) for name in shards) if isinstance(shards, Mapping) else None,
    }


def calibration_verdict(receipt: Mapping[str, Any]) -> str:
    """pass iff every calibration cell scored without error; an error is a fail."""
    n_error_rows = receipt.get("n_error_rows")
    return "pass" if n_error_rows == 0 else "fail"


def calibration_receipt_v2(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Wrap a ``calibration_eval.v1`` payload in the unified ``receipt.v2`` envelope.

    The v1 payload is embedded verbatim under ``payload``; the envelope binds
    the shard data digests, run params, this module's source hash, and the
    loaded numeric stack. Validates the v1 contract first — a malformed v1
    receipt is never wrapped.
    """
    if calibration_contract_errors(receipt):
        raise ValueError("calibration receipt violates its synthetic research contract")
    return build_receipt_v2(
        kind=str(receipt["kind"]),
        data_label=str(receipt["data_label"]),
        dataset=calibration_dataset_identity(receipt),
        params=calibration_params(receipt),
        code_files=(Path(__file__),),
        verdict=calibration_verdict(receipt),
        payload=dict(receipt),
        generated_at=str(receipt["generated_at"]),
        revision=str(receipt["git_revision"]),
    )


def calibration_v2_consistency_errors(envelope: Mapping[str, Any]) -> list[str]:
    """Re-derive a calibration receipt.v2 envelope's bound digests from its payload."""
    errors: list[str] = []
    payload = envelope.get("payload")
    if not isinstance(payload, Mapping):
        return ["payload_not_object"]
    contract_errors = calibration_contract_errors(payload)
    errors.extend(f"payload_{name}" for name in contract_errors)
    if contract_errors:
        return errors
    try:
        dataset = calibration_dataset_identity(payload)
    except ValueError as exc:
        return [*errors, f"payload_{exc}"]
    if hash_bytes(canonical_json_bytes(dataset)) != envelope.get("dataset_hash"):
        errors.append("dataset_hash_mismatch")
    if hash_bytes(canonical_json_bytes(calibration_params(payload))) != envelope.get("params_hash"):
        errors.append("params_hash_mismatch")
    if calibration_verdict(payload) != envelope.get("verdict"):
        errors.append("verdict_mismatch")
    return errors


def write_calibration_receipt(
    receipt: Mapping[str, Any],
    receipts_dir: Path | str = Path("receipts"),
    *,
    receipt_version: int = 1,
) -> Path:
    """Seal a calibration receipt and write ``receipts/calibration_eval_<hash>.json``.

    The filename hash is the sha256 of the canonical receipt payload; the
    same digest is embedded as ``receipt_sha256`` (the fleet_eval seal
    convention). The write is atomic and the contract is validated before
    sealing. ``receipt_version=2`` wraps the v1 payload in the unified
    ``receipt.v2`` envelope first.
    """
    if receipt_version == 1:
        if calibration_contract_errors(receipt):
            raise ValueError("calibration receipt violates its synthetic research contract")
        body: Mapping[str, Any] = receipt
    elif receipt_version == 2:
        body = calibration_receipt_v2(receipt)
    else:
        raise ValueError(f"receipt_version must be 1 or 2, got {receipt_version!r}")
    payload = seal_receipt(body)
    path = Path(receipts_dir) / f"calibration_eval_{payload['receipt_sha256'][:16]}.json"
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


__all__ = [
    "CALIBRATION_EVAL_KIND",
    "CALIBRATION_EVAL_SCHEMA",
    "DEFAULT_PIT_BINS",
    "calibration_contract_errors",
    "calibration_dataset_identity",
    "calibration_params",
    "calibration_receipt_v2",
    "calibration_v2_consistency_errors",
    "calibration_verdict",
    "median_mz_regression",
    "pit_histogram_block",
    "quantile_hit_rates",
    "reliability_regression",
    "run_calibration_eval",
    "write_calibration_receipt",
]
