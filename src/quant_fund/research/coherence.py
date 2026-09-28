"""Cross-asset distributional coherence: reconciled aggregate forecasts.

Fitting a head on each name gives marginal quantiles; the portfolio-level
question is what distribution the *sum* has. Quantiles do not add — the naive
sum-of-quantiles is only exact under comonotonicity and is badly
over-dispersed under diversification, while a head fit on the aggregate
series alone discards the margin structure. This module reconciles margins
through a Gaussian copula estimated on in-sample residual z-scores
(PIT -> Φ⁻¹ -> correlation), draws correlated uniforms, inverse-CDFs each
margin's quantile grid back to returns, and sums — a seeded Monte Carlo
convolution that is honest about dependence.

Compared methods (each emits an ``(n_eval, n_taus)`` grid on the realized
aggregate):

- ``direct``: empirical quantiles of the train aggregate — no margin info.
- ``naive_sum``: per-name train quantiles summed per τ — comonotone bound.
- ``independent_mc``: margins convolved under independence (no copula).
- ``copula_mc``: margins convolved through the fitted Gaussian copula.

Honesty contract: proper scores only (per-τ pinball, CRPS, coverage, PIT-KS),
every panel is SYNTHETIC and labeled, receipts are sealed receipt.v2
envelopes under ``receipts/``. Correctness evidence, never market claims.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy import stats as sstats

from quant_fund.metrics.probability import pit_ks
from quant_fund.metrics.scoring import (
    coverage,
    mean_pinball,
    pinball_loss,
    pit_values,
    rearrange_quantiles,
)
from quant_fund.research.fleet_eval import DEFAULT_TAUS, _atomic_write_text
from quant_fund.research.receipt_v2 import build_receipt_v2, seal_receipt
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

COHERENCE_SCHEMA = "coherence_eval.v1"
METHODS: tuple[str, ...] = ("direct", "naive_sum", "independent_mc", "copula_mc")
Array = np.ndarray


@dataclass(frozen=True)
class PanelShard:
    """A SYNTHETIC multivariate return panel (n_obs, n_names)."""

    name: str
    y: Array
    config: dict[str, Any]


def _factor_panel(
    name: str,
    n: int,
    seed: int,
    rho: float,
    *,
    df: float | None = None,
    n_names: int = 4,
    sigmas: Sequence[float] | None = None,
) -> PanelShard:
    """One-factor Gaussian (or Student-t) copula panel with per-name scales."""
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    rng = np.random.default_rng(seed)
    f = rng.standard_normal(n)
    e = rng.standard_normal((n, n_names))
    if df is not None:
        f = f / np.sqrt(rng.chisquare(df, n) / df)
        e = e / np.sqrt(rng.chisquare(df, n * n_names).reshape(n, n_names) / df)
    sig = np.asarray(sigmas if sigmas is not None else np.linspace(0.8, 1.4, n_names))
    y = (math.sqrt(rho) * f[:, None] + math.sqrt(1.0 - rho) * e) * sig[None, :]
    return PanelShard(
        name,
        y,
        {"data_label": "SYNTHETIC", "generator": name, "rho": float(rho), "df": df},
    )


def _regime_copula_panel(n: int, seed: int, n_names: int = 4) -> PanelShard:
    """Correlation breaks mid-panel: rho 0.1 -> 0.9 (contagion regime)."""
    half = n // 2
    a = _factor_panel("regime_copula", half, seed, 0.1, n_names=n_names).y
    b = _factor_panel("regime_copula", n - half, seed + 1, 0.9, n_names=n_names).y
    return PanelShard(
        "regime_copula",
        np.vstack([a, b]),
        {"data_label": "SYNTHETIC", "generator": "regime_copula", "rho_break": (0.1, 0.9)},
    )


def _gauss_factor(n: int, seed: int) -> PanelShard:
    return _factor_panel("gauss_factor", n, seed, 0.3)


def _independent_panel(n: int, seed: int) -> PanelShard:
    return _factor_panel("independent", n, seed, 0.0)


def _heavy_tail_factor(n: int, seed: int) -> PanelShard:
    return _factor_panel("heavy_tail_factor", n, seed, 0.5, df=5.0)


PANEL_GENERATORS: dict[str, Any] = {
    "gauss_factor": _gauss_factor,
    "independent": _independent_panel,
    "heavy_tail_factor": _heavy_tail_factor,
    "regime_copula": _regime_copula_panel,
}


def _margin_grids(y_train: Array, taus: Array) -> Array:
    """Per-name empirical quantile rows from the train slice: ``(k, n_taus)``."""
    return np.asarray(np.quantile(y_train, taus, axis=0).T, dtype=float)


def _pit_zscores(y_train: Array, grids: Array, taus: Array) -> Array:
    """In-sample margin PIT -> Φ⁻¹ z-scores, tie-jittered deterministically."""
    n, k = y_train.shape
    z = np.empty((n, k), dtype=float)
    rng = np.random.default_rng(1234567)
    for i in range(k):
        # CDF_i(y) on the discrete grid; ties jittered ±half a grid step so
        # discrete margins don't collapse correlation estimates.
        u = np.interp(y_train[:, i], grids[i], taus, left=0.0, right=1.0)
        u += rng.uniform(-1e-9, 1e-9, n)
        u = np.clip(u, 1e-6, 1.0 - 1e-6)
        z[:, i] = sstats.norm.ppf(u)
    return z


def _nearest_psd_corr(z: Array) -> Array:
    """Sample correlation of z-scores, projected to PSD if needed."""
    r = np.corrcoef(z.T)
    r = np.clip(r, -1.0, 1.0)
    np.fill_diagonal(r, 1.0)
    eigvals, eigvecs = np.linalg.eigh(r)
    if eigvals[0] < 0.0:
        eigvals = np.clip(eigvals, 1e-12, None)
        r = (eigvecs * eigvals[None, :]) @ eigvecs.T
        d = np.sqrt(np.diag(r))
        r = r / d[:, None] / d[None, :]
    return np.asarray(r, dtype=float)


def _mc_aggregate_quantiles(
    grids: Array,
    taus: Array,
    n_eval: int,
    n_mc: int,
    rng: np.random.Generator,
    corr: Array | None,
) -> Array:
    """Inverse-CDF each margin at (possibly correlated) uniforms; sum; quantiles."""
    k = grids.shape[0]
    if corr is None:
        u = rng.uniform(1e-9, 1.0 - 1e-9, (n_mc, k))
    else:
        z = rng.multivariate_normal(np.zeros(k), corr, size=n_mc)
        u = sstats.norm.cdf(z)
    samples = np.zeros((n_mc, k), dtype=float)
    for i in range(k):
        samples[:, i] = np.interp(u[:, i], taus, grids[i])
    total = samples.sum(axis=1)
    q = np.quantile(total, taus)
    return np.tile(q, (n_eval, 1))


def _method_grid(
    method: str,
    y_train: Array,
    n_eval: int,
    taus: Array,
    n_mc: int,
    rng: np.random.Generator,
) -> Array:
    agg_train = y_train.sum(axis=1)
    if method == "direct":
        return np.tile(np.quantile(agg_train, taus), (n_eval, 1))
    grids = _margin_grids(y_train, taus)
    if method == "naive_sum":
        return np.tile(grids.sum(axis=0), (n_eval, 1))
    if method == "independent_mc":
        return _mc_aggregate_quantiles(grids, taus, n_eval, n_mc, rng, None)
    if method == "copula_mc":
        r = _nearest_psd_corr(_pit_zscores(y_train, grids, taus))
        return _mc_aggregate_quantiles(grids, taus, n_eval, n_mc, rng, r)
    raise ValueError(f"method must be one of {METHODS}, got {method!r}")


def _grid_metrics(q: Array, y_eval: Array, taus: Array) -> dict[str, float | None]:
    q = rearrange_quantiles(q)
    y_eval = np.asarray(y_eval, dtype=float)
    dt = np.diff(np.concatenate([[0.0], taus]))
    crps = float(
        np.mean(
            sum(
                2.0 * pinball_loss(y_eval, q[:, j], float(taus[j])) * float(dt[j])
                for j in range(taus.size)
            )
        )
    )
    ks, ks_p = pit_ks(pit_values(y_eval, q, taus))
    row: dict[str, float | None] = {"crps": crps, "pit_ks": ks, "pit_ks_p": ks_p}
    for level, (lo, hi) in ((0.8, (0.1, 0.9)), (0.9, (0.05, 0.95))):
        i_lo = int(np.argmin(np.abs(taus - lo)))
        i_hi = int(np.argmin(np.abs(taus - hi)))
        if abs(taus[i_lo] - lo) < 1e-9 and abs(taus[i_hi] - hi) < 1e-9:
            row[f"coverage_{int(level * 100)}"] = coverage(y_eval, q[:, i_lo], q[:, i_hi])
    for j, tau in enumerate(taus):
        row[f"pinball_{tau:.2f}"] = mean_pinball(y_eval, q[:, j], float(tau))
    return row


def run_coherence(
    panels: Iterable[str] | Mapping[str, Any] | None = None,
    n_train: int = 384,
    n_eval: int = 128,
    *,
    seed: int = 0,
    taus: Sequence[float] = DEFAULT_TAUS,
    n_mc: int = 512,
    methods: Iterable[str] = METHODS,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Score each reconciliation method per panel; return frame + receipt.v2."""
    tau_arr = np.asarray(list(taus), dtype=float)
    if (
        tau_arr.size == 0
        or not np.isfinite(tau_arr).all()
        or np.any((tau_arr <= 0.0) | (tau_arr >= 1.0))
        or np.any(np.diff(tau_arr) <= 0.0)
    ):
        raise ValueError("taus must be a nonempty strictly increasing grid inside (0, 1)")
    if (
        isinstance(n_train, bool)
        or not isinstance(n_train, (int, np.integer))
        or isinstance(n_eval, bool)
        or not isinstance(n_eval, (int, np.integer))
        or n_train < 8
        or n_eval < 4
    ):
        raise ValueError("n_train must be >= 8 and n_eval >= 4")
    n_train = int(n_train)
    n_eval = int(n_eval)
    if int(n_mc) < 16:
        raise ValueError("n_mc must be >= 16")
    method_list = [str(m) for m in methods]
    bad = [m for m in method_list if m not in METHODS]
    if bad or not method_list:
        raise ValueError(f"unknown methods: {bad!r}")

    resolved: Mapping[str, Any]
    if panels is None:
        resolved = PANEL_GENERATORS
    elif isinstance(panels, Mapping):
        resolved = panels
    else:
        names = [str(s) for s in panels]
        unknown = [s for s in names if s not in PANEL_GENERATORS]
        if unknown:
            raise ValueError(f"unknown panels: {unknown!r}")
        resolved = {s: PANEL_GENERATORS[s] for s in names}
    if not resolved:
        raise ValueError("coherence requires at least one panel")

    n_obs = n_train + n_eval
    rows: list[dict[str, Any]] = []
    panel_meta: dict[str, Any] = {}
    n_error_rows = 0
    for panel_index, (panel_name, generator) in enumerate(resolved.items()):
        panel_seed = int(seed) + panel_index
        panel = generator(n_obs, panel_seed)
        if not isinstance(panel, PanelShard):
            raise ValueError(f"panel {panel_name!r} did not return a PanelShard")
        y = np.asarray(panel.y, dtype=float)
        if panel.name != panel_name or panel.config.get("data_label") != "SYNTHETIC":
            raise ValueError(f"panel {panel_name!r} must match its name and SYNTHETIC label")
        if y.ndim != 2 or y.shape[0] != n_obs or y.shape[1] < 2:
            raise ValueError(
                f"panel {panel_name!r} produced shape {y.shape}; needs ({n_obs}, >=2 names)"
            )
        if not np.isfinite(y).all():
            raise ValueError(f"panel {panel_name!r} produced non-finite data")
        panel_meta[panel_name] = {
            "n": int(y.shape[0]),
            "n_names": int(y.shape[1]),
            "seed": panel_seed,
            "y_sha256": hash_bytes(y.tobytes()),
            "config": panel.config,
        }
        y_train = y[:n_train]
        agg_eval = y[n_train : n_train + n_eval].sum(axis=1)
        for method in method_list:
            row: dict[str, Any] = {
                "panel": panel_name,
                "method": method,
                "status": "ok",
                "error": None,
                "n_train": n_train,
                "n_eval": n_eval,
            }
            try:
                rng = np.random.default_rng(panel_seed * 7919 + hash(method) % 7919)
                q = _method_grid(method, y_train, n_eval, tau_arr, int(n_mc), rng)
                row.update(_grid_metrics(q, agg_eval, tau_arr))
            except Exception as exc:  # recorded, never silent
                row["status"] = "error"
                row["error"] = str(exc)
                n_error_rows += 1
            rows.append(row)

    frame = pl.DataFrame(rows)
    payload: dict[str, Any] = {
        "schema": COHERENCE_SCHEMA,
        "kind": "coherence_eval",
        "data_label": "SYNTHETIC",
        "live_pnl_claim": False,
        "n_train": n_train,
        "n_eval": n_eval,
        "seed": int(seed),
        "taus": [float(t) for t in tau_arr],
        "n_mc": int(n_mc),
        "methods": method_list,
        "panels": panel_meta,
        "results": rows,
        "n_error_rows": n_error_rows,
        "scope_note": (
            "Distributional reconciliation on SYNTHETIC correlated panels: "
            "marginal quantile grids reconciled to the aggregate via "
            "naive-sum, independent-MC, and a Gaussian-copula MC fit on "
            "in-sample residual z-scores. Proper scores only."
        ),
    }
    envelope = build_receipt_v2(
        kind="coherence_eval",
        data_label="SYNTHETIC",
        dataset={
            name: {"y_sha256": m["y_sha256"], "n_names": m["n_names"]}
            for name, m in panel_meta.items()
        },
        params={
            "methods": method_list,
            "taus": [float(t) for t in tau_arr],
            "n_train": n_train,
            "n_eval": n_eval,
            "seed": int(seed),
            "panels": sorted(panel_meta),
            "n_mc": int(n_mc),
        },
        code_files=(Path(__file__),),
        verdict="pass" if n_error_rows == 0 else "fail",
        payload=payload,
    )
    return frame, envelope


def coherence_dataset_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The dataset identity bound by the envelope's ``dataset_hash``."""
    panels = payload.get("panels")
    if not isinstance(panels, Mapping):
        raise ValueError("coherence receipt has no panels block")
    out: dict[str, Any] = {}
    for name, meta in panels.items():
        if not isinstance(meta, Mapping):
            raise ValueError(f"coherence panel {name!r} metadata is not an object")
        out[str(name)] = {"y_sha256": meta.get("y_sha256"), "n_names": meta.get("n_names")}
    return out


def coherence_params(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The run parameters bound by the envelope's ``params_hash``."""
    panels = payload.get("panels")
    return {
        "methods": payload.get("methods"),
        "taus": payload.get("taus"),
        "n_train": payload.get("n_train"),
        "n_eval": payload.get("n_eval"),
        "seed": payload.get("seed"),
        "panels": sorted(str(name) for name in panels) if isinstance(panels, Mapping) else None,
        "n_mc": payload.get("n_mc"),
    }


def coherence_v2_consistency_errors(envelope: Mapping[str, Any]) -> list[str]:
    """Re-derive a coherence receipt.v2 envelope's bound digests from its payload."""
    payload = envelope.get("payload")
    if not isinstance(payload, Mapping):
        return ["payload_not_object"]
    errors = coherence_contract_errors(envelope)
    if errors:
        return errors
    try:
        dataset = coherence_dataset_identity(payload)
    except ValueError as exc:
        return [*errors, f"payload_{exc}"]
    if hash_bytes(canonical_json_bytes(dataset)) != envelope.get("dataset_hash"):
        errors.append("dataset_hash_mismatch")
    if hash_bytes(canonical_json_bytes(coherence_params(payload))) != envelope.get("params_hash"):
        errors.append("params_hash_mismatch")
    return errors


def coherence_contract_errors(receipt: Mapping[str, Any]) -> list[str]:
    """Contract check on the sealed receipt.v2 envelope's payload."""
    errors: list[str] = []
    payload = receipt.get("payload")
    if not isinstance(payload, Mapping):
        return ["payload_not_object"]
    if receipt.get("schema") != "receipt.v2":
        errors.append("schema_not_receipt_v2")
    if payload.get("schema") != COHERENCE_SCHEMA:
        errors.append("payload_schema_mismatch")
    if payload.get("kind") != "coherence_eval":
        errors.append("kind_mismatch")
    if payload.get("data_label") != "SYNTHETIC":
        errors.append("data_label_mismatch")
    if payload.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim")
    if not isinstance(payload.get("results"), list):
        errors.append("results_missing")
    return errors


def write_coherence_receipt(
    receipt: Mapping[str, Any],
    receipts_dir: Path | str = Path("receipts"),
) -> Path:
    """Seal the envelope and write ``receipts/coherence_<hash>.json``."""
    if coherence_contract_errors(receipt):
        raise ValueError("coherence receipt violates its contract")
    payload = seal_receipt(receipt)
    path = Path(receipts_dir) / f"coherence_{payload['receipt_sha256'][:16]}.json"
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def format_coherence_table(frame: pl.DataFrame) -> str:
    """Compact per-(panel, method) summary for CLI output."""
    cols = ("panel", "method", "status", "crps", "pit_ks", "coverage_80", "coverage_90")
    return frame.select([c for c in cols if c in frame.columns]).__str__()
