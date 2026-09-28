"""P3.2 — multi-horizon fleet evaluation on identical origins.

The daily fleet tournament scores one-step-ahead distributions; this lane
scores the same heads at horizons ``h ∈ {1,5,20}`` on the same walk-forward
origins, with the h-step construction stated explicitly per row:

- ``native`` — heads that emit multi-horizon distributions themselves
  (``HStepScaledDistribution`` via its h-block adapter) are scored on their
  own h-step quantiles.
- ``iid_sqrt`` — the universal honest extension for 1-step heads:
  ``q_h(τ) = h·μ̂ + √h·(q_1(τ) − μ̂)`` with ``μ̂`` the trapezoid mean of the
  1-step quantile grid. Accumulates under the iid assumption — declared on
  the row, never hidden.
- ``empirical_ratio`` — causal dispersion scaling: the spread is scaled by
  ``std(h-sums)/std(1-step)`` measured on a trailing ``8h``-bar window at
  each origin (lookback only — no forward data). Under clustering or mean
  reversion the realized h-step dispersion departs from ``√h``; this
  construction adapts where ``iid_sqrt`` cannot.

Scores are proper rules only — per-τ pinball, quantile CRPS, interval
coverage — against the realized h-step forward sum. Receipt kind
``multih_fleet_eval``; SYNTHETIC correctness evidence, never a market claim.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.scoring import crps_from_quantiles, mean_pinball, pit_values
from quant_fund.research.fleet_eval import (
    COVERAGE_LEVELS,
    FLEET_HEAD_REGISTRY,
    SyntheticShard,
    _central_interval_index,
    _coverage_key,
    _pinball_key,
    resolve_shard_generators,
)
from quant_fund.research.receipt_v2 import build_receipt_v2, seal_receipt

MULTIH_FLEET_SCHEMA = "multih_fleet_eval.v1"
MULTIH_FLEET_KIND = "multih_fleet_eval"

Array = NDArray[np.float64]

# Head names that emit real multi-horizon distributions (native blocks).
_NATIVE_MULTIH = {"hstep_t", "hstep_emp"}

# Construction tags.
CONSTRUCTION_NATIVE = "native"
CONSTRUCTION_IID = "iid_sqrt"
CONSTRUCTION_EMPIRICAL = "empirical_ratio"


@dataclass
class MultiHRow:
    shard: str
    model: str
    horizon: int
    construction: str
    status: str
    error: str | None
    n_eval: int
    seed: int
    metrics: dict[str, Any]


def _forward_sums(y: Array, h: int) -> Array:
    """Realized h-step sums aligned to origins: ``out[t] = y[t:t+h].sum()``.

    Matches the fleet's one-step convention (``predict`` at origin ``t`` is
    scored against ``y[t]``): the h-step target is the window starting at
    the origin bar.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    csum = np.concatenate([[0.0], np.cumsum(y)])
    out = np.full(n, np.nan)
    if h <= n:
        out[: n - h + 1] = csum[h:] - csum[: n - h + 1]
    return np.asarray(out, dtype=np.float64)


def _quantile_mean(q: Array, taus: Array) -> Array:
    """Quantile-grid mean: ``E[Y] ≈ ∫₀¹ Q(τ) dτ``.

    Trapezoid over the grid's interior plus endpoint-constant tails
    (``Q(τ) ≈ q_lo`` below ``τ_min``, ``≈ q_hi`` above ``τ_max``) — the
    standard convention when the grid does not reach 0/1.
    """
    interior = np.trapezoid(q, taus, axis=1)
    return np.asarray(
        interior + q[:, 0] * taus[0] + q[:, -1] * (1.0 - taus[-1]),
        dtype=np.float64,
    )


def _extend_iid(q1: Array, taus: Array, h: int) -> Array:
    """iid accumulation: mean scales by h, dispersion by √h."""
    mu = _quantile_mean(q1, taus)[:, None]
    return np.asarray(mu * h + (q1 - mu) * math.sqrt(h), dtype=np.float64)


def _h_step_dispersion_ratio(
    y: Array, origins: NDArray[np.integer[Any]], h: int, lookback: int
) -> Array:
    """Causal ``std(h-sums)/std(1-step)`` over a trailing ``lookback`` window.

    For origin index ``t`` only ``y[:t+1]`` is used: the trailing
    ``lookback`` one-step returns and the ``lookback - h`` overlapping
    h-sums ending within that window. Returns NaN where the window is
    degenerate (too few sums or zero variance) — the row then falls back
    to ``iid_sqrt`` scaling, declared in the row's ``construction``.
    """
    n = origins.size
    ratio = np.full(n, np.nan)
    for i, t in enumerate(origins):
        lo = max(0, t - lookback)
        window = y[lo : t + 1]  # returns up to and including the origin bar
        if window.size < h + 4:
            continue
        s1 = float(np.std(window))
        if not np.isfinite(s1) or s1 <= 0.0:
            continue
        sums = np.asarray(
            [window[k : k + h].sum() for k in range(window.size - h + 1)],
            dtype=float,
        )
        sh = float(np.std(sums))
        if not np.isfinite(sh) or sh <= 0.0:
            continue
        ratio[i] = sh / s1
    return ratio


def _extend_empirical(q1: Array, taus: Array, h: int, ratio: Array) -> Array:
    """Empirical-ratio extension: mean scales by h, dispersion by the causal
    trailing ratio; falls back to √h where the window was degenerate."""
    mu = _quantile_mean(q1, taus)[:, None]
    scale = np.where(np.isfinite(ratio), ratio, math.sqrt(h))[:, None]
    return np.asarray(mu * h + (q1 - mu) * scale, dtype=np.float64)


def _predict_native_h(model: Any, x: Array, h: int, n_taus: int) -> Array | None:
    """Slice the ``h`` block out of a multi-horizon head's predict output."""
    inner = getattr(model, "_inner", None)
    if inner is None or not hasattr(inner, "horizons"):
        return None
    if h not in inner.horizons:
        return None
    block = getattr(model, "block", "student_t")
    h_idx = inner.horizons.index(h)
    lo = (2 * h_idx + (0 if block == "student_t" else 1)) * n_taus
    full = np.asarray(inner.predict(x), dtype=float)
    return np.asarray(full[:, lo : lo + n_taus], dtype=np.float64)


def _score_head_horizon(
    *,
    shard: SyntheticShard,
    name: str,
    model: Any,
    h: int,
    n_train: int,
    n_eval: int,
    taus: Array,
    coverage_index: dict[float, tuple[int, int] | None],
    seed: int,
) -> list[MultiHRow]:
    """Score one head at horizon h on the shard's trailing origins."""
    n = shard.y.size
    max_origin = n - h  # y[t:t+h] must fit
    origins = np.arange(n_train, max_origin + 1)
    if origins.size > n_eval:
        origins = origins[:n_eval]
    if origins.size < 4:
        return []

    # 1-step predictions at each origin — the head's causal information set
    # is the trailing fit plus the origin's features, mirroring _score_row.
    if getattr(model, "fleet_lagged_predict", False):
        x_eval = shard.y[origins - 1].reshape(-1, 1)
    else:
        x_eval = np.asarray(shard.x[origins], dtype=float)
    q1 = np.asarray(model.predict(x_eval), dtype=float)
    if q1.ndim != 2 or q1.shape[0] != origins.size or q1.shape[1] != taus.size:
        raise ValueError(f"predict shape {q1.shape}; expected ({origins.size}, {taus.size})")
    if not np.isfinite(q1).all() or np.any(np.diff(q1, axis=1) < 0.0):
        raise ValueError("predict returned non-finite or crossing quantiles")

    y_target = _forward_sums(shard.y, h)[origins]
    lookback = 8 * h
    rows: list[MultiHRow] = []

    constructions: list[tuple[str, Array]]
    if name in _NATIVE_MULTIH:
        qn = _predict_native_h(model, x_eval, h, taus.size)
        if qn is not None:
            constructions = [(CONSTRUCTION_NATIVE, np.asarray(qn, dtype=float))]
        else:
            constructions = []
        # Always also record the iid extension for comparison.
        constructions.append((CONSTRUCTION_IID, _extend_iid(q1, taus, h)))
    else:
        ratio = _h_step_dispersion_ratio(shard.y, origins, h, lookback)
        constructions = [
            (CONSTRUCTION_IID, _extend_iid(q1, taus, h)),
            (CONSTRUCTION_EMPIRICAL, _extend_empirical(q1, taus, h, ratio)),
        ]

    for tag, qh in constructions:
        if not np.isfinite(qh).all() or np.any(np.diff(qh, axis=1) < -1e-12):
            qh = np.maximum.accumulate(np.asarray(qh, dtype=float), axis=1)
        metrics: dict[str, Any] = {"crps": None, "pit_ks": None}
        metrics["crps"] = crps_from_quantiles(y_target, qh, taus)
        metrics["pit_ks"] = float(
            np.max(
                np.abs(
                    np.sort(pit_values(y_target, qh, taus))
                    - (np.arange(y_target.size) + 0.5) / y_target.size
                )
            )
        )
        for level, key_ij in coverage_index.items():
            key = _coverage_key(level)
            metrics[key] = None
            if key_ij is not None:
                lo_i, hi_i = key_ij
                metrics[key] = float(np.mean((y_target >= qh[:, lo_i]) & (y_target <= qh[:, hi_i])))
        for j, tau in enumerate(taus):
            metrics[_pinball_key(float(tau))] = mean_pinball(y_target, qh[:, j], float(tau))
        rows.append(
            MultiHRow(
                shard=shard.name,
                model=name,
                horizon=h,
                construction=tag,
                status="ok",
                error=None,
                n_eval=int(origins.size),
                seed=seed,
                metrics=metrics,
            )
        )
    return rows


def run_multih_fleet_eval(
    factories: Mapping[str, Any],
    shard_generators: Mapping[str, Any],
    *,
    taus: Sequence[float] = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95),
    horizons: Sequence[int] = (1, 5, 20),
    n_train: int = 200,
    n_eval: int = 40,
    n: int = 400,
    seed: int = 11,
) -> tuple[list[MultiHRow], dict[str, Any]]:
    """Fit each head on the leading slice; score h-step targets on trailing origins."""
    taus_arr = np.asarray([float(t) for t in taus], dtype=np.float64)
    horizons_i = sorted({int(h) for h in horizons})
    if not horizons_i or min(horizons_i) < 1:
        raise ValueError("horizons must be positive integers")
    coverage_index = {level: _central_interval_index(taus_arr, level) for level in COVERAGE_LEVELS}

    rows: list[MultiHRow] = []
    for _shard_name, gen in shard_generators.items():
        shard: SyntheticShard = gen(n=n, seed=seed)
        for head_name, factory in factories.items():
            try:
                model = factory()
                model.fit(shard.x[:n_train], shard.y[:n_train])
                for h in horizons_i:
                    rows.extend(
                        _score_head_horizon(
                            shard=shard,
                            name=head_name,
                            model=model,
                            h=h,
                            n_train=n_train,
                            n_eval=n_eval,
                            taus=taus_arr,
                            coverage_index=coverage_index,
                            seed=seed,
                        )
                    )
            except Exception as exc:  # fail-closed: record, never fake
                for h in horizons_i:
                    rows.append(
                        MultiHRow(
                            shard=shard.name,
                            model=head_name,
                            horizon=h,
                            construction="n/a",
                            status="error",
                            error=f"{type(exc).__name__}: {exc}",
                            n_eval=0,
                            seed=seed,
                            metrics={},
                        )
                    )

    ok_rows = [r for r in rows if r.status == "ok"]
    # Leaderboard: min mean pinball per (shard, horizon).
    leaders: dict[str, str] = {}
    for shard_name in shard_generators:
        for h in horizons_i:
            cand = [r for r in ok_rows if r.shard == shard_name and r.horizon == h]
            if not cand:
                continue
            best = min(
                cand,
                key=lambda r: float(
                    np.nanmean([r.metrics[_pinball_key(float(t))] for t in taus_arr])
                ),
            )
            leaders[f"{shard_name}/h{h}"] = f"{best.model}:{best.construction}"

    receipt = build_receipt_v2(
        kind=MULTIH_FLEET_KIND,
        data_label="SYNTHETIC",
        dataset={
            "shards": sorted(shard_generators),
            "n": int(n),
            "seed": int(seed),
        },
        params={
            "taus": [float(t) for t in taus_arr],
            "horizons": horizons_i,
            "n_train": int(n_train),
            "n_eval_requested": int(n_eval),
        },
        code_files=(Path(__file__),),
        verdict="pass" if ok_rows else "fail",
        payload={
            "schema": MULTIH_FLEET_SCHEMA,
            "n_rows": len(rows),
            "n_ok": len(ok_rows),
            "leaders": leaders,
            "constructions": sorted({r.construction for r in ok_rows}),
            "rows": [
                {
                    "shard": r.shard,
                    "model": r.model,
                    "horizon": r.horizon,
                    "construction": r.construction,
                    "status": r.status,
                    "error": r.error,
                    "n_eval": r.n_eval,
                    **r.metrics,
                }
                for r in rows
            ],
            "live_pnl_claim": False,
        },
    )
    return rows, receipt


def multih_fleet_consistency_errors(body: Mapping[str, Any]) -> list[str]:
    """Re-derive the leaderboard from the sealed rows — catches a receipt
    whose ``leaders`` block was edited without recomputing the scores."""
    payload = body.get("payload")
    if not isinstance(payload, Mapping):
        return []
    rows = payload.get("rows")
    leaders = payload.get("leaders")
    if not isinstance(rows, list) or not isinstance(leaders, dict):
        return []
    pinball_keys = [k for k in rows[0] if k.startswith("pinball_")] if rows else []
    errors: list[str] = []
    recomputed: dict[str, str] = {}
    cells: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or row.get("status") != "ok":
            continue
        cells.setdefault(f"{row.get('shard')}/h{row.get('horizon')}", []).append(row)
    for cell, cell_rows in cells.items():
        best = min(
            cell_rows,
            key=lambda r: float(np.nanmean([float(r.get(k, np.nan)) for k in pinball_keys])),
        )
        recomputed[cell] = f"{best.get('model')}:{best.get('construction')}"
    for cell, winner in recomputed.items():
        if leaders.get(cell) != winner:
            errors.append(f"leader_mismatch:{cell}")
    return errors


def write_multih_receipt(receipt: Mapping[str, Any], out_dir: Path) -> Path:
    """Persist as ``multih_fleet_eval_<sha16>.json``."""
    import json

    sealed = seal_receipt(receipt)
    digest = str(sealed["receipt_sha256"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{MULTIH_FLEET_KIND}_{digest[:16]}.json"
    path.write_text(json.dumps(sealed, indent=2, sort_keys=True) + "\n")
    return path


def multih_factories(
    taus: Sequence[float],
    seed: int,
    names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Fleet factories — same registry as the one-step tournament."""
    from quant_fund.research.fleet_eval import fleet_head_factories

    return fleet_head_factories(taus, seed, names)


__all__ = [
    "CONSTRUCTION_EMPIRICAL",
    "CONSTRUCTION_IID",
    "CONSTRUCTION_NATIVE",
    "MULTIH_FLEET_KIND",
    "MULTIH_FLEET_SCHEMA",
    "MultiHRow",
    "FLEET_HEAD_REGISTRY",
    "multih_factories",
    "resolve_shard_generators",
    "run_multih_fleet_eval",
    "write_multih_receipt",
]
