"""Cross-sectional rank-IC evaluation lane (ULTRAPLAN P3.4).

Seeded multi-asset SYNTHETIC panels with a planted cross-sectional signal
are scored per date by Spearman rank-IC between a challenger transform of
the signal and h-step forward returns; the per-date IC series is then
summarized with a Newey-West mean-IC t-stat via
``metrics.cross_section.date_ic_series``. Reporting stays in proper-score
framing (rank-IC distribution and HAC significance), never P&L.

``write_rankic_receipt`` publishes the sealed JSON evidence under
``receipts/`` (canonical payload, sha256 filename, atomic write). All
output is SYNTHETIC correctness evidence, not market data.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_fund.metrics.cross_section import date_ic_series
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent
from quant_fund.research.fleet_eval import _atomic_write_text
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

Array = NDArray[np.float64]

RANKIC_SCHEMA = "cross_sectional_rankic.v1"

DEFAULT_HORIZONS = (1, 5, 20)
_MIN_ASSETS = 8
_MIN_DATES = 30


@dataclass(frozen=True)
class CrossSectionalPanel:
    """A planted-signal synthetic panel: ``signal[t, i]`` then ``fwd[t, i, h]``.

    ``signal`` is observed at date ``t`` before the horizon-``h`` forward
    return ``fwd[t, i, h]`` is realized — no look-ahead inside a shard.
    """

    dates: NDArray[Any]
    asset_ids: tuple[str, ...]
    signal: Array
    forward: dict[int, Array]
    description: str


PanelGenerator = Callable[[int, int, int, Sequence[int]], CrossSectionalPanel]


def _require_dims(n_assets: int, n_dates: int) -> tuple[int, int]:
    n_assets, n_dates = int(n_assets), int(n_dates)
    if n_assets < _MIN_ASSETS:
        raise ValueError(f"need at least {_MIN_ASSETS} assets per date, got {n_assets}")
    if n_dates < _MIN_DATES + max(DEFAULT_HORIZONS) + 1:
        raise ValueError(
            f"need at least {_MIN_DATES + max(DEFAULT_HORIZONS) + 1} dates, got {n_dates}"
        )
    return n_assets, n_dates


def _dates(n: int) -> NDArray[Any]:
    base = np.datetime64("2020-01-01")
    return base + np.arange(n)


def _forward_returns(
    rng: np.random.Generator,
    n_dates: int,
    n_assets: int,
    horizons: Sequence[int],
    sigma: float = 0.01,
) -> dict[int, Array]:
    """Independent Gaussian daily returns aggregated to h-step forwards."""
    daily = rng.normal(0.0, sigma, size=(n_dates, n_assets))
    out: dict[int, Array] = {}
    for h in horizons:
        h = int(h)
        if h < 1:
            raise ValueError("horizons must be >= 1")
        fwd = np.full((n_dates, n_assets), np.nan)
        cum = np.cumsum(daily, axis=0)
        fwd[h:] = cum[h:] - cum[:-h]
        out[h] = fwd
    return out


def _panel(
    name: str,
    description: str,
    n_dates: int,
    n_assets: int,
    seed: int,
    horizons: Sequence[int],
    beta: float,
    sigma: float,
    map_fn: Callable[[Array], Array] | None = None,
    flip_after: int | None = None,
) -> CrossSectionalPanel:
    rng = np.random.default_rng(seed)
    signal = rng.normal(0.0, 1.0, size=(n_dates, n_assets))
    mapped = signal if map_fn is None else map_fn(signal)
    rng_fwd = np.random.default_rng(seed + 7919)
    forward = _forward_returns(rng_fwd, n_dates, n_assets, horizons, sigma)
    loading = rng.normal(beta, 0.002, size=(n_assets,))
    for h, fwd in forward.items():
        effect = mapped * loading[None, :] * math.sqrt(h)
        if flip_after is not None:
            sign = np.ones(n_dates)
            sign[flip_after:] = -1.0
            effect = effect * sign[:, None]
        forward[h] = fwd + effect
    return CrossSectionalPanel(
        dates=_dates(n_dates),
        asset_ids=tuple(f"S{i:03d}" for i in range(n_assets)),
        signal=signal,
        forward=forward,
        description=f"{name}: {description}",
    )


def linear_signal(
    n_dates: int, n_assets: int, seed: int, horizons: Sequence[int]
) -> CrossSectionalPanel:
    """Rank signal maps linearly to forward returns (positive planted IC)."""
    return _panel(
        "linear_signal",
        "fwd = beta*signal + eps",
        n_dates,
        n_assets,
        seed,
        horizons,
        beta=0.02,
        sigma=0.05,
    )


def monotone_cubic(
    n_dates: int, n_assets: int, seed: int, horizons: Sequence[int]
) -> CrossSectionalPanel:
    """fwd = beta*signal^3 — rank-preserving: high Spearman, weaker Pearson."""
    return _panel(
        "monotone_cubic",
        "fwd = beta*signal^3 + eps",
        n_dates,
        n_assets,
        seed,
        horizons,
        beta=0.004,
        sigma=0.05,
        map_fn=lambda s: s**3,
    )


def pure_noise(
    n_dates: int, n_assets: int, seed: int, horizons: Sequence[int]
) -> CrossSectionalPanel:
    """Signal and returns independent — the zero-IC null."""
    return _panel(
        "pure_noise",
        "fwd independent of signal",
        n_dates,
        n_assets,
        seed,
        horizons,
        beta=0.0,
        sigma=0.05,
    )


def regime_flip(
    n_dates: int, n_assets: int, seed: int, horizons: Sequence[int]
) -> CrossSectionalPanel:
    """Loading sign flips mid-sample: |IC| large, mean IC near zero."""
    return _panel(
        "regime_flip",
        "sign(beta) flips at mid-sample",
        n_dates,
        n_assets,
        seed,
        horizons,
        beta=0.03,
        sigma=0.05,
        flip_after=n_dates // 2,
    )


def weak_signal(
    n_dates: int, n_assets: int, seed: int, horizons: Sequence[int]
) -> CrossSectionalPanel:
    """Small beta — the harness must call it borderline, not oversell."""
    return _panel(
        "weak_signal",
        "fwd = tiny*signal + eps",
        n_dates,
        n_assets,
        seed,
        horizons,
        beta=0.004,
        sigma=0.05,
    )


PANEL_GENERATORS: dict[str, PanelGenerator] = {
    "linear_signal": linear_signal,
    "monotone_cubic": monotone_cubic,
    "pure_noise": pure_noise,
    "regime_flip": regime_flip,
    "weak_signal": weak_signal,
}


def resolve_panels(names: Sequence[str] | None = None) -> dict[str, PanelGenerator]:
    if names is None:
        return dict(PANEL_GENERATORS)
    resolved: dict[str, PanelGenerator] = {}
    for name in names:
        if name not in PANEL_GENERATORS:
            raise ValueError(f"unknown panel {name!r}; have {sorted(PANEL_GENERATORS)}")
        resolved[name] = PANEL_GENERATORS[name]
    return resolved


Challenger = Callable[[Array, np.random.Generator], Array]


def _challenger_identity(signal: Array, rng: np.random.Generator) -> Array:
    return signal


def _challenger_noisy(signal: Array, rng: np.random.Generator) -> Array:
    return signal + rng.normal(0.0, 3.0, size=signal.shape)


def _challenger_lagged(signal: Array, rng: np.random.Generator) -> Array:
    out = np.full(signal.shape, np.nan)
    out[1:] = signal[:-1]
    return out


def _challenger_shuffled(signal: Array, rng: np.random.Generator) -> Array:
    out = np.empty(signal.shape)
    for t in range(signal.shape[0]):
        out[t] = signal[t, rng.permutation(signal.shape[1])]
    return out


def _challenger_inverted(signal: Array, rng: np.random.Generator) -> Array:
    return -signal


CHALLENGERS: dict[str, Challenger] = {
    "identity": _challenger_identity,
    "noisy": _challenger_noisy,
    "lagged": _challenger_lagged,
    "shuffled": _challenger_shuffled,
    "inverted": _challenger_inverted,
}


def _to_flat(score: Array, target: Array) -> tuple[Array, Array, Array]:
    mask = np.isfinite(score) & np.isfinite(target)
    n_dates = score.shape[0]
    return (
        score[mask],
        target[mask],
        np.repeat(np.arange(n_dates), mask.sum(axis=1)),
    )


def run_cross_sectional_bench(
    panels: Mapping[str, PanelGenerator] | None = None,
    challengers: Sequence[str] | None = None,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    n_assets: int = 32,
    n_dates: int = 96,
    seed: int = 11,
    min_names: int = 5,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Score each challenger transform per shard and horizon; return frame + receipt.

    Proper-score framing only: rank-IC distribution and HAC t-stat, never P&L.
    """
    n_assets, n_dates = _require_dims(n_assets, n_dates)
    horizons = tuple(int(h) for h in horizons)
    if not horizons or any(h < 1 for h in horizons):
        raise ValueError("horizons must be a nonempty tuple of ints >= 1")
    resolved_panels = dict(PANEL_GENERATORS) if panels is None else dict(panels)
    challenger_names = list(CHALLENGERS) if challengers is None else list(challengers)
    for name in challenger_names:
        if name not in CHALLENGERS:
            raise ValueError(f"unknown challenger {name!r}; have {sorted(CHALLENGERS)}")

    rows: list[dict[str, Any]] = []
    panel_meta: dict[str, Any] = {}
    for shard_index, (name, generator) in enumerate(resolved_panels.items()):
        shard_seed = int(seed) + 104729 * shard_index
        panel = generator(n_dates, n_assets, shard_seed, horizons)
        panel_meta[name] = {
            "n_dates": n_dates,
            "n_assets": n_assets,
            "seed": shard_seed,
            "signal_sha256": hash_bytes(panel.signal.tobytes()),
            "forward_sha256": {
                str(h): hash_bytes(fwd.tobytes()) for h, fwd in sorted(panel.forward.items())
            },
            "description": panel.description,
        }
        for chal_index, chal_name in enumerate(challenger_names):
            rng = np.random.default_rng(shard_seed + 811 + 37 * chal_index)
            transformed = CHALLENGERS[chal_name](panel.signal, rng)
            for h in horizons:
                fwd = panel.forward[h]
                scores_flat, target_flat, date_idx = _to_flat(transformed, fwd)
                if scores_flat.size == 0:
                    rows.append(
                        {
                            "shard": name,
                            "challenger": chal_name,
                            "horizon": h,
                            "status": "error",
                            "error": "no finite score/target pairs",
                            "n_dates": 0,
                        }
                    )
                    continue
                result = date_ic_series(
                    scores_flat,
                    target_flat,
                    date_idx,
                    min_names=min_names,
                )
                rows.append(
                    {
                        "shard": name,
                        "challenger": chal_name,
                        "horizon": h,
                        "status": "ok",
                        "error": "",
                        "n_dates": result.n_dates,
                        "mean_spearman": result.mean_spearman,
                        "t_spearman": result.t_spearman,
                        "p_spearman": result.p_spearman,
                        "mean_pearson": result.mean_pearson,
                        "t_pearson": result.t_pearson,
                        "icir_pearson": result.icir_pearson,
                        "icir_ann_pearson": result.icir_ann_pearson,
                    }
                )

    columns = [
        "shard",
        "challenger",
        "horizon",
        "status",
        "error",
        "n_dates",
        "mean_spearman",
        "t_spearman",
        "p_spearman",
        "mean_pearson",
        "t_pearson",
        "icir_pearson",
        "icir_ann_pearson",
    ]
    schema = {
        **{k: pl.String for k in ("shard", "challenger", "status", "error")},
        **{k: pl.Int64 for k in ("horizon", "n_dates")},
        **{
            k: pl.Float64
            for k in columns
            if k not in {"shard", "challenger", "status", "error", "horizon", "n_dates"}
        },
    }
    frame = pl.DataFrame(rows, schema=schema, orient="row").select(columns)

    inputs_sha256 = hash_bytes(
        canonical_json_bytes(
            {
                "panels": {
                    name: {
                        "signal_sha256": meta["signal_sha256"],
                        "forward_sha256": meta["forward_sha256"],
                    }
                    for name, meta in panel_meta.items()
                },
                "challengers": challenger_names,
                "horizons": list(horizons),
                "n_assets": n_assets,
                "n_dates": n_dates,
                "seed": int(seed),
            }
        )
    )
    receipt: dict[str, Any] = {
        "schema": RANKIC_SCHEMA,
        "kind": "cross_sectional_rankic_eval",
        "data_label": "SYNTHETIC",
        "live_pnl_claim": False,
        "generated_at": datetime.now(UTC).isoformat(),
        "git_revision": git_revision(),
        "seed": int(seed),
        "n_assets": n_assets,
        "n_dates": n_dates,
        "horizons": list(horizons),
        "challengers": challenger_names,
        "panels": panel_meta,
        "inputs_sha256": inputs_sha256,
        "n_rows": len(rows),
        "n_error_rows": sum(1 for row in rows if row["status"] != "ok"),
        "results": rows,
    }
    return frame, receipt


def write_rankic_receipt(
    receipt: Mapping[str, Any],
    receipts_dir: Path | str = Path("receipts"),
) -> Path:
    """Seal a rank-IC receipt as ``receipts/rankic_eval_<hash>.json`` (atomic)."""
    research_blob = {key: value for key, value in receipt.items() if key != "live_pnl_claim"}
    if (
        receipt.get("schema") != RANKIC_SCHEMA
        or receipt.get("data_label") != "SYNTHETIC"
        or receipt.get("live_pnl_claim") is not False
        or not isinstance(receipt.get("results"), list)
        or not receipt["results"]
        or not family_blob_forbidden_metrics_absent(research_blob)
    ):
        raise ValueError("rank-IC receipt violates its synthetic research contract")
    canonical = json.loads(canonical_json_bytes(dict(receipt)))
    digest = hash_bytes(canonical_json_bytes(canonical))
    payload = {**canonical, "receipt_sha256": digest}
    path = Path(receipts_dir) / f"rankic_eval_{digest[:16]}.json"
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def format_rankic_table(frame: pl.DataFrame) -> str:
    """Compact text table for the CLI echo."""
    ok = frame.filter(pl.col("status") == "ok")
    lines = ["shard | challenger | h | n_dates | mean_rankIC | t_NW | p"]
    for row in ok.iter_rows(named=True):
        lines.append(
            f"{row['shard']} | {row['challenger']} | {row['horizon']} | "
            f"{row['n_dates']} | {row['mean_spearman']:+.4f} | "
            f"{row['t_spearman']:+.2f} | {row['p_spearman']:.3g}"
        )
    return "\n".join(lines)
