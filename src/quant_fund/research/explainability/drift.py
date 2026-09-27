"""Feature-attribution drift over time.

The evaluation window is split into ``n_blocks`` contiguous, time-ordered
blocks (rows sorted by ``times`` ascending; row order kept when ``times`` is
None). Permutation importance is computed per block — the model is fixed, so
differences reflect shifting feature relevance in the data, not refits.

Drift statistics per consecutive block pair:

- **Spearman ρ** of the raw importance vectors (rank agreement; low/negative
  means the ranking changed),
- **Jensen–Shannon divergence** of the ε-smoothed normalized importance
  shares (natural log; bounded in ``[0, ln 2]``),
- **top-k overlap** ``|A ∩ B| / k`` of the per-block top features.

A report flags drift when any consecutive Spearman falls below
``spearman_floor``, any JS exceeds ``js_ceiling``, or mean top-k overlap
falls below ``topk_floor``. Defaults are honest heuristics, tunable by the
caller — they are diagnostic flags, never promotion gates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr

from quant_fund.research.explainability.attribution import (
    AttributionResult,
    _validate_xy,
    permutation_attribution,
)
from quant_fund.research.explainability.scoring import PredictFn, ProperScoreSpec

_JS_EPS = 1e-12


@dataclass(frozen=True)
class BlockAttribution:
    """Attribution computed on one contiguous time block."""

    block_index: int
    first_row: int  # position within the (sorted) eval window
    last_row: int
    n_rows: int
    result: AttributionResult


@dataclass(frozen=True)
class AttributionDrift:
    """Drift statistics across time-ordered attribution blocks."""

    n_blocks: int
    blocks: tuple[BlockAttribution, ...]
    consecutive_spearman: tuple[float, ...]
    consecutive_js: tuple[float, ...]
    consecutive_topk_overlap: tuple[float, ...]
    per_feature_max_share_change: dict[str, float]
    min_spearman: float
    max_js: float
    mean_topk_overlap: float
    drift_flagged: bool
    thresholds: dict[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _normalized_shares(importances: NDArray[np.float64]) -> NDArray[np.float64]:
    """ε-smoothed shares of positive importance mass; identical scale → uniform."""
    imp = np.asarray(importances, dtype=float)
    positive = np.clip(imp, 0.0, None)
    shares = positive + _JS_EPS
    total = float(shares.sum())
    if total <= 0.0 or not np.isfinite(total):
        uniform: NDArray[np.float64] = np.full(imp.shape, 1.0 / imp.size)
        return uniform
    return np.asarray(shares / total, dtype=float)


def js_divergence(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    """Jensen–Shannon divergence in nats between two share vectors (≤ ln 2)."""
    pp = _normalized_shares(np.asarray(p, dtype=float))
    qq = _normalized_shares(np.asarray(q, dtype=float))
    m = 0.5 * (pp + qq)
    kl_pm = float(np.sum(pp * np.log(pp / m)))
    kl_qm = float(np.sum(qq * np.log(qq / m)))
    return float(max(0.0, 0.5 * (kl_pm + kl_qm)))


def _topk_names(result: AttributionResult, k: int) -> set[str]:
    return {a.feature for a in result.top(k)}


def _block_ranges(n_rows: int, n_blocks: int) -> list[tuple[int, int]]:
    """Contiguous [start, end) row ranges, balanced by np.array_split."""
    edges = np.array_split(np.arange(n_rows), n_blocks)
    return [(int(block[0]), int(block[-1]) + 1) for block in edges if block.size]


def attribution_drift(
    predict: PredictFn,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    times: NDArray[Any] | None = None,
    feature_names: list[str] | None = None,
    scoring: ProperScoreSpec | str | None = None,
    n_blocks: int = 4,
    n_repeats: int = 5,
    seed: int = 42,
    top_k: int = 5,
    spearman_floor: float = 0.6,
    js_ceiling: float = 0.1,
    topk_floor: float = 0.5,
    max_rows_per_block: int | None = None,
) -> AttributionDrift:
    """Per-block permutation importance plus consecutive-block drift statistics.

    The model (``predict``) is held fixed; blocks vary only in which rows are
    evaluated. Each block gets an independent child RNG from
    ``np.random.SeedSequence(seed).spawn(n_blocks)`` so the whole table is
    deterministic under one seed.
    """
    xx, yy = _validate_xy(x, y)
    n_rows, n_features = xx.shape
    n_blocks = int(n_blocks)
    if n_blocks < 2:
        raise ValueError("n_blocks must be >= 2 for drift statistics")
    if n_rows < n_blocks:
        raise ValueError(f"cannot split {n_rows} rows into {n_blocks} blocks")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if not all(np.isfinite(v) for v in (spearman_floor, js_ceiling, topk_floor)):
        raise ValueError("drift thresholds must be finite")
    if not (-1.0 <= spearman_floor <= 1.0 and 0.0 <= js_ceiling <= np.log(2.0)):
        raise ValueError("drift thresholds are outside their statistic ranges")
    if not 0.0 <= topk_floor <= 1.0:
        raise ValueError("topk_floor must be in [0, 1]")

    if times is not None:
        tt = np.asarray(times)
        if tt.ndim != 1 or tt.shape[0] != n_rows:
            raise ValueError(f"times length {tt.shape[0]} != x rows {n_rows}")
        if tt.dtype.kind in "Mm" and np.isnat(tt).any():
            raise ValueError("times must not contain NaT")
        if tt.dtype.kind in "fi" and not np.isfinite(tt).all():
            raise ValueError("times must be finite")
        order = np.argsort(tt, kind="stable")
    else:
        order = np.arange(n_rows)
    x_sorted = xx[order]
    y_sorted = yy[order]

    names = (
        [f"x{j}" for j in range(n_features)]
        if feature_names is None
        else [str(v) for v in feature_names]
    )
    if len(names) != n_features:
        raise ValueError(f"feature_names length {len(names)} does not match x columns {n_features}")

    spec = scoring  # resolved once inside each block call (same object reuse)
    child_seeds = np.random.SeedSequence(int(seed)).spawn(n_blocks)
    warnings: list[str] = []
    blocks: list[BlockAttribution] = []
    ranges = _block_ranges(n_rows, n_blocks)
    for index, (start, end) in enumerate(ranges):
        result = permutation_attribution(
            predict,
            x_sorted[start:end],
            y_sorted[start:end],
            names,
            scoring=spec,
            n_repeats=n_repeats,
            seed=int(child_seeds[index].generate_state(1)[0]),
            max_rows=max_rows_per_block,
        )
        blocks.append(
            BlockAttribution(
                block_index=index,
                first_row=start,
                last_row=end,
                n_rows=end - start,
                result=result,
            )
        )

    vectors = [block.result.importance_vector(names) for block in blocks]
    spearman_vals: list[float] = []
    js_vals: list[float] = []
    overlap_vals: list[float] = []
    k_eff = min(int(top_k), n_features)
    for a, b in zip(blocks[:-1], blocks[1:], strict=True):
        va = a.result.importance_vector(names)
        vb = b.result.importance_vector(names)
        if np.all(va == va[0]) or np.all(vb == vb[0]):
            rho = float("nan")
            warnings.append(
                f"constant importance vector in block {a.block_index} or {b.block_index}"
            )
        else:
            rho = float(spearmanr(va, vb).statistic)
        spearman_vals.append(rho)
        js_vals.append(js_divergence(va, vb))
        overlap_vals.append(
            len(_topk_names(a.result, k_eff) & _topk_names(b.result, k_eff)) / float(k_eff)
        )

    shares = [_normalized_shares(v) for v in vectors]
    per_feature_change: dict[str, float] = {}
    for j, name in enumerate(names):
        series = [float(s[j]) for s in shares]
        per_feature_change[name] = float(max(series) - min(series))

    finite_spearman = [v for v in spearman_vals if np.isfinite(v)]
    min_spearman = float(min(finite_spearman)) if finite_spearman else float("nan")
    max_js = float(max(js_vals)) if js_vals else float("nan")
    mean_overlap = float(np.mean(overlap_vals)) if overlap_vals else float("nan")
    flagged = (
        (np.isfinite(min_spearman) and min_spearman < float(spearman_floor))
        or (np.isfinite(max_js) and max_js > float(js_ceiling))
        or (np.isfinite(mean_overlap) and mean_overlap < float(topk_floor))
    )
    return AttributionDrift(
        n_blocks=len(blocks),
        blocks=tuple(blocks),
        consecutive_spearman=tuple(spearman_vals),
        consecutive_js=tuple(js_vals),
        consecutive_topk_overlap=tuple(overlap_vals),
        per_feature_max_share_change=per_feature_change,
        min_spearman=min_spearman,
        max_js=max_js,
        mean_topk_overlap=mean_overlap,
        drift_flagged=bool(flagged),
        thresholds={
            "spearman_floor": float(spearman_floor),
            "js_ceiling": float(js_ceiling),
            "topk_floor": float(topk_floor),
            "top_k": float(k_eff),
        },
        warnings=tuple(warnings),
    )
