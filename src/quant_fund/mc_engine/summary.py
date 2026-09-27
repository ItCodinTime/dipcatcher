"""Per-chunk reduction and the chunk-id-order merge.

Workers return a :class:`ChunkSummary`. The coordinator merges those objects
in chunk-id order. Completion order is not an input to the merge.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from quant_fund.mc_engine.aggregate import TDigest, Welford, merge_welford
from quant_fund.mc_engine.tails import spectral_es, weighted_expected_shortfall
from quant_fund.mc_engine.variance import CrossStats, add_cross, cross_stats

FloatArray = NDArray[np.float64]
WelfordState = tuple[int, float, float]


def _finite_array(values: FloatArray, name: str) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return arr


@dataclass
class ChunkSummary:
    chunk_id: int
    n_paths: int
    retain_samples: bool
    loss: FloatArray | None
    max_drawdown: FloatArray | None
    ruined: NDArray[np.uint8] | None
    no_drawdown: NDArray[np.uint8] | None
    recovered: NDArray[np.uint8] | None
    recovery_steps: NDArray[np.int64] | None
    control: FloatArray | None
    weight: FloatArray | None
    welford_loss: WelfordState
    welford_dd: WelfordState
    td_loss_mean: FloatArray
    td_loss_weight: FloatArray
    td_dd_mean: FloatArray
    td_dd_weight: FloatArray
    td_recovery_mean: FloatArray
    td_recovery_weight: FloatArray
    ruin_count: int
    drawdown_count: int
    recovered_count: int
    recovery_step_sum: float
    sum_weight: float
    sum_weight_sq: float
    sum_weighted_loss: float
    sum_weighted_loss_sq: float
    sum_weighted_ruin: float
    sum_weight_recovered: float
    sum_weight_drawdown: float
    sum_weighted_recovery: float
    sum_weighted_dd: float
    has_control: bool
    control_mean: float | None
    pilot: CrossStats
    eval_cross: CrossStats
    crude_n: int
    crude_sum: float
    crude_sumsq: float
    es_chunk: FloatArray
    exceedances: FloatArray
    evt_truncated: bool

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "chunk_id": self.chunk_id,
            "n_paths": self.n_paths,
            "retain_samples": self.retain_samples,
            "ruin_count": self.ruin_count,
            "drawdown_count": self.drawdown_count,
            "recovered_count": self.recovered_count,
            "recovery_step_sum": self.recovery_step_sum,
            "sum_weight": self.sum_weight,
            "sum_weight_sq": self.sum_weight_sq,
            "sum_weighted_loss": self.sum_weighted_loss,
            "sum_weighted_loss_sq": self.sum_weighted_loss_sq,
            "sum_weighted_ruin": self.sum_weighted_ruin,
            "sum_weight_recovered": self.sum_weight_recovered,
            "sum_weight_drawdown": self.sum_weight_drawdown,
            "sum_weighted_recovery": self.sum_weighted_recovery,
            "sum_weighted_dd": self.sum_weighted_dd,
            "has_control": self.has_control,
            "control_mean": self.control_mean,
            "pilot": list(self.pilot),
            "eval_cross": list(self.eval_cross),
            "crude_n": self.crude_n,
            "crude_sum": self.crude_sum,
            "crude_sumsq": self.crude_sumsq,
            "evt_truncated": self.evt_truncated,
            "welford_loss": list(self.welford_loss),
            "welford_dd": list(self.welford_dd),
            "es_chunk": [float(v) for v in self.es_chunk.tolist()],
        }
        # NumPy appends ``.npz`` unless the name already ends with it.
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(
            tmp,
            meta=np.asarray(json.dumps(meta, sort_keys=True)),
            loss=_or_empty(self.loss),
            max_drawdown=_or_empty(self.max_drawdown),
            ruined=_or_empty_uint8(self.ruined),
            no_drawdown=_or_empty_uint8(self.no_drawdown),
            recovered=_or_empty_uint8(self.recovered),
            recovery_steps=_or_empty_int(self.recovery_steps),
            control=_or_empty(self.control),
            weight=_or_empty(self.weight),
            td_loss_mean=self.td_loss_mean,
            td_loss_weight=self.td_loss_weight,
            td_dd_mean=self.td_dd_mean,
            td_dd_weight=self.td_dd_weight,
            td_recovery_mean=self.td_recovery_mean,
            td_recovery_weight=self.td_recovery_weight,
            exceedances=self.exceedances,
        )
        tmp.replace(path)

    @staticmethod
    def load(path: Path) -> ChunkSummary:
        with np.load(path, allow_pickle=False) as blob:
            meta = json.loads(str(blob["meta"].item()))
            retain = bool(meta["retain_samples"])
            loss = None if not retain else np.asarray(blob["loss"], dtype=np.float64)
            return ChunkSummary(
                chunk_id=int(meta["chunk_id"]),
                n_paths=int(meta["n_paths"]),
                retain_samples=retain,
                loss=loss,
                max_drawdown=None
                if not retain
                else np.asarray(blob["max_drawdown"], dtype=np.float64),
                ruined=None if not retain else np.asarray(blob["ruined"], dtype=np.uint8),
                no_drawdown=None if not retain else np.asarray(blob["no_drawdown"], dtype=np.uint8),
                recovered=None if not retain else np.asarray(blob["recovered"], dtype=np.uint8),
                recovery_steps=None
                if not retain
                else np.asarray(blob["recovery_steps"], dtype=np.int64),
                control=None
                if (not retain or not meta["has_control"])
                else np.asarray(blob["control"], dtype=np.float64),
                weight=_optional_weight(blob["weight"], retain),
                welford_loss=_welford_tuple(meta["welford_loss"]),
                welford_dd=_welford_tuple(meta["welford_dd"]),
                td_loss_mean=np.asarray(blob["td_loss_mean"], dtype=np.float64),
                td_loss_weight=np.asarray(blob["td_loss_weight"], dtype=np.float64),
                td_dd_mean=np.asarray(blob["td_dd_mean"], dtype=np.float64),
                td_dd_weight=np.asarray(blob["td_dd_weight"], dtype=np.float64),
                td_recovery_mean=np.asarray(blob["td_recovery_mean"], dtype=np.float64),
                td_recovery_weight=np.asarray(blob["td_recovery_weight"], dtype=np.float64),
                ruin_count=int(meta["ruin_count"]),
                drawdown_count=int(meta["drawdown_count"]),
                recovered_count=int(meta["recovered_count"]),
                recovery_step_sum=float(meta["recovery_step_sum"]),
                sum_weight=float(meta["sum_weight"]),
                sum_weight_sq=float(meta["sum_weight_sq"]),
                sum_weighted_loss=float(meta["sum_weighted_loss"]),
                sum_weighted_loss_sq=float(meta["sum_weighted_loss_sq"]),
                sum_weighted_ruin=float(meta["sum_weighted_ruin"]),
                sum_weight_recovered=float(meta["sum_weight_recovered"]),
                sum_weight_drawdown=float(meta["sum_weight_drawdown"]),
                sum_weighted_recovery=float(meta["sum_weighted_recovery"]),
                sum_weighted_dd=float(meta["sum_weighted_dd"]),
                has_control=bool(meta["has_control"]),
                control_mean=None if meta["control_mean"] is None else float(meta["control_mean"]),
                pilot=_cross_tuple(meta["pilot"]),
                eval_cross=_cross_tuple(meta["eval_cross"]),
                crude_n=int(meta["crude_n"]),
                crude_sum=float(meta["crude_sum"]),
                crude_sumsq=float(meta["crude_sumsq"]),
                es_chunk=np.asarray(meta["es_chunk"], dtype=np.float64),
                exceedances=np.asarray(blob["exceedances"], dtype=np.float64),
                evt_truncated=bool(meta["evt_truncated"]),
            )


def _or_empty(values: FloatArray | None) -> FloatArray:
    if values is None:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


def _or_empty_uint8(values: NDArray[np.uint8] | None) -> NDArray[np.uint8]:
    if values is None:
        return np.zeros(0, dtype=np.uint8)
    return np.asarray(values, dtype=np.uint8)


def _or_empty_int(values: NDArray[np.int64] | None) -> NDArray[np.int64]:
    if values is None:
        return np.zeros(0, dtype=np.int64)
    return np.asarray(values, dtype=np.int64)


def _optional_weight(values: FloatArray, retain: bool) -> FloatArray | None:
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0 or not retain:
        return None
    return arr


def _meta_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("checkpoint metadata expected an int")
    return value


def _meta_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("checkpoint metadata expected a float")
    return float(value)


def _welford_tuple(values: list[object]) -> WelfordState:
    return (_meta_int(values[0]), _meta_float(values[1]), _meta_float(values[2]))


def _cross_tuple(values: list[object]) -> CrossStats:
    return (
        _meta_int(values[0]),
        _meta_float(values[1]),
        _meta_float(values[2]),
        _meta_float(values[3]),
        _meta_float(values[4]),
        _meta_float(values[5]),
    )


def build_chunk_summary(
    *,
    chunk_id: int,
    indices: NDArray[np.int64],
    loss: FloatArray,
    max_drawdown: FloatArray,
    ruined: NDArray[np.uint8],
    no_drawdown: NDArray[np.uint8],
    recovered: NDArray[np.uint8],
    recovery_steps: NDArray[np.int64],
    control: FloatArray | None,
    control_mean: float | None,
    weight: FloatArray | None,
    crude_loss: FloatArray | None,
    es_levels: tuple[float, ...],
    compression: float,
    memory_mode: str,
    pilot_every: int,
    evt_threshold: float | None,
    max_exceedances: int,
) -> ChunkSummary:
    """Reduce one path block. Sketch mode drops the per-path arrays after the digest."""
    y = _finite_array(loss, "loss")
    dd = _finite_array(max_drawdown, "max_drawdown")
    n = int(y.size)
    if dd.size != n or int(indices.size) != n:
        raise ValueError("path arrays must share the chunk length")
    w = None if weight is None else _finite_array(weight, "weight")
    if w is not None and (w.size != n or np.any(w < 0.0)):
        raise ValueError("weights must be non-negative and aligned with losses")
    ones = np.ones(n, dtype=np.float64)
    ww = ones if w is None else w
    moments = Welford()
    moments.add_all(y)
    dd_moments = Welford()
    dd_moments.add_all(dd)
    loss_digest = TDigest(compression)
    loss_digest.add_all(y)
    dd_digest = TDigest(compression)
    dd_digest.add_all(dd)
    recovered_mask = np.asarray(recovered, dtype=np.uint8).astype(bool)
    recovery = np.asarray(recovery_steps, dtype=np.int64)
    rec_values = recovery[recovered_mask].astype(np.float64)
    rec_digest = TDigest(compression)
    if rec_values.size:
        rec_digest.add_all(rec_values)
    drawdown_mask = ~np.asarray(no_drawdown, dtype=np.uint8).astype(bool)
    ruined_mask = np.asarray(ruined, dtype=np.uint8).astype(bool)
    es_values = np.full(len(es_levels), np.nan, dtype=np.float64)
    if n >= 5:
        for i, level in enumerate(es_levels):
            if w is None:
                es_values[i] = spectral_es(y, level)
            elif float(ww.sum()) > 0.0:
                es_values[i] = weighted_expected_shortfall(y, ww, level)
    pilot_mask = (np.asarray(indices, dtype=np.int64) % int(pilot_every)) == 0
    eval_mask = ~pilot_mask
    if control is None:
        pilot: CrossStats = (0, 0.0, 0.0, 0.0, 0.0, 0.0)
        evaluation: CrossStats = (0, 0.0, 0.0, 0.0, 0.0, 0.0)
        has_control = False
    else:
        x = _finite_array(control, "control")
        if x.size != n:
            raise ValueError("control must align with losses")
        pilot = cross_stats(y[pilot_mask], x[pilot_mask])
        evaluation = cross_stats(y[eval_mask], x[eval_mask])
        has_control = True
    if crude_loss is None:
        crude_n = 0
        crude_sum = 0.0
        crude_sumsq = 0.0
    else:
        crude = _finite_array(crude_loss, "crude_loss")
        crude_n = int(crude.size)
        crude_sum = float(crude.sum())
        crude_sumsq = float(np.square(crude).sum())
    if evt_threshold is None:
        exceedances = np.zeros(0, dtype=np.float64)
        truncated = False
    else:
        above = y[y > float(evt_threshold)]
        truncated = int(above.size) > int(max_exceedances)
        exceedances = above if not truncated else above[: int(max_exceedances)]
    retain = memory_mode == "exact"
    return ChunkSummary(
        chunk_id=chunk_id,
        n_paths=n,
        retain_samples=retain,
        loss=y if retain else None,
        max_drawdown=dd if retain else None,
        ruined=np.asarray(ruined, dtype=np.uint8) if retain else None,
        no_drawdown=np.asarray(no_drawdown, dtype=np.uint8) if retain else None,
        recovered=np.asarray(recovered, dtype=np.uint8) if retain else None,
        recovery_steps=recovery if retain else None,
        control=None if control is None or not retain else _finite_array(control, "control"),
        weight=None if w is None or not retain else w,
        welford_loss=moments.state(),
        welford_dd=dd_moments.state(),
        td_loss_mean=loss_digest.means,
        td_loss_weight=loss_digest.weights,
        td_dd_mean=dd_digest.means,
        td_dd_weight=dd_digest.weights,
        td_recovery_mean=rec_digest.means,
        td_recovery_weight=rec_digest.weights,
        ruin_count=int(ruined_mask.sum()),
        drawdown_count=int(drawdown_mask.sum()),
        recovered_count=int(recovered_mask.sum()),
        recovery_step_sum=float(rec_values.sum()) if rec_values.size else 0.0,
        sum_weight=float(ww.sum()),
        sum_weight_sq=float(np.square(ww).sum()),
        sum_weighted_loss=float(np.sum(ww * y)),
        sum_weighted_loss_sq=float(np.sum(ww * np.square(y))),
        sum_weighted_ruin=float(np.sum(ww[ruined_mask])) if ruined_mask.any() else 0.0,
        sum_weight_recovered=float(np.sum(ww[recovered_mask])) if recovered_mask.any() else 0.0,
        sum_weight_drawdown=float(np.sum(ww[drawdown_mask])) if drawdown_mask.any() else 0.0,
        sum_weighted_recovery=float(np.sum(ww[recovered_mask] * rec_values))
        if rec_values.size
        else 0.0,
        sum_weighted_dd=float(np.sum(ww * dd)),
        has_control=has_control,
        control_mean=control_mean,
        pilot=pilot,
        eval_cross=evaluation,
        crude_n=crude_n,
        crude_sum=crude_sum,
        crude_sumsq=crude_sumsq,
        es_chunk=es_values,
        exceedances=np.asarray(exceedances, dtype=np.float64),
        evt_truncated=truncated,
    )


@dataclass
class MergedSummary:
    chunks: list[ChunkSummary]
    loss: FloatArray | None
    max_drawdown: FloatArray | None
    ruined: NDArray[np.uint8] | None
    no_drawdown: NDArray[np.uint8] | None
    recovered: NDArray[np.uint8] | None
    recovery_steps: NDArray[np.int64] | None
    control: FloatArray | None
    weight: FloatArray | None
    welford_loss: WelfordState
    welford_dd: WelfordState
    loss_digest: TDigest
    dd_digest: TDigest
    recovery_digest: TDigest
    ruin_count: int
    drawdown_count: int
    recovered_count: int
    recovery_step_sum: float
    sum_weight: float
    sum_weight_sq: float
    sum_weighted_loss: float
    sum_weighted_loss_sq: float
    sum_weighted_ruin: float
    sum_weight_recovered: float
    sum_weight_drawdown: float
    sum_weighted_recovery: float
    sum_weighted_dd: float
    has_control: bool
    control_mean: float | None
    pilot: CrossStats
    eval_cross: CrossStats
    crude_n: int
    crude_sum: float
    crude_sumsq: float
    es_chunks: FloatArray
    exceedances: FloatArray
    evt_truncated: bool
    n_paths: int
    retain_samples: bool


def merge_summaries(chunks: list[ChunkSummary], *, compression: float) -> MergedSummary:
    """Fold chunks in chunk-id order. Ids must be exactly ``0 .. n_chunks-1``."""
    if not chunks:
        raise ValueError("no chunks to merge")
    ordered = sorted(chunks, key=lambda chunk: chunk.chunk_id)
    ids = [chunk.chunk_id for chunk in ordered]
    if ids != list(range(len(ordered))):
        raise ValueError("chunk ids must be a complete 0..n_chunks-1 set")
    retain = ordered[0].retain_samples
    if any(chunk.retain_samples != retain for chunk in ordered):
        raise ValueError("chunks disagree on retain_samples")
    has_control = ordered[0].has_control
    control_mean = ordered[0].control_mean
    for chunk in ordered[1:]:
        if chunk.has_control != has_control:
            raise ValueError("chunks disagree on control variate availability")
        if has_control and chunk.control_mean != control_mean:
            raise ValueError("chunks disagree on the control mean")
    welford_loss = ordered[0].welford_loss
    welford_dd = ordered[0].welford_dd
    loss_digest = TDigest(compression, ordered[0].td_loss_mean, ordered[0].td_loss_weight)
    dd_digest = TDigest(compression, ordered[0].td_dd_mean, ordered[0].td_dd_weight)
    recovery_digest = TDigest(
        compression, ordered[0].td_recovery_mean, ordered[0].td_recovery_weight
    )
    pilot = ordered[0].pilot
    evaluation = ordered[0].eval_cross
    ruin_count = ordered[0].ruin_count
    drawdown_count = ordered[0].drawdown_count
    recovered_count = ordered[0].recovered_count
    recovery_step_sum = ordered[0].recovery_step_sum
    sum_weight = ordered[0].sum_weight
    sum_weight_sq = ordered[0].sum_weight_sq
    sum_weighted_loss = ordered[0].sum_weighted_loss
    sum_weighted_loss_sq = ordered[0].sum_weighted_loss_sq
    sum_weighted_ruin = ordered[0].sum_weighted_ruin
    sum_weight_recovered = ordered[0].sum_weight_recovered
    sum_weight_drawdown = ordered[0].sum_weight_drawdown
    sum_weighted_recovery = ordered[0].sum_weighted_recovery
    sum_weighted_dd = ordered[0].sum_weighted_dd
    crude_n = ordered[0].crude_n
    crude_sum = ordered[0].crude_sum
    crude_sumsq = ordered[0].crude_sumsq
    evt_truncated = ordered[0].evt_truncated
    exceedance_parts = [ordered[0].exceedances]
    es_rows = [ordered[0].es_chunk]
    n_paths = ordered[0].n_paths
    for chunk in ordered[1:]:
        welford_loss = merge_welford(welford_loss, chunk.welford_loss)
        welford_dd = merge_welford(welford_dd, chunk.welford_dd)
        loss_digest = loss_digest.merge(
            TDigest(compression, chunk.td_loss_mean, chunk.td_loss_weight)
        )
        dd_digest = dd_digest.merge(TDigest(compression, chunk.td_dd_mean, chunk.td_dd_weight))
        recovery_digest = recovery_digest.merge(
            TDigest(compression, chunk.td_recovery_mean, chunk.td_recovery_weight)
        )
        pilot = add_cross(pilot, chunk.pilot)
        evaluation = add_cross(evaluation, chunk.eval_cross)
        ruin_count += chunk.ruin_count
        drawdown_count += chunk.drawdown_count
        recovered_count += chunk.recovered_count
        recovery_step_sum += chunk.recovery_step_sum
        sum_weight += chunk.sum_weight
        sum_weight_sq += chunk.sum_weight_sq
        sum_weighted_loss += chunk.sum_weighted_loss
        sum_weighted_loss_sq += chunk.sum_weighted_loss_sq
        sum_weighted_ruin += chunk.sum_weighted_ruin
        sum_weight_recovered += chunk.sum_weight_recovered
        sum_weight_drawdown += chunk.sum_weight_drawdown
        sum_weighted_recovery += chunk.sum_weighted_recovery
        sum_weighted_dd += chunk.sum_weighted_dd
        crude_n += chunk.crude_n
        crude_sum += chunk.crude_sum
        crude_sumsq += chunk.crude_sumsq
        evt_truncated = evt_truncated or chunk.evt_truncated
        exceedance_parts.append(chunk.exceedances)
        es_rows.append(chunk.es_chunk)
        n_paths += chunk.n_paths
    if retain:
        loss = _concat_float([chunk.loss for chunk in ordered])
        max_drawdown = _concat_float([chunk.max_drawdown for chunk in ordered])
        ruined = _concat_uint8([chunk.ruined for chunk in ordered])
        no_drawdown = _concat_uint8([chunk.no_drawdown for chunk in ordered])
        recovered = _concat_uint8([chunk.recovered for chunk in ordered])
        recovery_steps = _concat_int([chunk.recovery_steps for chunk in ordered])
        control = _concat_float([chunk.control for chunk in ordered]) if has_control else None
        weight = _concat_optional_weight([chunk.weight for chunk in ordered])
    else:
        loss = None
        max_drawdown = None
        ruined = None
        no_drawdown = None
        recovered = None
        recovery_steps = None
        control = None
        weight = None
    return MergedSummary(
        chunks=ordered,
        loss=loss,
        max_drawdown=max_drawdown,
        ruined=ruined,
        no_drawdown=no_drawdown,
        recovered=recovered,
        recovery_steps=recovery_steps,
        control=control,
        weight=weight,
        welford_loss=welford_loss,
        welford_dd=welford_dd,
        loss_digest=loss_digest,
        dd_digest=dd_digest,
        recovery_digest=recovery_digest,
        ruin_count=ruin_count,
        drawdown_count=drawdown_count,
        recovered_count=recovered_count,
        recovery_step_sum=recovery_step_sum,
        sum_weight=sum_weight,
        sum_weight_sq=sum_weight_sq,
        sum_weighted_loss=sum_weighted_loss,
        sum_weighted_loss_sq=sum_weighted_loss_sq,
        sum_weighted_ruin=sum_weighted_ruin,
        sum_weight_recovered=sum_weight_recovered,
        sum_weight_drawdown=sum_weight_drawdown,
        sum_weighted_recovery=sum_weighted_recovery,
        sum_weighted_dd=sum_weighted_dd,
        has_control=has_control,
        control_mean=control_mean,
        pilot=pilot,
        eval_cross=evaluation,
        crude_n=crude_n,
        crude_sum=crude_sum,
        crude_sumsq=crude_sumsq,
        es_chunks=np.vstack(es_rows),
        exceedances=np.concatenate(exceedance_parts)
        if exceedance_parts
        else np.zeros(0, dtype=np.float64),
        evt_truncated=evt_truncated,
        n_paths=n_paths,
        retain_samples=retain,
    )


def _concat_float(parts: list[FloatArray | None]) -> FloatArray:
    arrays = [np.asarray(part, dtype=np.float64) for part in parts if part is not None]
    if not arrays:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(arrays)


def _concat_uint8(parts: list[NDArray[np.uint8] | None]) -> NDArray[np.uint8]:
    arrays = [np.asarray(part, dtype=np.uint8) for part in parts if part is not None]
    if not arrays:
        return np.zeros(0, dtype=np.uint8)
    return np.concatenate(arrays)


def _concat_int(parts: list[NDArray[np.int64] | None]) -> NDArray[np.int64]:
    arrays = [np.asarray(part, dtype=np.int64) for part in parts if part is not None]
    if not arrays:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate(arrays)


def _concat_optional_weight(parts: list[FloatArray | None]) -> FloatArray | None:
    if all(part is None for part in parts):
        return None
    if any(part is None for part in parts):
        raise ValueError("either every chunk has weights or none do")
    return np.concatenate(
        [np.asarray(part, dtype=np.float64) for part in parts if part is not None]
    )
