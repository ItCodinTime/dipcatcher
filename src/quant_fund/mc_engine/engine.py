"""Parallel scenario runner.

Each chunk is a pure function of its chunk id. Results are merged in chunk-id
order, so the worker count and the completion order are not inputs. Checkpoints
store finished chunks and are safe to resume after a stop.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np

from quant_fund.mc_engine.engine_config import (
    BACKENDS,
    MC_ENGINE_VERSION,
    MEMORY_MODES,
    SHOCK_MODES,
    EngineConfig,
    config_fingerprint,
    config_from_payload,
    fingerprint_payload,
    resolved_importance_shift,
)
from quant_fund.mc_engine.report import (
    build_report,
    incomplete_report,
    scenario_mean,
    scenario_variance,
)
from quant_fund.mc_engine.scenario import ScenarioGenerator
from quant_fund.mc_engine.summary import (
    ChunkSummary,
    MergedSummary,
    build_chunk_summary,
    merge_summaries,
)
from quant_fund.mc_engine.tails import path_risk_stats
from quant_fund.mc_engine.variance import draw_standard_normals
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

ProgressCallback = Callable[[dict[str, float | int]], None]

__all__ = [
    "MC_ENGINE_VERSION",
    "EngineConfig",
    "resume_simulation",
    "run_simulation",
]


@dataclass(frozen=True)
class ChunkTask:
    chunk_id: int
    n_paths: int
    chunk_size: int
    seed: int
    n_steps: int
    n_factors: int
    shock_mode: str
    importance_shift: float
    qmc_scramble: bool
    qmc_seed: int
    memory_mode: str
    ruin_level: float
    es_levels: tuple[float, ...]
    compression: float
    pilot_every: int
    evt_threshold: float | None
    max_exceedances: int
    measure_reference_crude: bool
    accepts_external_shocks: bool
    generator: ScenarioGenerator


def _limit_threads() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def _worker_init() -> None:
    _limit_threads()


def simulate_chunk(task: ChunkTask) -> ChunkSummary:
    """Simulate one contiguous block of path indices. Pure in ``task``."""
    _limit_threads()
    start = task.chunk_id * task.chunk_size
    end = min(start + task.chunk_size, task.n_paths)
    indices = np.arange(start, end, dtype=np.int64)
    if task.shock_mode != "crude" and not task.accepts_external_shocks:
        raise ValueError(
            f"{task.generator.name} does not accept external shocks; only shock_mode='crude' "
            "is supported. Draw Philox numbers inside the generator, keyed by path index."
        )
    shocks = None
    weights = None
    if task.accepts_external_shocks:
        shocks, weights = draw_standard_normals(
            indices,
            task.n_steps,
            task.n_factors,
            seed=task.seed,
            shock_mode=task.shock_mode,
            importance_shift=task.importance_shift,
            qmc_scramble=task.qmc_scramble,
            qmc_seed=task.qmc_seed,
        )
    batch = task.generator.generate(indices, shocks=shocks, seed=task.seed)
    returns = np.asarray(batch.returns, dtype=np.float64)
    if returns.shape != (indices.size, task.n_steps):
        raise ValueError(
            f"generator returned shape {returns.shape}, expected {(indices.size, task.n_steps)}"
        )
    if batch.log_importance_weight is not None:
        extra = np.exp(np.asarray(batch.log_importance_weight, dtype=np.float64))
        weights = extra if weights is None else weights * extra
    stats = path_risk_stats(returns, ruin_level=task.ruin_level)
    crude_loss = None
    if (
        task.measure_reference_crude
        and task.shock_mode == "importance"
        and task.accepts_external_shocks
    ):
        crude_shocks, _unused = draw_standard_normals(
            indices,
            task.n_steps,
            task.n_factors,
            seed=task.seed,
            shock_mode="crude",
            importance_shift=0.0,
            qmc_scramble=False,
            qmc_seed=task.seed,
        )
        crude_batch = task.generator.generate(indices, shocks=crude_shocks, seed=task.seed)
        crude_stats = path_risk_stats(crude_batch.returns, ruin_level=task.ruin_level)
        crude_loss = np.asarray(crude_stats["loss"], dtype=np.float64)
    return build_chunk_summary(
        chunk_id=task.chunk_id,
        indices=indices,
        loss=np.asarray(stats["loss"], dtype=np.float64),
        max_drawdown=np.asarray(stats["max_drawdown"], dtype=np.float64),
        ruined=np.asarray(stats["ruined"], dtype=np.uint8),
        no_drawdown=np.asarray(stats["no_drawdown"], dtype=np.uint8),
        recovered=np.asarray(stats["recovered"], dtype=np.uint8),
        recovery_steps=np.asarray(stats["recovery_steps"], dtype=np.int64),
        control=None if batch.control is None else np.asarray(batch.control, dtype=np.float64),
        control_mean=batch.control_mean,
        weight=None if weights is None else np.asarray(weights, dtype=np.float64),
        crude_loss=crude_loss,
        es_levels=task.es_levels,
        compression=task.compression,
        memory_mode=task.memory_mode,
        pilot_every=task.pilot_every,
        evt_threshold=task.evt_threshold,
        max_exceedances=task.max_exceedances,
    )


def _validate(generator: ScenarioGenerator, config: EngineConfig) -> None:
    if (
        isinstance(config.n_paths, bool)
        or not isinstance(config.n_paths, int)
        or config.n_paths < 1
    ):
        raise ValueError("n_paths must be a positive int")
    if (
        isinstance(config.chunk_size, bool)
        or not isinstance(config.chunk_size, int)
        or config.chunk_size < 1
    ):
        raise ValueError("chunk_size must be a positive int")
    if isinstance(config.seed, bool) or not isinstance(config.seed, int) or config.seed < 0:
        raise ValueError("seed must be a non-negative int")
    if (
        isinstance(config.workers, bool)
        or not isinstance(config.workers, int)
        or config.workers < 1
    ):
        raise ValueError("workers must be a positive int")
    if config.backend not in BACKENDS:
        raise ValueError(f"backend must be one of {sorted(BACKENDS)}")
    if config.shock_mode not in SHOCK_MODES:
        raise ValueError(f"shock_mode must be one of {sorted(SHOCK_MODES)}")
    if config.memory_mode not in MEMORY_MODES:
        raise ValueError(f"memory_mode must be one of {sorted(MEMORY_MODES)}")
    if config.shock_mode == "antithetic" and (
        config.n_paths % 2 != 0 or config.chunk_size % 2 != 0
    ):
        raise ValueError("antithetic mode needs an even n_paths and an even chunk_size")
    if config.shock_mode != "qmc_sobol" and config.n_scrambles != 1:
        raise ValueError("n_scrambles > 1 is only valid for shock_mode='qmc_sobol'")
    if config.n_scrambles < 1:
        raise ValueError("n_scrambles must be >= 1")
    if config.pilot_every < 2:
        raise ValueError("pilot_every must be >= 2")
    if not math.isfinite(config.ruin_level):
        raise ValueError("ruin_level must be finite")
    if not config.es_levels or any(not 0.0 < level < 1.0 for level in config.es_levels):
        raise ValueError("es_levels must be a non-empty tuple of probabilities in (0, 1)")
    if not 0.0 < config.ci_level < 1.0:
        raise ValueError("ci_level must be in (0, 1)")
    if config.tdigest_compression < 10.0:
        raise ValueError("tdigest_compression must be >= 10")
    if config.importance_shift is not None and not math.isfinite(config.importance_shift):
        raise ValueError("importance_shift must be finite")
    if config.evt_threshold is not None and not math.isfinite(config.evt_threshold):
        raise ValueError("evt_threshold must be finite")
    if config.max_exceedances_per_chunk < 1:
        raise ValueError("max_exceedances_per_chunk must be positive")
    if generator.n_steps < 1 or generator.n_factors < 1:
        raise ValueError("generator n_steps and n_factors must be positive")
    if config.shock_mode != "crude" and not generator.accepts_external_shocks:
        raise ValueError(
            "shock modes other than crude require a generator that accepts external shocks"
        )


def _n_chunks(config: EngineConfig) -> int:
    return int(math.ceil(config.n_paths / config.chunk_size))


def _tasks(
    generator: ScenarioGenerator,
    config: EngineConfig,
    pending: list[int],
    *,
    scramble_id: int,
) -> list[ChunkTask]:
    shift = resolved_importance_shift(config, generator.n_steps)
    measure_crude = bool(config.measure_variance_reduction and config.shock_mode == "importance")
    return [
        ChunkTask(
            chunk_id=chunk_id,
            n_paths=config.n_paths,
            chunk_size=config.chunk_size,
            seed=config.seed,
            n_steps=generator.n_steps,
            n_factors=generator.n_factors,
            shock_mode=config.shock_mode,
            importance_shift=shift,
            qmc_scramble=config.qmc_scramble,
            qmc_seed=config.seed + scramble_id,
            memory_mode=config.memory_mode,
            ruin_level=config.ruin_level,
            es_levels=config.es_levels,
            compression=config.tdigest_compression,
            pilot_every=config.pilot_every,
            evt_threshold=config.evt_threshold,
            max_exceedances=config.max_exceedances_per_chunk,
            measure_reference_crude=measure_crude,
            accepts_external_shocks=generator.accepts_external_shocks,
            generator=generator,
        )
        for chunk_id in pending
    ]


def _chunk_file(directory: Path, chunk_id: int) -> Path:
    return directory / f"chunk_{chunk_id:06d}.npz"


def _write_manifest(directory: Path, manifest: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    body = {k: v for k, v in manifest.items() if k != "receipt_sha256"}
    manifest = {**body, "receipt_sha256": hash_bytes(canonical_json_bytes(body))}
    target = directory / "manifest.json"
    tmp = directory / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    tmp.replace(target)


def _load_manifest(directory: Path) -> dict[str, Any] | None:
    path = directory / "manifest.json"
    if not path.is_file():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("checkpoint manifest is not an object")
    seal = loaded.get("receipt_sha256")
    if seal is not None:
        body = {k: v for k, v in loaded.items() if k != "receipt_sha256"}
        if seal != hash_bytes(canonical_json_bytes(body)):
            raise ValueError("checkpoint manifest seal mismatch — refusing to resume")
    return cast(dict[str, Any], loaded)


def _execute(
    tasks: list[ChunkTask],
    *,
    backend: str,
    workers: int,
    on_result: Callable[[ChunkSummary], None],
    stop_after: int | None,
) -> None:
    if not tasks:
        return
    if backend == "serial":
        for produced, task in enumerate(tasks):
            if stop_after is not None and produced >= stop_after:
                return
            on_result(simulate_chunk(task))
        return
    if backend == "ray":
        _execute_ray(tasks, on_result)
        return
    ctx = __import__("multiprocessing").get_context("spawn")
    produced = 0
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_worker_init,
    ) as pool:
        futures = [pool.submit(simulate_chunk, task) for task in tasks]
        for future in as_completed(futures):
            on_result(future.result())
            produced += 1
            if stop_after is not None and produced >= stop_after:
                for pending in futures:
                    pending.cancel()
                return


def _execute_ray(
    tasks: list[ChunkTask],
    on_result: Callable[[ChunkSummary], None],
) -> None:
    try:
        import ray
    except ImportError as exc:
        raise ImportError(
            "Ray backend requested but ray is not installed. "
            "Install ray, or use backend='process' or backend='serial'."
        ) from exc
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False, logging_level="ERROR")
    remote = ray.remote(simulate_chunk)
    pending = [remote.remote(task) for task in tasks]
    while pending:
        done, pending = ray.wait(pending, num_returns=1)
        on_result(ray.get(done[0]))


def _run_chunks(
    generator: ScenarioGenerator,
    config: EngineConfig,
    *,
    scramble_id: int,
    progress: ProgressCallback | None,
    use_checkpoint: bool,
) -> tuple[MergedSummary | None, float, int, int, int]:
    """Returns merged (or None if stopped early), elapsed, workers_used, done, total."""
    directory = Path(config.checkpoint_dir) if config.checkpoint_dir and use_checkpoint else None
    spec = generator.spec_dict()
    fingerprint = config_fingerprint(spec, config)
    total = _n_chunks(config)
    completed: dict[int, ChunkSummary] = {}
    if directory is not None:
        manifest = _load_manifest(directory)
        if manifest is None:
            _write_manifest(
                directory,
                {
                    "mc_engine_version": MC_ENGINE_VERSION,
                    "fingerprint": fingerprint,
                    "payload": fingerprint_payload(spec, config),
                    "generator_spec": spec,
                    "n_chunks": total,
                    "completed_chunks": [],
                },
            )
        else:
            if int(manifest.get("mc_engine_version", -1)) != MC_ENGINE_VERSION:
                raise ValueError("checkpoint was written by a different mc_engine version")
            if manifest.get("fingerprint") != fingerprint:
                raise ValueError("checkpoint fingerprint does not match this generator and config")
            for chunk_id in manifest.get("completed_chunks", []):
                path = _chunk_file(directory, int(chunk_id))
                if not path.is_file():
                    raise ValueError(f"checkpoint is missing {path.name}")
                completed[int(chunk_id)] = ChunkSummary.load(path)
    pending_ids = [chunk_id for chunk_id in range(total) if chunk_id not in completed]
    tasks = _tasks(generator, config, pending_ids, scramble_id=scramble_id)
    backend = config.backend
    workers_used = 1 if backend == "serial" else config.workers
    started = time.perf_counter()
    new_completed = 0

    def on_result(summary: ChunkSummary) -> None:
        nonlocal new_completed
        completed[summary.chunk_id] = summary
        new_completed += 1
        if directory is not None:
            summary.save(_chunk_file(directory, summary.chunk_id))
            done_ids = sorted(completed)
            manifest = _load_manifest(directory)
            if manifest is None:
                raise ValueError("checkpoint manifest disappeared during the run")
            manifest["completed_chunks"] = done_ids
            _write_manifest(directory, manifest)
        if progress is not None:
            elapsed = time.perf_counter() - started
            paths_done = min(config.n_paths, len(completed) * config.chunk_size)
            progress(
                {
                    "chunks_completed": len(completed),
                    "chunks_total": total,
                    "paths_completed": paths_done,
                    "paths_total": config.n_paths,
                    "elapsed_s": elapsed,
                    "paths_per_s_wall_clock": paths_done / elapsed if elapsed > 0.0 else 0.0,
                }
            )

    stop = config.stop_after_new_chunks if use_checkpoint else None
    _execute(tasks, backend=backend, workers=workers_used, on_result=on_result, stop_after=stop)
    elapsed = time.perf_counter() - started
    if len(completed) < total:
        return None, elapsed, workers_used, len(completed), total
    merged = merge_summaries(
        [completed[chunk_id] for chunk_id in range(total)],
        compression=config.tdigest_compression,
    )
    return merged, elapsed, workers_used, total, total


def run_simulation(
    generator: ScenarioGenerator,
    config: EngineConfig,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run the scenario engine and return a JSON-ready report.

    An incomplete checkpoint returns ``status='incomplete'`` and does not
    invent tail numbers from the finished prefix.
    """
    _validate(generator, config)
    merged, elapsed, workers_used, done, total = _run_chunks(
        generator,
        config,
        scramble_id=0,
        progress=progress,
        use_checkpoint=config.checkpoint_dir is not None,
    )
    if merged is None:
        return incomplete_report(
            chunks_completed=done,
            chunks_total=total,
            checkpoint_dir=config.checkpoint_dir,
        )
    qmc_means: list[float] | None = None
    qmc_variance: float | None = None
    replicate_elapsed: float | None = None
    if config.shock_mode == "qmc_sobol" and config.n_scrambles > 1:
        replicate_started = time.perf_counter()
        qmc_means = [scenario_mean(merged, config.shock_mode)]
        replicate_config = replace(config, checkpoint_dir=None, stop_after_new_chunks=None)
        for scramble_id in range(1, config.n_scrambles):
            replicate, _elapsed, _workers, _done, _total = _run_chunks(
                generator,
                replicate_config,
                scramble_id=scramble_id,
                progress=None,
                use_checkpoint=False,
            )
            if replicate is None:
                raise RuntimeError("a scramble replicate stopped early without a checkpoint")
            qmc_means.append(scenario_mean(replicate, config.shock_mode))
        if config.measure_variance_reduction:
            crude_config = replace(
                replicate_config,
                shock_mode="crude",
                n_scrambles=1,
                measure_variance_reduction=False,
            )
            crude, _elapsed, _workers, _done, _total = _run_chunks(
                generator,
                crude_config,
                scramble_id=0,
                progress=None,
                use_checkpoint=False,
            )
            if crude is None:
                raise RuntimeError("crude reference stopped early")
            qmc_variance = scenario_variance(crude, "crude")
        replicate_elapsed = time.perf_counter() - replicate_started
    report_config = replace(
        config,
        importance_shift=resolved_importance_shift(config, generator.n_steps),
    )
    return build_report(
        merged,
        report_config,
        generator_name=generator.name,
        generator_type=str(generator.spec_dict().get("type", generator.name)),
        data_source=str(generator.data_source),
        n_steps=generator.n_steps,
        n_factors=generator.n_factors,
        elapsed_s=elapsed,
        workers_used=workers_used,
        qmc_means=qmc_means,
        qmc_crude_variance=qmc_variance,
        replicate_elapsed_s=replicate_elapsed,
    )


def resume_simulation(
    checkpoint_dir: str | Path,
    generator: ScenarioGenerator,
    *,
    workers: int | None = None,
    backend: str | None = None,
    progress: ProgressCallback | None = None,
    stop_after_new_chunks: int | None = None,
) -> dict[str, Any]:
    """Continue a checkpoint. The generator specification must match the manifest."""
    directory = Path(checkpoint_dir)
    manifest = _load_manifest(directory)
    if manifest is None:
        raise ValueError(f"no checkpoint manifest in {directory}")
    payload = manifest.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload is missing")
    payload_dict = cast(dict[str, Any], payload)
    used_workers = int(workers) if workers is not None else 1
    used_backend = backend if backend is not None else "serial"
    if used_backend not in BACKENDS:
        raise ValueError(f"backend must be one of {sorted(BACKENDS)}")
    config = config_from_payload(payload_dict, workers=used_workers, backend=used_backend)
    config = replace(
        config,
        checkpoint_dir=str(directory),
        stop_after_new_chunks=stop_after_new_chunks,
    )
    fingerprint = config_fingerprint(generator.spec_dict(), config)
    if fingerprint != manifest.get("fingerprint"):
        raise ValueError("generator does not match the checkpoint fingerprint")
    return run_simulation(generator, config, progress=progress)
