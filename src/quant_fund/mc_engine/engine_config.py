"""Run configuration and the checkpoint fingerprint.

The fingerprint covers every input that changes scenario numbers. Worker count,
backend, the checkpoint directory, and the early-stop hook are control-plane
settings and are not part of it.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

MC_ENGINE_VERSION = 1

SHOCK_MODES = frozenset({"crude", "antithetic", "qmc_sobol", "importance"})
BACKENDS = frozenset({"serial", "process", "ray"})
MEMORY_MODES = frozenset({"exact", "sketch"})


@dataclass(frozen=True)
class EngineConfig:
    """Controls of one scenario-risk run.

    ``importance_shift`` is the mean added to each factor-0 shock. ``None``
    resolves to ``-1/sqrt(n_steps)`` so the squared norm of the shift is 1.
    ``stop_after_new_chunks`` is a control hook for checkpoint tests.
    """

    n_paths: int
    chunk_size: int = 4096
    seed: int = 0
    workers: int = 1
    backend: str = "process"
    shock_mode: str = "crude"
    importance_shift: float | None = None
    qmc_scramble: bool = True
    n_scrambles: int = 1
    control_variate: bool = False
    pilot_every: int = 10
    memory_mode: str = "exact"
    ruin_level: float = 0.5
    es_levels: tuple[float, ...] = (0.975, 0.99)
    ci_level: float = 0.95
    tdigest_compression: float = 100.0
    measure_variance_reduction: bool = True
    evt_threshold: float | None = None
    max_exceedances_per_chunk: int = 100_000
    checkpoint_dir: str | None = None
    stop_after_new_chunks: int | None = None


def resolved_importance_shift(config: EngineConfig, n_steps: int) -> float:
    if config.importance_shift is None:
        return -1.0 / math.sqrt(n_steps)
    return float(config.importance_shift)


def fingerprint_payload(spec: dict[str, Any], config: EngineConfig) -> dict[str, Any]:
    return {
        "version": MC_ENGINE_VERSION,
        "generator": spec,
        "n_paths": config.n_paths,
        "chunk_size": config.chunk_size,
        "seed": config.seed,
        "shock_mode": config.shock_mode,
        "importance_shift": config.importance_shift,
        "qmc_scramble": config.qmc_scramble,
        "n_scrambles": config.n_scrambles,
        "control_variate": config.control_variate,
        "pilot_every": config.pilot_every,
        "memory_mode": config.memory_mode,
        "ruin_level": config.ruin_level,
        "es_levels": list(config.es_levels),
        "ci_level": config.ci_level,
        "tdigest_compression": config.tdigest_compression,
        "measure_variance_reduction": config.measure_variance_reduction,
        "evt_threshold": config.evt_threshold,
        "max_exceedances_per_chunk": config.max_exceedances_per_chunk,
    }


def config_fingerprint(spec: dict[str, Any], config: EngineConfig) -> str:
    raw = json.dumps(
        fingerprint_payload(spec, config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def config_from_payload(payload: dict[str, Any], *, workers: int, backend: str) -> EngineConfig:
    shift = payload["importance_shift"]
    threshold = payload["evt_threshold"]
    return EngineConfig(
        n_paths=int(payload["n_paths"]),
        chunk_size=int(payload["chunk_size"]),
        seed=int(payload["seed"]),
        workers=workers,
        backend=backend,
        shock_mode=str(payload["shock_mode"]),
        importance_shift=None if shift is None else float(shift),
        qmc_scramble=bool(payload["qmc_scramble"]),
        n_scrambles=int(payload["n_scrambles"]),
        control_variate=bool(payload["control_variate"]),
        pilot_every=int(payload["pilot_every"]),
        memory_mode=str(payload["memory_mode"]),
        ruin_level=float(payload["ruin_level"]),
        es_levels=tuple(float(level) for level in payload["es_levels"]),
        ci_level=float(payload["ci_level"]),
        tdigest_compression=float(payload["tdigest_compression"]),
        measure_variance_reduction=bool(payload["measure_variance_reduction"]),
        evt_threshold=None if threshold is None else float(threshold),
        max_exceedances_per_chunk=int(payload["max_exceedances_per_chunk"]),
    )
