"""Wall-clock throughput against worker count.

Every row is a real run on this process. ``measured_speedup_vs_first_row`` is
``min_elapsed(first) / min_elapsed(row)`` and is allowed to be below 1.
Nothing here is a hardware-roofline or a linear-scaling claim.
"""

from __future__ import annotations

import os
import time
from dataclasses import replace
from typing import Any

import numpy as np

from quant_fund.mc_engine.engine import EngineConfig, run_simulation
from quant_fund.mc_engine.scenario import GbmPortfolioGenerator, ScenarioGenerator


def _default_generator(n_steps: int) -> GbmPortfolioGenerator:
    return GbmPortfolioGenerator(
        mu=np.asarray([0.0, 0.0], dtype=np.float64),
        covariance=np.asarray([[0.04, 0.01], [0.01, 0.09]], dtype=np.float64),
        weights=np.asarray([0.5, 0.5], dtype=np.float64),
        n_steps=n_steps,
    )


def scaling_benchmark(
    *,
    n_paths: int,
    n_steps: int,
    worker_counts: list[int],
    repeats: int,
    seed: int = 0,
    chunk_size: int = 4096,
    generator: ScenarioGenerator | None = None,
    backend: str = "process",
) -> dict[str, Any]:
    """Time the same scenario at each worker count.

    The clock wraps :func:`run_simulation`, including pool startup and the
    reduction. ``paths_per_s_using_min_elapsed`` is ``n_paths / min(elapsed)``.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if not worker_counts or any(
        isinstance(count, bool) or not isinstance(count, int) or count < 1
        for count in worker_counts
    ):
        raise ValueError("worker_counts must be positive integers")
    scenario = generator if generator is not None else _default_generator(n_steps)
    if scenario.n_steps != n_steps:
        raise ValueError("generator.n_steps must equal n_steps")
    rows: list[dict[str, Any]] = []
    fingerprints: list[str] = []
    for workers in worker_counts:
        elapsed: list[float] = []
        engine_elapsed: list[float] = []
        fingerprint: str | None = None
        for _rep in range(repeats):
            config = EngineConfig(
                n_paths=n_paths,
                chunk_size=min(chunk_size, n_paths if n_paths % 2 == 0 else n_paths),
                seed=seed,
                workers=workers,
                backend=backend,
                shock_mode="crude",
                memory_mode="exact",
                measure_variance_reduction=False,
                checkpoint_dir=None,
            )
            # Keep the requested chunk size when it is legal. The min() above
            # only fires if a caller passes a chunk larger than the path count
            # together with an odd path count that antithetic would reject;
            # this benchmark is crude mode, so restore the caller's chunk size
            # capped at n_paths.
            config = replace(config, chunk_size=max(1, min(chunk_size, n_paths)))
            started = time.perf_counter()
            report = run_simulation(scenario, config)
            elapsed.append(time.perf_counter() - started)
            engine_elapsed.append(float(report["elapsed_s"]))
            fingerprint = str(report["fingerprint"])
        if fingerprint is None:
            raise RuntimeError("benchmark produced no run")
        fastest = min(elapsed)
        rows.append(
            {
                "workers": workers,
                "backend": backend,
                "repeats": repeats,
                "elapsed_s": elapsed,
                "engine_elapsed_s": engine_elapsed,
                "min_elapsed_s": fastest,
                "max_elapsed_s": max(elapsed),
                "paths_per_s_using_min_elapsed": n_paths / fastest,
                "fingerprint": fingerprint,
            }
        )
        fingerprints.append(fingerprint)
    baseline = float(rows[0]["min_elapsed_s"])
    for row in rows:
        row["measured_speedup_vs_first_row"] = baseline / float(row["min_elapsed_s"])
    return {
        "research_only": True,
        "live_pnl_claim": False,
        "market_evidence": False,
        "data_source": str(scenario.data_source),
        "simulation_claim": "scenario_simulation_not_market_evidence",
        "n_paths": n_paths,
        "n_steps": n_steps,
        "chunk_size": max(1, min(chunk_size, n_paths)),
        "seed": seed,
        "backend": backend,
        "host_cpu_count": os.cpu_count(),
        "fingerprints_identical_across_worker_counts": len(set(fingerprints)) == 1,
        "rows": rows,
        "limitation": (
            "Throughput is wall time on this host, including process startup and "
            "reduction. measured_speedup_vs_first_row is the ratio of those measured "
            "minimum elapsed times. A ratio below 1 means the extra workers were slower. "
            "This is not a claim of linear scaling."
        ),
    }
