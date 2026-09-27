"""Tests for research/benches_extra.py — optional scorecard families."""

from __future__ import annotations

import numpy as np
import polars as pl

from quant_fund.research.benches_extra import (
    bench_complexity,
    bench_roughness,
    bench_serial_randomness,
)
from quant_fund.research.catalog import (
    OPTIONAL_BENCHMARK_FAMILIES,
    family_blob_forbidden_metrics_absent,
    family_blob_has_finite_observation,
)


def _panel(n_assets: int = 6, n_days: int = 300, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for a in range(n_assets):
        # mildly persistent returns so roughness/serial diagnostics are non-trivial
        r = np.zeros(n_days)
        for t in range(1, n_days):
            r[t] = 0.2 * r[t - 1] + 0.01 * rng.standard_normal()
        for t in range(n_days):
            rows.append({"security_id": a, "event_time": t, "ret_1": float(r[t])})
    return pl.DataFrame(rows)


def test_families_registered_as_optional() -> None:
    for fam in ("complexity", "roughness", "serial_randomness"):
        assert fam in OPTIONAL_BENCHMARK_FAMILIES


def test_benches_return_clean_finite_blobs() -> None:
    frame = _panel()
    for bench in (bench_complexity, bench_roughness, bench_serial_randomness):
        blob = bench(frame)
        assert isinstance(blob, dict) and blob
        assert family_blob_has_finite_observation(blob)
        assert family_blob_forbidden_metrics_absent(blob)
        assert all(np.isfinite(v) for v in blob.values())


def test_complexity_entropy_ranges() -> None:
    blob = bench_complexity(_panel())
    assert 0.0 <= blob["permutation_entropy"] <= 1.0
    assert 0.0 <= blob["spectral_entropy"] <= 1.0
    assert blob["n_assets"] == 6.0


def test_empty_frame_returns_empty() -> None:
    empty = pl.DataFrame({"security_id": [], "event_time": [], "ret_1": []})
    assert bench_complexity(empty) == {}
    assert bench_roughness(empty) == {}
    assert bench_serial_randomness(empty) == {}
