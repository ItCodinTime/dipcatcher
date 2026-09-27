"""Session caches must reproduce uncached results and honor monkeypatches."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.pipeline.dataset import ensure_silver
from tests.support.session_cache import bench_stats, data_stats, reset_for_tests


def _nan_equal(left: Any, right: Any) -> bool:
    if isinstance(left, (float, np.floating)) and isinstance(right, (float, np.floating)):
        if math.isnan(float(left)) and math.isnan(float(right)):
            return True
        # arch/BLAS reductions are not bit-stable across calls. A cached
        # receipt is one valid run, within a tight tolerance of a fresh one.
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-12)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_nan_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _nan_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _small_config() -> AppConfig:
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 2
    cfg.northset.use_session_l2 = False
    cfg.northset.sweep_n_permutations = 50
    return cfg


def test_bench_cache_matches_uncached_call() -> None:
    reset_for_tests()
    bars = SyntheticMarketProvider(n_assets=4, n_days=16, seed=1).get_bars()
    cfg = _small_config()
    original = bench_northset.__wrapped__  # type: ignore[attr-defined]
    fresh = original(bars, cfg)
    cached = bench_northset(bars, cfg)
    again = bench_northset(bars, cfg)
    assert _nan_equal(fresh, cached)
    assert _nan_equal(fresh, again)
    assert cached is not again
    assert bench_stats()["hit"] >= 1


def test_bench_cache_bypasses_monkeypatch(monkeypatch: pytest.MonkeyPatch) -> None:
    import quant_fund.microstructure.book_metrics as book_metrics

    bars = SyntheticMarketProvider(n_assets=4, n_days=16, seed=3).get_bars()
    monkeypatch.setattr(book_metrics, "side_notional_finite_rate", lambda rows, **kwargs: 0.0)
    cfg = _small_config()
    cfg.northset.side_notional_finite_floor = 0.5
    with pytest.raises(ValueError, match="side_notional_finite_rate"):
        bench_northset(bars, cfg)


def test_synthetic_silver_cache_is_root_independent(tmp_path: Path) -> None:
    reset_for_tests()
    left_cfg = _small_config()
    left_cfg.data.synthetic_n_assets = 4
    left_cfg.data.synthetic_n_days = 16
    left_cfg.data.root = tmp_path / "a"
    right_cfg = left_cfg.model_copy(deep=True)
    right_cfg.data.root = tmp_path / "b"
    left = ensure_silver(left_cfg)
    right = ensure_silver(right_cfg)
    assert left.equals(right)
    assert (tmp_path / "b" / "silver" / "bars.parquet").is_file()
    assert data_stats()["hit"] >= 1
    left_manifest = json.loads((tmp_path / "a" / "metadata" / "data_manifest.json").read_text())
    right_manifest = json.loads((tmp_path / "b" / "metadata" / "data_manifest.json").read_text())
    assert left_manifest["source"] == "synthetic"
    assert (
        right_manifest["artifacts"]["silver"]["sha256"]
        == left_manifest["artifacts"]["silver"]["sha256"]
    )
    assert right_manifest["artifacts"]["silver"]["path"] == str(
        (tmp_path / "b" / "silver" / "bars.parquet").resolve()
    )


def test_silver_cache_does_not_replace_existing_bars(tmp_path: Path) -> None:
    """A lake that already has silver is read, not overwritten by the snapshot."""
    import polars as pl

    reset_for_tests()
    cfg = _small_config()
    cfg.data.synthetic_n_assets = 4
    cfg.data.synthetic_n_days = 16
    cfg.data.root = tmp_path / "fresh"
    ensure_silver(cfg)
    custom = tmp_path / "custom"
    custom.mkdir()
    (custom / "silver").mkdir()
    (custom / "bronze").mkdir()
    for name in ("bars.parquet", "universe.parquet"):
        source = tmp_path / "fresh" / "silver" / name
        target = custom / "silver" / name
        target.write_bytes(source.read_bytes())
    bars_path = custom / "silver" / "bars.parquet"
    frame = pl.read_parquet(bars_path)
    assert "close" in frame.columns
    edited = frame.with_columns(pl.lit(-123.0).alias("close"))
    edited.write_parquet(bars_path)
    cfg.data.root = custom
    got = ensure_silver(cfg)
    assert got["close"][0] == -123.0
    assert pl.read_parquet(bars_path)["close"][0] == -123.0
