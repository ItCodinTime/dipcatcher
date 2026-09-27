"""Day Wave 121: ranker cache identity is joblib bytes, not mtime."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from quant_fund.config import load_config
from quant_fund.models.ranking import RidgeRanker
from quant_fund.pipeline.forecast import (
    _joblib_artifact_digest,
    _load_ranker_cached,
    clear_forecast_caches,
    forecast_asof,
)
from quant_fund.utils.hashing import hash_file


def _cfg(tmp_path: Path):
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.data.source = "synthetic"
    cfg.fusion.skip_intervals = True
    cfg.fusion.apply_interval_caps = False
    return cfg


def _panel(n_days: int = 12, n_names: int = 6) -> pl.DataFrame:
    start = datetime(2020, 1, 2, 16, 0, 0)
    rows: list[dict[str, object]] = []
    for day in range(n_days):
        stamp = start + timedelta(days=day)
        for name in range(n_names):
            rows.append(
                {
                    "event_time": stamp,
                    "security_id": f"S{name:02d}",
                    "symbol": f"S{name:02d}",
                    "ret_1": 0.001 * name,
                    "vol_20": 0.02,
                    "cs_pct_mom_20": 0.1 * (name + 1),
                }
            )
    return pl.DataFrame(rows)


def _fit_signed_ranker(sign: float) -> RidgeRanker:
    x = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]], dtype=float)
    y = sign * x[:, 0]
    return RidgeRanker(1.0).fit(x, y)


def _save_ranker(tmp_path: Path, *, sign: float) -> Path:
    path = tmp_path / "metadata" / "ranker_ridge.joblib"
    _fit_signed_ranker(sign).save(path)
    return path


def _replace_ranker_preserving_mtime(path: Path, model: RidgeRanker) -> None:
    """Overwrite the artifact while keeping the previous mtime (Wave 121)."""
    stat = path.stat()
    model.save(path)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))


def _rank_scores(state) -> dict[str, float]:
    return {row.security_id: float(row.rank_score["5d"]) for row in state.forecasts}


def test_ranker_cache_missing_artifact_returns_none(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    clear_forecast_caches()
    assert _load_ranker_cached(cfg) is None


def test_ranker_spec_cache_uses_artifact_bytes_not_mtime(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    clear_forecast_caches()
    path = _save_ranker(tmp_path, sign=1.0)
    first = _load_ranker_cached(cfg)
    assert first is not None
    probe = np.array([[0.2], [0.8]], dtype=float)
    first_pred = np.asarray(first.predict(probe), dtype=float)
    digest = _joblib_artifact_digest(path)
    assert digest == hash_file(path)
    mtime_ns = path.stat().st_mtime_ns
    _replace_ranker_preserving_mtime(path, _fit_signed_ranker(-1.0))
    assert path.stat().st_mtime_ns == mtime_ns
    second = _load_ranker_cached(cfg)
    assert second is not None
    second_pred = np.asarray(second.predict(probe), dtype=float)
    assert _joblib_artifact_digest(path) != digest
    assert not np.allclose(first_pred, second_pred, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(first_pred, -second_pred, rtol=1e-6, atol=1e-8)


def test_forecast_asof_cache_misses_when_ranker_bytes_change_at_same_mtime(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    clear_forecast_caches()
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[-1]
    path = _save_ranker(tmp_path, sign=1.0)
    first = forecast_asof(cfg, asof, frame=frame)
    first_scores = _rank_scores(first)
    mtime_ns = path.stat().st_mtime_ns
    _replace_ranker_preserving_mtime(path, _fit_signed_ranker(-1.0))
    assert path.stat().st_mtime_ns == mtime_ns
    second = forecast_asof(cfg, asof, frame=frame)
    second_scores = _rank_scores(second)
    assert first_scores.keys() == second_scores.keys()
    assert first_scores != second_scores
    clear_forecast_caches()
    fresh = forecast_asof(cfg, asof, frame=frame)
    assert _rank_scores(fresh) == second_scores
