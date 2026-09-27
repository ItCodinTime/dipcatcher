"""Day Wave 112: causal per-security GARCH namespace.

Research/infrastructure only — no live broker / vendor MD / live_pnl_claim.
Name-level forecasts do not replace vol_20 or the date-level market overlay.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quant_fund.config import load_config
from quant_fund.data.lake import Lake
from quant_fund.models.volatility import (
    GARCH_DATE_LEVEL_SCOPE,
    GARCH_SECURITY_LEVEL_SCOPE,
    GARCHVol,
)
from quant_fund.pipeline.forecast import (
    clear_forecast_caches,
    forecast_asof,
    garch_market_forecast_asof,
    garch_name_forecasts_asof,
)
from quant_fund.pipeline.train import _garch_name_return_history, _garch_return_history
from quant_fund.schemas.errors import PointInTimeError

RESEARCH_ONLY = True
LIVE_PNL_CLAIM = False


def _cfg(tmp_path: Path):
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.data.source = "synthetic"
    cfg.fusion.skip_intervals = True
    cfg.fusion.apply_interval_caps = False
    return cfg


def _panel(n_days: int = 80, n_names: int = 2, seed: int = 17) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime(2020, 1, 2, 16, 0, 0)
    rows: list[dict[str, object]] = []
    for day in range(n_days):
        stamp = start + timedelta(days=day)
        for name in range(n_names):
            vol = 0.008 if name == 0 else 0.04
            rows.append(
                {
                    "event_time": stamp,
                    "security_id": f"S{name:02d}",
                    "symbol": f"S{name:02d}",
                    "ret_1": float(rng.normal(0.0, vol)),
                    "vol_20": vol,
                    "cs_pct_mom_20": 0.5 + 0.01 * name,
                    "available_time": stamp,
                }
            )
    return pl.DataFrame(rows)


def _save_garch(
    tmp_path: Path,
    returns: np.ndarray,
    *,
    series_scope: str = GARCH_DATE_LEVEL_SCOPE,
    mean: str = "Constant",
) -> Path:
    path = tmp_path / "metadata" / "vol_garch.joblib"
    GARCHVol(series_scope=series_scope, min_obs=20, mean=mean).fit_returns(returns).save(path)
    return path


def _replace_garch_preserving_mtime(path: Path, model: GARCHVol) -> None:
    stat = path.stat()
    model.save(path)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))


def _write_universe(tmp_path: Path, frame: pl.DataFrame, security_ids: list[str]) -> None:
    membership = (
        frame.filter(pl.col("security_id").is_in(security_ids))
        .select(pl.col("security_id"), pl.col("event_time").alias("asof"))
        .unique()
    )
    Lake(tmp_path).write_parquet(membership, "silver/universe.parquet")


@pytest.fixture(autouse=True)
def _clear_garch_caches() -> None:
    clear_forecast_caches()
    yield
    clear_forecast_caches()


def test_name_history_is_not_the_equal_weight_cross_section() -> None:
    start = datetime(2020, 1, 2, 16, 0, 0)
    frame = pl.DataFrame(
        {
            "event_time": [start, start],
            "security_id": ["A", "B"],
            "ret_1": [0.01, 0.99],
        }
    )
    _dates, pooled = _garch_return_history(frame)
    _name_dates, name_a = _garch_name_return_history(frame, "A")
    assert pooled.tolist() == pytest.approx([0.50])
    assert name_a.tolist() == pytest.approx([0.01])


def test_name_history_duplicate_keys_fail_closed() -> None:
    start = datetime(2020, 1, 2, 16, 0, 0)
    frame = pl.DataFrame(
        {
            "event_time": [start, start],
            "security_id": ["A", "A"],
            "ret_1": [0.01, 0.02],
        }
    )
    with pytest.raises(PointInTimeError, match="duplicate"):
        _garch_name_return_history(frame, "A")
    with pytest.raises(PointInTimeError, match="duplicate"):
        _garch_return_history(frame)


def test_name_history_blank_security_id_fail_closed() -> None:
    start = datetime(2020, 1, 2, 16, 0, 0)
    frame = pl.DataFrame(
        {
            "event_time": [start, start],
            "security_id": ["A", "  "],
            "ret_1": [0.01, 0.02],
        }
    )
    with pytest.raises(PointInTimeError, match="blank security_id"):
        _garch_name_return_history(frame, "A")


def test_name_history_missing_security_id_fail_closed() -> None:
    start = datetime(2020, 1, 2, 16, 0, 0)
    frame = pl.DataFrame({"event_time": [start], "ret_1": [0.01]})
    with pytest.raises(PointInTimeError, match="security_id"):
        _garch_name_return_history(frame, "A")


def test_name_forecasts_differ_across_heterogeneous_names(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[70]
    _save_garch(tmp_path, np.full(80, 0.01))
    names = garch_name_forecasts_asof(cfg, frame, asof)
    market = garch_market_forecast_asof(cfg, frame, asof)
    assert market is not None
    assert market.series_scope == GARCH_DATE_LEVEL_SCOPE
    assert set(names) == {"S00", "S01"}
    assert names["S00"].series_scope == GARCH_SECURITY_LEVEL_SCOPE
    assert names["S01"].series_scope == GARCH_SECURITY_LEVEL_SCOPE
    assert names["S00"].n_obs == names["S01"].n_obs == 70
    assert names["S01"].sigma != pytest.approx(names["S00"].sigma, rel=1e-3, abs=1e-8)
    assert names["S00"].sigma != pytest.approx(market.sigma, rel=1e-3, abs=1e-8)
    assert names["S01"].sigma != pytest.approx(market.sigma, rel=1e-3, abs=1e-8)


def test_name_forecasts_ignore_returns_on_or_after_origin(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[70]
    _save_garch(tmp_path, np.full(80, 0.01))
    poisoned = frame.with_columns(
        pl.when((pl.col("event_time") >= asof) & (pl.col("security_id") == "S00"))
        .then(pl.lit(0.5))
        .otherwise(pl.col("ret_1"))
        .alias("ret_1")
    )
    clean = garch_name_forecasts_asof(cfg, frame, asof)
    clear_forecast_caches()
    dirty = garch_name_forecasts_asof(cfg, poisoned, asof)
    assert clean["S00"].variance == pytest.approx(dirty["S00"].variance)
    assert clean["S01"].variance == pytest.approx(dirty["S01"].variance)


def test_name_forecasts_ignore_late_available_restatement(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _panel()
    dates = frame["event_time"].unique().sort().to_list()
    asof = dates[70]
    early = dates[20]
    _save_garch(tmp_path, np.full(80, 0.01))
    poisoned = frame.with_columns(
        pl.when((pl.col("security_id") == "S00") & (pl.col("event_time") == early))
        .then(pl.lit(0.8))
        .otherwise(pl.col("ret_1"))
        .alias("ret_1")
    ).with_columns(
        pl.when((pl.col("security_id") == "S00") & (pl.col("event_time") == early))
        .then(pl.lit(asof + timedelta(days=1)))
        .otherwise(pl.col("available_time"))
        .alias("available_time")
    )
    dropped = frame.filter(~((pl.col("security_id") == "S00") & (pl.col("event_time") == early)))
    dirty = garch_name_forecasts_asof(cfg, poisoned, asof, security_ids=["S00"])
    clear_forecast_caches()
    clean = garch_name_forecasts_asof(cfg, dropped, asof, security_ids=["S00"])
    assert dirty["S00"].variance == pytest.approx(clean["S00"].variance)
    assert dirty["S00"].n_obs == clean["S00"].n_obs == 69


def test_name_forecasts_exclude_non_members(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _panel(n_names=3)
    asof = frame["event_time"].unique().sort().to_list()[70]
    _write_universe(tmp_path, frame, ["S00"])
    _save_garch(tmp_path, np.full(80, 0.01))
    names = garch_name_forecasts_asof(cfg, frame, asof)
    assert set(names) == {"S00"}
    with pytest.raises(ValueError, match="no strictly prior returns"):
        garch_name_forecasts_asof(cfg, frame, asof, security_ids=["S01"])


def test_name_forecasts_missing_artifact_returns_empty(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[70]
    assert garch_name_forecasts_asof(cfg, frame, asof) == {}


def test_name_forecasts_wrong_series_scope_fails_closed(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[70]
    _save_garch(tmp_path, np.full(80, 0.01), series_scope="univariate_return_series")
    with pytest.raises(ValueError, match="series_scope"):
        garch_name_forecasts_asof(cfg, frame, asof)


def test_forecast_asof_keeps_name_vol_and_market_overlay(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[70]
    _save_garch(tmp_path, np.full(80, 0.01))
    names = garch_name_forecasts_asof(cfg, frame, asof)
    state = forecast_asof(cfg, asof, frame=frame)
    assert state.garch_series_scope == GARCH_DATE_LEVEL_SCOPE
    assert state.market_risk_overlay == "garch"
    by_id = {
        row["security_id"]: float(row["vol_20"])
        for row in frame.filter(pl.col("event_time") == asof).iter_rows(named=True)
    }
    for forecast in state.forecasts:
        assert forecast.volatility["5d"] == pytest.approx(by_id[forecast.security_id])
        assert forecast.diagnostics["garch_series_scope"] == GARCH_DATE_LEVEL_SCOPE
        assert "garch_name_sigma" not in forecast.diagnostics
        assert names[forecast.security_id].sigma != pytest.approx(
            float(forecast.diagnostics["garch_market_sigma"]), rel=1e-3, abs=1e-8
        )


def test_name_forecasts_explicit_blank_id_fails_closed(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[70]
    _save_garch(tmp_path, np.full(80, 0.01))
    with pytest.raises(ValueError, match="non-empty"):
        garch_name_forecasts_asof(cfg, frame, asof, security_ids=["S00", " "])


def test_name_forecasts_cache_misses_when_spec_bytes_change_at_same_mtime(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[70]
    path = _save_garch(tmp_path, np.full(80, 0.01), mean="Constant")
    first = garch_name_forecasts_asof(cfg, frame, asof, security_ids=["S00"])
    mtime_ns = path.stat().st_mtime_ns
    replacement = GARCHVol(
        series_scope=GARCH_DATE_LEVEL_SCOPE, min_obs=20, mean="Zero"
    ).fit_returns(np.full(80, 0.01))
    _replace_garch_preserving_mtime(path, replacement)
    assert path.stat().st_mtime_ns == mtime_ns
    second = garch_name_forecasts_asof(cfg, frame, asof, security_ids=["S00"])
    assert second["S00"].variance != pytest.approx(first["S00"].variance, rel=1e-8, abs=1e-12)
    clear_forecast_caches()
    fresh = garch_name_forecasts_asof(cfg, frame, asof, security_ids=["S00"])
    assert second["S00"].variance == pytest.approx(fresh["S00"].variance)
