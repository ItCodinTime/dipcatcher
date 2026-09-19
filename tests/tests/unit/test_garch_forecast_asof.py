"""Causal GARCH market overlay for forecast_asof / optimizer covariance."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from quant_fund.config import load_config
from quant_fund.data.lake import Lake
from quant_fund.models.volatility import GARCH_DATE_LEVEL_SCOPE, GARCHVol
from quant_fund.pipeline.forecast import (
    _load_garch_spec_cached,
    apply_market_variance_overlay_to_covariance,
    clear_forecast_caches,
    forecast_asof,
    garch_market_forecast_asof,
    optimize_asof,
    overlay_covariance_with_garch_market,
)
from quant_fund.pipeline.train import _garch_return_history
from quant_fund.schemas.errors import PointInTimeError
from quant_fund.schemas.forecast import MarketState


def _cfg(tmp_path: Path):
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.data.source = "synthetic"
    cfg.fusion.skip_intervals = True
    cfg.fusion.apply_interval_caps = False
    return cfg


def _panel(n_days: int = 80, n_names: int = 6, seed: int = 17) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime(2020, 1, 2, 16, 0, 0)
    rows: list[dict[str, object]] = []
    for day in range(n_days):
        stamp = start + timedelta(days=day)
        mkt = float(rng.normal(0.0, 0.01))
        for name in range(n_names):
            vol = 0.012 + 0.004 * name
            ret = mkt + float(rng.normal(0.0, vol))
            rows.append(
                {
                    "event_time": stamp,
                    "security_id": f"S{name:02d}",
                    "symbol": f"S{name:02d}",
                    "ret_1": ret,
                    "vol_20": vol,
                    "cs_pct_mom_20": 0.5 + 0.01 * name,
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
    """Overwrite the artifact while keeping the previous mtime (Wave 113)."""
    stat = path.stat()
    model.save(path)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))


@pytest.fixture(autouse=True)
def _clear_garch_caches() -> None:
    clear_forecast_caches()
    yield
    clear_forecast_caches()


def test_overlay_scales_equal_weight_variance_to_garch_level() -> None:
    rng = np.random.default_rng(3)
    factor = rng.normal(size=(8, 3))
    sig = factor @ factor.T + 0.04 * np.eye(8)
    target = 0.0009
    overlaid = overlay_covariance_with_garch_market(sig, target)
    weights = np.full(8, 1.0 / 8.0)
    assert float(weights @ overlaid @ weights) == pytest.approx(target, rel=1e-8, abs=1e-12)
    with pytest.raises(ValueError, match="strictly positive"):
        overlay_covariance_with_garch_market(sig, 0.0)


def test_forecast_asof_without_garch_artifact_leaves_name_vol_untouched(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _panel()
    asof = frame["event_time"].max()
    state = forecast_asof(cfg, asof, frame=frame)
    assert state.garch_market_sigma is None
    assert "garch_market_cross_section" not in state.notes
    by_id = {
        row["security_id"]: float(row["vol_20"])
        for row in frame.filter(pl.col("event_time") == asof).iter_rows(named=True)
    }
    for forecast in state.forecasts:
        assert forecast.volatility["5d"] == pytest.approx(by_id[forecast.security_id])
        assert "garch_market_sigma" not in forecast.diagnostics


def test_forecast_asof_attaches_causal_garch_without_replacing_name_vol(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[70]
    _save_garch(tmp_path, np.full(80, 0.01))
    state = forecast_asof(cfg, asof, frame=frame)
    assert state.garch_market_sigma is not None
    assert state.garch_market_variance is not None
    assert state.garch_series_scope == GARCH_DATE_LEVEL_SCOPE
    assert state.garch_horizon == 5
    assert state.garch_n_obs == 70
    assert state.market_risk_overlay == "garch"
    assert "garch_market_cross_section" in state.notes
    by_id = {
        row["security_id"]: float(row["vol_20"])
        for row in frame.filter(pl.col("event_time") == asof).iter_rows(named=True)
    }
    market_sigmas = {float(f.diagnostics["garch_market_sigma"]) for f in state.forecasts}
    assert len(market_sigmas) == 1
    for forecast in state.forecasts:
        assert forecast.volatility["5d"] == pytest.approx(by_id[forecast.security_id])
        assert forecast.diagnostics["garch_series_scope"] == GARCH_DATE_LEVEL_SCOPE
        assert float(forecast.diagnostics["garch_market_sigma"]) == pytest.approx(
            float(state.garch_market_sigma)
        )


def test_garch_asof_ignores_returns_on_or_after_origin(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[70]
    _save_garch(tmp_path, np.full(80, 0.01))
    poisoned = frame.with_columns(
        pl.when(pl.col("event_time") >= asof)
        .then(pl.lit(0.5))
        .otherwise(pl.col("ret_1"))
        .alias("ret_1")
    )
    clear_forecast_caches()
    dirty = garch_market_forecast_asof(cfg, poisoned, asof)
    clear_forecast_caches()
    clean = garch_market_forecast_asof(cfg, frame, asof)
    assert clean is not None and dirty is not None
    assert dirty.variance == pytest.approx(clean.variance)
    assert dirty.n_obs == clean.n_obs == 70


def test_garch_asof_does_not_reuse_persisted_full_sample_fit(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _panel()
    dates = frame["event_time"].unique().sort().to_list()
    asof = dates[70]
    early = frame.filter(pl.col("event_time") < asof)
    late = frame.filter(pl.col("event_time") >= asof).with_columns(pl.lit(0.25).alias("ret_1"))
    leaked = pl.concat([early, late])
    _dates, values = _garch_return_history(leaked)
    _save_garch(tmp_path, values)
    persisted = GARCHVol.load(tmp_path / "metadata" / "vol_garch.joblib")
    leaked_forecast = persisted.forecast(horizon=1)
    causal = garch_market_forecast_asof(cfg, leaked, asof)
    assert causal is not None
    assert float(leaked_forecast["variance"][0]) != pytest.approx(
        causal.variance, rel=1e-8, abs=1e-12
    )


def test_wrong_series_scope_fails_closed(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[70]
    _save_garch(tmp_path, np.full(80, 0.01), series_scope="univariate_return_series")
    with pytest.raises(ValueError, match="series_scope"):
        forecast_asof(cfg, asof, frame=frame)


def test_optimize_asof_persists_garch_overlay_columns(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[70]
    _save_garch(tmp_path, np.full(80, 0.01))
    weights = optimize_asof(cfg, asof, persist=False, frame=frame)
    assert "garch_market_sigma" in weights.columns
    assert "garch_series_scope" in weights.columns
    sigmas = weights["garch_market_sigma"].to_list()
    assert all(np.isfinite(float(value)) and float(value) > 0.0 for value in sigmas)
    assert set(weights["garch_series_scope"].to_list()) == {GARCH_DATE_LEVEL_SCOPE}
    assert set(weights["market_risk_overlay"].to_list()) == {"garch"}


def test_apply_market_variance_overlay_matches_direct_scale(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[70]
    _save_garch(tmp_path, np.full(80, 0.01))
    rng = np.random.default_rng(4)
    factor = rng.normal(size=(6, 3))
    sig = factor @ factor.T + 0.03 * np.eye(6)
    garch = garch_market_forecast_asof(cfg, frame, asof)
    assert garch is not None
    direct = overlay_covariance_with_garch_market(sig, garch.variance)
    applied, overlay, kind = apply_market_variance_overlay_to_covariance(cfg, frame, asof, sig)
    assert kind == "garch"
    assert overlay is not None
    assert overlay.variance == pytest.approx(garch.variance)
    assert applied == pytest.approx(direct)


def test_market_state_rejects_partial_garch_fields() -> None:
    asof = datetime(2020, 6, 1, 16, 0, 0)
    with pytest.raises(ValueError, match="together"):
        MarketState(asof=asof, forecasts=[], garch_market_sigma=0.02)
    with pytest.raises(ValueError, match="together"):
        MarketState(
            asof=asof,
            forecasts=[],
            garch_market_sigma=0.02,
            garch_market_variance=0.0004,
            garch_cumulative_variance=0.0004,
            garch_horizon=1,
            garch_series_scope=GARCH_DATE_LEVEL_SCOPE,
            garch_fit_status="fitted",
            garch_n_obs=20,
        )


def test_market_state_rejects_unknown_market_risk_overlay() -> None:
    asof = datetime(2020, 6, 1, 16, 0, 0)
    with pytest.raises(ValueError, match="market_risk_overlay"):
        MarketState(
            asof=asof,
            forecasts=[],
            garch_market_sigma=0.02,
            garch_market_variance=0.0004,
            garch_cumulative_variance=0.0004,
            garch_horizon=1,
            garch_series_scope=GARCH_DATE_LEVEL_SCOPE,
            garch_fit_status="fitted",
            garch_n_obs=20,
            market_risk_overlay="ewma",
        )


def _write_universe(tmp_path: Path, frame: pl.DataFrame, security_ids: list[str]) -> None:
    membership = (
        frame.filter(pl.col("security_id").is_in(security_ids))
        .select(pl.col("security_id"), pl.col("event_time").alias("asof"))
        .unique()
    )
    Lake(tmp_path).write_parquet(membership, "silver/universe.parquet")


def test_garch_asof_cache_misses_when_earlier_history_changes(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _panel()
    dates = frame["event_time"].unique().sort().to_list()
    asof = dates[70]
    early_cut = dates[40]
    _save_garch(tmp_path, np.full(80, 0.01))
    poisoned = frame.with_columns(
        pl.when(pl.col("event_time") < early_cut)
        .then(pl.lit(0.25))
        .otherwise(pl.col("ret_1"))
        .alias("ret_1")
    )
    first = garch_market_forecast_asof(cfg, frame, asof)
    second = garch_market_forecast_asof(cfg, poisoned, asof)
    assert first is not None and second is not None
    assert first.n_obs == second.n_obs == 70
    assert second.variance != pytest.approx(first.variance, rel=1e-8, abs=1e-12)


def test_garch_asof_excludes_non_member_returns(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[70]
    members = ["S00", "S01", "S02"]
    _write_universe(tmp_path, frame, members)
    _save_garch(tmp_path, np.full(80, 0.01))
    poisoned = frame.with_columns(
        pl.when(pl.col("security_id") == "S05")
        .then(pl.lit(0.8))
        .otherwise(pl.col("ret_1"))
        .alias("ret_1")
    )
    members_only = frame.filter(pl.col("security_id").is_in(members))
    dirty = garch_market_forecast_asof(cfg, poisoned, asof)
    clear_forecast_caches()
    clean = garch_market_forecast_asof(cfg, members_only, asof)
    assert dirty is not None and clean is not None
    assert dirty.variance == pytest.approx(clean.variance)
    assert dirty.n_obs == clean.n_obs
    (tmp_path / "silver" / "universe.parquet").unlink()
    clear_forecast_caches()
    unfiltered = garch_market_forecast_asof(cfg, poisoned, asof)
    assert unfiltered is not None
    assert unfiltered.variance != pytest.approx(clean.variance, rel=1e-8, abs=1e-12)


def test_garch_asof_universe_without_security_id_fails_closed(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[70]
    _write_universe(tmp_path, frame, ["S00"])
    _save_garch(tmp_path, np.full(80, 0.01))
    with pytest.raises(PointInTimeError, match="security_id"):
        garch_market_forecast_asof(cfg, frame.select("event_time", "ret_1"), asof)


def test_garch_asof_ignores_late_available_return_restatement(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _panel().with_columns(pl.col("event_time").alias("available_time"))
    dates = frame["event_time"].unique().sort().to_list()
    asof = dates[70]
    early = dates[20]
    _save_garch(tmp_path, np.full(80, 0.01))
    poisoned = frame.with_columns(
        pl.when((pl.col("security_id") == "S05") & (pl.col("event_time") == early))
        .then(pl.lit(0.9))
        .otherwise(pl.col("ret_1"))
        .alias("ret_1")
    ).with_columns(
        pl.when((pl.col("security_id") == "S05") & (pl.col("event_time") == early))
        .then(pl.lit(asof + timedelta(days=1)))
        .otherwise(pl.col("available_time"))
        .alias("available_time")
    )
    dropped = frame.filter(~((pl.col("security_id") == "S05") & (pl.col("event_time") == early)))
    clear_forecast_caches()
    dirty = garch_market_forecast_asof(cfg, poisoned, asof)
    clear_forecast_caches()
    clean = garch_market_forecast_asof(cfg, dropped, asof)
    assert clean is not None and dirty is not None
    assert dirty.variance == pytest.approx(clean.variance)
    assert dirty.n_obs == clean.n_obs


def test_garch_asof_includes_restatement_once_available(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _panel().with_columns(pl.col("event_time").alias("available_time"))
    dates = frame["event_time"].unique().sort().to_list()
    asof = dates[70]
    early = dates[20]
    _save_garch(tmp_path, np.full(80, 0.01))
    poisoned = frame.with_columns(
        pl.when((pl.col("security_id") == "S05") & (pl.col("event_time") == early))
        .then(pl.lit(0.9))
        .otherwise(pl.col("ret_1"))
        .alias("ret_1")
    )
    clear_forecast_caches()
    dirty = garch_market_forecast_asof(cfg, poisoned, asof)
    clear_forecast_caches()
    clean = garch_market_forecast_asof(cfg, frame, asof)
    assert clean is not None and dirty is not None
    assert dirty.variance != pytest.approx(clean.variance, rel=1e-8, abs=1e-12)


def test_garch_asof_null_available_time_fails_closed(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _panel().with_columns(pl.col("event_time").alias("available_time"))
    asof = frame["event_time"].unique().sort().to_list()[70]
    _save_garch(tmp_path, np.full(80, 0.01))
    poisoned = frame.with_columns(
        pl.when(pl.col("security_id") == "S00")
        .then(pl.lit(None, dtype=pl.Datetime))
        .otherwise(pl.col("available_time"))
        .alias("available_time")
    )
    with pytest.raises(PointInTimeError, match="available_time"):
        garch_market_forecast_asof(cfg, poisoned, asof)


def test_garch_asof_empty_universe_fails_closed(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[70]
    _save_garch(tmp_path, np.full(80, 0.01))
    empty = pl.DataFrame(schema={"security_id": pl.String, "asof": pl.Datetime})
    Lake(tmp_path).write_parquet(empty, "silver/universe.parquet")
    with pytest.raises(PointInTimeError, match="empty"):
        garch_market_forecast_asof(cfg, frame, asof)


def test_garch_spec_cache_uses_artifact_bytes_not_mtime(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    path = _save_garch(tmp_path, np.full(80, 0.01), mean="Constant")
    loaded = _load_garch_spec_cached(cfg)
    assert loaded is not None
    spec, digest = loaded
    assert spec.mean == "Constant"
    mtime_ns = path.stat().st_mtime_ns
    replacement = GARCHVol(
        series_scope=GARCH_DATE_LEVEL_SCOPE, min_obs=20, mean="Zero"
    ).fit_returns(np.full(80, 0.01))
    _replace_garch_preserving_mtime(path, replacement)
    assert path.stat().st_mtime_ns == mtime_ns
    reloaded = _load_garch_spec_cached(cfg)
    assert reloaded is not None
    spec2, digest2 = reloaded
    assert digest2 != digest
    assert spec2.mean == "Zero"


def test_garch_asof_cache_misses_when_spec_bytes_change_at_same_mtime(tmp_path: Path) -> None:
    """Mean is not in the old (p, q, dist, vol) as-of key; bytes must invalidate."""
    cfg = _cfg(tmp_path)
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[70]
    path = _save_garch(tmp_path, np.full(80, 0.01), mean="Constant")
    first = garch_market_forecast_asof(cfg, frame, asof)
    assert first is not None
    mtime_ns = path.stat().st_mtime_ns
    replacement = GARCHVol(
        series_scope=GARCH_DATE_LEVEL_SCOPE, min_obs=20, mean="Zero"
    ).fit_returns(np.full(80, 0.01))
    _replace_garch_preserving_mtime(path, replacement)
    assert path.stat().st_mtime_ns == mtime_ns
    second = garch_market_forecast_asof(cfg, frame, asof)
    assert second is not None
    assert second.variance != pytest.approx(first.variance, rel=1e-8, abs=1e-12)
    clear_forecast_caches()
    fresh = garch_market_forecast_asof(cfg, frame, asof)
    assert fresh is not None
    assert second.variance == pytest.approx(fresh.variance)


def _capture_optimize_sigma(cfg, frame: pl.DataFrame, asof: datetime) -> np.ndarray:
    seen: dict[str, np.ndarray] = {}

    def capture(alpha, sigma, w_prev, config, **kwargs):  # noqa: ANN001
        seen["sigma"] = np.asarray(sigma, dtype=float).copy()
        n = len(alpha)
        return np.zeros(n), type("D", (), {"feasible": True})()

    with patch("quant_fund.pipeline.forecast.optimize_mean_variance", side_effect=capture):
        optimize_asof(cfg, asof, persist=False, frame=frame, use_fused_alpha=False)
    assert "sigma" in seen
    return seen["sigma"]


def test_optimize_asof_covariance_ignores_late_available_restatement(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _panel().with_columns(pl.col("event_time").alias("available_time"))
    dates = frame["event_time"].unique().sort().to_list()
    asof = dates[70]
    early = dates[20]
    poisoned = frame.with_columns(
        pl.when((pl.col("security_id") == "S05") & (pl.col("event_time") == early))
        .then(pl.lit(0.9))
        .otherwise(pl.col("ret_1"))
        .alias("ret_1")
    ).with_columns(
        pl.when((pl.col("security_id") == "S05") & (pl.col("event_time") == early))
        .then(pl.lit(asof + timedelta(days=1)))
        .otherwise(pl.col("available_time"))
        .alias("available_time")
    )
    dropped = frame.filter(~((pl.col("security_id") == "S05") & (pl.col("event_time") == early)))
    clean = _capture_optimize_sigma(cfg, frame, asof)
    dirty = _capture_optimize_sigma(cfg, poisoned, asof)
    omitted = _capture_optimize_sigma(cfg, dropped, asof)
    np.testing.assert_allclose(dirty, omitted, rtol=1e-10, atol=1e-12)
    assert not np.allclose(dirty, clean, rtol=1e-5, atol=1e-8)


def test_optimize_asof_covariance_includes_restatement_once_available(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _panel().with_columns(pl.col("event_time").alias("available_time"))
    dates = frame["event_time"].unique().sort().to_list()
    asof = dates[70]
    early = dates[20]
    published = frame.with_columns(
        pl.when((pl.col("security_id") == "S05") & (pl.col("event_time") == early))
        .then(pl.lit(0.9))
        .otherwise(pl.col("ret_1"))
        .alias("ret_1")
    )
    clean = _capture_optimize_sigma(cfg, frame, asof)
    dirty = _capture_optimize_sigma(cfg, published, asof)
    assert not np.allclose(dirty, clean, rtol=1e-5, atol=1e-8)


def test_optimize_asof_covariance_null_available_time_fails_closed(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _panel().with_columns(pl.col("event_time").alias("available_time"))
    dates = frame["event_time"].unique().sort().to_list()
    asof = dates[70]
    early = dates[20]
    poisoned = frame.with_columns(
        pl.when((pl.col("security_id") == "S05") & (pl.col("event_time") == early))
        .then(pl.lit(None))
        .otherwise(pl.col("available_time"))
        .alias("available_time")
    )
    with pytest.raises(PointInTimeError, match="null available_time"):
        optimize_asof(cfg, asof, persist=False, frame=poisoned, use_fused_alpha=False)
