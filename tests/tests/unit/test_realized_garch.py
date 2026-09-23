"""Causal Realized GARCH on daily Parkinson; not high-frequency RV."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quant_fund.config import load_config
from quant_fund.models.realized_garch import (
    REALIZED_GARCH_FAMILY,
    REALIZED_GARCH_MEASURE,
    RealizedGARCHVol,
    parkinson_daily_variance,
)
from quant_fund.models.volatility import GARCH_DATE_LEVEL_SCOPE, GARCHVol
from quant_fund.pipeline.forecast import (
    MARKET_RISK_OVERLAY_GARCH,
    MARKET_RISK_OVERLAY_REALIZED_GARCH,
    clear_forecast_caches,
    forecast_asof,
    garch_market_forecast_asof,
    optimize_asof,
    realized_garch_market_forecast_asof,
)
from quant_fund.pipeline.train import _realized_garch_history
from quant_fund.schemas.errors import PointInTimeError


def _cfg(tmp_path: Path):
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.data.source = "synthetic"
    cfg.fusion.skip_intervals = True
    cfg.fusion.apply_interval_caps = False
    return cfg


def _rgarch_panel(n_days: int = 80, n_names: int = 4, seed: int = 17) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime(2020, 1, 2, 16, 0, 0)
    rows: list[dict[str, object]] = []
    close = {name: 100.0 + name for name in range(n_names)}
    for day in range(n_days):
        stamp = start + timedelta(days=day)
        mkt = float(rng.normal(0.0, 0.01))
        for name in range(n_names):
            vol = 0.012 + 0.003 * name
            ret = mkt + float(rng.normal(0.0, vol))
            px = close[name] * math.exp(ret)
            close[name] = px
            span = max(abs(ret), 0.004)
            rows.append(
                {
                    "event_time": stamp,
                    "security_id": f"S{name:02d}",
                    "symbol": f"S{name:02d}",
                    "ret_1": ret,
                    "vol_20": vol,
                    "vol_parkinson": vol,
                    "cs_pct_mom_20": 0.5 + 0.01 * name,
                    "high_split_adjusted": px * math.exp(span),
                    "low_split_adjusted": px * math.exp(-span),
                    "open_split_adjusted": px,
                    "close_split_adjusted": px,
                }
            )
    return pl.DataFrame(rows)


def _save_rgarch(
    tmp_path: Path,
    returns: np.ndarray,
    measure: np.ndarray,
    *,
    series_scope: str = GARCH_DATE_LEVEL_SCOPE,
) -> Path:
    path = tmp_path / "metadata" / "vol_realized_garch.joblib"
    RealizedGARCHVol(min_obs=20, series_scope=series_scope).fit_returns(returns, measure).save(path)
    return path


def _save_garch(tmp_path: Path, returns: np.ndarray) -> Path:
    path = tmp_path / "metadata" / "vol_garch.joblib"
    GARCHVol(series_scope=GARCH_DATE_LEVEL_SCOPE, min_obs=20).fit_returns(returns).save(path)
    return path


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    clear_forecast_caches()
    yield
    clear_forecast_caches()


def test_parkinson_identity_and_undefined_rows() -> None:
    high = np.array([math.e, 2.0, 0.0, 1.0, 1.2], dtype=float)
    low = np.array([1.0, 1.0, 1.0, 1.1, 1.2], dtype=float)
    park = parkinson_daily_variance(high, low)
    assert park[0] == pytest.approx(1.0 / (4.0 * math.log(2.0)))
    assert np.isnan(park[2])
    assert np.isnan(park[3])
    assert park[4] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="same length"):
        parkinson_daily_variance([1.0, 2.0], [1.0])


def test_realized_garch_rejects_inferred_or_invented_measures() -> None:
    returns = np.full(80, 0.01)
    with pytest.raises(ValueError, match="realized_measure"):
        RealizedGARCHVol().fit(np.empty((80, 0)), returns, returns=returns)
    with pytest.raises(ValueError, match="returns="):
        RealizedGARCHVol().fit(
            np.empty((80, 0)), returns, realized_measure=np.full(80, 0.0001)
        )
    with pytest.raises(ValueError, match="parkinson_daily_ohlc"):
        RealizedGARCHVol(realized_measure="close_to_close_squared")
    with pytest.raises(ValueError, match="same length"):
        RealizedGARCHVol(min_obs=20).fit_returns(returns, np.full(10, 0.0001))


def test_realized_garch_fits_and_stamps_daily_parkinson_honesty() -> None:
    rng = np.random.default_rng(5)
    returns = rng.normal(0.0, 0.01, size=120)
    high = np.exp(np.abs(returns) + 0.01)
    low = np.exp(-(np.abs(returns) + 0.01))
    measure = parkinson_daily_variance(high, low)
    model = RealizedGARCHVol(min_obs=40, mean="Zero").fit_returns(returns, measure)
    forecast = model.forecast(horizon=5)
    diag = model.diagnostics()
    assert diag["realized_measure"] == REALIZED_GARCH_MEASURE
    assert diag["intraday_realized_variance"] is False
    assert diag["variance_family"] == REALIZED_GARCH_FAMILY
    assert diag["vol"] == "realized_garch"
    assert forecast["intraday_realized_variance"] is False
    assert forecast["realized_measure"] == REALIZED_GARCH_MEASURE
    assert forecast["variance"].shape == (5,)
    assert np.all(forecast["variance"] > 0.0)
    assert forecast["multi_step_method"] == "expected_log_variance_plugin"
    pit = model.pit(returns[-1:], forecast["sigma"][:1])
    assert 0.0 <= float(pit[0]) <= 1.0
    score = model.log_density(returns[-1:], forecast["sigma"][:1])
    assert np.isfinite(score[0])


def test_one_step_forecast_moves_with_last_parkinson_not_squared_return() -> None:
    rng = np.random.default_rng(9)
    returns = rng.normal(0.0, 0.012, size=100)
    base = parkinson_daily_variance(
        np.exp(np.abs(returns) + 0.008), np.exp(-(np.abs(returns) + 0.008))
    )
    quiet = base.copy()
    quiet[-1] = 1e-8
    loud = base.copy()
    loud[-1] = 0.01
    quiet_model = RealizedGARCHVol(min_obs=30, mean="Zero").fit_returns(returns, quiet)
    loud_model = RealizedGARCHVol(min_obs=30, mean="Zero").fit_returns(returns, loud)
    assert quiet_model.fit_status == "fitted"
    assert loud_model.fit_status == "fitted"
    quiet_f = quiet_model.forecast(horizon=1)
    loud_f = loud_model.forecast(horizon=1)
    assert float(loud_f["variance"][0]) != pytest.approx(float(quiet_f["variance"][0]))


def test_history_uses_one_day_parkinson_not_rolling_vol_parkinson() -> None:
    frame = _rgarch_panel()
    poisoned = frame.with_columns(pl.lit(9.9).alias("vol_parkinson"))
    dates, values, measures = _realized_garch_history(frame)
    _d2, _v2, poisoned_measures = _realized_garch_history(poisoned)
    assert dates.size == 80
    assert values.size == measures.size
    assert np.allclose(measures, poisoned_measures)
    assert float(np.max(measures)) < 1.0


def test_history_fail_closed_without_ohlc() -> None:
    frame = _rgarch_panel().drop(["high_split_adjusted", "low_split_adjusted"])
    with pytest.raises(PointInTimeError, match="daily OHLC"):
        _realized_garch_history(frame)


def test_realized_garch_asof_ignores_origin_and_future_ohlc(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _rgarch_panel()
    asof = frame["event_time"].unique().sort().to_list()[70]
    _asof_dates, values, measures = _realized_garch_history(frame.filter(pl.col("event_time") < asof))
    _save_rgarch(tmp_path, values, measures, series_scope=GARCH_DATE_LEVEL_SCOPE)
    clean = realized_garch_market_forecast_asof(cfg, frame, asof)
    poisoned = frame.with_columns(
        pl.when(pl.col("event_time") >= asof)
        .then(pl.col("high_split_adjusted") * 4.0)
        .otherwise(pl.col("high_split_adjusted"))
        .alias("high_split_adjusted")
    )
    clear_forecast_caches()
    dirty = realized_garch_market_forecast_asof(cfg, poisoned, asof)
    assert clean is not None and dirty is not None
    assert dirty.variance == pytest.approx(clean.variance)
    assert dirty.n_obs == clean.n_obs
    assert dirty.realized_measure == REALIZED_GARCH_MEASURE
    assert dirty.intraday_realized_variance is False
    assert dirty.series_scope == GARCH_DATE_LEVEL_SCOPE


def test_realized_garch_asof_drops_unpublished_ohlc_restatement(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _rgarch_panel().with_columns(pl.col("event_time").alias("available_time"))
    dates = frame["event_time"].unique().sort().to_list()
    asof = dates[70]
    early = dates[20]
    _hist_dates, values, measures = _realized_garch_history(frame.filter(pl.col("event_time") < asof))
    _save_rgarch(tmp_path, values, measures, series_scope=GARCH_DATE_LEVEL_SCOPE)
    poisoned = frame.with_columns(
        pl.when((pl.col("security_id") == "S00") & (pl.col("event_time") == early))
        .then(pl.col("high_split_adjusted") * 4.0)
        .otherwise(pl.col("high_split_adjusted"))
        .alias("high_split_adjusted")
    ).with_columns(
        pl.when((pl.col("security_id") == "S00") & (pl.col("event_time") == early))
        .then(pl.lit(asof + timedelta(days=1)))
        .otherwise(pl.col("available_time"))
        .alias("available_time")
    )
    dropped = frame.filter(~((pl.col("security_id") == "S00") & (pl.col("event_time") == early)))
    clear_forecast_caches()
    dirty = realized_garch_market_forecast_asof(cfg, poisoned, asof)
    clear_forecast_caches()
    clean = realized_garch_market_forecast_asof(cfg, dropped, asof)
    assert clean is not None and dirty is not None
    assert dirty.variance == pytest.approx(clean.variance)
    assert dirty.n_obs == clean.n_obs


def test_realized_garch_asof_uses_restatement_once_available(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _rgarch_panel().with_columns(pl.col("event_time").alias("available_time"))
    dates = frame["event_time"].unique().sort().to_list()
    asof = dates[70]
    early = dates[20]
    _hist_dates, values, measures = _realized_garch_history(frame.filter(pl.col("event_time") < asof))
    _save_rgarch(tmp_path, values, measures, series_scope=GARCH_DATE_LEVEL_SCOPE)
    poisoned = frame.with_columns(
        pl.when((pl.col("security_id") == "S00") & (pl.col("event_time") == early))
        .then(pl.col("high_split_adjusted") * 4.0)
        .otherwise(pl.col("high_split_adjusted"))
        .alias("high_split_adjusted")
    )
    clear_forecast_caches()
    dirty = realized_garch_market_forecast_asof(cfg, poisoned, asof)
    clear_forecast_caches()
    clean = realized_garch_market_forecast_asof(cfg, frame, asof)
    assert clean is not None and dirty is not None
    assert dirty.variance != pytest.approx(clean.variance, rel=1e-8, abs=1e-12)


def test_realized_garch_asof_null_available_time_fail_closed(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _rgarch_panel().with_columns(pl.col("event_time").alias("available_time"))
    asof = frame["event_time"].unique().sort().to_list()[40]
    _asof_dates, values, measures = _realized_garch_history(frame.filter(pl.col("event_time") < asof))
    _save_rgarch(tmp_path, values, measures, series_scope=GARCH_DATE_LEVEL_SCOPE)
    poisoned = frame.with_columns(
        pl.when(pl.col("security_id") == "S00")
        .then(pl.lit(None, dtype=pl.Datetime))
        .otherwise(pl.col("available_time"))
        .alias("available_time")
    )
    with pytest.raises(PointInTimeError, match="available_time"):
        realized_garch_market_forecast_asof(cfg, poisoned, asof)


def test_wrong_scope_and_missing_artifact(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _rgarch_panel()
    asof = frame["event_time"].unique().sort().to_list()[50]
    assert realized_garch_market_forecast_asof(cfg, frame, asof) is None
    _asof_dates, values, measures = _realized_garch_history(frame.filter(pl.col("event_time") < asof))
    _save_rgarch(tmp_path, values, measures, series_scope="univariate_return_series")
    with pytest.raises(ValueError, match="series_scope"):
        realized_garch_market_forecast_asof(cfg, frame, asof)


def test_forecast_asof_consumes_realized_garch_overlay(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _rgarch_panel()
    asof = frame["event_time"].unique().sort().to_list()[60]
    _asof_dates, values, measures = _realized_garch_history(frame.filter(pl.col("event_time") < asof))
    _save_rgarch(tmp_path, values, measures, series_scope=GARCH_DATE_LEVEL_SCOPE)
    state = forecast_asof(cfg, asof, frame=frame)
    rgarch = realized_garch_market_forecast_asof(cfg, frame, asof)
    assert rgarch is not None
    assert state.garch_market_sigma == pytest.approx(rgarch.sigma)
    assert state.garch_market_variance == pytest.approx(rgarch.variance)
    assert state.market_risk_overlay == MARKET_RISK_OVERLAY_REALIZED_GARCH
    assert "realized_garch_market_cross_section" in state.notes
    assert "garch_market_cross_section" not in state.notes
    assert garch_market_forecast_asof(cfg, frame, asof) is None
    by_id = {
        row["security_id"]: float(row["vol_20"])
        for row in frame.filter(pl.col("event_time") == asof).iter_rows(named=True)
    }
    for forecast in state.forecasts:
        assert forecast.volatility["5d"] == pytest.approx(by_id[forecast.security_id])
        assert forecast.diagnostics["market_risk_overlay"] == MARKET_RISK_OVERLAY_REALIZED_GARCH
        assert forecast.diagnostics["realized_measure"] == REALIZED_GARCH_MEASURE
        assert forecast.diagnostics["intraday_realized_variance"] == "false"
        assert forecast.diagnostics["garch_variance_family"] == REALIZED_GARCH_FAMILY
        assert float(forecast.diagnostics["garch_market_sigma"]) == pytest.approx(rgarch.sigma)


def test_forecast_asof_prefers_rgarch_over_return_only_garch(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _rgarch_panel()
    asof = frame["event_time"].unique().sort().to_list()[60]
    prior = frame.filter(pl.col("event_time") < asof)
    _asof_dates, values, measures = _realized_garch_history(prior)
    _save_rgarch(tmp_path, values, measures)
    _save_garch(tmp_path, values)
    garch = garch_market_forecast_asof(cfg, frame, asof)
    rgarch = realized_garch_market_forecast_asof(cfg, frame, asof)
    assert garch is not None and rgarch is not None
    assert rgarch.variance != pytest.approx(garch.variance, rel=1e-5, abs=1e-12)
    state = forecast_asof(cfg, asof, frame=frame)
    assert state.market_risk_overlay == MARKET_RISK_OVERLAY_REALIZED_GARCH
    assert state.garch_market_variance == pytest.approx(rgarch.variance)
    assert state.garch_market_variance != pytest.approx(garch.variance, rel=1e-5, abs=1e-12)
    weights = optimize_asof(cfg, asof, persist=False, frame=frame)
    assert set(weights["market_risk_overlay"].to_list()) == {MARKET_RISK_OVERLAY_REALIZED_GARCH}
    assert float(weights["garch_market_variance"][0]) == pytest.approx(rgarch.variance)


def test_forecast_asof_rgarch_missing_ohlc_does_not_fall_back_to_garch(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _rgarch_panel()
    asof = frame["event_time"].unique().sort().to_list()[60]
    prior = frame.filter(pl.col("event_time") < asof)
    _asof_dates, values, measures = _realized_garch_history(prior)
    _save_rgarch(tmp_path, values, measures)
    _save_garch(tmp_path, values)
    stripped = frame.drop("high_split_adjusted", "low_split_adjusted")
    with pytest.raises(PointInTimeError, match="daily OHLC"):
        forecast_asof(cfg, asof, frame=stripped)


def test_forecast_asof_without_rgarch_keeps_return_only_garch(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _rgarch_panel()
    asof = frame["event_time"].unique().sort().to_list()[60]
    prior = frame.filter(pl.col("event_time") < asof)
    _asof_dates, values, _measures = _realized_garch_history(prior)
    _save_garch(tmp_path, values)
    garch = garch_market_forecast_asof(cfg, frame, asof)
    assert garch is not None
    state = forecast_asof(cfg, asof, frame=frame)
    assert state.market_risk_overlay == MARKET_RISK_OVERLAY_GARCH
    assert state.garch_market_variance == pytest.approx(garch.variance)
    assert "garch_market_cross_section" in state.notes
    assert "realized_garch_market_cross_section" not in state.notes
