"""Causal Realized GARCH Parkinson overlay for paper/backtest check_order."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quant_fund.backtest.engine import run_backtest
from quant_fund.config import load_config
from quant_fund.models.realized_garch import RealizedGARCHVol
from quant_fund.models.volatility import GARCH_DATE_LEVEL_SCOPE, GARCHVol
from quant_fund.paper.loop import run_paper_loop
from quant_fund.pipeline.forecast import (
    MARKET_RISK_OVERLAY_GARCH,
    MARKET_RISK_OVERLAY_REALIZED_GARCH,
    clear_forecast_caches,
    market_risk_overlay_asof,
)
from quant_fund.schemas.errors import PointInTimeError


def _cfg(tmp_path: Path, *, paper: bool = False):
    cfg = load_config("configs/paper.yaml" if paper else "configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.data.source = "synthetic"
    cfg.costs.frictionless = True
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_net = 1.0
    cfg.risk_gate.max_gross = 2.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.max_participation = 1.0
    return cfg


def _panel(
    *,
    n_days: int = 36,
    ret_scale: float = 0.01,
    name_vol: float = 0.02,
    hl_span: float = 0.006,
    seed: int = 11,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime(2020, 1, 2, 16, 0, 0)
    rows: list[dict[str, object]] = []
    for day in range(n_days):
        stamp = start + timedelta(days=day)
        mkt = float(rng.normal(0.0, ret_scale))
        for name, px0 in (("A", 100.0), ("B", 50.0)):
            high = px0 * float(np.exp(hl_span))
            low = px0 * float(np.exp(-hl_span))
            rows.append(
                {
                    "security_id": name,
                    "event_time": stamp,
                    "open": px0,
                    "high": high,
                    "low": low,
                    "close": px0,
                    "close_total_return": px0,
                    "volume": 1_000_000.0,
                    "adv": 100_000_000.0,
                    "vol_20": name_vol,
                    "ret_1": mkt,
                    "source": "synthetic",
                }
            )
    return pl.DataFrame(rows)


def _late_weights(frame: pl.DataFrame, *, scale: float = 0.2) -> pl.DataFrame:
    dates = frame["event_time"].unique().sort().to_list()
    asof = dates[len(dates) // 2 + 8]
    return pl.DataFrame(
        {
            "event_time": [asof, asof],
            "security_id": ["A", "B"],
            "target_weight": [scale, scale],
        }
    )


def _save_rgarch(
    tmp_path: Path,
    *,
    series_scope: str = GARCH_DATE_LEVEL_SCOPE,
    ret_scale: float = 0.01,
    park: float = 4.0e-5,
) -> Path:
    path = tmp_path / "metadata" / "vol_realized_garch.joblib"
    returns = np.full(80, ret_scale)
    measure = np.full(80, park)
    RealizedGARCHVol(series_scope=series_scope, min_obs=20).fit_returns(returns, measure).save(path)
    return path


def _save_garch(tmp_path: Path) -> Path:
    path = tmp_path / "metadata" / "vol_garch.joblib"
    GARCHVol(series_scope=GARCH_DATE_LEVEL_SCOPE, min_obs=20).fit_returns(np.full(80, 0.01)).save(
        path
    )
    return path


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    clear_forecast_caches()
    yield
    clear_forecast_caches()


def test_backtest_without_rgarch_artifact_still_gates_on_name_vol(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.risk_gate.max_predicted_vol = 0.05
    frame = _panel(name_vol=0.2, ret_scale=0.01, hl_span=0.006)
    result = run_backtest(frame, _late_weights(frame), cfg, initial_nav=100_000.0)
    assert result.fills.height == 0
    assert int(result.metrics["risk_gate_rejects"]) >= 1
    assert int(result.metrics["garch_risk_overlay_dates"]) == 0
    assert int(result.metrics["realized_garch_risk_overlay_dates"]) == 0


def test_backtest_rgarch_overlay_admits_high_name_vol(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.risk_gate.max_predicted_vol = 0.05
    frame = _panel(name_vol=0.2, ret_scale=0.01, hl_span=0.006)
    _save_rgarch(tmp_path)
    result = run_backtest(frame, _late_weights(frame), cfg, initial_nav=100_000.0)
    assert result.fills.height >= 1
    assert int(result.metrics["realized_garch_risk_overlay_dates"]) >= 1
    assert int(result.metrics["garch_risk_overlay_dates"]) == 0
    assert result.metrics["live_pnl_claim"] is False


def test_backtest_rgarch_overlay_rejects_high_parkinson_vol(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    # The gate consumes the causal one-step latent RGARCH sigma.  A high
    # observed Parkinson measure can reject some, but not necessarily every,
    # later latent forecast.
    cfg.risk_gate.max_predicted_vol = 0.03
    frame = _panel(name_vol=0.01, ret_scale=0.01, hl_span=0.22)
    without = run_backtest(frame, _late_weights(frame), cfg, initial_nav=100_000.0)
    assert without.fills.height >= 1
    _save_rgarch(tmp_path)
    clear_forecast_caches()
    with_overlay = run_backtest(frame, _late_weights(frame), cfg, initial_nav=100_000.0)
    assert with_overlay.fills.height <= without.fills.height
    assert with_overlay.fills.height >= 1
    assert int(with_overlay.metrics["risk_gate_rejects"]) >= 1
    assert (
        with_overlay.fills["signal_time"].unique().to_list()
        != without.fills["signal_time"].unique().to_list()
    )
    assert int(with_overlay.metrics["realized_garch_risk_overlay_dates"]) >= 1
    assert int(with_overlay.metrics["garch_risk_overlay_dates"]) == 0


def test_backtest_rgarch_overlay_ignores_ohlc_on_or_after_origin(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.risk_gate.max_predicted_vol = 0.05
    frame = _panel(name_vol=0.2, ret_scale=0.01, hl_span=0.006)
    asof = _late_weights(frame)["event_time"][0]
    poisoned = frame.with_columns(
        pl.when(pl.col("event_time") >= asof)
        .then(pl.col("close") * float(np.exp(0.5)))
        .otherwise(pl.col("high"))
        .alias("high"),
        pl.when(pl.col("event_time") >= asof)
        .then(pl.col("close") * float(np.exp(-0.5)))
        .otherwise(pl.col("low"))
        .alias("low"),
    )
    _save_rgarch(tmp_path)
    clean = run_backtest(frame, _late_weights(frame), cfg, initial_nav=100_000.0)
    clear_forecast_caches()
    dirty = run_backtest(poisoned, _late_weights(frame), cfg, initial_nav=100_000.0)
    clean_origin = clean.fills.filter(pl.col("signal_time") == asof)
    dirty_origin = dirty.fills.filter(pl.col("signal_time") == asof)
    assert clean_origin.height == dirty_origin.height >= 1


def test_backtest_rgarch_wrong_series_scope_fails_closed(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _panel()
    _save_rgarch(tmp_path, series_scope="univariate_return_series")
    with pytest.raises(ValueError, match="series_scope"):
        run_backtest(frame, _late_weights(frame), cfg, initial_nav=100_000.0)


def test_backtest_rgarch_artifact_without_ohlc_fails_closed(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _panel().drop("high", "low")
    _save_rgarch(tmp_path)
    with pytest.raises(PointInTimeError, match="daily OHLC"):
        run_backtest(frame, _late_weights(frame), cfg, initial_nav=100_000.0)


def test_backtest_rgarch_does_not_fall_back_to_garch_when_ohlc_missing(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    cfg.risk_gate.max_predicted_vol = 0.05
    frame = _panel(name_vol=0.2, ret_scale=0.01, hl_span=0.006).drop("high", "low")
    _save_garch(tmp_path)
    _save_rgarch(tmp_path)
    with pytest.raises(PointInTimeError, match="daily OHLC"):
        run_backtest(frame, _late_weights(frame), cfg, initial_nav=100_000.0)


def test_backtest_prefers_rgarch_over_return_only_garch(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    # Both artifacts are available; this threshold exposes different causal
    # one-step paths while keeping the assertion about RGARCH precedence clear.
    cfg.risk_gate.max_predicted_vol = 0.03
    frame = _panel(name_vol=0.01, ret_scale=0.01, hl_span=0.22)
    _save_garch(tmp_path)
    garch_only = run_backtest(frame, _late_weights(frame), cfg, initial_nav=100_000.0)
    assert garch_only.fills.height >= 1
    assert int(garch_only.metrics["garch_risk_overlay_dates"]) >= 1
    assert int(garch_only.metrics["realized_garch_risk_overlay_dates"]) == 0
    _save_rgarch(tmp_path)
    clear_forecast_caches()
    both = run_backtest(frame, _late_weights(frame), cfg, initial_nav=100_000.0)
    assert both.fills.height <= garch_only.fills.height
    assert both.fills.height >= 1
    assert int(both.metrics["risk_gate_rejects"]) >= 1
    assert (
        both.fills["signal_time"].unique().to_list()
        != garch_only.fills["signal_time"].unique().to_list()
    )
    assert int(both.metrics["realized_garch_risk_overlay_dates"]) >= 1
    assert int(both.metrics["garch_risk_overlay_dates"]) == 0


def test_market_risk_overlay_asof_source_stamps(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _panel()
    asof = _late_weights(frame)["event_time"][0]
    assert market_risk_overlay_asof(cfg, frame, asof) == (None, None)
    _save_garch(tmp_path)
    sigma, source = market_risk_overlay_asof(cfg, frame, asof)
    assert sigma is not None and sigma > 0.0
    assert source == MARKET_RISK_OVERLAY_GARCH
    _save_rgarch(tmp_path)
    clear_forecast_caches()
    sigma, source = market_risk_overlay_asof(cfg, frame, asof)
    assert sigma is not None and sigma > 0.0
    assert source == MARKET_RISK_OVERLAY_REALIZED_GARCH


def test_paper_loop_rgarch_overlay_admits_high_name_vol(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, paper=True)
    cfg.risk_gate.max_predicted_vol = 0.05
    frame = _panel(n_days=28, name_vol=0.2, ret_scale=0.01, hl_span=0.006)
    weights = _late_weights(frame)
    without = run_paper_loop(
        frame,
        cfg,
        champion_weights=weights,
        initial_nav=100_000.0,
        prefer_latest=False,
        run_id="paper-rgarch-name",
    )
    assert without.metrics["n_fills"] == 0
    assert int(without.metrics["risk_gate_rejects"]) >= 1
    _save_rgarch(tmp_path)
    clear_forecast_caches()
    with_overlay = run_paper_loop(
        frame,
        cfg,
        champion_weights=weights,
        initial_nav=100_000.0,
        prefer_latest=False,
        run_id="paper-rgarch-overlay",
    )
    assert with_overlay.metrics["n_fills"] >= 1
    assert int(with_overlay.metrics["realized_garch_risk_overlay_dates"]) >= 1
    assert int(with_overlay.metrics["garch_risk_overlay_dates"]) == 0
    assert with_overlay.metrics["live_pnl_claim"] is False
