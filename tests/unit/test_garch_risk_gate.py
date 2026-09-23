"""Causal GARCH market overlay for paper/backtest check_order."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quant_fund.backtest.engine import run_backtest
from quant_fund.config import load_config
from quant_fund.data.lake import Lake
from quant_fund.models.volatility import GARCH_DATE_LEVEL_SCOPE, GARCHVol
from quant_fund.paper.loop import run_paper_loop
from quant_fund.pipeline.forecast import clear_forecast_caches


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
    seed: int = 11,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime(2020, 1, 2, 16, 0, 0)
    rows: list[dict[str, object]] = []
    for day in range(n_days):
        stamp = start + timedelta(days=day)
        mkt = float(rng.normal(0.0, ret_scale))
        for name, px0 in (("A", 100.0), ("B", 50.0)):
            rows.append(
                {
                    "security_id": name,
                    "event_time": stamp,
                    "open": px0,
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


def _save_garch(
    tmp_path: Path,
    *,
    series_scope: str = GARCH_DATE_LEVEL_SCOPE,
) -> Path:
    path = tmp_path / "metadata" / "vol_garch.joblib"
    GARCHVol(series_scope=series_scope, min_obs=20).fit_returns(np.full(80, 0.01)).save(path)
    return path


@pytest.fixture(autouse=True)
def _clear_garch_caches() -> None:
    clear_forecast_caches()
    yield
    clear_forecast_caches()


def test_backtest_without_garch_artifact_still_gates_on_name_vol(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.risk_gate.max_predicted_vol = 0.05
    frame = _panel(name_vol=0.2, ret_scale=0.01)
    result = run_backtest(frame, _late_weights(frame), cfg, initial_nav=100_000.0)
    assert result.fills.height == 0
    assert int(result.metrics["risk_gate_rejects"]) >= 1
    assert int(result.metrics["garch_risk_overlay_dates"]) == 0


def test_backtest_garch_overlay_admits_high_name_vol(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.risk_gate.max_predicted_vol = 0.05
    frame = _panel(name_vol=0.2, ret_scale=0.01)
    _save_garch(tmp_path)
    result = run_backtest(frame, _late_weights(frame), cfg, initial_nav=100_000.0)
    assert result.fills.height >= 1
    assert int(result.metrics["garch_risk_overlay_dates"]) >= 1
    assert result.metrics["live_pnl_claim"] is False


def test_backtest_garch_overlay_rejects_high_market_vol(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.risk_gate.max_predicted_vol = 0.03
    frame = _panel(name_vol=0.01, ret_scale=0.08)
    without = run_backtest(frame, _late_weights(frame), cfg, initial_nav=100_000.0)
    assert without.fills.height >= 1
    _save_garch(tmp_path)
    clear_forecast_caches()
    with_overlay = run_backtest(frame, _late_weights(frame), cfg, initial_nav=100_000.0)
    assert with_overlay.fills.height == 0
    assert int(with_overlay.metrics["risk_gate_rejects"]) >= 1
    assert int(with_overlay.metrics["garch_risk_overlay_dates"]) >= 1


def test_backtest_garch_overlay_ignores_returns_on_or_after_origin(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.risk_gate.max_predicted_vol = 0.05
    frame = _panel(name_vol=0.2, ret_scale=0.01)
    dates = frame["event_time"].unique().sort().to_list()
    asof = _late_weights(frame)["event_time"][0]
    poisoned = frame.with_columns(
        pl.when(pl.col("event_time") >= asof)
        .then(pl.lit(0.5))
        .otherwise(pl.col("ret_1"))
        .alias("ret_1")
    )
    _save_garch(tmp_path)
    clean = run_backtest(frame, _late_weights(frame), cfg, initial_nav=100_000.0)
    clear_forecast_caches()
    dirty = run_backtest(poisoned, _late_weights(frame), cfg, initial_nav=100_000.0)
    # Later flatten dates may see post-origin returns under their own asof.
    # The decision origin itself must ignore ret_1 on or after that stamp.
    clean_origin = clean.fills.filter(pl.col("signal_time") == asof)
    dirty_origin = dirty.fills.filter(pl.col("signal_time") == asof)
    assert clean_origin.height == dirty_origin.height >= 1
    assert dates[0] < asof


def test_backtest_wrong_series_scope_fails_closed(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _panel()
    _save_garch(tmp_path, series_scope="univariate_return_series")
    with pytest.raises(ValueError, match="series_scope"):
        run_backtest(frame, _late_weights(frame), cfg, initial_nav=100_000.0)


def test_backtest_artifact_without_ret_1_fails_closed(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _panel().drop("ret_1")
    _save_garch(tmp_path)
    with pytest.raises(ValueError, match="ret_1"):
        run_backtest(frame, _late_weights(frame), cfg, initial_nav=100_000.0)


def test_backtest_garch_overlay_ignores_non_member_returns(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.risk_gate.max_predicted_vol = 0.03
    frame = _panel(name_vol=0.01, ret_scale=0.01)
    day_parity = pl.col("event_time").rank("dense").cast(pl.Int64) % 2
    shocked = frame.with_columns(
        pl.when(pl.col("security_id") == "B")
        .then(pl.when(day_parity == 0).then(pl.lit(0.25)).otherwise(pl.lit(-0.25)))
        .otherwise(pl.col("ret_1"))
        .alias("ret_1")
    )
    weights = _late_weights(shocked)
    _save_garch(tmp_path)
    without = run_backtest(shocked, weights, cfg, initial_nav=100_000.0)
    assert without.fills.height == 0
    assert int(without.metrics["risk_gate_rejects"]) >= 1
    membership = (
        shocked.filter(pl.col("security_id") == "A")
        .select(pl.col("security_id"), pl.col("event_time").alias("asof"))
        .unique()
    )
    Lake(tmp_path).write_parquet(membership, "silver/universe.parquet")
    clear_forecast_caches()
    with_universe = run_backtest(shocked, weights, cfg, initial_nav=100_000.0)
    assert with_universe.fills.height >= 1
    assert int(with_universe.metrics["garch_risk_overlay_dates"]) >= 1


def test_paper_loop_garch_overlay_admits_high_name_vol(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, paper=True)
    cfg.risk_gate.max_predicted_vol = 0.05
    frame = _panel(n_days=28, name_vol=0.2, ret_scale=0.01)
    weights = _late_weights(frame)
    without = run_paper_loop(
        frame,
        cfg,
        champion_weights=weights,
        initial_nav=100_000.0,
        prefer_latest=False,
        run_id="paper-garch-name",
    )
    assert without.metrics["n_fills"] == 0
    assert int(without.metrics["risk_gate_rejects"]) >= 1
    _save_garch(tmp_path)
    clear_forecast_caches()
    with_overlay = run_paper_loop(
        frame,
        cfg,
        champion_weights=weights,
        initial_nav=100_000.0,
        prefer_latest=False,
        run_id="paper-garch-overlay",
    )
    assert with_overlay.metrics["n_fills"] >= 1
    assert int(with_overlay.metrics["garch_risk_overlay_dates"]) >= 1
    assert with_overlay.metrics["live_pnl_claim"] is False
