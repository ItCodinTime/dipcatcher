"""Forecast conformal intervals. Coverage sets, not Sharpe."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quant_fund.config import load_config
from quant_fund.config.models import AppConfig
from quant_fund.metrics.cross_section import _date_keys
from quant_fund.pipeline.dataset import build_gold, panel
from quant_fund.pipeline.forecast import (
    conformal_sets_asof,
    forecast_asof,
    history_for_calibration,
)
from quant_fund.schemas.forecast import AssetForecast


def _asof() -> datetime:
    return datetime(2020, 6, 1, 16, 0, 0)


def test_schema_accepts_optional_intervals() -> None:
    bare = AssetForecast(
        security_id="S0",
        symbol="S0",
        asof=_asof(),
        model_version="fusion.v1",
    )
    assert bare.interval_lo == {}
    assert bare.interval_hi == {}
    assert bare.interval_alpha is None
    assert bare.interval_method is None
    filled = AssetForecast(
        security_id="S0",
        symbol="S0",
        asof=_asof(),
        model_version="fusion.v1",
        interval_lo={"5d": -0.04},
        interval_hi={"5d": 0.05},
        interval_alpha=0.10,
        interval_method="mondrian_cqr",
    )
    assert filled.interval_lo["5d"] < filled.interval_hi["5d"]
    dumped = filled.model_dump_json()
    assert "Sharpe" not in dumped
    assert "interval_lo" in dumped


def _toy_panel(n_days: int = 50, n_names: int = 8, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime(2019, 1, 2, 16, 0, 0)
    times = [start + timedelta(days=i) for i in range(n_days)]
    rows: list[dict[str, object]] = []
    for t_i, t in enumerate(times):
        for j in range(n_names):
            vol = 0.012 + 0.03 * (j / max(n_names - 1, 1)) + 0.01 * (t_i > n_days // 2)
            ret = float(rng.normal(0.0, vol))
            rows.append(
                {
                    "event_time": t,
                    "security_id": f"S{j:02d}",
                    "symbol": f"S{j:02d}",
                    "ret_1": ret,
                    "vol_20": vol,
                    "mom_20": ret * 2.0,
                    "future_log_return_5": float(rng.normal(0.0, vol * np.sqrt(5.0))),
                }
            )
    return pl.DataFrame(rows)


def test_toy_panel_finite_lo_lt_hi_default_alpha() -> None:
    df = _toy_panel()
    asof = df["event_time"].max()
    assert isinstance(asof, datetime)
    bundle = conformal_sets_asof(df, asof, AppConfig())
    assert bundle is not None
    assert bundle.alpha == 0.10
    assert bundle.method == "mondrian_cqr"
    assert bundle.lower
    for sid, lo in bundle.lower.items():
        hi = bundle.upper[sid]
        assert np.isfinite(lo) and np.isfinite(hi)
        assert lo < hi


def test_split_cqr_when_vol_20_missing() -> None:
    df = _toy_panel().drop("vol_20")
    asof = df["event_time"].max()
    assert isinstance(asof, datetime)
    bundle = conformal_sets_asof(df, asof, AppConfig())
    assert bundle is not None
    assert bundle.method == "split_cqr"
    assert bundle.alpha == 0.10


def test_history_and_sets_exclude_decision_bar_y() -> None:
    df = _toy_panel()
    asof = df["event_time"].max()
    assert isinstance(asof, datetime)
    hist = history_for_calibration(df, asof, horizon_bars=5)
    assert hist.filter(pl.col("event_time") >= asof).is_empty()
    poisoned = df.with_columns(
        pl.when(pl.col("event_time") == asof)
        .then(pl.lit(1.0e6))
        .otherwise(pl.col("future_log_return_5"))
        .alias("future_log_return_5")
    )
    clean = conformal_sets_asof(df, asof, AppConfig())
    dirty = conformal_sets_asof(poisoned, asof, AppConfig())
    assert clean is not None and dirty is not None
    asof_key = _date_keys(np.array([asof]))[0]
    assert asof_key not in _date_keys(list(clean.cal_event_times))
    for sid in clean.lower:
        assert clean.lower[sid] == pytest.approx(dirty.lower[sid])
        assert clean.upper[sid] == pytest.approx(dirty.upper[sid])


@pytest.mark.synthetic
def test_forecast_asof_attaches_intervals(tmp_path: Path) -> None:
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.data.synthetic_n_assets = 8
    cfg.data.synthetic_n_days = 90
    build_gold(cfg)
    state = forecast_asof(cfg)
    assert "SYNTHETIC" in state.notes
    assert state.forecasts
    row = state.forecasts[0]
    assert row.interval_alpha == 0.10
    assert row.interval_method in {"mondrian_cqr", "split_cqr"}
    assert row.interval_lo
    hz = next(iter(row.interval_lo))
    lo, hi = row.interval_lo[hz], row.interval_hi[hz]
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo < hi
    dumped = state.model_dump_json()
    assert "Sharpe" not in dumped
    assert "sharpe" not in dumped.lower()
    # Gold panel still has the decision-bar label in synthetic history; sets must ignore it.
    df = panel(cfg)
    asof = state.asof
    poisoned = df.with_columns(
        pl.when(pl.col("event_time") == asof)
        .then(pl.lit(1.0e6))
        .otherwise(pl.col(cfg.train.distribution_target))
        .alias(cfg.train.distribution_target)
    )
    a = conformal_sets_asof(df, asof, cfg)
    b = conformal_sets_asof(poisoned, asof, cfg)
    assert a is not None and b is not None
    for sid in a.lower:
        assert a.lower[sid] == pytest.approx(b.lower[sid])
        assert a.upper[sid] == pytest.approx(b.upper[sid])
