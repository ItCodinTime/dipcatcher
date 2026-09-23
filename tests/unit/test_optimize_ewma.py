"""Day Wave 130: named RiskMetrics EWMA path for optimize_asof / trailing covariance."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from quant_fund.config import load_config
from quant_fund.models.covariance import (
    DCC_COVARIANCE_OBJECT_ONE_STEP,
    DCC_FAMILY_GAUSSIAN,
    EWMA_MIN_OBS,
    EWMA_SPEC_RISKMETRICS,
    OPTIMIZER_COVARIANCE_EWMA,
)
from quant_fund.pipeline.forecast import (
    clear_forecast_caches,
    estimate_optimizer_covariance_asof,
    optimize_asof,
)


def _cfg(tmp_path: Path):
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.data.source = "synthetic"
    cfg.fusion.skip_intervals = True
    cfg.fusion.apply_interval_caps = False
    return cfg


def _panel(n_days: int = 80, n_names: int = 3, seed: int = 11) -> pl.DataFrame:
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


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    clear_forecast_caches()
    yield
    clear_forecast_caches()


def _fake_ewma(returns: np.ndarray, lam: float = 0.94) -> tuple[np.ndarray, dict[str, float | str]]:
    x = np.asarray(returns, dtype=float)
    n = int(x.shape[1])
    sigma = np.eye(n, dtype=float) * 0.0009
    finite = int(np.isfinite(x).all(axis=1).sum())
    return sigma, {
        "family": OPTIMIZER_COVARIANCE_EWMA,
        "spec": EWMA_SPEC_RISKMETRICS,
        "lambda": float(lam),
        "covariance_object": DCC_COVARIANCE_OBJECT_ONE_STEP,
        "n_obs": float(finite),
        "n_assets": float(n),
    }


def test_named_ewma_path_returns_one_step_matrix_without_overlay(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "ewma"
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[-1]
    ids = frame["security_id"].unique().sort().to_list()
    hist = frame.filter(pl.col("event_time") <= asof)
    seen: dict[str, np.ndarray | float] = {}

    def capture(
        returns: np.ndarray, lam: float = 0.94
    ) -> tuple[np.ndarray, dict[str, float | str]]:
        seen["mat"] = np.asarray(returns, dtype=float).copy()
        seen["lam"] = float(lam)
        return _fake_ewma(returns, lam=lam)

    with (
        patch("quant_fund.pipeline.forecast.ewma", side_effect=capture),
        patch(
            "quant_fund.pipeline.forecast.dcc_gaussian",
            side_effect=AssertionError("EWMA must not silently run dcc_gaussian"),
        ),
        patch(
            "quant_fund.pipeline.forecast.dcc_student_t",
            side_effect=AssertionError("EWMA must not silently run dcc_student_t"),
        ),
        patch(
            "quant_fund.pipeline.forecast.adcc",
            side_effect=AssertionError("EWMA must not silently run adcc"),
        ),
        patch(
            "quant_fund.pipeline.forecast.apply_market_variance_overlay_to_covariance",
            side_effect=AssertionError("EWMA H_{t+1} must not be GARCH-overlaid"),
        ),
    ):
        estimate = estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)
    assert estimate.estimator == OPTIMIZER_COVARIANCE_EWMA
    assert estimate.covariance_object == DCC_COVARIANCE_OBJECT_ONE_STEP
    assert estimate.spec == EWMA_SPEC_RISKMETRICS
    assert estimate.market_overlay is None
    assert estimate.overlay is None
    assert estimate.sigma == pytest.approx(np.eye(len(ids)) * 0.0009)
    assert "mat" in seen
    assert int(np.asarray(seen["mat"]).shape[0]) >= EWMA_MIN_OBS
    assert float(seen["lam"]) == pytest.approx(float(cfg.features.ewma_lambda))


def test_named_ewma_fails_closed_on_short_history(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "ewma"
    frame = _panel(n_days=EWMA_MIN_OBS - 1)
    asof = frame["event_time"].unique().sort().to_list()[-1]
    ids = frame["security_id"].unique().sort().to_list()
    hist = frame.filter(pl.col("event_time") <= asof)
    with pytest.raises(ValueError, match="optimizer_covariance_failed:ewma"):
        estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)


def test_named_ewma_does_not_fall_back_to_ledoit_wolf_or_dcc(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "ewma"
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[-1]
    ids = frame["security_id"].unique().sort().to_list()
    hist = frame.filter(pl.col("event_time") <= asof)

    def boom(returns: np.ndarray, lam: float = 0.94) -> tuple[np.ndarray, dict[str, float | str]]:
        raise ValueError("ewma failed")

    with (
        patch("quant_fund.pipeline.forecast.ewma", side_effect=boom),
        patch(
            "quant_fund.pipeline.forecast.dcc_gaussian",
            side_effect=AssertionError("failed EWMA must not run dcc_gaussian"),
        ),
        patch(
            "quant_fund.pipeline.forecast.ledoit_wolf",
            side_effect=AssertionError("failed EWMA must not run Ledoit-Wolf"),
        ),
        pytest.raises(ValueError, match="optimizer_covariance_failed:ewma"),
    ):
        estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)


def test_named_ewma_fails_closed_on_family_mismatch(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "ewma"
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[-1]
    ids = frame["security_id"].unique().sort().to_list()
    hist = frame.filter(pl.col("event_time") <= asof)

    def gaussian_params(
        returns: np.ndarray, lam: float = 0.94
    ) -> tuple[np.ndarray, dict[str, float | str]]:
        x = np.asarray(returns, dtype=float)
        n = int(x.shape[1])
        return np.eye(n) * 0.0004, {
            "family": DCC_FAMILY_GAUSSIAN,
            "spec": "engle_2002_gaussian_dcc",
            "covariance_object": DCC_COVARIANCE_OBJECT_ONE_STEP,
            "n_obs": float(np.isfinite(x).all(axis=1).sum()),
            "n_assets": float(n),
        }

    with (
        patch("quant_fund.pipeline.forecast.ewma", side_effect=gaussian_params),
        pytest.raises(ValueError, match="unexpected_family:dcc_gaussian"),
    ):
        estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)


def test_optimize_asof_named_ewma_stamps_identity_not_overlay(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "ewma"
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[-1]
    captured: dict[str, np.ndarray] = {}

    def capture_mv(alpha, sigma, w_prev, config, **kwargs):
        captured["sigma"] = np.asarray(sigma, dtype=float).copy()
        n = len(alpha)
        return np.zeros(n), type("D", (), {"feasible": True})()

    with (
        patch("quant_fund.pipeline.forecast.ewma", side_effect=_fake_ewma),
        patch(
            "quant_fund.pipeline.forecast.dcc_gaussian",
            side_effect=AssertionError("optimize_asof EWMA must not run dcc_gaussian"),
        ),
        patch("quant_fund.pipeline.forecast.optimize_mean_variance", side_effect=capture_mv),
        patch(
            "quant_fund.pipeline.forecast.apply_market_variance_overlay_to_covariance",
            side_effect=AssertionError("named EWMA must not overlay H_{t+1}"),
        ),
    ):
        weights = optimize_asof(cfg, asof, persist=False, frame=frame)
    assert set(weights["covariance_estimator"].to_list()) == {OPTIMIZER_COVARIANCE_EWMA}
    assert set(weights["covariance_object"].to_list()) == {DCC_COVARIANCE_OBJECT_ONE_STEP}
    assert set(weights["covariance_spec"].to_list()) == {EWMA_SPEC_RISKMETRICS}
    assert set(weights["covariance_horizon"].to_list()) == {1}
    assert "market_risk_overlay" not in weights.columns
    assert "garch_market_sigma" not in weights.columns
    n = captured["sigma"].shape[0]
    assert captured["sigma"] == pytest.approx(np.eye(n) * 0.0009)


def test_named_ewma_fails_closed_on_incomplete_asof_row(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "ewma"
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[-1]
    frame = frame.with_columns(
        pl.when((pl.col("security_id") == "S00") & (pl.col("event_time") == asof))
        .then(None)
        .otherwise(pl.col("ret_1"))
        .alias("ret_1")
    )
    ids = frame["security_id"].unique().sort().to_list()
    hist = frame.filter(pl.col("event_time") <= asof)
    with (
        patch(
            "quant_fund.pipeline.forecast.ewma",
            side_effect=AssertionError("incomplete asof must not run EWMA"),
        ),
        pytest.raises(ValueError, match="incomplete_terminal_row"),
    ):
        estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)
