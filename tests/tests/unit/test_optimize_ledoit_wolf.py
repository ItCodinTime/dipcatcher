"""Day Wave 139: default Ledoit-Wolf stays Ledoit-Wolf when T<=N."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from quant_fund.config import load_config
from quant_fund.models.covariance import (
    OPTIMIZER_COVARIANCE_HOMOSKEDASTIC_PROXY,
    OPTIMIZER_COVARIANCE_LEDOIT_WOLF,
    OPTIMIZER_COVARIANCE_OBJECT_TRAILING,
    OPTIMIZER_COVARIANCE_SAMPLE,
    OPTIMIZER_COVARIANCE_SPEC_LEDOIT_WOLF,
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


def _panel(n_days: int = 80, n_names: int = 3, seed: int = 13) -> pl.DataFrame:
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


def _fake_ledoit_wolf(returns: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
    x = np.asarray(returns, dtype=float)
    finite = x[np.isfinite(x).all(axis=1)]
    n = int(finite.shape[1]) if finite.ndim == 2 and finite.shape[1] else int(x.shape[1])
    sigma = np.eye(n, dtype=float) * 0.0004
    return sigma, {
        "family": OPTIMIZER_COVARIANCE_LEDOIT_WOLF,
        "spec": OPTIMIZER_COVARIANCE_SPEC_LEDOIT_WOLF,
        "covariance_object": OPTIMIZER_COVARIANCE_OBJECT_TRAILING,
        "sample": "listwise_complete",
        "shrinkage": 0.25,
        "n_obs": float(finite.shape[0]),
        "n_assets": float(n),
    }


def test_default_ledoit_wolf_path_returns_trailing_matrix_with_overlay(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    assert cfg.optimizer.covariance == "ledoit_wolf"
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[-1]
    ids = frame["security_id"].unique().sort().to_list()
    hist = frame.filter(pl.col("event_time") <= asof)
    seen: dict[str, np.ndarray] = {}

    def capture(returns: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
        seen["mat"] = np.asarray(returns, dtype=float).copy()
        return _fake_ledoit_wolf(returns)

    def scale_overlay(_config, _frame, _asof, sigma):
        scaled = np.asarray(sigma, dtype=float) * 4.0
        return scaled, None, None

    with (
        patch("quant_fund.pipeline.forecast.ledoit_wolf", side_effect=capture),
        patch(
            "quant_fund.pipeline.forecast.dcc_gaussian",
            side_effect=AssertionError("Ledoit-Wolf must not silently run dcc_gaussian"),
        ),
        patch(
            "quant_fund.pipeline.forecast.ewma",
            side_effect=AssertionError("Ledoit-Wolf must not silently run ewma"),
        ),
        patch(
            "quant_fund.pipeline.forecast.oas",
            side_effect=AssertionError("Ledoit-Wolf must not silently run OAS"),
        ),
        patch(
            "quant_fund.pipeline.forecast.sample",
            side_effect=AssertionError("Ledoit-Wolf must not silently run sample"),
        ),
        patch(
            "quant_fund.pipeline.forecast.apply_market_variance_overlay_to_covariance",
            side_effect=scale_overlay,
        ),
    ):
        estimate = estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)
    assert estimate.estimator == OPTIMIZER_COVARIANCE_LEDOIT_WOLF
    assert estimate.covariance_object == OPTIMIZER_COVARIANCE_OBJECT_TRAILING
    assert estimate.spec == OPTIMIZER_COVARIANCE_SPEC_LEDOIT_WOLF
    assert estimate.sigma == pytest.approx(np.eye(len(ids)) * 0.0016)
    assert "mat" in seen
    assert int(np.asarray(seen["mat"]).shape[0]) >= 2


def test_default_ledoit_wolf_unmeasured_on_short_history(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _panel(n_days=1)
    asof = frame["event_time"].unique().sort().to_list()[-1]
    ids = frame["security_id"].unique().sort().to_list()
    hist = frame.filter(pl.col("event_time") <= asof)
    with (
        patch(
            "quant_fund.pipeline.forecast.sample",
            side_effect=AssertionError("short Ledoit-Wolf must not switch to sample"),
        ),
        patch(
            "quant_fund.pipeline.forecast.ledoit_wolf",
            side_effect=AssertionError("short Ledoit-Wolf must stay unmeasured"),
        ),
    ):
        estimate = estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)
    assert estimate.estimator == OPTIMIZER_COVARIANCE_HOMOSKEDASTIC_PROXY
    assert estimate.unmeasured_reason == "insufficient_finite_rows"


def test_default_ledoit_wolf_keeps_ledoit_wolf_when_t_le_n(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _panel(n_days=5, n_names=8)
    asof = frame["event_time"].unique().sort().to_list()[-1]
    ids = frame["security_id"].unique().sort().to_list()
    hist = frame.filter(pl.col("event_time") <= asof)
    seen: dict[str, int] = {}

    def capture(returns: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
        mat = np.asarray(returns, dtype=float)
        seen["n_rows"] = int(mat.shape[0])
        seen["n_cols"] = int(mat.shape[1])
        return _fake_ledoit_wolf(returns)

    with (
        patch("quant_fund.pipeline.forecast.ledoit_wolf", side_effect=capture),
        patch(
            "quant_fund.pipeline.forecast.sample",
            side_effect=AssertionError("Ledoit-Wolf must not switch to sample when T<=N"),
        ),
        patch(
            "quant_fund.pipeline.forecast.oas",
            side_effect=AssertionError("Ledoit-Wolf must not switch to OAS when T<=N"),
        ),
        patch(
            "quant_fund.pipeline.forecast.apply_market_variance_overlay_to_covariance",
            side_effect=lambda _c, _f, _a, sigma: (np.asarray(sigma, dtype=float), None, None),
        ),
    ):
        estimate = estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)
    assert estimate.estimator == OPTIMIZER_COVARIANCE_LEDOIT_WOLF
    assert estimate.spec == OPTIMIZER_COVARIANCE_SPEC_LEDOIT_WOLF
    assert seen["n_rows"] <= seen["n_cols"]
    assert seen["n_rows"] >= 2


def test_default_ledoit_wolf_does_not_fall_back_to_sample(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[-1]
    ids = frame["security_id"].unique().sort().to_list()
    hist = frame.filter(pl.col("event_time") <= asof)

    def boom(returns: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
        raise ValueError("ledoit-wolf failed")

    with (
        patch("quant_fund.pipeline.forecast.ledoit_wolf", side_effect=boom),
        patch(
            "quant_fund.pipeline.forecast.sample",
            side_effect=AssertionError("failed Ledoit-Wolf must not run sample"),
        ),
        patch(
            "quant_fund.pipeline.forecast.oas",
            side_effect=AssertionError("failed Ledoit-Wolf must not run OAS"),
        ),
        patch(
            "quant_fund.pipeline.forecast.ewma",
            side_effect=AssertionError("failed Ledoit-Wolf must not run ewma"),
        ),
    ):
        estimate = estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)
    assert estimate.estimator == OPTIMIZER_COVARIANCE_HOMOSKEDASTIC_PROXY
    assert estimate.unmeasured_reason == "estimation_failed"


def test_default_ledoit_wolf_fails_closed_on_family_mismatch(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[-1]
    ids = frame["security_id"].unique().sort().to_list()
    hist = frame.filter(pl.col("event_time") <= asof)

    def sample_params(returns: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
        x = np.asarray(returns, dtype=float)
        n = int(x.shape[1])
        return np.eye(n) * 0.0004, {
            "family": OPTIMIZER_COVARIANCE_SAMPLE,
            "spec": "unbiased_sample",
            "covariance_object": OPTIMIZER_COVARIANCE_OBJECT_TRAILING,
            "n_obs": float(np.isfinite(x).all(axis=1).sum()),
            "n_assets": float(n),
        }

    with (
        patch("quant_fund.pipeline.forecast.ledoit_wolf", side_effect=sample_params),
        pytest.raises(ValueError, match="unexpected_family:sample"),
    ):
        estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)


def test_optimize_asof_default_ledoit_wolf_stamps_trailing_identity(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[-1]
    captured: dict[str, np.ndarray] = {}

    def capture_mv(alpha, sigma, w_prev, config, **kwargs):
        captured["sigma"] = np.asarray(sigma, dtype=float).copy()
        n = len(alpha)
        return np.zeros(n), type("D", (), {"feasible": True})()

    with (
        patch("quant_fund.pipeline.forecast.ledoit_wolf", side_effect=_fake_ledoit_wolf),
        patch(
            "quant_fund.pipeline.forecast.sample",
            side_effect=AssertionError("optimize_asof Ledoit-Wolf must not run sample"),
        ),
        patch("quant_fund.pipeline.forecast.optimize_mean_variance", side_effect=capture_mv),
        patch(
            "quant_fund.pipeline.forecast.apply_market_variance_overlay_to_covariance",
            side_effect=lambda _c, _f, _a, sigma: (np.asarray(sigma, dtype=float), None, None),
        ),
    ):
        weights = optimize_asof(cfg, asof, persist=False, frame=frame)
    assert set(weights["covariance_estimator"].to_list()) == {OPTIMIZER_COVARIANCE_LEDOIT_WOLF}
    assert set(weights["covariance_object"].to_list()) == {OPTIMIZER_COVARIANCE_OBJECT_TRAILING}
    assert set(weights["covariance_spec"].to_list()) == {OPTIMIZER_COVARIANCE_SPEC_LEDOIT_WOLF}
    assert "covariance_horizon" not in weights.columns
    n = captured["sigma"].shape[0]
    assert captured["sigma"] == pytest.approx(np.eye(n) * 0.0004)


def test_optimize_asof_default_ledoit_wolf_stays_when_t_le_n(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _panel(n_days=25, n_names=30)
    asof = frame["event_time"].unique().sort().to_list()[-1]
    seen: dict[str, int] = {}

    def capture(returns: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
        mat = np.asarray(returns, dtype=float)
        seen["n_rows"] = int(mat.shape[0])
        seen["n_cols"] = int(mat.shape[1])
        return _fake_ledoit_wolf(returns)

    def capture_mv(alpha, sigma, w_prev, config, **kwargs):
        n = len(alpha)
        return np.zeros(n), type("D", (), {"feasible": True})()

    with (
        patch("quant_fund.pipeline.forecast.ledoit_wolf", side_effect=capture),
        patch(
            "quant_fund.pipeline.forecast.sample",
            side_effect=AssertionError("optimize_asof Ledoit-Wolf must not switch to sample"),
        ),
        patch("quant_fund.pipeline.forecast.optimize_mean_variance", side_effect=capture_mv),
        patch(
            "quant_fund.pipeline.forecast.apply_market_variance_overlay_to_covariance",
            side_effect=lambda _c, _f, _a, sigma: (np.asarray(sigma, dtype=float), None, None),
        ),
    ):
        weights = optimize_asof(cfg, asof, persist=False, frame=frame)
    assert set(weights["covariance_estimator"].to_list()) == {OPTIMIZER_COVARIANCE_LEDOIT_WOLF}
    assert set(weights["covariance_spec"].to_list()) == {OPTIMIZER_COVARIANCE_SPEC_LEDOIT_WOLF}
    assert seen["n_rows"] <= seen["n_cols"]
    assert seen["n_rows"] >= 2
