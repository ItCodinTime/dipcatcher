"""Day Wave 124: named Gaussian DCC path for optimize_asof / trailing covariance.

Day Wave 125: DCC trailing complete-window honesty on the optimize_asof path.
Day Wave 127: named Student-t DCC path; must not silently run Gaussian DCC.
Day Wave 129: named scalar ADCC path; must not silently run Engle DCC.
Day Wave 134: named Bollerslev CCC path; must not silently run Engle DCC.
Day Wave 136: named diagonal CES AG-DCC path; must not silently run scalar ADCC.
Day Wave 138: named unrestricted CES AG-DCC path; must not silently run diagonal AG-DCC.
"""

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
    DCC_FAMILY_ADCC,
    DCC_FAMILY_AGDCC,
    DCC_FAMILY_AGDCC_FULL,
    DCC_FAMILY_CCC,
    DCC_FAMILY_GAUSSIAN,
    DCC_FAMILY_STUDENT_T,
    DCC_SPEC_ADCC,
    DCC_SPEC_AGDCC,
    DCC_SPEC_AGDCC_FULL,
    DCC_SPEC_CCC,
    DCC_SPEC_ENGLE_2002,
    DCC_SPEC_STUDENT_T,
    DCC_STAGE1_MIN_OBS,
    OPTIMIZER_COVARIANCE_LEDOIT_WOLF,
    OPTIMIZER_COVARIANCE_OBJECT_TRAILING,
    OPTIMIZER_COVARIANCE_SPEC_LEDOIT_WOLF,
)
from quant_fund.models.volatility import GARCH_DATE_LEVEL_SCOPE, GARCHVol
from quant_fund.pipeline.forecast import (
    clear_forecast_caches,
    estimate_optimizer_covariance_asof,
    optimize_asof,
    overlay_covariance_with_garch_market,
)
from quant_fund.pipeline.train import _garch_return_history


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


def _save_garch(tmp_path: Path, returns: np.ndarray) -> Path:
    path = tmp_path / "metadata" / "vol_garch.joblib"
    GARCHVol(series_scope=GARCH_DATE_LEVEL_SCOPE, min_obs=20).fit_returns(returns).save(path)
    return path


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    clear_forecast_caches()
    yield
    clear_forecast_caches()


def _fake_dcc(returns: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
    x = np.asarray(returns, dtype=float)
    n = int(x.shape[1])
    sigma = np.eye(n, dtype=float) * 0.0004
    finite = int(np.isfinite(x).all(axis=1).sum())
    return sigma, {
        "a": 0.05,
        "b": 0.9,
        "family": DCC_FAMILY_GAUSSIAN,
        "spec": DCC_SPEC_ENGLE_2002,
        "covariance_object": DCC_COVARIANCE_OBJECT_ONE_STEP,
        "n_obs": float(finite),
        "n_assets": float(n),
    }


def test_default_optimizer_covariance_is_trailing_ledoit_wolf(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[-1]
    ids = frame["security_id"].unique().sort().to_list()
    hist = frame.filter(pl.col("event_time") <= asof)
    estimate = estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)
    assert estimate.estimator == OPTIMIZER_COVARIANCE_LEDOIT_WOLF
    assert estimate.covariance_object == OPTIMIZER_COVARIANCE_OBJECT_TRAILING
    assert estimate.spec == OPTIMIZER_COVARIANCE_SPEC_LEDOIT_WOLF
    assert estimate.unmeasured_reason is None
    assert estimate.sigma.shape == (len(ids), len(ids))


def test_named_dcc_path_returns_one_step_matrix_without_overlay(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "dcc_gaussian"
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[-1]
    ids = frame["security_id"].unique().sort().to_list()
    hist = frame.filter(pl.col("event_time") <= asof)
    _dates, values = _garch_return_history(frame.filter(pl.col("event_time") < asof))
    _save_garch(tmp_path, values)
    seen: dict[str, np.ndarray] = {}

    def capture(returns: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
        seen["mat"] = np.asarray(returns, dtype=float).copy()
        return _fake_dcc(returns)

    with (
        patch("quant_fund.pipeline.forecast.dcc_gaussian", side_effect=capture),
        patch(
            "quant_fund.pipeline.forecast.apply_market_variance_overlay_to_covariance",
            side_effect=AssertionError("DCC H_{t+1} must not be GARCH-overlaid"),
        ),
    ):
        estimate = estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)
    assert estimate.estimator == DCC_FAMILY_GAUSSIAN
    assert estimate.covariance_object == DCC_COVARIANCE_OBJECT_ONE_STEP
    assert estimate.spec == DCC_SPEC_ENGLE_2002
    assert estimate.market_overlay is None
    assert estimate.overlay is None
    assert estimate.sigma == pytest.approx(np.eye(len(ids)) * 0.0004)
    assert "mat" in seen
    assert seen["mat"].shape[0] >= DCC_STAGE1_MIN_OBS


def test_named_dcc_path_fails_closed_on_short_history(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "dcc_gaussian"
    frame = _panel(n_days=20)
    asof = frame["event_time"].unique().sort().to_list()[-1]
    ids = frame["security_id"].unique().sort().to_list()
    hist = frame.filter(pl.col("event_time") <= asof)
    with pytest.raises(ValueError, match="insufficient_finite_rows"):
        estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)


def test_named_dcc_path_does_not_fall_back_to_ledoit_wolf(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "dcc_gaussian"
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[-1]
    ids = frame["security_id"].unique().sort().to_list()
    hist = frame.filter(pl.col("event_time") <= asof)

    def boom(returns: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
        raise ValueError("stage-1 failed")

    with (
        patch("quant_fund.pipeline.forecast.dcc_gaussian", side_effect=boom),
        pytest.raises(ValueError, match="optimizer_covariance_failed:dcc_gaussian"),
    ):
        estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)


def test_named_dcc_ignores_unpublished_return_restatement(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "dcc_gaussian"
    frame = _panel()
    dates = frame["event_time"].unique().sort().to_list()
    asof = dates[-1]
    early = dates[10]
    late = asof + timedelta(days=1)
    unpublished = frame.with_columns(
        pl.col("event_time").alias("available_time"),
        pl.when((pl.col("security_id") == "S00") & (pl.col("event_time") == early))
        .then(pl.lit(8.0))
        .otherwise(pl.col("ret_1"))
        .alias("ret_1"),
    )
    unpublished = unpublished.with_columns(
        pl.when((pl.col("security_id") == "S00") & (pl.col("event_time") == early))
        .then(pl.lit(late))
        .otherwise(pl.col("available_time"))
        .alias("available_time")
    )
    omitted = unpublished.filter(
        ~((pl.col("security_id") == "S00") & (pl.col("event_time") == early))
    )
    ids = frame["security_id"].unique().sort().to_list()
    from quant_fund.data.point_in_time import filter_trailing_returns_asof

    dirty_hist = filter_trailing_returns_asof(
        unpublished.filter(pl.col("event_time") <= asof), asof
    )
    clean_hist = filter_trailing_returns_asof(
        omitted.filter(pl.col("event_time") <= asof), asof
    )
    seen: dict[str, list[np.ndarray]] = {}

    def capture(returns: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
        seen.setdefault("mats", []).append(np.asarray(returns, dtype=float).copy())
        return _fake_dcc(returns)

    with patch("quant_fund.pipeline.forecast.dcc_gaussian", side_effect=capture):
        dirty = estimate_optimizer_covariance_asof(cfg, unpublished, asof, ids, dirty_hist)
        clean = estimate_optimizer_covariance_asof(cfg, omitted, asof, ids, clean_hist)
    dirty_mat = seen["mats"][0]
    clean_mat = seen["mats"][1]
    dirty_finite = dirty_mat[np.isfinite(dirty_mat).all(axis=1)]
    clean_finite = clean_mat[np.isfinite(clean_mat).all(axis=1)]
    assert dirty.n_obs == clean.n_obs
    assert dirty_finite == pytest.approx(clean_finite)
    assert not np.any(np.abs(dirty_finite) > 1.0)


def test_optimize_asof_named_dcc_stamps_identity_not_overlay(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "dcc_gaussian"
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[-1]
    _dates, values = _garch_return_history(frame.filter(pl.col("event_time") < asof))
    _save_garch(tmp_path, values)
    captured: dict[str, np.ndarray] = {}

    def capture_mv(alpha, sigma, w_prev, config, **kwargs):
        captured["sigma"] = np.asarray(sigma, dtype=float).copy()
        n = len(alpha)
        return np.zeros(n), type("D", (), {"feasible": True})()

    with (
        patch("quant_fund.pipeline.forecast.dcc_gaussian", side_effect=_fake_dcc),
        patch("quant_fund.pipeline.forecast.optimize_mean_variance", side_effect=capture_mv),
        patch(
            "quant_fund.pipeline.forecast.apply_market_variance_overlay_to_covariance",
            side_effect=AssertionError("named DCC must not overlay H_{t+1}"),
        ),
    ):
        weights = optimize_asof(cfg, asof, persist=False, frame=frame)
    assert set(weights["covariance_estimator"].to_list()) == {DCC_FAMILY_GAUSSIAN}
    assert set(weights["covariance_object"].to_list()) == {DCC_COVARIANCE_OBJECT_ONE_STEP}
    assert set(weights["covariance_spec"].to_list()) == {DCC_SPEC_ENGLE_2002}
    assert "market_risk_overlay" not in weights.columns
    assert "garch_market_sigma" not in weights.columns
    n = captured["sigma"].shape[0]
    assert captured["sigma"] == pytest.approx(np.eye(n) * 0.0004)


def test_ledoit_path_still_applies_garch_overlay(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[-1]
    ids = frame["security_id"].unique().sort().to_list()
    hist = frame.filter(pl.col("event_time") <= asof)
    _dates, values = _garch_return_history(frame.filter(pl.col("event_time") < asof))
    _save_garch(tmp_path, values)
    estimate = estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)
    assert estimate.market_overlay == "garch"
    assert estimate.overlay is not None
    from quant_fund.models.covariance import ledoit_wolf_cov, repair_psd

    wide = hist.select(["event_time", "security_id", "ret_1"]).pivot(
        on="security_id", index="event_time", values="ret_1"
    )
    mat = wide.select(ids).to_numpy()
    unscaled, _ = repair_psd(ledoit_wolf_cov(mat), cfg.train.psd_eigen_tol)
    overlaid = overlay_covariance_with_garch_market(unscaled, float(estimate.overlay.variance))
    assert estimate.sigma == pytest.approx(overlaid)


def test_named_dcc_fails_closed_on_incomplete_asof_row(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "dcc_gaussian"
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
            "quant_fund.pipeline.forecast.dcc_gaussian",
            side_effect=AssertionError("incomplete asof must not run DCC"),
        ),
        pytest.raises(ValueError, match="incomplete_terminal_row"),
    ):
        estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)


def test_named_dcc_fails_closed_on_short_contiguous_suffix(tmp_path: Path) -> None:
    """Scattered complete rows must not be concatenated into a DCC sample."""
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "dcc_gaussian"
    frame = _panel()
    dates = frame["event_time"].unique().sort().to_list()
    asof = dates[-1]
    hole = dates[-(DCC_STAGE1_MIN_OBS - 1)]
    frame = frame.with_columns(
        pl.when(pl.col("event_time") == hole)
        .then(None)
        .otherwise(pl.col("ret_1"))
        .alias("ret_1")
    )
    ids = frame["security_id"].unique().sort().to_list()
    hist = frame.filter(pl.col("event_time") <= asof)
    complete = hist.filter(pl.col("ret_1").is_not_null())["event_time"].n_unique()
    assert complete >= DCC_STAGE1_MIN_OBS
    with (
        patch(
            "quant_fund.pipeline.forecast.dcc_gaussian",
            side_effect=AssertionError("stitched holes must not run DCC"),
        ),
        pytest.raises(ValueError, match="insufficient_contiguous_rows"),
    ):
        estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)


def test_named_dcc_sorts_trailing_returns_chronologically(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "dcc_gaussian"
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[-1]
    ids = frame["security_id"].unique().sort().to_list()
    hist = frame.filter(pl.col("event_time") <= asof)
    reversed_hist = hist.sort(["event_time", "security_id"], descending=True)
    seen: list[np.ndarray] = []

    def capture(returns: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
        seen.append(np.asarray(returns, dtype=float).copy())
        return _fake_dcc(returns)

    with patch("quant_fund.pipeline.forecast.dcc_gaussian", side_effect=capture):
        estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)
        estimate_optimizer_covariance_asof(cfg, frame, asof, ids, reversed_hist)
    assert len(seen) == 2
    assert seen[0] == pytest.approx(seen[1])
    assert np.isfinite(seen[0][-1]).all()


def _fake_student_t_dcc(returns: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
    x = np.asarray(returns, dtype=float)
    n = int(x.shape[1])
    sigma = np.eye(n, dtype=float) * 0.0009
    finite = int(np.isfinite(x).all(axis=1).sum())
    return sigma, {
        "a": 0.04,
        "b": 0.91,
        "nu": 8.0,
        "family": DCC_FAMILY_STUDENT_T,
        "spec": DCC_SPEC_STUDENT_T,
        "dist": "student_t",
        "covariance_object": DCC_COVARIANCE_OBJECT_ONE_STEP,
        "n_obs": float(finite),
        "n_assets": float(n),
    }


def test_named_student_t_dcc_path_returns_one_step_matrix_without_overlay(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "dcc_student_t"
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[-1]
    ids = frame["security_id"].unique().sort().to_list()
    hist = frame.filter(pl.col("event_time") <= asof)
    _dates, values = _garch_return_history(frame.filter(pl.col("event_time") < asof))
    _save_garch(tmp_path, values)
    seen: dict[str, np.ndarray] = {}

    def capture(returns: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
        seen["mat"] = np.asarray(returns, dtype=float).copy()
        return _fake_student_t_dcc(returns)

    with (
        patch("quant_fund.pipeline.forecast.dcc_student_t", side_effect=capture),
        patch(
            "quant_fund.pipeline.forecast.dcc_gaussian",
            side_effect=AssertionError("student-t DCC must not silently run dcc_gaussian"),
        ),
        patch(
            "quant_fund.pipeline.forecast.apply_market_variance_overlay_to_covariance",
            side_effect=AssertionError("DCC H_{t+1} must not be GARCH-overlaid"),
        ),
    ):
        estimate = estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)
    assert estimate.estimator == DCC_FAMILY_STUDENT_T
    assert estimate.covariance_object == DCC_COVARIANCE_OBJECT_ONE_STEP
    assert estimate.spec == DCC_SPEC_STUDENT_T
    assert estimate.market_overlay is None
    assert estimate.overlay is None
    assert estimate.sigma == pytest.approx(np.eye(len(ids)) * 0.0009)
    assert "mat" in seen
    assert seen["mat"].shape[0] >= DCC_STAGE1_MIN_OBS


def test_named_student_t_dcc_fails_closed_on_short_history(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "dcc_student_t"
    frame = _panel(n_days=20)
    asof = frame["event_time"].unique().sort().to_list()[-1]
    ids = frame["security_id"].unique().sort().to_list()
    hist = frame.filter(pl.col("event_time") <= asof)
    with pytest.raises(ValueError, match="optimizer_covariance_failed:dcc_student_t"):
        estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)


def test_named_student_t_dcc_does_not_fall_back_to_ledoit_wolf_or_gaussian(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "dcc_student_t"
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[-1]
    ids = frame["security_id"].unique().sort().to_list()
    hist = frame.filter(pl.col("event_time") <= asof)

    def boom(returns: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
        raise ValueError("stage-1 failed")

    with (
        patch("quant_fund.pipeline.forecast.dcc_student_t", side_effect=boom),
        patch(
            "quant_fund.pipeline.forecast.dcc_gaussian",
            side_effect=AssertionError("failed student-t DCC must not run dcc_gaussian"),
        ),
        pytest.raises(ValueError, match="optimizer_covariance_failed:dcc_student_t"),
    ):
        estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)


def test_named_student_t_dcc_fails_closed_on_family_mismatch(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "dcc_student_t"
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[-1]
    ids = frame["security_id"].unique().sort().to_list()
    hist = frame.filter(pl.col("event_time") <= asof)

    def gaussian_params(returns: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
        return _fake_dcc(returns)

    with (
        patch("quant_fund.pipeline.forecast.dcc_student_t", side_effect=gaussian_params),
        pytest.raises(ValueError, match="unexpected_family:dcc_gaussian"),
    ):
        estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)


def test_optimize_asof_named_student_t_dcc_stamps_identity_not_overlay(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "dcc_student_t"
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[-1]
    _dates, values = _garch_return_history(frame.filter(pl.col("event_time") < asof))
    _save_garch(tmp_path, values)
    captured: dict[str, np.ndarray] = {}

    def capture_mv(alpha, sigma, w_prev, config, **kwargs):
        captured["sigma"] = np.asarray(sigma, dtype=float).copy()
        n = len(alpha)
        return np.zeros(n), type("D", (), {"feasible": True})()

    with (
        patch("quant_fund.pipeline.forecast.dcc_student_t", side_effect=_fake_student_t_dcc),
        patch(
            "quant_fund.pipeline.forecast.dcc_gaussian",
            side_effect=AssertionError("optimize_asof student-t must not run dcc_gaussian"),
        ),
        patch("quant_fund.pipeline.forecast.optimize_mean_variance", side_effect=capture_mv),
        patch(
            "quant_fund.pipeline.forecast.apply_market_variance_overlay_to_covariance",
            side_effect=AssertionError("named DCC must not overlay H_{t+1}"),
        ),
    ):
        weights = optimize_asof(cfg, asof, persist=False, frame=frame)
    assert set(weights["covariance_estimator"].to_list()) == {DCC_FAMILY_STUDENT_T}
    assert set(weights["covariance_object"].to_list()) == {DCC_COVARIANCE_OBJECT_ONE_STEP}
    assert set(weights["covariance_spec"].to_list()) == {DCC_SPEC_STUDENT_T}
    assert set(weights["covariance_horizon"].to_list()) == {1}
    assert "market_risk_overlay" not in weights.columns
    assert "garch_market_sigma" not in weights.columns
    n = captured["sigma"].shape[0]
    assert captured["sigma"] == pytest.approx(np.eye(n) * 0.0009)


def test_named_student_t_dcc_fails_closed_on_incomplete_asof_row(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "dcc_student_t"
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
            "quant_fund.pipeline.forecast.dcc_student_t",
            side_effect=AssertionError("incomplete asof must not run DCC"),
        ),
        pytest.raises(ValueError, match="incomplete_terminal_row"),
    ):
        estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)


def _fake_adcc(returns: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
    x = np.asarray(returns, dtype=float)
    n = int(x.shape[1])
    sigma = np.eye(n, dtype=float) * 0.0016
    finite = int(np.isfinite(x).all(axis=1).sum())
    return sigma, {
        "a": 0.03,
        "b": 0.9,
        "g": 0.05,
        "kappa": 1.2,
        "family": DCC_FAMILY_ADCC,
        "spec": DCC_SPEC_ADCC,
        "dist": "normal",
        "asymmetric": "true",
        "covariance_object": DCC_COVARIANCE_OBJECT_ONE_STEP,
        "n_obs": float(finite),
        "n_assets": float(n),
    }


def test_named_adcc_path_returns_one_step_matrix_without_overlay(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "adcc"
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[-1]
    ids = frame["security_id"].unique().sort().to_list()
    hist = frame.filter(pl.col("event_time") <= asof)
    _dates, values = _garch_return_history(frame.filter(pl.col("event_time") < asof))
    _save_garch(tmp_path, values)
    seen: dict[str, np.ndarray] = {}

    def capture(returns: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
        seen["mat"] = np.asarray(returns, dtype=float).copy()
        return _fake_adcc(returns)

    with (
        patch("quant_fund.pipeline.forecast.adcc", side_effect=capture),
        patch(
            "quant_fund.pipeline.forecast.dcc_gaussian",
            side_effect=AssertionError("ADCC must not silently run dcc_gaussian"),
        ),
        patch(
            "quant_fund.pipeline.forecast.dcc_student_t",
            side_effect=AssertionError("ADCC must not silently run dcc_student_t"),
        ),
        patch(
            "quant_fund.pipeline.forecast.apply_market_variance_overlay_to_covariance",
            side_effect=AssertionError("DCC H_{t+1} must not be GARCH-overlaid"),
        ),
    ):
        estimate = estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)
    assert estimate.estimator == DCC_FAMILY_ADCC
    assert estimate.covariance_object == DCC_COVARIANCE_OBJECT_ONE_STEP
    assert estimate.spec == DCC_SPEC_ADCC
    assert estimate.market_overlay is None
    assert estimate.overlay is None
    assert estimate.sigma == pytest.approx(np.eye(len(ids)) * 0.0016)
    assert "mat" in seen
    assert seen["mat"].shape[0] >= DCC_STAGE1_MIN_OBS


def test_named_adcc_fails_closed_on_short_history(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "adcc"
    frame = _panel(n_days=20)
    asof = frame["event_time"].unique().sort().to_list()[-1]
    ids = frame["security_id"].unique().sort().to_list()
    hist = frame.filter(pl.col("event_time") <= asof)
    with pytest.raises(ValueError, match="optimizer_covariance_failed:adcc"):
        estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)


def test_named_adcc_does_not_fall_back_to_ledoit_wolf_or_engle_dcc(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "adcc"
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[-1]
    ids = frame["security_id"].unique().sort().to_list()
    hist = frame.filter(pl.col("event_time") <= asof)

    def boom(returns: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
        raise ValueError("stage-1 failed")

    with (
        patch("quant_fund.pipeline.forecast.adcc", side_effect=boom),
        patch(
            "quant_fund.pipeline.forecast.dcc_gaussian",
            side_effect=AssertionError("failed ADCC must not run dcc_gaussian"),
        ),
        patch(
            "quant_fund.pipeline.forecast.dcc_student_t",
            side_effect=AssertionError("failed ADCC must not run dcc_student_t"),
        ),
        pytest.raises(ValueError, match="optimizer_covariance_failed:adcc"),
    ):
        estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)


def test_named_adcc_fails_closed_on_family_mismatch(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "adcc"
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[-1]
    ids = frame["security_id"].unique().sort().to_list()
    hist = frame.filter(pl.col("event_time") <= asof)

    def gaussian_params(returns: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
        return _fake_dcc(returns)

    with (
        patch("quant_fund.pipeline.forecast.adcc", side_effect=gaussian_params),
        pytest.raises(ValueError, match="unexpected_family:dcc_gaussian"),
    ):
        estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)


def test_optimize_asof_named_adcc_stamps_identity_not_overlay(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "adcc"
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[-1]
    _dates, values = _garch_return_history(frame.filter(pl.col("event_time") < asof))
    _save_garch(tmp_path, values)
    captured: dict[str, np.ndarray] = {}

    def capture_mv(alpha, sigma, w_prev, config, **kwargs):
        captured["sigma"] = np.asarray(sigma, dtype=float).copy()
        n = len(alpha)
        return np.zeros(n), type("D", (), {"feasible": True})()

    with (
        patch("quant_fund.pipeline.forecast.adcc", side_effect=_fake_adcc),
        patch(
            "quant_fund.pipeline.forecast.dcc_gaussian",
            side_effect=AssertionError("optimize_asof ADCC must not run dcc_gaussian"),
        ),
        patch(
            "quant_fund.pipeline.forecast.dcc_student_t",
            side_effect=AssertionError("optimize_asof ADCC must not run dcc_student_t"),
        ),
        patch("quant_fund.pipeline.forecast.optimize_mean_variance", side_effect=capture_mv),
        patch(
            "quant_fund.pipeline.forecast.apply_market_variance_overlay_to_covariance",
            side_effect=AssertionError("named DCC must not overlay H_{t+1}"),
        ),
    ):
        weights = optimize_asof(cfg, asof, persist=False, frame=frame)
    assert set(weights["covariance_estimator"].to_list()) == {DCC_FAMILY_ADCC}
    assert set(weights["covariance_object"].to_list()) == {DCC_COVARIANCE_OBJECT_ONE_STEP}
    assert set(weights["covariance_spec"].to_list()) == {DCC_SPEC_ADCC}
    assert set(weights["covariance_horizon"].to_list()) == {1}
    assert "market_risk_overlay" not in weights.columns
    assert "garch_market_sigma" not in weights.columns
    n = captured["sigma"].shape[0]
    assert captured["sigma"] == pytest.approx(np.eye(n) * 0.0016)


def test_named_adcc_fails_closed_on_incomplete_asof_row(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "adcc"
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
            "quant_fund.pipeline.forecast.adcc",
            side_effect=AssertionError("incomplete asof must not run ADCC"),
        ),
        pytest.raises(ValueError, match="incomplete_terminal_row"),
    ):
        estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)


def _fake_ccc(returns: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
    x = np.asarray(returns, dtype=float)
    n = int(x.shape[1])
    sigma = np.eye(n, dtype=float) * 0.0025
    finite = int(np.isfinite(x).all(axis=1).sum())
    return sigma, {
        "family": DCC_FAMILY_CCC,
        "spec": DCC_SPEC_CCC,
        "dist": "normal",
        "asymmetric": "false",
        "dynamic_correlation": "false",
        "covariance_object": DCC_COVARIANCE_OBJECT_ONE_STEP,
        "n_obs": float(finite),
        "n_assets": float(n),
    }


def test_named_ccc_path_returns_one_step_matrix_without_overlay(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "ccc"
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[-1]
    ids = frame["security_id"].unique().sort().to_list()
    hist = frame.filter(pl.col("event_time") <= asof)
    _dates, values = _garch_return_history(frame.filter(pl.col("event_time") < asof))
    _save_garch(tmp_path, values)
    seen: dict[str, np.ndarray] = {}

    def capture(returns: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
        seen["mat"] = np.asarray(returns, dtype=float).copy()
        return _fake_ccc(returns)

    with (
        patch("quant_fund.pipeline.forecast.ccc", side_effect=capture),
        patch(
            "quant_fund.pipeline.forecast.dcc_gaussian",
            side_effect=AssertionError("CCC must not silently run dcc_gaussian"),
        ),
        patch(
            "quant_fund.pipeline.forecast.dcc_student_t",
            side_effect=AssertionError("CCC must not silently run dcc_student_t"),
        ),
        patch(
            "quant_fund.pipeline.forecast.adcc",
            side_effect=AssertionError("CCC must not silently run adcc"),
        ),
        patch(
            "quant_fund.pipeline.forecast.apply_market_variance_overlay_to_covariance",
            side_effect=AssertionError("CCC H_{t+1} must not be GARCH-overlaid"),
        ),
    ):
        estimate = estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)
    assert estimate.estimator == DCC_FAMILY_CCC
    assert estimate.covariance_object == DCC_COVARIANCE_OBJECT_ONE_STEP
    assert estimate.spec == DCC_SPEC_CCC
    assert estimate.market_overlay is None
    assert estimate.overlay is None
    assert estimate.sigma == pytest.approx(np.eye(len(ids)) * 0.0025)
    assert "mat" in seen
    assert seen["mat"].shape[0] >= DCC_STAGE1_MIN_OBS


def test_named_ccc_fails_closed_on_short_history(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "ccc"
    frame = _panel(n_days=20)
    asof = frame["event_time"].unique().sort().to_list()[-1]
    ids = frame["security_id"].unique().sort().to_list()
    hist = frame.filter(pl.col("event_time") <= asof)
    with pytest.raises(ValueError, match="optimizer_covariance_failed:ccc"):
        estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)


def test_named_ccc_does_not_fall_back_to_ledoit_wolf_or_engle_dcc(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "ccc"
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[-1]
    ids = frame["security_id"].unique().sort().to_list()
    hist = frame.filter(pl.col("event_time") <= asof)

    def boom(returns: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
        raise ValueError("stage-1 failed")

    with (
        patch("quant_fund.pipeline.forecast.ccc", side_effect=boom),
        patch(
            "quant_fund.pipeline.forecast.dcc_gaussian",
            side_effect=AssertionError("failed CCC must not run dcc_gaussian"),
        ),
        patch(
            "quant_fund.pipeline.forecast.dcc_student_t",
            side_effect=AssertionError("failed CCC must not run dcc_student_t"),
        ),
        patch(
            "quant_fund.pipeline.forecast.adcc",
            side_effect=AssertionError("failed CCC must not run adcc"),
        ),
        pytest.raises(ValueError, match="optimizer_covariance_failed:ccc"),
    ):
        estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)


def test_named_ccc_fails_closed_on_family_mismatch(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "ccc"
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[-1]
    ids = frame["security_id"].unique().sort().to_list()
    hist = frame.filter(pl.col("event_time") <= asof)

    def gaussian_params(returns: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
        return _fake_dcc(returns)

    with (
        patch("quant_fund.pipeline.forecast.ccc", side_effect=gaussian_params),
        pytest.raises(ValueError, match="unexpected_family:dcc_gaussian"),
    ):
        estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)


def test_optimize_asof_named_ccc_stamps_identity_not_overlay(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "ccc"
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[-1]
    _dates, values = _garch_return_history(frame.filter(pl.col("event_time") < asof))
    _save_garch(tmp_path, values)
    captured: dict[str, np.ndarray] = {}

    def capture_mv(alpha, sigma, w_prev, config, **kwargs):
        captured["sigma"] = np.asarray(sigma, dtype=float).copy()
        n = len(alpha)
        return np.zeros(n), type("D", (), {"feasible": True})()

    with (
        patch("quant_fund.pipeline.forecast.ccc", side_effect=_fake_ccc),
        patch(
            "quant_fund.pipeline.forecast.dcc_gaussian",
            side_effect=AssertionError("optimize_asof CCC must not run dcc_gaussian"),
        ),
        patch(
            "quant_fund.pipeline.forecast.dcc_student_t",
            side_effect=AssertionError("optimize_asof CCC must not run dcc_student_t"),
        ),
        patch(
            "quant_fund.pipeline.forecast.adcc",
            side_effect=AssertionError("optimize_asof CCC must not run adcc"),
        ),
        patch("quant_fund.pipeline.forecast.optimize_mean_variance", side_effect=capture_mv),
        patch(
            "quant_fund.pipeline.forecast.apply_market_variance_overlay_to_covariance",
            side_effect=AssertionError("named CCC must not overlay H_{t+1}"),
        ),
    ):
        weights = optimize_asof(cfg, asof, persist=False, frame=frame)
    assert set(weights["covariance_estimator"].to_list()) == {DCC_FAMILY_CCC}
    assert set(weights["covariance_object"].to_list()) == {DCC_COVARIANCE_OBJECT_ONE_STEP}
    assert set(weights["covariance_spec"].to_list()) == {DCC_SPEC_CCC}
    assert set(weights["covariance_horizon"].to_list()) == {1}
    assert "market_risk_overlay" not in weights.columns
    assert "garch_market_sigma" not in weights.columns
    n = captured["sigma"].shape[0]
    assert captured["sigma"] == pytest.approx(np.eye(n) * 0.0025)


def test_named_ccc_fails_closed_on_incomplete_asof_row(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "ccc"
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
            "quant_fund.pipeline.forecast.ccc",
            side_effect=AssertionError("incomplete asof must not run CCC"),
        ),
        pytest.raises(ValueError, match="incomplete_terminal_row"),
    ):
        estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)


def _fake_agdcc(returns: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
    x = np.asarray(returns, dtype=float)
    n = int(x.shape[1])
    sigma = np.eye(n, dtype=float) * 0.0036
    finite = int(np.isfinite(x).all(axis=1).sum())
    return sigma, {
        "family": DCC_FAMILY_AGDCC,
        "spec": DCC_SPEC_AGDCC,
        "parameterization": "diagonal",
        "dist": "normal",
        "asymmetric": "true",
        "dynamic_correlation": "true",
        "covariance_object": DCC_COVARIANCE_OBJECT_ONE_STEP,
        "n_obs": float(finite),
        "n_assets": float(n),
    }


def test_named_agdcc_path_returns_one_step_matrix_without_overlay(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "agdcc"
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[-1]
    ids = frame["security_id"].unique().sort().to_list()
    hist = frame.filter(pl.col("event_time") <= asof)
    _dates, values = _garch_return_history(frame.filter(pl.col("event_time") < asof))
    _save_garch(tmp_path, values)
    seen: dict[str, np.ndarray] = {}

    def capture(returns: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
        seen["mat"] = np.asarray(returns, dtype=float).copy()
        return _fake_agdcc(returns)

    with (
        patch("quant_fund.pipeline.forecast.agdcc", side_effect=capture),
        patch(
            "quant_fund.pipeline.forecast.adcc",
            side_effect=AssertionError("AG-DCC must not silently run adcc"),
        ),
        patch(
            "quant_fund.pipeline.forecast.dcc_gaussian",
            side_effect=AssertionError("AG-DCC must not silently run dcc_gaussian"),
        ),
        patch(
            "quant_fund.pipeline.forecast.dcc_student_t",
            side_effect=AssertionError("AG-DCC must not silently run dcc_student_t"),
        ),
        patch(
            "quant_fund.pipeline.forecast.ccc",
            side_effect=AssertionError("AG-DCC must not silently run ccc"),
        ),
        patch(
            "quant_fund.pipeline.forecast.agdcc_full",
            side_effect=AssertionError("AG-DCC must not silently run agdcc_full"),
        ),
        patch(
            "quant_fund.pipeline.forecast.apply_market_variance_overlay_to_covariance",
            side_effect=AssertionError("AG-DCC H_{t+1} must not be GARCH-overlaid"),
        ),
    ):
        estimate = estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)
    assert estimate.estimator == DCC_FAMILY_AGDCC
    assert estimate.covariance_object == DCC_COVARIANCE_OBJECT_ONE_STEP
    assert estimate.spec == DCC_SPEC_AGDCC
    assert estimate.market_overlay is None
    assert estimate.overlay is None
    assert estimate.sigma == pytest.approx(np.eye(len(ids)) * 0.0036)
    assert "mat" in seen
    assert seen["mat"].shape[0] >= DCC_STAGE1_MIN_OBS


def test_named_agdcc_fails_closed_on_short_history(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "agdcc"
    frame = _panel(n_days=20)
    asof = frame["event_time"].unique().sort().to_list()[-1]
    ids = frame["security_id"].unique().sort().to_list()
    hist = frame.filter(pl.col("event_time") <= asof)
    with pytest.raises(ValueError, match="optimizer_covariance_failed:agdcc"):
        estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)


def test_named_agdcc_does_not_fall_back_to_ledoit_wolf_or_scalar_adcc(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "agdcc"
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[-1]
    ids = frame["security_id"].unique().sort().to_list()
    hist = frame.filter(pl.col("event_time") <= asof)

    def boom(returns: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
        raise ValueError("stage-1 failed")

    with (
        patch("quant_fund.pipeline.forecast.agdcc", side_effect=boom),
        patch(
            "quant_fund.pipeline.forecast.adcc",
            side_effect=AssertionError("failed AG-DCC must not run adcc"),
        ),
        patch(
            "quant_fund.pipeline.forecast.dcc_gaussian",
            side_effect=AssertionError("failed AG-DCC must not run dcc_gaussian"),
        ),
        patch(
            "quant_fund.pipeline.forecast.dcc_student_t",
            side_effect=AssertionError("failed AG-DCC must not run dcc_student_t"),
        ),
        patch(
            "quant_fund.pipeline.forecast.ccc",
            side_effect=AssertionError("failed AG-DCC must not run ccc"),
        ),
        patch(
            "quant_fund.pipeline.forecast.agdcc_full",
            side_effect=AssertionError("failed AG-DCC must not run agdcc_full"),
        ),
        pytest.raises(ValueError, match="optimizer_covariance_failed:agdcc"),
    ):
        estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)


def test_named_agdcc_fails_closed_on_family_mismatch(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "agdcc"
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[-1]
    ids = frame["security_id"].unique().sort().to_list()
    hist = frame.filter(pl.col("event_time") <= asof)

    def gaussian_params(returns: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
        return _fake_dcc(returns)

    with (
        patch("quant_fund.pipeline.forecast.agdcc", side_effect=gaussian_params),
        pytest.raises(ValueError, match="unexpected_family:dcc_gaussian"),
    ):
        estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)


def test_named_agdcc_fails_closed_on_scalar_adcc_family_mismatch(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "agdcc"
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[-1]
    ids = frame["security_id"].unique().sort().to_list()
    hist = frame.filter(pl.col("event_time") <= asof)

    def adcc_params(returns: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
        return _fake_adcc(returns)

    with (
        patch("quant_fund.pipeline.forecast.agdcc", side_effect=adcc_params),
        pytest.raises(ValueError, match="unexpected_family:adcc"),
    ):
        estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)


def test_named_agdcc_fails_closed_on_full_family_mismatch(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "agdcc"
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[-1]
    ids = frame["security_id"].unique().sort().to_list()
    hist = frame.filter(pl.col("event_time") <= asof)

    def full_params(returns: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
        return _fake_agdcc_full(returns)

    with (
        patch("quant_fund.pipeline.forecast.agdcc", side_effect=full_params),
        pytest.raises(ValueError, match="unexpected_family:agdcc_full"),
    ):
        estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)


def test_optimize_asof_named_agdcc_stamps_identity_not_overlay(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "agdcc"
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[-1]
    _dates, values = _garch_return_history(frame.filter(pl.col("event_time") < asof))
    _save_garch(tmp_path, values)
    captured: dict[str, np.ndarray] = {}

    def capture_mv(alpha, sigma, w_prev, config, **kwargs):
        captured["sigma"] = np.asarray(sigma, dtype=float).copy()
        n = len(alpha)
        return np.zeros(n), type("D", (), {"feasible": True})()

    with (
        patch("quant_fund.pipeline.forecast.agdcc", side_effect=_fake_agdcc),
        patch(
            "quant_fund.pipeline.forecast.adcc",
            side_effect=AssertionError("optimize_asof AG-DCC must not run adcc"),
        ),
        patch(
            "quant_fund.pipeline.forecast.dcc_gaussian",
            side_effect=AssertionError("optimize_asof AG-DCC must not run dcc_gaussian"),
        ),
        patch(
            "quant_fund.pipeline.forecast.dcc_student_t",
            side_effect=AssertionError("optimize_asof AG-DCC must not run dcc_student_t"),
        ),
        patch(
            "quant_fund.pipeline.forecast.ccc",
            side_effect=AssertionError("optimize_asof AG-DCC must not run ccc"),
        ),
        patch(
            "quant_fund.pipeline.forecast.agdcc_full",
            side_effect=AssertionError("optimize_asof AG-DCC must not run agdcc_full"),
        ),
        patch("quant_fund.pipeline.forecast.optimize_mean_variance", side_effect=capture_mv),
        patch(
            "quant_fund.pipeline.forecast.apply_market_variance_overlay_to_covariance",
            side_effect=AssertionError("named AG-DCC must not overlay H_{t+1}"),
        ),
    ):
        weights = optimize_asof(cfg, asof, persist=False, frame=frame)
    assert set(weights["covariance_estimator"].to_list()) == {DCC_FAMILY_AGDCC}
    assert set(weights["covariance_object"].to_list()) == {DCC_COVARIANCE_OBJECT_ONE_STEP}
    assert set(weights["covariance_spec"].to_list()) == {DCC_SPEC_AGDCC}
    assert set(weights["covariance_horizon"].to_list()) == {1}
    assert "market_risk_overlay" not in weights.columns
    assert "garch_market_sigma" not in weights.columns
    n = captured["sigma"].shape[0]
    assert captured["sigma"] == pytest.approx(np.eye(n) * 0.0036)


def test_named_agdcc_fails_closed_on_incomplete_asof_row(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "agdcc"
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
            "quant_fund.pipeline.forecast.agdcc",
            side_effect=AssertionError("incomplete asof must not run AG-DCC"),
        ),
        pytest.raises(ValueError, match="incomplete_terminal_row"),
    ):
        estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)


def _fake_agdcc_full(returns: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
    x = np.asarray(returns, dtype=float)
    n = int(x.shape[1])
    sigma = np.eye(n, dtype=float) * 0.0049
    finite = int(np.isfinite(x).all(axis=1).sum())
    return sigma, {
        "family": DCC_FAMILY_AGDCC_FULL,
        "spec": DCC_SPEC_AGDCC_FULL,
        "parameterization": "full",
        "dist": "normal",
        "asymmetric": "true",
        "dynamic_correlation": "true",
        "covariance_object": DCC_COVARIANCE_OBJECT_ONE_STEP,
        "n_obs": float(finite),
        "n_assets": float(n),
    }


def test_named_agdcc_full_path_returns_one_step_matrix_without_overlay(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "agdcc_full"
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[-1]
    ids = frame["security_id"].unique().sort().to_list()
    hist = frame.filter(pl.col("event_time") <= asof)
    _dates, values = _garch_return_history(frame.filter(pl.col("event_time") < asof))
    _save_garch(tmp_path, values)
    seen: dict[str, np.ndarray] = {}

    def capture(returns: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
        seen["mat"] = np.asarray(returns, dtype=float).copy()
        return _fake_agdcc_full(returns)

    with (
        patch("quant_fund.pipeline.forecast.agdcc_full", side_effect=capture),
        patch(
            "quant_fund.pipeline.forecast.agdcc",
            side_effect=AssertionError("unrestricted AG-DCC must not silently run agdcc"),
        ),
        patch(
            "quant_fund.pipeline.forecast.adcc",
            side_effect=AssertionError("unrestricted AG-DCC must not silently run adcc"),
        ),
        patch(
            "quant_fund.pipeline.forecast.dcc_gaussian",
            side_effect=AssertionError("unrestricted AG-DCC must not silently run dcc_gaussian"),
        ),
        patch(
            "quant_fund.pipeline.forecast.dcc_student_t",
            side_effect=AssertionError("unrestricted AG-DCC must not silently run dcc_student_t"),
        ),
        patch(
            "quant_fund.pipeline.forecast.ccc",
            side_effect=AssertionError("unrestricted AG-DCC must not silently run ccc"),
        ),
        patch(
            "quant_fund.pipeline.forecast.apply_market_variance_overlay_to_covariance",
            side_effect=AssertionError("unrestricted AG-DCC H_{t+1} must not be GARCH-overlaid"),
        ),
    ):
        estimate = estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)
    assert estimate.estimator == DCC_FAMILY_AGDCC_FULL
    assert estimate.covariance_object == DCC_COVARIANCE_OBJECT_ONE_STEP
    assert estimate.spec == DCC_SPEC_AGDCC_FULL
    assert estimate.market_overlay is None
    assert estimate.overlay is None
    assert estimate.sigma == pytest.approx(np.eye(len(ids)) * 0.0049)
    assert "mat" in seen
    assert seen["mat"].shape[0] >= DCC_STAGE1_MIN_OBS


def test_named_agdcc_full_fails_closed_on_short_history(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "agdcc_full"
    frame = _panel(n_days=20)
    asof = frame["event_time"].unique().sort().to_list()[-1]
    ids = frame["security_id"].unique().sort().to_list()
    hist = frame.filter(pl.col("event_time") <= asof)
    with pytest.raises(ValueError, match="optimizer_covariance_failed:agdcc_full"):
        estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)


def test_named_agdcc_full_does_not_fall_back_to_ledoit_wolf_or_diagonal(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "agdcc_full"
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[-1]
    ids = frame["security_id"].unique().sort().to_list()
    hist = frame.filter(pl.col("event_time") <= asof)

    def boom(returns: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
        raise ValueError("stage-1 failed")

    with (
        patch("quant_fund.pipeline.forecast.agdcc_full", side_effect=boom),
        patch(
            "quant_fund.pipeline.forecast.agdcc",
            side_effect=AssertionError("failed unrestricted AG-DCC must not run agdcc"),
        ),
        patch(
            "quant_fund.pipeline.forecast.adcc",
            side_effect=AssertionError("failed unrestricted AG-DCC must not run adcc"),
        ),
        patch(
            "quant_fund.pipeline.forecast.dcc_gaussian",
            side_effect=AssertionError("failed unrestricted AG-DCC must not run dcc_gaussian"),
        ),
        patch(
            "quant_fund.pipeline.forecast.dcc_student_t",
            side_effect=AssertionError("failed unrestricted AG-DCC must not run dcc_student_t"),
        ),
        patch(
            "quant_fund.pipeline.forecast.ccc",
            side_effect=AssertionError("failed unrestricted AG-DCC must not run ccc"),
        ),
        pytest.raises(ValueError, match="optimizer_covariance_failed:agdcc_full"),
    ):
        estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)


def test_named_agdcc_full_fails_closed_on_family_mismatch(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "agdcc_full"
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[-1]
    ids = frame["security_id"].unique().sort().to_list()
    hist = frame.filter(pl.col("event_time") <= asof)

    def gaussian_params(returns: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
        return _fake_dcc(returns)

    with (
        patch("quant_fund.pipeline.forecast.agdcc_full", side_effect=gaussian_params),
        pytest.raises(ValueError, match="unexpected_family:dcc_gaussian"),
    ):
        estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)


def test_named_agdcc_full_fails_closed_on_diagonal_family_mismatch(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "agdcc_full"
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[-1]
    ids = frame["security_id"].unique().sort().to_list()
    hist = frame.filter(pl.col("event_time") <= asof)

    def diagonal_params(returns: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
        return _fake_agdcc(returns)

    with (
        patch("quant_fund.pipeline.forecast.agdcc_full", side_effect=diagonal_params),
        pytest.raises(ValueError, match="unexpected_family:agdcc"),
    ):
        estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)


def test_optimize_asof_named_agdcc_full_stamps_identity_not_overlay(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "agdcc_full"
    frame = _panel()
    asof = frame["event_time"].unique().sort().to_list()[-1]
    _dates, values = _garch_return_history(frame.filter(pl.col("event_time") < asof))
    _save_garch(tmp_path, values)
    captured: dict[str, np.ndarray] = {}

    def capture_mv(alpha, sigma, w_prev, config, **kwargs):
        captured["sigma"] = np.asarray(sigma, dtype=float).copy()
        n = len(alpha)
        return np.zeros(n), type("D", (), {"feasible": True})()

    with (
        patch("quant_fund.pipeline.forecast.agdcc_full", side_effect=_fake_agdcc_full),
        patch(
            "quant_fund.pipeline.forecast.agdcc",
            side_effect=AssertionError("optimize_asof unrestricted AG-DCC must not run agdcc"),
        ),
        patch(
            "quant_fund.pipeline.forecast.adcc",
            side_effect=AssertionError("optimize_asof unrestricted AG-DCC must not run adcc"),
        ),
        patch(
            "quant_fund.pipeline.forecast.dcc_gaussian",
            side_effect=AssertionError(
                "optimize_asof unrestricted AG-DCC must not run dcc_gaussian"
            ),
        ),
        patch(
            "quant_fund.pipeline.forecast.dcc_student_t",
            side_effect=AssertionError(
                "optimize_asof unrestricted AG-DCC must not run dcc_student_t"
            ),
        ),
        patch(
            "quant_fund.pipeline.forecast.ccc",
            side_effect=AssertionError("optimize_asof unrestricted AG-DCC must not run ccc"),
        ),
        patch("quant_fund.pipeline.forecast.optimize_mean_variance", side_effect=capture_mv),
        patch(
            "quant_fund.pipeline.forecast.apply_market_variance_overlay_to_covariance",
            side_effect=AssertionError("named unrestricted AG-DCC must not overlay H_{t+1}"),
        ),
    ):
        weights = optimize_asof(cfg, asof, persist=False, frame=frame)
    assert set(weights["covariance_estimator"].to_list()) == {DCC_FAMILY_AGDCC_FULL}
    assert set(weights["covariance_object"].to_list()) == {DCC_COVARIANCE_OBJECT_ONE_STEP}
    assert set(weights["covariance_spec"].to_list()) == {DCC_SPEC_AGDCC_FULL}
    assert set(weights["covariance_horizon"].to_list()) == {1}
    assert "market_risk_overlay" not in weights.columns
    assert "garch_market_sigma" not in weights.columns
    n = captured["sigma"].shape[0]
    assert captured["sigma"] == pytest.approx(np.eye(n) * 0.0049)


def test_named_agdcc_full_fails_closed_on_incomplete_asof_row(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.optimizer.covariance = "agdcc_full"
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
            "quant_fund.pipeline.forecast.agdcc_full",
            side_effect=AssertionError("incomplete asof must not run unrestricted AG-DCC"),
        ),
        pytest.raises(ValueError, match="incomplete_terminal_row"),
    ):
        estimate_optimizer_covariance_asof(cfg, frame, asof, ids, hist)
