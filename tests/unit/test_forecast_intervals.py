"""Forecast conformal intervals. Coverage sets, not Sharpe."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quant_fund.config import load_config
from quant_fund.config.models import AppConfig
from quant_fund.metrics.cross_section import _date_keys
from quant_fund.pipeline.dataset import build_gold, panel
from quant_fund.pipeline.forecast import (
    _load_rl_cached,
    conformal_sets_asof,
    forecast_asof,
    history_for_calibration,
)
from quant_fund.schemas.forecast import AssetForecast


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("rl_linucb.joblib", "RL_LINUCB"),
        ("rl_thompson.joblib", "RL_THOMPSON"),
        ("rl_quantile_thompson.joblib", "RL_QUANTILE_THOMPSON"),
        ("rl_policy_gradient.joblib", "RL_POLICY_GRADIENT"),
    ],
)
def test_rl_artifact_identity_is_preserved(tmp_path: Path, filename: str, expected: str) -> None:
    import joblib

    from quant_fund.models.rl import LinUCBRanker
    from quant_fund.pipeline import forecast as forecast_module

    root = tmp_path / "metadata"
    root.mkdir()
    joblib.dump({"policy": LinUCBRanker(1), "features": ["ret_1"]}, root / filename)
    cfg = AppConfig()
    cfg.data.root = tmp_path
    forecast_module._RANKER_CACHE.clear()
    loaded = _load_rl_cached(cfg)
    assert loaded is not None
    assert loaded[2] == expected


def test_rl_artifact_checksum_mismatch_fails_closed(tmp_path: Path) -> None:
    from quant_fund.models.base import save_joblib_artifact
    from quant_fund.models.rl import LinUCBRanker
    from quant_fund.pipeline import forecast as forecast_module

    root = tmp_path / "metadata"
    root.mkdir()
    artifact = root / "rl_linucb.joblib"
    save_joblib_artifact(
        {"policy": LinUCBRanker(1), "features": ["ret_1"]}, artifact
    )
    artifact.write_bytes(artifact.read_bytes() + b"tampered")
    cfg = AppConfig()
    cfg.data.root = tmp_path
    forecast_module._RANKER_CACHE.clear()
    with pytest.raises(ValueError, match="checksum mismatch"):
        _load_rl_cached(cfg)


def test_probability_calibrator_loader_fails_closed_for_missing_artifact(tmp_path: Path) -> None:
    from quant_fund.pipeline import forecast as forecast_module

    cfg = AppConfig()
    cfg.data.root = tmp_path
    cfg.fusion.apply_probability_calibration = True
    with pytest.raises(ValueError, match="artifact is missing"):
        forecast_module._load_probability_calibrator(cfg)


def test_probability_calibrator_loader_enforces_score_identity(tmp_path: Path) -> None:
    from quant_fund.models.base import save_joblib_artifact
    from quant_fund.models.calibration import ProbabilityCalibrator
    from quant_fund.pipeline import forecast as forecast_module

    calibrator = ProbabilityCalibrator("isotonic").fit(np.linspace(0, 1, 20), np.tile([0.0, 1.0], 10))
    calibrator.score_feature = "wrong_score"
    calibrator.label = "future_label"
    path = tmp_path / "metadata" / "calibrator_auto.joblib"
    save_joblib_artifact(calibrator, path)
    cfg = AppConfig()
    cfg.data.root = tmp_path
    with pytest.raises(ValueError, match="score identity mismatch"):
        forecast_module._load_probability_calibrator(cfg)


def test_probability_calibrator_loader_accepts_fitted_bound_artifact(tmp_path: Path) -> None:
    from quant_fund.models.calibration import ProbabilityCalibrator
    from quant_fund.pipeline import forecast as forecast_module

    calibrator = ProbabilityCalibrator("platt").fit(np.linspace(0, 1, 20), np.tile([0.0, 1.0], 10))
    calibrator.score_feature = "cs_pct_mom_20"
    calibrator.label = "future_excess_return_5"
    calibrator.horizon = "future_excess_return_5"
    calibrator.fit_start = "2020-01-01"
    calibrator.fit_end = "2020-02-01"
    calibrator.oos_start = "2020-02-02"
    calibrator.oos_end = "2020-02-10"
    path = tmp_path / "metadata" / "calibrator_auto.joblib"
    calibrator.save(path)
    cfg = AppConfig()
    cfg.data.root = tmp_path
    loaded = forecast_module._load_probability_calibrator(cfg)
    assert loaded.method == "platt"


def test_probability_calibrator_loader_rejects_configured_label_mismatch(tmp_path: Path) -> None:
    from quant_fund.models.calibration import ProbabilityCalibrator
    from quant_fund.pipeline import forecast as forecast_module

    calibrator = ProbabilityCalibrator("platt").fit(np.linspace(0, 1, 20), np.tile([0.0, 1.0], 10))
    calibrator.score_feature = "cs_pct_mom_20"
    calibrator.label = "future_excess_return_5"
    calibrator.horizon = "future_excess_return_5"
    calibrator.fit_start = "2020-01-01"
    calibrator.fit_end = "2020-02-01"
    calibrator.oos_start = "2020-02-02"
    calibrator.oos_end = "2020-02-10"
    calibrator.save(tmp_path / "metadata" / "calibrator_auto.joblib")
    cfg = AppConfig()
    cfg.data.root = tmp_path
    cfg.fusion.probability_calibration_label = "future_excess_return_20"
    with pytest.raises(ValueError, match="label identity mismatch"):
        forecast_module._load_probability_calibrator(cfg)


def test_probability_calibrator_loader_rejects_missing_window_provenance(tmp_path: Path) -> None:
    from quant_fund.models.calibration import ProbabilityCalibrator
    from quant_fund.pipeline import forecast as forecast_module

    calibrator = ProbabilityCalibrator("platt").fit(np.linspace(0, 1, 20), np.tile([0.0, 1.0], 10))
    calibrator.score_feature = "cs_pct_mom_20"
    calibrator.label = "future_excess_return_5"
    calibrator.horizon = "future_excess_return_5"
    calibrator.save(tmp_path / "metadata" / "calibrator_auto.joblib")
    cfg = AppConfig()
    cfg.data.root = tmp_path
    with pytest.raises(ValueError, match="fit_start is missing"):
        forecast_module._load_probability_calibrator(cfg)


def test_probability_calibrator_loader_rejects_stale_asof(tmp_path: Path) -> None:
    from quant_fund.models.calibration import ProbabilityCalibrator
    from quant_fund.pipeline import forecast as forecast_module

    calibrator = ProbabilityCalibrator("platt").fit(np.linspace(0, 1, 20), np.tile([0.0, 1.0], 10))
    calibrator.score_feature = "cs_pct_mom_20"
    calibrator.label = "future_excess_return_5"
    calibrator.horizon = "future_excess_return_5"
    calibrator.fit_start = "2020-01-01"
    calibrator.fit_end = "2020-02-01"
    calibrator.oos_start = "2020-02-02"
    calibrator.oos_end = "2020-02-10"
    calibrator.save(tmp_path / "metadata" / "calibrator_auto.joblib")
    cfg = AppConfig()
    cfg.data.root = tmp_path
    cfg.fusion.probability_calibration_max_age_days = 5
    with pytest.raises(ValueError, match="stale"):
        forecast_module._load_probability_calibrator(
            cfg, asof=datetime(2020, 2, 20, tzinfo=UTC)
        )


def test_ranker_loader_accepts_ensemble_artifact(tmp_path: Path) -> None:
    from quant_fund.models.ranking import EnsembleRanker
    from quant_fund.pipeline import forecast as forecast_module

    root = tmp_path / "metadata"
    root.mkdir()
    x = np.arange(40, dtype=float).reshape(20, 2)
    y = np.sin(x[:, 0])
    model = EnsembleRanker(seed=3).fit(x, y)
    model.features = ["ret_1", "vol_20"]
    model.save(root / "ranker_ensemble.joblib")
    cfg = AppConfig()
    cfg.data.root = tmp_path
    forecast_module._RANKER_CACHE.clear()
    loaded = forecast_module._load_ranker_cached(cfg)
    assert isinstance(loaded, EnsembleRanker)
    assert loaded.features == ["ret_1", "vol_20"]


def test_ranker_loader_discovers_neural_artifact(tmp_path: Path) -> None:
    from quant_fund.models.ranking import NeuralRanker
    from quant_fund.pipeline import forecast as forecast_module

    root = tmp_path / "metadata"
    root.mkdir()
    x = np.arange(40, dtype=float).reshape(20, 2)
    y = np.sin(x[:, 0])
    model = NeuralRanker(seed=4).fit(x, y)
    model.features = ["ret_1", "vol_20"]
    model.save(root / "ranker_neural.joblib")
    cfg = AppConfig()
    cfg.data.root = tmp_path
    forecast_module._RANKER_CACHE.clear()
    loaded = forecast_module._load_ranker_cached(cfg)
    assert isinstance(loaded, NeuralRanker)


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
