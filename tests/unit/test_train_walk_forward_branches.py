"""Walk-forward training happy paths for distribution/volatility/regime/tail.

Complements test_pipeline_training (fail-closed edges) by exercising each family's
walk-forward loop and persisted-artifact path on a small synthetic gold panel.
Tracking is monkeypatched so tests stay hermetic (no MLflow writes).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quant_fund.config import load_config
from quant_fund.config.models import ValidationConfig
from quant_fund.pipeline.train import (
    _fit_ranker,
    _garch_oos_predictions,
    _garch_return_history,
    _split_fold,
    train_alpha,
    train_distribution,
    train_regime,
    train_tail,
    train_volatility,
)
from quant_fund.validation.walk_forward import Fold


def _cfg(tmp_path: Path, **validation: int):
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.validation = ValidationConfig(
        scheme="expanding",
        train_bars=16,
        val_bars=4,
        test_bars=4,
        seed=42,
        **validation,
    )
    return cfg


def _stub_panel(n_dates: int = 64, n_secs: int = 6) -> pl.DataFrame:
    rng = np.random.default_rng(20260917)
    rows = []
    for day in range(n_dates):
        stamp = datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=day)
        mkt_vol = 0.02 + abs(float(rng.normal(0, 0.003)))
        for sec in range(n_secs):
            rows.append(
                {
                    "event_time": stamp,
                    "security_id": f"S{sec:02d}",
                    "ret_1": float(rng.normal(0.0004, 0.01)),
                    "vol_20": 0.02 + abs(float(rng.normal(0, 0.003))),
                    "vol_ewma": 0.02 + abs(float(rng.normal(0, 0.002))),
                    "vol_parkinson": 0.02 + abs(float(rng.normal(0, 0.003))),
                    "vol_of_vol": 0.005 + abs(float(rng.normal(0, 0.001))),
                    "mom_20": float(rng.normal(0, 0.01)),
                    "future_idio_return_1": float(rng.normal(0, 0.01)),
                    "future_log_return_5": float(rng.normal(0, 0.02)),
                    "future_return_5": float(rng.normal(0, 0.02)),
                    "future_realized_var_5": 0.0004 + abs(float(rng.normal(0, 0.0002))),
                    "future_tail_event_5": float(rng.random() < 0.15),
                    "mkt_ret_1": float(rng.normal(0, 0.01)),
                    "mkt_vol_20": mkt_vol,
                    "cs_dispersion": 0.01 + abs(float(rng.normal(0, 0.002))),
                    "breadth": 0.5 + float(rng.normal(0, 0.05)),
                }
            )
    return pl.DataFrame(rows)


@pytest.fixture()
def stub_train(monkeypatch: pytest.MonkeyPatch):
    def _apply(frame: pl.DataFrame) -> None:
        monkeypatch.setattr("quant_fund.pipeline.train.panel", lambda *a, **k: frame)
        monkeypatch.setattr("quant_fund.pipeline.train.configure_tracking", lambda: None)
        monkeypatch.setattr("quant_fund.pipeline.train.log_run", lambda **kwargs: "run_test_0001")

    return _apply


def test_train_distribution_empirical_walk_forward(
    tmp_path: Path, stub_train, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_train(_stub_panel())
    cfg = _cfg(tmp_path)
    result = train_distribution(cfg, "empirical")
    metrics = result["metrics"]
    assert np.isfinite(metrics["mean_pinball"])
    assert np.isfinite(metrics["crossing_rate"])
    assert Path(result["path"]).is_file()
    assert Path(result["path"]).parent == tmp_path / "metadata"


def test_train_volatility_ewma_walk_forward(tmp_path: Path, stub_train) -> None:
    stub_train(_stub_panel())
    cfg = _cfg(tmp_path)
    result = train_volatility(cfg, "ewma")
    assert np.isfinite(result["metrics"]["qlike"])
    assert Path(result["path"]).name == "vol_ewma.joblib"
    assert Path(result["path"]).is_file()


def test_train_volatility_har_uses_production_feature_schema(tmp_path: Path, stub_train) -> None:
    stub_train(_stub_panel())
    cfg = _cfg(tmp_path)

    result = train_volatility(cfg, "har")

    assert np.isfinite(result["metrics"]["qlike"])
    artifact = Path(result["path"])
    assert artifact.name == "vol_har.joblib"
    assert artifact.is_file()


@pytest.mark.parametrize(("model_name", "feature"), [("rolling", "vol_20"), ("ewma", "vol_ewma")])
def test_train_volatility_baselines_score_selected_sigma_as_variance(
    tmp_path: Path,
    stub_train,
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    feature: str,
) -> None:
    frame = _stub_panel().with_columns(
        pl.lit(0.02).alias(feature),
        (pl.int_range(0, pl.len()) % 11 * 0.001 + 0.001).alias("vol_of_vol"),
    )
    stub_train(frame)
    observed: list[np.ndarray] = []

    def capture_qlike(_y, forecast, _floor):  # noqa: ANN001
        observed.append(np.asarray(forecast, dtype=float).copy())
        return 0.0

    monkeypatch.setattr("quant_fund.pipeline.train.qlike", capture_qlike)
    result = train_volatility(_cfg(tmp_path), model_name)

    assert result["metrics"]["qlike"] == 0.0
    assert observed
    assert np.allclose(np.concatenate(observed), 0.02**2)


def test_garch_oos_predictions_use_full_causal_return_history() -> None:
    dates = np.array([0, 1, 3, 4, 3])
    x = np.zeros((5, 1))
    y = np.ones(5)
    test_mask = dates >= 3
    return_dates = np.arange(6)
    return_values = return_dates.astype(float)
    fit_lengths: list[int] = []

    class FakeGarch:
        fit_status = "fitted"

        def fit(self, _x, _y, *, returns):  # noqa: ANN001
            fit_lengths.append(len(returns))
            self.last_return = float(returns[-1])
            return self

        def forecast(self, *, horizon):
            return {"cumulative_variance": np.array([self.last_return + horizon])}

    predictions, statuses = _garch_oos_predictions(
        lambda: FakeGarch(),
        x,
        y,
        dates,
        test_mask,
        label_horizon=1,
        return_dates=return_dates,
        return_values=return_values,
    )

    assert predictions.tolist() == [3.0, 4.0, 3.0]
    assert statuses == ["fitted", "fitted"]
    assert fit_lengths == [3, 4]


def test_garch_return_history_keeps_latest_unlabeled_returns() -> None:
    frame = pl.DataFrame(
        {
            "event_time": [0, 1, 2],
            "ret_1": [0.01, 0.02, 0.03],
            "future_realized_var_5": [0.1, None, None],
        }
    )

    dates, values = _garch_return_history(frame)

    assert dates.tolist() == [0, 1, 2]
    assert values.tolist() == pytest.approx([0.01, 0.02, 0.03])


def test_train_garch_persisted_fit_uses_latest_return_history(
    tmp_path: Path, stub_train, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _stub_panel(n_dates=64, n_secs=2)
    latest = frame.select(pl.col("event_time").max()).item()
    frame = frame.with_columns(
        pl.when(pl.col("event_time") == latest)
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(pl.col("future_realized_var_5"))
        .alias("future_realized_var_5")
    )
    stub_train(frame)
    observed: list[np.ndarray] = []

    class FakeGarch:
        def __init__(self, **_kwargs):
            pass

        fit_status = "fitted"
        converged = True
        fallback_reason = None
        n_obs = 0
        returns_scale = 100.0

        def fit(self, _x, _y, *, returns):  # noqa: ANN001
            observed.append(np.asarray(returns, dtype=float).copy())
            self.n_obs = len(returns)
            self.last_return = float(returns[-1])
            return self

        def forecast(self, *, horizon):
            return {"cumulative_variance": np.repeat(self.last_return + horizon, horizon)}

        def save(self, path):  # noqa: ANN001
            output = Path(path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"fake-garch")

    monkeypatch.setattr("quant_fund.pipeline.train.GARCHVol", FakeGarch)
    result = train_volatility(_cfg(tmp_path), "garch")

    assert Path(result["path"]).is_file()
    assert len(observed[-1]) == 64
    assert result["diagnostics"]["series_scope"] == "date_level_equal_weight_cross_section"
    assert observed[-1][-1] == pytest.approx(
        float(frame.filter(pl.col("event_time") == latest)["ret_1"].mean())
    )


@pytest.mark.parametrize("model_name", ["threshold", "single_state"])
def test_train_regime_non_hmm_persists_model(tmp_path: Path, stub_train, model_name: str) -> None:
    stub_train(_stub_panel())
    cfg = _cfg(tmp_path)
    result = train_regime(cfg, model_name)
    assert result["metrics"] == {}
    assert Path(result["path"]).name == f"regime_{model_name}.joblib"
    assert Path(result["path"]).is_file()


def test_train_regime_hmm_reports_oos_likelihood(tmp_path: Path, stub_train) -> None:
    stub_train(_stub_panel())
    cfg = _cfg(tmp_path)
    cfg.train.n_hmm_states = 2
    result = train_regime(cfg, "hmm")
    assert np.isfinite(result["metrics"]["oos_avg_ll"])
    assert Path(result["path"]).is_file()


def test_train_tail_historical_and_drawdown(tmp_path: Path, stub_train) -> None:
    stub_train(_stub_panel())
    cfg = _cfg(tmp_path)
    historical = train_tail(cfg, "historical")
    metrics = historical["metrics"]
    assert np.isfinite(metrics["oos_var_breach_rate"])
    assert metrics["var"] <= metrics["es"]
    assert Path(historical["path"]).is_file()

    drawdown = train_tail(cfg, "drawdown")
    assert np.isfinite(drawdown["metrics"]["oos_brier"])
    assert Path(drawdown["path"]).name == "tail_drawdown.joblib"


def test_train_alpha_mean_walk_forward_free(tmp_path: Path, stub_train) -> None:
    stub_train(_stub_panel())
    cfg = _cfg(tmp_path)
    result = train_alpha(cfg, "mean")
    assert np.isfinite(result["metrics"]["mean"])
    assert Path(result["path"]).name == "alpha_mean.joblib"
    assert Path(result["path"]).is_file()


def test_split_fold_masks_align_to_dates() -> None:
    dates = np.array(
        [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(6)], dtype=object
    )
    x = np.arange(6.0).reshape(-1, 1)
    y = np.arange(6.0)
    fold = Fold(
        train_times=list(dates[:3]),
        val_times=list(dates[3:4]),
        test_times=list(dates[4:]),
    )
    tr, va, te = _split_fold(dates, x, y, fold)
    assert tr.tolist() == [True, True, True, False, False, False]
    assert va.tolist() == [False, False, False, True, False, False]
    assert te.tolist() == [False, False, False, False, True, True]


def test_fit_ranker_sorts_by_date_for_grouped_models() -> None:
    class _StubRanker:
        def __init__(self) -> None:
            self.seen: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

        def fit(self, x, y, group=None) -> None:  # noqa: ANN001
            self.seen = (np.asarray(x), np.asarray(y), np.asarray(group))

    dates = np.array(
        [
            datetime(2020, 1, 3, tzinfo=UTC),
            datetime(2020, 1, 1, tzinfo=UTC),
            datetime(2020, 1, 1, tzinfo=UTC),
            datetime(2020, 1, 2, tzinfo=UTC),
        ],
        dtype=object,
    )
    x = np.arange(4.0).reshape(-1, 1)
    y = np.arange(4.0)
    model = _StubRanker()
    _fit_ranker(model, "lambdarank", x, y, dates)
    assert model.seen is not None
    seen_x, _seen_y, group = model.seen
    assert seen_x[:, 0].tolist() == [1.0, 2.0, 3.0, 0.0]  # date-sorted order
    assert group.tolist() == [2, 1, 1]
