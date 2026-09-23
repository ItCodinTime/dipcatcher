"""Day Wave 114: name-level GARCH walk-forward QLIKE/density.

Research/infrastructure only — no live broker / vendor MD / live_pnl_claim.
Name-level scores do not replace the date-level market overlay.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from scipy.stats import norm

from quant_fund.config import load_config
from quant_fund.config.models import ValidationConfig
from quant_fund.metrics.scoring import (
    crps_gaussian,
    date_level_equal_weight,
    log_score_gaussian,
    name_level_one_step_density_summary,
    name_level_qlike,
    overlap_aware_qlike,
    qlike,
)
from quant_fund.models.volatility import GARCH_SECURITY_LEVEL_SCOPE
from quant_fund.pipeline.train import (
    _garch_name_oos_predictions,
    garch_name_walk_forward,
)
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent
from quant_fund.schemas.errors import PointInTimeError

RESEARCH_ONLY = True
LIVE_PNL_CLAIM = False


class _NameGarch:
    fit_status = "fitted"
    last_fit_returns: np.ndarray | None = None

    def fit(self, _x, _y, *, returns):  # noqa: ANN001
        self.last_fit_returns = np.asarray(returns, dtype=float).copy()
        self.last_return = float(returns[-1])
        self.last_sigma = 0.02
        return self

    def forecast(self, *, horizon, quantiles=None):  # noqa: ANN001
        sigma = np.full(horizon, self.last_sigma)
        out = {
            "cumulative_variance": np.full(horizon, self.last_return + horizon),
            "variance": np.square(sigma),
            "sigma": sigma,
            "mean": 0.0,
            "distribution": "normal",
            "requested_distribution": "normal",
            "fit_status": self.fit_status,
        }
        if quantiles is not None:
            levels = np.asarray(quantiles, dtype=float)
            out["quantiles"] = self.last_sigma * norm.ppf(levels)[None, :]
        return out

    def log_density(self, returns, sigma=None):  # noqa: ANN001
        values = np.asarray(returns, dtype=float)
        mu = np.zeros(values.size)
        scales = (
            np.full(values.size, self.last_sigma)
            if sigma is None
            else np.asarray(sigma, dtype=float)
        )
        return log_score_gaussian(values, mu, scales)

    def pit(self, returns, sigma=None):  # noqa: ANN001
        values = np.asarray(returns, dtype=float).reshape(-1)
        scales = (
            np.full(values.size, self.last_sigma)
            if sigma is None
            else np.asarray(sigma, dtype=float)
        )
        return norm.cdf(values / scales)


def _two_name_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "event_time": [0, 0, 1, 1, 2, 2, 3, 3],
            "security_id": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "ret_1": [0.01, 0.10, 0.02, 0.20, 0.03, 0.30, 0.04, 0.40],
        }
    )


def test_name_level_qlike_does_not_equal_weight_collapse() -> None:
    sids = np.array(["A", "B", "A", "B"])
    dates = np.array(["d0", "d0", "d1", "d1"], dtype=object)
    forecast = np.array([0.04, 0.16, 0.04, 0.16])
    realized = np.array([0.16, 0.04, 0.16, 0.04])
    named = name_level_qlike(sids, dates, forecast, realized, horizon_bars=1)
    _keys, forecast_d = date_level_equal_weight(dates, forecast)
    _keys, realized_d = date_level_equal_weight(dates, realized)
    collapsed = qlike(realized_d, forecast_d)
    assert named["scoring_scope"] == GARCH_SECURITY_LEVEL_SCOPE
    assert named["n_names"] == 2
    assert named["n_origins_overlapping"] == 4
    assert named["qlike"] == pytest.approx(qlike(realized, forecast))
    assert collapsed == pytest.approx(0.0)
    assert named["qlike"] > collapsed
    with pytest.raises(ValueError, match="unique within each date"):
        overlap_aware_qlike(dates, forecast, realized, horizon_bars=1)
    assert family_blob_forbidden_metrics_absent(named) is True
    assert "live_pnl_claim" not in named


def test_name_level_qlike_stride_is_per_name() -> None:
    sids = np.array(["A", "B", "A", "B", "A", "B"])
    dates = np.array(["d0", "d0", "d1", "d1", "d2", "d2"], dtype=object)
    forecast = np.full(6, 0.04)
    realized = np.array([0.04, 0.09, 0.04, 0.09, 0.04, 0.09])
    scored = name_level_qlike(sids, dates, forecast, realized, horizon_bars=2)
    assert scored["n_origins_overlapping"] == 6
    assert scored["n_origins_nonoverlapping"] == 4
    kept = qlike(np.array([0.04, 0.09, 0.04, 0.09]), np.full(4, 0.04))
    assert scored["qlike"] == pytest.approx(kept)


def test_name_level_qlike_fail_closed_on_duplicate_or_empty() -> None:
    with pytest.raises(ValueError, match="unique"):
        name_level_qlike(
            ["A", "A"],
            np.array(["d0", "d0"], dtype=object),
            np.array([0.04, 0.04]),
            np.array([0.04, 0.04]),
            horizon_bars=1,
        )
    with pytest.raises(ValueError, match="at least one"):
        name_level_qlike([], np.array([], dtype=object), np.array([]), np.array([]), horizon_bars=1)
    with pytest.raises(ValueError, match="strings"):
        name_level_qlike(
            [True],
            np.array(["d0"], dtype=object),
            np.array([0.04]),
            np.array([0.04]),
            horizon_bars=1,
        )


def test_name_level_density_summary_allows_shared_dates() -> None:
    sids = np.array(["A", "B", "A", "B"])
    dates = np.array(["d0", "d0", "d1", "d1"], dtype=object)
    log_scores = np.array([-0.5, -1.5, -0.5, -1.5])
    crps = np.array([0.1, 0.3, 0.1, 0.3])
    pits = np.linspace(0.1, 0.9, 4)
    scored = name_level_one_step_density_summary(
        sids, dates, log_scores, crps, pits, horizon_bars=2
    )
    assert scored["density_target"] == "security_level_ret_1"
    assert scored["scoring_scope"] == GARCH_SECURITY_LEVEL_SCOPE
    assert scored["n_density_origins"] == 4
    assert scored["n_density_origins_qlike_stride"] == 2
    assert scored["n_names"] == 2
    assert scored["log_score_one_step"] == pytest.approx(-1.0)
    assert scored["log_score_one_step_qlike_origins"] == pytest.approx(-1.0)
    assert family_blob_forbidden_metrics_absent(scored) is True


def test_garch_name_oos_uses_per_name_history_not_cross_section() -> None:
    frame = _two_name_frame()
    dates = np.asarray(frame["event_time"].to_numpy())
    ids = np.asarray(frame["security_id"].to_numpy())
    x = np.zeros((frame.height, 1))
    y = np.ones(frame.height)
    test_mask = dates >= 3
    fits: list[list[float]] = []

    class Recording(_NameGarch):
        def fit(self, _x, _y, *, returns):  # noqa: ANN001
            fits.append(np.asarray(returns, dtype=float).tolist())
            return super().fit(_x, _y, returns=returns)

    predictions, statuses, density = _garch_name_oos_predictions(
        Recording,
        x,
        y,
        dates,
        ids,
        test_mask,
        label_horizon=1,
        return_frame=frame,
    )
    assert predictions.tolist() == pytest.approx([1.03, 1.30])
    assert statuses == ["fitted", "fitted"]
    assert fits == [[0.01, 0.02, 0.03], [0.10, 0.20, 0.30]]
    assert ("A", 3) in density and ("B", 3) in density
    assert density[("A", 3)]["y"] == pytest.approx(0.04)
    assert density[("B", 3)]["y"] == pytest.approx(0.40)
    assert 0.04 not in fits[0]
    assert 0.40 not in fits[1]


def test_garch_name_oos_density_scores_name_ret_1_not_variance_label() -> None:
    frame = _two_name_frame()
    dates = np.asarray(frame["event_time"].to_numpy())
    ids = np.asarray(frame["security_id"].to_numpy())
    x = np.zeros((frame.height, 1))
    y = np.full(frame.height, 9.0)
    test_mask = (dates >= 3) & (ids == "A")
    _predictions, _statuses, density = _garch_name_oos_predictions(
        _NameGarch,
        x,
        y,
        dates,
        ids,
        test_mask,
        label_horizon=5,
        return_frame=frame,
    )
    record = density[("A", 3)]
    assert record["y"] == pytest.approx(0.04)
    assert record["y"] != pytest.approx(9.0)
    assert record["density_horizon"] == 1
    assert record["sigma"] == pytest.approx(0.02)
    assert record["sigma"] != pytest.approx(math.sqrt(5.04))
    assert record["crps"] == pytest.approx(
        float(crps_gaussian(np.array([0.04]), np.array([0.0]), np.array([0.02]))[0])
    )


def test_garch_name_oos_fail_closed_when_origin_ret_1_missing() -> None:
    frame = pl.DataFrame(
        {
            "event_time": [0, 1, 2],
            "security_id": ["A", "A", "A"],
            "ret_1": [0.01, 0.02, 0.03],
        }
    )
    dates = np.array([0, 1, 2, 5])
    ids = np.array(["A", "A", "A", "A"])
    with pytest.raises(ValueError, match="missing origin ret_1"):
        _garch_name_oos_predictions(
            _NameGarch,
            np.zeros((4, 1)),
            np.ones(4),
            dates,
            ids,
            np.array([False, False, False, True]),
            label_horizon=1,
            return_frame=frame,
        )


def test_garch_name_oos_fail_closed_without_prior_history() -> None:
    frame = _two_name_frame()
    dates = np.asarray(frame["event_time"].to_numpy())
    ids = np.asarray(frame["security_id"].to_numpy())
    with pytest.raises(ValueError, match="no strictly prior returns"):
        _garch_name_oos_predictions(
            _NameGarch,
            np.zeros((frame.height, 1)),
            np.ones(frame.height),
            dates,
            ids,
            dates == 0,
            label_horizon=1,
            return_frame=frame,
        )


def test_garch_name_oos_duplicate_test_keys_fail_closed() -> None:
    frame = _two_name_frame()
    dates = np.array([3, 3])
    ids = np.array(["A", "A"])
    with pytest.raises(ValueError, match="duplicate"):
        _garch_name_oos_predictions(
            _NameGarch,
            np.zeros((2, 1)),
            np.ones(2),
            dates,
            ids,
            np.array([True, True]),
            label_horizon=1,
            return_frame=frame,
        )


def test_garch_name_oos_ignores_late_available_restatement() -> None:
    start = datetime(2020, 1, 2, 16, 0, 0)
    dates = [start + timedelta(days=i) for i in range(6)]
    rows: list[dict[str, object]] = []
    for stamp in dates:
        rows.append(
            {"event_time": stamp, "security_id": "A", "ret_1": 0.01, "available_time": stamp}
        )
        rows.append(
            {"event_time": stamp, "security_id": "B", "ret_1": 0.02, "available_time": stamp}
        )
    frame = (
        pl.DataFrame(rows)
        .with_columns(
            pl.when((pl.col("security_id") == "B") & (pl.col("event_time") == dates[0]))
            .then(pl.lit(0.80))
            .otherwise(pl.col("ret_1"))
            .alias("ret_1")
        )
        .with_columns(
            pl.when((pl.col("security_id") == "B") & (pl.col("event_time") == dates[0]))
            .then(pl.lit(dates[5]))
            .otherwise(pl.col("available_time"))
            .alias("available_time")
        )
    )
    panel_dates = np.asarray(frame["event_time"].to_numpy())
    ids = np.asarray(frame["security_id"].to_numpy())
    origin = dates[3]
    test_mask = np.array(
        [stamp == origin for stamp in panel_dates.tolist()],
        dtype=bool,
    )
    captured: list[list[float]] = []

    class Capture(_NameGarch):
        def fit(self, _x, _y, *, returns):  # noqa: ANN001
            captured.append(np.asarray(returns, dtype=float).tolist())
            return super().fit(_x, _y, returns=returns)

    _garch_name_oos_predictions(
        Capture,
        np.zeros((frame.height, 1)),
        np.ones(frame.height),
        panel_dates,
        ids,
        test_mask,
        label_horizon=1,
        return_frame=frame,
    )
    assert captured[0] == pytest.approx([0.01, 0.01, 0.01])
    assert 0.80 not in captured[1]
    assert captured[1] == pytest.approx([0.02, 0.02])


def _cfg(tmp_path: Path):
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.fusion.skip_intervals = True
    cfg.validation = ValidationConfig(
        scheme="expanding",
        train_bars=8,
        val_bars=2,
        test_bars=2,
        seed=42,
        embargo_bars=0,
    )
    return cfg


def _labeled_panel(n_days: int = 24, n_names: int = 2) -> pl.DataFrame:
    start = datetime(2020, 1, 2, 16, 0, 0)
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(17)
    for day in range(n_days):
        stamp = start + timedelta(days=day)
        for name in range(n_names):
            vol = 0.008 if name == 0 else 0.04
            rows.append(
                {
                    "event_time": stamp,
                    "security_id": f"S{name:02d}",
                    "ret_1": float(rng.normal(0.0, vol)),
                    "vol_20": vol,
                    "vol_ewma": vol,
                    "vol_parkinson": vol,
                    "vol_of_vol": 0.001,
                    "future_realized_var_5": vol**2,
                    "available_time": stamp,
                }
            )
    return pl.DataFrame(rows)


def test_garch_name_walk_forward_stamps_security_scope(tmp_path: Path) -> None:
    result = garch_name_walk_forward(
        _cfg(tmp_path),
        frame=_labeled_panel(),
        make_model=_NameGarch,
    )
    metrics = result["metrics"]
    assert metrics["scoring_scope"] == GARCH_SECURITY_LEVEL_SCOPE
    assert metrics["density_target"] == "security_level_ret_1"
    assert metrics["n_names"] == 2
    assert metrics["n_origins_overlapping"] >= 2
    assert np.isfinite(metrics["qlike"])
    assert np.isfinite(metrics["log_score_one_step"])
    assert result["diagnostics"]["series_scope"] == GARCH_SECURITY_LEVEL_SCOPE
    assert family_blob_forbidden_metrics_absent(metrics) is True
    assert "live_pnl_claim" not in metrics
    assert "live_pnl_claim" not in result["diagnostics"]


def test_garch_name_walk_forward_explicit_missing_id_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no rows"):
        garch_name_walk_forward(
            _cfg(tmp_path),
            frame=_labeled_panel(),
            security_ids=["S00", "MISSING"],
            make_model=_NameGarch,
        )


def test_garch_name_walk_forward_blank_id_fails_closed(tmp_path: Path) -> None:
    frame = _labeled_panel().with_columns(pl.lit(" ").alias("security_id"))
    with pytest.raises(PointInTimeError, match="blank security_id"):
        garch_name_walk_forward(_cfg(tmp_path), frame=frame, make_model=_NameGarch)
