"""API fail-closed branches: middleware, artifact integrity, risk fallbacks.

Complements test_cli_api by exercising the branches that previously had no
coverage: non-loopback refusal, HSTS, absolute-path allowlist, backtest artifact
rejection paths, forecast distribution envelope, and risk/portfolio fallbacks.
"""

from __future__ import annotations

import importlib
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from quant_fund.api.app import app
from quant_fund.config import load_config
from quant_fund.config.models import AppConfig
from quant_fund.models.covariance import ledoit_wolf_cov, repair_psd
from quant_fund.models.realized_garch import REALIZED_GARCH_MEASURE, RealizedGARCHVol
from quant_fund.models.volatility import GARCH_DATE_LEVEL_SCOPE, GARCHVol
from quant_fund.pipeline.forecast import (
    MARKET_RISK_OVERLAY_GARCH,
    MARKET_RISK_OVERLAY_REALIZED_GARCH,
    GarchMarketForecast,
    RealizedGarchMarketForecast,
    apply_market_variance_overlay_to_covariance,
    clear_forecast_caches,
)
from quant_fund.pipeline.train import _garch_return_history, _realized_garch_history
from quant_fund.portfolio.optimizer import component_risk


def _api():  # noqa: ANN202
    return importlib.import_module("quant_fund.api.app")


def _write_config(configs_dir: Path, tmp_path: Path, extra_yaml: str = "") -> Path:
    configs_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = configs_dir / "research.yaml"
    body = f"data:\n  root: {(tmp_path / 'data').as_posix()}\n"
    if extra_yaml:
        body += extra_yaml
    cfg_path.write_text(body)
    return cfg_path


def test_non_loopback_client_refused_when_key_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QUANT_API_KEY", raising=False)
    client = TestClient(app, client=("203.0.113.5", 50123))
    assert client.get("/health").status_code == 200  # public liveness only
    refused = client.get("/models")
    assert refused.status_code == 403
    assert "non-localhost clients are refused" in refused.json()["detail"]


def test_hsts_header_present_on_https() -> None:
    client = TestClient(app, base_url="https://testserver")
    response = client.get("/health")
    assert response.status_code == 200
    assert response.headers["strict-transport-security"] == ("max-age=31536000; includeSubDomains")


def test_absolute_config_path_allowlisted_and_traversal_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _api()
    configs_dir = tmp_path / "configs"
    cfg_path = _write_config(configs_dir, tmp_path)
    monkeypatch.setattr(api, "_CONFIGS_DIR", configs_dir.resolve())
    # Absolute path under the allowlisted configs dir is accepted.
    assert api.resolve_allowed_config_path(str(cfg_path.resolve())) == cfg_path.resolve()
    outside = tmp_path / "outside.yaml"
    outside.write_text("data:\n  root: data\n")
    with pytest.raises(HTTPException) as excinfo:
        api.resolve_allowed_config_path(str(outside.resolve()))
    assert excinfo.value.status_code == 400


def test_resolve_backtest_artifact_path_rejects_empty(tmp_path: Path) -> None:
    api = _api()
    with pytest.raises(HTTPException) as excinfo:
        api._resolve_backtest_artifact_path(tmp_path, "")
    assert excinfo.value.status_code == 422
    assert "non-empty string" in excinfo.value.detail


def test_forecast_distribution_endpoint_stamps_honesty(monkeypatch: pytest.MonkeyPatch) -> None:
    from quant_fund.schemas.forecast import AssetForecast

    api = _api()
    forecast = AssetForecast(
        security_id="SEC_1",
        symbol="AAA",
        asof=datetime(2026, 1, 1, tzinfo=UTC),
        model_version="test",
        rank_percentile={"5d": 0.9},
        interval_lo={"5d": -0.05},
        interval_hi={"5d": 0.06},
        interval_alpha=0.1,
        interval_method="split_cqr",
    )

    class _State:
        forecasts = [forecast]

    monkeypatch.setattr(api, "_load_cfg", lambda _path: SimpleNamespace())
    monkeypatch.setattr(api, "forecast_asof", lambda _cfg: _State())
    response = TestClient(app).get("/forecast/AAA/distribution")
    assert response.status_code == 200
    body = response.json()
    assert body["research_only"] is True
    assert body["live_pnl_claim"] is False
    assert body["interval_lo"]["5d"] == -0.05
    assert body["interval_method"] == "split_cqr"


def _weights_frame(asof_day: int = 1) -> pl.DataFrame:
    asof = datetime(2026, 1, asof_day, tzinfo=UTC)
    return pl.DataFrame(
        {
            "event_time": [asof, asof],
            "security_id": ["SEC_1", "SEC_2"],
            "target_weight": [0.1, -0.05],
        }
    )


def _risk_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    weights: pl.DataFrame | None,
    panel_frame: pl.DataFrame,
    extra_yaml: str = "",
) -> TestClient:
    api = _api()
    configs_dir = tmp_path / "configs"
    _write_config(configs_dir, tmp_path, extra_yaml=extra_yaml)
    gold = tmp_path / "data" / "gold"
    gold.mkdir(parents=True, exist_ok=True)
    if weights is not None:
        weights.write_parquet(gold / "target_weights.parquet")
    monkeypatch.setattr(api, "_CONFIGS_DIR", configs_dir.resolve())
    import quant_fund.pipeline.dataset as dataset

    monkeypatch.setattr(dataset, "panel", lambda _cfg: panel_frame)
    return TestClient(app)


def test_risk_portfolio_empty_weights_artifact_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = pl.DataFrame(
        {
            "event_time": pl.Series([], dtype=pl.Datetime),
            "security_id": pl.Series([], dtype=pl.Utf8),
            "target_weight": pl.Series([], dtype=pl.Float64),
        }
    )
    client = _risk_client(tmp_path, monkeypatch, weights=empty, panel_frame=empty)
    response = client.get("/risk/portfolio", params={"config_path": "research.yaml"})
    assert response.status_code == 422
    assert "no as-of rows" in response.json()["detail"]


def test_risk_portfolio_missing_return_history_unmeasured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = pl.DataFrame(
        {
            "event_time": [datetime(2026, 1, 1, tzinfo=UTC)],
            "security_id": ["SEC_1"],
            "close": [100.0],
        }
    )
    client = _risk_client(tmp_path, monkeypatch, weights=_weights_frame(), panel_frame=frame)
    body = client.get("/risk/portfolio", params={"config_path": "research.yaml"}).json()
    assert body["status"] == "UNMEASURED"
    assert body["reason"] == "point-in-time return history is unavailable"
    assert body["live_pnl_claim"] is False


def test_risk_portfolio_too_few_securities_unmeasured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = pl.DataFrame(
        {
            "event_time": [datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 2, tzinfo=UTC)],
            "security_id": ["SEC_1", "SEC_1"],
            "ret_1": [0.01, -0.02],
        }
    )
    client = _risk_client(tmp_path, monkeypatch, weights=_weights_frame(), panel_frame=frame)
    body = client.get("/risk/portfolio", params={"config_path": "research.yaml"}).json()
    assert body["status"] == "UNMEASURED"
    assert body["reason"] == "fewer than two securities have point-in-time return history"


def test_risk_portfolio_insufficient_finite_rows_unmeasured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    day_one = datetime(2026, 1, 1, tzinfo=UTC)
    day_two = datetime(2026, 1, 2, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "event_time": [day_one, day_one, day_two, day_two],
            "security_id": ["SEC_1", "SEC_2", "SEC_1", "SEC_2"],
            "ret_1": [float("nan")] * 4,
        }
    )
    client = _risk_client(tmp_path, monkeypatch, weights=_weights_frame(), panel_frame=frame)
    body = client.get("/risk/portfolio", params={"config_path": "research.yaml"}).json()
    assert body["status"] == "UNMEASURED"
    assert body["reason"] == "insufficient finite return observations for covariance"


def test_risk_portfolio_rejects_duplicate_target_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    duplicate = pl.DataFrame(
        {
            "event_time": [datetime(2026, 1, 5, tzinfo=UTC)] * 3,
            "security_id": ["SEC_1", "SEC_1", "SEC_2"],
            "target_weight": [0.1, 0.2, -0.05],
        }
    )
    frame = pl.DataFrame(
        {
            "event_time": [datetime(2026, 1, 1, tzinfo=UTC)] * 2,
            "security_id": ["SEC_1", "SEC_2"],
            "ret_1": [0.01, -0.01],
        }
    )
    client = _risk_client(tmp_path, monkeypatch, weights=duplicate, panel_frame=frame)
    response = client.get("/risk/portfolio", params={"config_path": "research.yaml"})
    assert response.status_code == 422
    assert "duplicate" in response.json()["detail"]


def test_risk_portfolio_measured_component_risk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = []
    for day in range(1, 6):
        rows.append({"event_time": datetime(2026, 1, day, tzinfo=UTC), "security_id": "SEC_1"})
        rows.append({"event_time": datetime(2026, 1, day, tzinfo=UTC), "security_id": "SEC_2"})
    rng_returns = {
        ("SEC_1", 1): 0.01,
        ("SEC_2", 1): -0.01,
        ("SEC_1", 2): 0.02,
        ("SEC_2", 2): 0.00,
        ("SEC_1", 3): -0.01,
        ("SEC_2", 3): 0.015,
        ("SEC_1", 4): 0.005,
        ("SEC_2", 4): -0.02,
        ("SEC_1", 5): 0.012,
        ("SEC_2", 5): 0.004,
    }
    frame = pl.DataFrame(
        {
            "event_time": [row["event_time"] for row in rows],
            "security_id": [row["security_id"] for row in rows],
            "ret_1": [rng_returns[(row["security_id"], row["event_time"].day)] for row in rows],
        }
    )
    client = _risk_client(
        tmp_path, monkeypatch, weights=_weights_frame(asof_day=5), panel_frame=frame
    )
    response = client.get("/risk/portfolio", params={"config_path": "research.yaml"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "MEASURED"
    assert body["securities"] == 2
    assert body["observations"] == 5
    assert body["research_only"] is True
    assert len(body["components"]) == 2
    assert body["predicted_volatility"] >= 0.0
    assert body["market_risk_overlay"] is None
    assert "garch_market_variance" not in body


def test_risk_portfolio_ignores_late_available_return_restatement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    early = datetime(2026, 1, 1, tzinfo=UTC)
    later = datetime(2026, 1, 6, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    for day in range(1, 6):
        stamp = datetime(2026, 1, day, tzinfo=UTC)
        rows.append(
            {
                "event_time": stamp,
                "available_time": stamp,
                "security_id": "SEC_1",
                "ret_1": 0.01 * day,
            }
        )
        rows.append(
            {
                "event_time": stamp,
                "available_time": stamp,
                "security_id": "SEC_2",
                "ret_1": -0.008 * day,
            }
        )
    clean = pl.DataFrame(rows)
    poison = (pl.col("security_id") == "SEC_2") & (pl.col("event_time") == early)
    unpublished = clean.with_columns(
        pl.when(poison).then(pl.lit(8.0)).otherwise(pl.col("ret_1")).alias("ret_1"),
        pl.when(poison)
        .then(pl.lit(later))
        .otherwise(pl.col("available_time"))
        .alias("available_time"),
    )
    published = clean.with_columns(
        pl.when(poison).then(pl.lit(8.0)).otherwise(pl.col("ret_1")).alias("ret_1")
    )
    omitted = clean.filter(~poison)
    weights = _weights_frame(asof_day=5)
    omitted_body = (
        _risk_client(tmp_path, monkeypatch, weights=weights, panel_frame=omitted)
        .get("/risk/portfolio", params={"config_path": "research.yaml"})
        .json()
    )
    dirty_body = (
        _risk_client(tmp_path, monkeypatch, weights=weights, panel_frame=unpublished)
        .get("/risk/portfolio", params={"config_path": "research.yaml"})
        .json()
    )
    published_body = (
        _risk_client(tmp_path, monkeypatch, weights=weights, panel_frame=published)
        .get("/risk/portfolio", params={"config_path": "research.yaml"})
        .json()
    )
    assert omitted_body["status"] == dirty_body["status"] == "MEASURED"
    assert dirty_body["predicted_volatility"] == pytest.approx(omitted_body["predicted_volatility"])
    assert dirty_body["observations"] == omitted_body["observations"] == 4
    assert published_body["status"] == "MEASURED"
    assert published_body["observations"] == 5
    assert published_body["predicted_volatility"] != pytest.approx(
        omitted_body["predicted_volatility"]
    )


def test_risk_portfolio_null_available_time_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stamp = datetime(2026, 1, 1, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "event_time": [stamp, stamp],
            "available_time": [stamp, None],
            "security_id": ["SEC_1", "SEC_2"],
            "ret_1": [0.01, -0.02],
        }
    )
    client = _risk_client(tmp_path, monkeypatch, weights=_weights_frame(), panel_frame=frame)
    response = client.get("/risk/portfolio", params={"config_path": "research.yaml"})
    assert response.status_code == 422
    assert "null available_time" in response.json()["detail"]


def _overlay_panel(n_days: int = 40, *, ohlc: bool = False, seed: int = 17) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    close = {"SEC_1": 100.0, "SEC_2": 101.0}
    for day in range(n_days):
        stamp = start + timedelta(days=day)
        mkt = float(rng.normal(0.0, 0.01))
        for i, sid in enumerate(("SEC_1", "SEC_2")):
            vol = 0.012 + 0.004 * i
            ret = mkt + float(rng.normal(0.0, vol))
            px = close[sid] * math.exp(ret)
            close[sid] = px
            row: dict[str, object] = {
                "event_time": stamp,
                "security_id": sid,
                "ret_1": ret,
            }
            if ohlc:
                span = max(abs(ret), 0.004)
                row["high_split_adjusted"] = px * math.exp(span)
                row["low_split_adjusted"] = px * math.exp(-span)
            rows.append(row)
    return pl.DataFrame(rows)


def _save_garch_artifact(data_root: Path, returns: np.ndarray) -> None:
    path = data_root / "metadata" / "vol_garch.joblib"
    GARCHVol(series_scope=GARCH_DATE_LEVEL_SCOPE, min_obs=20).fit_returns(returns).save(path)


def _save_rgarch_artifact(data_root: Path, returns: np.ndarray, measure: np.ndarray) -> None:
    path = data_root / "metadata" / "vol_realized_garch.joblib"
    RealizedGARCHVol(min_obs=20, series_scope=GARCH_DATE_LEVEL_SCOPE).fit_returns(
        returns, measure
    ).save(path)


def _raw_and_overlaid_vol(
    cfg: AppConfig,
    frame: pl.DataFrame,
    asof: datetime,
    weights: pl.DataFrame,
) -> tuple[
    float,
    float,
    GarchMarketForecast | RealizedGarchMarketForecast | None,
    str | None,
]:
    hist = frame.filter(pl.col("event_time") <= asof)
    wide = hist.select(["event_time", "security_id", "ret_1"]).pivot(
        on="security_id", index="event_time", values="ret_1"
    )
    ids = [str(value) for value in weights["security_id"].to_list()]
    cols = [sid for sid in ids if sid in wide.columns]
    mat = wide.select(cols).to_numpy().astype(float)
    mat = mat[np.isfinite(mat).all(axis=1)]
    sigma, _ = repair_psd(ledoit_wolf_cov(mat), cfg.train.psd_eigen_tol)
    w = np.asarray(weights["target_weight"].to_list(), dtype=float)
    col_idx = [ids.index(sid) for sid in cols]
    _mcr, _cr, raw_vol = component_risk(w[col_idx], sigma)
    overlaid, overlay, kind = apply_market_variance_overlay_to_covariance(cfg, frame, asof, sigma)
    _mcr_o, _cr_o, overlay_vol = component_risk(w[col_idx], overlaid)
    return raw_vol, overlay_vol, overlay, kind


def test_risk_portfolio_applies_garch_overlay_and_stamps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _overlay_panel()
    asof = frame["event_time"].max()
    assert isinstance(asof, datetime)
    weights = _weights_frame().with_columns(pl.lit(asof).alias("event_time"))
    client = _risk_client(tmp_path, monkeypatch, weights=weights, panel_frame=frame)
    _dates, values = _garch_return_history(frame.filter(pl.col("event_time") < asof))
    _save_garch_artifact(tmp_path / "data", values)
    clear_forecast_caches()
    cfg = load_config(tmp_path / "configs" / "research.yaml")
    raw_vol, overlay_vol, overlay, kind = _raw_and_overlaid_vol(cfg, frame, asof, weights)
    assert kind == MARKET_RISK_OVERLAY_GARCH
    assert overlay is not None
    assert overlay_vol != pytest.approx(raw_vol, rel=1e-8, abs=1e-12)
    body = client.get("/risk/portfolio", params={"config_path": "research.yaml"}).json()
    assert body["status"] == "MEASURED"
    assert body["market_risk_overlay"] == MARKET_RISK_OVERLAY_GARCH
    assert body["predicted_volatility"] == pytest.approx(overlay_vol)
    assert body["garch_market_variance"] == pytest.approx(overlay.variance)
    assert body["garch_series_scope"] == GARCH_DATE_LEVEL_SCOPE
    assert body["research_only"] is True
    assert body["live_pnl_claim"] is False


def test_risk_portfolio_prefers_realized_garch_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _overlay_panel(ohlc=True)
    asof = frame["event_time"].max()
    assert isinstance(asof, datetime)
    weights = _weights_frame().with_columns(pl.lit(asof).alias("event_time"))
    client = _risk_client(tmp_path, monkeypatch, weights=weights, panel_frame=frame)
    prior = frame.filter(pl.col("event_time") < asof)
    _dates, values, measures = _realized_garch_history(prior)
    _save_rgarch_artifact(tmp_path / "data", values, measures)
    _save_garch_artifact(tmp_path / "data", values)
    clear_forecast_caches()
    cfg = load_config(tmp_path / "configs" / "research.yaml")
    raw_vol, overlay_vol, overlay, kind = _raw_and_overlaid_vol(cfg, frame, asof, weights)
    assert kind == MARKET_RISK_OVERLAY_REALIZED_GARCH
    assert overlay is not None
    assert overlay_vol != pytest.approx(raw_vol, rel=1e-8, abs=1e-12)
    body = client.get("/risk/portfolio", params={"config_path": "research.yaml"}).json()
    assert body["status"] == "MEASURED"
    assert body["market_risk_overlay"] == MARKET_RISK_OVERLAY_REALIZED_GARCH
    assert body["predicted_volatility"] == pytest.approx(overlay_vol)
    assert body["garch_market_variance"] == pytest.approx(overlay.variance)
    assert body["realized_measure"] == REALIZED_GARCH_MEASURE
    assert body["intraday_realized_variance"] is False


def test_risk_portfolio_named_dcc_does_not_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import patch

    from quant_fund.models.covariance import (
        DCC_COVARIANCE_OBJECT_ONE_STEP,
        DCC_FAMILY_GAUSSIAN,
        DCC_SPEC_ENGLE_2002,
    )

    frame = _overlay_panel(n_days=60)
    asof = frame["event_time"].max()
    assert isinstance(asof, datetime)
    weights = _weights_frame().with_columns(pl.lit(asof).alias("event_time"))
    client = _risk_client(
        tmp_path,
        monkeypatch,
        weights=weights,
        panel_frame=frame,
        extra_yaml="optimizer:\n  covariance: dcc_gaussian\n",
    )
    _dates, values = _garch_return_history(frame.filter(pl.col("event_time") < asof))
    _save_garch_artifact(tmp_path / "data", values)
    clear_forecast_caches()
    fake = np.eye(2) * 0.0004
    w = np.asarray(weights["target_weight"].to_list(), dtype=float)
    _mcr, _cr, expected_vol = component_risk(w, fake)

    def fake_dcc(returns: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
        x = np.asarray(returns, dtype=float)
        return fake, {
            "family": DCC_FAMILY_GAUSSIAN,
            "spec": DCC_SPEC_ENGLE_2002,
            "covariance_object": DCC_COVARIANCE_OBJECT_ONE_STEP,
            "n_obs": float(np.isfinite(x).all(axis=1).sum()),
            "n_assets": float(x.shape[1]),
        }

    with (
        patch("quant_fund.pipeline.forecast.dcc_gaussian", side_effect=fake_dcc),
        patch(
            "quant_fund.pipeline.forecast.apply_market_variance_overlay_to_covariance",
            side_effect=AssertionError("named DCC must not overlay H_{t+1}"),
        ),
    ):
        body = client.get("/risk/portfolio", params={"config_path": "research.yaml"}).json()
    assert body["status"] == "MEASURED"
    assert body["covariance_estimator"] == DCC_FAMILY_GAUSSIAN
    assert body["covariance_object"] == DCC_COVARIANCE_OBJECT_ONE_STEP
    assert body["covariance_spec"] == DCC_SPEC_ENGLE_2002
    assert body["market_risk_overlay"] is None
    assert "garch_market_variance" not in body
    assert body["predicted_volatility"] == pytest.approx(expected_vol)
    assert body["research_only"] is True
    assert body["live_pnl_claim"] is False


def test_risk_portfolio_named_dcc_short_history_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _overlay_panel(n_days=10)
    asof = frame["event_time"].max()
    assert isinstance(asof, datetime)
    weights = _weights_frame().with_columns(pl.lit(asof).alias("event_time"))
    client = _risk_client(
        tmp_path,
        monkeypatch,
        weights=weights,
        panel_frame=frame,
        extra_yaml="optimizer:\n  covariance: dcc_gaussian\n",
    )
    response = client.get("/risk/portfolio", params={"config_path": "research.yaml"})
    assert response.status_code == 422
    assert "optimizer_covariance_failed:dcc_gaussian" in response.json()["detail"]


def test_risk_portfolio_named_student_t_dcc_does_not_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import patch

    from quant_fund.models.covariance import (
        DCC_COVARIANCE_OBJECT_ONE_STEP,
        DCC_FAMILY_STUDENT_T,
        DCC_SPEC_STUDENT_T,
    )

    frame = _overlay_panel(n_days=60)
    asof = frame["event_time"].max()
    assert isinstance(asof, datetime)
    weights = _weights_frame().with_columns(pl.lit(asof).alias("event_time"))
    client = _risk_client(
        tmp_path,
        monkeypatch,
        weights=weights,
        panel_frame=frame,
        extra_yaml="optimizer:\n  covariance: dcc_student_t\n",
    )
    _dates, values = _garch_return_history(frame.filter(pl.col("event_time") < asof))
    _save_garch_artifact(tmp_path / "data", values)
    clear_forecast_caches()
    fake = np.eye(2) * 0.0009
    w = np.asarray(weights["target_weight"].to_list(), dtype=float)
    _mcr, _cr, expected_vol = component_risk(w, fake)

    def fake_dcc(returns: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
        x = np.asarray(returns, dtype=float)
        return fake, {
            "family": DCC_FAMILY_STUDENT_T,
            "spec": DCC_SPEC_STUDENT_T,
            "covariance_object": DCC_COVARIANCE_OBJECT_ONE_STEP,
            "n_obs": float(np.isfinite(x).all(axis=1).sum()),
            "n_assets": float(x.shape[1]),
        }

    with (
        patch("quant_fund.pipeline.forecast.dcc_student_t", side_effect=fake_dcc),
        patch(
            "quant_fund.pipeline.forecast.dcc_gaussian",
            side_effect=AssertionError("student-t DCC must not silently run dcc_gaussian"),
        ),
        patch(
            "quant_fund.pipeline.forecast.apply_market_variance_overlay_to_covariance",
            side_effect=AssertionError("named DCC must not overlay H_{t+1}"),
        ),
    ):
        body = client.get("/risk/portfolio", params={"config_path": "research.yaml"}).json()
    assert body["status"] == "MEASURED"
    assert body["covariance_estimator"] == DCC_FAMILY_STUDENT_T
    assert body["covariance_object"] == DCC_COVARIANCE_OBJECT_ONE_STEP
    assert body["covariance_spec"] == DCC_SPEC_STUDENT_T
    assert body["covariance_horizon"] == 1
    assert body["market_risk_overlay"] is None
    assert "garch_market_variance" not in body
    assert body["predicted_volatility"] == pytest.approx(expected_vol)
    assert body["research_only"] is True
    assert body["live_pnl_claim"] is False


def test_risk_portfolio_named_student_t_dcc_short_history_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _overlay_panel(n_days=10)
    asof = frame["event_time"].max()
    assert isinstance(asof, datetime)
    weights = _weights_frame().with_columns(pl.lit(asof).alias("event_time"))
    client = _risk_client(
        tmp_path,
        monkeypatch,
        weights=weights,
        panel_frame=frame,
        extra_yaml="optimizer:\n  covariance: dcc_student_t\n",
    )
    response = client.get("/risk/portfolio", params={"config_path": "research.yaml"})
    assert response.status_code == 422
    assert "optimizer_covariance_failed:dcc_student_t" in response.json()["detail"]


def test_risk_portfolio_named_adcc_does_not_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import patch

    from quant_fund.models.covariance import (
        DCC_COVARIANCE_OBJECT_ONE_STEP,
        DCC_FAMILY_ADCC,
        DCC_SPEC_ADCC,
    )

    frame = _overlay_panel(n_days=60)
    asof = frame["event_time"].max()
    assert isinstance(asof, datetime)
    weights = _weights_frame().with_columns(pl.lit(asof).alias("event_time"))
    client = _risk_client(
        tmp_path,
        monkeypatch,
        weights=weights,
        panel_frame=frame,
        extra_yaml="optimizer:\n  covariance: adcc\n",
    )
    _dates, values = _garch_return_history(frame.filter(pl.col("event_time") < asof))
    _save_garch_artifact(tmp_path / "data", values)
    clear_forecast_caches()
    fake = np.eye(2) * 0.0016
    w = np.asarray(weights["target_weight"].to_list(), dtype=float)
    _mcr, _cr, expected_vol = component_risk(w, fake)

    def fake_adcc(returns: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
        x = np.asarray(returns, dtype=float)
        return fake, {
            "family": DCC_FAMILY_ADCC,
            "spec": DCC_SPEC_ADCC,
            "covariance_object": DCC_COVARIANCE_OBJECT_ONE_STEP,
            "n_obs": float(np.isfinite(x).all(axis=1).sum()),
            "n_assets": float(x.shape[1]),
        }

    with (
        patch("quant_fund.pipeline.forecast.adcc", side_effect=fake_adcc),
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
            side_effect=AssertionError("named DCC must not overlay H_{t+1}"),
        ),
    ):
        body = client.get("/risk/portfolio", params={"config_path": "research.yaml"}).json()
    assert body["status"] == "MEASURED"
    assert body["covariance_estimator"] == DCC_FAMILY_ADCC
    assert body["covariance_object"] == DCC_COVARIANCE_OBJECT_ONE_STEP
    assert body["covariance_spec"] == DCC_SPEC_ADCC
    assert body["covariance_horizon"] == 1
    assert body["market_risk_overlay"] is None
    assert "garch_market_variance" not in body
    assert body["predicted_volatility"] == pytest.approx(expected_vol)
    assert body["research_only"] is True
    assert body["live_pnl_claim"] is False


def test_risk_portfolio_named_adcc_short_history_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _overlay_panel(n_days=10)
    asof = frame["event_time"].max()
    assert isinstance(asof, datetime)
    weights = _weights_frame().with_columns(pl.lit(asof).alias("event_time"))
    client = _risk_client(
        tmp_path,
        monkeypatch,
        weights=weights,
        panel_frame=frame,
        extra_yaml="optimizer:\n  covariance: adcc\n",
    )
    response = client.get("/risk/portfolio", params={"config_path": "research.yaml"})
    assert response.status_code == 422
    assert "optimizer_covariance_failed:adcc" in response.json()["detail"]


def test_risk_portfolio_named_ccc_does_not_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import patch

    from quant_fund.models.covariance import (
        DCC_COVARIANCE_OBJECT_ONE_STEP,
        DCC_FAMILY_CCC,
        DCC_SPEC_CCC,
    )

    frame = _overlay_panel(n_days=60)
    asof = frame["event_time"].max()
    assert isinstance(asof, datetime)
    weights = _weights_frame().with_columns(pl.lit(asof).alias("event_time"))
    client = _risk_client(
        tmp_path,
        monkeypatch,
        weights=weights,
        panel_frame=frame,
        extra_yaml="optimizer:\n  covariance: ccc\n",
    )
    _dates, values = _garch_return_history(frame.filter(pl.col("event_time") < asof))
    _save_garch_artifact(tmp_path / "data", values)
    clear_forecast_caches()
    fake = np.eye(2) * 0.0025
    w = np.asarray(weights["target_weight"].to_list(), dtype=float)
    _mcr, _cr, expected_vol = component_risk(w, fake)

    def fake_ccc(returns: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
        x = np.asarray(returns, dtype=float)
        return fake, {
            "family": DCC_FAMILY_CCC,
            "spec": DCC_SPEC_CCC,
            "covariance_object": DCC_COVARIANCE_OBJECT_ONE_STEP,
            "n_obs": float(np.isfinite(x).all(axis=1).sum()),
            "n_assets": float(x.shape[1]),
        }

    with (
        patch("quant_fund.pipeline.forecast.ccc", side_effect=fake_ccc),
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
            side_effect=AssertionError("named CCC must not overlay H_{t+1}"),
        ),
    ):
        body = client.get("/risk/portfolio", params={"config_path": "research.yaml"}).json()
    assert body["status"] == "MEASURED"
    assert body["covariance_estimator"] == DCC_FAMILY_CCC
    assert body["covariance_object"] == DCC_COVARIANCE_OBJECT_ONE_STEP
    assert body["covariance_spec"] == DCC_SPEC_CCC
    assert body["covariance_horizon"] == 1
    assert body["market_risk_overlay"] is None
    assert "garch_market_variance" not in body
    assert body["predicted_volatility"] == pytest.approx(expected_vol)
    assert body["research_only"] is True
    assert body["live_pnl_claim"] is False


def test_risk_portfolio_named_ccc_short_history_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _overlay_panel(n_days=10)
    asof = frame["event_time"].max()
    assert isinstance(asof, datetime)
    weights = _weights_frame().with_columns(pl.lit(asof).alias("event_time"))
    client = _risk_client(
        tmp_path,
        monkeypatch,
        weights=weights,
        panel_frame=frame,
        extra_yaml="optimizer:\n  covariance: ccc\n",
    )
    response = client.get("/risk/portfolio", params={"config_path": "research.yaml"})
    assert response.status_code == 422
    assert "optimizer_covariance_failed:ccc" in response.json()["detail"]


def test_risk_portfolio_named_agdcc_does_not_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import patch

    from quant_fund.models.covariance import (
        DCC_COVARIANCE_OBJECT_ONE_STEP,
        DCC_FAMILY_AGDCC,
        DCC_SPEC_AGDCC,
    )

    frame = _overlay_panel(n_days=60)
    asof = frame["event_time"].max()
    assert isinstance(asof, datetime)
    weights = _weights_frame().with_columns(pl.lit(asof).alias("event_time"))
    client = _risk_client(
        tmp_path,
        monkeypatch,
        weights=weights,
        panel_frame=frame,
        extra_yaml="optimizer:\n  covariance: agdcc\n",
    )
    _dates, values = _garch_return_history(frame.filter(pl.col("event_time") < asof))
    _save_garch_artifact(tmp_path / "data", values)
    clear_forecast_caches()
    fake = np.eye(2) * 0.0036
    w = np.asarray(weights["target_weight"].to_list(), dtype=float)
    _mcr, _cr, expected_vol = component_risk(w, fake)

    def fake_agdcc(returns: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
        x = np.asarray(returns, dtype=float)
        return fake, {
            "family": DCC_FAMILY_AGDCC,
            "spec": DCC_SPEC_AGDCC,
            "parameterization": "diagonal",
            "covariance_object": DCC_COVARIANCE_OBJECT_ONE_STEP,
            "n_obs": float(np.isfinite(x).all(axis=1).sum()),
            "n_assets": float(x.shape[1]),
        }

    with (
        patch("quant_fund.pipeline.forecast.agdcc", side_effect=fake_agdcc),
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
            side_effect=AssertionError("named AG-DCC must not overlay H_{t+1}"),
        ),
    ):
        body = client.get("/risk/portfolio", params={"config_path": "research.yaml"}).json()
    assert body["status"] == "MEASURED"
    assert body["covariance_estimator"] == DCC_FAMILY_AGDCC
    assert body["covariance_object"] == DCC_COVARIANCE_OBJECT_ONE_STEP
    assert body["covariance_spec"] == DCC_SPEC_AGDCC
    assert body["covariance_horizon"] == 1
    assert body["market_risk_overlay"] is None
    assert "garch_market_variance" not in body
    assert body["predicted_volatility"] == pytest.approx(expected_vol)
    assert body["research_only"] is True
    assert body["live_pnl_claim"] is False


def test_risk_portfolio_named_agdcc_short_history_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _overlay_panel(n_days=10)
    asof = frame["event_time"].max()
    assert isinstance(asof, datetime)
    weights = _weights_frame().with_columns(pl.lit(asof).alias("event_time"))
    client = _risk_client(
        tmp_path,
        monkeypatch,
        weights=weights,
        panel_frame=frame,
        extra_yaml="optimizer:\n  covariance: agdcc\n",
    )
    response = client.get("/risk/portfolio", params={"config_path": "research.yaml"})
    assert response.status_code == 422
    assert "optimizer_covariance_failed:agdcc" in response.json()["detail"]


def test_risk_portfolio_named_agdcc_full_does_not_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import patch

    from quant_fund.models.covariance import (
        DCC_COVARIANCE_OBJECT_ONE_STEP,
        DCC_FAMILY_AGDCC_FULL,
        DCC_SPEC_AGDCC_FULL,
    )

    frame = _overlay_panel(n_days=60)
    asof = frame["event_time"].max()
    assert isinstance(asof, datetime)
    weights = _weights_frame().with_columns(pl.lit(asof).alias("event_time"))
    client = _risk_client(
        tmp_path,
        monkeypatch,
        weights=weights,
        panel_frame=frame,
        extra_yaml="optimizer:\n  covariance: agdcc_full\n",
    )
    _dates, values = _garch_return_history(frame.filter(pl.col("event_time") < asof))
    _save_garch_artifact(tmp_path / "data", values)
    clear_forecast_caches()
    fake = np.eye(2) * 0.0049
    w = np.asarray(weights["target_weight"].to_list(), dtype=float)
    _mcr, _cr, expected_vol = component_risk(w, fake)

    def fake_agdcc_full(returns: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
        x = np.asarray(returns, dtype=float)
        return fake, {
            "family": DCC_FAMILY_AGDCC_FULL,
            "spec": DCC_SPEC_AGDCC_FULL,
            "parameterization": "full",
            "covariance_object": DCC_COVARIANCE_OBJECT_ONE_STEP,
            "n_obs": float(np.isfinite(x).all(axis=1).sum()),
            "n_assets": float(x.shape[1]),
        }

    with (
        patch("quant_fund.pipeline.forecast.agdcc_full", side_effect=fake_agdcc_full),
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
            side_effect=AssertionError("named unrestricted AG-DCC must not overlay H_{t+1}"),
        ),
    ):
        body = client.get("/risk/portfolio", params={"config_path": "research.yaml"}).json()
    assert body["status"] == "MEASURED"
    assert body["covariance_estimator"] == DCC_FAMILY_AGDCC_FULL
    assert body["covariance_object"] == DCC_COVARIANCE_OBJECT_ONE_STEP
    assert body["covariance_spec"] == DCC_SPEC_AGDCC_FULL
    assert body["covariance_horizon"] == 1
    assert body["market_risk_overlay"] is None
    assert "garch_market_variance" not in body
    assert body["predicted_volatility"] == pytest.approx(expected_vol)
    assert body["research_only"] is True
    assert body["live_pnl_claim"] is False


def test_risk_portfolio_named_agdcc_full_short_history_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _overlay_panel(n_days=10)
    asof = frame["event_time"].max()
    assert isinstance(asof, datetime)
    weights = _weights_frame().with_columns(pl.lit(asof).alias("event_time"))
    client = _risk_client(
        tmp_path,
        monkeypatch,
        weights=weights,
        panel_frame=frame,
        extra_yaml="optimizer:\n  covariance: agdcc_full\n",
    )
    response = client.get("/risk/portfolio", params={"config_path": "research.yaml"})
    assert response.status_code == 422
    assert "optimizer_covariance_failed:agdcc_full" in response.json()["detail"]


def test_risk_portfolio_named_ewma_does_not_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import patch

    from quant_fund.models.covariance import (
        DCC_COVARIANCE_OBJECT_ONE_STEP,
        EWMA_SPEC_RISKMETRICS,
        OPTIMIZER_COVARIANCE_EWMA,
    )

    frame = _overlay_panel(n_days=40)
    asof = frame["event_time"].max()
    assert isinstance(asof, datetime)
    weights = _weights_frame().with_columns(pl.lit(asof).alias("event_time"))
    client = _risk_client(
        tmp_path,
        monkeypatch,
        weights=weights,
        panel_frame=frame,
        extra_yaml="optimizer:\n  covariance: ewma\n",
    )
    fake = np.eye(2) * 0.0009
    w = np.asarray(weights["target_weight"].to_list(), dtype=float)
    _mcr, _cr, expected_vol = component_risk(w, fake)

    def fake_ewma(
        returns: np.ndarray, lam: float = 0.94
    ) -> tuple[np.ndarray, dict[str, float | str]]:
        x = np.asarray(returns, dtype=float)
        return fake, {
            "family": OPTIMIZER_COVARIANCE_EWMA,
            "spec": EWMA_SPEC_RISKMETRICS,
            "lambda": float(lam),
            "covariance_object": DCC_COVARIANCE_OBJECT_ONE_STEP,
            "n_obs": float(np.isfinite(x).all(axis=1).sum()),
            "n_assets": float(x.shape[1]),
        }

    with (
        patch("quant_fund.pipeline.forecast.ewma", side_effect=fake_ewma),
        patch(
            "quant_fund.pipeline.forecast.dcc_gaussian",
            side_effect=AssertionError("EWMA must not silently run dcc_gaussian"),
        ),
        patch(
            "quant_fund.pipeline.forecast.apply_market_variance_overlay_to_covariance",
            side_effect=AssertionError("named EWMA must not overlay H_{t+1}"),
        ),
    ):
        body = client.get("/risk/portfolio", params={"config_path": "research.yaml"}).json()
    assert body["status"] == "MEASURED"
    assert body["covariance_estimator"] == OPTIMIZER_COVARIANCE_EWMA
    assert body["covariance_object"] == DCC_COVARIANCE_OBJECT_ONE_STEP
    assert body["covariance_spec"] == EWMA_SPEC_RISKMETRICS
    assert body["covariance_horizon"] == 1
    assert body["market_risk_overlay"] is None
    assert "garch_market_variance" not in body
    assert body["predicted_volatility"] == pytest.approx(expected_vol)
    assert body["research_only"] is True
    assert body["live_pnl_claim"] is False


def test_risk_portfolio_named_ewma_short_history_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _overlay_panel(n_days=10)
    asof = frame["event_time"].max()
    assert isinstance(asof, datetime)
    weights = _weights_frame().with_columns(pl.lit(asof).alias("event_time"))
    client = _risk_client(
        tmp_path,
        monkeypatch,
        weights=weights,
        panel_frame=frame,
        extra_yaml="optimizer:\n  covariance: ewma\n",
    )
    response = client.get("/risk/portfolio", params={"config_path": "research.yaml"})
    assert response.status_code == 422
    assert "optimizer_covariance_failed:ewma" in response.json()["detail"]


def test_risk_portfolio_named_oas_uses_trailing_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import patch

    from quant_fund.models.covariance import (
        OAS_SPEC_CHEN_2010,
        OPTIMIZER_COVARIANCE_OAS,
        OPTIMIZER_COVARIANCE_OBJECT_TRAILING,
    )

    frame = _overlay_panel(n_days=40)
    asof = frame["event_time"].max()
    assert isinstance(asof, datetime)
    weights = _weights_frame().with_columns(pl.lit(asof).alias("event_time"))
    client = _risk_client(
        tmp_path,
        monkeypatch,
        weights=weights,
        panel_frame=frame,
        extra_yaml="optimizer:\n  covariance: oas\n",
    )
    fake = np.eye(2) * 0.0004
    overlaid = fake * 4.0
    w = np.asarray(weights["target_weight"].to_list(), dtype=float)
    _mcr, _cr, expected_vol = component_risk(w, overlaid)

    def fake_oas(returns: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
        x = np.asarray(returns, dtype=float)
        return fake, {
            "family": OPTIMIZER_COVARIANCE_OAS,
            "spec": OAS_SPEC_CHEN_2010,
            "covariance_object": OPTIMIZER_COVARIANCE_OBJECT_TRAILING,
            "n_obs": float(np.isfinite(x).all(axis=1).sum()),
            "n_assets": float(x.shape[1]),
        }

    def scale_overlay(_config, _frame, _asof, sigma):
        return np.asarray(sigma, dtype=float) * 4.0, None, None

    with (
        patch("quant_fund.pipeline.forecast.oas", side_effect=fake_oas),
        patch(
            "quant_fund.pipeline.forecast.ledoit_wolf",
            side_effect=AssertionError("OAS must not silently run Ledoit-Wolf"),
        ),
        patch(
            "quant_fund.pipeline.forecast.ewma",
            side_effect=AssertionError("OAS must not silently run ewma"),
        ),
        patch(
            "quant_fund.pipeline.forecast.apply_market_variance_overlay_to_covariance",
            side_effect=scale_overlay,
        ),
    ):
        body = client.get("/risk/portfolio", params={"config_path": "research.yaml"}).json()
    assert body["status"] == "MEASURED"
    assert body["covariance_estimator"] == OPTIMIZER_COVARIANCE_OAS
    assert body["covariance_object"] == OPTIMIZER_COVARIANCE_OBJECT_TRAILING
    assert body["covariance_spec"] == OAS_SPEC_CHEN_2010
    assert "covariance_horizon" not in body
    assert body["predicted_volatility"] == pytest.approx(expected_vol)
    assert body["research_only"] is True
    assert body["live_pnl_claim"] is False


def test_risk_portfolio_named_oas_short_history_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _overlay_panel(n_days=1)
    asof = frame["event_time"].max()
    assert isinstance(asof, datetime)
    weights = _weights_frame().with_columns(pl.lit(asof).alias("event_time"))
    client = _risk_client(
        tmp_path,
        monkeypatch,
        weights=weights,
        panel_frame=frame,
        extra_yaml="optimizer:\n  covariance: oas\n",
    )
    response = client.get("/risk/portfolio", params={"config_path": "research.yaml"})
    assert response.status_code == 422
    assert "optimizer_covariance_failed:oas" in response.json()["detail"]


def test_risk_portfolio_named_sample_uses_trailing_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import patch

    from quant_fund.models.covariance import (
        OPTIMIZER_COVARIANCE_OBJECT_TRAILING,
        OPTIMIZER_COVARIANCE_SAMPLE,
        SAMPLE_SPEC_UNBIASED,
    )

    frame = _overlay_panel(n_days=40)
    asof = frame["event_time"].max()
    assert isinstance(asof, datetime)
    weights = _weights_frame().with_columns(pl.lit(asof).alias("event_time"))
    client = _risk_client(
        tmp_path,
        monkeypatch,
        weights=weights,
        panel_frame=frame,
        extra_yaml="optimizer:\n  covariance: sample\n",
    )
    fake = np.eye(2) * 0.0004
    overlaid = fake * 4.0
    w = np.asarray(weights["target_weight"].to_list(), dtype=float)
    _mcr, _cr, expected_vol = component_risk(w, overlaid)

    def fake_sample(returns: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
        x = np.asarray(returns, dtype=float)
        return fake, {
            "family": OPTIMIZER_COVARIANCE_SAMPLE,
            "spec": SAMPLE_SPEC_UNBIASED,
            "covariance_object": OPTIMIZER_COVARIANCE_OBJECT_TRAILING,
            "n_obs": float(np.isfinite(x).all(axis=1).sum()),
            "n_assets": float(x.shape[1]),
        }

    def scale_overlay(_config, _frame, _asof, sigma):
        return np.asarray(sigma, dtype=float) * 4.0, None, None

    with (
        patch("quant_fund.pipeline.forecast.sample", side_effect=fake_sample),
        patch(
            "quant_fund.pipeline.forecast.ledoit_wolf",
            side_effect=AssertionError("sample must not silently run Ledoit-Wolf"),
        ),
        patch(
            "quant_fund.pipeline.forecast.oas",
            side_effect=AssertionError("sample must not silently run OAS"),
        ),
        patch(
            "quant_fund.pipeline.forecast.ewma",
            side_effect=AssertionError("sample must not silently run ewma"),
        ),
        patch(
            "quant_fund.pipeline.forecast.apply_market_variance_overlay_to_covariance",
            side_effect=scale_overlay,
        ),
    ):
        body = client.get("/risk/portfolio", params={"config_path": "research.yaml"}).json()
    assert body["status"] == "MEASURED"
    assert body["covariance_estimator"] == OPTIMIZER_COVARIANCE_SAMPLE
    assert body["covariance_object"] == OPTIMIZER_COVARIANCE_OBJECT_TRAILING
    assert body["covariance_spec"] == SAMPLE_SPEC_UNBIASED
    assert "covariance_horizon" not in body
    assert body["predicted_volatility"] == pytest.approx(expected_vol)
    assert body["research_only"] is True
    assert body["live_pnl_claim"] is False


def test_risk_portfolio_named_sample_short_history_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _overlay_panel(n_days=1)
    asof = frame["event_time"].max()
    assert isinstance(asof, datetime)
    weights = _weights_frame().with_columns(pl.lit(asof).alias("event_time"))
    client = _risk_client(
        tmp_path,
        monkeypatch,
        weights=weights,
        panel_frame=frame,
        extra_yaml="optimizer:\n  covariance: sample\n",
    )
    response = client.get("/risk/portfolio", params={"config_path": "research.yaml"})
    assert response.status_code == 422
    assert "optimizer_covariance_failed:sample" in response.json()["detail"]


def test_risk_portfolio_named_nonlinear_uses_trailing_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import patch

    from quant_fund.models.covariance import (
        OPTIMIZER_COVARIANCE_LEDOIT_WOLF_NONLINEAR,
        OPTIMIZER_COVARIANCE_OBJECT_TRAILING,
        OPTIMIZER_COVARIANCE_SPEC_LEDOIT_WOLF_NONLINEAR,
    )

    frame = _overlay_panel(n_days=40)
    asof = frame["event_time"].max()
    assert isinstance(asof, datetime)
    weights = _weights_frame().with_columns(pl.lit(asof).alias("event_time"))
    client = _risk_client(
        tmp_path,
        monkeypatch,
        weights=weights,
        panel_frame=frame,
        extra_yaml="optimizer:\n  covariance: ledoit_wolf_nonlinear\n",
    )
    fake = np.eye(2) * 0.0004
    overlaid = fake * 4.0
    w = np.asarray(weights["target_weight"].to_list(), dtype=float)
    _mcr, _cr, expected_vol = component_risk(w, overlaid)

    def fake_nl(returns: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
        x = np.asarray(returns, dtype=float)
        return fake, {
            "family": OPTIMIZER_COVARIANCE_LEDOIT_WOLF_NONLINEAR,
            "spec": OPTIMIZER_COVARIANCE_SPEC_LEDOIT_WOLF_NONLINEAR,
            "covariance_object": OPTIMIZER_COVARIANCE_OBJECT_TRAILING,
            "n_obs": float(np.isfinite(x).all(axis=1).sum()),
            "n_assets": float(x.shape[1]),
        }

    def scale_overlay(_config, _frame, _asof, sigma):
        return np.asarray(sigma, dtype=float) * 4.0, None, None

    with (
        patch("quant_fund.pipeline.forecast.ledoit_wolf_nonlinear", side_effect=fake_nl),
        patch(
            "quant_fund.pipeline.forecast.ledoit_wolf",
            side_effect=AssertionError("nonlinear must not silently run 2004 Ledoit-Wolf"),
        ),
        patch(
            "quant_fund.pipeline.forecast.oas",
            side_effect=AssertionError("nonlinear must not silently run OAS"),
        ),
        patch(
            "quant_fund.pipeline.forecast.ewma",
            side_effect=AssertionError("nonlinear must not silently run ewma"),
        ),
        patch(
            "quant_fund.pipeline.forecast.apply_market_variance_overlay_to_covariance",
            side_effect=scale_overlay,
        ),
    ):
        body = client.get("/risk/portfolio", params={"config_path": "research.yaml"}).json()
    assert body["status"] == "MEASURED"
    assert body["covariance_estimator"] == OPTIMIZER_COVARIANCE_LEDOIT_WOLF_NONLINEAR
    assert body["covariance_object"] == OPTIMIZER_COVARIANCE_OBJECT_TRAILING
    assert body["covariance_spec"] == OPTIMIZER_COVARIANCE_SPEC_LEDOIT_WOLF_NONLINEAR
    assert "covariance_horizon" not in body
    assert body["predicted_volatility"] == pytest.approx(expected_vol)
    assert body["research_only"] is True
    assert body["live_pnl_claim"] is False


def test_risk_portfolio_named_nonlinear_short_history_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _overlay_panel(n_days=1)
    asof = frame["event_time"].max()
    assert isinstance(asof, datetime)
    weights = _weights_frame().with_columns(pl.lit(asof).alias("event_time"))
    client = _risk_client(
        tmp_path,
        monkeypatch,
        weights=weights,
        panel_frame=frame,
        extra_yaml="optimizer:\n  covariance: ledoit_wolf_nonlinear\n",
    )
    response = client.get("/risk/portfolio", params={"config_path": "research.yaml"})
    assert response.status_code == 422
    assert "optimizer_covariance_failed:ledoit_wolf_nonlinear" in response.json()["detail"]


def test_risk_portfolio_present_rgarch_missing_ohlc_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ohlc = _overlay_panel(ohlc=True)
    asof = ohlc["event_time"].max()
    assert isinstance(asof, datetime)
    prior = ohlc.filter(pl.col("event_time") < asof)
    _dates, values, measures = _realized_garch_history(prior)
    stripped = ohlc.drop("high_split_adjusted", "low_split_adjusted")
    weights = _weights_frame().with_columns(pl.lit(asof).alias("event_time"))
    client = _risk_client(tmp_path, monkeypatch, weights=weights, panel_frame=stripped)
    _save_rgarch_artifact(tmp_path / "data", values, measures)
    _save_garch_artifact(tmp_path / "data", values)
    clear_forecast_caches()
    response = client.get("/risk/portfolio", params={"config_path": "research.yaml"})
    assert response.status_code == 422
    assert "OHLC" in response.json()["detail"]


def test_research_latest_missing_receipt_is_404(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _api()
    configs_dir = tmp_path / "configs"
    _write_config(configs_dir, tmp_path)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(api, "_CONFIGS_DIR", configs_dir.resolve())
    response = TestClient(app).get("/research/latest", params={"config_path": "research.yaml"})
    assert response.status_code == 404
    assert "run `quant research` first" in response.json()["detail"]


def test_research_latest_invalid_receipt_is_422(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _api()
    configs_dir = tmp_path / "configs"
    _write_config(configs_dir, tmp_path)
    receipt_dir = tmp_path / "data" / "metadata" / "research"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    (receipt_dir / "latest.json").write_text(json.dumps({"not": "a notebook"}))
    monkeypatch.setattr(api, "_CONFIGS_DIR", configs_dir.resolve())
    response = TestClient(app).get("/research/latest", params={"config_path": "research.yaml"})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["message"] == "latest research receipt failed integrity verification"
    assert detail["errors"]


def test_drift_without_receipt_reports_none_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _api()
    configs_dir = tmp_path / "configs"
    _write_config(configs_dir, tmp_path)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(api, "_CONFIGS_DIR", configs_dir.resolve())
    response = TestClient(app).get("/monitoring/drift", params={"config_path": "research.yaml"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "UNMEASURED"
    assert body["source"] == "none"
    assert body["research_only"] is True


def _backtest_artifact_dir(tmp_path: Path) -> Path:
    artifact_dir = tmp_path / "metadata" / "backtests"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def _artifact_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, artifact_id: str = "a" * 16
) -> tuple[Path, dict]:
    api = _api()
    root = tmp_path / "data"
    root.mkdir(parents=True, exist_ok=True)
    artifact_dir = _backtest_artifact_dir(root)
    fills = artifact_dir / f"{artifact_id}.fills.parquet"
    equity = artifact_dir / f"{artifact_id}.equity.parquet"
    pl.DataFrame({"security_id": ["SEC_1"]}).write_parquet(fills)
    pl.DataFrame({"nav": [1.0]}).write_parquet(equity)
    artifact = {
        "id": artifact_id,
        "claim": "research_only",
        "research_only": True,
        "live_pnl_claim": False,
        "metrics": {"observations": 1},
        "scope": {},
        "fills_path": str(fills),
        "equity_path": str(equity),
        "fills_sha256": api._file_sha256(fills),
        "equity_sha256": api._file_sha256(equity),
    }
    artifact["artifact_sha256"] = api._backtest_artifact_digest(artifact)
    monkeypatch.setattr(
        api, "_load_cfg", lambda _path: SimpleNamespace(data=SimpleNamespace(root=str(root)))
    )
    return artifact_dir / f"{artifact_id}.json", artifact


def test_backtest_lookup_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    api = _api()
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setattr(
        api, "_load_cfg", lambda _path: SimpleNamespace(data=SimpleNamespace(root=str(root)))
    )
    response = TestClient(app).get("/backtest/" + "b" * 16)
    assert response.status_code == 404
    assert "backtest not found" in response.json()["detail"]


def test_backtest_lookup_rejects_identity_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, artifact = _artifact_fixture(tmp_path, monkeypatch)
    artifact["id"] = "c" * 16  # receipt no longer matches its filename identity
    path.write_text(json.dumps(artifact))
    response = TestClient(app).get("/backtest/" + "a" * 16)
    assert response.status_code == 422
    assert "identity or claim validation" in response.json()["detail"]


def test_backtest_lookup_rejects_live_pnl_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, artifact = _artifact_fixture(tmp_path, monkeypatch)
    artifact["live_pnl_claim"] = True
    artifact["artifact_sha256"] = _api()._backtest_artifact_digest(artifact)
    path.write_text(json.dumps(artifact))
    response = TestClient(app).get("/backtest/" + "a" * 16)
    assert response.status_code == 422
    assert "invalid live-P&L claim" in response.json()["detail"]


def test_backtest_lookup_rejects_invalid_digest_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, artifact = _artifact_fixture(tmp_path, monkeypatch)
    artifact["artifact_sha256"] = "not-a-sha256"
    path.write_text(json.dumps(artifact))
    response = TestClient(app).get("/backtest/" + "a" * 16)
    assert response.status_code == 422
    assert "invalid digest" in response.json()["detail"]


def test_backtest_lookup_rejects_missing_data_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, artifact = _artifact_fixture(tmp_path, monkeypatch)
    Path(artifact["fills_path"]).unlink()
    path.write_text(json.dumps(artifact))
    response = TestClient(app).get("/backtest/" + "a" * 16)
    assert response.status_code == 422
    assert response.json()["detail"]["message"] == "backtest data artifact missing"
    assert response.json()["detail"]["path"] == "fills_path"


def test_backtest_lookup_returns_honest_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, artifact = _artifact_fixture(tmp_path, monkeypatch)
    path.write_text(json.dumps(artifact))
    response = TestClient(app).get("/backtest/" + "a" * 16)
    assert response.status_code == 200
    body = response.json()
    assert body["research_only"] is True
    assert body["live_pnl_claim"] is False
    assert body["id"] == "a" * 16
