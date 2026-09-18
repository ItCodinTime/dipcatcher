"""API fail-closed branches: middleware, artifact integrity, risk fallbacks.

Complements test_cli_api by exercising the branches that previously had no
coverage: non-loopback refusal, HSTS, absolute-path allowlist, backtest artifact
rejection paths, forecast distribution envelope, and risk/portfolio fallbacks.
"""

from __future__ import annotations

import importlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from quant_fund.api.app import app


def _api():  # noqa: ANN202
    return importlib.import_module("quant_fund.api.app")


def _write_config(configs_dir: Path, tmp_path: Path) -> Path:
    configs_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = configs_dir / "research.yaml"
    cfg_path.write_text(f"data:\n  root: {(tmp_path / 'data').as_posix()}\n")
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
) -> TestClient:
    api = _api()
    configs_dir = tmp_path / "configs"
    _write_config(configs_dir, tmp_path)
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
