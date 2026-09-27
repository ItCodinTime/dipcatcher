"""API coverage: streamed-body middleware, weight-artifact validation, drift.

Complements test_cli_api and test_api_fail_closed_branches by exercising the
remaining uncovered lines/branches in ``src/quant_fund/api/app.py``:

- ``limited_receive`` byte accounting (including non-http.request messages) and
  the ``_RequestBodyTooLarge`` → 413 path, exercised by invoking
  ``api_auth_middleware`` directly. Through the full ASGI stack, an oversized
  streamed body is instead surfaced by FastAPI's body parser as a generic 400 —
  the 64 KiB guard still fails closed either way.
- ``Content-Length`` parse/negative rejections (400).
- ``/risk/portfolio`` target-weight artifact validation branches and the
  ``unmeasured_reason`` → UNMEASURED mapping.
- ``/backtest/{id}`` with a *valid* ``analytics_export`` continuing to the
  data-artifact integrity checks.
- ``/monitoring/drift`` receipt read-race, mutation, decode, provenance and
  evidence-report branches.
- ``/research/latest`` unreadable receipt, post-verification deletion, and
  non-object payload rejection.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.responses import JSONResponse

from quant_fund.api.app import app


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


def _weights_frame(asof_day: int = 1) -> pl.DataFrame:
    asof = datetime(2026, 1, asof_day, tzinfo=UTC)
    return pl.DataFrame(
        {
            "event_time": [asof, asof],
            "security_id": ["SEC_1", "SEC_2"],
            "target_weight": [0.1, -0.05],
        }
    )


def _returns_frame(days: int = 5) -> pl.DataFrame:
    rows = []
    for day in range(1, days + 1):
        stamp = datetime(2026, 1, day, tzinfo=UTC)
        rows.append({"event_time": stamp, "security_id": "SEC_1", "ret_1": 0.01 * day})
        rows.append({"event_time": stamp, "security_id": "SEC_2", "ret_1": -0.005 * day})
    return pl.DataFrame(rows)


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


def _research_cfg_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[TestClient, Path]:
    """Client whose ``cfg.data.root`` points at ``tmp_path``."""
    api = _api()
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        api,
        "_load_cfg",
        lambda _path: SimpleNamespace(data=SimpleNamespace(root=str(tmp_path))),
    )
    return TestClient(app), tmp_path


def _request_for(messages: list[dict]) -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "scheme": "http",
        "path": "/portfolio/optimize",
        "query_string": b"",
        "headers": [],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }
    queue = iter(messages)

    async def receive() -> dict:
        return next(queue, {"type": "http.disconnect"})

    return Request(scope, receive)


# ---------------------------------------------------------------------------
# Middleware: Content-Length validation + streamed byte accounting
# ---------------------------------------------------------------------------


def test_middleware_rejects_malformed_content_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("QUANT_API_KEY", raising=False)
    client = TestClient(app)
    for declared in ("abc", "-1", "12.5"):
        response = client.post(
            "/portfolio/optimize",
            content=b"{}",
            headers={"Content-Length": declared},
        )
        assert response.status_code == 400
        assert response.json() == {"detail": "invalid Content-Length"}


def test_middleware_streamed_oversized_body_fails_closed() -> None:
    api = _api()
    request = _request_for(
        [
            {
                "type": "http.request",
                "body": b"x" * (api._MAX_REQUEST_BYTES + 1),
                "more_body": True,
            },
            {"type": "http.request", "body": b"", "more_body": False},
        ]
    )

    async def call_next(r: Request) -> JSONResponse:
        await r.body()  # pulls the oversized chunk through limited_receive
        return JSONResponse({"unreachable": True})

    async def run() -> JSONResponse:
        return await api.api_auth_middleware(request, call_next)

    response = asyncio.run(run())
    assert response.status_code == 413
    assert json.loads(response.body) == {"detail": "request body exceeds the 64 KiB limit"}


def test_middleware_limited_receive_counts_only_http_request() -> None:
    api = _api()
    request = _request_for(
        [
            {"type": "http.request", "body": b"{}", "more_body": True},
            {"type": "http.request", "body": b"", "more_body": False},
            {"type": "http.disconnect"},
        ]
    )
    observed: list[str] = []

    async def call_next(r: Request) -> JSONResponse:
        body = await r.body()
        assert body == b"{}"
        # A non-http.request message bypasses the byte accounting.
        disconnect = await r._receive()
        observed.append(disconnect["type"])
        return JSONResponse({"ok": True})

    async def run() -> JSONResponse:
        return await api.api_auth_middleware(request, call_next)

    response = asyncio.run(run())
    assert response.status_code == 200
    assert observed == ["http.disconnect"]


# ---------------------------------------------------------------------------
# /risk/portfolio: target-weight artifact validation
# ---------------------------------------------------------------------------


def test_risk_portfolio_unreadable_weights_artifact_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _api()
    configs_dir = tmp_path / "configs"
    _write_config(configs_dir, tmp_path)
    gold = tmp_path / "data" / "gold"
    gold.mkdir(parents=True, exist_ok=True)
    (gold / "target_weights.parquet").write_bytes(b"not a parquet file")
    monkeypatch.setattr(api, "_CONFIGS_DIR", configs_dir.resolve())
    response = TestClient(app).get("/risk/portfolio", params={"config_path": "research.yaml"})
    assert response.status_code == 422
    assert "could not be read" in response.json()["detail"]


def test_risk_portfolio_missing_columns_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights = pl.DataFrame(
        {
            "event_time": [datetime(2026, 1, 1, tzinfo=UTC)],
            "security_id": ["SEC_1"],
            "not_target_weight": [0.5],
        }
    )
    client = _risk_client(tmp_path, monkeypatch, weights=weights, panel_frame=pl.DataFrame())
    response = client.get("/risk/portfolio", params={"config_path": "research.yaml"})
    assert response.status_code == 422
    assert "missing columns" in response.json()["detail"]
    assert "target_weight" in response.json()["detail"]


def test_risk_portfolio_non_datetime_event_time_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights = pl.DataFrame(
        {
            "event_time": ["2026-01-01"],
            "security_id": ["SEC_1"],
            "target_weight": [0.5],
        }
    )
    client = _risk_client(tmp_path, monkeypatch, weights=weights, panel_frame=pl.DataFrame())
    response = client.get("/risk/portfolio", params={"config_path": "research.yaml"})
    assert response.status_code == 422
    assert "must be datetime" in response.json()["detail"]


def test_risk_portfolio_null_identity_fields_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stamp = datetime(2026, 1, 1, tzinfo=UTC)
    weights = pl.DataFrame(
        {
            "event_time": [stamp, stamp],
            "security_id": ["SEC_1", None],
            "target_weight": [0.5, -0.1],
        }
    )
    client = _risk_client(tmp_path, monkeypatch, weights=weights, panel_frame=pl.DataFrame())
    response = client.get("/risk/portfolio", params={"config_path": "research.yaml"})
    assert response.status_code == 422
    assert "null identity fields" in response.json()["detail"]


def test_risk_portfolio_uncastable_weights_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stamp = datetime(2026, 1, 1, tzinfo=UTC)
    weights = pl.DataFrame(
        {
            "event_time": [stamp],
            "security_id": ["SEC_1"],
            "target_weight": pl.Series([[0.5]]),
        }
    )
    client = _risk_client(tmp_path, monkeypatch, weights=weights, panel_frame=pl.DataFrame())
    response = client.get("/risk/portfolio", params={"config_path": "research.yaml"})
    assert response.status_code == 422
    assert "invalid weights" in response.json()["detail"]


def test_risk_portfolio_non_finite_weights_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stamp = datetime(2026, 1, 1, tzinfo=UTC)
    weights = pl.DataFrame(
        {
            "event_time": [stamp, stamp],
            "security_id": ["SEC_1", "SEC_2"],
            "target_weight": [0.5, float("nan")],
        }
    )
    client = _risk_client(tmp_path, monkeypatch, weights=weights, panel_frame=pl.DataFrame())
    response = client.get("/risk/portfolio", params={"config_path": "research.yaml"})
    assert response.status_code == 422
    assert "non-finite weights" in response.json()["detail"]


def test_risk_portfolio_non_datetime_asof_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The as-of type guard still fails closed if max() ever yields a non-datetime."""
    api = _api()
    client = _risk_client(
        tmp_path, monkeypatch, weights=_weights_frame(), panel_frame=_returns_frame()
    )
    monkeypatch.setattr(api, "datetime", str)
    response = client.get("/risk/portfolio", params={"config_path": "research.yaml"})
    assert response.status_code == 422
    assert "must be datetime" in response.json()["detail"]


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        (
            "fewer_than_two_securities",
            "fewer than two securities have point-in-time return history",
        ),
        (
            "estimation_failed",
            "insufficient finite return observations for covariance",
        ),
        (
            "unknown_future_reason",
            "insufficient finite return observations for covariance",
        ),
    ],
)
def test_risk_portfolio_unmeasured_estimate_reasons(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reason: str,
    expected: str,
) -> None:
    client = _risk_client(
        tmp_path,
        monkeypatch,
        weights=_weights_frame(asof_day=5),
        panel_frame=_returns_frame(),
    )
    import quant_fund.pipeline.forecast as forecast

    monkeypatch.setattr(
        forecast,
        "estimate_optimizer_covariance_asof",
        lambda *args, **kwargs: SimpleNamespace(unmeasured_reason=reason),
    )
    response = client.get("/risk/portfolio", params={"config_path": "research.yaml"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "UNMEASURED"
    assert body["reason"] == expected
    assert body["research_only"] is True
    assert body["live_pnl_claim"] is False


# ---------------------------------------------------------------------------
# /backtest/{id}: valid analytics_export proceeds to integrity checks
# ---------------------------------------------------------------------------


def _artifact_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, artifact_id: str = "a" * 16
) -> tuple[Path, dict]:
    api = _api()
    root = tmp_path / "data"
    root.mkdir(parents=True, exist_ok=True)
    artifact_dir = root / "metadata" / "backtests"
    artifact_dir.mkdir(parents=True, exist_ok=True)
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


def test_backtest_lookup_accepts_valid_analytics_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _api()
    path, artifact = _artifact_fixture(tmp_path, monkeypatch)
    export = {
        "label": "unit",
        "data_source": "synthetic",
        "research_only": True,
        "live_pnl_claim": False,
        "disclaimer": "research artifact",
        "exposure": {},
        "execution": {},
        "equity": {},
        "drawdown_duration": {},
        "var_es": {},
        "var_backtest": {},
        "mean_turnover": 0.0,
        "capacity": {},
        "stress": {},
        "stress_report": {},
    }
    artifact["metrics"] = {"analytics_export": export}
    artifact["artifact_sha256"] = api._backtest_artifact_digest(artifact)
    path.write_text(json.dumps(artifact))
    response = TestClient(app).get("/backtest/" + "a" * 16)
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "a" * 16
    assert body["metrics"]["analytics_export"]["data_source"] == "synthetic"
    assert body["research_only"] is True
    assert body["live_pnl_claim"] is False


# ---------------------------------------------------------------------------
# /monitoring/drift: receipt races, provenance, evidence report
# ---------------------------------------------------------------------------


def _write_receipt(root: Path, content: bytes | str) -> Path:
    receipt_dir = root / "metadata" / "research"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    latest = receipt_dir / "latest.json"
    latest.write_bytes(content if isinstance(content, bytes) else content.encode())
    return latest


def _stub_verify(monkeypatch: pytest.MonkeyPatch, fn: Callable[[Path], dict]) -> None:
    verifier = importlib.import_module("quant_fund.research.verify")
    monkeypatch.setattr(verifier, "verify_research_artifact", fn)


def test_drift_unreadable_receipt_reports_none_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, root = _research_cfg_client(tmp_path, monkeypatch)
    latest = _write_receipt(root, "{}")
    latest.chmod(0o000)
    body = client.get("/monitoring/drift").json()
    assert body["status"] == "UNMEASURED"
    assert body["reason"] == "research artifact could not be read"
    assert body["source"] == "none"
    assert body["research_only"] is True


def test_drift_receipt_deleted_between_verify_and_reread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, root = _research_cfg_client(tmp_path, monkeypatch)
    _write_receipt(root, "{}")

    def delete_on_verify(path: Path) -> dict:
        path.unlink()
        return {"valid": True, "errors": []}

    _stub_verify(monkeypatch, delete_on_verify)
    body = client.get("/monitoring/drift").json()
    assert body["status"] == "UNMEASURED"
    assert body["reason"] == "research artifact could not be read"
    assert body["source"] == "none"


def test_drift_receipt_changed_after_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, root = _research_cfg_client(tmp_path, monkeypatch)
    _write_receipt(root, '{"claim": "original"}')

    def mutate_on_verify(path: Path) -> dict:
        path.write_text('{"claim": "replacement"}')
        return {"valid": True, "errors": []}

    _stub_verify(monkeypatch, mutate_on_verify)
    body = client.get("/monitoring/drift").json()
    assert body["status"] == "UNMEASURED"
    assert body["reason"] == "research artifact changed after integrity verification"
    assert body["source"] == "none"


def test_drift_receipt_undecodable_after_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, root = _research_cfg_client(tmp_path, monkeypatch)
    _write_receipt(root, "not-json")
    _stub_verify(monkeypatch, lambda _path: {"valid": True, "errors": []})
    body = client.get("/monitoring/drift").json()
    assert body["status"] == "UNMEASURED"
    assert body["reason"] == "research artifact could not be read"
    assert body["source"] == "none"


def test_drift_valid_receipt_reports_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, root = _research_cfg_client(tmp_path, monkeypatch)
    receipt = {
        "data_source": "synthetic",
        "provenance": {
            "run_id": "run-1",
            "git_revision": "abc123",
            "point_in_time": True,
        },
    }
    _write_receipt(root, json.dumps(receipt))
    _stub_verify(monkeypatch, lambda _path: {"valid": True, "errors": []})
    body = client.get("/monitoring/drift").json()
    assert body["status"] == "UNMEASURED"
    assert body["source"] == "research_artifact"
    assert body["latest_run_id"] == "run-1"
    assert body["latest_git_revision"] == "abc123"
    assert body["data_source"] == "synthetic"
    assert body["point_in_time"] is True
    assert body["research_only"] is True


def _write_evidence_report(root: Path, payload: str, *, correct_sha: bool = True) -> Path:
    report_dir = root / "metadata" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    evidence = report_dir / "evidence_report.json"
    evidence.write_text(payload)
    sidecar = report_dir / "evidence_report.json.sha256"
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    sidecar.write_text(digest if correct_sha else "0" * 64, encoding="ascii")
    return evidence


def test_drift_evidence_report_with_health_sets_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, root = _research_cfg_client(tmp_path, monkeypatch)
    _write_evidence_report(
        root,
        json.dumps(
            {
                "status": "sufficient_evidence",
                "health": {"status": "HEALTHY", "warnings": []},
                "warnings": ["w1"],
            }
        ),
    )
    body = client.get("/monitoring/drift").json()
    assert body["status"] == "HEALTHY"
    assert body["health"] == {"status": "HEALTHY", "warnings": []}
    assert body["evidence_report"] == {"status": "sufficient_evidence", "warnings": ["w1"]}
    assert body["research_only"] is True


def test_drift_evidence_report_without_health_keeps_unmeasured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, root = _research_cfg_client(tmp_path, monkeypatch)
    _write_evidence_report(root, json.dumps({"status": "insufficient_evidence"}))
    body = client.get("/monitoring/drift").json()
    assert body["status"] == "UNMEASURED"
    assert "health" not in body
    assert body["evidence_report"]["status"] == "insufficient_evidence"


def test_drift_evidence_report_hash_mismatch_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, root = _research_cfg_client(tmp_path, monkeypatch)
    _write_evidence_report(root, '{"status": "x"}', correct_sha=False)
    body = client.get("/monitoring/drift").json()
    assert body["evidence_report"] == {"status": "invalid", "reason": "hash_mismatch"}


def test_drift_evidence_report_malformed_json_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, root = _research_cfg_client(tmp_path, monkeypatch)
    _write_evidence_report(root, "not-json")
    body = client.get("/monitoring/drift").json()
    assert body["evidence_report"] == {"status": "invalid"}


# ---------------------------------------------------------------------------
# /research/latest: read races and non-object payload
# ---------------------------------------------------------------------------


def test_research_latest_unreadable_receipt_is_422(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, root = _research_cfg_client(tmp_path, monkeypatch)
    latest = _write_receipt(root, "{}")
    latest.chmod(0o000)
    response = client.get("/research/latest")
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["message"] == "latest research receipt could not be read before verification"


def test_research_latest_deleted_after_verification_is_422(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, root = _research_cfg_client(tmp_path, monkeypatch)
    _write_receipt(root, "{}")

    def delete_on_verify(path: Path) -> dict:
        path.unlink()
        return {"valid": True}

    _stub_verify(monkeypatch, delete_on_verify)
    response = client.get("/research/latest")
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["message"] == "latest research receipt could not be read after verification"


def test_research_latest_non_object_payload_is_422(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, root = _research_cfg_client(tmp_path, monkeypatch)
    _write_receipt(root, "[1, 2, 3]")
    _stub_verify(monkeypatch, lambda _path: {"valid": True})
    response = client.get("/research/latest")
    assert response.status_code == 422
    assert "not an object" in response.json()["detail"]
