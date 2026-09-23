import json
from datetime import UTC
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from fastapi.testclient import TestClient

from quant_fund.api.app import _backtest_artifact_digest, _write_backtest_artifact, app
from quant_fund.pipeline.doctor import doctor


def test_doctor() -> None:
    info = doctor("configs/research.yaml")
    assert info["core_imports"] == "ok"
    assert info["mode"] == "research"
    assert info["firm"] == "Artificial Hedge"
    assert info["benchmark_catalog_version"] == "2"
    assert "portfolio_conformal" in info["families"]
    assert "northset" in info["families"]


def test_doctor_rejects_manifest_with_missing_required_artifact(tmp_path: Path) -> None:
    cfg = tmp_path / "research.yaml"
    cfg.write_text(f"data:\n  root: {tmp_path.as_posix()}\n  source: synthetic\n")
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "data_manifest.json").write_text(
        '{"schema_version":1,"source":"synthetic","artifacts":{"bars":{}}}'
    )
    assert doctor(str(cfg))["data_manifest"] == "invalid"


def test_doctor_rejects_manifest_artifact_outside_data_root(tmp_path: Path) -> None:
    import hashlib
    import json

    cfg = tmp_path / "research.yaml"
    root = tmp_path / "data"
    cfg.write_text(f"data:\n  root: {root.as_posix()}\n  source: synthetic\n")
    outside = tmp_path / "unrelated.parquet"
    outside.write_bytes(b"not the configured data lake")
    metadata = root / "metadata"
    metadata.mkdir(parents=True)
    artifacts = {
        name: {
            "path": str(outside),
            "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
            "rows": 0,
        }
        for name in ("bars", "actions", "master", "silver", "universe")
    }
    (metadata / "data_manifest.json").write_text(
        json.dumps({"source": "synthetic", "artifacts": artifacts})
    )
    assert doctor(str(cfg))["data_manifest"] == "invalid"


def test_doctor_rejects_unknown_manifest_schema_version(tmp_path: Path) -> None:
    cfg = tmp_path / "research.yaml"
    root = tmp_path / "data"
    cfg.write_text(f"data:\n  root: {root.as_posix()}\n  source: synthetic\n")
    metadata = root / "metadata"
    metadata.mkdir(parents=True)
    (metadata / "data_manifest.json").write_text(
        '{"schema_version":99,"source":"synthetic","artifacts":{}}'
    )
    assert doctor(str(cfg))["data_manifest"] == "invalid"


def test_doctor_rejects_invalid_research_receipt(tmp_path: Path) -> None:
    cfg = tmp_path / "research.yaml"
    root = tmp_path / "data"
    cfg.write_text(f"data:\n  root: {root.as_posix()}\n  source: synthetic\n")
    receipt_dir = root / "metadata" / "research"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "latest.json").write_text('{"claim":"live"}')
    assert doctor(str(cfg))["research_receipt"] == "invalid"


def test_doctor_rejects_manifest_with_stale_row_count(tmp_path: Path) -> None:
    import hashlib

    cfg = tmp_path / "research.yaml"
    root = tmp_path / "data"
    cfg.write_text(f"data:\n  root: {root.as_posix()}\n  source: synthetic\n")
    bars = root / "bronze" / "bars.parquet"
    bars.parent.mkdir(parents=True)
    frame = pl.DataFrame({"security_id": ["A", "B"], "close": [1.0, 2.0]})
    frame.write_parquet(bars)
    artifacts = {
        name: {
            "path": str(bars),
            "sha256": hashlib.sha256(bars.read_bytes()).hexdigest(),
            "rows": 99 if name == "bars" else 2,
            "columns": ["security_id", "close"],
        }
        for name in ("bars", "actions", "master", "silver", "universe")
    }
    metadata = root / "metadata"
    metadata.mkdir(parents=True)
    (metadata / "data_manifest.json").write_text(
        json.dumps({"schema_version": 1, "source": "synthetic", "artifacts": artifacts})
    )
    assert doctor(str(cfg))["data_manifest"] == "invalid"


def test_health_endpoint() -> None:
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["firm"] == "Artificial Hedge"
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["X-Frame-Options"] == "DENY"
    assert r.headers["Cache-Control"] == "no-store"


def test_readiness_is_fail_closed_for_missing_manifest(tmp_path: Path, monkeypatch) -> None:
    """Missing data_manifest must fail closed (503), isolated from the repo lake.

    Hardens against macOS ``/var`` vs ``/private/var`` when monkeypatching
    ``_CONFIGS_DIR`` with an unresolved tmp path, and against cwd pollution
    from the real ``data/metadata/data_manifest.json``.
    """
    import importlib

    api = importlib.import_module("quant_fund.api.app")

    root = tmp_path.resolve()
    (root / "research.yaml").write_text(
        f"runtime:\n  mode: research\ndata:\n  root: {root.as_posix()}\n  source: synthetic\n"
    )
    # Prefer unresolved /var alias when pytest tmp lives under /private/var
    # (historical intermittent 400 from relative_to without resolve).
    patch_dir: Path = root
    if root.parts[:3] == ("/", "private", "var"):
        candidate = Path("/var").joinpath(*root.parts[3:])
        if candidate.exists():
            patch_dir = candidate
    monkeypatch.setattr(api, "_CONFIGS_DIR", patch_dir)
    response = TestClient(api.app).get("/ready", params={"config_path": "research.yaml"})
    assert response.status_code == 503, response.json()
    payload = response.json()
    assert payload["status"] == "not_ready"
    assert "checks" in payload
    assert payload["checks"]["data_manifest"] is False
    assert payload["report"]["data_manifest"] == "missing"


def test_readiness_is_fail_closed_for_missing_research_receipt(tmp_path: Path, monkeypatch) -> None:
    """A complete data lake without an integrity receipt is not ready."""
    import importlib

    api = importlib.import_module("quant_fund.api.app")
    root = tmp_path.resolve()
    (root / "research.yaml").write_text(
        f"runtime:\n  mode: research\ndata:\n  root: {root.as_posix()}\n  source: synthetic\n"
    )
    monkeypatch.setattr(api, "_CONFIGS_DIR", root)
    response = TestClient(api.app).get("/ready", params={"config_path": "research.yaml"})
    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "not_ready"
    assert payload["checks"]["research_receipt"] is False
    assert payload["report"]["research_receipt"] == "missing"


def test_resolve_allowed_config_path_tolerates_unresolved_configs_dir(
    tmp_path: Path, monkeypatch
) -> None:
    """Allowlist must resolve configs_dir before relative_to (macOS symlink)."""
    import importlib

    api = importlib.import_module("quant_fund.api.app")
    root = tmp_path.resolve()
    (root / "research.yaml").write_text("data:\n  root: data\n  source: synthetic\n")
    patch_dir = root
    if root.parts[:3] == ("/", "private", "var"):
        alias = Path("/var").joinpath(*root.parts[3:])
        if alias.exists():
            patch_dir = alias
            assert patch_dir != root  # unresolved vs resolved
    monkeypatch.setattr(api, "_CONFIGS_DIR", patch_dir)
    resolved = api.resolve_allowed_config_path("research.yaml")
    assert resolved == (root / "research.yaml")


def test_readiness_returns_structured_failure_for_invalid_config(
    tmp_path: Path, monkeypatch
) -> None:
    import importlib

    api = importlib.import_module("quant_fund.api.app")
    (tmp_path / "broken.yaml").write_text("data: [not: valid")
    monkeypatch.setattr(api, "_CONFIGS_DIR", tmp_path.resolve())
    response = TestClient(api.app).get("/ready", params={"config_path": "broken.yaml"})
    assert response.status_code == 503
    assert response.json()["checks"]["configuration"] is False


def test_readiness_does_not_mask_internal_bugs_as_config_invalid(
    tmp_path: Path, monkeypatch
) -> None:
    """Bugbot regression: only genuine config errors fail closed as 'invalid'.

    A code bug inside ``doctor`` (e.g. AttributeError) must surface as an
    HTTP 500 traceback, not be swallowed into a 503 "configuration invalid"
    that hides the real defect from operators.
    """
    import importlib

    api = importlib.import_module("quant_fund.api.app")
    root = tmp_path.resolve()
    (root / "research.yaml").write_text(
        f"runtime:\n  mode: research\ndata:\n  root: {root.as_posix()}\n  source: synthetic\n"
    )
    monkeypatch.setattr(api, "_CONFIGS_DIR", root)

    def boom(_config_path: str):
        raise AttributeError("simulated internal bug in doctor()")

    monkeypatch.setattr(api, "doctor", boom)
    with pytest.raises(AttributeError, match="simulated internal bug"):
        TestClient(api.app, raise_server_exceptions=True).get(
            "/ready", params={"config_path": "research.yaml"}
        )


def test_models_endpoint_reports_backend_availability() -> None:
    payload = TestClient(app).get("/models").json()
    assert "xgboost" in payload["backend_availability"]
    assert "lightgbm" in payload["backend_availability"]
    assert "torch" in payload["backend_availability"]
    assert "robinhood_plus" in payload["kline_foundation"]
    assert payload["robinhood_plus"]["core_engine"] is True
    assert payload["catalog_claim"]
    from quant_fund.pipeline.train import RANKING_MODEL_NAMES

    assert set(payload["ranking"]) == RANKING_MODEL_NAMES
    assert {
        "rff",
        "sdf_ridge",
        "ipca",
        "fm",
        "pcr",
        "pls",
        "tprf",
        "gbrt",
        "pp",
        "combo",
        "alasso",
        "classic",
        "fm_ridge",
        "combo_ic",
        "reversal",
        "classic_st",
        "ridge_st",
        "ridge_neut",
        "fm_st",
        "combo_ic_st",
        "combo_msfe",
    } <= set(payload["ranking"])


def test_drift_endpoint_is_explicitly_unmeasured() -> None:
    client = TestClient(app)
    response = client.get("/monitoring/drift")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "UNMEASURED"
    assert "reason" in payload


def test_regime_endpoint_is_explicitly_unmeasured() -> None:
    response = TestClient(app).get("/regime")
    assert response.status_code == 200
    assert response.json()["status"] == "UNMEASURED"
    assert response.json()["live_pnl_claim"] is False


def test_cli_doctor_surfaces_unready_default_state(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from quant_fund.cli.main import app as cli

    # Isolate the readiness check from ignored/generated artifacts left by
    # prior synthetic runs in the repository's default data root.
    config = tmp_path / "research.yaml"
    config.write_text(f"data:\n  root: {tmp_path.as_posix()}\n  source: synthetic\n")
    runner = CliRunner()
    result = runner.invoke(cli, ["doctor", "--config", str(config)])
    # An unprepared checkout must fail closed while still rendering diagnostics.
    assert result.exit_code == 1
    assert "core_imports" in result.stdout


def test_cli_doctor_fails_closed_on_invalid_manifest(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from quant_fund.cli.main import app as cli

    config = tmp_path / "research.yaml"
    config.write_text(f"data:\n  root: {tmp_path.as_posix()}\n  source: synthetic\n")
    result = CliRunner().invoke(cli, ["doctor", "--config", str(config)])
    assert result.exit_code == 1
    assert "data_manifest: missing" in result.stdout


def test_cli_api_refuses_non_loopback_without_key(monkeypatch) -> None:
    from typer.testing import CliRunner

    from quant_fund.cli.main import app as cli

    monkeypatch.delenv("QUANT_API_KEY", raising=False)
    result = CliRunner().invoke(cli, ["api", "--host", "0.0.0.0"])
    assert result.exit_code != 0
    assert "requires QUANT_API_KEY" in result.output


def test_config_path_traversal_rejected() -> None:
    from fastapi import HTTPException

    from quant_fund.api.app import resolve_allowed_config_path

    try:
        resolve_allowed_config_path("../pyproject.toml")
        raise AssertionError("expected HTTPException")
    except HTTPException as exc:
        assert exc.status_code == 400


def test_config_path_allowlisted() -> None:
    from quant_fund.api.app import resolve_allowed_config_path

    path = resolve_allowed_config_path("configs/research.yaml")
    assert path.name == "research.yaml"
    assert path.parent.name == "configs"


def test_api_key_required_when_set(monkeypatch) -> None:
    monkeypatch.setenv("QUANT_API_KEY", "secret-test-key")
    # Re-import not needed: middleware reads env each request
    client = TestClient(app)
    r = client.get("/doctor")
    assert r.status_code == 401
    r_ok = client.get("/doctor", headers={"X-API-Key": "secret-test-key"})
    assert r_ok.status_code == 200
    assert r_ok.headers["content-security-policy"].startswith("default-src 'self'")
    assert r_ok.headers["permissions-policy"] == "camera=(), microphone=(), geolocation=()"
    # health stays public
    assert client.get("/health").status_code == 200
    # Configured-key deployments protect metadata and documentation routes too.
    assert client.get("/models").status_code == 401
    assert client.get("/openapi.json").status_code == 401
    assert client.get("/models", headers={"X-API-Key": "secret-test-key"}).status_code == 200


def test_api_rejects_declared_oversized_request_before_route(monkeypatch) -> None:
    import importlib

    api = importlib.import_module("quant_fund.api.app")
    monkeypatch.delenv("QUANT_API_KEY", raising=False)
    response = TestClient(api.app).post(
        "/portfolio/optimize",
        headers={"Content-Length": str(api._MAX_REQUEST_BYTES + 1)},
        content=b"{}",
    )
    assert response.status_code == 413
    assert response.json() == {"detail": "request body exceeds the 64 KiB limit"}


def test_api_request_models_reject_unknown_fields() -> None:
    response = TestClient(app).post(
        "/portfolio/optimize",
        json={"config_path": "configs/research.yaml", "unexpected": "value"},
    )
    assert response.status_code == 422
    assert "extra_forbidden" in response.text


def test_api_key_comparison_is_constant_time(monkeypatch) -> None:
    import hmac
    import importlib

    api = importlib.import_module("quant_fund.api.app")

    calls: list[tuple[str, str]] = []
    original = hmac.compare_digest

    def recording_compare(left: str, right: str) -> bool:
        calls.append((left, right))
        return original(left, right)

    monkeypatch.setattr(api.hmac, "compare_digest", recording_compare)
    monkeypatch.setenv("QUANT_API_KEY", "secret-test-key")
    response = TestClient(app).get("/doctor", headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 401
    assert calls == [("wrong-key", "secret-test-key")]


def test_drift_endpoint_fails_closed_on_invalid_research_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    import importlib

    api = importlib.import_module("quant_fund.api.app")
    cfg = tmp_path / "research.yaml"
    cfg.write_text(f"data:\n  root: {tmp_path.as_posix()}\n  source: synthetic\n")
    research_dir = tmp_path / "metadata" / "research"
    research_dir.mkdir(parents=True)
    (research_dir / "latest.json").write_text('{"synthetic": true}')
    monkeypatch.setattr(api, "_CONFIGS_DIR", tmp_path)
    response = TestClient(api.app).get("/monitoring/drift", params={"config_path": "research.yaml"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "UNMEASURED"
    assert payload["source"] == "none"
    assert payload["integrity_errors"]


def test_backtest_artifact_is_retrievable_and_research_only(tmp_path: Path) -> None:
    cfg = SimpleNamespace(data=SimpleNamespace(root=str(tmp_path)))
    request = SimpleNamespace(config_path="configs/backtest.yaml")
    bars = pl.DataFrame(
        {"event_time": ["2026-01-01", "2026-01-02"], "source": ["synthetic", "synthetic"]}
    )
    result = SimpleNamespace(
        frictionless=False,
        source_note="synthetic_only",
        metrics={"observations": 2},
        fills=pl.DataFrame({"security_id": ["A"]}),
        equity=pl.DataFrame({"nav": [100.0]}),
    )

    artifact = _write_backtest_artifact(cfg, request, bars, result)
    saved = Path(artifact["artifact_path"])
    assert saved.exists()
    assert artifact["claim"] == "research_only"
    assert artifact["live_pnl_claim"] is False
    assert Path(artifact["fills_path"]).exists()
    assert Path(artifact["equity_path"]).exists()
    assert len(artifact["fills_sha256"]) == 64
    assert len(artifact["equity_sha256"]) == 64
    assert artifact["artifact_sha256"] == _backtest_artifact_digest(artifact)


def test_backtest_lookup_rejects_tampered_claim(tmp_path: Path, monkeypatch) -> None:
    import importlib

    api = importlib.import_module("quant_fund.api.app")

    cfg = SimpleNamespace(data=SimpleNamespace(root=str(tmp_path)))
    artifact_dir = tmp_path / "metadata" / "backtests"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / ("a" * 16 + ".json")).write_text("{}")
    monkeypatch.setattr(api, "_load_cfg", lambda _path: cfg)
    response = TestClient(app).get("/backtest/aaaaaaaaaaaaaaaa")
    assert response.status_code == 422


def test_backtest_lookup_rejects_poisoned_analytics_export(tmp_path: Path, monkeypatch) -> None:
    import importlib

    api = importlib.import_module("quant_fund.api.app")
    cfg = SimpleNamespace(data=SimpleNamespace(root=str(tmp_path)))
    artifact_dir = tmp_path / "metadata" / "backtests"
    artifact_dir.mkdir(parents=True)
    fills = artifact_dir / "f.fills.parquet"
    equity = artifact_dir / "f.equity.parquet"
    pl.DataFrame({"security_id": ["A"]}).write_parquet(fills)
    pl.DataFrame({"nav": [100.0]}).write_parquet(equity)
    artifact = {
        "id": "f" * 16,
        "claim": "research_only",
        "research_only": True,
        "live_pnl_claim": False,
        "metrics": {"analytics_export": {"live_pnl_claim": True}},
        "scope": {},
        "fills_path": str(fills),
        "equity_path": str(equity),
        "fills_sha256": api._file_sha256(fills),
        "equity_sha256": api._file_sha256(equity),
    }
    (artifact_dir / ("f" * 16 + ".json")).write_text(json.dumps(artifact))
    monkeypatch.setattr(api, "_load_cfg", lambda _path: cfg)

    response = TestClient(app).get("/backtest/" + "f" * 16)

    assert response.status_code == 422
    assert "analytics export" in response.json()["detail"]["message"]


def test_backtest_lookup_rejects_non_hex_id() -> None:
    response = TestClient(app).get("/backtest/AAAAAAAAAAAAAAAA")
    assert response.status_code == 400


def test_research_latest_fails_closed_on_read_race(tmp_path: Path, monkeypatch) -> None:
    import importlib

    api = importlib.import_module("quant_fund.api.app")
    cfg = SimpleNamespace(data=SimpleNamespace(root=str(tmp_path)))
    receipt = tmp_path / "metadata" / "research"
    receipt.mkdir(parents=True)
    latest = receipt / "latest.json"
    latest.write_text("not-json")
    monkeypatch.setattr(api, "_load_cfg", lambda _path: cfg)
    verifier = importlib.import_module("quant_fund.research.verify")
    monkeypatch.setattr(verifier, "verify_research_artifact", lambda _path: {"valid": True})

    response = TestClient(app).get("/research/latest")

    assert response.status_code == 422
    assert "could not be read" in response.json()["detail"]["message"]


def test_research_latest_does_not_serve_bytes_changed_after_verification(
    tmp_path: Path, monkeypatch
) -> None:
    import importlib

    api = importlib.import_module("quant_fund.api.app")
    cfg = SimpleNamespace(data=SimpleNamespace(root=str(tmp_path)))
    receipt = tmp_path / "metadata" / "research"
    receipt.mkdir(parents=True)
    latest = receipt / "latest.json"
    latest.write_text(json.dumps({"claim": "original"}))
    monkeypatch.setattr(api, "_load_cfg", lambda _path: cfg)
    verifier = importlib.import_module("quant_fund.research.verify")

    def replace_after_verification(path: Path) -> dict[str, object]:
        path.write_text(json.dumps({"claim": "replacement"}))
        return {"valid": True}

    monkeypatch.setattr(verifier, "verify_research_artifact", replace_after_verification)

    response = TestClient(app).get("/research/latest")

    assert response.status_code == 422
    assert "changed after verification" in response.json()["detail"]["message"]


def test_research_latest_applies_honesty_envelope(tmp_path: Path, monkeypatch) -> None:
    import importlib

    api = importlib.import_module("quant_fund.api.app")
    cfg = SimpleNamespace(data=SimpleNamespace(root=str(tmp_path)))
    receipt = tmp_path / "metadata" / "research"
    receipt.mkdir(parents=True)
    (receipt / "latest.json").write_text(
        json.dumps({"claim": "research_only", "live_pnl_claim": True})
    )
    monkeypatch.setattr(api, "_load_cfg", lambda _path: cfg)
    verifier = importlib.import_module("quant_fund.research.verify")
    monkeypatch.setattr(verifier, "verify_research_artifact", lambda _path: {"valid": True})

    response = TestClient(app).get("/research/latest")

    assert response.status_code == 200
    payload = response.json()
    assert payload["research_only"] is True
    assert payload["live_pnl_claim"] is False


def test_research_latest_fails_closed_on_verifier_exception(tmp_path: Path, monkeypatch) -> None:
    import importlib

    api = importlib.import_module("quant_fund.api.app")
    cfg = SimpleNamespace(data=SimpleNamespace(root=str(tmp_path)))
    receipt = tmp_path / "metadata" / "research"
    receipt.mkdir(parents=True)
    (receipt / "latest.json").write_text("{}")
    monkeypatch.setattr(api, "_load_cfg", lambda _path: cfg)
    verifier = importlib.import_module("quant_fund.research.verify")

    def explode(_path):
        raise OSError("simulated verifier failure")

    monkeypatch.setattr(verifier, "verify_research_artifact", explode)

    response = TestClient(app).get("/research/latest")

    assert response.status_code == 422
    assert response.json()["detail"]["message"] == "latest research receipt verification failed"


def test_backtest_lookup_rejects_mutated_data_file(tmp_path: Path, monkeypatch) -> None:
    import importlib

    api = importlib.import_module("quant_fund.api.app")
    cfg = SimpleNamespace(data=SimpleNamespace(root=str(tmp_path)))
    request = SimpleNamespace(config_path="configs/backtest.yaml")
    bars = pl.DataFrame({"event_time": ["2026-01-01"], "source": ["file"]})
    result = SimpleNamespace(
        frictionless=False,
        source_note="file",
        metrics={"observations": 1},
        fills=pl.DataFrame({"security_id": ["A"]}),
        equity=pl.DataFrame({"nav": [100.0]}),
    )
    artifact = _write_backtest_artifact(cfg, request, bars, result)
    pl.DataFrame({"security_id": ["TAMPERED"]}).write_parquet(artifact["fills_path"])
    monkeypatch.setattr(api, "_load_cfg", lambda _path: cfg)
    response = TestClient(app).get(f"/backtest/{artifact['id']}")
    assert response.status_code == 422


def test_backtest_lookup_rejects_tampered_receipt_digest(tmp_path: Path, monkeypatch) -> None:
    import importlib

    api = importlib.import_module("quant_fund.api.app")
    cfg = SimpleNamespace(data=SimpleNamespace(root=str(tmp_path)))
    request = SimpleNamespace(config_path="configs/backtest.yaml")
    bars = pl.DataFrame({"event_time": ["2026-01-01"], "source": ["file"]})
    result = SimpleNamespace(
        frictionless=False,
        source_note="file",
        metrics={"observations": 1},
        fills=pl.DataFrame({"security_id": ["A"]}),
        equity=pl.DataFrame({"nav": [100.0]}),
    )
    artifact = _write_backtest_artifact(cfg, request, bars, result)
    receipt_path = Path(artifact["artifact_path"])
    receipt = json.loads(receipt_path.read_text())
    receipt["scope"]["source"] = "tampered"
    receipt_path.write_text(json.dumps(receipt))
    monkeypatch.setattr(api, "_load_cfg", lambda _path: cfg)

    response = TestClient(app).get(f"/backtest/{artifact['id']}")

    assert response.status_code == 422
    assert response.json()["detail"] == "backtest artifact digest mismatch"


def test_backtest_lookup_rejects_artifact_path_escape(tmp_path: Path, monkeypatch) -> None:
    import importlib

    api = importlib.import_module("quant_fund.api.app")
    cfg = SimpleNamespace(data=SimpleNamespace(root=str(tmp_path)))
    artifact_dir = tmp_path / "metadata" / "backtests"
    artifact_dir.mkdir(parents=True)
    outside = tmp_path / "outside.parquet"
    pl.DataFrame({"security_id": ["A"]}).write_parquet(outside)
    artifact = {
        "id": "b" * 16,
        "claim": "research_only",
        "live_pnl_claim": False,
        "metrics": {},
        "scope": {},
        "fills_path": str(outside),
        "equity_path": str(outside),
        "fills_sha256": "0" * 64,
        "equity_sha256": "0" * 64,
    }
    (artifact_dir / ("b" * 16 + ".json")).write_text(json.dumps(artifact))
    monkeypatch.setattr(api, "_load_cfg", lambda _path: cfg)
    response = TestClient(app).get("/backtest/" + "b" * 16)
    assert response.status_code == 422


def _assert_research_honesty(payload: dict) -> None:
    assert payload.get("research_only") is True
    assert payload.get("live_pnl_claim") is False


def test_regime_and_drift_force_research_honesty() -> None:
    client = TestClient(app)
    regime = client.get("/regime").json()
    _assert_research_honesty(regime)
    assert regime["status"] == "UNMEASURED"
    drift = client.get("/monitoring/drift").json()
    _assert_research_honesty(drift)
    assert drift["status"] == "UNMEASURED"


def test_risk_portfolio_unmeasured_forces_honesty(tmp_path: Path, monkeypatch) -> None:
    import importlib

    api = importlib.import_module("quant_fund.api.app")
    cfg = SimpleNamespace(data=SimpleNamespace(root=str(tmp_path)))
    monkeypatch.setattr(api, "_load_cfg", lambda _path: cfg)
    payload = TestClient(app).get("/risk/portfolio").json()
    assert payload["status"] == "UNMEASURED"
    _assert_research_honesty(payload)
    # Poisoned upstream cannot flip honesty via stamp helper
    stamped = api._stamp_research_honesty(
        {"status": "MEASURED", "research_only": False, "live_pnl_claim": True, "sharpe": 9.9}
    )
    _assert_research_honesty(stamped)
    assert stamped["status"] == "MEASURED"


def test_portfolio_optimize_and_exposures_force_honesty(monkeypatch) -> None:
    import importlib

    import polars as pl

    api = importlib.import_module("quant_fund.api.app")
    fake = pl.DataFrame(
        {
            "event_time": ["2026-01-01"],
            "security_id": ["A"],
            "target_weight": [0.5],
            "alpha": [0.1],
        }
    )
    monkeypatch.setattr(api, "_load_cfg", lambda _path: SimpleNamespace())
    monkeypatch.setattr(api, "optimize_asof", lambda _cfg, _asof=None: fake)

    opt = TestClient(app).post("/portfolio/optimize", json={"config_path": "configs/research.yaml"})
    assert opt.status_code == 200
    body = opt.json()
    assert isinstance(body, dict)
    assert body["n"] == 1
    assert body["weights"][0]["security_id"] == "A"
    _assert_research_honesty(body)
    assert body["claim"] == "research_only"

    exp = TestClient(app).get("/portfolio/exposures")
    assert exp.status_code == 200
    ebody = exp.json()
    assert ebody["gross"] == 0.5
    assert ebody["net"] == 0.5
    _assert_research_honesty(ebody)

    tgt = TestClient(app).get("/portfolio/target")
    assert tgt.status_code == 200
    tbody = tgt.json()
    assert tbody["weights"][0]["target_weight"] == 0.5
    _assert_research_honesty(tbody)


def test_backtest_post_bounds_engine_to_causal_window(tmp_path: Path, monkeypatch) -> None:
    import importlib

    api = importlib.import_module("quant_fund.api.app")
    cfg = SimpleNamespace(data=SimpleNamespace(root=str(tmp_path)))
    all_dates = list(range(100))
    bars = pl.DataFrame({"event_time": all_dates})
    result = SimpleNamespace(
        frictionless=False,
        source_note="synthetic_only",
        metrics={"observations": 30},
        fills=pl.DataFrame({"security_id": ["A"]}),
        equity=pl.DataFrame({"nav": [100.0]}),
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(api, "_load_cfg", lambda _path: cfg)
    monkeypatch.setattr(
        "quant_fund.pipeline.dataset.ensure_silver",
        lambda _cfg: bars,
    )

    def capture_weights(_cfg, dates):
        captured["weight_dates"] = dates
        return pl.DataFrame()

    def capture_backtest(engine_bars, _weights, _cfg):
        captured["engine_dates"] = engine_bars["event_time"].unique().sort().to_list()
        return result

    monkeypatch.setattr(api, "build_causal_weight_panel", capture_weights)
    monkeypatch.setattr("quant_fund.backtest.engine.run_backtest", capture_backtest)

    response = TestClient(app).post("/backtest", json={"config_path": "configs/backtest.yaml"})

    assert response.status_code == 200
    assert captured["weight_dates"] == all_dates[:30]
    assert captured["engine_dates"] == all_dates[:31]


def test_backtest_post_forces_honesty_over_poisoned_metrics(tmp_path: Path, monkeypatch) -> None:
    import importlib

    api = importlib.import_module("quant_fund.api.app")
    cfg = SimpleNamespace(data=SimpleNamespace(root=str(tmp_path)))

    bars = pl.DataFrame({"event_time": ["2026-01-01", "2026-01-02"]})
    result = SimpleNamespace(
        frictionless=False,
        source_note="synthetic_only",
        metrics={
            "observations": 2,
            "research_only": False,
            "live_pnl_claim": True,
            "total_return": 0.01,
        },
        fills=pl.DataFrame({"security_id": ["A"]}),
        equity=pl.DataFrame({"nav": [100.0]}),
    )

    monkeypatch.setattr(api, "_load_cfg", lambda _path: cfg)
    monkeypatch.setattr(
        "quant_fund.pipeline.dataset.ensure_silver",
        lambda _cfg: bars,
    )
    monkeypatch.setattr(api, "build_causal_weight_panel", lambda _cfg, _dates: pl.DataFrame())
    monkeypatch.setattr(
        "quant_fund.backtest.engine.run_backtest",
        lambda _bars, _weights, _cfg: result,
    )

    response = TestClient(app).post("/backtest", json={"config_path": "configs/backtest.yaml"})
    assert response.status_code == 200
    payload = response.json()
    _assert_research_honesty(payload)
    assert payload["claim"] == "research_only"
    assert payload["observations"] == 2


def test_forecast_missing_symbol_fail_closed(monkeypatch) -> None:
    import importlib

    api = importlib.import_module("quant_fund.api.app")

    class _State:
        forecasts: list = []

    monkeypatch.setattr(api, "_load_cfg", lambda _path: SimpleNamespace())
    monkeypatch.setattr(api, "forecast_asof", lambda _cfg: _State())
    response = TestClient(app).get("/forecast/NO_SUCH_SYMBOL")
    assert response.status_code == 404
    detail = response.json()
    # Fail closed: error body must not invent a live P&L claim
    assert detail.get("live_pnl_claim") is not True
    assert detail.get("research_only") is not False
    assert "NO_SUCH_SYMBOL" in detail["detail"]


def test_forecast_and_ranking_stamp_honesty(monkeypatch) -> None:
    import importlib
    from datetime import datetime

    from quant_fund.schemas.forecast import AssetForecast

    api = importlib.import_module("quant_fund.api.app")
    forecast = AssetForecast(
        security_id="SEC_1",
        symbol="AAA",
        asof=datetime(2026, 1, 1, tzinfo=UTC),
        model_version="test",
        rank_percentile={"5d": 0.9},
    )

    class _State:
        forecasts = [forecast]

    monkeypatch.setattr(api, "_load_cfg", lambda _path: SimpleNamespace())
    monkeypatch.setattr(api, "forecast_asof", lambda _cfg: _State())

    fr = TestClient(app).get("/forecast/AAA")
    assert fr.status_code == 200
    fbody = fr.json()
    assert fbody["symbol"] == "AAA"
    _assert_research_honesty(fbody)

    rk = TestClient(app).get("/ranking")
    assert rk.status_code == 200
    rbody = rk.json()
    assert rbody["n"] == 1
    assert rbody["rankings"][0]["symbol"] == "AAA"
    _assert_research_honesty(rbody)


def test_forecast_bad_config_fail_closed() -> None:
    response = TestClient(app).get(
        "/forecast/AAA", params={"config_path": "configs/does_not_exist.yaml"}
    )
    assert response.status_code == 404
    detail = response.json()
    assert detail.get("live_pnl_claim") is not True


def test_ranking_path_traversal_fail_closed() -> None:
    response = TestClient(app).get("/ranking", params={"config_path": "../pyproject.toml"})
    assert response.status_code == 400
    assert response.json().get("live_pnl_claim") is not True


def test_backtest_artifact_includes_research_only_bool(tmp_path: Path) -> None:
    cfg = SimpleNamespace(data=SimpleNamespace(root=str(tmp_path)))
    request = SimpleNamespace(config_path="configs/backtest.yaml")
    bars = pl.DataFrame({"event_time": ["2026-01-01"], "source": ["synthetic"]})
    result = SimpleNamespace(
        frictionless=False,
        source_note="synthetic_only",
        metrics={"observations": 1, "live_pnl_claim": True},
        fills=pl.DataFrame({"security_id": ["A"]}),
        equity=pl.DataFrame({"nav": [100.0]}),
    )
    artifact = _write_backtest_artifact(cfg, request, bars, result)
    assert artifact["research_only"] is True
    assert artifact["live_pnl_claim"] is False
    assert artifact["claim"] == "research_only"
