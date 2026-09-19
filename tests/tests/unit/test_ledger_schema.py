"""Paper ledger schema validation helper (Wave 6)."""

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from quant_fund.config.loader import load_config
from quant_fund.paper.ledger import (
    PaperLedger,
    latest_run_id,
    promotion_dry_run,
    validate_ledger_schema,
    validate_promotion_dry_run_receipt,
)
from quant_fund.paper.loop import run_paper_loop


def _mini_cfg(tmp_path: Path):
    cfg = load_config("configs/paper.yaml")
    cfg.data.root = tmp_path
    cfg.risk_gate.max_name = 0.5
    cfg.risk_gate.max_net = 0.5
    cfg.risk_gate.max_gross = 2.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.max_participation = 1.0
    cfg.costs.frictionless = True
    return cfg


def _bars(n=6):
    rows = []
    for d in range(n):
        t = datetime(2024, 1, 1 + d, tzinfo=UTC)
        for sid, px in [("A", 100.0 + d), ("B", 50.0)]:
            rows.append(
                {
                    "security_id": sid,
                    "event_time": t,
                    "open": px,
                    "close": px,
                    "close_total_return": px,
                    "volume": 1e6,
                    "adv": 1e8,
                    "vol_20": 0.02,
                    "source": "synthetic",
                }
            )
    return pl.DataFrame(rows)


def _weights(n=5):
    rows = []
    for d in range(n):
        t = datetime(2024, 1, 1 + d, tzinfo=UTC)
        rows.append({"event_time": t, "security_id": "A", "target_weight": 0.1})
        rows.append({"event_time": t, "security_id": "B", "target_weight": -0.05})
    return pl.DataFrame(rows)


def test_validate_ledger_schema_ok_after_paper_run(tmp_path: Path) -> None:
    cfg = _mini_cfg(tmp_path)
    result = run_paper_loop(
        _bars(),
        cfg,
        champion_weights=_weights(),
        initial_nav=100_000.0,
        max_steps=3,
        run_id="schema-ok",
        prefer_latest=False,
    )
    report = validate_ledger_schema(tmp_path / "metadata" / "paper" / "schema-ok")
    assert report["ok"] is True, report
    assert report["schema_version"] == PaperLedger.SCHEMA_VERSION
    assert report["present"]["meta.json"]
    assert report["present"]["broker_state.json"]
    assert report["present"]["analytics_export.json"]
    assert (Path(result.paths["analytics_export"])).is_file()


def test_validate_ledger_schema_strict_rejects_missing_completion_artifacts(tmp_path: Path) -> None:
    cfg = _mini_cfg(tmp_path)
    run_paper_loop(
        _bars(),
        cfg,
        champion_weights=_weights(),
        initial_nav=100_000.0,
        max_steps=3,
        run_id="strict-missing",
        prefer_latest=False,
    )
    root = tmp_path / "metadata" / "paper" / "strict-missing"
    (root / "broker_state.json").unlink()
    (root / "analytics_export.json").unlink()

    legacy = validate_ledger_schema(root)
    strict = validate_ledger_schema(root, require_complete=True)

    assert legacy["ok"] is True, legacy
    assert strict["ok"] is False, strict
    assert "broker_state.json_missing" in strict["errors"]
    assert "analytics_export.json_missing" in strict["errors"]
    assert strict["require_complete"] is True


def test_validate_ledger_schema_strict_rejects_meta_only_run(tmp_path: Path) -> None:
    root = tmp_path / "metadata" / "paper" / "meta-only"
    root.mkdir(parents=True)
    (root / "meta.json").write_text(
        json.dumps(
            {
                "run_id": "meta-only",
                "schema_version": PaperLedger.SCHEMA_VERSION,
                "label": "PAPER_SIMULATED",
                "disclaimer": "research only",
            }
        )
    )

    legacy = validate_ledger_schema(root)
    strict = validate_ledger_schema(root, require_complete=True)

    assert legacy["ok"] is True, legacy
    assert strict["ok"] is False, strict
    assert "broker_state.json_missing" in strict["errors"]
    assert "equity.parquet_missing" in strict["errors"]
    assert "orders.parquet_missing" in strict["errors"]
    assert "cash_ledger.parquet_missing" in strict["errors"]


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ("not-json", "analytics_export_invalid_json:"),
        (
            json.dumps(["not", "an", "export"]),
            "analytics_export_invalid:analytics_export_not_a_dict",
        ),
        (
            json.dumps({"live_pnl_claim": True}),
            "analytics_export_invalid:live_pnl_claim_must_be_false",
        ),
    ],
)
def test_validate_ledger_schema_validates_persisted_analytics_export(
    tmp_path: Path,
    payload: str,
    expected: str,
) -> None:
    cfg = _mini_cfg(tmp_path)
    run_paper_loop(
        _bars(),
        cfg,
        champion_weights=_weights(),
        initial_nav=100_000.0,
        max_steps=3,
        run_id="analytics-schema",
        prefer_latest=False,
    )
    analytics_path = tmp_path / "metadata" / "paper" / "analytics-schema" / "analytics_export.json"
    analytics_path.write_text(payload)

    report = validate_ledger_schema(analytics_path.parent)

    assert report["ok"] is False
    assert any(error.startswith(expected) for error in report["errors"])


def test_validate_ledger_schema_binds_analytics_export_identity(tmp_path: Path) -> None:
    cfg = _mini_cfg(tmp_path)
    run_paper_loop(
        _bars(),
        cfg,
        champion_weights=_weights(),
        initial_nav=100_000.0,
        max_steps=3,
        run_id="analytics-identity",
        prefer_latest=False,
    )
    analytics_path = (
        tmp_path / "metadata" / "paper" / "analytics-identity" / "analytics_export.json"
    )
    analytics = json.loads(analytics_path.read_text())
    assert analytics["run_id"] == "analytics-identity"
    analytics["run_id"] = "another-run"
    analytics_path.write_text(json.dumps(analytics))

    report = validate_ledger_schema(analytics_path.parent)

    assert report["ok"] is False
    assert "analytics_export_run_id_mismatch" in report["errors"]


def test_validate_ledger_schema_rejects_tampered_analytics_export(tmp_path: Path) -> None:
    cfg = _mini_cfg(tmp_path)
    run_paper_loop(
        _bars(),
        cfg,
        champion_weights=_weights(),
        initial_nav=100_000.0,
        max_steps=3,
        run_id="analytics-digest",
        prefer_latest=False,
    )
    analytics_path = tmp_path / "metadata" / "paper" / "analytics-digest" / "analytics_export.json"
    analytics = json.loads(analytics_path.read_text())
    assert isinstance(analytics["analytics_export_sha256"], str)
    analytics["label"] = "TAMPERED"
    analytics_path.write_text(json.dumps(analytics))

    report = validate_ledger_schema(analytics_path.parent)

    assert report["ok"] is False
    assert "analytics_export_sha256_mismatch" in report["errors"]


def test_validate_ledger_schema_flags_live_promote(tmp_path: Path) -> None:
    root = tmp_path / "metadata" / "paper" / "bad"
    root.mkdir(parents=True)
    (root / "meta.json").write_text(
        json.dumps(
            {
                "run_id": "bad",
                "schema_version": 2,
                "label": "PAPER_SIMULATED",
                "disclaimer": "x",
                "live_pnl_claim": True,
            }
        )
    )
    (root / "broker_state.json").write_text(
        json.dumps(
            {
                "run_id": "bad",
                "schema_version": 2,
                "step": 1,
                "champion": {"slot": "champion", "cash": 1.0, "shares": {}, "allow_capital": True},
            }
        )
    )
    (root / "promotion_dry_run.json").write_text(
        json.dumps(
            {
                "would_promote_paper": True,
                "would_promote_live": True,
                "live_pnl_claim": False,
            }
        )
    )
    report = validate_ledger_schema(root)
    assert report["ok"] is False
    assert any("live_pnl_claim" in e for e in report["errors"])
    assert any("would_promote_live" in e for e in report["errors"])


def test_validate_ledger_schema_missing_meta(tmp_path: Path) -> None:
    root = tmp_path / "empty_run"
    root.mkdir()
    report = validate_ledger_schema(root)
    assert report["ok"] is False
    assert "meta.json_missing" in report["errors"]


def test_validate_ledger_schema_is_total_for_non_object_and_string_version(tmp_path: Path) -> None:
    root = tmp_path / "malformed_run"
    root.mkdir()
    (root / "meta.json").write_text(json.dumps(["not", "metadata"]))
    (root / "broker_state.json").write_text(json.dumps("not-state"))
    report = validate_ledger_schema(root)
    assert report["ok"] is False
    assert "meta.json_not_object" in report["errors"]
    assert "broker_state_not_object" in report["errors"]

    (root / "meta.json").write_text(
        json.dumps(
            {
                "run_id": root.name,
                "schema_version": "2",
                "label": "PAPER_SIMULATED",
                "disclaimer": "x",
            }
        )
    )
    report = validate_ledger_schema(root)
    assert "meta_schema_version_not_int" in report["errors"]


@pytest.mark.parametrize(
    ("filename", "error_prefix"),
    [
        ("meta.json", "meta.json_invalid_json:"),
        ("broker_state.json", "broker_state_invalid_json:"),
        ("promotion_dry_run.json", "promotion_dry_run_invalid_json:"),
    ],
)
def test_validate_ledger_schema_reports_unreadable_json_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    error_prefix: str,
) -> None:
    root = tmp_path / "unreadable-run"
    root.mkdir()
    (root / "meta.json").write_text(
        json.dumps(
            {
                "run_id": root.name,
                "schema_version": 2,
                "label": "PAPER_SIMULATED",
                "disclaimer": "x",
            }
        )
    )
    (root / "broker_state.json").write_text(
        json.dumps(
            {
                "run_id": root.name,
                "schema_version": 2,
                "step": 1,
                "champion": {
                    "slot": "champion",
                    "cash": 1.0,
                    "shares": {},
                    "allow_capital": True,
                },
            }
        )
    )
    (root / "promotion_dry_run.json").write_text(
        json.dumps(
            {
                "would_promote_paper": False,
                "would_promote_live": False,
                "live_pnl_claim": False,
                "research_only": True,
            }
        )
    )
    original_read_text = Path.read_text

    def fail_for_target(self: Path, *args: object, **kwargs: object) -> str:
        if self == root / filename:
            raise OSError("simulated unreadable artifact")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_for_target)

    report = validate_ledger_schema(root)

    assert report["ok"] is False
    assert any(error.startswith(error_prefix) for error in report["errors"])


def test_validate_ledger_schema_binds_artifact_run_identity(tmp_path: Path) -> None:
    root = tmp_path / "metadata" / "paper" / "run-a"
    root.mkdir(parents=True)
    (root / "meta.json").write_text(
        json.dumps(
            {
                "run_id": "run-b",
                "schema_version": 2,
                "label": "PAPER_SIMULATED",
                "disclaimer": "x",
            }
        )
    )
    (root / "broker_state.json").write_text(
        json.dumps(
            {
                "run_id": "run-c",
                "schema_version": 2,
                "step": 0,
                "champion": {"slot": "champion", "cash": 1.0, "shares": {}, "allow_capital": False},
            }
        )
    )
    report = validate_ledger_schema(root)
    assert report["ok"] is False
    assert "meta_run_id_mismatch" in report["errors"]
    assert "broker_state_run_id_mismatch" in report["errors"]


def test_validate_promotion_missing_honesty_flags_fail_closed(tmp_path: Path) -> None:
    """Wave 9: missing would_promote_live / live_pnl_claim / research_only → errors."""
    root = tmp_path / "metadata" / "paper" / "missing-flags"
    root.mkdir(parents=True)
    (root / "meta.json").write_text(
        json.dumps(
            {
                "run_id": "missing-flags",
                "schema_version": 2,
                "label": "PAPER_SIMULATED",
                "disclaimer": "x",
            }
        )
    )
    (root / "broker_state.json").write_text(
        json.dumps(
            {
                "run_id": "missing-flags",
                "schema_version": 2,
                "step": 1,
                "champion": {"slot": "champion", "cash": 1.0, "shares": {}, "allow_capital": True},
            }
        )
    )
    (root / "promotion_dry_run.json").write_text(
        json.dumps({"would_promote_paper": True})  # honesty flags absent
    )
    report = validate_ledger_schema(root)
    assert report["ok"] is False
    errs = report["errors"]
    assert any("promotion_missing:would_promote_live" in e for e in errs)
    assert any("promotion_missing:live_pnl_claim" in e for e in errs)
    assert any("promotion_missing:research_only" in e for e in errs)


def test_validate_promotion_research_only_false_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "metadata" / "paper" / "bad-research"
    root.mkdir(parents=True)
    (root / "meta.json").write_text(
        json.dumps(
            {
                "run_id": "bad-research",
                "schema_version": 2,
                "label": "PAPER_SIMULATED",
                "disclaimer": "x",
            }
        )
    )
    (root / "broker_state.json").write_text(
        json.dumps(
            {
                "run_id": "bad-research",
                "schema_version": 2,
                "step": 1,
                "champion": {"slot": "champion", "cash": 1.0, "shares": {}, "allow_capital": True},
            }
        )
    )
    (root / "promotion_dry_run.json").write_text(
        json.dumps(
            {
                "would_promote_paper": False,
                "would_promote_live": False,
                "live_pnl_claim": False,
                "research_only": False,
            }
        )
    )
    report = validate_ledger_schema(root)
    assert report["ok"] is False
    assert any("research_only_must_be_true" in e for e in report["errors"])


def test_validate_promotion_dry_run_receipt_helper() -> None:
    ok = validate_promotion_dry_run_receipt(
        {
            "would_promote_paper": True,
            "would_promote_live": False,
            "live_pnl_claim": False,
            "research_only": True,
        }
    )
    assert ok == []
    bad = validate_promotion_dry_run_receipt({"would_promote_live": True})
    assert any("would_promote_live_must_be_false" in e for e in bad)
    assert any("promotion_missing:live_pnl_claim" in e for e in bad)


def test_validate_promotion_dry_run_rejects_non_string_run_id() -> None:
    errors = validate_promotion_dry_run_receipt(
        {
            "run_id": 123,
            "would_promote_live": False,
            "live_pnl_claim": False,
            "research_only": True,
        }
    )
    assert "promotion_run_id_invalid" in errors


@pytest.mark.parametrize("payload", [["bad"], "bad"])
def test_latest_run_id_rejects_malformed_identity(tmp_path: Path, payload: object) -> None:
    latest = tmp_path / "metadata" / "paper"
    latest.mkdir(parents=True)
    (latest / "latest_run.json").write_text(json.dumps(payload))
    assert latest_run_id(tmp_path) is None


@pytest.mark.parametrize("payload", [["bad"], "bad"])
def test_load_broker_state_rejects_malformed_json_objects(tmp_path: Path, payload: object) -> None:
    run_dir = tmp_path / "metadata" / "paper" / "safe-run"
    run_dir.mkdir(parents=True)
    (run_dir / "broker_state.json").write_text(json.dumps(payload))
    from quant_fund.paper.ledger import load_broker_state

    assert load_broker_state(tmp_path, "safe-run") is None


def test_promotion_dry_run_builder_hard_closes_live() -> None:
    receipt = promotion_dry_run(
        mean_l1=0.1,
        max_l1=0.2,
        n_steps=10,
        champion_nav=1e6,
        shadow_gross=1.0,
        max_mean_l1=0.5,
        min_steps=5,
        data_source="SYNTHETIC",
    )
    assert receipt["would_promote_live"] is False
    assert receipt["live_pnl_claim"] is False
    assert receipt["research_only"] is True
    assert validate_promotion_dry_run_receipt(receipt) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mean_l1": float("nan")},
        {"max_l1": float("inf")},
        {"max_mean_l1": -1.0},
        {"n_steps": -1},
        {"champion_nav": float("nan")},
        {"rolling_window": 0},
    ],
)
def test_promotion_dry_run_rejects_invalid_diagnostics(kwargs: dict[str, object]) -> None:
    base: dict[str, object] = {
        "mean_l1": 0.1,
        "max_l1": 0.2,
        "n_steps": 10,
        "champion_nav": 1e6,
        "shadow_gross": 1.0,
        "max_mean_l1": 0.5,
        "min_steps": 5,
        "data_source": "SYNTHETIC",
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        promotion_dry_run(**base)  # type: ignore[arg-type]


def test_cash_ledger_parquet_schema_roundtrip_and_fail_closed(tmp_path: Path) -> None:
    """Wave 11: cash_ledger.parquet required columns + nonfinite fail-closed."""
    cfg = _mini_cfg(tmp_path)
    result = run_paper_loop(
        _bars(n=8),
        cfg,
        champion_weights=_weights(n=6),
        initial_nav=100_000.0,
        max_steps=4,
        run_id="cash-schema",
        prefer_latest=False,
    )
    root = tmp_path / "metadata" / "paper" / "cash-schema"
    cash_path = root / "cash_ledger.parquet"
    assert cash_path.is_file(), f"expected cash_ledger from fills; paths={result.paths}"
    cash = pl.read_parquet(cash_path)
    for col in (
        "asof",
        "slot",
        "order_id",
        "security_id",
        "cash_delta",
        "notional",
        "fees",
        "side",
    ):
        assert col in cash.columns
    report = validate_ledger_schema(root)
    assert report["ok"] is True, report
    assert report["present"].get("cash_ledger.parquet") is True

    # Fail-closed: strip a required column
    broken = tmp_path / "metadata" / "paper" / "cash-broken"
    broken.mkdir(parents=True)
    (broken / "meta.json").write_text(
        json.dumps(
            {
                "run_id": "cash-broken",
                "schema_version": 2,
                "label": "PAPER_SIMULATED",
                "disclaimer": "x",
            }
        )
    )
    (broken / "broker_state.json").write_text(
        json.dumps(
            {
                "run_id": "cash-broken",
                "schema_version": 2,
                "step": 1,
                "champion": {"slot": "champion", "cash": 1.0, "shares": {}, "allow_capital": True},
            }
        )
    )
    cash.select([c for c in cash.columns if c != "cash_delta"]).write_parquet(
        broken / "cash_ledger.parquet"
    )
    bad = validate_ledger_schema(broken)
    assert bad["ok"] is False
    assert any("cash_ledger_missing:cash_delta" in e for e in bad["errors"])

    # Fail-closed: nonfinite cash_delta
    broken2 = tmp_path / "metadata" / "paper" / "cash-nan"
    broken2.mkdir(parents=True)
    (broken2 / "meta.json").write_text((broken / "meta.json").read_text())
    (broken2 / "broker_state.json").write_text((broken / "broker_state.json").read_text())
    bad_cash = cash.with_columns(pl.lit(float("nan")).alias("cash_delta"))
    bad_cash.write_parquet(broken2 / "cash_ledger.parquet")
    bad2 = validate_ledger_schema(broken2)
    assert bad2["ok"] is False
    assert any("cash_ledger_cash_delta_nonfinite" in e for e in bad2["errors"])
