"""Wave 7: analytics_export.json schema validation + paper roundtrip."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from quant_fund.config.loader import load_config
from quant_fund.metrics.analytics import (
    ANALYTICS_SCHEMA_KEYS,
    analytics_export_digest,
    book_diagnostics,
    export_analytics_dict,
    validate_analytics_export,
)
from quant_fund.paper.loop import run_paper_loop


def test_validate_analytics_export_ok_and_fail_closed_live_claim() -> None:
    diag = book_diagnostics(
        np.random.default_rng(1).normal(0, 0.01, size=40),
        weights=np.array([0.05, -0.02]),
        cov=np.eye(2) * 0.0004,
        data_source="SYNTHETIC",
    )
    blob = export_analytics_dict(diag)
    report = validate_analytics_export(blob)
    assert report["ok"] is True
    assert report["live_pnl_claim"] is False
    assert not report["errors"]
    for key in ANALYTICS_SCHEMA_KEYS:
        assert key in report["present_schema_keys"]

    # Mutate to claim live P&L — must fail closed
    bad = dict(blob)
    bad["live_pnl_claim"] = True
    bad_report = validate_analytics_export(bad)
    assert bad_report["ok"] is False
    assert "live_pnl_claim_must_be_false" in bad_report["errors"]

    # research_only=False also fail-closed
    bad2 = dict(blob)
    bad2["research_only"] = False
    r2 = validate_analytics_export(bad2)
    assert r2["ok"] is False
    assert "research_only_must_be_true_or_absent" in r2["errors"]

    # Missing schema keys
    thin = {"label": "x", "live_pnl_claim": False, "research_only": True}
    r3 = validate_analytics_export(thin)
    assert r3["ok"] is False
    assert any(e.startswith("missing_schema_key:") for e in r3["errors"])

    # None / wrong type
    assert validate_analytics_export(None)["ok"] is False
    assert validate_analytics_export([1, 2, 3])["ok"] is False  # type: ignore[arg-type]


def test_analytics_export_digest_is_canonical_and_self_excluding() -> None:
    blob = {"label": "PAPER_SIMULATED", "nested": {"b": 2, "a": 1}}
    digest = analytics_export_digest(blob)

    assert len(digest) == 64
    assert digest == analytics_export_digest({**blob, "analytics_export_sha256": digest})
    assert digest != analytics_export_digest({**blob, "nested": {"a": 1, "b": 3}})


def test_validate_analytics_export_rejects_tampered_digest() -> None:
    diag = book_diagnostics(
        np.random.default_rng(2).normal(0, 0.01, size=40), data_source="SYNTHETIC"
    )
    blob = export_analytics_dict(diag)
    blob["analytics_export_sha256"] = analytics_export_digest(blob)
    blob["label"] = "TAMPERED"

    report = validate_analytics_export(blob)

    assert report["ok"] is False
    assert "analytics_export_sha256_mismatch" in report["errors"]


def test_export_forces_live_pnl_claim_false() -> None:
    dirty = {
        "label": "HACK",
        "data_source": "SYNTHETIC",
        "research_only": False,
        "live_pnl_claim": True,
        "disclaimer": "x",
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
    out = export_analytics_dict(dirty)
    assert out["live_pnl_claim"] is False
    assert out["research_only"] is True
    assert validate_analytics_export(out)["ok"] is True


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


def _bars(n: int = 6):
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


def _weights(n: int = 6):
    rows = []
    for d in range(n):
        t = datetime(2024, 1, 1 + d, tzinfo=UTC)
        rows.append({"event_time": t, "security_id": "A", "target_weight": 0.1})
        rows.append({"event_time": t, "security_id": "B", "target_weight": -0.05})
    return pl.DataFrame(rows)


def test_paper_analytics_export_json_roundtrip(tmp_path: Path) -> None:
    cfg = _mini_cfg(tmp_path)
    result = run_paper_loop(
        _bars(6),
        cfg,
        champion_weights=_weights(6),
        run_id="wave7-ae-roundtrip",
        initial_nav=100_000.0,
        max_steps=5,
        prefer_latest=False,
    )
    assert result.metrics.get("analytics_export_ok") is True
    ae_path = Path(result.paths["analytics_export"])
    assert ae_path.is_file()
    loaded = json.loads(ae_path.read_text())
    report = validate_analytics_export(loaded)
    assert report["ok"] is True, report["errors"]
    assert loaded["live_pnl_claim"] is False
    assert loaded["research_only"] is True
    # Roundtrip: export → disk → load → re-validate → re-export forces flags
    again = export_analytics_dict(loaded)
    assert again["live_pnl_claim"] is False
    assert validate_analytics_export(again)["ok"] is True

    # Fail-closed: if disk is mutated to claim live P&L, validator rejects
    mutated = dict(loaded)
    mutated["live_pnl_claim"] = True
    ae_path.write_text(json.dumps(mutated, indent=2, default=str))
    disk_bad = json.loads(ae_path.read_text())
    assert validate_analytics_export(disk_bad)["ok"] is False
