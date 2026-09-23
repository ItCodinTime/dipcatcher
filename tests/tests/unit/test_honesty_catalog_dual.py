"""Day Wave 10: dual honesty catalogs — research scorecard vs analytics_export.

Research family blobs: FORBIDDEN_RESEARCH_METRIC_KEYS / family_blob_forbidden_metrics_absent.
Paper analytics_export: equity/stress may carry pnl/nav diagnostics; live_pnl_claim must
be false (validate_analytics_export fail-closed). Not a live capital claim.
"""

from __future__ import annotations

import numpy as np

from quant_fund.metrics.analytics import (
    book_diagnostics,
    export_analytics_dict,
    validate_analytics_export,
)
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent


def test_research_family_blob_rejects_pnl_sharpe_tokens() -> None:
    clean = {"mean_ic": 0.1, "coverage": 0.9, "research_only": True}
    assert family_blob_forbidden_metrics_absent(clean) is True
    assert family_blob_forbidden_metrics_absent({"mean_ic": 0.1, "shock_down_pnl": -0.2}) is False
    assert family_blob_forbidden_metrics_absent({"coverage": 0.9, "sharpe": 1.2}) is False
    assert family_blob_forbidden_metrics_absent({"nav_end": 1e5}) is False


def test_analytics_export_allows_equity_nav_pnl_when_live_claim_false() -> None:
    diag = book_diagnostics(
        np.random.default_rng(10).normal(0, 0.01, size=32),
        weights=np.array([0.04, -0.01]),
        cov=np.eye(2) * 0.0004,
        data_source="SYNTHETIC",
    )
    blob = export_analytics_dict(diag)
    assert blob.get("live_pnl_claim") is False
    equity = blob.get("equity") or {}
    assert "nav_start" in equity or "nav_end" in equity
    # Nested stress diagnostics intentionally use *_pnl keys on the export path.
    stress = blob.get("stress") or {}
    assert any("pnl" in str(k).lower() for k in stress) or any(
        "pnl" in str(k).lower() for k in (blob.get("stress_report") or {})
    )
    report = validate_analytics_export(blob)
    assert report["ok"] is True
    assert "live_pnl_claim_must_be_false" not in report["errors"]
    # Different catalog: full export is *not* a research family blob.
    assert family_blob_forbidden_metrics_absent(blob) is False


def test_analytics_export_fail_closed_on_live_pnl_claim_true() -> None:
    diag = book_diagnostics(
        np.random.default_rng(11).normal(0, 0.01, size=24),
        weights=np.array([0.03]),
        cov=np.eye(1) * 0.0004,
        data_source="SYNTHETIC",
    )
    blob = export_analytics_dict(diag)
    bad = dict(blob)
    bad["live_pnl_claim"] = True
    report = validate_analytics_export(bad)
    assert report["ok"] is False
    assert "live_pnl_claim_must_be_false" in report["errors"]
