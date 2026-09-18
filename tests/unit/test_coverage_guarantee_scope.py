"""Day Wave 41: Jackknife+/CV+ coverage floors are marginal, not training-conditional.

Research-only honesty keys on model metadata and bench blobs. No invented
training-conditional bounds. No live P&L claim.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models.cv_plus import CVPlus
from quant_fund.models.jackknife_plus import JackknifePlus
from quant_fund.research.benches import bench_cv_plus
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent

_SCOPE = "marginal_exchangeable"
_CLAIM_NEEDLE = "not training-conditional"


def _assert_marginal_honesty(blob: dict) -> None:
    assert blob.get("coverage_guarantee_scope") == _SCOPE
    claim = str(blob.get("coverage_guarantee_claim", ""))
    assert "marginal" in claim.lower()
    assert _CLAIM_NEEDLE in claim
    assert blob.get("research_only") is True
    assert blob.get("live_pnl_claim") is not True
    assert family_blob_forbidden_metrics_absent(blob) is True


def test_jackknife_plus_metadata_marginal_scope() -> None:
    y = np.linspace(-1.0, 1.0, 40)
    meta = JackknifePlus(0.10).fit(y, np.zeros_like(y)).metadata().extra
    _assert_marginal_honesty(meta)
    assert meta["coverage_identity"] == "1-2*alpha"


def test_cv_plus_metadata_marginal_scope_minmax_and_plus() -> None:
    y = np.linspace(-1.0, 1.0, 40)
    for agg in ("minmax", "plus"):
        meta = CVPlus(0.10, n_folds=5, aggregation=agg).fit(y, np.zeros_like(y)).metadata().extra
        _assert_marginal_honesty(meta)
        expected = "1-alpha" if agg == "minmax" else "1-2*alpha"
        assert meta["coverage_identity"] == expected


def test_bench_cv_plus_fixture_surfaces_marginal_scope() -> None:
    """SYNTHETIC/unit fixture path — no panel required."""
    blob = bench_cv_plus(seed=23)
    assert blob  # fixture path always returns
    assert blob.get("dgp") == "fixture"
    _assert_marginal_honesty(blob)
    assert "coverage_identity" in blob
    assert "coverage_floor" in blob


def test_scope_string_exact_and_floor_identities() -> None:
    """Lab contract: exact scope token; floors stay marginal identities only."""
    from quant_fund.models.cv_plus import cv_plus_coverage_level
    from quant_fund.models.jackknife_plus import jackknife_plus_coverage_level

    assert _SCOPE == "marginal_exchangeable"
    assert jackknife_plus_coverage_level(0.10) == 0.80
    assert cv_plus_coverage_level(0.10, "minmax") == 0.90
    assert cv_plus_coverage_level(0.10, "plus") == 0.80
    assert cv_plus_coverage_level(0.10, "jaw") == 0.80
