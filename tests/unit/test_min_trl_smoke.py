"""Day Wave 5: thin Bailey–LdP MinTRL research smoke.

Gaussian fixture path → finite MinTRL; payload keys omit forbidden
sharpe/sortino/calmar/pnl/nav tokens. Research-diagnostic only — not live P&L.
Edges for ``min_track_record_length`` live in ``test_overfitting_edges.py``.
"""

from __future__ import annotations

import math

import numpy as np

from quant_fund.metrics.analytics import book_diagnostics
from quant_fund.metrics.overfitting import min_trl_from_returns
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent


def test_min_trl_from_returns_gaussian_finite() -> None:
    """Positive-drift Gaussian returns → finite research-only MinTRL."""
    rng = np.random.default_rng(42)
    # Positive mean / moderate vol so per-period observed ratio > 0 vs sr*=0.
    r = rng.normal(0.02, 0.05, size=200)
    blob = min_trl_from_returns(r, conf=0.95)
    assert blob["research_only"] is True
    assert "live_pnl_claim" not in blob  # pnl token forbidden
    assert blob["claim"] == "bailey_ldp_min_trl_diagnostic_only"
    assert blob["n_obs"] == 200
    assert math.isfinite(float(blob["min_track_record_length"]))
    assert math.isfinite(float(blob["min_trl"]))
    assert float(blob["min_trl"]) == float(blob["min_track_record_length"])
    assert float(blob["min_trl"]) > 1.0
    assert math.isfinite(float(blob["track_record_bars"]))
    assert float(blob["track_record_bars"]) >= float(blob["min_trl"])
    assert family_blob_forbidden_metrics_absent(blob) is True


def test_min_trl_from_returns_short_honest_nan() -> None:
    blob = min_trl_from_returns(np.array([0.01, -0.02, 0.0]))
    assert blob["n_obs"] == 3
    assert math.isnan(float(blob["min_trl"]))
    assert math.isnan(float(blob["min_track_record_length"]))
    assert family_blob_forbidden_metrics_absent(blob) is True


def test_book_diagnostics_min_trl_smoke_hygiene() -> None:
    """book_diagnostics nests MinTRL smoke; nested blob stays forbidden-clean."""
    rng = np.random.default_rng(7)
    r = rng.normal(0.015, 0.04, size=120)
    diag = book_diagnostics(r, label="MIN_TRL_SMOKE", data_source="SYNTHETIC")
    assert "min_trl" in diag
    nested = diag["min_trl"]
    assert isinstance(nested, dict)
    assert nested["research_only"] is True
    assert "live_pnl_claim" not in nested  # pnl token forbidden
    assert math.isfinite(float(nested["min_trl"]))
    assert family_blob_forbidden_metrics_absent(nested) is True
    # Nested keys themselves must not tokenize to forbidden headlines.
    for key in nested:
        parts = str(key).lower().replace("-", "_").split("_")
        assert not any(t in {"sharpe", "sortino", "calmar", "pnl", "nav"} for t in parts if t)
