"""P6.2 statistical-core audit: known-answer tests.

Deterministic, offline, SYNTHETIC fixtures only — research-diagnostic
machinery, never a live Sharpe / P&L claim. Each test pins either a
hand-computed value (closed form / reference implementation) or an exact
structural identity of the cited method.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.scoring import coverage


def test_coverage_masks_nonfinite() -> None:
    """Regression: non-finite observations are masked, not counted as misses."""
    assert coverage(
        np.array([0.5, np.nan]), np.array([0.0, 0.0]), np.array([1.0, 1.0])
    ) == pytest.approx(1.0)
    assert np.isnan(coverage(np.array([np.nan]), np.array([0.0]), np.array([1.0])))
    assert coverage(
        np.array([0.0, 5.0]), np.array([-1.0, 0.0]), np.array([0.5, 2.0])
    ) == pytest.approx(0.5)
