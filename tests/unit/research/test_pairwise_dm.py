"""Pairwise Diebold–Mariano helper."""

import numpy as np
import pytest

from quant_fund.metrics.inference import pairwise_diebold_mariano


def test_pairwise_dm_prefers_lower_loss():
    rng = np.random.default_rng(0)
    good = rng.normal(0.1, 0.05, size=80)
    bad = rng.normal(0.5, 0.05, size=80)
    rows = pairwise_diebold_mariano({"good": good, "bad": bad})
    assert len(rows) == 1
    assert rows[0]["preferred"] == "good"
    assert rows[0]["p_value"] < 0.05


def test_pairwise_dm_mismatched_lengths_fail_closed():
    """Bugbot regression: length mismatch must raise, never silently truncate."""
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 1.0, 200)
    b = rng.normal(0.0, 1.0, 150)
    with pytest.raises(ValueError, match="must align"):
        pairwise_diebold_mariano({"a": a, "b": b})
    # Same-length series still work
    rows = pairwise_diebold_mariano({"a": a, "b": rng.normal(0.0, 1.0, 200)})
    assert rows[0]["n"] == 200
