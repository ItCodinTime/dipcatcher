"""Christoffersen independence fixtures with known clustering / independence."""

import numpy as np
import pytest

from quant_fund.metrics.probability import christoffersen_cc, christoffersen_independence


def test_independence_rejects_clustered_hits():
    # Alternating blocks of hits: strongly dependent Markov chain.
    # 20 zeros, 5 ones, 20 zeros, 5 ones, ...
    blocks = []
    for _ in range(8):
        blocks.extend([0] * 20)
        blocks.extend([1] * 5)
    hits = np.asarray(blocks, dtype=float)
    lr, p, counts = christoffersen_independence(hits)
    assert counts["n11"] > 0
    assert counts["n00"] > 0
    assert np.isfinite(lr)
    assert lr > 0
    # Clustered hits should reject independence at conventional levels
    assert p < 0.05


def test_independence_does_not_wildly_reject_iid():
    rng = np.random.default_rng(42)
    hits = (rng.random(500) < 0.05).astype(float)
    lr, p, counts = christoffersen_independence(hits)
    assert "n00" in counts
    assert np.isfinite(p)
    # Under true independence, p should not be tiny (allow rare false reject)
    assert p > 0.001


def test_independence_transition_counts_fixture():
    # Deterministic: 0,1,0,1,... → n01=n10 large, n00=n11=0
    hits = np.array([0, 1] * 40, dtype=float)
    lr, p, counts = christoffersen_independence(hits)
    assert counts["n00"] == 0.0
    assert counts["n11"] == 0.0
    assert counts["n01"] == 40.0
    assert counts["n10"] == 39.0
    assert np.isfinite(lr)


def test_cc_combines_kupiec_and_independence():
    rng = np.random.default_rng(0)
    hits = (rng.random(300) < 0.05).astype(float)
    lr_cc, p_cc, extras = christoffersen_cc(hits, 0.05)
    assert "kupiec_lr" in extras
    assert "ind_lr" in extras
    if np.isfinite(extras["kupiec_lr"]) and np.isfinite(extras["ind_lr"]):
        assert lr_cc == pytest.approx(extras["kupiec_lr"] + extras["ind_lr"])


def test_cc_lr_identity_explicit_fixture():
    """Hardcoded short Markov path with finite UC+IND → exact sum."""
    # 40 zeros + 10 ones + 40 zeros + 10 ones: clustered, elevated rate
    hits = np.asarray([0] * 40 + [1] * 10 + [0] * 40 + [1] * 10, dtype=float)
    lr_cc, _, extras = christoffersen_cc(hits, 0.05)
    assert np.isfinite(lr_cc)
    assert lr_cc == pytest.approx(extras["kupiec_lr"] + extras["ind_lr"], abs=1e-12)
