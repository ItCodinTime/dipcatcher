"""Wave 8: Christoffersen CC extremes — LR identity, joint reject, boundaries."""

import numpy as np
import pytest

from quant_fund.metrics.probability import (
    christoffersen_cc,
    christoffersen_independence,
    kupiec_pof,
)


def test_lr_cc_equals_lr_uc_plus_lr_ind_within_float_tol() -> None:
    """Identity: LR_cc = LR_uc + LR_ind (Christoffersen 1998)."""
    rng = np.random.default_rng(7)
    # Mix a few regimes so both UC and IND are finite
    hits = (rng.random(400) < 0.05).astype(float)
    lr_cc, _, extras = christoffersen_cc(hits, 0.05)
    rate, lr_uc, _ = kupiec_pof(hits, 0.05)
    lr_ind, _, _ = christoffersen_independence(hits)
    assert np.isfinite(lr_cc) and np.isfinite(lr_uc) and np.isfinite(lr_ind)
    assert lr_cc == pytest.approx(lr_uc + lr_ind, rel=0.0, abs=1e-12)
    assert extras["kupiec_lr"] == pytest.approx(lr_uc, abs=1e-12)
    assert extras["ind_lr"] == pytest.approx(lr_ind, abs=1e-12)
    assert extras["hit_rate"] == pytest.approx(rate, abs=1e-12)


def test_joint_reject_under_clustering_and_miscalibration() -> None:
    """Clustered + wrong rate → CC should reject hard (p tiny / LR large)."""
    # Long runs of hits then quiet — dependent; hit rate ~0.25 vs alpha=0.05
    blocks: list[float] = []
    for _ in range(12):
        blocks.extend([0.0] * 15)
        blocks.extend([1.0] * 5)
    hits = np.asarray(blocks, dtype=float)
    assert 0.20 < hits.mean() < 0.30
    lr_cc, p_cc, extras = christoffersen_cc(hits, 0.05)
    assert np.isfinite(lr_cc)
    assert lr_cc > extras["kupiec_lr"]  # independence adds positive LR
    assert lr_cc > extras["ind_lr"]
    assert p_cc < 0.01


def test_cc_empty_and_short_series_boundaries() -> None:
    empty = np.asarray([], dtype=float)
    lr, p, extras = christoffersen_cc(empty, 0.05)
    assert np.isnan(lr) and np.isnan(p)

    short = np.zeros(8, dtype=float)  # below independence n<12 and kupiec n<10
    lr2, p2, _ = christoffersen_cc(short, 0.05)
    assert np.isnan(lr2) and np.isnan(p2)

    # All hits / no hits → Kupiec LR nan → CC nan (even if IND finite)
    all_hit = np.ones(50, dtype=float)
    lr3, p3, ex3 = christoffersen_cc(all_hit, 0.05)
    assert np.isnan(lr3) and np.isnan(p3)
    assert np.isnan(ex3["kupiec_lr"])

    none_hit = np.zeros(50, dtype=float)
    lr4, p4, ex4 = christoffersen_cc(none_hit, 0.05)
    assert np.isnan(lr4) and np.isnan(p4)
    assert np.isnan(ex4["kupiec_lr"])


def test_cc_identity_on_iid_near_calibration() -> None:
    rng = np.random.default_rng(0)
    hits = (rng.random(500) < 0.05).astype(float)
    lr_cc, _, extras = christoffersen_cc(hits, 0.05)
    if np.isfinite(extras["kupiec_lr"]) and np.isfinite(extras["ind_lr"]):
        assert lr_cc == pytest.approx(extras["kupiec_lr"] + extras["ind_lr"], abs=1e-12)
