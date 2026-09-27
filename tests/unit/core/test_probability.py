"""Probability / PIT / Kupiec / Christoffersen metrics.

Happy-path + Wave 46 empty/all-NaN/mismatch/bad-alpha edges.
Research-diagnostic only — live_pnl_claim=false; not a live edge.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.probability import (
    acerbi_szekely_z1,
    acerbi_szekely_z2,
    brier_score,
    christoffersen_cc,
    christoffersen_independence,
    expected_calibration_error,
    kupiec_pof,
    log_loss,
    pit_ks,
)

RESEARCH_ONLY = True
LIVE_PNL_CLAIM = False


def test_brier_perfect() -> None:
    y = np.array([0.0, 1.0, 1.0, 0.0])
    assert brier_score(y, y) == 0.0


def test_log_loss_better_when_confident_and_right() -> None:
    y = np.array([1.0, 0.0])
    assert log_loss(np.array([0.9, 0.1]), y) < log_loss(np.array([0.6, 0.4]), y)


def test_kupiec_matches_when_rate_equals_alpha() -> None:
    rng = np.random.default_rng(1)
    hits = (rng.random(2000) < 0.05).astype(float)
    rate, _, p = kupiec_pof(hits, 0.05)
    assert abs(rate - 0.05) < 0.02
    assert p > 0.01


def test_pit_ks_uniform() -> None:
    u = np.linspace(0.01, 0.99, 200)
    stat, p = pit_ks(u)
    assert stat < 0.1
    assert p > 0.05


def test_pit_ks_spiked_rejects_uniform() -> None:
    """DayWave7 thin honesty: mass piled near 0/1 fails Uniform(0,1) KS.

    Reuses pit_ks only — no new API. Research-diagnostic; live_pnl_claim=false.
    """
    # Half mass at ~0, half at ~1 (after clip still extreme U-shape)
    spiked = np.concatenate([np.full(100, 1e-6), np.full(100, 1.0 - 1e-6)])
    stat, p = pit_ks(spiked)
    assert np.isfinite(stat) and np.isfinite(p)
    assert stat > 0.3
    assert p < 0.01


def test_ece_zero_when_aligned() -> None:
    p = np.array([0.1, 0.1, 0.9, 0.9])
    y = np.array([0.1, 0.1, 0.9, 0.9])
    assert expected_calibration_error(p, y, n_bins=2) < 0.05


# --- Wave 46 edges ---


def test_brier_empty_all_nan_and_mismatch() -> None:
    assert np.isnan(brier_score(np.array([]), np.array([])))
    assert np.isnan(brier_score(np.array([np.nan, np.nan]), np.array([np.nan, 1.0])))
    # Finite pair survives alongside NaNs
    assert np.isclose(
        brier_score(np.array([0.5, np.nan]), np.array([0.5, np.nan])),
        0.0,
    )
    with pytest.raises(ValueError, match="length mismatch"):
        brier_score(np.array([0.1, 0.2]), np.array([1.0]))
    with pytest.raises(ValueError, match="must be 1d"):
        brier_score(np.array([[0.1, 0.2]]), np.array([0.0, 1.0]))


def test_log_loss_empty_all_nan_and_mismatch() -> None:
    assert np.isnan(log_loss(np.array([]), np.array([])))
    assert np.isnan(log_loss(np.array([np.nan]), np.array([np.nan])))
    with pytest.raises(ValueError, match="length mismatch"):
        log_loss(np.array([0.9, 0.1]), np.array([1.0]))


def test_ece_empty_short_bad_bins_and_mismatch() -> None:
    assert np.isnan(expected_calibration_error(np.array([]), np.array([]), n_bins=10))
    assert np.isnan(
        expected_calibration_error(np.array([0.1, 0.2]), np.array([0.0, 1.0]), n_bins=10)
    )
    assert np.isnan(
        expected_calibration_error(np.array([np.nan, np.nan]), np.array([np.nan, np.nan]), n_bins=2)
    )
    assert np.isnan(expected_calibration_error(np.ones(20), np.zeros(20), n_bins=0))
    with pytest.raises(ValueError, match="length mismatch"):
        expected_calibration_error(np.array([0.1, 0.9]), np.array([0.0]), n_bins=2)


def test_kupiec_empty_short_bad_alpha() -> None:
    rate, lr, p = kupiec_pof(np.array([]), 0.05)
    assert np.isnan(rate) and np.isnan(lr) and np.isnan(p)
    rate2, lr2, p2 = kupiec_pof(np.zeros(9), 0.05)  # n < 10
    assert np.isnan(rate2) and np.isnan(lr2) and np.isnan(p2)
    for bad in (0.0, 1.0, -0.1, float("nan"), float("inf")):
        r, lr_bad, pv = kupiec_pof(np.zeros(20), bad)
        assert np.isnan(r) and np.isnan(lr_bad) and np.isnan(pv)
    # All hits / no hits → rate finite, LR NaN
    r0, l0, p0 = kupiec_pof(np.zeros(20), 0.05)
    assert r0 == 0.0 and np.isnan(l0) and np.isnan(p0)
    r1, l1, p1 = kupiec_pof(np.ones(20), 0.05)
    assert r1 == 1.0 and np.isnan(l1) and np.isnan(p1)


def test_pit_ks_empty_short_all_nan() -> None:
    assert all(np.isnan(x) for x in pit_ks(np.array([])))
    assert all(np.isnan(x) for x in pit_ks(np.linspace(0.1, 0.9, 7)))
    assert all(np.isnan(x) for x in pit_ks(np.full(20, np.nan)))


def test_christoffersen_independence_empty_short() -> None:
    lr, p, counts = christoffersen_independence(np.array([]))
    assert np.isnan(lr) and np.isnan(p) and counts == {}
    lr2, p2, counts2 = christoffersen_independence(np.zeros(11))
    assert np.isnan(lr2) and np.isnan(p2) and counts2 == {}
    # All-NaN hits → empty after finite filter
    lr3, p3, _ = christoffersen_independence(np.full(30, np.nan))
    assert np.isnan(lr3) and np.isnan(p3)


def test_christoffersen_cc_bad_alpha_empty() -> None:
    lr, p, extras = christoffersen_cc(np.array([]), 0.05)
    assert np.isnan(lr) and np.isnan(p)
    assert np.isnan(extras["hit_rate"])
    lr2, p2, _ = christoffersen_cc(np.zeros(50), 0.0)
    assert np.isnan(lr2) and np.isnan(p2)


def test_acerbi_szekely_empty_no_hits_mismatch() -> None:
    """Empty / no hits / length mismatch — honest NaN or fail-closed."""
    empty = np.array([])
    z1, n1 = acerbi_szekely_z1(empty, empty, empty, 0.05)
    z2, n2 = acerbi_szekely_z2(empty, empty, empty)
    assert np.isnan(z1) and n1 == 0
    assert np.isnan(z2) and n2 == 0

    losses = np.array([0.01, 0.02, 0.03])
    var = np.full(3, 0.10)  # no breaches
    es = np.full(3, 0.15)
    z1b, n1b = acerbi_szekely_z1(losses, var, es, 0.05)
    z2b, n2b = acerbi_szekely_z2(losses, var, es)
    assert np.isnan(z1b) and n1b == 0
    assert np.isnan(z2b) and n2b == 0

    with pytest.raises(ValueError, match="length mismatch"):
        acerbi_szekely_z1(losses, var[:2], es, 0.05)
    with pytest.raises(ValueError, match="length mismatch"):
        acerbi_szekely_z2(losses, var, es[:1])


def test_acerbi_szekely_nonpositive_es_and_alpha_fail_closed() -> None:
    losses = np.array([0.10, 0.20, 0.05])
    var = np.full(3, 0.08)
    es_bad = np.array([0.15, 0.0, 0.15])  # zero ES on a hit (0.20 > 0.08)
    z1, n1 = acerbi_szekely_z1(losses, var, es_bad, 0.25)
    z2, n2 = acerbi_szekely_z2(losses, var, es_bad)
    assert n1 >= 1 and np.isnan(z1)
    assert n2 >= 1 and np.isnan(z2)

    es_neg = np.array([-0.1, 0.15, 0.15])
    z1n, _ = acerbi_szekely_z1(losses, var, es_neg, 0.25)
    assert np.isnan(z1n)

    for bad_a in (0.0, 1.0, -0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="alpha"):
            acerbi_szekely_z1(losses, var, np.full(3, 0.15), bad_a)


def test_acerbi_szekely_z1_z2_distinct_known_fixture() -> None:
    """When n_hits ≠ N·α, Z1 and Z2 diverge (Acerbi–Székely 2014).

    losses=[0.10, 0.20, 0.05, 0.01], VaR=0.08, ES=0.15, α=0.25 → 2 hits, Nα=1.
    Z1 = (0.10/0.15 + 0.20/0.15) / 1 − 1 = 1.0
    Z2 = mean(0.10/0.15, 0.20/0.15) − 1 = 0.0
    Research-diagnostic only — not a live capital claim.
    """
    losses = np.array([0.10, 0.20, 0.05, 0.01])
    var = np.full(4, 0.08)
    es = np.full(4, 0.15)
    z1, n1 = acerbi_szekely_z1(losses, var, es, 0.25)
    z2, n2 = acerbi_szekely_z2(losses, var, es)
    assert n1 == n2 == 2
    assert z1 == pytest.approx(1.0)
    assert z2 == pytest.approx(0.0)
    assert z1 != pytest.approx(z2)


def test_honesty_flags_not_live_claim() -> None:
    assert RESEARCH_ONLY is True
    assert LIVE_PNL_CLAIM is False


def test_probability_scores_mask_out_of_range_never_clip() -> None:
    """Out-of-range probabilities are invalid observations, not clipped.

    A forecast of 1.7 must not be reshaped into a low-looking Brier/log-loss;
    the invalid pair is masked out and only genuine probabilities score. If no
    valid pair remains the score is honest NaN. Research-diagnostic only.
    """
    p = np.array([0.9, 1.7, -0.5, 0.2])
    y = np.array([1.0, 1.0, 0.0, 0.0])
    assert np.isclose(brier_score(p, y), ((0.9 - 1.0) ** 2 + (0.2 - 0.0) ** 2) / 2)
    assert np.isclose(log_loss(p, y), -(np.log(0.9) + np.log(1.0 - 0.2)) / 2)
    assert np.isfinite(expected_calibration_error(p, y, n_bins=2))
    # All out-of-range → honest NaN (never a fabricated finite score)
    bad = np.array([1.7, -0.5])
    hits = np.array([1.0, 0.0])
    assert np.isnan(brier_score(bad, hits))
    assert np.isnan(log_loss(bad, hits))
    assert np.isnan(expected_calibration_error(bad, hits, n_bins=2))
