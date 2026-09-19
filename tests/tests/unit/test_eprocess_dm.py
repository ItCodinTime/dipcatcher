"""DayWave4: anytime-valid e-process on forecast loss differentials (Choe–Ramdas).

Research-diagnostic only — live_pnl_claim=false; not a live capital claim.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.evalues import (
    e_process_dm,
    e_process_loss_diff,
    e_process_threshold,
)
from quant_fund.metrics.inference import pairwise_diebold_mariano


def test_constant_positive_diff_grows_wealth() -> None:
    """A clearly worse than B → d = L_A − L_B > 0 → capital grows under H0: E[d]≤0."""
    d = np.full(40, 0.5, dtype=float)
    path = e_process_loss_diff(d=d, lam=0.25, initial_bound=1.0)
    assert path.shape == (40,)
    assert np.all(path >= 0.0)
    assert np.all(np.isfinite(path))
    assert path[-1] > path[0]
    assert path[-1] > 1.0
    assert np.all(np.diff(path) > 0.0)


def test_constant_negative_diff_shrinks_wealth() -> None:
    """A better → d < 0 → capital shrinks (no evidence against H0)."""
    d = np.full(40, -0.5, dtype=float)
    path = e_process_loss_diff(d=d, lam=0.25, initial_bound=1.0)
    assert path[-1] < 1.0
    assert path[-1] < path[0]
    out = e_process_threshold(path, level=0.05)
    assert out["reject"] is False


def test_two_series_align_matches_d() -> None:
    loss_a = np.array([1.0, 1.2, 0.9, 1.1, 1.3], dtype=float)
    loss_b = np.array([0.5, 0.6, 0.4, 0.55, 0.7], dtype=float)
    from_ab = e_process_loss_diff(loss_a, loss_b, lam=0.2)
    from_d = e_process_loss_diff(d=loss_a - loss_b, lam=0.2)
    assert np.allclose(from_ab, from_d)


def test_rng_worse_a_often_crosses_ville() -> None:
    rng = np.random.default_rng(7)
    # A losses ~ N(1.0, 0.1); B ~ N(0.2, 0.1) → strong positive d
    loss_a = rng.normal(1.0, 0.1, size=120)
    loss_b = rng.normal(0.2, 0.1, size=120)
    ep = e_process_dm(loss_a, loss_b, lam=0.3, level=0.05)
    assert ep["n"] == 120
    assert float(ep["mean_loss_diff"]) > 0.5
    assert float(ep["e_final"]) > 20.0
    assert ep["reject"] is True
    assert ep["first_cross"] is not None
    assert float(ep["threshold"]) == 20.0


def test_nullish_diff_rarely_rejects() -> None:
    rng = np.random.default_rng(11)
    d = rng.normal(0.0, 0.15, size=80)
    path = e_process_loss_diff(d=d, lam=0.2, initial_bound=1.0)
    out = e_process_threshold(path, level=0.05)
    assert out["reject"] is False
    assert float(np.max(path)) < 20.0


def test_empty_returns_unit() -> None:
    path = e_process_loss_diff(d=np.asarray([], dtype=float))
    assert path.shape == (1,)
    assert path[0] == 1.0
    ep = e_process_dm(np.asarray([], dtype=float), np.asarray([], dtype=float))
    assert ep["n"] == 0
    assert ep["e_final"] == 1.0
    assert ep["mean_loss_diff"] != ep["mean_loss_diff"]  # NaN
    assert ep["reject"] is False


def test_length_mismatch_fail_closed() -> None:
    with pytest.raises(ValueError, match="align"):
        e_process_loss_diff(np.ones(3), np.ones(4))
    with pytest.raises(ValueError, match="align"):
        e_process_dm(np.ones(3), np.ones(2))


def test_non_finite_pairs_dropped() -> None:
    a = np.array([1.0, np.nan, 1.5, 2.0], dtype=float)
    b = np.array([0.5, 0.4, np.inf, 0.8], dtype=float)
    # finite pairs: index 0 and 3 → d = [0.5, 1.2]
    path = e_process_loss_diff(a, b, lam=0.25)
    assert path.size == 2
    assert np.all(np.isfinite(path))


def test_bad_lam_and_bound() -> None:
    d = np.ones(5)
    for lam in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="lam"):
            e_process_loss_diff(d=d, lam=lam)
    with pytest.raises(ValueError, match="initial_bound"):
        e_process_loss_diff(d=d, initial_bound=0.0)
    with pytest.raises(ValueError, match="initial_bound"):
        e_process_loss_diff(d=d, initial_bound=-1.0)


def test_both_d_and_losses_rejected() -> None:
    with pytest.raises(ValueError, match="either"):
        e_process_loss_diff(np.ones(3), np.zeros(3), d=np.ones(3))


def test_pairwise_optional_e_process_fields() -> None:
    rng = np.random.default_rng(0)
    good = rng.normal(0.1, 0.05, size=80)
    bad = rng.normal(0.5, 0.05, size=80)
    rows = pairwise_diebold_mariano({"good": good, "bad": bad}, include_e_process=True)
    assert len(rows) == 1
    r = rows[0]
    assert "e_final" in r and "e_reject" in r and "e_n" in r
    # bad − good has positive mean → evidence bad is worse when ordered alphabetically?
    # names sorted: bad, good → row a=bad, b=good → d = L_bad − L_good > 0 → reject
    assert r["a"] == "bad" and r["b"] == "good"
    assert float(r["mean_loss_diff"]) > 0.0
    assert float(r["e_final"]) > 1.0
    assert int(r["e_n"]) == 80
    # default path unchanged
    plain = pairwise_diebold_mariano({"good": good, "bad": bad})
    assert "e_final" not in plain[0]


def test_one_step_factor_closed_form() -> None:
    """Single obs x=1, λ=0.25 → e = exp(λ − ψ_E(λ))."""
    lam = 0.25
    psi = float(-np.log(1.0 - lam) - lam)
    expected = float(np.exp(lam * 1.0 - psi * 1.0))
    path = e_process_loss_diff(d=np.array([1.0]), lam=lam, initial_bound=1.0)
    assert np.isclose(path[0], expected)
