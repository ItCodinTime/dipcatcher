"""FDR/FWER corrections: ordering, FWER size simulation, fail-closed edges."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.validation.fdr import (
    adjust,
    benjamini_hochberg,
    benjamini_yekutieli,
    bonferroni,
    hochberg,
    holm,
    sidak,
    simes,
    storey_pi0,
    storey_qvalues,
)

P = np.array([0.001, 0.01, 0.02, 0.03, 0.04, 0.5, 0.9])


def test_bonferroni_and_sidak_values() -> None:
    assert bonferroni(P)["adjusted"][0] == pytest.approx(0.007)
    assert sidak(P)["adjusted"][0] == pytest.approx(1 - (1 - 0.001) ** 7)


def test_holm_stepdown_monotone() -> None:
    out = holm(P)
    adj_sorted = np.sort(out["adjusted"])
    assert np.all(np.diff(adj_sorted) >= -1e-12)
    assert out["adjusted"][0] == pytest.approx(0.007)
    assert out["adjusted"][1] == pytest.approx(0.06)


def test_hochberg_dominates_holm() -> None:
    assert np.all(hochberg(P)["adjusted"] <= holm(P)["adjusted"] + 1e-12)


def test_bh_known_values() -> None:
    out = benjamini_hochberg(P)
    # BH adjusted for p_(i): m*p_(i)/i with backward monotone enforcement.
    assert out["adjusted"][0] == pytest.approx(7 * 0.001 / 1)
    assert out["adjusted"][4] == pytest.approx(7 * 0.04 / 5, rel=1e-9)
    assert out["rejected"].sum() == 3  # p=0.03 adjusts to 0.0525 > 0.05


def test_by_more_conservative_than_bh() -> None:
    assert np.all(benjamini_yekutieli(P)["adjusted"] >= benjamini_hochberg(P)["adjusted"] - 1e-12)


def test_simes_global_stat() -> None:
    assert simes(P) == pytest.approx(7 * 0.001)
    assert simes(np.full(10, 0.9)) == pytest.approx(0.9)


def test_storey_pi0_and_qvalues() -> None:
    p = np.concatenate([np.full(50, 0.001), np.random.default_rng(0).uniform(0.4, 0.99, 950)])
    pi0 = storey_pi0(p)
    assert 0.8 <= pi0 <= 1.0
    q = storey_qvalues(p)
    assert q["qvalues"].shape == p.shape
    assert q["qvalues"][0] < 0.05


def test_adjust_dispatcher() -> None:
    for m in ("bonferroni", "sidak", "holm", "hochberg", "bh", "by", "storey", "simes"):
        out = adjust(P, m)
        assert "adjusted" in out or "qvalues" in out
    with pytest.raises(ValueError):
        adjust(P, "bogus")


def test_fwer_simulation_global_null() -> None:
    # Under the global null Bonferroni/Holm reject at ~alpha (conservative).
    rng = np.random.default_rng(0)
    rej_b, rej_h = 0, 0
    trials = 400
    for _ in range(trials):
        pv = rng.uniform(size=20)
        rej_b += bonferroni(pv, 0.05)["rejected"].any()
        rej_h += holm(pv, 0.05)["rejected"].any()
    assert rej_b / trials <= 0.08
    assert rej_h / trials <= 0.08


def test_bh_monotone_and_bounded() -> None:
    rng = np.random.default_rng(1)
    pv = rng.uniform(size=50)
    adj = benjamini_hochberg(pv)["adjusted"]
    assert np.all((adj >= pv - 1e-12) | (adj == 1.0))
    order = np.argsort(pv)
    assert np.all(np.diff(adj[order]) >= -1e-12)  # monotone in sorted p


def test_fail_closed_edges() -> None:
    with pytest.raises(ValueError):
        benjamini_hochberg(np.array([]))
    with pytest.raises(ValueError):
        holm(np.array([0.1, np.nan, 0.2]))
    with pytest.raises(ValueError):
        sidak(np.array([1.5]))
    with pytest.raises(ValueError):
        storey_pi0(P, lambdas=np.array([1.5]))
