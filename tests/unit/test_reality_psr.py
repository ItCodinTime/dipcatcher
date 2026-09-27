"""Unit pins for the unit-safe PSR / MinTRL API (PROOFCORE W4 §7.1, §12)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from quant_fund.reality.psr import min_trl_from_returns, psr_from_returns


def _closed_form_psr(r: np.ndarray, sr_star: float = 0.0) -> float:
    """Independent recomputation of the Bailey–LdP PSR from raw returns."""
    mu = float(np.mean(r))
    sig = float(np.std(r, ddof=1))
    z = (r - mu) / sig
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4))
    sr = mu / sig
    se = np.sqrt(max(1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2, 1e-18))
    return float(norm.cdf((sr - sr_star) * np.sqrt(r.size - 1) / se))


def test_psr_gaussian_closed_form_pin() -> None:
    """Known closed-form Gaussian-ish case pinned to 6 decimals."""
    rng = np.random.default_rng(42)
    r = rng.normal(0.0008, 0.01, size=500)
    got = psr_from_returns(r, sr_star=0.0, periods_per_year=252.0)
    expected = _closed_form_psr(r)
    assert got["psr"] == pytest.approx(expected, abs=1e-12)
    assert got["sr_periodic"] == pytest.approx(float(np.mean(r) / np.std(r, ddof=1)), rel=1e-12)
    assert got["sr_annualized"] == pytest.approx(got["sr_periodic"] * np.sqrt(252.0), rel=1e-12)
    assert got["n_obs"] == 500.0
    # hard pin of the deterministic value itself (6 decimals)
    assert got["psr"] == pytest.approx(0.940553, abs=5e-7)


def test_psr_zero_edge_iid_book_is_about_half() -> None:
    """F1 regression at API level: zero-edge iid daily book, n=252 -> ~0.5, NOT ~1.0."""
    rng = np.random.default_rng(7)
    r = rng.normal(0.0, 0.01, size=252)
    r = r - float(np.mean(r))  # exactly zero sample mean -> SR == 0
    got = psr_from_returns(r, sr_star=0.0, periods_per_year=252.0)
    assert got["psr"] == pytest.approx(0.5, abs=1e-9)


def test_psr_invariant_to_return_scaling_units() -> None:
    """Unit safety: rescaling returns (bps vs decimal) must not move the PSR."""
    rng = np.random.default_rng(123)
    r = rng.normal(0.0005, 0.008, size=400)
    a = psr_from_returns(r, sr_star=0.0, periods_per_year=252.0)
    b = psr_from_returns(r * 1e4, sr_star=0.0, periods_per_year=252.0)
    assert a["psr"] == pytest.approx(b["psr"], abs=1e-12)
    assert a["sr_periodic"] == pytest.approx(b["sr_periodic"], abs=1e-12)


def test_psr_fail_closed_on_short_and_degenerate() -> None:
    out = psr_from_returns(np.array([0.01, -0.01]), periods_per_year=252.0)
    assert np.isnan(out["psr"]) and out["n_obs"] == 2.0
    assert np.isnan(psr_from_returns(np.full(100, 0.001), periods_per_year=252.0)["psr"])
    poisoned = np.concatenate([np.linspace(-0.01, 0.01, 50), [np.nan, np.inf]])
    got = psr_from_returns(poisoned, periods_per_year=252.0)
    assert got["n_obs"] == 50.0
    assert np.isfinite(got["psr"])


def test_psr_rejects_bad_periods_per_year() -> None:
    with pytest.raises(ValueError, match="periods_per_year"):
        psr_from_returns(np.linspace(-0.01, 0.01, 10), periods_per_year=0.0)


def test_min_trl_positive_sr_positive_periods() -> None:
    rng = np.random.default_rng(11)
    r = rng.normal(0.002, 0.01, size=600)
    mtrl = min_trl_from_returns(r, sr_star=0.0, periods_per_year=252.0)
    assert np.isfinite(mtrl) and mtrl > 1.0
    # Units: a per-period convention means a strong daily book needs on the
    # order of tens-to-hundreds of PERIODS, not hundreds of years.
    assert mtrl < 100_000.0


def test_min_trl_fail_closed_when_sr_not_above_star() -> None:
    rng = np.random.default_rng(13)
    r = rng.normal(-0.001, 0.01, size=300)
    assert np.isnan(min_trl_from_returns(r, sr_star=0.0, periods_per_year=252.0))
    assert np.isnan(min_trl_from_returns(np.array([0.01]), periods_per_year=252.0))
