"""Tests for models/archimedean_extra.py — Frank and Joe copulas."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.archimedean_extra import (
    frank_cdf,
    frank_fit,
    frank_sim,
    frank_tau,
    joe_cdf,
    joe_fit,
    joe_pdf,
    joe_sim,
)


def test_frank_tau_monotone_and_signed() -> None:
    taus = [frank_tau(th) for th in (-8.0, -3.0, 0.0, 3.0, 8.0)]
    assert all(b > a for a, b in zip(taus, taus[1:], strict=False))
    assert frank_tau(6.0) > 0 and frank_tau(-6.0) < 0


def test_frank_cdf_margins() -> None:
    u = np.linspace(0.05, 0.95, 10)
    assert np.allclose(frank_cdf(u, np.ones_like(u), 5.0), u, atol=1e-9)


def test_frank_fit_recovers_theta() -> None:
    rng = np.random.default_rng(0)
    uv = frank_sim(5.0, 4000, rng=rng)
    out = frank_fit(uv)
    assert abs(out["theta"] - 5.0) < 1.5
    assert out["theta"] > 0


def test_joe_cdf_margin_and_pdf_positive() -> None:
    u = np.linspace(0.05, 0.95, 10)
    assert np.allclose(joe_cdf(u, np.ones_like(u), 3.0), u, atol=1e-9)
    assert (joe_pdf(u, u, 3.0) > 0).all()


def test_joe_fit_recovers_theta() -> None:
    rng = np.random.default_rng(1)
    uv = joe_sim(3.0, 4000, rng=rng)
    out = joe_fit(uv)
    assert abs(out["theta"] - 3.0) < 1.0
    assert out["theta"] >= 1.0


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        joe_cdf(np.array([0.5]), np.array([0.5]), 0.5)
    with pytest.raises(ValueError):
        frank_fit(np.ones((5, 2)))  # too few rows
