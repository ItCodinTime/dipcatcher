"""Tests for models/empirical_copula.py — Deheuvels empirical copula."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.empirical_copula import (
    empirical_copula,
    pseudo_observations,
    spearman_rho,
    tail_dependence,
)


def test_pseudo_observations_uniform() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((500, 2))
    u = pseudo_observations(x)
    assert u.shape == (500, 2)
    assert (u > 0).all() and (u < 1).all()
    assert abs(float(u.mean()) - 0.5) < 0.02


def test_independence_and_comonotone_copula() -> None:
    rng = np.random.default_rng(1)
    indep = np.column_stack([rng.standard_normal(4000), rng.standard_normal(4000)])
    u_ind = pseudo_observations(indep)
    c_ind = float(empirical_copula(u_ind, np.array([[0.5, 0.5]]))[0])
    assert abs(c_ind - 0.25) < 0.03  # product copula

    z = rng.standard_normal(4000)
    como = np.column_stack([z, z])  # perfectly comonotone
    u_c = pseudo_observations(como)
    c_com = float(empirical_copula(u_c, np.array([[0.5, 0.5]]))[0])
    assert abs(c_com - 0.5) < 0.02  # min copula


def test_spearman_rho_bounds() -> None:
    rng = np.random.default_rng(2)
    z = rng.standard_normal(3000)
    assert spearman_rho(z, z) > 0.99
    assert abs(spearman_rho(z, rng.standard_normal(3000))) < 0.06


def test_tail_dependence_comonotone() -> None:
    rng = np.random.default_rng(3)
    z = rng.standard_normal(5000)
    td = tail_dependence(z, z, q=0.95)
    assert td["upper"] > 0.9 and td["lower"] > 0.9


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        pseudo_observations(np.ones((3, 2)))  # too few rows
    with pytest.raises(ValueError):
        tail_dependence(
            np.random.default_rng(0).standard_normal(100),
            np.random.default_rng(1).standard_normal(100),
            q=0.4,
        )
