"""Avellaneda–Stoikov quote boundary: fail-closed validation on every input."""

from __future__ import annotations

import math

import pytest

from quant_fund.market_sim.quotes import avellaneda_stoikov_quotes

pytestmark = pytest.mark.synthetic


@pytest.mark.parametrize("mid_tick", [math.nan, math.inf, -math.inf])
def test_non_finite_mid_tick_rejected(mid_tick: float) -> None:
    with pytest.raises(ValueError, match="mid_tick"):
        avellaneda_stoikov_quotes(mid_tick, 0.0, 0.1, 1.5, 4.0)


@pytest.mark.parametrize("inventory", [math.nan, math.inf])
def test_non_finite_inventory_rejected(inventory: float) -> None:
    with pytest.raises(ValueError, match="inventory_units"):
        avellaneda_stoikov_quotes(100.0, inventory, 0.1, 1.5, 4.0)


@pytest.mark.parametrize("gamma", [0.0, -1.0, math.nan, math.inf])
def test_non_positive_gamma_rejected(gamma: float) -> None:
    with pytest.raises(ValueError, match="gamma"):
        avellaneda_stoikov_quotes(100.0, 0.0, gamma, 1.5, 4.0)


@pytest.mark.parametrize("k", [0.0, -0.5, math.nan])
def test_non_positive_intensity_rejected(k: float) -> None:
    with pytest.raises(ValueError, match="k must"):
        avellaneda_stoikov_quotes(100.0, 0.0, 0.1, k, 4.0)


@pytest.mark.parametrize("sigma2", [-1e-9, math.nan])
def test_negative_variance_rejected(sigma2: float) -> None:
    with pytest.raises(ValueError, match="sigma2"):
        avellaneda_stoikov_quotes(100.0, 0.0, 0.1, 1.5, sigma2)


def test_valid_quotes_straddle_reservation() -> None:
    bid, ask, half = avellaneda_stoikov_quotes(100.0, 2.0, 0.1, 1.5, 4.0)
    reservation = 100.0 - 2.0 * 0.1 * 4.0
    assert bid <= math.floor(reservation)
    assert ask >= math.ceil(reservation)
    assert ask > bid
    assert half >= 1.0
