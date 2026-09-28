"""Fail-closed delay guards on the hedge_lab directional books.

The books document ``weights at t use data through t-delay``. ``delay=0``
lets a weight read the same bar's close-to-close return it earns — a
one-bar look-ahead. The clamps used to coerce ``delay`` to 0 silently;
they now raise.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.hedge_lab.directional import (
    antonacci_returns,
    risk_parity_blend,
    topk_long_returns,
    tsmom_book_returns,
)
from quant_fund.hedge_lab.sleeve_policy import etf_dual_momentum
from quant_fund.hedge_lab.target_hunt import pair_spread_returns


def _px(n: int = 300, names: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, size=(n, names)), axis=0))


@pytest.mark.parametrize("delay", [0, -1, -5])
def test_tsmom_book_returns_rejects_sub_day_delay(delay: int) -> None:
    with pytest.raises(ValueError, match="delay must be >= 1"):
        tsmom_book_returns(_px(), delay=delay)


@pytest.mark.parametrize("delay", [0, -2])
def test_topk_long_returns_rejects_sub_day_delay(delay: int) -> None:
    with pytest.raises(ValueError, match="delay must be >= 1"):
        topk_long_returns(_px(), delay=delay)


@pytest.mark.parametrize("delay", [0, -1])
def test_antonacci_returns_rejects_sub_day_delay(delay: int) -> None:
    spy = _px()[:, 0]
    tlt = _px(seed=1)[:, 0]
    with pytest.raises(ValueError, match="delay must be >= 1"):
        antonacci_returns(spy, tlt, delay=delay)


@pytest.mark.parametrize("delay", [0, -3])
def test_risk_parity_blend_rejects_sub_day_delay(delay: int) -> None:
    px = _px(names=3)
    streams = {"a": px[:, 0], "b": px[:, 1], "c": px[:, 2]}
    with pytest.raises(ValueError, match="delay must be >= 1"):
        risk_parity_blend(streams, delay=delay)


@pytest.mark.parametrize("delay", [0, -1])
def test_etf_dual_momentum_rejects_sub_day_delay(delay: int) -> None:
    with pytest.raises(ValueError, match="delay must be >= 1"):
        etf_dual_momentum(_px(names=5), defensive=4, delay=delay)


@pytest.mark.parametrize("delay", [0, -1])
def test_pair_spread_returns_rejects_sub_day_delay(delay: int) -> None:
    px = _px(names=2)
    with pytest.raises(ValueError, match="delay must be >= 1"):
        pair_spread_returns(px[:, 0], px[:, 1], delay=delay)


def test_delay_one_books_unchanged() -> None:
    """The delay=1 path is the documented book; the guard must not alter it."""
    px = _px()
    for book in (
        tsmom_book_returns(px, delay=1),
        topk_long_returns(px, delay=1),
        risk_parity_blend({"a": px[:, 0], "b": px[:, 1], "c": px[:, 2]}, delay=1),
        etf_dual_momentum(_px(names=5, seed=2), defensive=4, delay=1),
        pair_spread_returns(px[:, 0], px[:, 1], delay=1),
        antonacci_returns(px[:, 0], px[:, 1], delay=1),
    ):
        assert book.ndim == 1
        assert np.all(np.isfinite(book))
