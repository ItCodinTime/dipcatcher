"""Return accounting and causal position sizing."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from quant_fund.metrics.returns import (
    annualized_vol,
    cagr,
    downside_deviation,
    max_drawdown,
    sharpe_ratio,
    turnover,
    wealth_index,
)
from quant_fund.portfolio.allocators import inverse_volatility, kelly_weights, volatility_target
from quant_fund.risk.gates import (
    GateSpec,
    apply_gate_stack,
    kelly_leverage,
    vol_target,
    vol_target_leverage,
)
from tests.property._profiles import adversarial_settings

_ret = st.floats(min_value=-0.4, max_value=0.4, allow_nan=False, allow_infinity=False)


@given(rs=st.lists(_ret, min_size=1, max_size=30))
@adversarial_settings()
def test_wealth_is_the_compound_product(rs: list[float]) -> None:
    r = np.asarray(rs, dtype=float)
    wealth = wealth_index(r)
    assert wealth[-1] == pytest.approx(float(np.prod(1.0 + r)))
    if np.all(r >= 0.0):
        assert max_drawdown(r) == pytest.approx(0.0)


@given(rs=st.lists(_ret, min_size=4, max_size=40), scale=st.floats(0.2, 5.0))
@adversarial_settings()
def test_sharpe_is_invariant_to_positive_scale(rs: list[float], scale: float) -> None:
    r = np.asarray(rs, dtype=float)
    assume(float(np.std(r, ddof=1)) > 1e-6)
    base = float(sharpe_ratio(r)["sharpe"])
    scaled = float(sharpe_ratio(scale * r)["sharpe"])
    flipped = float(sharpe_ratio(-r)["sharpe"])
    assert scaled == pytest.approx(base, rel=1e-9, abs=1e-9)
    assert flipped == pytest.approx(-base, rel=1e-9, abs=1e-9)


@given(rs=st.lists(_ret, min_size=2, max_size=20))
@adversarial_settings()
def test_nonfinite_returns_do_not_become_a_number(rs: list[float]) -> None:
    r = np.asarray(rs, dtype=float)
    r[0] = np.nan
    assert np.isnan(sharpe_ratio(r)["sharpe"])
    assert np.isnan(annualized_vol(r))
    assert np.isnan(max_drawdown(r))


@given(n=st.integers(min_value=2, max_value=30))
@adversarial_settings()
def test_flat_returns_have_zero_cagr_and_zero_downside(n: int) -> None:
    flat = np.zeros(n)
    assert cagr(flat) == pytest.approx(0.0)
    assert downside_deviation(flat) == pytest.approx(0.0)
    gains = np.full(n, 0.01)
    assert downside_deviation(gains) == pytest.approx(0.0)


@given(
    a=st.lists(_ret, min_size=2, max_size=6),
    b=st.lists(_ret, min_size=2, max_size=6),
    c=st.lists(_ret, min_size=2, max_size=6),
)
@adversarial_settings()
def test_turnover_is_an_l1_metric(a: list[float], b: list[float], c: list[float]) -> None:
    n = min(len(a), len(b), len(c))
    wa, wb, wc = (np.asarray(x[:n], dtype=float) for x in (a, b, c))
    assert turnover(wa, wa) == pytest.approx(0.0)
    assert turnover(wa, wb) == pytest.approx(turnover(wb, wa))
    assert turnover(wa, wc) <= turnover(wa, wb) + turnover(wb, wc) + 1e-9


@given(
    variances=st.lists(
        st.floats(min_value=0.01, max_value=4.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=5,
        unique=True,
    )
)
@adversarial_settings()
def test_inverse_volatility_sums_to_one_and_is_permutation_equivariant(
    variances: list[float],
) -> None:
    cov = np.diag(np.asarray(variances, dtype=float))
    weights = inverse_volatility(cov)
    assert weights.sum() == pytest.approx(1.0)
    assert np.all(weights > 0.0)
    order = np.arange(len(variances))[::-1]
    flipped = inverse_volatility(cov[np.ix_(order, order)])
    assert flipped == pytest.approx(weights[order])


@given(
    mu=st.lists(
        st.floats(min_value=0.0, max_value=0.05, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=4,
    ),
    fraction=st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@adversarial_settings()
def test_diagonal_kelly_is_fraction_times_mu_over_variance(
    mu: list[float],
    fraction: float,
) -> None:
    means = np.asarray(mu, dtype=float)
    cov = np.eye(means.size)
    weights = kelly_weights(means, cov, fraction=fraction, long_only=False)
    assert weights == pytest.approx(fraction * means)


@given(
    target=st.floats(min_value=0.05, max_value=0.4, allow_nan=False, allow_infinity=False),
    cap=st.floats(min_value=0.5, max_value=4.0, allow_nan=False, allow_infinity=False),
)
@adversarial_settings()
def test_volatility_target_hits_target_or_the_cap(target: float, cap: float) -> None:
    weights = np.array([0.5, 0.5])
    cov = np.diag([0.04, 0.09])
    scaled, leverage = volatility_target(weights, cov, target, max_leverage=cap)
    raw = float(np.sqrt(weights @ cov @ weights))
    expected = min(target / raw, cap)
    assert leverage == pytest.approx(expected)
    assert float(np.sqrt(scaled @ cov @ scaled)) == pytest.approx(raw * expected)


_spec = GateSpec(
    vol_target=0.10,
    vol_lookback=8,
    vol_cap=3.0,
    kelly_fraction=0.25,
    kelly_lookback=8,
    crc_alpha=None,
    es_limit=None,
    crash_lookback=5,
    crash_return=-0.15,
    dd_limit=0.25,
    dd_mode="halt",
    stepm_enable=False,
)


@given(
    rs=st.lists(
        st.floats(min_value=-0.05, max_value=0.05, allow_nan=False, allow_infinity=False),
        min_size=24,
        max_size=36,
    ),
    cut=st.integers(min_value=8, max_value=20),
)
@adversarial_settings()
def test_gate_stack_and_vol_target_ignore_the_future(rs: list[float], cut: int) -> None:
    r = np.asarray(rs, dtype=float)
    assume(cut < r.size - 1)
    future = r.copy()
    future[cut + 1 :] = 0.04
    original = apply_gate_stack(r, _spec)
    shifted = apply_gate_stack(future, _spec)
    assert original.returns[: cut + 1] == pytest.approx(shifted.returns[: cut + 1])
    vt = vol_target(r, target=0.02, lookback=8)
    vt2 = vol_target(future, target=0.02, lookback=8)
    assert vt[: cut + 1] == pytest.approx(vt2[: cut + 1])
    assert np.all(vt[:9] == 0.0)
    lev = vol_target_leverage(r, target=0.02, lookback=8)
    lev2 = vol_target_leverage(future, target=0.02, lookback=8)
    assert lev[: cut + 1] == pytest.approx(lev2[: cut + 1])
    kelly = kelly_leverage(r, fraction=0.25, lookback=8)
    assert np.all(kelly >= -1e-12)
