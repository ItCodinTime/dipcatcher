"""Hypothesis properties for the allocation pack (SYNTHETIC only).

- Feasibility: post-constraint weights satisfy every declared bound on
  random well-conditioned inputs, for every engine.
- Determinism: identical inputs give bitwise-identical weights.
- No lookahead: permuting observations strictly after a decision index
  cannot change that decision's weights (structural causality).
- ERC parity: risk-parity risk-contribution shares are ~1/n on
  well-conditioned covariance.
- Turnover bound: per-rebalance gross weight change never exceeds
  2 * leverage_cap.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from quant_fund.research.allocation import (
    AllocationConstraints,
    fit_weights,
    risk_contributions,
    risk_parity_weights,
    run_walk_forward,
    weights_satisfy,
)
from quant_fund.research.allocation.engines import EngineName

Array = NDArray[np.float64]

_ENGINES: tuple[EngineName, ...] = ("inverse_volatility", "risk_parity", "kelly")


def _cov_from_seed(seed: int, n: int) -> Array:
    """Deterministic well-conditioned SPD covariance."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    return np.asarray(a @ a.T * 1e-3 + np.diag(rng.uniform(1e-3, 5e-3, n)), dtype=float)


def _returns_from_seed(seed: int, t: int, n: int) -> Array:
    rng = np.random.default_rng(seed)
    cov = _cov_from_seed(seed + 1, n)
    mu = rng.uniform(-0.001, 0.001, n)
    return np.asarray(rng.multivariate_normal(mu, cov, t), dtype=float)


@given(
    seed=st.integers(0, 10_000),
    n=st.integers(2, 6),
    engine=st.sampled_from(_ENGINES),
    cap=st.floats(0.5, 3.0, allow_nan=False, allow_infinity=False),
    max_w=st.floats(0.2, 1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=60, deadline=None)
def test_fit_weights_respects_constraints(
    seed: int, n: int, engine: EngineName, cap: float, max_w: float
) -> None:
    cons = AllocationConstraints(leverage_cap=cap, max_weight=max_w)
    returns = _returns_from_seed(seed, t=60, n=n)
    w = fit_weights(engine, returns, cons, kelly_fraction=0.5)
    assert weights_satisfy(w, cons) == []


@given(seed=st.integers(0, 10_000), n=st.integers(2, 6))
@settings(max_examples=40, deadline=None)
def test_fit_weights_deterministic(seed: int, n: int) -> None:
    returns = _returns_from_seed(seed, t=60, n=n)
    for engine in _ENGINES:
        a = fit_weights(engine, returns)
        b = fit_weights(engine, returns)
        assert np.array_equal(a, b)


@given(
    seed=st.integers(0, 10_000),
    n=st.integers(2, 5),
    split=st.integers(50, 90),
)
@settings(max_examples=30, deadline=None)
def test_no_lookahead_future_permutation_changes_nothing(seed: int, n: int, split: int) -> None:
    """Weights at decisions < split are bitwise identical when every row at
    or after ``split`` is permuted."""
    window = 40
    returns = _returns_from_seed(seed, t=100, n=n)
    permuted = returns.copy()
    rng = np.random.default_rng(seed + 99)
    order = rng.permutation(permuted.shape[0] - split)
    permuted[split:] = permuted[split:][order]

    ev_a = run_walk_forward(returns, "inverse_volatility", window=window, step=5)
    ev_b = run_walk_forward(permuted, "inverse_volatility", window=window, step=5)
    assert np.array_equal(ev_a.decision_index, ev_b.decision_index)
    early = ev_a.decision_index < split
    assert np.array_equal(ev_a.weights[early], ev_b.weights[early])


@given(seed=st.integers(0, 10_000), n=st.integers(2, 7))
@settings(max_examples=50, deadline=None)
def test_erc_residual_near_zero_well_conditioned(seed: int, n: int) -> None:
    cov = _cov_from_seed(seed, n)
    w = risk_parity_weights(cov, tol=1e-12)
    shares = risk_contributions(w, cov)
    shares = shares / shares.sum()
    assert float(np.abs(shares - 1.0 / n).max()) < 1e-6


@given(seed=st.integers(0, 10_000), n=st.integers(2, 5))
@settings(max_examples=40, deadline=None)
def test_turnover_bounded_by_twice_cap(seed: int, n: int) -> None:
    cap = 1.5
    returns = _returns_from_seed(seed, t=120, n=n)
    cons = AllocationConstraints(leverage_cap=cap)
    ev = run_walk_forward(returns, "kelly", window=30, step=10, constraints=cons)
    assert float(ev.turnover.max()) <= 2.0 * cap + 1e-9
    assert ev.constraint_violations == 0


@given(seed=st.integers(0, 10_000), n=st.integers(2, 5))
@settings(max_examples=40, deadline=None)
def test_weights_stay_finite_and_bounded(seed: int, n: int) -> None:
    returns = _returns_from_seed(seed, t=100, n=n)
    cons = AllocationConstraints(leverage_cap=2.0, max_weight=0.8)
    ev = run_walk_forward(
        returns, "vol_target", window=30, step=10, target_vol=0.05, constraints=cons
    )
    assert np.all(np.isfinite(ev.weights))
    assert np.abs(ev.weights).max() <= 0.8 * (1 + 1e-9)
    for w in ev.weights:
        assert float(np.abs(w).sum()) <= 2.0 + 1e-9
