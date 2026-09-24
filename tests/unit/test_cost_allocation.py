"""Engineering checks on generated inputs, not market performance evidence."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import cvxpy as cp
import numpy as np
import pytest

from quant_fund.research.cost_allocation import AllocationConfig, allocate
from quant_fund.research.net_replay import MarketPanel, ReplayConfig, Strategy, replay
from quant_fund.research.net_tournament import allocation_ablations


def solve(alpha=(0.01,), previous=None, covariance=None, config=None, **kwargs):
    a = np.array(alpha)
    n = len(a)
    fields = dict(
        uncertainty=np.zeros(n),
        previous=np.zeros(n) if previous is None else np.array(previous),
        lower=np.zeros(n),
        upper=np.full(n, 0.5),
        capacity=np.ones(n),
        impact=np.zeros(n),
        config=config or AllocationConfig(uncertainty_aversion=0, turnover_limit=1),
        linear_cost=0,
        gross_limit=0.95,
        name_limit=0.5,
        cash_buffer=0.01,
    )
    fields.update(kwargs)
    return allocate(a, np.eye(n) * 0.001 if covariance is None else covariance, **fields)


def test_analytic_risk_solution_and_risk_aversion():
    # argmax alpha*w - lambda*variance*w^2 = alpha/(2*lambda*variance).
    w, diagnostic = solve()
    assert w[0] == pytest.approx(0.25, abs=1e-6)
    higher, _ = solve(
        config=AllocationConfig(risk_aversion=40, uncertainty_aversion=0, turnover_limit=1)
    )
    assert higher[0] == pytest.approx(0.125, abs=1e-6)
    assert diagnostic["max_constraint_violation"] < 1e-7


def test_linear_cost_no_trade_region_from_cash_and_existing_holdings():
    w, _ = solve(alpha=(0.0001,), linear_cost=0.001)
    assert w[0] == 0
    # The marginal risk-adjusted alpha is smaller than the spread in either direction.
    w, _ = solve(alpha=(0.0041,), previous=(0.1,), linear_cost=0.001)
    assert w[0] == 0.1


def test_impact_and_uncertainty_reduce_exposure():
    plain, _ = solve()
    impacted, diag = solve(impact=np.array([0.01]))
    uncertain, _ = solve(
        uncertainty=np.array([0.005]),
        config=AllocationConfig(uncertainty_aversion=1, turnover_limit=1),
    )
    assert 0 < impacted[0] < plain[0]
    assert uncertain[0] == pytest.approx(0.125, abs=1e-6)
    assert diag["predicted_impact_cost"] == pytest.approx(0.01 * impacted[0] ** 1.5)


def test_liquidity_turnover_and_post_fee_exposure_constraints():
    w, diag = solve(
        alpha=(1.0, 1.0),
        capacity=np.array([0.02, 1.0]),
        config=AllocationConfig(turnover_limit=0.12),
        linear_cost=0.001,
    )
    assert w[0] <= 0.0200001
    assert abs(w).sum() <= 0.1200001
    w, diag = solve(alpha=(1.0, 1.0), linear_cost=0.02, gross_limit=0.7)
    fees = diag["predicted_linear_cost"]
    assert abs(w).sum() / (1 - fees) <= 0.7000001
    assert abs(w).max() / (1 - fees) <= 0.5000001


def test_short_borrow_and_cash_opportunity_cost():
    short, _ = solve(alpha=(-0.01,), lower=np.array([-0.5]), upper=np.zeros(1))
    borrowed, _ = solve(
        alpha=(-0.01,), lower=np.array([-0.5]), upper=np.zeros(1), borrow_cost=0.005
    )
    assert short[0] == pytest.approx(-0.25, abs=1e-6)
    assert borrowed[0] == pytest.approx(-0.125, abs=1e-6)
    cash, _ = solve(cash_return=0.005, funding_cost=0.005)
    assert cash[0] == pytest.approx(0.125, abs=1e-6)


def test_infeasible_and_failed_solver_never_fall_back(monkeypatch):
    with pytest.raises(ValueError, match="infeasible"):
        solve(previous=(0.8,), capacity=np.array([0.01]))

    def broken(*args, **kwargs):
        raise cp.error.SolverError("injected")

    monkeypatch.setattr(cp.Problem, "solve", broken)
    with pytest.raises(ValueError, match="solver failed"):
        solve()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"covariance": np.array([[-0.001]])},
        {"capacity": np.array([np.nan])},
        {"uncertainty": np.array([-0.1])},
        {"previous": (np.inf,)},
        {"cash_return": 0.1},
        {"covariance": np.eye(2)},
    ],
)
def test_invalid_inputs_rejected(kwargs):
    with pytest.raises(ValueError):
        solve(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"risk_aversion": 0},
        {"risk_window": True},
        {"turnover_limit": np.nan},
        {"covariance_shrinkage": 1.1},
        {"alpha_scale": -1},
    ],
)
def test_invalid_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        AllocationConfig(**kwargs).validate()


def market():
    rng = np.random.default_rng(7)
    close = 100 * np.exp(np.cumsum(rng.normal(0.002, 0.01, (90, 4)), axis=0))
    opening = np.vstack([close[0], close[:-1]])
    return MarketPanel(
        [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(90)],
        ["a", "b", "c", "d"],
        close,
        opening,
        np.full_like(close, 1e5),
        np.ones_like(close, dtype=bool),
    )


def play(panel, strategy, **kwargs):
    return replay(
        panel, strategy, ReplayConfig(), start="2020-03-01", end="2020-03-30", history=60, **kwargs
    )


def optimized():
    return Strategy(
        "optimized", "momentum", fraction=0.5, allocation=AllocationConfig(uncertainty_aversion=0)
    )


def test_causal_decision_and_stress_uses_frozen_planning_costs():
    p, s = market(), optimized()
    original = play(p, s)
    assert original["allocations"][0]["turnover"] > 0
    stressed = play(p, s, impact_multiplier=2)
    assert original["allocations"][0] == stressed["allocations"][0]
    p.close[61:] *= 1.1
    p.volume[61:] *= 10
    changed = play(p, s)
    assert original["allocations"][0] == changed["allocations"][0]
    first = p.dates[61].isoformat()
    assert [f for f in original["fills"] if f["execution_session"] == first] == [
        f for f in changed["fills"] if f["execution_session"] == first
    ]


def test_allocator_uses_filled_holdings_and_reports_matched_ablation():
    p, s = market(), optimized()
    # A large next-open gap forces a rejection/partial execution relative to the target.
    p.opening[61] *= 1.5
    result = play(p, s)
    assert result["rejected_orders"] > 0
    previous = result["allocations"][1]
    shares = dict.fromkeys(p.names, 0.0)
    first_date = p.dates[61].isoformat()
    for fill in result["fills"]:
        if fill["execution_session"] == first_date:
            shares[fill["security_id"]] += fill["quantity"]
    for name, weight in zip(previous["security_ids"], previous["previous_weights"], strict=True):
        assert weight == pytest.approx(
            shares[name] * p.close[61, p.names.index(name)] / previous["planning_nav"]
        )
    control = replace(s, name="ranked", allocation=None)
    baseline = play(p, control)
    report = allocation_ablations({"optimized": result, "ranked": baseline}, [s, control])
    assert len(report) == 1
    assert report[0]["total_return_difference"] == pytest.approx(
        result["total_return"] - baseline["total_return"]
    )
    assert report[0]["inference"] == "descriptive_only"
    assert allocation_ablations({"optimized": result}, [s]) == []


def test_cost_aware_flat_market_avoids_control_round_trip_cost():
    p = market()
    p.close[:] = p.opening[:] = 100
    s = optimized()
    candidate = play(p, s)
    control = play(p, replace(s, name="ranked", allocation=None))
    assert candidate["fills"] == []
    assert candidate["total_return"] == 0
    assert control["total_return"] < 0


def test_debt_funding_reduces_leveraged_allocation():
    fields = dict(
        alpha=(0.005, 0.005),
        covariance=np.eye(2) * 0.0001,
        upper=np.ones(2),
        name_limit=1,
        gross_limit=2,
        config=AllocationConfig(turnover_limit=2, uncertainty_aversion=0),
    )
    free, _ = solve(**fields)
    funded, _ = solve(**fields, funding_cost=0.004)
    assert free.sum() > 1.9
    assert funded.sum() == pytest.approx(1.0, abs=1e-5)


def test_long_short_adapter_and_unavailable_held_history():
    from quant_fund.research.net_replay import _cost_weights

    p = market()
    s = replace(optimized(), long_short=True)
    result = play(p, s)
    assert result["liquidation_complete"]
    assert any(w < 0 for d in result["allocations"] for w in d["target_weights"])
    p.known[50, 0] = False
    with pytest.raises(ValueError, match="known risk history"):
        _cost_weights(
            p,
            60,
            np.array([0.0, 0.1, 0.0, 0.0]),
            np.array([0.1, 0.0, 0.0, 0.0]),
            np.ones(4) * 1e6,
            np.ones(4) * 0.01,
            s,
            ReplayConfig(),
            1e5,
        )
