"""Z3 proofs of the accounting closed forms. Quantifier-free reals.

Each proof is the unsatisfiability of the negated claim under the
preconditions, so the claim holds for every real assignment of those
symbols. IEEE float is outside the proof; a separate check compares
``total_cost`` to the closed form on an interior point.
"""

from __future__ import annotations

import pytest
import z3

from quant_fund.backtest.fast_replay import _order_costs
from quant_fund.config.models import CostConfig
from quant_fund.execution.costs import total_cost
from quant_fund.formal.accounting import algebraic_total_cost
from quant_fund.metrics.returns import turnover


def prove(constraints: list[z3.BoolRef], claim: z3.BoolRef, *, timeout_ms: int = 20_000) -> None:
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    for item in constraints:
        solver.add(item)
    solver.add(z3.Not(claim))
    result = solver.check()
    if result != z3.unsat:
        model = solver.model() if result == z3.sat else None
        raise AssertionError(f"not proved ({result}): {model}")


def test_one_step_nav_identity_opening_buy() -> None:
    initial, q, price, mark, cost = z3.Reals("initial q price mark cost")
    cash = initial - q * price - cost
    realized = -cost
    unrealized = q * (mark - price)
    nav = cash + q * mark
    prove(
        [q > 0, price > 0, mark > 0, cost >= 0],
        nav == initial + realized + unrealized,
    )


def test_partial_close_preserves_pnl_identity() -> None:
    initial, shares, avg, realized, q, price, mark, cost = z3.Reals(
        "initial shares avg realized q price mark cost"
    )
    cash = initial + realized - shares * avg
    cash2 = cash + q * price - cost
    realized2 = realized - cost + q * (price - avg)
    shares2 = shares - q
    unrealized2 = shares2 * (mark - avg)
    nav2 = cash2 + shares2 * mark
    prove(
        [shares > 0, q > 0, q < shares, price > 0, mark > 0, cost >= 0],
        nav2 == initial + realized2 + unrealized2,
    )


def test_flatten_preserves_pnl_identity() -> None:
    initial, shares, avg, realized, price, cost = z3.Reals("initial shares avg realized price cost")
    cash = initial + realized - shares * avg
    cash2 = cash + shares * price - cost
    realized2 = realized - cost + shares * (price - avg)
    nav2 = cash2
    prove(
        [shares > 0, price > 0, cost >= 0],
        nav2 == initial + realized2,
    )


def test_flip_long_to_short_preserves_pnl_identity() -> None:
    initial, shares, avg, realized, q, price, mark, cost = z3.Reals(
        "initial shares avg realized q price mark cost"
    )
    cash = initial + realized - shares * avg
    cash2 = cash + q * price - cost
    realized2 = realized - cost + shares * (price - avg)
    shares2 = shares - q
    unrealized2 = shares2 * (mark - price)
    nav2 = cash2 + shares2 * mark
    prove(
        [shares > 0, q > shares, price > 0, mark > 0, cost >= 0],
        nav2 == initial + realized2 + unrealized2,
    )


def test_three_trade_unroll_preserves_identity() -> None:
    """Buy, add, partial sell. Symbolic sizes, fixed branch structure."""
    initial, q1, q2, q3, p1, p2, p3, c1, c2, c3, mark = z3.Reals(
        "initial q1 q2 q3 p1 p2 p3 c1 c2 c3 mark"
    )
    # Open.
    cash = initial - q1 * p1 - c1
    realized = -c1
    qty = q1
    avg = p1
    # Add. Average is the notional-weighted price.
    cash = cash - q2 * p2 - c2
    realized = realized - c2
    avg = (qty * avg + q2 * p2) / (qty + q2)
    qty = qty + q2
    # Partial sell of q3.
    cash = cash + q3 * p3 - c3
    realized = realized - c3 + q3 * (p3 - avg)
    qty = qty - q3
    nav = cash + qty * mark
    unrealized = qty * (mark - avg)
    prove(
        [
            q1 > 0,
            q2 > 0,
            q3 > 0,
            q3 < q1 + q2,
            p1 > 0,
            p2 > 0,
            p3 > 0,
            c1 >= 0,
            c2 >= 0,
            c3 >= 0,
            mark > 0,
        ],
        nav == initial + realized + unrealized,
    )


def _cost_symbols():
    names = "q1 q2 price adv sigma upsilon bps_c bps_s bps_t s1 s2"
    q1, q2, price, adv, sigma, upsilon, bps_c, bps_s, bps_t, s1, s2 = z3.Reals(names)
    pre = [
        q2 >= q1,
        q1 >= 0,
        price > 0,
        adv > 0,
        sigma >= 0,
        upsilon >= 0,
        bps_c >= 0,
        bps_s >= 0,
        bps_t >= 0,
        s1 >= 0,
        s2 >= 0,
        s1 * s1 * adv == q1 * price,
        s2 * s2 * adv == q2 * price,
    ]

    def total(q, s):
        notional = q * price
        return (
            notional * bps_c / 10000
            + notional * bps_s / 10000
            + notional * upsilon * sigma * s
            + notional * bps_t / 10000
        )

    return pre, total(q1, s1), total(q2, s2), (q1, price, adv, sigma, upsilon, bps_c, bps_s, bps_t)


def test_costs_are_nonnegative_and_monotone_in_quantity() -> None:
    pre, low, high, _symbols = _cost_symbols()
    prove(pre, z3.And(low >= 0, high >= low))


def test_costs_monotone_in_each_rate() -> None:
    q, price, adv, sigma, upsilon, bps, bps2, s = z3.Reals("q price adv sigma upsilon bps bps2 s")
    pre = [
        q >= 0,
        price > 0,
        adv > 0,
        sigma >= 0,
        upsilon >= 0,
        bps2 >= bps,
        bps >= 0,
        s >= 0,
        s * s * adv == q * price,
    ]
    notional = q * price
    base = notional * upsilon * sigma * s
    low = notional * bps / 10000 + base
    high = notional * bps2 / 10000 + base
    prove(pre, high >= low)


def test_split_and_dividend_conserve_market_value() -> None:
    shares, price, factor, cash, dividend = z3.Reals("shares price factor cash dividend")
    prove([factor > 0], shares * price == (shares * factor) * (price / factor))
    prove(
        [dividend >= 0],
        cash + shares * price == (cash + shares * dividend) + shares * (price - dividend),
    )
    shares2 = shares * factor
    price2 = price / factor - dividend
    cash2 = cash + shares2 * dividend
    prove(
        [factor > 0, dividend >= 0],
        cash2 + shares2 * price2 == cash + shares * price,
    )


def test_turnover_is_a_metric_on_three_names() -> None:
    def vec(prefix: str):
        return z3.Reals(f"{prefix}0 {prefix}1 {prefix}2")

    a = vec("a")
    b = vec("b")
    c = vec("c")

    def l1(left, right):
        return sum(
            z3.If(left[i] - right[i] >= 0, left[i] - right[i], right[i] - left[i]) for i in range(3)
        )

    prove([], l1(a, b) >= 0)
    prove([z3.And(*[a[i] == b[i] for i in range(3)])], l1(a, b) == 0)
    prove([], l1(a, b) == l1(b, a))
    prove([], l1(a, c) <= l1(a, b) + l1(b, c))


def test_closed_form_matches_total_cost_and_fast_replay() -> None:
    """Interior point of the proved precondition, not a substitute for it."""
    cfg = CostConfig(
        commission_bps=1.5,
        half_spread_bps=2.5,
        impact_y=0.2,
        bps_per_turnover=4.0,
        financing_bps_per_year=80.0,
        frictionless=False,
        participation_limit=1.0,
    )
    out = total_cost(4.0, 25.0, 100.0, 0.05, cfg)
    closed = algebraic_total_cost(
        4.0,
        25.0,
        100.0,
        0.05,
        commission_bps=1.5,
        half_spread_bps=2.5,
        impact_y=0.2,
        bps_per_turnover=4.0,
    )
    assert out["total"] == pytest.approx(closed["total"])
    assert out["commission"] == pytest.approx(closed["commission"])
    assert out["impact"] == pytest.approx(closed["impact"])
    assert float(out["total"]) >= 0.0
    _comm, _spr, _imp, fast_total = _order_costs(4.0, 25.0, 100.0, 0.05, cfg)
    assert fast_total == pytest.approx(out["total"])
    ignored = CostConfig(
        commission_bps=1.5,
        half_spread_bps=2.5,
        impact_y=0.2,
        bps_per_turnover=4.0,
        financing_bps_per_year=0.0,
        participation_limit=1.0,
    )
    assert total_cost(4.0, 25.0, 100.0, 0.05, ignored)["total"] == pytest.approx(out["total"])
    assert turnover([0.2, -0.1, 0.0], [0.0, -0.1, 0.4]) == pytest.approx(0.6)
