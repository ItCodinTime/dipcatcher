"""Implementation-shortfall identity: the six components sum to the book gap."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_fund.parity.shortfall import attribute_shortfall

MARK = 110.0
STAMP = datetime(2024, 1, 2, tzinfo=UTC)


def _fill(
    *,
    qty: float,
    price: float,
    fee: float = 0.0,
    spread: float = 0.0,
    impact: float = 0.0,
    sid: str = "A",
    when: datetime = STAMP,
) -> dict:
    explicit = fee + spread + impact
    return {
        "event_time": when,
        "security_id": sid,
        "signed_qty": qty,
        "price": price,
        "fee": fee,
        "spread": spread,
        "impact": impact,
        "explicit_cost": explicit,
        "decision_price": 100.0,
    }


def _gap(report: dict) -> None:
    assert report["algebraic_residual"] == pytest.approx(0.0, abs=1e-8)
    assert report["component_sum"] == pytest.approx(report["fill_gap"], abs=1e-8)
    assert report["live_pnl_claim"] is False
    assert report["research_only"] is True
    assert report["paper_arrival_shortfall"]["in_terminal_gap"] is False


def test_delay_on_shared_quantity() -> None:
    report = attribute_shortfall(
        [_fill(qty=10, price=100, fee=1, spread=2, impact=3)],
        [_fill(qty=10, price=101, fee=1, spread=2, impact=3)],
        {"A": MARK},
    )
    _gap(report)
    assert report["components"]["delay"] == pytest.approx(10.0)
    assert report["components"]["spread"] == pytest.approx(0.0)
    assert report["components"]["fees"] == pytest.approx(0.0)
    assert report["components"]["missed_fills"] == pytest.approx(0.0)
    assert report["components"]["opportunity"] == pytest.approx(0.0)
    assert report["fill_gap"] == pytest.approx(10.0)


def test_missed_fill_and_its_costs() -> None:
    report = attribute_shortfall(
        [_fill(qty=10, price=100, fee=1, spread=2, impact=3)],
        [],
        {"A": MARK},
    )
    _gap(report)
    assert report["components"]["missed_fills"] == pytest.approx(100.0)
    assert report["components"]["fees"] == pytest.approx(-1.0)
    assert report["components"]["spread"] == pytest.approx(-2.0)
    assert report["components"]["impact"] == pytest.approx(-3.0)
    assert report["fill_gap"] == pytest.approx(94.0)


def test_opportunity_on_paper_only_quantity() -> None:
    report = attribute_shortfall(
        [],
        [_fill(qty=4, price=105)],
        {"A": MARK},
    )
    _gap(report)
    assert report["components"]["opportunity"] == pytest.approx(-20.0)
    assert report["fill_gap"] == pytest.approx(-20.0)
    assert report["opportunity_detail"]["unmatched_shadow_quantity"] == pytest.approx(-20.0)


def test_shared_terminal_mark_gap_sits_in_opportunity() -> None:
    report = attribute_shortfall(
        [_fill(qty=10, price=100)],
        [_fill(qty=10, price=100)],
        {"A": (110.0, 112.0)},
    )
    _gap(report)
    assert report["components"]["opportunity"] == pytest.approx(-20.0)
    assert report["components"]["delay"] == pytest.approx(0.0)
    assert report["opportunity_detail"]["shared_mark_gap"] == pytest.approx(-20.0)
    assert report["fill_gap"] == pytest.approx(-20.0)


def test_round_trip_delay_is_not_netted_away() -> None:
    later = datetime(2024, 1, 3, tzinfo=UTC)
    backtest = [
        _fill(qty=10, price=100),
        _fill(qty=-10, price=120, when=later),
    ]
    paper = [
        _fill(qty=10, price=100),
        _fill(qty=-10, price=121, when=later),
    ]
    report = attribute_shortfall(backtest, paper, {"A": MARK})
    _gap(report)
    assert report["components"]["delay"] == pytest.approx(-10.0)
    assert report["fill_gap"] == pytest.approx(-10.0)


@given(
    bt=st.lists(
        st.fixed_dictionaries(
            {
                "event_time": st.just(STAMP),
                "security_id": st.sampled_from(["A", "B"]),
                "signed_qty": st.one_of(
                    st.just(0.0),
                    st.floats(
                        min_value=-40,
                        max_value=-0.05,
                        allow_nan=False,
                        allow_infinity=False,
                    ),
                    st.floats(
                        min_value=0.05,
                        max_value=40,
                        allow_nan=False,
                        allow_infinity=False,
                    ),
                ),
                "price": st.floats(
                    min_value=5, max_value=200, allow_nan=False, allow_infinity=False
                ),
                "fee": st.floats(min_value=0, max_value=5, allow_nan=False, allow_infinity=False),
                "spread": st.floats(
                    min_value=0, max_value=5, allow_nan=False, allow_infinity=False
                ),
                "impact": st.floats(
                    min_value=0, max_value=5, allow_nan=False, allow_infinity=False
                ),
                "decision_price": st.floats(
                    min_value=5, max_value=200, allow_nan=False, allow_infinity=False
                ),
            }
        ),
        max_size=6,
    ),
    sh=st.lists(
        st.fixed_dictionaries(
            {
                "event_time": st.just(STAMP),
                "security_id": st.sampled_from(["A", "B"]),
                "signed_qty": st.one_of(
                    st.just(0.0),
                    st.floats(
                        min_value=-40,
                        max_value=-0.05,
                        allow_nan=False,
                        allow_infinity=False,
                    ),
                    st.floats(
                        min_value=0.05,
                        max_value=40,
                        allow_nan=False,
                        allow_infinity=False,
                    ),
                ),
                "price": st.floats(
                    min_value=5, max_value=200, allow_nan=False, allow_infinity=False
                ),
                "fee": st.floats(min_value=0, max_value=5, allow_nan=False, allow_infinity=False),
                "spread": st.floats(
                    min_value=0, max_value=5, allow_nan=False, allow_infinity=False
                ),
                "impact": st.floats(
                    min_value=0, max_value=5, allow_nan=False, allow_infinity=False
                ),
                "decision_price": st.floats(
                    min_value=5, max_value=200, allow_nan=False, allow_infinity=False
                ),
            }
        ),
        max_size=6,
    ),
)
@settings(max_examples=40, deadline=None)
def test_random_fills_keep_the_identity(bt: list[dict], sh: list[dict]) -> None:
    for row in (*bt, *sh):
        row["explicit_cost"] = row["fee"] + row["spread"] + row["impact"]
    report = attribute_shortfall(bt, sh, {"A": 80.0, "B": 40.0})
    assert report["algebraic_residual"] == pytest.approx(0.0, abs=1e-6)
    assert report["component_sum"] == pytest.approx(report["fill_gap"], abs=1e-6)
