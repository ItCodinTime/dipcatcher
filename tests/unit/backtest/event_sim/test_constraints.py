"""backtest/event_sim/constraints: T+1 settlement, PDT, GFV, tax-lot,
and buying-power branch coverage on pure in-memory state."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from quant_fund.backtest.event_sim.constraints import (
    ConstraintBook,
    TaxLot,
    below_min_notional,
    round_shares,
    session_date,
)


class TestHelpers:
    def test_session_date(self) -> None:
        assert session_date(datetime(2024, 3, 4, 15, 30)) == date(2024, 3, 4)

    def test_round_shares(self) -> None:
        assert round_shares(3.7, fractional=True) == 3.7
        assert round_shares(3.7, fractional=False) == 3.0
        assert round_shares(-2.9, fractional=False) == -2.0
        with pytest.raises(ValueError, match="finite"):
            round_shares(float("nan"), fractional=True)
        with pytest.raises(ValueError, match="finite"):
            round_shares(float("inf"), fractional=False)

    def test_below_min_notional(self) -> None:
        assert below_min_notional(1.0, 10.0, 100.0)
        assert not below_min_notional(20.0, 10.0, 100.0)
        assert below_min_notional(-0.5, 10.0, 100.0)  # abs() on quantity
        with pytest.raises(ValueError, match="non-negative"):
            below_min_notional(1.0, 10.0, -1.0)
        with pytest.raises(ValueError, match="positive"):
            below_min_notional(1.0, 0.0, 100.0)
        with pytest.raises(ValueError, match="positive"):
            below_min_notional(1.0, float("nan"), 100.0)


def _book(**kw: object) -> ConstraintBook:
    base: dict[str, object] = {"settlement_bars": 1, "allow_margin": False}
    base.update(kw)
    return ConstraintBook(**base)  # type: ignore[arg-type]


class TestInit:
    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="settlement_bars"):
            ConstraintBook(settlement_bars=-1, allow_margin=False)
        with pytest.raises(ValueError, match="pdt_mode"):
            ConstraintBook(settlement_bars=1, allow_margin=False, pdt_mode="yolo")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="gfv_mode"):
            ConstraintBook(settlement_bars=1, allow_margin=False, gfv_mode="yolo")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="PDT window"):
            ConstraintBook(settlement_bars=1, allow_margin=False, pdt_window_sessions=0)

    def test_cash_accounting(self) -> None:
        book = _book()
        book.settled = 100.0
        book.unsettled = [(5, 30.0), (6, 20.0)]
        assert book.economic_cash() == 150.0
        assert book.buying_power() == 150.0
        book.mature(5)
        assert book.settled == 130.0
        assert book.unsettled == [(6, 20.0)]
        book.mature(7)
        assert book.economic_cash() == 150.0
        assert book.unsettled == []


class TestPdt:
    def test_recent_day_trades_window(self) -> None:
        book = _book()
        sessions = [date(2024, 3, 1), date(2024, 3, 4), date(2024, 3, 5)]
        book.day_trades = [date(2024, 2, 1), date(2024, 3, 4), date(2024, 3, 5)]
        assert book.recent_day_trades(date(2024, 3, 5), sessions) == 2
        assert book.recent_day_trades(date(2024, 3, 5), []) == 0

    def test_would_block_modes(self) -> None:
        session = date(2024, 3, 4)
        # mode != block -> never blocks
        assert not _book(pdt_mode="warn").pdt_would_block("A", 10.0, -10.0, session, 1.0, [session])
        # nav above threshold -> free
        assert not _book(pdt_mode="block").pdt_would_block(
            "A", 10.0, -10.0, session, 1e9, [session]
        )
        # position not closing today -> no block
        assert not _book(pdt_mode="block").pdt_would_block("A", 10.0, -5.0, session, 1.0, [session])
        # close same-session-opened position past day-trade limit -> block
        book = _book(pdt_mode="block", pdt_max_day_trades=1)
        book.opened_session["A"] = session
        book.day_trades = [session]
        assert book.pdt_would_block("A", 10.0, -10.0, session, 1.0, [session])
        # opened a different session -> not a day trade
        book2 = _book(pdt_mode="block", pdt_max_day_trades=0)
        book2.opened_session["A"] = date(2024, 3, 1)
        assert not book2.pdt_would_block("A", 10.0, -10.0, session, 1.0, [session])

    def test_note_round_trip_day_trade(self) -> None:
        session = date(2024, 3, 4)
        book = _book(pdt_mode="warn", pdt_max_day_trades=0)
        book.opened_session["A"] = session
        book.note_round_trip("A", 10.0, 0.0, session, 1.0, [session])
        assert book.day_trades == [session]
        assert any(w["code"] == "PDT" for w in book.warnings)
        assert "A" not in book.opened_session  # flat -> entry popped
        # open -> records opened_session
        book.note_round_trip("B", 0.0, 5.0, session, 1.0, [session])
        assert book.opened_session["B"] == session
        # off mode: no bookkeeping
        book2 = _book(pdt_mode="off")
        book2.note_round_trip("A", 10.0, 0.0, session, 1.0, [session])
        assert book2.day_trades == []


class TestGfv:
    def test_would_block(self) -> None:
        book = _book(gfv_mode="block")
        book.lots["A"] = [TaxLot(10.0, settles_bar=5, unsettled_funded=True)]
        # selling before settles_bar -> blocked
        assert book.gfv_would_block("A", 10.0, -10.0, bar_index=3)
        # after settles -> free
        assert not book.gfv_would_block("A", 10.0, -10.0, bar_index=6)
        # buy-side delta -> free
        assert not book.gfv_would_block("A", 10.0, 5.0, bar_index=3)
        # mode off -> free
        book.gfv_mode = "off"
        assert not book.gfv_would_block("A", 10.0, -10.0, bar_index=3)
        # funded lots (settled cash) -> free
        book2 = _book(gfv_mode="block")
        book2.lots["A"] = [TaxLot(10.0, settles_bar=5, unsettled_funded=False)]
        assert not book2.gfv_would_block("A", 10.0, -10.0, bar_index=3)

    def test_consume_lots_partial_and_warn(self) -> None:
        book = _book(gfv_mode="warn")
        book.lots["A"] = [
            TaxLot(5.0, settles_bar=9, unsettled_funded=True),
            TaxLot(5.0, settles_bar=9, unsettled_funded=False),
        ]
        book.consume_lots_for_sell("A", 7.0, bar_index=3)
        assert book.gfv_count == 1  # only the unsettled-funded lot warns
        remaining = book.lots["A"]
        assert len(remaining) == 1
        assert remaining[0].quantity == 3.0
        # consume the rest -> lot list dropped
        book.consume_lots_for_sell("A", 3.0, bar_index=9)
        assert "A" not in book.lots
        # no-op
        book.consume_lots_for_sell("B", 0.0, bar_index=0)
        assert "B" not in book.lots


class TestBuySell:
    def test_buy_no_margin_overdraw_rejected(self) -> None:
        book = _book()
        book.settled = 50.0
        assert not book.apply_buy("A", 1.0, 100.0, bar_index=0)
        assert book.blocked == 1
        assert book.settled == 50.0

    def test_buy_negative_need_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            _book().apply_buy("A", 1.0, -5.0, bar_index=0)

    def test_buy_from_settled_creates_settled_lot(self) -> None:
        book = _book()
        book.settled = 100.0
        assert book.apply_buy("A", 2.0, 60.0, bar_index=0)
        assert book.settled == 40.0
        lot = book.lots["A"][0]
        assert lot.quantity == 2.0
        assert not lot.unsettled_funded
        assert lot.settles_bar == 0

    def test_buy_from_unsettled_marks_lot(self) -> None:
        book = _book()
        book.settled = 30.0
        book.unsettled = [(8, 50.0)]
        assert book.apply_buy("A", 1.0, 60.0, bar_index=0)
        assert book.settled == 0.0
        assert book.unsettled == [(8, 20.0)]
        lot = book.lots["A"][0]
        assert lot.unsettled_funded
        assert lot.settles_bar == 8

    def test_buy_margin_allows_overdraw(self) -> None:
        book = _book(allow_margin=True)
        book.settled = 10.0
        assert book.apply_buy("A", 1.0, 100.0, bar_index=0)
        assert book.settled == -90.0

    def test_buy_zero_quantity_no_lot(self) -> None:
        book = _book()
        book.settled = 100.0
        assert book.apply_buy("A", 0.0, 50.0, bar_index=0)
        assert "A" not in book.lots

    def test_sell_proceeds_park_unsettled(self) -> None:
        book = _book(settlement_bars=2)
        book.apply_sell_proceeds(100.0, bar_index=3)
        assert book.unsettled == [(5, 100.0)]
        book.settled = 50.0
        book.apply_sell_proceeds(-10.0, bar_index=3)
        assert book.settled == 40.0  # costs hit settled cash

    def test_sell_negative_proceeds_no_margin_raises(self) -> None:
        book = _book()
        book.settled = 5.0
        with pytest.raises(ValueError, match="negative without margin"):
            book.apply_sell_proceeds(-20.0, bar_index=0)
        assert book.settled == 5.0  # rolled back
