"""Crypto (24/7 UTC) and FX (Sun 17:00 ET → Fri 17:00 ET) calendars.

Research/infrastructure only — no live broker / vendor MD.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from quant_fund.calendars import CRYPTO, FX, get_calendar

ET = ZoneInfo("America/New_York")
RESEARCH_ONLY = True
LIVE_PNL_CLAIM = False


# --- crypto -----------------------------------------------------------------


def test_crypto_every_day_is_a_session() -> None:
    for d in (date(2024, 2, 29), date(2024, 3, 10), date(2024, 12, 25)):
        assert CRYPTO.is_session_day(d)
        session = CRYPTO.session_on(d)
        assert session.date == d
        assert session.duration == timedelta(hours=24)
        assert not session.is_half_day


def test_crypto_session_bounds_are_utc_day() -> None:
    session = CRYPTO.session_on(date(2024, 6, 1))
    assert session.open_utc == datetime(2024, 6, 1, 0, 0, tzinfo=UTC)
    assert session.close_utc == datetime(2024, 6, 2, 0, 0, tzinfo=UTC)


def test_crypto_sessions_tile_without_gaps() -> None:
    sessions = CRYPTO.sessions(date(2024, 3, 9), date(2024, 3, 12))
    assert len(sessions) == 4
    for prev, nxt in zip(sessions[:-1], sessions[1:], strict=True):
        assert prev.close_utc == nxt.open_utc


# --- FX ----------------------------------------------------------------------


def test_fx_week_has_five_24h_sessions() -> None:
    sessions = FX.sessions(date(2024, 3, 4), date(2024, 3, 10))
    assert [s.date for s in sessions] == [
        date(2024, 3, 4),
        date(2024, 3, 5),
        date(2024, 3, 6),
        date(2024, 3, 7),
        date(2024, 3, 8),
    ]
    for session in sessions:
        assert session.duration == timedelta(hours=24)


def test_fx_monday_session_opens_sunday_1700_et() -> None:
    session = FX.session_on(date(2024, 3, 4))  # Monday label
    assert session is not None
    # Sunday 2024-03-03 17:00 EST = 22:00 UTC (pre-spring-forward).
    assert session.open_utc == datetime(2024, 3, 3, 22, 0, tzinfo=UTC)
    assert session.close_utc == datetime(2024, 3, 4, 22, 0, tzinfo=UTC)
    assert session.open_utc.astimezone(ET).time() == time(17, 0)


def test_fx_friday_close_is_weekend_gap_start() -> None:
    fri = FX.session_on(date(2024, 3, 8))
    assert fri is not None
    assert fri.close_utc == datetime(2024, 3, 8, 22, 0, tzinfo=UTC)  # 17:00 EST
    # Nothing trades between Friday close and Sunday 17:00 ET open.
    assert FX.session_on(date(2024, 3, 9)) is None
    assert FX.session_on(date(2024, 3, 10)) is None


def test_fx_sessions_after_dst_change() -> None:
    # After 2024-03-10 spring-forward, 17:00 ET = 21:00 UTC.
    session = FX.session_on(date(2024, 3, 11))
    assert session is not None
    assert session.open_utc == datetime(2024, 3, 10, 21, 0, tzinfo=UTC)
    assert session.close_utc == datetime(2024, 3, 11, 21, 0, tzinfo=UTC)
    assert session.duration == timedelta(hours=24)  # DST shift lands in gap


def test_fx_weekend_has_no_sessions_even_across_dst() -> None:
    # The 2024-11-03 fall-back Sunday still opens 17:00 ET (= 22:00 UTC after).
    session = FX.session_on(date(2024, 11, 4))
    assert session is not None
    assert session.open_utc == datetime(2024, 11, 3, 22, 0, tzinfo=UTC)


def test_fx_session_labels_use_close_date() -> None:
    # A Sunday-evening-ET open belongs to the Monday-labeled session.
    sessions = FX.sessions(date(2024, 3, 3), date(2024, 3, 3))
    assert sessions == []  # Sunday is never a session label


def test_fx_next_previous_session_day_skips_weekend() -> None:
    assert FX.next_session_day(date(2024, 3, 8)) == date(2024, 3, 11)
    assert FX.previous_session_day(date(2024, 3, 11)) == date(2024, 3, 8)


def test_fx_registry_alias() -> None:
    assert get_calendar("forex") is FX


def test_crypto_registry_alias() -> None:
    assert get_calendar("crypto24x7") is CRYPTO
