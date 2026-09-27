"""US equity holiday rules and unscheduled-closure tables (NYSE/Nasdaq).

Self-contained: no third-party calendar dependency. Rules are the NYSE
schedule conventions used by ``exchange_calendars`` (XNYS); ad-hoc closures
and early closes are explicit tables of verified dates.

Scope: regular-schedule rules cover 1995 onward (MLK Day applies from
1998; Juneteenth from 2022). The standard early close has been 13:00 ET
since 1993. Earlier ordinary sessions are unsupported.

Verified full-day unscheduled closures in ``UNSCHEDULED_CLOSURES`` include
the 9/11 closure week (2001-09-11..14), Hurricane Sandy (2012-10-29/30) and
national days of mourning (Ford 2007-01-02, Reagan 2004-06-11,
G.H.W. Bush 2018-12-05, Carter 2025-01-09, Nixon 1994-04-27).
"""

from __future__ import annotations

from datetime import date, timedelta

SATURDAY = 5
SUNDAY = 6


def nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    """The n-th (1-based) ``weekday`` of ``month``."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    """The last ``weekday`` of ``month``."""
    if month == 12:
        last = date(year, 12, 31)
    else:
        last = date(year, month + 1, 1) - timedelta(days=1)
    offset = (last.weekday() - weekday) % 7
    return last - timedelta(days=offset)


def easter_sunday(year: int) -> date:
    """Gregorian Easter Sunday (Anonymous Gregorian algorithm / computus)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month, day = divmod(h + ell - 7 * m + 114, 31)
    return date(year, month, day + 1)


def observed(d: date, *, saturday_to_friday: bool = True) -> set[date]:
    """Holiday closure dates for a fixed-date holiday landing on ``d``.

    Saturday → observed Friday before (when ``saturday_to_friday``), Sunday →
    observed Monday after. Weekday holidays close on the day itself.
    """
    if d.weekday() == SATURDAY:
        return {d - timedelta(days=1)} if saturday_to_friday else set()
    if d.weekday() == SUNDAY:
        return {d + timedelta(days=1)}
    return {d}


def new_years_closures(year: int) -> set[date]:
    """New Year's Day closures for calendar ``year``.

    NYSE special case: when January 1 falls on a Saturday the preceding
    Friday (December 31) is NOT a holiday — the exchange stays open (e.g.
    2021-12-31 and 2010-12-31 were full sessions). Sunday → Monday January 2.
    """
    jan1 = date(year, 1, 1)
    if jan1.weekday() == SATURDAY:
        return set()
    return observed(jan1)


def us_equity_holidays(year: int) -> set[date]:
    """Scheduled full-day NYSE/Nasdaq closures for ``year``."""
    if year < 1995:
        raise ValueError("US equity regular schedules support years >= 1995")
    out: set[date] = set()
    out |= new_years_closures(year)
    if year >= 1998:
        out.add(nth_weekday_of_month(year, 1, 0, 3))  # MLK Day
    out.add(nth_weekday_of_month(year, 2, 0, 3))  # Washington's Birthday
    out.add(easter_sunday(year) - timedelta(days=2))  # Good Friday
    out.add(last_weekday_of_month(year, 5, 0))  # Memorial Day
    if year >= 2022:
        out |= observed(date(year, 6, 19))  # Juneteenth
    out |= observed(date(year, 7, 4))  # Independence Day
    out.add(nth_weekday_of_month(year, 9, 0, 1))  # Labor Day
    out.add(nth_weekday_of_month(year, 11, 3, 4))  # Thanksgiving
    out |= observed(date(year, 12, 25))  # Christmas
    # A Sunday New Year pushes its Monday observance into this year only via
    # new_years_closures(year); nothing leaks across the year boundary.
    return out


def us_equity_early_closes(year: int, holidays: set[date] | None = None) -> set[date]:
    """Scheduled 13:00 ET early closes for ``year``.

    - July 3 on Monday/Tuesday/Thursday; Wednesday from 2013 onward.
    - July 5 on Friday before 2013. July 3 is closed when July 4 is Saturday.
    - The day after Thanksgiving.
    - December 24 when it is a weekday and not the observed Christmas holiday
      (December 25 on Saturday → December 24 is a full closure).
    """
    if year < 1995:
        raise ValueError("US equity regular schedules support years >= 1995")
    holidays = us_equity_holidays(year) if holidays is None else holidays
    out: set[date] = set()
    july3 = date(year, 7, 3)
    if (
        july3.weekday() in {0, 1, 3} or (year >= 2013 and july3.weekday() == 2)
    ) and july3 not in holidays:
        out.add(july3)
    july5 = date(year, 7, 5)
    if year < 2013 and july5.weekday() == 4:
        out.add(july5)
    out.add(nth_weekday_of_month(year, 11, 3, 4) + timedelta(days=1))
    dec24 = date(year, 12, 24)
    if dec24.weekday() < 5 and dec24 not in holidays:
        out.add(dec24)
    out |= {d for d in ADHOC_EARLY_CLOSES if d.year == year}
    return out


# Verified unscheduled full-day closures (NYSE announcements / standard
# reference tables; identical to the exchange_calendars XNYS ad-hoc list for
# the covered years). Dates outside the supported rule range are still exact.
UNSCHEDULED_CLOSURES: frozenset[date] = frozenset(
    {
        date(1985, 9, 27),  # Hurricane Gloria
        date(1994, 4, 27),  # Richard Nixon funeral (national day of mourning)
        date(2001, 9, 11),  # September 11 attacks — closed through 9/14
        date(2001, 9, 12),
        date(2001, 9, 13),
        date(2001, 9, 14),
        date(2004, 6, 11),  # Ronald Reagan funeral
        date(2007, 1, 2),  # Gerald Ford funeral
        date(2012, 10, 29),  # Hurricane Sandy
        date(2012, 10, 30),
        date(2018, 12, 5),  # George H.W. Bush funeral
        date(2025, 1, 9),  # Jimmy Carter funeral (national day of mourning)
    }
)

# Verified ad-hoc early closes not covered by the recurring rules (NYSE
# announcements, as tabulated by the exchange_calendars XNYS early-close list):
# the Friday after a Thursday Christmas.
ADHOC_EARLY_CLOSES: frozenset[date] = frozenset(
    {
        date(1997, 12, 26),
        date(1999, 12, 31),  # millennium observance
        date(2003, 12, 26),
    }
)
