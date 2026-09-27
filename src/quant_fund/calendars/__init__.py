"""Trading-calendar layer: sessions, DST-safe bar alignment, tz policy.

Single source of truth for exchange session structure:

- :class:`UsEquitiesCalendar` (``XNYS``/``NYSE``/``NASDAQ``): 09:30–16:00
  America/New_York regular hours, 13:00 ET early closes, scheduled holidays
  (rule-based) plus a verified unscheduled-closure table (9/11 week, Sandy,
  national days of mourning).
- :class:`CryptoCalendar` (``CRYPTO``): 24/7/365 UTC-day sessions.
- :class:`FxCalendar` (``FX``): five 24h sessions per week on the
  New-York-close convention — Sunday 17:00 ET → Friday 17:00 ET.

Timezone policy: all session/bar bounds are tz-aware UTC; exchange-local
wall time appears only as label columns and inside calendar construction
(stdlib ``zoneinfo``; naive datetimes are refused at the boundary).
"""

from quant_fund.calendars.alignment import (
    assign_bars,
    bars_in_session,
    grid_invariants,
    session_grid,
    slot_for,
)
from quant_fund.calendars.crypto import CRYPTO, CryptoCalendar
from quant_fund.calendars.fx import FX, FxCalendar
from quant_fund.calendars.holidays import (
    ADHOC_EARLY_CLOSES,
    UNSCHEDULED_CLOSURES,
    us_equity_early_closes,
    us_equity_holidays,
)
from quant_fund.calendars.registry import calendar_names, get_calendar
from quant_fund.calendars.sessions import (
    Session,
    SessionIndex,
    TradingCalendar,
    require_aware,
    session_at,
)
from quant_fund.calendars.us_equities import XNYS, UsEquitiesCalendar

__all__ = [
    "ADHOC_EARLY_CLOSES",
    "CRYPTO",
    "FX",
    "UNSCHEDULED_CLOSURES",
    "XNYS",
    "CryptoCalendar",
    "FxCalendar",
    "Session",
    "SessionIndex",
    "TradingCalendar",
    "UsEquitiesCalendar",
    "assign_bars",
    "bars_in_session",
    "calendar_names",
    "get_calendar",
    "grid_invariants",
    "require_aware",
    "session_at",
    "session_grid",
    "slot_for",
    "us_equity_early_closes",
    "us_equity_holidays",
]
