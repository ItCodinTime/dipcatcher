"""Calendar registry: stable names → calendar instances."""

from __future__ import annotations

from quant_fund.calendars.crypto import CRYPTO, CryptoCalendar
from quant_fund.calendars.fx import FX, FxCalendar
from quant_fund.calendars.sessions import TradingCalendar
from quant_fund.calendars.us_equities import XNYS, UsEquitiesCalendar

_CALENDARS: dict[str, TradingCalendar] = {
    "XNYS": XNYS,
    "NYSE": XNYS,
    "NASDAQ": XNYS,  # Nasdaq-listed equities follow the NYSE schedule
    "CRYPTO": CRYPTO,
    "CRYPTO24X7": CRYPTO,
    "FX": FX,
    "FOREX": FX,
}


def get_calendar(name: str) -> TradingCalendar:
    """Look up a calendar by name (case-insensitive); fail closed on unknown."""
    key = name.strip().upper()
    try:
        return _CALENDARS[key]
    except KeyError as exc:
        raise ValueError(f"unknown calendar {name!r}; supported: {sorted(_CALENDARS)}") from exc


def calendar_names() -> list[str]:
    return sorted(_CALENDARS)


__all__ = [
    "CRYPTO",
    "FX",
    "XNYS",
    "CryptoCalendar",
    "FxCalendar",
    "UsEquitiesCalendar",
    "calendar_names",
    "get_calendar",
]
