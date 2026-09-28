"""Dividend reinvestment for research backtests.

Quote OHLC stays available in ``*_quote`` columns. The prices the research
backtest fills and marks (``open`` / ``high`` / ``low`` / ``close`` /
``close_total_return``) sit on one total-return basis, so a name bought on
the ex-date open is not credited with that ex-date dividend.

This module does not place orders and does not touch broker adapters.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import cast

import polars as pl

from quant_fund.data.adapters.stooq import session_close
from quant_fund.data.adapters.yahoo_eod import REVISION, SOURCE
from quant_fund.data.corporate_actions import adjust_prices
from quant_fund.schemas.errors import PointInTimeError

YAHOO_CHART_EVENTS = "div,split"

_CASH = ("cash_dividend", "special_dividend")
_REQUIRED_BARS = ("security_id", "event_time", "open", "high", "low", "close", "volume")
_ACTION_SCHEMA: dict[str, pl.DataType] = {
    "security_id": pl.String(),
    "event_time": pl.Datetime(time_zone="UTC"),
    "available_time": pl.Datetime(time_zone="UTC"),
    "action_type": pl.String(),
    "amount": pl.Float64(),
    "factor": pl.Float64(),
    "source": pl.String(),
    "revision_id": pl.String(),
}

__all__ = [
    "YAHOO_CHART_EVENTS",
    "apply_research_total_return",
    "parse_yahoo_corporate_actions",
]


def parse_yahoo_corporate_actions(
    payload: dict[str, object],
    *,
    security_id: str,
    yahoo_symbol: str,
) -> pl.DataFrame:
    """Chart ``events.dividends`` / ``events.splits`` as corporate-action rows.

    Ex-dates use the UTC calendar date of the Yahoo timestamp, then the same
    session close as ``parse_yahoo_chart``. Dividend rows are ``cash_dividend``
    because the chart payload does not separate specials. ``available_time``
    equals that session close.
    """
    if not security_id.strip() or not yahoo_symbol.strip():
        raise PointInTimeError("security_id and yahoo_symbol must be non-blank")
    block = _chart_block(payload)
    if block is None:
        return pl.DataFrame(schema=_ACTION_SCHEMA)
    events = block.get("events")
    if events is None:
        return pl.DataFrame(schema=_ACTION_SCHEMA)
    if not isinstance(events, dict):
        raise PointInTimeError("yahoo events is not an object")
    event_map = cast(dict[str, object], events)
    rows = _dividend_rows(event_map.get("dividends"), security_id, yahoo_symbol)
    rows.extend(_split_rows(event_map.get("splits"), security_id, yahoo_symbol))
    frame = _action_frame(rows)
    if frame.is_empty():
        return frame
    return frame.sort(["event_time", "action_type"])


def apply_research_total_return(
    bars: pl.DataFrame,
    actions: pl.DataFrame,
    *,
    prices_already_split_adjusted: bool = True,
) -> pl.DataFrame:
    """Reinvest cash dividends and put OHLC on the total-return basis.

    Yahoo quote history is already split-adjusted. The default leaves split
    rows unused so those factors are not applied twice. Raw prints must pass
    ``prices_already_split_adjusted=False``.

    Dividends on or before the first bar, or after the last bar, are outside
    the sample and are dropped. An ex-date strictly inside the sample that
    does not equal a bar fails closed.
    """
    if type(prices_already_split_adjusted) is not bool:
        raise PointInTimeError("prices_already_split_adjusted must be a bool")
    if any(name in bars.columns for name in ("open_quote", "close_quote")):
        raise PointInTimeError("research bars are already total-return adjusted")
    if bars.is_empty():
        if not actions.is_empty():
            raise PointInTimeError("corporate actions were supplied for an empty bar frame")
        return bars
    missing = [name for name in _REQUIRED_BARS if name not in bars.columns]
    if missing:
        raise PointInTimeError(f"research bars missing required columns: {missing}")
    ordered = bars.sort(["security_id", "event_time"])
    if ordered.select(["security_id", "event_time"]).is_duplicated().any():
        raise PointInTimeError("duplicate security_id/event_time in research bars")
    usable = _actions_for_adjustment(
        ordered,
        actions,
        prices_already_split_adjusted=prices_already_split_adjusted,
    )
    adjusted = adjust_prices(ordered, usable)
    return _scale_onto_total_return(adjusted)


def _chart_block(payload: dict[str, object]) -> dict[str, object] | None:
    chart = payload.get("chart")
    if not isinstance(chart, dict):
        return None
    result = cast(dict[str, object], chart).get("result")
    if not isinstance(result, list) or not result:
        return None
    block = result[0]
    if not isinstance(block, dict):
        return None
    return cast(dict[str, object], block)


def _event_entries(raw: object, *, what: str) -> list[tuple[object, dict[str, object]]]:
    if raw is None:
        return []
    if isinstance(raw, list):
        entries: list[tuple[object, dict[str, object]]] = []
        for item in raw:
            if not isinstance(item, dict):
                raise PointInTimeError(f"yahoo {what} entry is not an object")
            entries.append(("", cast(dict[str, object], item)))
        return entries
    if isinstance(raw, dict):
        entries = []
        for key, item in raw.items():
            if not isinstance(item, dict):
                raise PointInTimeError(f"yahoo {what} entry is not an object")
            entries.append((key, cast(dict[str, object], item)))
        return entries
    raise PointInTimeError(f"yahoo {what} is not a map or list")


def _float(value: object, *, what: str) -> float:
    if isinstance(value, bool) or value is None:
        raise PointInTimeError(f"{what} is not a number")
    number: float
    if isinstance(value, int | float):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value)
        except ValueError as exc:
            raise PointInTimeError(f"{what} is not a number") from exc
    else:
        raise PointInTimeError(f"{what} is not a number")
    if not math.isfinite(number):
        raise PointInTimeError(f"{what} is not finite")
    return number


def _positive_float(value: object, *, what: str) -> float:
    number = _float(value, what=what)
    if number <= 0.0:
        raise PointInTimeError(f"{what} must be positive")
    return number


def _unix_seconds(value: object) -> int:
    number = _float(value, what="yahoo event timestamp")
    if not number.is_integer():
        raise PointInTimeError("yahoo event timestamp is not an integer second")
    stamp = int(number)
    if stamp <= 0 or stamp > 10_000_000_000:
        raise PointInTimeError("yahoo event timestamp is outside the seconds range")
    return stamp


def _session_time(stamp: int, yahoo_symbol: str) -> datetime:
    day = datetime.fromtimestamp(stamp, tz=UTC).date()
    suffix = ".uk" if yahoo_symbol.endswith(".L") else ".us"
    return session_close(day, suffix)


def _stamp_from(item: dict[str, object], key: object, *, what: str) -> int:
    raw = item.get("date")
    if raw is None:
        raw = key if key != "" else None
    if raw is None:
        raise PointInTimeError(f"yahoo {what} is missing a timestamp")
    return _unix_seconds(raw)


def _action_row(
    security_id: str,
    when: datetime,
    action_type: str,
    *,
    amount: float | None,
    factor: float | None,
) -> dict[str, object]:
    return {
        "security_id": security_id,
        "event_time": when,
        "available_time": when,
        "action_type": action_type,
        "amount": amount,
        "factor": factor,
        "source": SOURCE,
        "revision_id": REVISION,
    }


def _action_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=_ACTION_SCHEMA)
    return pl.DataFrame(rows, schema=_ACTION_SCHEMA)


def _dividend_rows(raw: object, security_id: str, yahoo_symbol: str) -> list[dict[str, object]]:
    totals: dict[datetime, float] = {}
    for key, item in _event_entries(raw, what="dividend"):
        amount = _positive_float(item.get("amount"), what="dividend amount")
        when = _session_time(_stamp_from(item, key, what="dividend"), yahoo_symbol)
        totals[when] = totals.get(when, 0.0) + amount
    return [
        _action_row(security_id, when, "cash_dividend", amount=amount, factor=None)
        for when, amount in sorted(totals.items())
    ]


def _split_rows(raw: object, security_id: str, yahoo_symbol: str) -> list[dict[str, object]]:
    factors: dict[datetime, float] = {}
    for key, item in _event_entries(raw, what="split"):
        numerator = _positive_float(item.get("numerator"), what="split numerator")
        denominator = _positive_float(item.get("denominator"), what="split denominator")
        when = _session_time(_stamp_from(item, key, what="split"), yahoo_symbol)
        if when in factors:
            raise PointInTimeError("duplicate yahoo split on one session")
        factors[when] = numerator / denominator
    return [
        _action_row(security_id, when, "split", amount=None, factor=factor)
        for when, factor in sorted(factors.items())
    ]


def _actions_for_adjustment(
    bars: pl.DataFrame,
    actions: pl.DataFrame,
    *,
    prices_already_split_adjusted: bool,
) -> pl.DataFrame:
    if actions.is_empty():
        return actions
    if "action_type" not in actions.columns:
        raise PointInTimeError("corporate actions missing action_type")
    if actions.filter(pl.col("action_type").is_null()).height:
        raise PointInTimeError("corporate actions contain null action_type")
    kinds = list(_CASH)
    if not prices_already_split_adjusted:
        kinds.append("split")
    kept = actions.filter(pl.col("action_type").is_in(kinds))
    if kept.is_empty():
        return kept
    _reject_unknown_ids(bars, kept)
    if "available_time" in kept.columns:
        late = kept.filter(
            pl.col("available_time").is_null() | (pl.col("available_time") > pl.col("event_time"))
        )
        if late.height:
            raise PointInTimeError(
                "corporate action available_time must be at or before event_time"
            )
        kept = kept.drop("available_time")
    dividends = _dividends_on_bars(bars, kept.filter(pl.col("action_type").is_in(list(_CASH))))
    if prices_already_split_adjusted:
        return dividends
    splits = kept.filter(pl.col("action_type") == "split")
    if splits.is_empty():
        return dividends
    if dividends.is_empty():
        return splits
    return pl.concat([dividends, splits], how="diagonal_relaxed")


def _reject_unknown_ids(bars: pl.DataFrame, actions: pl.DataFrame) -> None:
    unknown = actions.join(bars.select("security_id").unique(), on="security_id", how="anti")
    if unknown.height:
        raise PointInTimeError("corporate action security_id is absent from the research bars")


def _dividends_on_bars(bars: pl.DataFrame, dividends: pl.DataFrame) -> pl.DataFrame:
    if dividends.is_empty():
        return dividends
    span = bars.group_by("security_id").agg(
        pl.col("event_time").min().alias("_first"),
        pl.col("event_time").max().alias("_last"),
    )
    window = dividends.join(span, on="security_id", how="inner").filter(
        (pl.col("event_time") > pl.col("_first")) & (pl.col("event_time") <= pl.col("_last"))
    )
    in_window = window.drop("_first", "_last")
    unmatched = in_window.join(
        bars.select(["security_id", "event_time"]),
        on=["security_id", "event_time"],
        how="anti",
    )
    if unmatched.height:
        raise PointInTimeError("in-sample dividend ex-date does not match a research bar")
    return in_window


def _scale_onto_total_return(adjusted: pl.DataFrame) -> pl.DataFrame:
    scaled = adjusted.with_columns(
        pl.col("open").alias("open_quote"),
        pl.col("high").alias("high_quote"),
        pl.col("low").alias("low_quote"),
        pl.col("close").alias("close_quote"),
        (pl.col("close_total_return") / pl.col("close_split_adjusted")).alias("_tr_scale"),
    )
    bad = scaled.filter(
        pl.col("_tr_scale").is_null()
        | (~pl.col("_tr_scale").is_finite())
        | (pl.col("_tr_scale") <= 0.0)
    )
    if bad.height:
        raise PointInTimeError("total-return scale is not finite and positive")
    return scaled.with_columns(
        (pl.col("open_split_adjusted") * pl.col("_tr_scale")).alias("open"),
        (pl.col("high_split_adjusted") * pl.col("_tr_scale")).alias("high"),
        (pl.col("low_split_adjusted") * pl.col("_tr_scale")).alias("low"),
        pl.col("close_total_return").alias("close"),
        pl.lit("total_return").alias("return_basis"),
    ).drop("_tr_scale")
