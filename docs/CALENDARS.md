# Trading calendars and bar alignment

`src/quant_fund/calendars/` is the single trading-calendar layer: exchange
session structure, DST-safe bar grids, and timestamp→slot assignment.
Research/infrastructure only — nothing here reaches a broker or live path.

## Supported calendars

| Registry names | Class | Sessions |
|---|---|---|
| `XNYS`, `NYSE`, `NASDAQ` | `UsEquitiesCalendar` | 09:30–16:00 `America/New_York` regular hours; 13:00 ET early closes (`is_half_day`); weekends + scheduled holidays + unscheduled closures are non-session days |
| `CRYPTO`, `CRYPTO24X7` | `CryptoCalendar` | UTC calendar days `[00:00, 24:00)`; every day is a session (24/7/365) |
| `FX`, `FOREX` | `FxCalendar` | Five 24h sessions/week on the New-York-close convention: session labeled `d` (Mon–Fri) runs 17:00 ET on `d-1` → 17:00 ET on `d`; Sunday 17:00 ET opens the Monday session |

Look up by name with `get_calendar(name)` (case-insensitive, fail-closed).

### NYSE holidays and closures

`holidays.us_equity_holidays(year)` is rule-based and self-contained (no
third-party dependency): New Year's Day (with the NYSE special case — a
*Saturday* January 1 is **not** observed on December 31; Sunday → Monday
January 2), MLK Day (3rd Mon Jan, ≥1998), Washington's Birthday, Good Friday
(Gregorian computus), Memorial Day, Juneteenth (≥2022), Independence Day,
Labor Day, Thanksgiving, Christmas — each with Saturday→Friday /
Sunday→Monday observance.

`holidays.UNSCHEDULED_CLOSURES` is an explicit verified table including the
9/11 closure week (2001-09-11..14), Hurricane Sandy (2012-10-29/30),
Hurricane Gloria (1985-09-27), and national days of mourning (Nixon
1994-04-27, Reagan 2004-06-11, Ford 2007-01-02, G.H.W. Bush 2018-12-05,
Carter 2025-01-09).

Early closes (`us_equity_early_closes`, 13:00 ET): July 3 (weekday, unless
it is the observed holiday), the day after Thanksgiving, December 24
(weekday, unless it is the observed holiday), plus `ADHOC_EARLY_CLOSES`
(1997-12-26, 2003-12-26 — the Friday after a Thursday Christmas).

Reliability window: rules are validated for 1998+. Pre-2003 early closes
that ran to 14:00 ET are modeled at 13:00 ET (documented approximation).
The FX calendar carries no holiday table — FX trades through US holidays.

## Timezone policy

- **Storage is tz-aware UTC.** `Session.open_utc`/`close_utc` and every
  `*_utc` grid column are `Datetime("us", "UTC")` in polars / aware
  `datetime` in Python. `Session.__post_init__` rejects naive or non-UTC
  bounds.
- **Exchange-local wall time is a label, not storage.** Session bounds are
  constructed from local wall times via stdlib `zoneinfo`, so 09:30 ET maps
  to 14:30 UTC under EST and 13:30 UTC under EDT automatically — no manual
  offset tables. Grid `bar_open_local`/`bar_close_local` columns keep the
  exchange-local tz for display.
- **Naive datetimes are refused at the boundary** (`require_aware`,
  `assign_bars`, `slot_for`); elsewhere the convention is *naive = UTC
  wall-clock* (see `validation/walk_forward.py`, `_stamp_utc_date` in
  `pipeline/forecast.py`).

## Bar alignment

`session_grid(calendar, start, end, bar_size)` tiles each session with bars
`[open + i·bar_size, min(open + (i+1)·bar_size, close))` — left-labeled,
contiguous, covering the session exactly. When the session length is not a
multiple of `bar_size`, the final bar is **truncated at the close** (never
extended past it, never zero-length).

`assign_bars(calendar, timestamps, bar_size)` maps each aware timestamp to
its `(session_date, bar_index, bar_open_utc, bar_close_utc)` slot via
floor-division on elapsed UTC; timestamps outside every session (weekends,
holidays, the FX weekend gap, the exclusive close instant) get nulls.

`grid_invariants(frame)` returns violation strings — the same checks the
Hypothesis suite asserts: bars inside their session, no gaps/overlaps, no
duplicate slots, contiguous `bar_index`.

```python
from datetime import date, timedelta
from quant_fund.calendars import XNYS, session_grid, assign_bars

grid = session_grid(XNYS, date(2024, 3, 8), date(2024, 3, 11), timedelta(minutes=5))
slots = assign_bars(XNYS, [some_aware_ts], timedelta(minutes=5))
```

### Relationship to `data/calendars.py`

`quant_fund.data.calendars` remains the lightweight weekday-only helper used
by the `synthetic`/`hf_ohlcv_1m` adapters (it predates this layer and is on
the leakage-rule allowlist). New exchange-aware work should use
`quant_fund.calendars`; the two are intentionally not rewired together in
this change to avoid perturbing sealed research flows.

## Naive-datetime audit (this change)

Audited `src/quant_fund` research/data paths for naive producers
(`datetime.now()`, `utcnow()`, naive `pd.Timestamp`, `fromisoformat`
without an offset, `replace(tzinfo=None)` wall-clock comparisons). All
`datetime.now` call sites were already tz-aware. Findings and fixes:

Fixed (research/data code):

- `cli/research_cmds.py` — `datetime.fromisoformat(date)` produced a naive
  `asof` for `--date 2024-01-05` on the `forecast`, `kronos-forecast` and
  `optimize` commands; now `_parse_asof` localizes naive input to UTC.
- `pipeline/kronos.py::_resolve_asof` — a naive/date `asof` could return a
  naive `MarketState.asof` while asset forecasts were stamped UTC; now
  normalized (naive→UTC, aware→UTC).
- `pipeline/train.py::_stamp_strictly_before/_stamp_at_or_before` —
  mixed naive/aware comparisons stripped `tzinfo` *without* converting,
  comparing the aware stamp's local wall time; now both sides normalize to
  UTC first (behavior identical for the tested UTC convention).
- `pipeline/forecast.py` — calibrator staleness compared `fit_end.date()`
  in the stamp's own offset; now `_stamp_utc_date` compares UTC dates.
- `models/robinhood_plus/torch_backend.py::_future_stamps` — empty-history
  fallback `datetime(2020, 1, 2)` was naive; now UTC-aware.

Listed, not fixed (live/paper path — off-limits per task):

- `paper/loop.py::_parse_iso` — `datetime.fromisoformat(value)` returns
  whatever offset the stored string carries, including naive → downstream
  paper-clock comparisons can mix naive/aware. Owner decision required.

Adapters (`data/adapters/vendor_http.py`, `data/sources/base.py`,
`data/adapters/hf_ohlcv_1m.py`, `stooq.py`, `yahoo_eod.py`) already reject
or localize naive input — audited, no change needed.

## Tests

- `tests/unit/calendars/` — known holidays, unscheduled closures, early
  closes, DST open/close UTC values, crypto/FX session shape, alignment and
  edge cases.
- `tests/property/test_calendar_alignment.py` — Hypothesis invariants
  across the 2024 DST transitions, Sandy/9-11 closure weeks, half days and
  14 bar sizes on all three calendars: exact tiling, valid-slot assignment,
  no duplicates, no gaps within session bounds.
