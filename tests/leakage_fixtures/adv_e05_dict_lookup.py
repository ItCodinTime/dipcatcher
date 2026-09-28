"""ADVERSARIAL §1a-E5 (DOCUMENTED NEGATIVE): future info via dict lookup.

`earnings_by_date.get(dates[i+1])` — index arithmetic into a precomputed
mapping is outside the syntactic rule pack. Pins the residual ceiling.
"""

from __future__ import annotations


def next_day_fact(earnings_by_date, dates, i):
    return earnings_by_date.get(dates[i + 1])
