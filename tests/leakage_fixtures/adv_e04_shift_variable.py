"""ADVERSARIAL §1a-E4 (POSITIVE, LH001): negative shift via constant name."""

from __future__ import annotations

HALF = 10


def smooth(close):
    k = -HALF
    return close.rolling_mean(21).shift(k)
