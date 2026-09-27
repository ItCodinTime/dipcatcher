"""ADVERSARIAL §1a-E14 (POSITIVE, LH001): negative shift via resolved name."""

from __future__ import annotations

HORIZON = 5


def momentum(df):
    h = -HORIZON
    df["momentum"] = df["close"].shift(h) / df["close"] - 1
    return df
