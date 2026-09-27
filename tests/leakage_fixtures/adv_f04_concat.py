"""ADVERSARIAL §1a-F4 (POSITIVE, LH008): string-concat constant folding."""

from __future__ import annotations


def claim() -> str:
    return "Sharpe of " + "2.1" + " in the backtest"
