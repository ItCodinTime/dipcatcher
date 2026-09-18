"""Executable-alpha filter."""

from __future__ import annotations

import math


def executable_alpha(
    alpha_gross: float,
    spread: float,
    fees: float,
    impact: float,
    borrow: float,
    uncertainty_penalty: float = 0.0,
) -> float:
    values = (alpha_gross, spread, fees, impact, borrow, uncertainty_penalty)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("executable-alpha inputs must be finite")
    return alpha_gross - spread - fees - impact - borrow - uncertainty_penalty


def should_trade(exec_alpha: float, min_alpha: float) -> bool:
    if not math.isfinite(float(exec_alpha)) or not math.isfinite(float(min_alpha)):
        return False
    return exec_alpha >= min_alpha
