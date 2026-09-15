"""Executable-alpha filter."""

from __future__ import annotations


def executable_alpha(
    alpha_gross: float,
    spread: float,
    fees: float,
    impact: float,
    borrow: float,
    uncertainty_penalty: float = 0.0,
) -> float:
    return alpha_gross - spread - fees - impact - borrow - uncertainty_penalty


def should_trade(exec_alpha: float, min_alpha: float) -> bool:
    return exec_alpha >= min_alpha
