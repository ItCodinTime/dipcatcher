"""SEEDED LEAK (LH007): annualized Sharpe fed into PSR (audit F1 pattern).

Deliberately leaky strategy file for the leakage-hunter CI gate
(DESIGN.md §6.4). MUST trip LH007. Do not import from strategy code.
"""

from __future__ import annotations


def sharpe_ratio(returns: list[float], periods_per_year: float = 252.0) -> float:
    """Self-contained stand-in: annualized by default, like metrics.returns."""
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / max(len(returns) - 1, 1)
    return mean / (var**0.5) * periods_per_year**0.5


def probabilistic_sharpe(sr: float, sr_star: float, n: int, skew: float, kurt: float) -> float:
    """Self-contained stand-in for metrics.overfitting.probabilistic_sharpe."""
    return 0.5 + (sr - sr_star) * (n - 1) ** 0.5 / max(1.0 - skew * sr, 1e-9) ** 0.5


def scorebook(returns: list[float], skew: float, kurt: float) -> float:
    """Leaky: annualized SR plugged into PSR with per-period n (z inflated)."""
    sr = sharpe_ratio(returns)
    sharpe = float(sr)
    return probabilistic_sharpe(sharpe, 0.0, len(returns), skew, kurt)
