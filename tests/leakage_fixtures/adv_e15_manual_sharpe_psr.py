"""ADVERSARIAL §1a-E15 (DOCUMENTED NEGATIVE): manual annualized Sharpe to PSR.

`sr = mean / std * sqrt(252)` fed to the PSR path without a ``sharpe_ratio``
call is below LH007's syntactic bar. Pins the residual ceiling.
"""

from __future__ import annotations

from math import sqrt


def psr_input(returns, benchmark=0.0):
    mean = sum(returns) / len(returns)
    std = sqrt(sum((r - mean) ** 2 for r in returns) / (len(returns) - 1))
    sr = mean / std * sqrt(252)
    return sr, benchmark
