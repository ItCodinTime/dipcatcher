"""ADVERSARIAL §1a-E10 (POSITIVE, LH013 warning): f-string headline template.

The number arrives at runtime; the template plus a numeric format spec is
the disclosure channel.
"""

from __future__ import annotations


def render(sharpe: float) -> str:
    return f"The backtest Sharpe was {sharpe:.2f} for the evaluation window"
