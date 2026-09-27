"""SEEDED LEAK (LH003): scaler fit before the fold split.

Deliberately leaky strategy file for the leakage-hunter CI gate
(DESIGN.md §6.4). MUST trip LH003. Do not import from strategy code.
"""

from __future__ import annotations


class MinMaxScaler:
    """Tiny self-contained scaler stand-in (keeps the fixture importable)."""

    def __init__(self) -> None:
        self.lo = 0.0
        self.hi = 1.0

    def fit(self, values: list[float]) -> None:
        self.lo = min(values)
        self.hi = max(values)

    def transform(self, values: list[float]) -> list[float]:
        span = self.hi - self.lo or 1.0
        return [(v - self.lo) / span for v in values]


def train_full_sample(features: list[float]) -> list[float]:
    """Leaky: scaler statistics are computed over the WHOLE sample."""
    scaler = MinMaxScaler()
    scaler.fit(features)
    return scaler.transform(features)
