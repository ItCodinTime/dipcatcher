"""CLEAN CONTROL: fold-scoped scaler fit.

Mirrors the `pipeline/train.py` idiom — scaler fit lexically inside the fold
loop over train indices only. MUST trip ZERO error-severity findings
(DESIGN.md §6.4).
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


def train_folds(features: list[float], folds: list[tuple[list[int], list[int]]]) -> list[float]:
    """Legitimate: scaler statistics come from each fold's train slice only."""
    scores: list[float] = []
    for train_idx, _test_idx in folds:
        scaler = MinMaxScaler()
        scaler.fit([features[i] for i in train_idx])
        scores.append(sum(scaler.transform(features)) / len(features))
    return scores
