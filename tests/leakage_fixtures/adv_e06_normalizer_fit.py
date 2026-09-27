"""ADVERSARIAL §1a-E6 (POSITIVE, LH003): global fit via `normalizer` attr."""

from __future__ import annotations


class FeaturePipeline:
    """Leaky: the normalizer statistics come from the FULL sample."""

    def __init__(self, normalizer, x_full) -> None:
        self.normalizer = normalizer
        self.normalizer.fit(x_full)
