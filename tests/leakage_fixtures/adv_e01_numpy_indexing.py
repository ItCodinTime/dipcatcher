"""ADVERSARIAL §1a-E1 (DOCUMENTED NEGATIVE): lookahead via numpy indexing.

`px[1:] - px[:-1]` on a numpy view is invisible to the syntactic rule pack
(no rule covers array index arithmetic). Kept as a fixture so the residual
ceiling is pinned in CI: this file MUST NOT trip any error finding; if a
future rule catches it, move it to the positive set.
"""

from __future__ import annotations

import numpy as np


def forward_gap(bars) -> object:
    """Leaky in spirit: tomorrow's close minus today's, as a numpy view."""
    px = np.asarray(bars.to_numpy(), dtype=float)
    return px[1:] - px[:-1]
