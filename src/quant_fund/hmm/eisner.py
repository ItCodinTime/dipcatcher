"""Jason Eisner (2002) ice-cream HMM from Jurafsky & Martin SLP3 Fig. A.2.

States: 0=COLD, 1=HOT. Observations are ice-cream counts {1,2,3} stored as
0-based indices {0,1,2}.
"""

from __future__ import annotations

import numpy as np

from quant_fund.hmm.discrete import DiscreteHMM

# π = [P(C), P(H)] = [0.2, 0.8]
# A: rows from, cols to. COLD→COLD 0.5, COLD→HOT 0.5, HOT→COLD 0.4, HOT→HOT 0.6
# B: P(1,2,3 | COLD) = [0.5, 0.4, 0.1], P(1,2,3 | HOT) = [0.2, 0.4, 0.4]
EISNER = DiscreteHMM(
    A=np.array([[0.5, 0.5], [0.4, 0.6]], dtype=float),
    B=np.array([[0.5, 0.4, 0.1], [0.2, 0.4, 0.4]], dtype=float),
    pi=np.array([0.2, 0.8], dtype=float),
)

COLD, HOT = 0, 1
STATE_NAMES = ("COLD", "HOT")


def ice_cream(counts: list[int]) -> list[int]:
    """Map ice-cream counts {1,2,3} to 0-based observation indices."""
    out = []
    for c in counts:
        if c not in (1, 2, 3):
            raise ValueError("Eisner observations are ice-cream counts 1, 2, or 3")
        out.append(int(c) - 1)
    return out
