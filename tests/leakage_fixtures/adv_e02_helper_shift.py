"""ADVERSARIAL §1a-E2 (DOCUMENTED NEGATIVE): leaky helper, non-price arg name.

`panel[col].shift(-1)` behind a helper whose receiver is not price-named and
whose shift amount is an unresolved parameter stays below LH001's syntactic
bar, so LH014 (helper-indirection) has no body finding to propagate. The
positive twin is adv_helper_indirection.py.
"""

from __future__ import annotations


def _align(panel, col, n=1):
    return panel[col].shift(-n)


def signal(panel):
    return _align(panel, "adj_close")
