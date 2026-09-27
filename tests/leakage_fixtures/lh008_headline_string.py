"""SEEDED LEAK (LH008): forbidden-metric headline in an emitted string.

The string below is one of the audit F3 evasion battery entries verbatim —
it bypasses the legacy fx1 honesty regex. Deliberately leaky strategy file
for the leakage-hunter CI gate (DESIGN.md §6.4). MUST trip LH008.
Do not import from strategy code.
"""

from __future__ import annotations


def render_summary() -> str:
    """Leaky: headlines a forbidden metric with the number out of regex reach."""
    return "Sharpe ratio came in at 2.1 for the evaluation window"
