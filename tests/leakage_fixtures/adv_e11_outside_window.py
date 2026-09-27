"""ADVERSARIAL §1a-E11 (DOCUMENTED NEGATIVE): lone number outside the window.

A single digit literal more than PROXIMITY_WINDOW chars from the token, with
fewer than two numbers per paragraph, is below the matcher's bar. Pins the
residual ceiling.
"""

from __future__ import annotations

NOTE = (
    "The strategy Sharpe was debated at length by the committee, covering "
    "methodology, data hygiene, and execution assumptions across several "
    "review sessions without conclusion. Separately the memo cites 2.13."
)
