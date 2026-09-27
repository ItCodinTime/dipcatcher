"""fx-1 honesty inheritance.

The lab's integrity rules (``quant_fund.research.catalog``) are enforced on
research *artifacts*. This module enforces the same rules on fx-1 *outputs*:
the agent must not emit forbidden research-headline metric claims, must not
present synthetic evidence as market evidence, and must not claim live
performance. Fail-closed: any violation raises :class:`Fx1HonestyError`.
"""

from __future__ import annotations

import re

# Mirrors FORBIDDEN_RESEARCH_METRIC_KEYS in quant_fund.research.catalog:
# headline P&L / ratio tokens that must never appear as fx-1 research results.
FORBIDDEN_HEADLINE_TOKENS: frozenset[str] = frozenset({"sharpe", "sortino", "calmar", "pnl", "nav"})

# Phrases that constitute a live-performance or synthetic-as-live claim.
_FORBIDDEN_CLAIM_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\blive p(?:&|and)l\b",
        r"\blive trading profit",
        r"\breal money (returns?|gains?|profits?)\b",
        r"\bguaranteed (returns?|alpha|profits?)\b",
        r"\bsynthetic results? (show|prove|demonstrate)s? (live|real|market)\b",
    )
)

# Synthetic evidence must carry an explicit SYNTHETIC label nearby.
_SYNTHETIC_TOKEN = re.compile(r"\bsynthetic\b", re.IGNORECASE)
# The label must be the explicit uppercase SYNTHETIC marker (case-sensitive).
_SYNTHETIC_LABEL = re.compile(r"\bSYNTHETIC\b")


class Fx1HonestyError(ValueError):
    """Raised when an fx-1 output violates the lab honesty contract."""


def _contains_forbidden_headline(text: str) -> str | None:
    """Return the offending token if *text* headlines a forbidden metric.

    A headline claim is a forbidden token immediately followed by a numeric
    value or ratio phrasing (e.g. "Sharpe 2.1", "pnl: +$4,200"). Bare
    discussion of why these metrics are forbidden is allowed.
    """
    for token in FORBIDDEN_HEADLINE_TOKENS:
        pattern = re.compile(rf"\b{token}\b\s*(?:of|=|:)?\s*[-+$]?\d[\d,.%$]*", re.IGNORECASE)
        if pattern.search(text):
            return token
    return None


def validate_fx1_output(text: str) -> str:
    """Fail-closed validation of a single fx-1 natural-language output.

    Returns *text* unchanged when clean; raises :class:`Fx1HonestyError`
    otherwise. Applied to every agent message that leaves a session.
    """
    token = _contains_forbidden_headline(text)
    if token is not None:
        raise Fx1HonestyError(
            f"fx-1 output headlines forbidden research metric token {token!r}; "
            "research results are proper scores, not Sharpe/P&L headlines."
        )
    for pattern in _FORBIDDEN_CLAIM_PATTERNS:
        if pattern.search(text):
            raise Fx1HonestyError(
                "fx-1 output contains a live-performance or synthetic-as-live "
                "claim; the lab's evidence gates forbid this."
            )
    if _SYNTHETIC_TOKEN.search(text) and not _SYNTHETIC_LABEL.search(text):
        # The label rule targets *presentation* of synthetic results, not
        # discussion. If no numeric claim accompanies the mention, the text
        # is commentary (e.g. a refusal) and passes.
        numeric = re.search(r"\d", text)
        presenting = re.search(
            r"\b(shows?|prove[sd]?|achiev\w+|scor\w+|result\w*|accuracy|"
            r"recover\w+|performance)\b",
            text,
            re.IGNORECASE,
        )
        if numeric and presenting:
            raise Fx1HonestyError(
                "synthetic evidence presented (with numeric claims) without "
                "an explicit SYNTHETIC label."
            )
    return text
