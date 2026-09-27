"""Proximity-window forbidden-headline matcher (LH008 core — audit F3 fix).

The old ``fx1/honesty.py`` regex requires the number to follow the token
*immediately* (``\\btoken\\b\\s*(?:of|=|:)?\\s*[-+$]?\\d``), so "Sharpe ratio
came in at 2.1", "the model's P&L was $4,200", or "hit rate — 61 percent"
sail through. This matcher instead:

1. Unicode-normalizes (NFKC) and casefolds the text.
2. Expands each forbidden token into aliases (``p&l``, ``net asset value``...).
3. Flags any alias occurrence within ``PROXIMITY_WINDOW`` chars of a numeric
   literal (incl. ``2.1``, ``$4,200``, ``61%``, ``61 percent``).
4. Second stage: any paragraph containing a forbidden token AND >= 2 numeric
   literals flags (numeric-density check).

``FORBIDDEN_HEADLINE_TOKENS`` is sourced from
``quant_fund.research.catalog.FORBIDDEN_RESEARCH_METRIC_KEYS`` (the catalog
constant — no copy; the fx1 drift-guard test keeps ``fx1/honesty.py``
honest). The import is lazy to respect the PROOFCORE layering contract
(leakage top-level imports are limited to contracts + schemas, §1.3).
"""

from __future__ import annotations

import re
import unicodedata

_TOKEN_FALLBACK: frozenset[str] = frozenset({"sharpe", "sortino", "calmar", "pnl", "nav"})


def _load_forbidden_tokens() -> frozenset[str]:
    """Import the catalog constant lazily; fall back to the pinned set."""
    try:
        from quant_fund.research.catalog import FORBIDDEN_RESEARCH_METRIC_KEYS
    except Exception:
        return _TOKEN_FALLBACK
    return frozenset(str(tok) for tok in FORBIDDEN_RESEARCH_METRIC_KEYS)


FORBIDDEN_HEADLINE_TOKENS: frozenset[str] = _load_forbidden_tokens()

_TOKEN_ALIASES: dict[str, tuple[str, ...]] = {
    "pnl": ("pnl", "p&l", "p/l", "profit and loss"),
    "nav": ("nav", "net asset value"),
    "sharpe": ("sharpe", "sharpe ratio"),
    "sortino": ("sortino",),
    "calmar": ("calmar",),
}

PROXIMITY_WINDOW = 80  # chars between token and numeric literal

# Numeric literal: optional currency sign, digits with optional ,/. grouping,
# optional % or "percent" suffix. Always contains at least one digit.
_NUMERIC_RE = re.compile(r"[-+]?[$€£]?\s*\d[\d,]*(?:\.\d+)?\s*(?:%|percent\b)?")


def _alias_pattern(alias: str) -> re.Pattern[str]:
    """Word-boundary-ish match that also works for non-word aliases (p&l)."""
    return re.compile(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])")


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def _aliases_for(token: str) -> tuple[str, ...]:
    return _TOKEN_ALIASES.get(token, (token,))


def find_forbidden_headline(text: str) -> list[str]:
    """Return the forbidden tokens headlined in *text* (empty = clean)."""
    norm = _normalize(text)
    hits: set[str] = set()
    for token in FORBIDDEN_HEADLINE_TOKENS:
        for alias in _aliases_for(token):
            pattern = _alias_pattern(alias)
            for match in pattern.finditer(norm):
                lo = max(0, match.start() - PROXIMITY_WINDOW)
                hi = min(len(norm), match.end() + PROXIMITY_WINDOW)
                if _NUMERIC_RE.search(norm, lo, hi):
                    hits.add(token)
                    break
            if token in hits:
                break
    # Second stage: paragraph numeric-density check — a paragraph that names a
    # forbidden token and cites >= 2 numbers is a headline even when each
    # individual number sits outside the proximity window.
    for paragraph in re.split(r"\n\s*\n|\n", norm):
        paragraph_hits = {
            token
            for token in FORBIDDEN_HEADLINE_TOKENS
            if any(_alias_pattern(alias).search(paragraph) for alias in _aliases_for(token))
        }
        if paragraph_hits and len(_NUMERIC_RE.findall(paragraph)) >= 2:
            hits.update(paragraph_hits)
    return sorted(hits)
