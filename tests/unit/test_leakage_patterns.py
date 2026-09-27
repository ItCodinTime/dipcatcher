"""Proximity-window headline matcher tests — audit F3 evasion battery verbatim."""

from __future__ import annotations

import hypothesis.strategies as st
from hypothesis import given, settings

from quant_fund.leakage.patterns import (
    FORBIDDEN_HEADLINE_TOKENS,
    find_forbidden_headline,
)

# The audit F3 evasion battery: every one of these sails through the legacy
# fx1/honesty.py regex (number must follow the token immediately).
EVASION_BATTERY: dict[str, list[str]] = {
    "Sharpe ratio came in at 2.1": ["sharpe"],
    "the model's P&L was $4,200": ["pnl"],
    "net asset value printed 1.23": ["nav"],
    "we observed a Sharpe... (details omitted) ... of 2.1": ["sharpe"],
    "P&L $4,200": ["pnl"],
    "Sortino: 1.8": ["sortino"],
    "Calmar = 0.9": ["calmar"],
    "pnl came to -$1,250.50 on the book": ["pnl"],
    "the NAV prints at 103.44": ["nav"],
    "Sharpe ratio of 2.1": ["sharpe"],  # the legacy regex DID catch this one
}

CLEAN_STRINGS: tuple[str, ...] = (
    "Sharpe is a forbidden headline metric; use proper scores.",
    "we never headline pnl or nav in research output",
    "pinball loss improved and coverage calibrated",  # no forbidden token at all
    "the navigator pattern routes orders",  # 'navi...' must not match 'nav'
    "planar calibration improved the pit histogram",  # 'planar' must not match 'pnl'/'p&l'
    "",
)

# Boundary documentation: "hit rate" is NOT in the catalog's forbidden token
# set (only sharpe/sortino/calmar/pnl/nav are), so the matcher deliberately
# does not flag it. If the catalog grows the token, this test must flip.
HIT_RATE_BOUNDARY = "hit rate — 61 percent"


def test_evasion_battery_caught() -> None:
    for text, expected in EVASION_BATTERY.items():
        assert find_forbidden_headline(text) == expected, f"missed evasion: {text!r}"


def test_clean_strings_not_flagged() -> None:
    for text in CLEAN_STRINGS:
        assert find_forbidden_headline(text) == [], f"false positive on: {text!r}"


def test_hit_rate_boundary_documented() -> None:
    assert find_forbidden_headline(HIT_RATE_BOUNDARY) == []


def test_unicode_normalization() -> None:
    # Full-width "Ｓｈａｒｐｅ" + full-width digits must still flag after NFKC.
    assert find_forbidden_headline("Ｓｈａｒｐｅ ２.１") == ["sharpe"]
    # Casefold: uppercase alias.
    assert find_forbidden_headline("SHARPE RATIO: 2.1") == ["sharpe"]


def test_paragraph_numeric_density_second_stage() -> None:
    # Token and numbers > 80 chars apart, but >= 2 numerics in the paragraph.
    text = "sharpe is discussed here" + "x" * 120 + "see table 4 and figure 7"
    assert find_forbidden_headline(text) == ["sharpe"]


def test_single_distant_number_not_flagged() -> None:
    text = "sharpe is discussed here" + "x" * 120 + "see table 4 at the end"
    assert find_forbidden_headline(text) == []


def test_token_set_matches_catalog() -> None:
    """Drift guard: matcher tokens come from research.catalog (no copy)."""
    from quant_fund.research.catalog import FORBIDDEN_RESEARCH_METRIC_KEYS

    assert frozenset(FORBIDDEN_RESEARCH_METRIC_KEYS) == FORBIDDEN_HEADLINE_TOKENS


_SAFE_ALPHABET = "bdgfjquwxyz0123456789 \n$%,.()-—"  # cannot spell any alias


@given(st.text(alphabet=_SAFE_ALPHABET, max_size=400))
@settings(max_examples=200, derandomize=True)
def test_matcher_never_flags_token_free_text(text: str) -> None:
    assert find_forbidden_headline(text) == []
