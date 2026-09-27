"""Blocking honesty inheritance: fx-1's forbidden set must never drift
from the lab's catalog. If the lab adds a forbidden research metric key,
this suite fails until fx-1 inherits it."""

import pytest

from fx1.honesty import FORBIDDEN_HEADLINE_TOKENS, Fx1HonestyError, validate_fx1_output
from quant_fund.research.catalog import FORBIDDEN_RESEARCH_METRIC_KEYS


def test_honesty_forbidden_tokens_mirror_lab_catalog() -> None:
    assert FORBIDDEN_HEADLINE_TOKENS == FORBIDDEN_RESEARCH_METRIC_KEYS


@pytest.mark.parametrize("key", sorted(FORBIDDEN_RESEARCH_METRIC_KEYS))
def test_honesty_every_lab_forbidden_key_rejected_when_headlined(key: str) -> None:
    with pytest.raises(Fx1HonestyError):
        validate_fx1_output(f"The strategy achieved {key}: 2.35 over the panel.")


def test_honesty_no_unmirrored_fx1_only_tokens() -> None:
    extra = FORBIDDEN_HEADLINE_TOKENS - FORBIDDEN_RESEARCH_METRIC_KEYS
    assert not extra, f"fx-1 blocks tokens the lab does not: {sorted(extra)}"
