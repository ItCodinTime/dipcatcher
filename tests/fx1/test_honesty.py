"""Honesty-inheritance tests: fx-1's output contract, tested like code."""

import pytest

from fx1.honesty import Fx1HonestyError, validate_fx1_output


def test_clean_research_output_passes():
    text = (
        "The distribution family scored CRPS 0.42 and PIT uniformity passed; "
        "verify with `uv run dipcatcher verify-research`."
    )
    assert validate_fx1_output(text) == text


@pytest.mark.parametrize("token", ["sharpe", "sortino", "calmar", "pnl", "nav"])
def test_forbidden_headline_metric_fails_closed(token: str):
    with pytest.raises(Fx1HonestyError):
        validate_fx1_output(f"The strategy achieved {token}: 2.35 over the panel.")


def test_bare_discussion_of_forbidden_metrics_allowed():
    # Explaining *why* a metric is forbidden is not a headline claim.
    text = "The lab forbids sharpe headlines in research scorecards by construction."
    assert validate_fx1_output(text) == text


@pytest.mark.parametrize(
    "claim",
    [
        "This produced live P&L of $12,000 last quarter.",
        "These are real money returns you can expect.",
        "The strategy offers guaranteed returns.",
    ],
)
def test_live_performance_claims_fail_closed(claim: str):
    with pytest.raises(Fx1HonestyError):
        validate_fx1_output(claim)


def test_synthetic_evidence_requires_label():
    with pytest.raises(Fx1HonestyError):
        validate_fx1_output("On synthetic data the model achieves 0.99 recovery accuracy.")
    # Explicit SYNTHETIC label passes.
    assert validate_fx1_output("On SYNTHETIC data the model achieves 0.99 recovery accuracy.")


def test_synthetic_commentary_without_numbers_allowed():
    # Refusals and contract discussion mention synthetic evidence without
    # presenting results - not a violation.
    text = "I will not rename synthetic evidence; the label stays."
    assert validate_fx1_output(text) == text
