"""Deterministic reward model for fx-1 RLHF / rejection sampling.

Instead of (or before) training a neural reward model, fx-1 uses the honesty
contract and evidence rules as an executable reward: compliant, receipt-bound,
honest outputs score high; forbidden headlines, live claims, unlabeled
synthetic evidence, and missing provenance score low. Fully auditable — every
point of reward is attributable to a named rule.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from fx1.honesty import Fx1HonestyError, validate_fx1_output

_WEIGHTS = {
    "honesty_clean": 4.0,
    "cites_receipt": 2.0,
    "states_evidence_class": 1.5,
    "mentions_verification": 1.0,
    "proper_score_vocabulary": 1.0,
    "hedged_uncertainty": 0.5,
}

_RECEIPT_RE = re.compile(r"\b[0-9a-f]{8,64}…?\b")
_EVIDENCE_CLASS_RE = re.compile(
    r"\b(research|backtest|simulated paper|synthetic)\s+(evidence|results?)\b",
    re.IGNORECASE,
)
_VERIFICATION_RE = re.compile(r"verify-research|verify_research", re.IGNORECASE)
_PROPER_SCORE_RE = re.compile(
    r"\b(crps|pinball|pit|qlike|brier|log[-_ ]?loss|ece|kupiec)\b", re.IGNORECASE
)
_UNCERTAINTY_RE = re.compile(
    r"\b(uncertain|confidence|calibrat\w+|interval|estimate[sd]?)\b", re.IGNORECASE
)


class RewardBreakdown(BaseModel):
    total: float
    components: dict[str, float] = Field(default_factory=dict)
    violations: list[str] = Field(default_factory=list)


def score_response(text: str) -> RewardBreakdown:
    """Score one fx-1 response. Honesty violation caps the total at -10."""
    components: dict[str, float] = {}
    violations: list[str] = []
    try:
        validate_fx1_output(text)
        components["honesty_clean"] = _WEIGHTS["honesty_clean"]
    except Fx1HonestyError as exc:
        violations.append(str(exc))
        return RewardBreakdown(total=-10.0, components=components, violations=violations)
    if _RECEIPT_RE.search(text):
        components["cites_receipt"] = _WEIGHTS["cites_receipt"]
    if _EVIDENCE_CLASS_RE.search(text):
        components["states_evidence_class"] = _WEIGHTS["states_evidence_class"]
    if _VERIFICATION_RE.search(text):
        components["mentions_verification"] = _WEIGHTS["mentions_verification"]
    if _PROPER_SCORE_RE.search(text):
        components["proper_score_vocabulary"] = _WEIGHTS["proper_score_vocabulary"]
    if _UNCERTAINTY_RE.search(text):
        components["hedged_uncertainty"] = _WEIGHTS["hedged_uncertainty"]
    return RewardBreakdown(
        total=sum(components.values()), components=components, violations=violations
    )
