"""Rephrased-gap (ConStat-style) evaluation tests."""

from __future__ import annotations

import pytest

from fx1.eval.bank import DOMAIN_TASKS, HONESTY_BAITS
from fx1.eval.rephrased import (
    canonical_rephrased_pair,
    rephrased_twins,
    run_rephrased_gap,
)


def test_every_bait_and_domain_task_has_a_rephrasing():
    # Fail-closed: adding a bank task without curating its rephrasing breaks
    # this test, so the twin set can never silently lag the bank.
    canonical, rephrased = canonical_rephrased_pair()
    assert len(canonical) == len(HONESTY_BAITS) + len(DOMAIN_TASKS)
    assert len(rephrased) == len(canonical)


def test_rephrased_twins_preserve_contract_change_surface():
    canonical, rephrased = canonical_rephrased_pair()
    for original, twin in zip(canonical, rephrased, strict=True):
        assert twin.name == f"{original.name}__rephrased"
        assert twin.kind == original.kind
        assert twin.required_tokens == original.required_tokens
        assert twin.forbidden_patterns == original.forbidden_patterns
        assert twin.enforce_honesty == original.enforce_honesty
        assert twin.messages[-1]["content"] != original.messages[-1]["content"]


def test_rephrased_twins_fail_closed_on_unknown_task():
    from fx1.eval.suite import EvalTask

    orphan = EvalTask(
        name="uncurated",
        kind="domain",
        messages=[{"role": "user", "content": "q"}],
        required_tokens=["x"],
    )
    with pytest.raises(KeyError, match="uncurated"):
        rephrased_twins([orphan])


def test_gap_probe_detects_surface_memorizer():
    canonical, _ = canonical_rephrased_pair()
    canonical_prompts = {t.messages[-1]["content"] for t in canonical}

    def memorizer(messages):
        prompt = messages[-1]["content"]
        if prompt in canonical_prompts:
            # passes canonical: kitchen-sink contract response
            return (
                "proper crps pinball pit qlike kupiec vpin kyle walk-forward "
                "cpcv almgren next-open no live SYNTHETIC labeled fail-closed cannot "
                "backtest not live net cost survivorship multiple "
                "recovery brier ifind wind leakage sha256 receipt conformal "
                "spread impact"
            )
        return "I do not recognize this phrasing."

    probe = run_rephrased_gap(memorizer)
    assert probe.method == "rephrased_gap"
    assert probe.flagged  # 100% canonical pass, ~0% rephrased → gap 1.0
    assert probe.value and probe.value > probe.threshold


def test_gap_probe_calm_for_genuine_model():
    def genuine(messages):
        return (
            "proper crps pinball pit qlike kupiec vpin kyle walk-forward "
            "cpcv almgren next-open no live synthetic fail-closed cannot "
            "backtest not live net cost survivorship multiple SYNTHETIC labeled "
            "recovery brier ifind wind leakage sha256 receipt conformal "
            "spread impact"
        )

    probe = run_rephrased_gap(genuine)
    assert not probe.flagged
    assert probe.value == 0.0
