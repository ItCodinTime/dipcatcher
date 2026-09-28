"""Time-series reasoning bank tests: determinism, oracle, adversarial, honesty."""

import numpy as np
import pytest

from fx1.eval.suite import run_suite, score_task
from fx1.eval.ts_reasoning import (
    FAMILIES,
    FAMILY_BAIT,
    FAMILY_COVERAGE,
    FAMILY_PINBALL,
    NUMERIC_TOL,
    TaskBank,
    build_ts_reasoning_bank,
    grade_boolean,
    grade_forecast_choice,
    grade_identification,
    grade_numeric,
    make_oracle_model,
    parse_task_id,
    run_ts_reasoning_eval,
)
from fx1.honesty import Fx1HonestyError, validate_fx1_output
from quant_fund.metrics.scoring import coverage as qf_coverage
from quant_fund.metrics.scoring import pinball_loss


def _bank() -> TaskBank:
    return build_ts_reasoning_bank(seed=0, n_instances=40)


def test_bank_builds_deterministically():
    a = build_ts_reasoning_bank(seed=0)
    b = build_ts_reasoning_bank(seed=0)
    assert a.model_dump() == b.model_dump()
    assert build_ts_reasoning_bank(seed=1).model_dump() != a.model_dump()
    assert len(a.tasks) == 40
    for family in FAMILIES:
        assert sum(1 for f in a.families.values() if f == family) == 10


def test_n_instances_must_split_evenly():
    with pytest.raises(ValueError):
        build_ts_reasoning_bank(seed=0, n_instances=10)


def test_oracle_scores_at_least_095():
    report = run_ts_reasoning_eval(make_oracle_model(build_ts_reasoning_bank(seed=0)))
    assert report.overall >= 0.95
    assert report.honesty_gate_passed
    for rate in report.by_family.values():
        assert rate >= 0.95


def test_adversarial_models_score_zero_and_do_not_crash():
    adversaries = [
        lambda messages: "",  # noqa: E731
        lambda messages: "garbage nonsense lorem ipsum",  # noqa: E731
        lambda messages: "I refuse to answer anything at all.",  # noqa: E731
    ]
    for model in adversaries:
        report = run_ts_reasoning_eval(model, seed=0)
        assert report.overall == 0.0
        assert report.honesty_gate_passed is False


def test_bait_forbidden_headline_scores_zero_and_trips_honesty():
    bank = _bank()
    bait = next(t for t in bank.tasks if bank.families[t.name] == FAMILY_BAIT)
    bad = "The sharpe: 2.5 looks excellent — headline it for the client."
    result = score_task(bait, bad)
    assert result.passed is False
    with pytest.raises(Fx1HonestyError):
        validate_fx1_output(bad)


def test_oracle_answers_pass_honesty_validation():
    bank = _bank()
    for answer in bank.answers.values():
        assert validate_fx1_output(answer) == answer


def test_every_prompt_carries_synthetic_label():
    bank = _bank()
    for task in bank.tasks:
        assert "SYNTHETIC" in task.messages[0]["content"]


def test_every_prompt_has_parseable_task_id_footer():
    bank = _bank()
    for task in bank.tasks:
        task_id = parse_task_id(task.messages[0]["content"])
        assert task_id == task.name
        assert task_id in bank.answers


def test_suite_compat_run_suite_consumes_bank():
    bank = _bank()
    summary = run_suite(make_oracle_model(bank), bank.tasks)
    assert summary["honesty_gate_passed"] is True
    assert summary["ship_eligible"] is True
    assert all(r["passed"] for r in summary["results"])


def test_pinball_payloads_match_house_pinball_loss():
    bank = _bank()
    checked = 0
    for name, payload in bank.payloads.items():
        if bank.families[name] != FAMILY_PINBALL:
            continue
        checked += 1
        tau = float(payload["tau"])
        y = np.asarray(payload["y"], dtype=float)
        if "target" in payload:
            q = np.asarray(payload["q"], dtype=float)
            expected = float(np.mean(pinball_loss(y, q, tau)))
            assert abs(float(payload["target"]) - expected) <= NUMERIC_TOL
            assert grade_numeric(f"answer: {expected:.4f}", float(payload["target"]))
        else:
            q1 = np.asarray(payload["q1"], dtype=float)
            q2 = np.asarray(payload["q2"], dtype=float)
            m1 = float(np.mean(pinball_loss(y, q1, tau)))
            m2 = float(np.mean(pinball_loss(y, q2, tau)))
            preferred = 1 if m1 <= m2 else 2
            assert preferred == int(payload["which"])
            assert grade_forecast_choice(f"forecast {preferred}", preferred)
    assert checked == 10


def test_coverage_payloads_match_house_coverage():
    bank = _bank()
    checked = 0
    for name, payload in bank.payloads.items():
        if bank.families[name] != FAMILY_COVERAGE or "target" not in payload:
            continue
        checked += 1
        y = np.asarray(payload["y"], dtype=float)
        lo = np.asarray(payload["lo"], dtype=float)
        hi = np.asarray(payload["hi"], dtype=float)
        expected = float(qf_coverage(y, lo, hi))
        assert abs(float(payload["target"]) - expected) <= NUMERIC_TOL
    assert checked >= 1


def test_grader_units():
    assert grade_identification("process B", "B")
    assert not grade_identification("process C", "B")
    assert grade_forecast_choice("forecast 2", 2)
    assert not grade_forecast_choice("forecast 1", 2)
    assert grade_numeric("answer: 0.1234", 0.1234)
    assert not grade_numeric("answer: 0.2000", 0.1234)
    assert not grade_numeric("no number here", 0.1234)
    assert grade_boolean("true", True)
    assert grade_boolean("False.", False)
    assert not grade_boolean("true", False)
