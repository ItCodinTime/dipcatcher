"""Calibrated-uncertainty eval tests: determinism, ground truth, oracles, honesty."""

import math
import re

import pytest
from scipy.stats import binom, norm

from fx1.eval.calibration_eval import (
    FAMILIES,
    FAMILY_BINOMIAL_HITS,
    FAMILY_GAUSSIAN_TAIL,
    build_calibration_bank,
    extract_probability,
    parse_question_id,
    run_calibration_eval,
    synthetic_oracle,
)
from fx1.honesty import Fx1HonestyError, validate_fx1_output


def _bank():
    return build_calibration_bank(seed=0, n_questions=60)


# ---------------------------------------------------------------------------
# (a) bank determinism, SYNTHETIC labels, open-interval ground truth
# ---------------------------------------------------------------------------


def test_bank_builds_deterministically_with_labels_and_open_interval_truth():
    a = _bank()
    b = _bank()
    assert a == b
    assert len(a) == 60
    for family in FAMILIES:
        assert sum(1 for q in a if q.family == family) == 15
    for q in a:
        assert "SYNTHETIC" in q.prompt
        assert 0.0 < q.true_probability < 1.0
        assert parse_question_id(q.prompt) == q.question_id
        # Generated prompts must pass the house honesty contract (fail-closed).
        assert validate_fx1_output(q.prompt) == q.prompt


def test_n_questions_must_split_evenly():
    with pytest.raises(ValueError):
        build_calibration_bank(seed=0, n_questions=10)


# ---------------------------------------------------------------------------
# (b) ground truths match closed-form recomputation from the printed params
# ---------------------------------------------------------------------------


def _recomputed_gaussian(prompt: str) -> float:
    mu = float(re.search(r"mu = (-?[\d.]+)", prompt).group(1))
    sigma = float(re.search(r"sigma = (-?[\d.]+)", prompt).group(1))
    k = float(re.search(r"k = (-?[\d.]+)", prompt).group(1))
    z = (k - mu) / sigma
    if "exceeds" in prompt:
        return float(norm.sf(z))
    return float(norm.cdf(z))


def _recomputed_binomial(prompt: str) -> float:
    p_hit = float(re.search(r"hit rate of p = (-?[\d.]+)", prompt).group(1))
    k = int(re.search(r"at most k = (\d+)", prompt).group(1))
    m = int(re.search(r"in m = (\d+)", prompt).group(1))
    return float(binom.cdf(k, m, p_hit))


def test_ground_truth_matches_closed_form_recomputation():
    bank = _bank()
    checked = 0
    for q in bank:
        if q.family == FAMILY_GAUSSIAN_TAIL:
            expected = _recomputed_gaussian(q.prompt)
        elif q.family == FAMILY_BINOMIAL_HITS:
            expected = _recomputed_binomial(q.prompt)
        else:
            continue  # covered by the round-trip determinism + oracle tests
        assert abs(q.true_probability - expected) <= 1e-6, q.question_id
        checked += 1
    assert checked == 30


# ---------------------------------------------------------------------------
# (c) oracle calibration behavior
# ---------------------------------------------------------------------------


def test_true_oracle_is_calibrated_and_passes():
    report = run_calibration_eval(synthetic_oracle("true"), seed=0)
    assert report.n_unparseable == 0
    assert report.extracted.size == report.n_questions
    assert abs(report.ece) <= 0.03
    assert abs(report.spiegelhalter_z) <= 2.0
    assert report.passed


def test_miscalibrated_oracle_is_detected_and_fails():
    # The mandated shrinkage p -> 0.5 + 0.9 (p - 0.5) caps |f - p| at 0.05, so
    # its ECE (~0.1 * mean |p - 0.5| ~ 0.04) can never reach 0.10; the
    # meaningful gate is a tightened threshold, which it fails decisively.
    report = run_calibration_eval(synthetic_oracle("miscalibrated"), seed=0, ece_threshold=0.02)
    assert report.n_unparseable == 0
    assert 0.03 <= report.ece < 0.05
    assert not report.passed
    true_ece = run_calibration_eval(synthetic_oracle("true"), seed=0).ece
    assert report.ece > 10 * true_ece


# ---------------------------------------------------------------------------
# (d) garbage / crashing models: counted, never propagated
# ---------------------------------------------------------------------------


def _exploding_model(messages: list[dict[str, str]]) -> str:
    raise RuntimeError("model exploded")


def test_garbage_models_count_unparseable_and_fail_closed():
    bank = _bank()
    models = [
        lambda messages: "lorem ipsum, no numbers at all",  # noqa: E731
        lambda messages: "the answer is 1.75, definitely out of range",  # noqa: E731
        lambda messages: "maybe 0.5, maybe 0.9, or 2e6",  # noqa: E731
        _exploding_model,
    ]
    for model in models:
        report = run_calibration_eval(model, seed=0)
        assert report.n_unparseable == len(bank) == report.n_questions
        assert report.extracted.size == 0
        assert math.isnan(report.ece)
        assert math.isnan(report.spiegelhalter_z)
        assert not report.passed


# ---------------------------------------------------------------------------
# (e) honesty: a forbidden live-trading claim is flagged fail-closed
# ---------------------------------------------------------------------------


def test_forbidden_live_claim_trips_honesty_validation():
    bad = "We made $100,000 in live P&L last month with Sharpe 2.5 — trust it."
    with pytest.raises(Fx1HonestyError):
        validate_fx1_output(bad)
    # ...and both oracles answer every prompt with honesty-clean output.
    bank = _bank()
    for mode in ("true", "miscalibrated"):
        oracle = synthetic_oracle(mode)
        for q in bank:
            assert validate_fx1_output(oracle([{"role": "user", "content": q.prompt}]))


# ---------------------------------------------------------------------------
# (f) bin bookkeeping: counts partition the bank, ECE is internally consistent
# ---------------------------------------------------------------------------


def test_bins_partition_questions_and_ece_is_consistent():
    report = run_calibration_eval(synthetic_oracle("true"), seed=0)
    assert len(report.bins) == 10
    assert sum(b.count for b in report.bins) + report.n_unparseable == report.n_questions
    assert {round(b.upper - b.lower, 9) for b in report.bins} == {0.1}
    assert report.bins[0].lower == 0.0
    assert report.bins[-1].upper == 1.0
    nonempty = [b for b in report.bins if b.count]
    assert len(nonempty) >= 4  # tail-concentrated bank still covers the edges
    recomputed = sum(
        (b.count / report.n_questions) * abs(b.mean_forecast - b.observed_frequency)
        for b in nonempty
    )
    assert abs(recomputed - report.ece) <= 1e-12


# ---------------------------------------------------------------------------
# (g) seed variation changes the bank
# ---------------------------------------------------------------------------


def test_seed_variation_changes_bank():
    a = _bank()
    b = build_calibration_bank(seed=1)
    assert [q.prompt for q in a] != [q.prompt for q in b]
    assert [q.true_probability for q in a] != [q.true_probability for q in b]


# ---------------------------------------------------------------------------
# extraction units
# ---------------------------------------------------------------------------


def test_extract_probability_units():
    assert extract_probability("The probability is 0.025.") == pytest.approx(0.025)
    assert extract_probability("about 5%") == pytest.approx(0.05)
    assert extract_probability("p = 9.6e-2, so roughly 0.096") == pytest.approx(0.096)
    assert extract_probability("first 0.4 then 0.9") == pytest.approx(0.9)  # last number wins
    assert extract_probability("no numbers here") is None
    assert extract_probability("") is None
    assert extract_probability("the answer is 1.75") is None
    assert extract_probability("maybe -0.3") is None
