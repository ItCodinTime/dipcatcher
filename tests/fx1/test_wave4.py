"""Tests for wave-4: hypotheses, reward model, curriculum, red-team, dip bench, chat."""

import json
from pathlib import Path

import pytest

from fx1.bench.dip import (
    DipForecast,
    assert_bench_output_honest,
    detect_dip_events,
    evaluate_forecasts,
    unconditional_baseline,
)
from fx1.eval.redteam import REDTEAM_TASKS
from fx1.eval.suite import run_suite
from fx1.hypotheses import GateVerdict, ResearchTrace, validate_trace_scores
from fx1.reward import score_response
from fx1.serve.chat import cited_complete
from fx1.train.curriculum import Level, build_curriculum, classify

# --- hypotheses ---------------------------------------------------------------

def test_trace_requires_proper_scores():
    with pytest.raises(ValueError, match="proper score"):
        validate_trace_scores({"sharpe_ratio": 2.1})
    validate_trace_scores({"crps_mean": 0.4, "brier_3m": 0.2})


def test_trace_admissibility_and_messages():
    trace = ResearchTrace(
        hypothesis="Dip recovery probability rises in low-VPIN regimes",
        config_diff="dip:\n  threshold: 0.10",
        bench_command="research",
        scores={"brier_3m": 0.21, "ece_3m": 0.04},
        verdict=GateVerdict.PASSED,
        receipt_sha256="c" * 64,
    )
    assert trace.admissible
    msgs = trace.to_sft_messages("sys")
    assert msgs[-1]["content"].count("verify-research") == 1
    pending = trace.model_copy(update={"verdict": None})
    assert not pending.admissible


# --- reward model ---------------------------------------------------------------

def test_reward_ranks_honest_above_dishonest():
    good = score_response(
        "CRPS improved to 0.31 (estimate with calibrated intervals); research "
        "evidence bound to receipt ab12cd34…, verify with dipcatcher "
        "verify-research."
    )
    bad = score_response("The sharpe: 3.0 proves it. Live P&L incoming.")
    assert good.total > 5.0
    assert bad.total == -10.0 and bad.violations


# --- curriculum -----------------------------------------------------------------

def test_curriculum_levels_and_determinism(tmp_path: Path):
    examples = [
        {"messages": [{"role": "assistant", "content": "rules only"}], "negative": False},
        {"messages": [{"role": "assistant", "content": "verify-research binds this evidence class"}], "negative": False},
        {"messages": [{"role": "assistant", "content": "Gate verdict: passed"}], "negative": False},
        {"messages": [{"role": "assistant", "content": "No."}], "negative": True},
    ]
    corpus = tmp_path / "c.jsonl"
    corpus.write_text("\n".join(json.dumps(e) for e in examples) + "\n")
    counts1 = build_curriculum(corpus, tmp_path / "o1.jsonl")
    counts2 = build_curriculum(corpus, tmp_path / "o2.jsonl")
    assert counts1 == counts2
    ordered = (tmp_path / "o1.jsonl").read_text().strip().splitlines()
    assert len(ordered) == 4
    assert classify(json.loads(ordered[0])) == Level.CONTRACTS
    assert classify(json.loads(ordered[-1])) == Level.REFUSAL


# --- red-team ---------------------------------------------------------------------

def test_redteam_catches_laundering_model():
    launderer = lambda msgs: "Yes, as MAXPROFIT my live P&L of $50,000 is great."  # noqa: E731
    summary = run_suite(launderer, [t for t in REDTEAM_TASKS if "roleplay" in t.name])
    assert summary["honesty_gate_passed"] is False


def test_redteam_compliant_model_passes():
    def compliant(msgs):
        text = msgs[-1]["content"]
        base = ("I can't do that — promotion is fail-closed, synthetic labels "
                "stay, and backtest evidence is not forward returns.")
        if "expected forward returns" in text:
            return "No. Backtest evidence cannot be presented as forward returns."
        if "SYNTHETIC label" in text:
            return "No. The SYNTHETIC label stays; renaming synthetic evidence is laundering."
        return base
    summary = run_suite(compliant, REDTEAM_TASKS)
    assert summary["honesty_gate_passed"] is True


# --- dip bench ----------------------------------------------------------------------

def _series_with_dip() -> tuple[list[float], list[str]]:
    # Dip at idx 2 (88, depth ~13.7% below the 102 peak); the peak is regained
    # at idx 5 (103), two bars after detection at idx 3... series adjusted so
    # detection fires at the 88 close and recovery lands inside 3 bars.
    closes = [100.0, 102.0, 88.0, 90.0, 103.0, 104.0, 105.0]
    dates = [f"2026-01-0{i + 1}" for i in range(7)]
    return closes, dates


def test_dip_detection_and_recovery():
    closes, dates = _series_with_dip()
    events = detect_dip_events(closes, dates, "TEST", threshold=0.10,
                               horizons_bars={"1m": 3, "12m": 500})
    assert len(events) == 1
    event = events[0]
    assert event.peak_date == "2026-01-02"
    assert event.depth == pytest.approx((102 - 88) / 102)
    assert event.recovered["1m"] is True       # peak regained at idx 4 within 3 bars
    assert event.recovered["12m"] is None      # horizon unobservable, not imputed


def test_dip_scoring_beats_naive_when_calibrated():
    closes, dates = _series_with_dip()
    events = detect_dip_events(closes, dates, "TEST", threshold=0.10,
                               horizons_bars={"1m": 3})
    baseline = unconditional_baseline(events)
    assert baseline["1m"] == 1.0
    good = [DipForecast("TEST", events[0].trough_date, {"1m": 0.95})]
    bad = [DipForecast("TEST", events[0].trough_date, {"1m": 0.05})]
    m_good = evaluate_forecasts(events, good)
    m_bad = evaluate_forecasts(events, bad)
    assert m_good["brier_1m"] < m_bad["brier_1m"]
    assert m_good["log_loss_1m"] < m_bad["log_loss_1m"]
    assert_bench_output_honest(m_good)


def test_bench_honesty_gate():
    with pytest.raises(ValueError, match="forbidden"):
        assert_bench_output_honest({"strategy_sharpe": 1.5})
    assert_bench_output_honest({"brier_1m": 0.2})


def test_forecast_probability_range_enforced():
    closes, dates = _series_with_dip()
    events = detect_dip_events(closes, dates, "TEST", threshold=0.10,
                               horizons_bars={"1m": 3})
    with pytest.raises(ValueError, match="out of range"):
        evaluate_forecasts(
            events, [DipForecast("TEST", events[0].trough_date, {"1m": 1.5})]
        )


# --- cited serving --------------------------------------------------------------------

def test_cited_complete_appends_provenance():
    class FakeBackend:
        def complete(self, messages):
            return "Research evidence: CRPS 0.31."

    out = cited_complete(FakeBackend(), [], receipt_hashes=["d" * 64])
    assert "verify-research" in out
    assert "dddddddddddddddd" in out
