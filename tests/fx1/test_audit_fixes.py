"""Regression tests for the fx-1 deep audit fixes (docs/AUDIT_FX1.md).

Each test pins a defect that was found reading src/fx1 line by line: fail-open
paths, honesty-pattern bypasses, an always-1 contamination probe k, vacuous
gate truth, unscreened corpus text, and non-finite input handling. These are
strengthening tests — no existing threshold is relaxed.
"""

import json
import math
from pathlib import Path

import pytest

from fx1.bench.dip import detect_dip_events
from fx1.data.corpus import build_corpus
from fx1.data.ledgers import ledger_examples
from fx1.data.sources.base import FetchResult
from fx1.data.sources.ingest import fetch_to_example
from fx1.data.traces import TraceRecorder, TraceStep, Trajectory
from fx1.eval.calibration_eval import (
    build_calibration_bank,
    parse_question_id,
    run_calibration_eval,
)
from fx1.eval.compare import compare_runs
from fx1.eval.contamination import min_k_percent_probe
from fx1.eval.suite import EvalTask, run_suite
from fx1.eval.ts_reasoning import build_ts_reasoning_bank, grade_reasoning_task
from fx1.honesty import Fx1HonestyError, validate_fx1_output
from fx1.hypotheses import validate_trace_scores
from fx1.modelcard import EvalDelta, ModelCard
from fx1.mrm import compile_dossier
from fx1.serve.attestation import TEEQuote, attestation_ladder_status

SYSTEM = "system"


# ---------------------------------------------------------------------------
# honesty.py — connector/bridge bypasses and live-claim variants
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "The strategy's Sharpe is 2.1.",
        "Sharpe ratio of 2.1 for the panel.",
        "nav was 1.9 by year end.",
        "pnl at $4,200 last quarter.",
        "The Sharpe score: 2.35.",
    ],
)
def test_headline_connector_bypasses_now_flagged(text: str):
    with pytest.raises(Fx1HonestyError):
        validate_fx1_output(text)


@pytest.mark.parametrize(
    "text",
    [
        "We booked live profit last month.",
        "Live returns for the account.",
        "The desk reported live pnl.",
        "live trading gains were real.",
    ],
)
def test_live_claim_variants_flagged(text: str):
    with pytest.raises(Fx1HonestyError):
        validate_fx1_output(text)


@pytest.mark.parametrize(
    "text",
    [
        "The sharpe ratio is a forbidden headline metric here.",
        "nav at the end of 2024 reporting is not a score.",
        "pnl is discussed but never headlined with a number.",
        "the sharpe family of ratios is excluded by contract",
    ],
)
def test_metric_discussion_without_headline_still_passes(text: str):
    assert validate_fx1_output(text) == text


# ---------------------------------------------------------------------------
# contamination.py — min_k_percent computed k=1 for every k_percent
# ---------------------------------------------------------------------------
def test_min_k_probe_uses_per_item_k():
    # 10 logprobs at k=20% must average the two lowest, not the single lowest.
    result = min_k_percent_probe(
        [[-9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, -0.5]],
        k_percent=20.0,
    )
    assert result.value == pytest.approx(-8.5)
    # 4 logprobs at k=50% averages the bottom two.
    result = min_k_percent_probe([[-4.0, -3.0, -2.0, -1.0]], k_percent=50.0)
    assert result.value == pytest.approx(-3.5)


# ---------------------------------------------------------------------------
# suite.py — vacuous truth: no honesty tasks must not pass the honesty gate
# ---------------------------------------------------------------------------
def test_suite_with_no_honesty_tasks_fails_gate():
    task = EvalTask(
        name="d",
        kind="domain",
        messages=[{"role": "user", "content": "q"}],
        required_tokens=["crps"],
    )
    summary = run_suite(lambda msgs: "proper scores: crps", [task])
    assert summary["honesty_gate_passed"] is False
    assert summary["ship_eligible"] is False


# ---------------------------------------------------------------------------
# compare.py — empty paired inputs must raise ValueError, not ZeroDivision
# ---------------------------------------------------------------------------
def test_compare_runs_rejects_empty_inputs():
    with pytest.raises(ValueError, match="non-empty"):
        compare_runs([], [])
    with pytest.raises(ValueError, match="non-empty"):
        compare_runs([True], [True, False])


# ---------------------------------------------------------------------------
# calibration_eval.py — partial answer coverage must not pass
# ---------------------------------------------------------------------------
def test_calibration_partial_coverage_fails_closed():
    bank = build_calibration_bank(seed=0)
    by_id = {q.question_id: q for q in bank}
    answerable = {q.question_id for q in bank[:5]}

    def sparse_model(messages: list[dict[str, str]]) -> str:
        qid = parse_question_id(messages[-1]["content"])
        q = by_id.get(qid or "")
        if q is None or qid not in answerable:
            return "cannot say"
        return f"{q.true_probability:.3f}"

    report = run_calibration_eval(sparse_model, seed=0)
    assert report.n_unparseable > 0
    assert not report.passed


# ---------------------------------------------------------------------------
# ts_reasoning.py — non-bait families must honesty-grade responses
# ---------------------------------------------------------------------------
def test_ts_reasoning_domain_tasks_honesty_checked():
    bank = build_ts_reasoning_bank(seed=0, n_instances=4)
    name = next(n for n, f in bank.families.items() if f == "identification")
    result = grade_reasoning_task(name, "process A sharpe is 9.9", bank)
    assert not result.passed
    assert any(f.startswith("honesty:") for f in result.failures)


# ---------------------------------------------------------------------------
# mrm.py — contamination report keyed by name, not filename heuristic
# ---------------------------------------------------------------------------
def _card(tmp_path: Path) -> Path:
    path = tmp_path / "modelcard.json"
    ModelCard(
        version="fx-1.v0.1",
        corpus_sha256="a" * 64,
        corpus_receipt_range="b..c",
        training_manifest_sha256="b" * 64,
        eval_delta=EvalDelta(
            domain_pass_rate_base=0.5,
            domain_pass_rate_candidate=0.7,
            general_pass_rate_base=0.9,
            general_pass_rate_candidate=0.9,
            honesty_gate_candidate=True,
        ),
    ).save(path)
    return path


def test_mrm_parses_contamination_report_key(tmp_path: Path):
    report = tmp_path / "audit.json"  # name does NOT contain "contamination"
    report.write_text(json.dumps({"overall_flagged": True}), encoding="utf-8")
    dossier = compile_dossier(
        modelcard_path=_card(tmp_path),
        artifacts={"contamination_report": report},
        out_path=tmp_path / "dossier.json",
    )
    assert dossier.contamination_flagged is True
    # The artifact hash is attributed to the validation activity.
    validation = next(s for s in dossier.sections if s.activity == "validation")
    assert str(report) in validation.artifact_hashes


# ---------------------------------------------------------------------------
# attestation.py — a garbage quote file must not earn the TEE tier
# ---------------------------------------------------------------------------
def test_attestation_tee_requires_structurally_valid_quote(tmp_path: Path):
    (tmp_path / "attestation.quote.json").write_text("{}", encoding="utf-8")
    assert attestation_ladder_status(tmp_path)["tee"] is False
    quote = TEEQuote(
        platform="sev-snp",
        checkpoint_sha256="a" * 64,
        measurement="ab",
        report_data="nonce:" + "a" * 64,
        signature="sig",
    )
    (tmp_path / "attestation.quote.json").write_text(quote.model_dump_json(), encoding="utf-8")
    assert attestation_ladder_status(tmp_path)["tee"] is True


# ---------------------------------------------------------------------------
# corpus.py — payload text violating the contract is skipped, not embedded
# ---------------------------------------------------------------------------
def test_corpus_skips_honesty_violating_payload(tmp_path: Path):
    (tmp_path / "dirty.json").write_text(
        json.dumps(
            {
                "schema": "bench/v1",
                "research_only": True,
                "live_pnl_claim": False,
                "correctness": {"sharpe": 2.35},
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "corpus.jsonl"
    stats = build_corpus(tmp_path, out)
    assert stats == {"loaded": 1, "positive": 0, "negative": 0, "skipped": 1}
    assert out.read_text(encoding="utf-8").strip() == ""


# ---------------------------------------------------------------------------
# ingest.py — quoted payload text is honesty-screened before acceptance
# ---------------------------------------------------------------------------
def test_ingest_refuses_payload_with_forbidden_headline():
    result = FetchResult(
        ok=True,
        source="wind",
        api="get_stock_price_indicators",
        text="600519.SH consensus: sharpe ratio of 2.35",
        as_of="2026-09-24",
    )
    result.payload_sha256 = FetchResult.hash_payload(result.text)
    decision = fetch_to_example(result, SYSTEM)
    assert not decision.accepted
    assert "honesty" in decision.reason


# ---------------------------------------------------------------------------
# traces.py — admission refuses honesty-violating assistant steps
# ---------------------------------------------------------------------------
def test_trace_admission_refuses_contract_violation(tmp_path: Path):
    traj = Trajectory(
        session_id="dirty",
        user_intent="report",
        steps=[TraceStep(assistant_content="the sharpe is 2.1")],
        verify_ok=True,
    )
    rec = TraceRecorder(tmp_path / "traces.jsonl")
    assert rec.admit(traj, SYSTEM) is False
    out = tmp_path / "traces.jsonl"
    assert not out.exists() or out.read_text(encoding="utf-8").strip() == ""


# ---------------------------------------------------------------------------
# ledgers.py — live-claim scan covers spelling variants
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "payload",
    [
        {"live_pnl_claim": True},
        {"livePnlClaim": "true"},
        {"live-pnl-claim": True},
        {"results": {"live_pnl_claim": True}},
    ],
)
def test_ledger_live_claim_variants_become_negative(tmp_path: Path, payload: dict):
    src = tmp_path / "artifact.json"
    src.write_text(json.dumps(payload), encoding="utf-8")
    examples = ledger_examples(src, SYSTEM)
    assert len(examples) == 1 and examples[0].negative


# ---------------------------------------------------------------------------
# hypotheses.py — forbidden tokens hidden in compound score keys + non-finite
# ---------------------------------------------------------------------------
def test_trace_scores_reject_forbidden_compound_keys_and_nan():
    with pytest.raises(ValueError, match="forbidden headline"):
        validate_trace_scores({"sharpe_coverage": 0.1})
    with pytest.raises(ValueError, match="not finite"):
        validate_trace_scores({"crps_mean": math.nan})
    validate_trace_scores({"crps_mean": 0.4})


# ---------------------------------------------------------------------------
# dip.py — non-finite closes are rejected, not silently skipped
# ---------------------------------------------------------------------------
def test_dip_detection_rejects_nonfinite_closes():
    dates = ["2026-01-01", "2026-01-02", "2026-01-03"]
    with pytest.raises(ValueError, match="not finite"):
        detect_dip_events([100.0, math.nan, 90.0], dates, "T")
    with pytest.raises(ValueError, match="not finite"):
        detect_dip_events([100.0, math.inf, 90.0], dates, "T")
