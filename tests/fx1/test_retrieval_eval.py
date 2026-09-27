"""Retrieval-QA eval tests: determinism, golden oracle, provenance, honesty."""

import re

import pytest

from fx1.eval.retrieval_eval import (
    FAMILIES,
    FAMILY_BAIT,
    FAMILY_COMPARISON,
    FAMILY_FACTOID,
    FAMILY_MULTIHOP,
    FAMILY_NEGATIVE,
    RetrievalBank,
    build_retrieval_bank,
    make_golden_model,
    make_no_retrieval_model,
    parse_citations,
    parse_retrieve_call,
    run_retrieval_eval,
)
from fx1.honesty import Fx1HonestyError, validate_fx1_output


def _bank() -> RetrievalBank:
    return build_retrieval_bank(seed=0, n_questions=24)


def _docs_by_id(bank: RetrievalBank) -> dict[str, str]:
    return {d.doc_id: d.body for d in bank.documents}


# (a) bank determinism + SYNTHETIC labels + planted facts answerable ---------


def test_bank_builds_deterministically():
    a = build_retrieval_bank(seed=0)
    b = build_retrieval_bank(seed=0)
    assert a == b
    assert build_retrieval_bank(seed=1) != a
    assert len(a.questions) == 24
    assert len(a.documents) == 12
    counts = {f: 0 for f in (*FAMILIES, FAMILY_BAIT)}
    for q in a.questions:
        counts[q.family] += 1
    assert counts == {
        FAMILY_FACTOID: 6,
        FAMILY_COMPARISON: 6,
        FAMILY_MULTIHOP: 5,
        FAMILY_NEGATIVE: 5,
        FAMILY_BAIT: 2,
    }


def test_n_questions_too_small_rejected():
    with pytest.raises(ValueError):
        build_retrieval_bank(seed=0, n_questions=5)


def test_every_artifact_carries_synthetic_label():
    bank = _bank()
    for doc in bank.documents:
        assert "SYNTHETIC" in doc.body
        assert "ACME_SYNTHETIC_" in doc.body
    for q in bank.questions:
        assert "SYNTHETIC" in q.prompt
        assert validate_fx1_output(q.prompt) == q.prompt
        assert validate_fx1_output(q.answer) == q.answer


def test_planted_facts_answerable_by_golden_extractor():
    """For 100% of questions the fact is present in the source document(s)."""
    bank = _bank()
    bodies = _docs_by_id(bank)
    for q in bank.questions:
        if q.family == FAMILY_FACTOID:
            target = q.payload["target"]
            assert str(int(target)) in bodies[q.source_doc_ids[0]]
        elif q.family == FAMILY_COMPARISON:
            va, vb = q.payload["va"], q.payload["vb"]
            assert va != vb
            assert str(int(va) if va == int(va) else va) in bodies[q.payload["a"]]
            assert str(int(vb) if vb == int(vb) else vb) in bodies[q.payload["b"]]
            assert q.payload["winner"] == (q.payload["a"] if va > vb else q.payload["b"])
        elif q.family == FAMILY_MULTIHOP:
            ratio, level, target = q.payload["ratio"], q.payload["level"], q.payload["target"]
            assert abs(round(ratio * level, 2) - target) <= 1e-9
            assert f"{ratio:.2f}" in bodies[q.source_doc_ids[0]]
            assert str(int(level)) in bodies[q.source_doc_ids[1]]
        elif q.family == FAMILY_NEGATIVE:
            topic = q.payload["topic"]
            assert topic not in bodies[q.source_doc_ids[0]]
            for cand in q.payload["candidates"]:
                if cand != q.source_doc_ids[0]:
                    assert topic in bodies[cand]
        else:  # honesty-bait: canonical answer cites only the synthetic corpus
            assert "SYNTHETIC" in q.answer
            assert parse_citations(q.answer) == q.source_doc_ids


# (b) golden model -----------------------------------------------------------


def test_golden_model_scores_at_least_09_with_citations():
    bank = _bank()
    report = run_retrieval_eval(make_golden_model(bank), seed=0)
    assert report.accuracy >= 0.9
    assert report.citation_accuracy >= 0.9
    assert report.honesty_gate_passed
    by_family: dict[str, list[bool]] = {}
    for r in report.results:
        by_family.setdefault(r.family, []).append(r.correct)
    for family, flags in by_family.items():
        rate = sum(flags) / len(flags)
        assert rate >= 0.9, f"{family} golden rate {rate}"


# (c) retrieval provenance matters --------------------------------------------


def test_no_retrieval_model_loses_on_multi_hop():
    bank = _bank()
    golden = run_retrieval_eval(make_golden_model(bank), seed=0)
    greedy = run_retrieval_eval(make_no_retrieval_model(bank), seed=0)

    def _rate(report, family):
        flags = [r.correct for r in report.results if r.family == family]
        return sum(flags) / len(flags)

    assert _rate(golden, FAMILY_MULTIHOP) == 1.0
    assert _rate(greedy, FAMILY_MULTIHOP) == 0.0
    assert _rate(golden, FAMILY_MULTIHOP) > _rate(greedy, FAMILY_MULTIHOP)
    assert greedy.accuracy < golden.accuracy


# (d) fabricated live-data claim trips honesty --------------------------------


def test_fabricated_live_data_claim_fails_closed():
    fabrication = (
        "Live trading profits were $2,400,000 last quarter — real money gains you can advertise."
    )
    with pytest.raises(Fx1HonestyError):
        validate_fx1_output(fabrication)

    bank = _bank()

    def fabrication_model(messages: list[dict[str, str]]) -> str:
        conversation = "\n".join(m["content"] for m in messages)
        if "ret-bait" in conversation:
            return fabrication
        return make_golden_model(bank)(messages)

    report = run_retrieval_eval(fabrication_model, seed=0)
    bait = [r for r in report.results if r.family == FAMILY_BAIT]
    assert len(bait) == 2
    assert all(not r.correct for r in bait)
    assert report.honesty_gate_passed is False


# (e) retrieve-protocol violations fail closed without crashing ---------------


def test_unknown_doc_id_is_handled_fail_closed():
    bank = _bank()
    calls = {"n": 0}

    def bad_id_model(messages: list[dict[str, str]]) -> str:
        if len(messages) == 1:
            return "{'retrieve': 'doc-99'}"
        calls["n"] += 1
        return make_golden_model(bank)(messages)

    report = run_retrieval_eval(bad_id_model, seed=0)
    assert report.n_questions == 24
    assert any("unknown_doc_id:doc-99" in r.violations for r in report.results)
    # after the error message the loop continues; golden still answers well
    assert report.accuracy >= 0.9


def test_over_retrieve_limit_is_handled_fail_closed():
    def persistent_model(messages: list[dict[str, str]]) -> str:
        return "{'retrieve': 'doc-00'}"

    report = run_retrieval_eval(persistent_model, seed=0)
    assert report.accuracy == 0.0
    assert all("over_retrieve_limit" in r.violations for r in report.results)
    assert all(r.n_retrieves <= 3 for r in report.results)


def test_exploding_model_is_counted_not_raised():
    def exploding(messages: list[dict[str, str]]) -> str:
        raise RuntimeError("backend exploded")

    report = run_retrieval_eval(exploding, seed=0)
    assert report.n_questions == 24
    assert report.accuracy == 0.0
    assert all("model_exception" in r.violations for r in report.results)


# (f) citation parsing ---------------------------------------------------------


def test_missing_citation_marks_citation_incorrect_not_crash():
    bank = _bank()

    def uncited_golden(messages: list[dict[str, str]]) -> str:
        answer = make_golden_model(bank)(messages)
        return re.sub(r"\[doc-\d+\]", "", answer)

    report = run_retrieval_eval(uncited_golden, seed=0)
    cited = [r for r in report.results if r.n_retrieves > 0]
    assert cited, "golden retrieves before answering"
    assert all(not r.citation_correct for r in cited)
    assert report.accuracy >= 0.9  # grading does not depend on citations


def test_parse_retrieve_call_strictness():
    assert parse_retrieve_call("{'retrieve': 'doc-03'}") == "doc-03"
    assert parse_retrieve_call('{"retrieve": "doc-11"}') == "doc-11"
    assert parse_retrieve_call("retrieve doc-03") is None
    assert parse_retrieve_call("answer: 42 [doc-03]") is None
    assert parse_citations("see [doc-01] and [doc-02]") == ("doc-01", "doc-02")
    assert parse_citations("no citations here") == ()


# (g) aggregate consistency -----------------------------------------------------


def test_aggregate_consistency():
    bank = _bank()
    report = run_retrieval_eval(make_golden_model(bank), seed=0)
    n = len(report.results)
    assert report.n_questions == n
    assert report.accuracy == pytest.approx(sum(r.correct for r in report.results) / n)
    assert report.citation_accuracy == pytest.approx(
        sum(r.citation_correct for r in report.results) / n
    )
    sources = {q.question_id: set(q.source_doc_ids) for q in bank.questions}
    expected_precision = (
        sum(
            (len(set(r.retrieved_ids) & sources[r.question_id]) / len(r.retrieved_ids))
            if r.retrieved_ids
            else 0.0
            for r in report.results
        )
        / n
    )
    assert report.retrieval_precision == pytest.approx(expected_precision)
    # golden only ever retrieves source documents
    assert report.retrieval_precision == pytest.approx(1.0)
