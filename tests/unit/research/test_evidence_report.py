import json

from quant_fund.reporting.report import build_evidence_report, write_evidence_report


def test_evidence_report_is_conservative_when_inputs_are_missing() -> None:
    report = build_evidence_report(candidates={})
    assert report["schema"] == "evidence_report.v1"
    assert report["status"] == "insufficient_evidence"
    assert "candidate_metrics_missing" in report["warnings"]
    assert report["research_only"] is True
    assert report["live_pnl_claim"] is False


def test_evidence_report_marks_complete_only_when_all_inputs_are_clean() -> None:
    report = build_evidence_report(
        candidates={"ranker": {"mean_ic": 0.1}},
        provenance={"data_source": "vendor", "artifact_sha256": "a" * 64, "manifest_valid": True},
        health={"status": "ok"},
        promotion={"promote": True},
    )
    assert report["status"] == "complete"
    assert report["warnings"] == []


def test_write_evidence_report_persists_json_and_markdown(tmp_path) -> None:
    paths = write_evidence_report(
        tmp_path,
        candidates={"ranker": {"mean_ic": 0.1}},
        provenance={"data_source": "vendor"},
        health={"status": "ok"},
        promotion={"promote": True},
    )
    assert paths["json"].is_file()
    assert paths["markdown"].is_file()
    assert json.loads(paths["json"].read_text())["schema"] == "evidence_report.v1"
    assert "Institutional Evidence Report" in paths["markdown"].read_text()
