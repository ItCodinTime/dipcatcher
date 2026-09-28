"""The findings page copies recorded trial figures and leaves the rest blank."""

from __future__ import annotations

import json
import re
from pathlib import Path

from scripts.research_findings import (
    NOT_IN_ARTIFACT,
    PENDING,
    SURVIVORSHIP_PR,
    BatchRow,
    assert_pending_has_no_measured_figures,
    collect_batches,
    render,
)

from quant_fund.research.catalog import FORBIDDEN_RESEARCH_METRIC_KEYS

_ROOT = Path(__file__).resolve().parents[3]
_PAGE = _ROOT / "docs" / "research" / "findings.md"


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _reality_payload(*, dsr: object = 0.5, pbo: object = None) -> dict[str, object]:
    return {
        "schema": "dipcatcher.reality_sweep_receipt.v1",
        "study_id": "fixture-study",
        "live_pnl_claim": False,
        "selected_trial_id": "trial-b",
        "deflated_probability_raw_count": 0.25,
        "pbo": {"pbo": 0.4},
        "reality_gate": {
            "created_utc": "2026-01-02T00:00:00+00:00",
            "n_trials": 2,
            "n_effective_trials": 2.0,
            "dsr": dsr,
            "pbo": pbo,
            "best_trial_id": "trial-b",
            "verdict": "deflated",
        },
        "trials": [
            {
                "trial_id": "trial-a",
                "strategy": "searched",
                "windows": {"validation": {"periodic_ratio": 0.1}},
            },
            {
                "trial_id": "trial-b",
                "strategy": "baseline_book",
                "windows": {"validation": {"periodic_ratio": 0.2}},
            },
        ],
    }


def test_missing_dsr_is_not_filled_and_null_pbo_stays_null(tmp_path: Path) -> None:
    receipt = tmp_path / "research" / "reality" / "studies" / "fixture-study" / "receipt.json"
    payload = _reality_payload()
    gate = payload["reality_gate"]
    assert isinstance(gate, dict)
    del gate["dsr"]
    _write(receipt, payload)
    _write(
        receipt.with_name("preregistration.json"),
        {
            "reality_filter": {"dsr_pass": 0.95},
            "why_these_strategies": {"baseline_book": "One unswept baseline."},
        },
    )
    rows, _notes = collect_batches(tmp_path)
    recorded = [row for row in rows if not row.pending]
    assert len(recorded) == 1
    row = recorded[0]
    assert row.dsr_vs_bar.startswith("gate dsr not in artifact; bar 0.95")
    assert "deflated_probability_raw_count 0.25" in row.dsr_vs_bar
    assert "reality_gate.pbo null" in row.pbo
    assert "cscv pbo 0.4" in row.pbo
    assert "0.2" in row.best_vs_baselines
    assert "0.1" in row.best_vs_baselines
    assert "preregistration calls baseline_book a baseline" in row.best_vs_baselines
    assert row.verdict.startswith("verdict deflated")


def test_pending_draft_has_no_figures_until_a_survivorship_receipt_exists(tmp_path: Path) -> None:
    rows, _notes = collect_batches(tmp_path)
    assert_pending_has_no_measured_figures(rows)
    pending = [row for row in rows if row.batch == SURVIVORSHIP_PR]
    assert len(pending) == 1
    assert pending[0].date == PENDING
    for column in ("trials", "dsr_vs_bar", "pbo", "best_vs_baselines", "verdict", "artifact"):
        assert getattr(pending[0], column) == PENDING
        assert not any(character.isdigit() for character in getattr(pending[0], column))

    receipt = tmp_path / "research" / "reality" / "survivorship" / "receipt.json"
    payload = _reality_payload(dsr=0.61)
    payload["study_id"] = "survivorship-fixture"
    _write(receipt, payload)
    rows, _notes = collect_batches(tmp_path)
    assert not any(row.batch == SURVIVORSHIP_PR for row in rows)
    survivorship = [
        row for row in rows if "survivorship" in row.batch or "survivorship" in row.artifact
    ]
    assert len(survivorship) == 1
    assert "0.61" in survivorship[0].dsr_vs_bar
    assert PENDING not in survivorship[0].dsr_vs_bar
    assert "bar not in artifact" in survivorship[0].dsr_vs_bar


def test_band_search_does_not_copy_forbidden_headline_fields(tmp_path: Path) -> None:
    path = tmp_path / "receipts" / "band.json"
    _write(
        path,
        {
            "schema": "adaptive_mix_band_search.v1",
            "created_at": "2026-09-22T00:00:00+00:00",
            "selected_band": None,
            "live_pnl_claim": False,
            "candidates": [
                {"band": 0.0, "eligible": False, "sharpe": 1.5, "pnl": 10, "nav": 11},
                {"band": 0.1, "eligible": False, "sharpe": 2.5},
            ],
        },
    )
    text = render(tmp_path)
    assert "1.5" not in text
    assert "2.5" not in text
    assert "length of candidates; no n_trials field" in text
    assert f"DSR vs bar | {NOT_IN_ARTIFACT}" in text or f"| {NOT_IN_ARTIFACT} |" in text
    scrubbed = re.sub(r"live_pnl_claim", "", text, flags=re.IGNORECASE)
    tokens = {token.lower() for token in re.split(r"[^A-Za-z]+", scrubbed) if token}
    assert tokens.isdisjoint(FORBIDDEN_RESEARCH_METRIC_KEYS)


def test_committed_page_matches_the_generator_and_the_reality_receipt() -> None:
    text = render(_ROOT)
    assert _PAGE.read_text(encoding="utf-8") == text
    receipt_path = (
        _ROOT
        / "research"
        / "reality"
        / "studies"
        / "reality-us-liquid-daily-2026-09-27"
        / "receipt.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    gate = receipt["reality_gate"]
    assert json.dumps(gate["dsr"]) in text
    assert json.dumps(gate["n_trials"]) in text
    assert json.dumps(receipt["pbo"]["pbo"]) in text
    assert "reality_gate.pbo null" in text
    assert gate["verdict"] in text
    prereg = json.loads(receipt_path.with_name("preregistration.json").read_text(encoding="utf-8"))
    assert json.dumps(prereg["reality_filter"]["dsr_pass"]) in text
    rows, _notes = collect_batches(_ROOT)
    assert_pending_has_no_measured_figures(rows)
    pending = next(row for row in rows if isinstance(row, BatchRow) and row.pending)
    assert pending.batch == SURVIVORSHIP_PR
    scrubbed = re.sub(r"live_pnl_claim", "", text, flags=re.IGNORECASE)
    tokens = {token.lower() for token in re.split(r"[^A-Za-z]+", scrubbed) if token}
    assert tokens.isdisjoint(FORBIDDEN_RESEARCH_METRIC_KEYS)
