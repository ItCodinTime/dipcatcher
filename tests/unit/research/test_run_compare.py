"""Unit tests for ``quant_fund.research.compare`` on SYNTHETIC fixtures."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from quant_fund.research.compare import (
    compare_runs,
    compare_series,
    load_run,
    receipt_payload,
)

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "run_compare"
RUN_A = FIXTURE_DIR / "run_a.json"
RUN_B = FIXTURE_DIR / "run_b.json"


def _series(comparison, key: str):
    matches = [s for s in comparison.series if s.key == key]
    assert len(matches) == 1, f"series {key} missing"
    return matches[0]


def test_planted_improvement_detected_significant() -> None:
    comp = compare_runs(RUN_A, RUN_B)
    pinball = _series(comp, "scores.pinball_per_fold")
    assert pinball.n_paired == 40
    # B was planted lower (better under the loss convention).
    assert pinball.mean_delta > 0
    assert pinball.verdict == "b_better"
    assert pinball.dm_p < 0.05
    assert pinball.bootstrap_lo > 0


def test_identical_runs_have_no_effect() -> None:
    comp = compare_runs(RUN_A, RUN_A)
    for s in comp.series:
        assert s.verdict in {"no_effect", "insufficient paired observations"}
        assert s.mean_delta == pytest.approx(0.0)
    pinball = _series(comp, "scores.pinball_per_fold")
    assert pinball.verdict == "no_effect"


def test_short_series_flagged_insufficient() -> None:
    comp = compare_runs(RUN_A, RUN_B)
    short = _series(comp, "scores.qlike_short")
    assert short.verdict == "insufficient paired observations"
    assert "min_paired" in short.note


def test_unaligned_lengths_not_paired() -> None:
    comp = compare_runs(RUN_A, RUN_B)
    pit = _series(comp, "scores.pit_hist")
    assert pit.verdict == "unaligned series lengths"
    assert pit.n_paired == 0


def test_config_diff_and_scalar_deltas() -> None:
    comp = compare_runs(RUN_A, RUN_B)
    diff_paths = {row["path"] for row in comp.config_diff}
    assert "params.temperature" in diff_paths
    delta_paths = {row["path"] for row in comp.metric_deltas}
    assert "metrics.pinball_mean" in delta_paths


def test_forbidden_diagnostics_excluded_from_verdicts() -> None:
    comp = compare_runs(RUN_A, RUN_B)
    assert all("sharpe" not in s.key for s in comp.series)
    excluded = {row["path"] for row in comp.excluded_diagnostics}
    assert "diagnostics.sharpe" in excluded
    assert all(row["path"] != "diagnostics.sharpe" for row in comp.metric_deltas)


def test_symmetry_order_swap_flips_sign() -> None:
    ab = compare_runs(RUN_A, RUN_B)
    ba = compare_runs(RUN_B, RUN_A)
    s_ab = {s.key: s for s in ab.series}
    s_ba = {s.key: s for s in ba.series}
    for key in s_ab:
        a, b = s_ab[key], s_ba[key]
        assert a.mean_delta == pytest.approx(-b.mean_delta, nan_ok=True)
        assert a.dm_stat == pytest.approx(-b.dm_stat, nan_ok=True)
        assert a.dm_p == pytest.approx(b.dm_p, nan_ok=True)
        if a.verdict == "a_better":
            assert b.verdict == "b_better"
        elif a.verdict == "b_better":
            assert b.verdict == "a_better"
        else:
            assert a.verdict == b.verdict


def test_determinism_same_seed_same_result() -> None:
    first = compare_runs(RUN_A, RUN_B, seed=11).to_dict()
    second = compare_runs(RUN_A, RUN_B, seed=11).to_dict()
    assert first == second


def test_higher_is_better_flips_verdict_direction() -> None:
    comp = compare_runs(RUN_A, RUN_B, higher_is_better=True)
    pinball = _series(comp, "scores.pinball_per_fold")
    assert pinball.verdict == "a_better"


def test_load_run_directory_of_jsons(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    shutil.copy(RUN_A, run_dir / "part1.json")
    shutil.copy(RUN_B, run_dir / "part2.json")
    data = load_run(run_dir)
    assert "part1.scores.pinball_per_fold" in data.series
    assert "part2.scores.pinball_per_fold" in data.series


def test_load_run_single_json_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    shutil.copy(RUN_A, run_dir / "receipt.json")
    data = load_run(run_dir)
    assert "scores.pinball_per_fold" in data.series


def test_load_run_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_run(tmp_path / "nope")


def test_load_run_empty_dir(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"no \*\.json results"):
        load_run(tmp_path)


def test_receipt_payload_conventions() -> None:
    comp = compare_runs(RUN_A, RUN_B)
    payload = receipt_payload(comp)
    assert payload["schema"] == "run_compare.v1"
    assert payload["research_only"] is True
    assert payload["live_pnl_claim"] is False
    # Serializable under strict allow_nan (NaN -> null).
    json.dumps(payload, allow_nan=False)


def test_report_json_and_markdown() -> None:
    comp = compare_runs(RUN_A, RUN_B)
    json.dumps(comp.to_dict(), allow_nan=False)
    md = comp.to_markdown()
    assert "# Run comparison" in md
    assert "scores.pinball_per_fold" in md
    assert "Excluded diagnostics" in md


def test_series_keys_filter() -> None:
    comp = compare_runs(RUN_A, RUN_B, series_keys=["scores.pinball_per_fold"])
    assert [s.key for s in comp.series] == ["scores.pinball_per_fold"]


def test_compare_series_direct() -> None:
    a = [0.5] * 30
    b = [0.4] * 30
    comp = compare_series("k", a, b)
    # Constant nonzero delta -> degenerate variance, honestly inconclusive.
    assert comp.verdict in {"inconclusive", "ambiguous", "a_better", "b_better"}
    assert comp.mean_delta == pytest.approx(0.1)
