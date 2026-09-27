"""Seeded-leak CI gate (DESIGN.md §6.4): fixtures MUST trip their rule.

A fixture that stops tripping = CI red. Clean controls MUST NOT trip any
error-severity finding (false-positive discipline).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from quant_fund.leakage import scan_paths

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "leakage_fixtures"

SEEDED: dict[str, str] = {
    "lh001_shift_forward.py": "LH001",
    "lh002_centered_rolling.py": "LH002",
    "lh003_fit_before_split.py": "LH003",
    "lh004_join_asof_event_time.py": "LH004",
    "lh005_frozen_universe.py": "LH005",
    "lh006_delta_mid_forward.py": "LH006",
    "lh007_psr_annualized.py": "LH007",
    "lh008_headline_string.py": "LH008",
    "lh010_bfill_event_time.py": "LH010",
}

CONTROLS: tuple[str, ...] = (
    "clean_forward_label.py",
    "clean_fold_scaler.py",
    "clean_fwd_delta_mid.py",
)

# ADVERSARIAL §1a evasion pack (proofcore/hardening): each fixture pins one
# adversarial snippet. Positives MUST trip their rule (at the mapped minimum
# severity); the mapping documents the post-hardening bar. New error-severity
# catches land in SEEDED above on promotion.
ADV_SEEDED: dict[str, str] = {
    "adv_e03_shift_keyword.py": "LH001",
    "adv_e04_shift_variable.py": "LH001",
    "adv_e06_normalizer_fit.py": "LH003",
    "adv_e07_fold_name_abuse.py": "LH003",
    "adv_e08_spelled_out.py": "LH013",
    "adv_e09_docstring_headline.py": "LH013",
    "adv_e10_fstring.py": "LH013",
    "adv_e12_getattr_parquet.py": "LH009",
    "adv_e13_duckdb_sql.py": "LH009",
    "adv_e14_shift_var_nonliteral.py": "LH001",
    "adv_e16_frozen_universe_lowercase.py": "LH005",
    "adv_e17_join_asof_keyvar.py": "LH004",
    "adv_f02_comment_headline.py": "LH013",
    "adv_f03_zero_width.py": "LH008",
    "adv_f04_concat.py": "LH008",
    "adv_f06_nav_words.py": "LH013",
    "adv_helper_indirection.py": "LH014",
}

# DOCUMENTED NEGATIVES (the residual ceiling, pinned so a future rule that
# catches one promotes it to ADV_SEEDED): numpy index arithmetic, dict-lookup
# time travel, non-price helper shift with unresolved amount, lone number
# outside the proximity window, manual annualized Sharpe without a
# sharpe_ratio call.
ADV_NEGATIVE: tuple[str, ...] = (
    "adv_e01_numpy_indexing.py",
    "adv_e02_helper_shift.py",
    "adv_e05_dict_lookup.py",
    "adv_e11_outside_window.py",
    "adv_e15_manual_sharpe_psr.py",
)


def test_fixture_dir_complete() -> None:
    on_disk = sorted(p.name for p in FIXTURE_DIR.glob("*.py"))
    assert on_disk == sorted([*SEEDED, *CONTROLS, *ADV_SEEDED, *ADV_NEGATIVE])


@pytest.mark.parametrize("filename,rule_id", sorted(SEEDED.items()))
def test_seeded_leak_caught(filename: str, rule_id: str) -> None:
    report = scan_paths([FIXTURE_DIR / filename])
    fired = {f.rule_id for f in report.findings}
    assert rule_id in fired, f"{filename} no longer trips {rule_id} (fired: {sorted(fired)})"


@pytest.mark.parametrize("filename,rule_id", sorted(ADV_SEEDED.items()))
def test_adversarial_leak_caught(filename: str, rule_id: str) -> None:
    report = scan_paths([FIXTURE_DIR / filename])
    fired = {f.rule_id for f in report.findings}
    assert rule_id in fired, f"{filename} no longer trips {rule_id} (fired: {sorted(fired)})"


@pytest.mark.parametrize("filename", ADV_NEGATIVE)
def test_adversarial_documented_negative(filename: str) -> None:
    """Pins the residual ceiling: zero error findings today. If a future rule
    catches the evasion, promote the fixture to ADV_SEEDED."""
    report = scan_paths([FIXTURE_DIR / filename])
    errors = [f for f in report.findings if f.severity == "error"]
    assert errors == [], f"{filename} now trips error findings: {errors}"


@pytest.mark.parametrize("filename", CONTROLS)
def test_clean_control_not_flagged(filename: str) -> None:
    report = scan_paths([FIXTURE_DIR / filename])
    errors = [f for f in report.findings if f.severity == "error"]
    assert errors == [], f"clean control {filename} flagged: {errors}"


@pytest.mark.parametrize("filename", sorted([*SEEDED, *CONTROLS, *ADV_SEEDED, *ADV_NEGATIVE]))
def test_fixture_importable(filename: str) -> None:
    """Fixtures are self-contained and importable (DESIGN.md §6.4)."""
    path = FIXTURE_DIR / filename
    spec = importlib.util.spec_from_file_location(f"leakage_fixture_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:  # minimal-env runs: polars absent
        if exc.name == "polars":
            pytest.skip("polars not installed in this environment")
        raise
