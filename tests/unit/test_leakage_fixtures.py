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


def test_fixture_dir_complete() -> None:
    on_disk = sorted(p.name for p in FIXTURE_DIR.glob("*.py"))
    assert on_disk == sorted([*SEEDED, *CONTROLS])


@pytest.mark.parametrize("filename,rule_id", sorted(SEEDED.items()))
def test_seeded_leak_caught(filename: str, rule_id: str) -> None:
    report = scan_paths([FIXTURE_DIR / filename])
    fired = {f.rule_id for f in report.findings}
    assert rule_id in fired, f"{filename} no longer trips {rule_id} (fired: {sorted(fired)})"


@pytest.mark.parametrize("filename", CONTROLS)
def test_clean_control_not_flagged(filename: str) -> None:
    report = scan_paths([FIXTURE_DIR / filename])
    errors = [f for f in report.findings if f.severity == "error"]
    assert errors == [], f"clean control {filename} flagged: {errors}"


@pytest.mark.parametrize("filename", sorted([*SEEDED, *CONTROLS]))
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
