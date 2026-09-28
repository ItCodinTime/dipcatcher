"""Ratchets for the harness quality bar.

The strict-module allowlist and the mccabe ceiling may tighten.
They must not shrink or rise.
"""

from __future__ import annotations

import hashlib
import re
import runpy
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST = ROOT / "quality" / "mypy_strict_modules.txt"
BASELINE = ROOT / "quality" / "mypy_strict_baseline.txt"
CHECKER = runpy.run_path(str(ROOT / "scripts" / "check_mypy_strict_allowlist.py"))
strict_entries = CHECKER["_entries"]
strict_allowlisted = CHECKER["allowlisted"]
# Initial set of strict-clean modules. Add to the allowlist; never remove these.
STRICT_MODULE_FLOOR = 397
STRICT_BASELINE_SHA256 = "452034ec90dbc11dc2a8ca78f22d950c591ae0fd67b3ecbfabe08d5906f7cdcd"
# validate_ledger_schema. verify_research_artifact was 196 before the split.
MCCABE_CEILING = 74
# `except Exception` handlers under src/quant_fund. Origin/main sat at 75;
# three catalog lazy-import guards narrowed to ImportError, so the ceiling
# tightens to 72. New handlers that push the total above this fail the test.
EXCEPT_EXCEPTION_CEILING = 72


def test_mypy_strict_allowlist_only_grows() -> None:
    lines = strict_entries(ALLOWLIST)
    baseline = strict_entries(BASELINE)
    assert lines == sorted(set(lines))
    assert baseline == sorted(set(baseline))
    assert hashlib.sha256("\n".join(baseline).encode()).hexdigest() == STRICT_BASELINE_SHA256
    assert set(baseline) <= set(lines)
    assert len(lines) >= STRICT_MODULE_FLOOR
    for line in lines:
        assert line.startswith("src/quant_fund/")
        assert line.endswith(".py")
        assert (ROOT / line).is_file(), line


def test_allowlist_parser_ignores_indented_comments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = "src/quant_fund/sample.py"
    file = tmp_path / module
    file.parent.mkdir(parents=True)
    file.write_text("pass\n")
    listing = tmp_path / "allowlist.txt"
    listing.write_text(f"  # comment\n{module}\n")
    baseline = tmp_path / "baseline.txt"
    baseline.write_text(f"{module}\n")
    monkeypatch.setitem(strict_allowlisted.__globals__, "ROOT", tmp_path)
    monkeypatch.setitem(strict_allowlisted.__globals__, "ALLOWLIST", listing)
    monkeypatch.setitem(strict_allowlisted.__globals__, "BASELINE", baseline)
    assert strict_allowlisted() == [module]


def test_allowlist_rejects_dropped_baseline_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("src/quant_fund/pinned.py\n")
    listing = tmp_path / "allowlist.txt"
    listing.write_text("src/quant_fund/replacement.py\n")
    monkeypatch.setitem(strict_allowlisted.__globals__, "ROOT", tmp_path)
    monkeypatch.setitem(strict_allowlisted.__globals__, "ALLOWLIST", listing)
    monkeypatch.setitem(strict_allowlisted.__globals__, "BASELINE", baseline)
    with pytest.raises(SystemExit):
        strict_allowlisted()


def test_mccabe_ceiling_not_raised() -> None:
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text())
    lint = cfg["tool"]["ruff"]["lint"]
    assert "C901" in lint["select"]
    assert lint["mccabe"]["max-complexity"] <= MCCABE_CEILING


def test_no_bare_except_and_exception_ceiling() -> None:
    bare = 0
    broad = 0
    for path in (ROOT / "src" / "quant_fund").rglob("*.py"):
        for line in path.read_text().splitlines():
            if re.match(r"\s*except\s*:", line):
                bare += 1
            elif re.match(r"\s*except\s+Exception\b", line):
                broad += 1
    assert bare == 0
    assert broad <= EXCEPT_EXCEPTION_CEILING


def test_broad_exception_manifest_matches_tree() -> None:
    """The global ceiling alone lets a new handler trade against an unrelated
    narrowing. The manifest pins the count per file so every broad catch is
    accounted to a place a reviewer can look at."""
    checker = runpy.run_path(str(ROOT / "scripts" / "check_broad_exceptions.py"))
    actual = checker["actual_counts"]()
    manifest = checker["manifest_counts"]()
    assert manifest == actual, (
        "broad-exception manifest drifted — regenerate with "
        "python scripts/update_broad_exceptions.py"
    )


def test_broad_exception_manifest_rejects_drift_and_dupes(tmp_path: Path) -> None:
    checker = runpy.run_path(str(ROOT / "scripts" / "check_broad_exceptions.py"))
    manifest_counts = checker["manifest_counts"]
    bad = tmp_path / "broad_exceptions.txt"
    bad.write_text("src/quant_fund/a.py 1\nsrc/quant_fund/a.py 2\n")
    with pytest.raises(SystemExit):
        manifest_counts(bad)
    bad.write_text("src/quant_fund/a.py 0\n")
    with pytest.raises(SystemExit):
        manifest_counts(bad)
    ok = tmp_path / "ok.txt"
    ok.write_text("# comment\nsrc/quant_fund/a.py 1\n")
    assert manifest_counts(ok) == {"src/quant_fund/a.py": 1}
