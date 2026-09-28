"""Ratchets for the harness quality bar.

The strict-module allowlist and the mccabe ceiling may tighten.
They must not shrink or rise. Per-function McCabe ceilings live in
``quality/mccabe_baseline.txt`` and are enforced by
``scripts/check_mccabe_ratchet.py``.
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
MCCABE_BASELINE = ROOT / "quality" / "mccabe_baseline.txt"
CHECKER = runpy.run_path(str(ROOT / "scripts" / "check_mypy_strict_allowlist.py"))
MCCABE = runpy.run_path(str(ROOT / "scripts" / "check_mccabe_ratchet.py"))
strict_entries = CHECKER["_entries"]
strict_allowlisted = CHECKER["allowlisted"]
mccabe_entries = MCCABE["_entries"]
# Initial set of strict-clean modules. Add to the allowlist; never remove these.
STRICT_MODULE_FLOOR = 397
STRICT_BASELINE_SHA256 = "452034ec90dbc11dc2a8ca78f22d950c591ae0fd67b3ecbfabe08d5906f7cdcd"
# validate_ledger_schema. verify_research_artifact was 196 before the split.
MCCABE_CEILING = 74
# Soft threshold + floor count for the per-function C901 ratchet. The
# floor may rise as debt is paid down via --write; it must not fall below
# this commit's recorded size without an intentional baseline rewrite.
MCCABE_SOFT_THRESHOLD = 10
MCCABE_BASELINE_FLOOR = 1
# `except Exception` handlers under src/quant_fund. This is origin/main's
# count at 7d2e01e (75). This branch narrows three of them, so the tree is
# at 72. New handlers that push the total above main fail this test.
EXCEPT_EXCEPTION_CEILING = 75


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


def test_mccabe_per_function_baseline_shape() -> None:
    assert MCCABE["SOFT_THRESHOLD"] == MCCABE_SOFT_THRESHOLD
    rows = mccabe_entries(MCCABE_BASELINE)
    assert len(rows) >= MCCABE_BASELINE_FLOOR
    assert rows == dict(sorted(rows.items(), key=lambda item: (item[0][0], item[0][1])))
    for (path, name), complexity in rows.items():
        assert path.startswith("src/")
        assert path.endswith(".py")
        assert (ROOT / path).is_file(), path
        assert name
        assert complexity >= MCCABE_SOFT_THRESHOLD
    # Northset worst offender must stay on the ratchet (may only decrease).
    northset_bench = rows.get(("src/quant_fund/northset/benches.py", "bench_northset"))
    assert northset_bench is not None
    assert northset_bench <= 14


def test_mccabe_ratchet_rejects_raised_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = tmp_path / "mccabe_baseline.txt"
    rows = {("src/quant_fund/sample.py", "too_complex"): 12}
    baseline.write_text(MCCABE["_format_baseline"](rows))
    monkeypatch.setitem(MCCABE["check"].__globals__, "BASELINE", baseline)
    monkeypatch.setitem(
        MCCABE["check"].__globals__,
        "_ruff_complexities",
        lambda _paths: {("src/quant_fund/sample.py", "too_complex"): 15},
    )
    assert MCCABE["check"]() == 1


def test_mccabe_write_refuses_raised_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = tmp_path / "quality" / "mccabe_baseline.txt"
    baseline.parent.mkdir(parents=True)
    rows = {("src/quant_fund/sample.py", "too_complex"): 12}
    baseline.write_text(MCCABE["_format_baseline"](rows))
    monkeypatch.setitem(MCCABE["write_baseline"].__globals__, "ROOT", tmp_path)
    monkeypatch.setitem(MCCABE["write_baseline"].__globals__, "BASELINE", baseline)
    monkeypatch.setitem(
        MCCABE["write_baseline"].__globals__,
        "_ruff_complexities",
        lambda _paths: {("src/quant_fund/sample.py", "too_complex"): 15},
    )
    assert MCCABE["write_baseline"]() == 1
    assert "12" in baseline.read_text()


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
