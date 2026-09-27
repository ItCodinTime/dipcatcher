"""Ratchets for the harness quality bar.

The strict-module allowlist and the mccabe ceiling may tighten.
They must not shrink or rise.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST = ROOT / "quality" / "mypy_strict_modules.txt"
# Initial size of quality/mypy_strict_modules.txt. Add modules; do not drop below.
STRICT_MODULE_FLOOR = 393
# validate_ledger_schema. verify_research_artifact was 196 before the split.
MCCABE_CEILING = 74
# `except Exception` handlers under src/quant_fund. This is origin/main's
# count at 46bfa4b (75). This branch narrows three of them, so the tree is
# at 72. New handlers that push the total above main fail this test.
EXCEPT_EXCEPTION_CEILING = 75


def test_mypy_strict_allowlist_only_grows() -> None:
    lines = [
        line.strip()
        for line in ALLOWLIST.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert lines == sorted(set(lines))
    assert len(lines) >= STRICT_MODULE_FLOOR
    for line in lines:
        assert line.startswith("src/quant_fund/")
        assert line.endswith(".py")
        assert (ROOT / line).is_file(), line


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
