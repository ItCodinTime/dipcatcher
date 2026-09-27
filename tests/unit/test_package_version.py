"""The harness, fx-1, and the installed distribution share one version."""

from __future__ import annotations

import re
from importlib.metadata import version
from pathlib import Path

import fx1
import quant_fund

_ROOT = Path(__file__).resolve().parents[2]


def test_quant_fund_version_matches_fx1_and_distribution() -> None:
    assert quant_fund.__version__ == fx1.__version__
    assert version("fx-1") == fx1.__version__


def test_citation_cff_version_matches_distribution() -> None:
    text = (_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    match = re.search(r"(?m)^version: ([0-9]+\.[0-9]+\.[0-9]+)\s*$", text)
    assert match is not None
    assert match.group(1) == fx1.__version__
