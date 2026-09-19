"""Audit docs must not drift from the runtime benchmark catalog."""

import re
from pathlib import Path

from quant_fund.research.catalog import BENCHMARK_FAMILY_ORDER

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOC_GLOBS = ("README.md", "docs/**/*.md")
_FAMILY_COUNT = re.compile(r"(\d+)-family")


def _doc_paths() -> list[Path]:
    paths = [path for pattern in _DOC_GLOBS for path in _REPO_ROOT.glob(pattern)]
    return sorted({path for path in paths if path.is_file()})


def test_documented_family_counts_match_catalog() -> None:
    expected = len(BENCHMARK_FAMILY_ORDER)
    mismatches = []
    for path in _doc_paths():
        for match in _FAMILY_COUNT.finditer(path.read_text()):
            if int(match.group(1)) != expected:
                mismatches.append(f"{path.relative_to(_REPO_ROOT)}: {match.group(0)} != {expected}")
    assert not mismatches, mismatches


def test_catalog_family_order_is_unique_and_required() -> None:
    from quant_fund.research.verify import REQUIRED_BENCHMARK_FAMILIES

    assert len(set(BENCHMARK_FAMILY_ORDER)) == len(BENCHMARK_FAMILY_ORDER)
    assert set(BENCHMARK_FAMILY_ORDER) == REQUIRED_BENCHMARK_FAMILIES
