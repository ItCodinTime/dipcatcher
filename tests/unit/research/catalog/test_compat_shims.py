"""Coverage for the catalog compatibility re-export shims.

These modules only re-export names from the parent package; importing them
and touching a few names is the whole contract.
"""

from __future__ import annotations

import importlib

import pytest

_SHIMS = [
    "quant_fund.research.catalog.constants",
    "quant_fund.research.catalog.families",
    "quant_fund.research.catalog.northset",
    "quant_fund.research.catalog.predicates",
    "quant_fund.research.catalog.consistency",
]


@pytest.mark.parametrize("module", _SHIMS)
def test_shim_imports_and_exports(module: str) -> None:
    mod = importlib.import_module(module)
    assert mod.__all__
    for name in mod.__all__[:3]:
        assert getattr(mod, name) is not None


def test_constants_values() -> None:
    from quant_fund.research.catalog import constants

    assert isinstance(constants.BENCHMARK_CATALOG_VERSION, int)
    assert constants.REQUIRED_BENCHMARK_FAMILIES
    assert set(constants.BENCHMARK_FAMILY_ORDER) >= constants.REQUIRED_BENCHMARK_FAMILIES
    assert constants.RESEARCH_RECEIPT_SCHEMA_VERSION in (
        constants.RESEARCH_RECEIPT_SCHEMA_VERSIONS_ACCEPTED
    )
    assert isinstance(constants.FORBIDDEN_RESEARCH_METRIC_KEYS, (set, frozenset))
