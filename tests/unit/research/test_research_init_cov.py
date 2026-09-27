"""Coverage for the lazy ``quant_fund.research`` package __getattr__.

``quant_fund.research`` defers importing ``agent`` (which pulls northset →
microstructure) until ``run_research`` is actually requested, to avoid
circular imports. These tests pin both branches of the lazy loader.
"""

import subprocess
import sys

import pytest

import quant_fund.research as research_pkg
from quant_fund.research import __getattr__ as research_getattr


def test_run_research_lazy_attr_resolves_to_agent_callable() -> None:
    """Accessing run_research triggers the lazy import and returns the real fn."""
    from quant_fund.research.agent import run_research

    resolved = research_getattr("run_research")

    assert resolved is run_research
    # Accessing via the package goes through __getattr__ too.
    assert research_pkg.run_research is run_research
    assert "run_research" in research_pkg.__all__


def test_run_research_is_not_eagerly_bound_on_package() -> None:
    """The package must not bind run_research in its own namespace."""
    assert "run_research" not in vars(research_pkg)


def test_research_package_does_not_eagerly_import_agent() -> None:
    """In a fresh interpreter, importing the package alone must not import agent."""
    code = (
        "import sys; import quant_fund.research; "
        "sys.exit(0 if 'quant_fund.research.agent' not in sys.modules else 1)"
    )
    result = subprocess.run([sys.executable, "-c", code], check=False)
    assert result.returncode == 0


def test_unknown_attribute_raises_attribute_error() -> None:
    """Miss branch: unknown names fail closed with a standard AttributeError."""
    with pytest.raises(AttributeError, match="no attribute 'nonexistent'"):
        research_getattr("nonexistent")

    with pytest.raises(AttributeError):
        _ = research_pkg.nonexistent
