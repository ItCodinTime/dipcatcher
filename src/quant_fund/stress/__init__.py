"""Research-only stress engine with lazy public exports.

Importing the CLI must not load numerical and plotting dependencies before a
stress command is invoked.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "CRISIS_CATALOG": "catalog",
    "Crisis": "catalog",
    "crisis_by_id": "catalog",
    "replay_crisis": "replay",
    "replay_portfolio": "replay",
    "build_stress_report": "report",
    "render_html": "report",
    "render_markdown": "report",
    "reverse_stress": "reverse",
    "worst_linear_scenario": "reverse",
    "ResearchStrategy": "strategy",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
