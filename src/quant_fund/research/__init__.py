"""Research package.

Do not eagerly import ``agent`` here — that pulls northset → microstructure and
creates circular imports when leaf modules import from ``quant_fund.research.*``.
"""

from __future__ import annotations

from typing import Any

__all__ = ["run_research"]


def __getattr__(name: str) -> Any:
    if name == "run_research":
        from quant_fund.research.agent import run_research

        return run_research
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
