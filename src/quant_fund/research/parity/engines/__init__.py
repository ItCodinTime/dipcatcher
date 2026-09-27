"""Engine adapters for the parity harness.

Each adapter exposes ``run(panel, plan, *, initial_nav, commission_bps) ->
EngineRun`` and is lazily probed: a missing third-party library marks the
engine ``available() == False`` and the harness records a SKIP, never a
fabricated result.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable

import polars as pl

from quant_fund.research.parity.canonical import EngineRun
from quant_fund.research.parity.strategies import StrategyPlan

Runner = Callable[..., EngineRun]

_ADAPTERS: dict[str, str] = {
    "native": "quant_fund.research.parity.engines.native",
    "vectorbt": "quant_fund.research.parity.engines.vectorbt_engine",
    "backtrader": "quant_fund.research.parity.engines.backtrader_engine",
    "nautilus": "quant_fund.research.parity.engines.nautilus_engine",
}

# Third-party import each adapter needs (None = always available).
_REQUIRES: dict[str, str | None] = {
    "native": None,
    "vectorbt": "vectorbt",
    "backtrader": "backtrader",
    "nautilus": "nautilus_trader",
}

KNOWN_ENGINES = tuple(_ADAPTERS)


def available(engine: str) -> bool:
    """True when the adapter's third-party import can succeed."""
    req = _REQUIRES[engine]
    return req is None or importlib.util.find_spec(req) is not None


def available_engines() -> list[str]:
    return [name for name in _ADAPTERS if available(name)]


def skip_reason(engine: str) -> str | None:
    req = _REQUIRES[engine]
    if req is None or importlib.util.find_spec(req) is not None:
        return None
    return f"optional dependency {req!r} not installed"


def run_engine(
    engine: str,
    panel: pl.DataFrame,
    plan: StrategyPlan,
    *,
    initial_nav: float = 1_000_000.0,
    commission_bps: float = 0.0,
) -> EngineRun:
    """Dispatch one (engine, strategy) run; SKIP/fail become statuses."""
    if engine not in _ADAPTERS:
        raise KeyError(f"unknown engine {engine!r}; have {sorted(_ADAPTERS)}")
    reason = skip_reason(engine)
    if reason is not None:
        return EngineRun.skip(engine, plan.name, reason)
    import importlib

    module = importlib.import_module(_ADAPTERS[engine])
    try:
        return module.run(
            panel,
            plan,
            initial_nav=initial_nav,
            commission_bps=commission_bps,
        )
    except Exception as exc:  # noqa: BLE001 - an engine crash is data, not a crash of the harness
        return EngineRun.fail(engine, plan.name, f"{type(exc).__name__}: {exc}")
