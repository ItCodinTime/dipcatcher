"""Key hygiene for simulation diagnostics.

``live_pnl_claim`` is the flag that says these numbers are not a live
result. It is the only key allowed to contain the token ``pnl``. Sharpe,
Sortino, Calmar, and NAV tokens are rejected everywhere.
"""

from __future__ import annotations

from quant_fund.research.catalog.constants import FORBIDDEN_RESEARCH_METRIC_KEYS


def _keys(obj: object) -> list[str]:
    found: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            found.append(str(key))
            found.extend(_keys(value))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            found.extend(_keys(item))
    return found


def diagnostic_keys_ok(payload: object) -> bool:
    """True when headline-metric tokens are absent aside from ``live_pnl_claim``."""
    for key in _keys(payload):
        if key == "live_pnl_claim":
            continue
        parts = key.lower().replace("-", "_").split("_")
        if any(part in FORBIDDEN_RESEARCH_METRIC_KEYS for part in parts if part):
            return False
    return True
