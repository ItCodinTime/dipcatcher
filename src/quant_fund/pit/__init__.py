"""PIT Vault (PROOFCORE W1): write-once, content-addressed, bitemporal store.

One physical choke point for all time-series data reads: the ONLY legal read
path is ``PitVault.asof(t)``, which physically cannot return rows with
``known_at > t``. Layer 2 of the PROOFCORE layering contract (DESIGN.md §1.3):
the implementation modules import contracts, schemas, utils, and data.lake
only. Those modules load on first use so ``pit.cli`` does not import polars.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "DataAccessRecorder",
    "ManifestError",
    "PitFrame",
    "PitVault",
    "RestatementPolicy",
    "VaultError",
    "VaultUnavailableError",
    "WatchdogProtocol",
]

_EXPORTS: dict[str, str] = {
    "DataAccessRecorder": "quant_fund.pit.vault",
    "ManifestError": "quant_fund.proofcore.contracts",
    "PitFrame": "quant_fund.pit.frame",
    "PitVault": "quant_fund.pit.vault",
    "RestatementPolicy": "quant_fund.pit.corrections",
    "VaultError": "quant_fund.proofcore.contracts",
    "VaultUnavailableError": "quant_fund.pit.frame",
    "WatchdogProtocol": "quant_fund.pit.vault",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
