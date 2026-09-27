"""PIT Vault (PROOFCORE W1): write-once, content-addressed, bitemporal store.

One physical choke point for all time-series data reads: the ONLY legal read
path is ``PitVault.asof(t)``, which physically cannot return rows with
``known_at > t``. Layer 2 of the PROOFCORE layering contract (DESIGN.md §1.3):
imports contracts, schemas, utils, data.lake only.
"""

from __future__ import annotations

from quant_fund.pit.corrections import RestatementPolicy
from quant_fund.pit.frame import PitFrame, VaultUnavailableError
from quant_fund.pit.vault import DataAccessRecorder, PitVault, WatchdogProtocol
from quant_fund.proofcore.contracts import ManifestError, VaultError

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
