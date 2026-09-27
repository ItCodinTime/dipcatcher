"""Reality Filter (PROOFCORE W4) — statistically honest multi-trial inference.

Research diagnostics only. Public names resolve on first use so mounting the
``quant reality`` CLI does not load numerical libraries during CLI startup.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "RealityFilterError",
    "RealityReport",
    "TrialLedgerRow",
    "bh_fdr",
    "build_reality_report",
    "cscv_pbo",
    "cscv_splits",
    "dsr_from_ledger",
    "effective_trials",
    "min_trl_from_returns",
    "pbo_from_performance",
    "psr_from_returns",
    "reality_check_from_trials",
    "spa_from_trials",
    "stepm_from_trials",
    "trial_pvalue",
]

_EXPORTS: dict[str, str] = {
    "RealityFilterError": "quant_fund.proofcore.contracts",
    "RealityReport": "quant_fund.proofcore.contracts",
    "TrialLedgerRow": "quant_fund.proofcore.contracts",
    "bh_fdr": "quant_fund.reality.fdr",
    "build_reality_report": "quant_fund.reality.report",
    "cscv_pbo": "quant_fund.reality.cscv",
    "cscv_splits": "quant_fund.reality.cscv",
    "dsr_from_ledger": "quant_fund.reality.dsr",
    "effective_trials": "quant_fund.reality.dsr",
    "min_trl_from_returns": "quant_fund.reality.psr",
    "pbo_from_performance": "quant_fund.reality.cscv",
    "psr_from_returns": "quant_fund.reality.psr",
    "reality_check_from_trials": "quant_fund.reality.spa",
    "spa_from_trials": "quant_fund.reality.spa",
    "stepm_from_trials": "quant_fund.reality.spa",
    "trial_pvalue": "quant_fund.reality.fdr",
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
