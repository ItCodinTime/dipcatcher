"""Reality Filter (PROOFCORE W4) — statistically honest multi-trial inference.

Unit-safe PSR/MinTRL, Deflated Sharpe with effective-trials clustering,
CSCV/PBO, SPA / White reality check, and BH-FDR over the trial ledger.
Research diagnostics only — proper scores remain the headline (AGENTS.md).
"""

from __future__ import annotations

from quant_fund.proofcore.contracts import RealityFilterError, RealityReport, TrialLedgerRow
from quant_fund.reality.cscv import cscv_pbo, cscv_splits, pbo_from_performance
from quant_fund.reality.dsr import dsr_from_ledger, effective_trials
from quant_fund.reality.fdr import bh_fdr, trial_pvalue
from quant_fund.reality.psr import min_trl_from_returns, psr_from_returns
from quant_fund.reality.report import build_reality_report
from quant_fund.reality.spa import (
    reality_check_from_trials,
    spa_from_trials,
    stepm_from_trials,
)

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
