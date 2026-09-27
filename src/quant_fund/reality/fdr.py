"""Benjamini–Hochberg FDR over the research trial ledger, split by family.

Per-trial p-values are ``1 - PSR`` (PSR vs ``sr_star = 0`` computed from the
ledger row's per-period Sharpe, ``n_obs``, skew and raw kurtosis — the
unit-safe convention; A1 F1). Families are never pooled: ``calibration`` and
``discovery`` are BH-adjusted separately, and ``bound`` trials are excluded
entirely (mirrors ``cli/_app.py:format_fdr_families``).

Research diagnostics only — never a live P&L / promotion claim.
"""

from __future__ import annotations

import numpy as np

from quant_fund.metrics.inference import benjamini_hochberg
from quant_fund.metrics.overfitting import probabilistic_sharpe
from quant_fund.proofcore.contracts import TrialLedgerRow

_FDR_FAMILIES = ("calibration", "discovery")


def trial_pvalue(row: TrialLedgerRow) -> float:
    """p_i = 1 - PSR_i vs sr_star = 0 from the ledger row's stored moments.

    Fail-closed: degenerate rows (non-finite moments, n_obs < 2) yield NaN,
    and NaN p-values are NEVER rejected downstream.
    """
    psr = probabilistic_sharpe(
        float(row.sharpe_periodic),
        0.0,
        int(row.n_obs),
        float(row.skew),
        float(row.kurtosis_raw),
    )
    return float(1.0 - psr)


def bh_fdr(rows: list[TrialLedgerRow], *, q: float = 0.05) -> dict[str, list[str]]:
    """BH-FDR rejections per family. Returns
    ``{'calibration': [trial_id...], 'discovery': [...]}`` — rejected trial
    ids in ascending p-value order (ties broken by trial_id for determinism).
    'bound' rows are never pooled nor returned.
    """
    if not np.isfinite(q) or not 0.0 < float(q) < 1.0:
        raise ValueError("q must be finite and in (0, 1)")
    out: dict[str, list[str]] = {fam: [] for fam in _FDR_FAMILIES}
    for fam in _FDR_FAMILIES:
        fam_rows = [r for r in rows if r.family == fam]
        if not fam_rows:
            continue
        pvals = np.asarray([trial_pvalue(r) for r in fam_rows], dtype=float)
        finite = np.isfinite(pvals)
        reject, _ = benjamini_hochberg(np.where(finite, pvals, 1.0), alpha=float(q))
        reject &= finite  # NaN p-values can never reject (fail-closed)
        rejected = [fam_rows[i] for i in np.nonzero(reject)[0]]
        rejected.sort(key=lambda r: (trial_pvalue(r), r.trial_id))
        out[fam] = [r.trial_id for r in rejected]
    return out
