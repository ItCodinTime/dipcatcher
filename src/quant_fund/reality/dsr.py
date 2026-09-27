"""Deflated Sharpe Ratio with effective-trials clustering.

Bailey & López de Prado (2014): the expected maximum Sharpe under the null
grows with the number of *independent* trials. Clusters of correlated
strategy variants (same ``cluster_id`` in the trial ledger) count once each,
so the deflation uses ``n_effective = n_clusters``, not the raw trial count.

Research diagnostics only — never a live P&L / promotion claim.
"""

from __future__ import annotations

import numpy as np

from quant_fund.metrics.overfitting import deflated_sharpe
from quant_fund.proofcore.contracts import RealityFilterError, TrialLedgerRow


def effective_trials(cluster_sizes: list[int]) -> float:
    """Bailey–LdP effective number of independent trials.

    Clusters of correlated variants count once: ``eff = sum over clusters of
    1`` (= number of non-empty clusters), NOT the raw trial count. Empty
    input or a non-positive cluster size -> ``RealityFilterError``
    (fail-closed; a silently zero effective count would fabricate evidence).
    """
    if not cluster_sizes:
        raise RealityFilterError("effective_trials requires at least one cluster")
    sizes = [int(s) for s in cluster_sizes]
    if any(s < 1 for s in sizes):
        raise RealityFilterError(f"cluster sizes must be >= 1, got {cluster_sizes!r}")
    return float(len(sizes))


def dsr_from_ledger(rows: list[TrialLedgerRow]) -> float:
    """DSR of the best trial in a ledger, deflated by effective trials.

    Groups rows by ``cluster_id`` -> ``effective_trials``; ``var_sr`` is the
    cross-trial (ddof=1) variance of per-period Sharpes; then
    ``metrics.overfitting.deflated_sharpe`` with the best trial's per-period
    SR, ``n_obs``, skew and raw kurtosis. Degeneracy: a single-trial ledger
    yields DSR == PSR vs ``sr_star = 0`` exactly. Empty ledger ->
    ``RealityFilterError``.
    """
    if not rows:
        raise RealityFilterError("dsr_from_ledger requires at least one trial row")
    clusters: dict[str, int] = {}
    for row in rows:
        clusters[row.cluster_id] = clusters.get(row.cluster_id, 0) + 1
    n_eff = effective_trials(list(clusters.values()))
    srs = np.asarray([float(r.sharpe_periodic) for r in rows], dtype=float)
    var_sr = float(np.var(srs, ddof=1)) if srs.size > 1 else 0.0
    best = max(rows, key=lambda r: (float(r.sharpe_periodic), r.trial_id))
    return deflated_sharpe(
        float(best.sharpe_periodic),
        int(best.n_obs),
        float(best.skew),
        float(best.kurtosis_raw),
        int(round(n_eff)),
        var_sr,
    )
