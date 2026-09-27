"""RealityReport assembly — orchestrates §7.1–7.5 into the contracts schema.

Verdict logic (fail-closed):
  - ``insufficient_evidence`` if n_trials < 10 or n_effective < 3;
  - ``deflated`` if DSR is non-finite or < 0.95, or PBO > 0.5, or the SPA
    p-value > q;
  - else ``pass``.

``pass`` is a research-diagnostic verdict only — the AGENTS.md honesty
contract stands: proper scores are the headline; DSR/PSR/PBO are overfitting
diagnostics, never promotion evidence, and no Sharpe/P&L/NAV is printed as a
headline by the CLI.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.overfitting import min_track_record_length, probabilistic_sharpe
from quant_fund.proofcore.contracts import (
    RealityFilterError,
    RealityReport,
    TrialLedgerRow,
    sha256_hex_json,
)
from quant_fund.reality.cscv import cscv_pbo
from quant_fund.reality.dsr import dsr_from_ledger, effective_trials
from quant_fund.reality.fdr import bh_fdr
from quant_fund.reality.spa import spa_from_trials

Array = NDArray[np.float64]

MIN_TRIALS = 10
MIN_EFFECTIVE_TRIALS = 3.0
DSR_PASS_THRESHOLD = 0.95
PBO_DEFLATE_THRESHOLD = 0.5

#: Honesty framing carried on every reality report (AGENTS.md rules 1–4).
DISCLAIMER = (
    "research diagnostics only — proper scores are the headline; "
    "DSR/PSR/PBO/SPA are overfitting diagnostics, never promotion evidence"
)


def _report_hash(fields: dict[str, object]) -> str:
    """sha256 over the canonical JSON of all report fields minus the hash.

    Floats are rounded to 12 decimals per the §8.2 determinism policy before
    serialization; non-finite values serialize as JSON null.
    """

    def _round(obj: object) -> object:
        if isinstance(obj, float):
            return round(obj, 12) if np.isfinite(obj) else None
        if isinstance(obj, list):
            return [_round(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _round(v) for k, v in obj.items()}
        return obj

    return sha256_hex_json(_round(fields))


def _verdict(
    *,
    n_trials: int,
    n_effective: float,
    dsr: float,
    pbo: float,
    spa_pvalue: float,
    q: float,
) -> str:
    if n_trials < MIN_TRIALS or n_effective < MIN_EFFECTIVE_TRIALS:
        return "insufficient_evidence"
    if not np.isfinite(dsr) or dsr < DSR_PASS_THRESHOLD:
        # Fail-closed: a non-finite DSR can never mint a 'pass'.
        return "deflated"
    if np.isfinite(pbo) and pbo > PBO_DEFLATE_THRESHOLD:
        return "deflated"
    if np.isfinite(spa_pvalue) and spa_pvalue > q:
        return "deflated"
    return "pass"


def build_reality_report(
    rows: list[TrialLedgerRow],
    *,
    q: float = 0.05,
    s_blocks: int = 16,
    returns_by_trial: dict[str, Array] | None = None,
    spa_n_boot: int = 2000,
    seed: int = 7,
    created_utc: str | None = None,
) -> RealityReport:
    """Orchestrate §7.1–7.5 into the contracts ``RealityReport`` schema.

    Without ``returns_by_trial`` the ledger rows alone carry no return
    series, so CSCV/PBO and SPA are not computable: ``pbo`` / ``spa_pvalue``
    are honest NaN ("NaN if n/a" per the schema). With
    ``returns_by_trial`` (trial_id -> aligned per-period returns, all one
    length), CSCV uses ``s_blocks`` blocks and SPA uses ``spa_n_boot``
    stationary-bootstrap draws. Empty ledger -> ``RealityFilterError``
    (fail-closed).
    """
    if not rows:
        raise RealityFilterError("build_reality_report requires a non-empty trial ledger")
    if not np.isfinite(q) or not 0.0 < float(q) < 1.0:
        raise RealityFilterError(f"q must be in (0, 1), got {q}")
    n_trials = len(rows)
    clusters = {r.cluster_id for r in rows}
    n_eff = effective_trials([sum(1 for r in rows if r.cluster_id == c) for c in clusters])
    best = max(rows, key=lambda r: (float(r.sharpe_periodic), r.trial_id))
    psr = probabilistic_sharpe(
        float(best.sharpe_periodic),
        0.0,
        int(best.n_obs),
        float(best.skew),
        float(best.kurtosis_raw),
    )
    min_trl = min_track_record_length(
        float(best.sharpe_periodic), float(best.skew), float(best.kurtosis_raw)
    )
    dsr = dsr_from_ledger(rows)

    pbo = float("nan")
    pbo_logit: list[float] = []
    spa_pvalue = float("nan")
    if returns_by_trial:
        missing = [r.trial_id for r in rows if r.trial_id not in returns_by_trial]
        if missing:
            raise RealityFilterError(
                f"returns_by_trial missing series for {len(missing)} ledger trials"
            )
        lengths = {
            int(np.asarray(v, dtype=float).reshape(-1).size) for v in returns_by_trial.values()
        }
        if len(lengths) != 1:
            raise RealityFilterError("returns_by_trial series must share one length")
        n_periods = lengths.pop()
        mat = np.column_stack(
            [np.asarray(returns_by_trial[r.trial_id], dtype=float).reshape(-1) for r in rows]
        )
        if n_periods >= 2 * int(s_blocks) and n_periods % int(s_blocks) == 0:
            pbo_res = cscv_pbo(mat, s_blocks=int(s_blocks))
            pbo = float(pbo_res["pbo"])  # type: ignore[arg-type]
            raw_logits = pbo_res["logits"]
            if not isinstance(raw_logits, list):
                raise RealityFilterError("CSCV returned malformed logits")
            pbo_logit = [float(x) for x in raw_logits]
        spa_pvalue = float(
            spa_from_trials(
                {r.trial_id: returns_by_trial[r.trial_id] for r in rows},
                n_boot=int(spa_n_boot),
                seed=int(seed),
            ).p_consistent
        )

    rejects = bh_fdr(rows, q=float(q))
    bh_fdr_rejects = rejects["calibration"] + rejects["discovery"]
    verdict = _verdict(
        n_trials=n_trials,
        n_effective=n_eff,
        dsr=dsr,
        pbo=pbo,
        spa_pvalue=spa_pvalue,
        q=float(q),
    )
    created = created_utc or datetime.now(UTC).isoformat()
    fields: dict[str, object] = {
        "schema_version": "proofcore/1",
        "created_utc": created,
        "n_trials": n_trials,
        "n_effective_trials": float(n_eff),
        "best_trial_id": best.trial_id,
        "psr": float(psr),
        "min_trl_periods": float(min_trl),
        "dsr": float(dsr),
        "pbo": pbo,
        "pbo_logit": pbo_logit,
        "spa_pvalue": spa_pvalue,
        "bh_fdr_rejects": bh_fdr_rejects,
        "fdr_q": float(q),
        "verdict": verdict,
    }
    fields["report_sha256"] = _report_hash(fields)
    return RealityReport.model_validate(fields)
