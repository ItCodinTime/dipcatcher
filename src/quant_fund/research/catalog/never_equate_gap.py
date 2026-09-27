"""Never-equate guards against gap_finite_rate.

Split out of the original module. Import the parent path; it re-exports these names.
"""

from __future__ import annotations

from .hypotheses import (
    H20_HYPOTHESIS_ID,
    H21_HYPOTHESIS_ID,
    H22_HYPOTHESIS_ID,
    H23_HYPOTHESIS_ID,
    H24_HYPOTHESIS_ID,
    H25_HYPOTHESIS_ID,
    H26_HYPOTHESIS_ID,
    H27_HYPOTHESIS_ID,
    H28_HYPOTHESIS_ID,
    H29_HYPOTHESIS_ID,
    H30_HYPOTHESIS_ID,
    H31_HYPOTHESIS_ID,
    H32_HYPOTHESIS_ID,
    H33_HYPOTHESIS_ID,
    H34_HYPOTHESIS_ID,
    H35_HYPOTHESIS_ID,
    H36_HYPOTHESIS_ID,
    H37_HYPOTHESIS_ID,
    H38_HYPOTHESIS_ID,
    H39_HYPOTHESIS_ID,
    H40_HYPOTHESIS_ID,
    H41_HYPOTHESIS_ID,
    H42_HYPOTHESIS_ID,
    H43_HYPOTHESIS_ID,
    H44_HYPOTHESIS_ID,
    H45_HYPOTHESIS_ID,
    NORTHSET_H23_H28_SPECS,
    northset_has_finite_book_uncrossed_rate,
    northset_has_finite_imbalance_p_ic,
    northset_has_finite_ohlc_identity_rate,
    northset_has_finite_session_book_vpin_p_ic,
)
from .primitives import _finite_scalar


def ohlc_identity_vs_gap_finite_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate ``ohlc_identity_rate`` with ``gap_finite_rate``.

    DATA_CONTRACTS: OHLC envelope identity ≠ overnight gap finiteness. When both
    keys are present:

    - Distinct key identity
    - H20 finite gate binds to ``ohlc_identity_rate`` only (gap has **no** H20 claim)
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_ohlc = "ohlc_identity_rate"
    k_gap = "gap_finite_rate"
    if k_ohlc not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_ohlc == k_gap:
        errs.append("ohlc_identity_gap_finite_keys_collapsed")

    # H20 watches daily ohlc only — gap must not share H20 id with session binds check
    if H20_HYPOTHESIS_ID in {H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h20_hypothesis_collapsed_into_h21_h22")

    # Positive: northset_has_finite_ohlc_identity_rate uses ohlc key, not gap
    if not northset_has_finite_ohlc_identity_rate({k_ohlc: 1.0}):
        errs.append("ohlc_identity_h20_gate_helper_broken")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    return errs


def gap_finite_rate_vs_book_uncrossed_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate ``gap_finite_rate`` with H21 ``book_uncrossed_rate``.

    DATA_CONTRACTS: overnight gap finiteness ≠ uncrossed book. When both keys
    are present:

    - Distinct key identity
    - Distinct hyp/gate: H21 binds to ``book_uncrossed_rate`` only (gap has
      **no** H21 claim); H21 id stays distinct from H20/H22
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off H20/H21/H22 triad polish (already landed). Off nest invent / kyle_ofi.
    """
    if not isinstance(blob, dict):
        return []
    k_gap = "gap_finite_rate"
    k_book = "book_uncrossed_rate"
    if k_gap not in blob or k_book not in blob:
        return []

    errs: list[str] = []
    if k_gap == k_book:
        errs.append("gap_finite_book_uncrossed_keys_collapsed")
    if H21_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h21_hypothesis_collapsed_into_h20_h22")

    elig = {"book_hypothesis_eligible": True}
    if not northset_has_finite_book_uncrossed_rate({k_book: 1.0, **elig}):
        errs.append("book_uncrossed_h21_gate_helper_broken")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def imbalance_top_p_ic_vs_gap_finite_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H22 ``imbalance_top_p_ic`` with ``gap_finite_rate``.

    DATA_CONTRACTS: imbalance discovery IC p-value ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - Distinct hyp/gate: H22 binds imbalance_top_p_ic only; gap has **no** H22
      claim and must not trigger H20/H21 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_imb = "imbalance_top_p_ic"
    k_gap = "gap_finite_rate"
    if k_imb not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_imb == k_gap:
        errs.append("imbalance_p_ic_gap_finite_keys_collapsed")
    if H22_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID}:
        errs.append("h22_hypothesis_collapsed_into_h20_or_h21")

    elig = {"book_hypothesis_eligible": True}
    if not northset_has_finite_imbalance_p_ic({k_imb: 0.05, **elig}):
        errs.append("imbalance_p_ic_h22_gate_helper_broken")
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")

    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_ohlc_identity_rate({k_imb: 0.05}):
        errs.append("imbalance_p_ic_incorrectly_triggers_h20_gate")

    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    if northset_has_finite_book_uncrossed_rate({k_imb: 0.05, **elig}):
        errs.append("imbalance_p_ic_incorrectly_triggers_h21_gate")
    return errs


def microprice_p_ic_vs_gap_finite_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H25 ``microprice_p_ic`` with ``gap_finite_rate``.

    DATA_CONTRACTS: microprice discovery IC p ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H25 binds ``microprice_p_ic`` only (gap has **no** H25 claim)
    - No dedicated microprice finite-gate helper — ``_finite_scalar`` on the
      ``microprice_p_ic`` key only (a gap-only blob must not count as finite microprice)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_mp = "microprice_p_ic"
    k_gap = "gap_finite_rate"
    if k_mp not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_mp == k_gap:
        errs.append("microprice_p_ic_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_mp) != H25_HYPOTHESIS_ID:
        errs.append("microprice_p_ic_not_bound_h25")
    if by_key.get(k_gap) == H25_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h25")
    if H25_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h25_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_mp: 0.05}.get(k_mp)):
        errs.append("microprice_p_ic_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_mp)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_microprice")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def clv_p_ic_vs_gap_finite_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H30 ``clv_p_ic`` with ``gap_finite_rate``.

    DATA_CONTRACTS: CLV discovery IC p ≠ overnight gap finiteness.
    Mirrors :func:`microprice_p_ic_vs_gap_finite_never_equate_honesty_errors`.
    When both keys are present:

    - Distinct key identity
    - H30 binds ``clv_p_ic`` only (gap has **no** H30 claim)
    - No dedicated clv finite-gate helper — ``_finite_scalar`` on the
      ``clv_p_ic`` key only (a gap-only blob must not count as finite clv)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite. Not Commander #236+; not Sergeant
    ofi↔gap; not Lt wick↔gap.
    """
    if not isinstance(blob, dict):
        return []
    k_clv = "clv_p_ic"
    k_gap = "gap_finite_rate"
    if k_clv not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_clv == k_gap:
        errs.append("clv_p_ic_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_clv) != H30_HYPOTHESIS_ID:
        errs.append("clv_p_ic_not_bound_h30")
    if by_key.get(k_gap) == H30_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h30")
    if H30_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h30_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_clv: 0.05}.get(k_clv)):
        errs.append("clv_p_ic_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_clv)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_clv")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def dm_split_vs_park_p_vs_gap_finite_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H31 ``dm_split_vs_park_p`` with ``gap_finite_rate``.

    DATA_CONTRACTS: overnight-split vs Parkinson DM p ≠ overnight gap finiteness.
    Mirrors :func:`clv_p_ic_vs_gap_finite_never_equate_honesty_errors`.
    When both keys are present:

    - Distinct key identity
    - H31 binds ``dm_split_vs_park_p`` only (gap has **no** H31 claim)
    - No dedicated dm_split finite-gate helper — ``_finite_scalar`` on the
      ``dm_split_vs_park_p`` key only (a gap-only blob must not count as finite dm)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite. Not Sergeant H28; not Lt H33 sweep.
    """
    if not isinstance(blob, dict):
        return []
    k_dm = "dm_split_vs_park_p"
    k_gap = "gap_finite_rate"
    if k_dm not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_dm == k_gap:
        errs.append("dm_split_vs_park_p_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_dm) != H31_HYPOTHESIS_ID:
        errs.append("dm_split_vs_park_p_not_bound_h31")
    if by_key.get(k_gap) == H31_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h31")
    if H31_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h31_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_dm: 0.05}.get(k_dm)):
        errs.append("dm_split_vs_park_p_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_dm)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_dm_split")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def sweep_follow_event_p_vs_gap_finite_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H36 ``sweep_follow_event_p`` with ``gap_finite_rate``.

    DATA_CONTRACTS: sweep-follow event p ≠ overnight gap finiteness.
    Mirrors :func:`clv_p_ic_vs_gap_finite_never_equate_honesty_errors`.
    When both keys are present:

    - Distinct key identity
    - H36 binds ``sweep_follow_event_p`` only (gap has **no** H36 claim)
    - No dedicated sweep_follow_event finite-gate helper — ``_finite_scalar`` on
      the ``sweep_follow_event_p`` key only (a gap-only blob must not count as
      finite sweep_follow_event_p)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite. Not Sergeant H35; not Lt H34.
    """
    if not isinstance(blob, dict):
        return []
    k_ev = "sweep_follow_event_p"
    k_gap = "gap_finite_rate"
    if k_ev not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_ev == k_gap:
        errs.append("sweep_follow_event_p_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_ev) != H36_HYPOTHESIS_ID:
        errs.append("sweep_follow_event_p_not_bound_h36")
    if by_key.get(k_gap) == H36_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h36")
    if H36_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h36_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_ev: 0.05}.get(k_ev)):
        errs.append("sweep_follow_event_p_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_ev)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_sweep_follow_event_p")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def sweep_reject_cost_adjusted_mean_bps_vs_gap_finite_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H39 ``sweep_reject_cost_adjusted_mean_bps`` with ``gap_finite_rate``.

    DATA_CONTRACTS: sweep-reject cost-adjusted mean bps ≠ overnight gap finiteness.
    Mirrors :func:`sweep_follow_event_p_vs_gap_finite_never_equate_honesty_errors`.
    When both keys are present:

    - Distinct key identity
    - H39 binds ``sweep_reject_cost_adjusted_mean_bps`` only (gap has **no** H39 claim)
    - No dedicated reject-cost finite-gate helper — ``_finite_scalar`` on the
      ``sweep_reject_cost_adjusted_mean_bps`` key only (a gap-only blob must not
      count as finite reject-cost bps)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite. Not Sergeant H38.
    """
    if not isinstance(blob, dict):
        return []
    k_cost = "sweep_reject_cost_adjusted_mean_bps"
    k_gap = "gap_finite_rate"
    if k_cost not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_cost == k_gap:
        errs.append("sweep_reject_cost_adjusted_mean_bps_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_cost) != H39_HYPOTHESIS_ID:
        errs.append("sweep_reject_cost_adjusted_mean_bps_not_bound_h39")
    if by_key.get(k_gap) == H39_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h39")
    if H39_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h39_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_cost: 1.0}.get(k_cost)):
        errs.append("sweep_reject_cost_adjusted_mean_bps_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_cost)):
        errs.append(
            "gap_finite_rate_incorrectly_counts_as_finite_sweep_reject_cost_adjusted_mean_bps"
        )

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def sweep_reject_control_diff_p_vs_gap_finite_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H44 ``sweep_reject_control_diff_p`` with ``gap_finite_rate``.

    DATA_CONTRACTS: sweep-reject control-diff p ≠ overnight gap finiteness.
    Mirrors :func:`sweep_reject_cost_adjusted_mean_bps_vs_gap_finite_never_equate_honesty_errors`.
    When both keys are present:

    - Distinct key identity
    - H44 binds ``sweep_reject_control_diff_p`` only (gap has **no** H44 claim)
    - No dedicated reject-control finite-gate helper — ``_finite_scalar`` on the
      ``sweep_reject_control_diff_p`` key only (a gap-only blob must not count as
      finite reject-control p)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite. Not Sergeant H42.
    """
    if not isinstance(blob, dict):
        return []
    k_ctrl = "sweep_reject_control_diff_p"
    k_gap = "gap_finite_rate"
    if k_ctrl not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_ctrl == k_gap:
        errs.append("sweep_reject_control_diff_p_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_ctrl) != H44_HYPOTHESIS_ID:
        errs.append("sweep_reject_control_diff_p_not_bound_h44")
    if by_key.get(k_gap) == H44_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h44")
    if H44_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h44_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_ctrl: 0.05}.get(k_ctrl)):
        errs.append("sweep_reject_control_diff_p_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_ctrl)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_sweep_reject_control_diff_p")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def ofi_p_ic_vs_gap_finite_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H27 ``ofi_p_ic`` with ``gap_finite_rate``.

    DATA_CONTRACTS: OFI discovery IC p ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H27 binds ``ofi_p_ic`` only (gap has **no** H27 claim)
    - No dedicated ofi finite-gate helper — ``_finite_scalar`` on the
      ``ofi_p_ic`` key only (a gap-only blob must not count as finite ofi)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite. Not Commander #211+ props.
    """
    if not isinstance(blob, dict):
        return []
    k_ofi = "ofi_p_ic"
    k_gap = "gap_finite_rate"
    if k_ofi not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_ofi == k_gap:
        errs.append("ofi_p_ic_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_ofi) != H27_HYPOTHESIS_ID:
        errs.append("ofi_p_ic_not_bound_h27")
    if by_key.get(k_gap) == H27_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h27")
    if H27_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h27_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_ofi: 0.05}.get(k_ofi)):
        errs.append("ofi_p_ic_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_ofi)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_ofi")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def dm_gk_vs_park_p_vs_gap_finite_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H28 ``dm_gk_vs_park_p`` with ``gap_finite_rate``.

    DATA_CONTRACTS: GK vs Park DM p-value ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H28 binds ``dm_gk_vs_park_p`` only (gap has **no** H28 claim)
    - No dedicated DM finite-gate helper — ``_finite_scalar`` on the
      ``dm_gk_vs_park_p`` key only (a gap-only blob must not count as finite DM p)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off CoS dm_split H31; off Lt sweep H33; off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_dm = "dm_gk_vs_park_p"
    k_gap = "gap_finite_rate"
    if k_dm not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_dm == k_gap:
        errs.append("dm_gk_vs_park_p_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_dm) != H28_HYPOTHESIS_ID:
        errs.append("dm_gk_vs_park_p_not_bound_h28")
    if by_key.get(k_gap) == H28_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h28")
    if H28_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h28_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_dm: 0.05}.get(k_dm)):
        errs.append("dm_gk_vs_park_p_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_dm)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_dm_gk")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def sweep_reject_event_p_vs_gap_finite_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H35 ``sweep_reject_event_p`` with ``gap_finite_rate``.

    DATA_CONTRACTS: sweep reject-event IC p ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H35 binds ``sweep_reject_event_p`` only (gap has **no** H35 claim)
    - No dedicated sweep-p finite-gate helper — ``_finite_scalar`` on the
      ``sweep_reject_event_p`` key only (a gap-only blob must not count as finite reject p)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off CoS H36; off Lt H34; off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_sw = "sweep_reject_event_p"
    k_gap = "gap_finite_rate"
    if k_sw not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_sw == k_gap:
        errs.append("sweep_reject_event_p_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_sw) != H35_HYPOTHESIS_ID:
        errs.append("sweep_reject_event_p_not_bound_h35")
    if by_key.get(k_gap) == H35_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h35")
    if H35_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h35_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_sw: 0.05}.get(k_sw)):
        errs.append("sweep_reject_event_p_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_sw)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_sweep_reject_event_p")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def sweep_follow_placebo_p_vs_gap_finite_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H38 ``sweep_follow_placebo_p`` with ``gap_finite_rate``.

    DATA_CONTRACTS: sweep follow-placebo IC p ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H38 binds ``sweep_follow_placebo_p`` only (gap has **no** H38 claim)
    - No dedicated sweep-p finite-gate helper — ``_finite_scalar`` on the
      ``sweep_follow_placebo_p`` key only (a gap-only blob must not count as finite placebo p)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off CoS H39 cost; off Lt H37; off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_sw = "sweep_follow_placebo_p"
    k_gap = "gap_finite_rate"
    if k_sw not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_sw == k_gap:
        errs.append("sweep_follow_placebo_p_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_sw) != H38_HYPOTHESIS_ID:
        errs.append("sweep_follow_placebo_p_not_bound_h38")
    if by_key.get(k_gap) == H38_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h38")
    if H38_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h38_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_sw: 0.05}.get(k_sw)):
        errs.append("sweep_follow_placebo_p_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_sw)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_sweep_follow_placebo_p")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def sweep_follow_fold_positive_fraction_vs_gap_finite_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H42 ``sweep_follow_fold_positive_fraction`` with ``gap_finite_rate``.

    DATA_CONTRACTS: sweep follow fold-positive fraction ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H42 binds ``sweep_follow_fold_positive_fraction`` only (gap has **no** H42 claim)
    - No dedicated fold-fraction finite-gate helper — ``_finite_scalar`` on the
      fold key only (a gap-only blob must not count as finite fold fraction)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off CoS H44; off Lt H41; off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_sw = "sweep_follow_fold_positive_fraction"
    k_gap = "gap_finite_rate"
    if k_sw not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_sw == k_gap:
        errs.append("sweep_follow_fold_positive_fraction_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_sw) != H42_HYPOTHESIS_ID:
        errs.append("sweep_follow_fold_positive_fraction_not_bound_h42")
    if by_key.get(k_gap) == H42_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h42")
    if H42_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h42_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_sw: 0.05}.get(k_sw)):
        errs.append("sweep_follow_fold_positive_fraction_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_sw)):
        errs.append(
            "gap_finite_rate_incorrectly_counts_as_finite_sweep_follow_fold_positive_fraction"
        )

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def wick_skew_p_ic_vs_gap_finite_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H26 ``wick_skew_p_ic`` with ``gap_finite_rate``.

    DATA_CONTRACTS: wick-skew discovery IC p ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H26 binds ``wick_skew_p_ic`` only (gap has **no** H26 claim)
    - Finite probe on ``wick_skew_p_ic`` only (gap-only blob must not count)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_wick = "wick_skew_p_ic"
    k_gap = "gap_finite_rate"
    if k_wick not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_wick == k_gap:
        errs.append("wick_skew_p_ic_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_wick) != H26_HYPOTHESIS_ID:
        errs.append("wick_skew_p_ic_not_bound_h26")
    if by_key.get(k_gap) == H26_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h26")
    if H26_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h26_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_wick: 0.05}.get(k_wick)):
        errs.append("wick_skew_p_ic_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_wick)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_wick_skew")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def vpin_p_ic_vs_gap_finite_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H32 ``vpin_p_ic`` with ``gap_finite_rate``.

    DATA_CONTRACTS: VPIN discovery IC p ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H32 binds ``vpin_p_ic`` only (gap has **no** H32 claim)
    - Finite probe on ``vpin_p_ic`` only (gap-only blob must not count)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_vpin = "vpin_p_ic"
    k_gap = "gap_finite_rate"
    if k_vpin not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_vpin == k_gap:
        errs.append("vpin_p_ic_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_vpin) != H32_HYPOTHESIS_ID:
        errs.append("vpin_p_ic_not_bound_h32")
    if by_key.get(k_gap) == H32_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h32")
    if H32_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h32_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_vpin: 0.05}.get(k_vpin)):
        errs.append("vpin_p_ic_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_vpin)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_vpin")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def sweep_reject_signed_p_ic_vs_gap_finite_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H33 ``sweep_reject_signed_p_ic`` with ``gap_finite_rate``.

    DATA_CONTRACTS: sweep-reject discovery IC p ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H33 binds ``sweep_reject_signed_p_ic`` only (gap has **no** H33 claim)
    - Finite probe on sweep-reject key only (gap-only blob must not count)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_sw = "sweep_reject_signed_p_ic"
    k_gap = "gap_finite_rate"
    if k_sw not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_sw == k_gap:
        errs.append("sweep_reject_signed_p_ic_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_sw) != H33_HYPOTHESIS_ID:
        errs.append("sweep_reject_signed_p_ic_not_bound_h33")
    if by_key.get(k_gap) == H33_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h33")
    if H33_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h33_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_sw: 0.05}.get(k_sw)):
        errs.append("sweep_reject_signed_p_ic_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_sw)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_sweep_reject")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def sweep_follow_signed_p_ic_vs_gap_finite_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H34 ``sweep_follow_signed_p_ic`` with ``gap_finite_rate``.

    DATA_CONTRACTS: sweep-follow discovery IC p ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H34 binds ``sweep_follow_signed_p_ic`` only (gap has **no** H34 claim)
    - Finite probe on sweep-follow key only (gap-only blob must not count)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_sw = "sweep_follow_signed_p_ic"
    k_gap = "gap_finite_rate"
    if k_sw not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_sw == k_gap:
        errs.append("sweep_follow_signed_p_ic_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_sw) != H34_HYPOTHESIS_ID:
        errs.append("sweep_follow_signed_p_ic_not_bound_h34")
    if by_key.get(k_gap) == H34_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h34")
    if H34_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h34_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_sw: 0.05}.get(k_sw)):
        errs.append("sweep_follow_signed_p_ic_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_sw)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_sweep_follow")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def sweep_reject_placebo_p_vs_gap_finite_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H37 ``sweep_reject_placebo_p`` with ``gap_finite_rate``.

    DATA_CONTRACTS: sweep-reject placebo p ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H37 binds ``sweep_reject_placebo_p`` only (gap has **no** H37 claim)
    - Finite probe on placebo key only (gap-only blob must not count)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_sw = "sweep_reject_placebo_p"
    k_gap = "gap_finite_rate"
    if k_sw not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_sw == k_gap:
        errs.append("sweep_reject_placebo_p_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_sw) != H37_HYPOTHESIS_ID:
        errs.append("sweep_reject_placebo_p_not_bound_h37")
    if by_key.get(k_gap) == H37_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h37")
    if H37_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h37_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_sw: 0.05}.get(k_sw)):
        errs.append("sweep_reject_placebo_p_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_sw)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_sweep_reject_placebo")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def sweep_follow_cost_adjusted_mean_bps_vs_gap_finite_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H40 ``sweep_follow_cost_adjusted_mean_bps`` with ``gap_finite_rate``.

    DATA_CONTRACTS: sweep-follow cost-adjusted mean bps ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H40 binds cost-adjusted follow mean only (gap has **no** H40 claim)
    - Finite probe on cost key only (gap-only blob must not count)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_sw = "sweep_follow_cost_adjusted_mean_bps"
    k_gap = "gap_finite_rate"
    if k_sw not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_sw == k_gap:
        errs.append("sweep_follow_cost_adjusted_mean_bps_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_sw) != H40_HYPOTHESIS_ID:
        errs.append("sweep_follow_cost_adjusted_mean_bps_not_bound_h40")
    if by_key.get(k_gap) == H40_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h40")
    if H40_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h40_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_sw: 0.05}.get(k_sw)):
        errs.append("sweep_follow_cost_adjusted_mean_bps_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_sw)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_sweep_follow_cost")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def sweep_reject_fold_positive_fraction_vs_gap_finite_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H41 ``sweep_reject_fold_positive_fraction`` with ``gap_finite_rate``.

    DATA_CONTRACTS: sweep-reject fold-positive fraction ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H41 binds fold-positive reject fraction only (gap has **no** H41 claim)
    - Finite probe on fold key only (gap-only blob must not count)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_sw = "sweep_reject_fold_positive_fraction"
    k_gap = "gap_finite_rate"
    if k_sw not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_sw == k_gap:
        errs.append("sweep_reject_fold_positive_fraction_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_sw) != H41_HYPOTHESIS_ID:
        errs.append("sweep_reject_fold_positive_fraction_not_bound_h41")
    if by_key.get(k_gap) == H41_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h41")
    if H41_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h41_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_sw: 0.05}.get(k_sw)):
        errs.append("sweep_reject_fold_positive_fraction_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_sw)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_sweep_reject_fold")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def sweep_follow_control_diff_p_vs_gap_finite_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H45 ``sweep_follow_control_diff_p`` with ``gap_finite_rate``.

    DATA_CONTRACTS: sweep-follow control-diff p ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H45 binds follow control-diff p only (gap has **no** H45 claim)
    - Finite probe on control key only (gap-only blob must not count)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_sw = "sweep_follow_control_diff_p"
    k_gap = "gap_finite_rate"
    if k_sw not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_sw == k_gap:
        errs.append("sweep_follow_control_diff_p_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_sw) != H45_HYPOTHESIS_ID:
        errs.append("sweep_follow_control_diff_p_not_bound_h45")
    if by_key.get(k_gap) == H45_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h45")
    if H45_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h45_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_sw: 0.05}.get(k_sw)):
        errs.append("sweep_follow_control_diff_p_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_sw)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_sweep_follow_control")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def session_book_vpin_p_ic_vs_gap_finite_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H43 ``session_book_vpin_p_ic`` with ``gap_finite_rate``.

    DATA_CONTRACTS: session-book VPIN discovery IC p ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H43 gate helper binds session_book_vpin_p_ic only (gap must not trigger it)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_vpin = "session_book_vpin_p_ic"
    k_gap = "gap_finite_rate"
    if k_vpin not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_vpin == k_gap:
        errs.append("session_book_vpin_p_ic_gap_finite_keys_collapsed")

    if H43_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h43_hypothesis_collapsed_into_h20_h21_h22")

    elig = {"session_book_hypothesis_eligible": True, "book_hypothesis_eligible": True}
    if not northset_has_finite_session_book_vpin_p_ic({k_vpin: 0.05, **elig}):
        errs.append("session_book_vpin_p_ic_h43_gate_helper_broken")
    if northset_has_finite_session_book_vpin_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h43_gate")

    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def gap_finite_rate_vs_session_reconstructs_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate ``gap_finite_rate`` with H23 ``session_reconstructs_daily_rate``.

    DATA_CONTRACTS: overnight gap finiteness ≠ session→daily reconstruction.
    When both keys are present:

    - Distinct key identity
    - Hyp/gate: reconstructs binds ``H23_HYPOTHESIS_ID``; gap is **not** H23
      and must not trigger H21/H20 gates
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off H20/H21/H22 triad. Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_gap = "gap_finite_rate"
    k_recon = "session_reconstructs_daily_rate"
    if k_gap not in blob or k_recon not in blob:
        return []

    errs: list[str] = []
    if k_gap == k_recon:
        errs.append("gap_finite_session_reconstructs_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_recon) != H23_HYPOTHESIS_ID:
        errs.append("session_reconstructs_daily_rate_not_bound_h23")
    if by_key.get(k_gap) == H23_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h23")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    if northset_has_finite_book_uncrossed_rate({k_recon: 1.0, **elig}):
        errs.append("session_reconstructs_incorrectly_triggers_h21_gate")

    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_ohlc_identity_rate({k_recon: 1.0}):
        errs.append("session_reconstructs_incorrectly_triggers_h20_gate")

    if H23_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID}:
        errs.append("h23_hypothesis_collapsed_into_h20_or_h21")
    return errs


def gap_finite_rate_vs_session_volume_conservation_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate ``gap_finite_rate`` with H24 ``session_volume_conservation_rate``.

    DATA_CONTRACTS: overnight gap finiteness ≠ session volume conservation.
    When both keys are present:

    - Distinct key identity
    - Hyp/gate: conservation binds ``H24_HYPOTHESIS_ID``; gap is **not** H24
      and must not trigger H21/H20 gates
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off H20/H21/H22 triad. Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_gap = "gap_finite_rate"
    k_vol = "session_volume_conservation_rate"
    if k_gap not in blob or k_vol not in blob:
        return []

    errs: list[str] = []
    if k_gap == k_vol:
        errs.append("gap_finite_session_volume_conservation_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_vol) != H24_HYPOTHESIS_ID:
        errs.append("session_volume_conservation_rate_not_bound_h24")
    if by_key.get(k_gap) == H24_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h24")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    if northset_has_finite_book_uncrossed_rate({k_vol: 1.0, **elig}):
        errs.append("session_volume_conservation_incorrectly_triggers_h21_gate")

    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_ohlc_identity_rate({k_vol: 1.0}):
        errs.append("session_volume_conservation_incorrectly_triggers_h20_gate")

    if H24_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID}:
        errs.append("h24_hypothesis_collapsed_into_h20_or_h21")
    return errs


def gap_finite_rate_vs_session_chain_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate ``gap_finite_rate`` with H29 ``session_chain_rate``.

    DATA_CONTRACTS: overnight gap finiteness ≠ session chain identity. When both
    keys are present:

    - Distinct key identity
    - Hyp/gate: chain binds ``H29_HYPOTHESIS_ID``; gap is **not** H29 and must
      not trigger H21/H20 gates
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off H20/H21/H22 triad. Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_gap = "gap_finite_rate"
    k_chain = "session_chain_rate"
    if k_gap not in blob or k_chain not in blob:
        return []

    errs: list[str] = []
    if k_gap == k_chain:
        errs.append("gap_finite_session_chain_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_chain) != H29_HYPOTHESIS_ID:
        errs.append("session_chain_rate_not_bound_h29")
    if by_key.get(k_gap) == H29_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h29")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    if northset_has_finite_book_uncrossed_rate({k_chain: 1.0, **elig}):
        errs.append("session_chain_incorrectly_triggers_h21_gate")

    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_ohlc_identity_rate({k_chain: 1.0}):
        errs.append("session_chain_incorrectly_triggers_h20_gate")

    if H29_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID}:
        errs.append("h29_hypothesis_collapsed_into_h20_or_h21")
    return errs


def session_ohlc_vs_gap_finite_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate ``session_ohlc_identity_rate`` with ``gap_finite_rate``.

    DATA_CONTRACTS: session-candle OHLC identity ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - Hyp/gate: session OHLC is **not** H23; gap has **no** H21 claim;
      H20 gate binds daily ``ohlc_identity_rate`` only (neither key is H20)
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off H20/H21/H22 triad. Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_sess = "session_ohlc_identity_rate"
    k_gap = "gap_finite_rate"
    if k_sess not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_sess == k_gap:
        errs.append("session_ohlc_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_sess) == H23_HYPOTHESIS_ID:
        errs.append("session_ohlc_identity_rate_incorrectly_bound_h23")
    if by_key.get(k_gap) == H23_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h23")

    # H21 is book_uncrossed — gap must not share that gate
    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    if northset_has_finite_book_uncrossed_rate({k_sess: 1.0, **elig}):
        errs.append("session_ohlc_incorrectly_triggers_h21_gate")

    # H20 is daily ohlc only — neither session nor gap is H20
    if northset_has_finite_ohlc_identity_rate({k_sess: 1.0}):
        errs.append("session_ohlc_incorrectly_triggers_h20_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")

    if H20_HYPOTHESIS_ID in {H23_HYPOTHESIS_ID, H21_HYPOTHESIS_ID}:
        errs.append("h20_hypothesis_collapsed_into_session_or_book_binds")
    return errs


__all__ = [
    "clv_p_ic_vs_gap_finite_never_equate_honesty_errors",
    "dm_gk_vs_park_p_vs_gap_finite_never_equate_honesty_errors",
    "dm_split_vs_park_p_vs_gap_finite_never_equate_honesty_errors",
    "gap_finite_rate_vs_book_uncrossed_never_equate_honesty_errors",
    "gap_finite_rate_vs_session_chain_never_equate_honesty_errors",
    "gap_finite_rate_vs_session_reconstructs_never_equate_honesty_errors",
    "gap_finite_rate_vs_session_volume_conservation_never_equate_honesty_errors",
    "imbalance_top_p_ic_vs_gap_finite_never_equate_honesty_errors",
    "microprice_p_ic_vs_gap_finite_never_equate_honesty_errors",
    "ofi_p_ic_vs_gap_finite_never_equate_honesty_errors",
    "ohlc_identity_vs_gap_finite_never_equate_honesty_errors",
    "session_book_vpin_p_ic_vs_gap_finite_never_equate_honesty_errors",
    "session_ohlc_vs_gap_finite_never_equate_honesty_errors",
    "sweep_follow_control_diff_p_vs_gap_finite_never_equate_honesty_errors",
    "sweep_follow_cost_adjusted_mean_bps_vs_gap_finite_never_equate_honesty_errors",
    "sweep_follow_event_p_vs_gap_finite_never_equate_honesty_errors",
    "sweep_follow_fold_positive_fraction_vs_gap_finite_never_equate_honesty_errors",
    "sweep_follow_placebo_p_vs_gap_finite_never_equate_honesty_errors",
    "sweep_follow_signed_p_ic_vs_gap_finite_never_equate_honesty_errors",
    "sweep_reject_control_diff_p_vs_gap_finite_never_equate_honesty_errors",
    "sweep_reject_cost_adjusted_mean_bps_vs_gap_finite_never_equate_honesty_errors",
    "sweep_reject_event_p_vs_gap_finite_never_equate_honesty_errors",
    "sweep_reject_fold_positive_fraction_vs_gap_finite_never_equate_honesty_errors",
    "sweep_reject_placebo_p_vs_gap_finite_never_equate_honesty_errors",
    "sweep_reject_signed_p_ic_vs_gap_finite_never_equate_honesty_errors",
    "vpin_p_ic_vs_gap_finite_never_equate_honesty_errors",
    "wick_skew_p_ic_vs_gap_finite_never_equate_honesty_errors",
]
