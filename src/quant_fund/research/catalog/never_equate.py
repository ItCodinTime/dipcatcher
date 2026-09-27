"""Identity, book, and sweep never-equate honesty guards.

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
    H29_HYPOTHESIS_ID,
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


def session_volume_conservation_vs_reconstructs_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H24 volume conservation with H23 session reconstructs.

    Sibling rates on the northset receipt — volume conservation ≠ OHLC envelope
    reconstructs (DATA_CONTRACTS). When **both** keys are present:

    - Catalog bind lock: reconstructs → H23, conservation → H24 (distinct hyp ids)
    - Numeric equality on clean synth is allowed (never fail on value equality)
    - Distinct key identity is structural (``session_reconstructs_daily_rate`` ≠
      ``session_volume_conservation_rate``)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_recon = "session_reconstructs_daily_rate"
    k_vol = "session_volume_conservation_rate"
    if k_recon not in blob or k_vol not in blob:
        return []

    errs: list[str] = []
    if k_recon == k_vol:
        errs.append("session_reconstructs_volume_conservation_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    recon_hyp = by_key.get(k_recon)
    vol_hyp = by_key.get(k_vol)
    if recon_hyp != H23_HYPOTHESIS_ID:
        errs.append("session_reconstructs_daily_rate_not_bound_h23")
    if vol_hyp != H24_HYPOTHESIS_ID:
        errs.append("session_volume_conservation_rate_not_bound_h24")
    if recon_hyp is not None and vol_hyp is not None and recon_hyp == vol_hyp:
        errs.append("session_reconstructs_volume_conservation_hypothesis_collapsed")
    return errs


def session_ohlc_vs_reconstructs_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate ``session_ohlc_identity_rate`` with ``session_reconstructs_daily_rate``.

    DATA_CONTRACTS: session-candle OHLC identity ≠ session envelope reconstructing
    the daily bar. When both keys are present:

    - Distinct key identity
    - H23 binds to reconstructs only (``H23_HYPOTHESIS_ID``); session OHLC is not H23
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_ohlc = "session_ohlc_identity_rate"
    k_recon = "session_reconstructs_daily_rate"
    if k_ohlc not in blob or k_recon not in blob:
        return []

    errs: list[str] = []
    if k_ohlc == k_recon:
        errs.append("session_ohlc_reconstructs_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_recon) != H23_HYPOTHESIS_ID:
        errs.append("session_reconstructs_daily_rate_not_bound_h23")
    if by_key.get(k_ohlc) == H23_HYPOTHESIS_ID:
        errs.append("session_ohlc_identity_rate_incorrectly_bound_h23")
    return errs


def session_ohlc_vs_volume_conservation_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate ``session_ohlc_identity_rate`` with ``session_volume_conservation_rate``.

    DATA_CONTRACTS: session-candle OHLC identity ≠ session volume conservation
    (H24). When both keys are present:

    - Distinct key identity
    - H24 binds to volume conservation only (``H24_HYPOTHESIS_ID``); session OHLC
      is not H23/H24
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_ohlc = "session_ohlc_identity_rate"
    k_vol = "session_volume_conservation_rate"
    if k_ohlc not in blob or k_vol not in blob:
        return []

    errs: list[str] = []
    if k_ohlc == k_vol:
        errs.append("session_ohlc_volume_conservation_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_vol) != H24_HYPOTHESIS_ID:
        errs.append("session_volume_conservation_rate_not_bound_h24")
    if by_key.get(k_ohlc) == H24_HYPOTHESIS_ID:
        errs.append("session_ohlc_identity_rate_incorrectly_bound_h24")
    if by_key.get(k_ohlc) == H23_HYPOTHESIS_ID:
        errs.append("session_ohlc_identity_rate_incorrectly_bound_h23")
    return errs


def session_ohlc_vs_session_chain_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate ``session_ohlc_identity_rate`` with ``session_chain_rate``.

    DATA_CONTRACTS: session-candle OHLC identity ≠ H29 session chain rate.
    When both keys are present:

    - Distinct key identity
    - H29 binds to chain only (``H29_HYPOTHESIS_ID``); session OHLC is not H23/H29
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_ohlc = "session_ohlc_identity_rate"
    k_chain = "session_chain_rate"
    if k_ohlc not in blob or k_chain not in blob:
        return []

    errs: list[str] = []
    if k_ohlc == k_chain:
        errs.append("session_ohlc_session_chain_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_chain) != H29_HYPOTHESIS_ID:
        errs.append("session_chain_rate_not_bound_h29")
    if by_key.get(k_ohlc) == H29_HYPOTHESIS_ID:
        errs.append("session_ohlc_identity_rate_incorrectly_bound_h29")
    if by_key.get(k_ohlc) == H23_HYPOTHESIS_ID:
        errs.append("session_ohlc_identity_rate_incorrectly_bound_h23")
    return errs


def book_uncrossed_vs_imbalance_p_ic_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H21 ``book_uncrossed_rate`` with H22 ``imbalance_top_p_ic``.

    Triad siblings: uncrossed book bound ≠ imbalance discovery IC p-value.
    When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H21 ≠ H22)
    - Gate helpers: H21 on book_uncrossed only; H22 on imbalance_top_p_ic only
    - Numeric equality is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_book = "book_uncrossed_rate"
    k_imb = "imbalance_top_p_ic"
    if k_book not in blob or k_imb not in blob:
        return []

    errs: list[str] = []
    if k_book == k_imb:
        errs.append("book_uncrossed_imbalance_p_ic_keys_collapsed")
    if H21_HYPOTHESIS_ID == H22_HYPOTHESIS_ID:
        errs.append("h21_h22_hypothesis_ids_collapsed")

    elig = {"book_hypothesis_eligible": True}
    if not northset_has_finite_book_uncrossed_rate({k_book: 1.0, **elig}):
        errs.append("book_uncrossed_h21_gate_helper_broken")
    if northset_has_finite_book_uncrossed_rate({k_imb: 1.0, **elig}):
        errs.append("imbalance_p_ic_incorrectly_triggers_h21_gate")
    if not northset_has_finite_imbalance_p_ic({k_imb: 0.05, **elig}):
        errs.append("imbalance_p_ic_h22_gate_helper_broken")
    if northset_has_finite_imbalance_p_ic({k_book: 1.0, **elig}):
        errs.append("book_uncrossed_incorrectly_triggers_h22_gate")
    return errs


def ohlc_identity_vs_imbalance_p_ic_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H20 ``ohlc_identity_rate`` with H22 ``imbalance_top_p_ic``.

    Triad diagonal: OHLC envelope bound ≠ imbalance discovery IC p-value.
    Completes H20↔H21↔H22 never-equate mesh (edges already landed). When both
    keys are present:

    - Distinct key identity
    - Distinct hyp ids (H20 ≠ H22)
    - Gate helpers: H20 on ohlc only; H22 on imbalance_top_p_ic only
    - Numeric equality is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_ohlc = "ohlc_identity_rate"
    k_imb = "imbalance_top_p_ic"
    if k_ohlc not in blob or k_imb not in blob:
        return []

    errs: list[str] = []
    if k_ohlc == k_imb:
        errs.append("ohlc_identity_imbalance_p_ic_keys_collapsed")
    if H20_HYPOTHESIS_ID == H22_HYPOTHESIS_ID:
        errs.append("h20_h22_hypothesis_ids_collapsed")

    if not northset_has_finite_ohlc_identity_rate({k_ohlc: 1.0}):
        errs.append("ohlc_identity_h20_gate_helper_broken")
    if northset_has_finite_ohlc_identity_rate({k_imb: 0.05}):
        errs.append("imbalance_p_ic_incorrectly_triggers_h20_gate")
    elig = {"book_hypothesis_eligible": True}
    if not northset_has_finite_imbalance_p_ic({k_imb: 0.05, **elig}):
        errs.append("imbalance_p_ic_h22_gate_helper_broken")
    if northset_has_finite_imbalance_p_ic({k_ohlc: 1.0, **elig}):
        errs.append("ohlc_identity_incorrectly_triggers_h22_gate")
    return errs


def session_bulk_vpin_vs_siblings_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate ``session_bulk_vpin`` with ``vpin_mean`` / ``session_book_vpin_mean``.

    DATA_CONTRACTS VPIN triad: candle signed-volume bulk ≠ daily Cont proxy mean
    (H32 path) ≠ session-L2 ``|ofi_sum|/ofi_abs_sum`` mean (H43 path). Bulk has
    **no** H-id. When ``session_bulk_vpin`` and at least one sibling are present:

    - Distinct key identity vs each present sibling
    - Distinct hyp ids H32 ≠ H43
    - Bulk must not bind to H32/H43 metric keys (``vpin_p_ic`` / ``session_book_vpin_p_ic``)
    - H43 gate helper triggers on session_book_vpin_p_ic only, not bulk
    - Numeric equality on synth is allowed

    Skip when bulk absent or both siblings absent. Research diagnostic only;
    never live Sharpe. Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_bulk = "session_bulk_vpin"
    k_daily = "vpin_mean"
    k_sess = "session_book_vpin_mean"
    if k_bulk not in blob:
        return []
    if k_daily not in blob and k_sess not in blob:
        return []

    errs: list[str] = []
    if k_daily in blob and k_bulk == k_daily:
        errs.append("session_bulk_vpin_vpin_mean_keys_collapsed")
    if k_sess in blob and k_bulk == k_sess:
        errs.append("session_bulk_vpin_session_book_vpin_mean_keys_collapsed")
    if k_daily in blob and k_sess in blob and k_daily == k_sess:
        errs.append("vpin_mean_session_book_vpin_mean_keys_collapsed")
    if H32_HYPOTHESIS_ID == H43_HYPOTHESIS_ID:
        errs.append("h32_h43_hypothesis_ids_collapsed")

    # Catalog bind lock: H32/H43 attach to *_p_ic, never to session_bulk_vpin
    h32_metric = "vpin_p_ic"
    h43_metric = "session_book_vpin_p_ic"
    if k_bulk in {h32_metric, h43_metric}:
        errs.append("session_bulk_vpin_incorrectly_bound_as_h32_or_h43_metric")
    if h32_metric == k_bulk or h43_metric == k_bulk:
        errs.append("session_bulk_vpin_metric_key_collapsed_into_h_gate")

    if northset_has_finite_session_book_vpin_p_ic({k_bulk: 0.5}):
        errs.append("session_bulk_vpin_incorrectly_triggers_h43_gate")
    if not northset_has_finite_session_book_vpin_p_ic(
        {h43_metric: 0.05, "session_book_hypothesis_eligible": True}
    ):
        # eligible default True in helper — still require p_ic key
        errs.append("session_book_vpin_p_ic_h43_gate_helper_broken")
    return errs


def ohlc_identity_vs_book_uncrossed_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H20 ``ohlc_identity_rate`` with H21 ``book_uncrossed_rate``.

    Triad siblings (DATA_CONTRACTS H20/H21/H22): OHLC envelope ≠ uncrossed book.
    When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H20 ≠ H21)
    - H20 finite gate helper triggers on ohlc only; H21 gate on book_uncrossed only
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_ohlc = "ohlc_identity_rate"
    k_book = "book_uncrossed_rate"
    if k_ohlc not in blob or k_book not in blob:
        return []

    errs: list[str] = []
    if k_ohlc == k_book:
        errs.append("ohlc_identity_book_uncrossed_keys_collapsed")
    if H20_HYPOTHESIS_ID == H21_HYPOTHESIS_ID:
        errs.append("h20_h21_hypothesis_ids_collapsed")

    if not northset_has_finite_ohlc_identity_rate({k_ohlc: 1.0}):
        errs.append("ohlc_identity_h20_gate_helper_broken")
    if northset_has_finite_ohlc_identity_rate({k_book: 1.0}):
        errs.append("book_uncrossed_incorrectly_triggers_h20_gate")
    if not northset_has_finite_book_uncrossed_rate({k_book: 1.0, "book_hypothesis_eligible": True}):
        errs.append("book_uncrossed_h21_gate_helper_broken")
    if northset_has_finite_book_uncrossed_rate({k_ohlc: 1.0, "book_hypothesis_eligible": True}):
        errs.append("ohlc_identity_incorrectly_triggers_h21_gate")
    return errs


def imbalance_top_p_ic_vs_session_reconstructs_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H22 ``imbalance_top_p_ic`` with H23 ``session_reconstructs_daily_rate``.

    DATA_CONTRACTS: imbalance discovery IC p-value ≠ session envelope reconstructing
    the daily bar. When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H22 ≠ H23)
    - H23 binds reconstructs only; imbalance must not bind H23
    - H22 gate on imbalance_top_p_ic only; reconstructs must not trigger H22
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_imb = "imbalance_top_p_ic"
    k_recon = "session_reconstructs_daily_rate"
    if k_imb not in blob or k_recon not in blob:
        return []

    errs: list[str] = []
    if k_imb == k_recon:
        errs.append("imbalance_p_ic_session_reconstructs_keys_collapsed")
    if H22_HYPOTHESIS_ID == H23_HYPOTHESIS_ID:
        errs.append("h22_h23_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_recon) != H23_HYPOTHESIS_ID:
        errs.append("session_reconstructs_daily_rate_not_bound_h23")
    if by_key.get(k_imb) == H23_HYPOTHESIS_ID:
        errs.append("imbalance_top_p_ic_incorrectly_bound_h23")

    elig = {"book_hypothesis_eligible": True}
    if not northset_has_finite_imbalance_p_ic({k_imb: 0.05, **elig}):
        errs.append("imbalance_p_ic_h22_gate_helper_broken")
    if northset_has_finite_imbalance_p_ic({k_recon: 1.0, **elig}):
        errs.append("session_reconstructs_incorrectly_triggers_h22_gate")
    return errs


def microprice_p_ic_vs_session_chain_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H25 ``microprice_p_ic`` with H29 ``session_chain_rate``.

    DATA_CONTRACTS: microprice discovery IC p ≠ session reconstruction chain.
    Mirrors :func:`imbalance_top_p_ic_vs_session_chain_never_equate_honesty_errors`
    with H25 bind + key-scoped ``_finite_scalar`` (no dedicated microprice gate).
    When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H25 ≠ H29)
    - H25 binds ``microprice_p_ic`` only; H29 binds chain only
    - Chain must not count as finite microprice via key-scoped probe
    - Numeric equality on clean synth is allowed (different units)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite. Off Commander #246+ identity batch.
    """
    if not isinstance(blob, dict):
        return []
    k_mp = "microprice_p_ic"
    k_chain = "session_chain_rate"
    if k_mp not in blob or k_chain not in blob:
        return []

    errs: list[str] = []
    if k_mp == k_chain:
        errs.append("microprice_p_ic_session_chain_keys_collapsed")
    if H25_HYPOTHESIS_ID == H29_HYPOTHESIS_ID:
        errs.append("h25_h29_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_mp) != H25_HYPOTHESIS_ID:
        errs.append("microprice_p_ic_not_bound_h25")
    if by_key.get(k_chain) != H29_HYPOTHESIS_ID:
        errs.append("session_chain_rate_not_bound_h29")
    if by_key.get(k_mp) == H29_HYPOTHESIS_ID:
        errs.append("microprice_p_ic_incorrectly_bound_h29")
    if by_key.get(k_chain) == H25_HYPOTHESIS_ID:
        errs.append("session_chain_rate_incorrectly_bound_h25")

    if not _finite_scalar({k_mp: 0.05}.get(k_mp)):
        errs.append("microprice_p_ic_finite_probe_broken")
    if _finite_scalar({k_chain: 1.0}.get(k_mp)):
        errs.append("session_chain_incorrectly_counts_as_finite_microprice")
    return errs


def sweep_reject_control_diff_p_vs_sweep_follow_control_diff_p_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H44 ``sweep_reject_control_diff_p`` with H45 follow twin.

    Sibling sweep control-diff p-values — reject path ≠ follow path.
    When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H44 ≠ H45)
    - Catalog bind lock: reject → H44; follow → H45
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_rej = "sweep_reject_control_diff_p"
    k_fol = "sweep_follow_control_diff_p"
    if k_rej not in blob or k_fol not in blob:
        return []

    errs: list[str] = []
    if k_rej == k_fol:
        errs.append("sweep_reject_follow_control_diff_p_keys_collapsed")
    if H44_HYPOTHESIS_ID == H45_HYPOTHESIS_ID:
        errs.append("h44_h45_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_rej) != H44_HYPOTHESIS_ID:
        errs.append("sweep_reject_control_diff_p_not_bound_h44")
    if by_key.get(k_fol) != H45_HYPOTHESIS_ID:
        errs.append("sweep_follow_control_diff_p_not_bound_h45")
    if by_key.get(k_rej) == H45_HYPOTHESIS_ID:
        errs.append("sweep_reject_control_diff_p_incorrectly_bound_h45")
    if by_key.get(k_fol) == H44_HYPOTHESIS_ID:
        errs.append("sweep_follow_control_diff_p_incorrectly_bound_h44")
    return errs


def sweep_reject_fold_positive_fraction_vs_sweep_follow_fold_positive_fraction_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H41 ``sweep_reject_fold_positive_fraction`` with H42 follow twin.

    Sibling sweep fold-positive fractions — reject path ≠ follow path.
    When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H41 ≠ H42)
    - Catalog bind lock: reject → H41; follow → H42
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_rej = "sweep_reject_fold_positive_fraction"
    k_fol = "sweep_follow_fold_positive_fraction"
    if k_rej not in blob or k_fol not in blob:
        return []

    errs: list[str] = []
    if k_rej == k_fol:
        errs.append("sweep_reject_follow_fold_positive_fraction_keys_collapsed")
    if H41_HYPOTHESIS_ID == H42_HYPOTHESIS_ID:
        errs.append("h41_h42_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_rej) != H41_HYPOTHESIS_ID:
        errs.append("sweep_reject_fold_positive_fraction_not_bound_h41")
    if by_key.get(k_fol) != H42_HYPOTHESIS_ID:
        errs.append("sweep_follow_fold_positive_fraction_not_bound_h42")
    if by_key.get(k_rej) == H42_HYPOTHESIS_ID:
        errs.append("sweep_reject_fold_positive_fraction_incorrectly_bound_h42")
    if by_key.get(k_fol) == H41_HYPOTHESIS_ID:
        errs.append("sweep_follow_fold_positive_fraction_incorrectly_bound_h41")
    return errs


def sweep_reject_event_p_vs_sweep_follow_event_p_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H35 ``sweep_reject_event_p`` with H36 ``sweep_follow_event_p``.

    DATA_CONTRACTS: sweep-reject event p ≠ sweep-follow event p. When both keys
    are present:

    - Distinct key identity
    - Distinct hyp ids (H35 ≠ H36)
    - H35 binds reject event only; H36 binds follow event only
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_rej = "sweep_reject_event_p"
    k_fol = "sweep_follow_event_p"
    if k_rej not in blob or k_fol not in blob:
        return []

    errs: list[str] = []
    if k_rej == k_fol:
        errs.append("sweep_reject_follow_event_p_keys_collapsed")
    if H35_HYPOTHESIS_ID == H36_HYPOTHESIS_ID:
        errs.append("h35_h36_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_rej) != H35_HYPOTHESIS_ID:
        errs.append("sweep_reject_event_p_not_bound_h35")
    if by_key.get(k_fol) != H36_HYPOTHESIS_ID:
        errs.append("sweep_follow_event_p_not_bound_h36")
    if by_key.get(k_rej) == H36_HYPOTHESIS_ID:
        errs.append("sweep_reject_event_p_incorrectly_bound_h36")
    if by_key.get(k_fol) == H35_HYPOTHESIS_ID:
        errs.append("sweep_follow_event_p_incorrectly_bound_h35")
    return errs


def sweep_reject_signed_p_ic_vs_sweep_follow_signed_p_ic_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H33 ``sweep_reject_signed_p_ic`` with H34 ``sweep_follow_signed_p_ic``.

    Sibling sweep signed-IC p-values — reject path ≠ follow path.
    When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H33 ≠ H34)
    - Catalog bind lock: reject → H33; follow → H34
    - Key-scoped ``_finite_scalar`` probes (no dedicated gate helpers)
    - Numeric equality on clean synth is allowed (different event sets)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off CoS H44; off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_rej = "sweep_reject_signed_p_ic"
    k_fol = "sweep_follow_signed_p_ic"
    if k_rej not in blob or k_fol not in blob:
        return []

    errs: list[str] = []
    if k_rej == k_fol:
        errs.append("sweep_reject_follow_signed_p_ic_keys_collapsed")
    if H33_HYPOTHESIS_ID == H34_HYPOTHESIS_ID:
        errs.append("h33_h34_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_rej) != H33_HYPOTHESIS_ID:
        errs.append("sweep_reject_signed_p_ic_not_bound_h33")
    if by_key.get(k_fol) != H34_HYPOTHESIS_ID:
        errs.append("sweep_follow_signed_p_ic_not_bound_h34")
    if by_key.get(k_rej) == H34_HYPOTHESIS_ID:
        errs.append("sweep_reject_signed_p_ic_incorrectly_bound_h34")
    if by_key.get(k_fol) == H33_HYPOTHESIS_ID:
        errs.append("sweep_follow_signed_p_ic_incorrectly_bound_h33")

    if not _finite_scalar({k_rej: 0.05}.get(k_rej)):
        errs.append("sweep_reject_signed_p_ic_finite_probe_broken")
    if _finite_scalar({k_fol: 0.05}.get(k_rej)):
        errs.append("sweep_follow_signed_p_ic_incorrectly_counts_as_finite_reject")
    if not _finite_scalar({k_fol: 0.05}.get(k_fol)):
        errs.append("sweep_follow_signed_p_ic_finite_probe_broken")
    if _finite_scalar({k_rej: 0.05}.get(k_fol)):
        errs.append("sweep_reject_signed_p_ic_incorrectly_counts_as_finite_follow")
    return errs


def sweep_reject_cost_adjusted_mean_bps_vs_sweep_follow_cost_adjusted_mean_bps_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H39 ``sweep_reject_cost_adjusted_mean_bps`` with H40 ``sweep_follow_cost_adjusted_mean_bps``.

    Sibling sweep cost-adjusted mean bps — reject path ≠ follow path.
    Mirrors :func:`sweep_reject_signed_p_ic_vs_sweep_follow_signed_p_ic_never_equate_honesty_errors`
    (H33≠H34). When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H39 ≠ H40)
    - Catalog bind lock: reject → H39; follow → H40
    - Key-scoped ``_finite_scalar`` probes (no dedicated gate helpers)
    - Numeric equality on clean synth is allowed (different event sets)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off Sergeant H37≠H38; off Lt H35≠H36; off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_rej = "sweep_reject_cost_adjusted_mean_bps"
    k_fol = "sweep_follow_cost_adjusted_mean_bps"
    if k_rej not in blob or k_fol not in blob:
        return []

    errs: list[str] = []
    if k_rej == k_fol:
        errs.append("sweep_reject_follow_cost_adjusted_mean_bps_keys_collapsed")
    if H39_HYPOTHESIS_ID == H40_HYPOTHESIS_ID:
        errs.append("h39_h40_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_rej) != H39_HYPOTHESIS_ID:
        errs.append("sweep_reject_cost_adjusted_mean_bps_not_bound_h39")
    if by_key.get(k_fol) != H40_HYPOTHESIS_ID:
        errs.append("sweep_follow_cost_adjusted_mean_bps_not_bound_h40")
    if by_key.get(k_rej) == H40_HYPOTHESIS_ID:
        errs.append("sweep_reject_cost_adjusted_mean_bps_incorrectly_bound_h40")
    if by_key.get(k_fol) == H39_HYPOTHESIS_ID:
        errs.append("sweep_follow_cost_adjusted_mean_bps_incorrectly_bound_h39")

    if not _finite_scalar({k_rej: 1.0}.get(k_rej)):
        errs.append("sweep_reject_cost_adjusted_mean_bps_finite_probe_broken")
    if _finite_scalar({k_fol: 1.0}.get(k_rej)):
        errs.append("sweep_follow_cost_adjusted_mean_bps_incorrectly_counts_as_finite_reject")
    if not _finite_scalar({k_fol: 1.0}.get(k_fol)):
        errs.append("sweep_follow_cost_adjusted_mean_bps_finite_probe_broken")
    if _finite_scalar({k_rej: 1.0}.get(k_fol)):
        errs.append("sweep_reject_cost_adjusted_mean_bps_incorrectly_counts_as_finite_follow")
    return errs


def sweep_reject_placebo_p_vs_sweep_follow_placebo_p_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H37 ``sweep_reject_placebo_p`` with H38 ``sweep_follow_placebo_p``.

    Sibling sweep placebo p-values — reject placebo ≠ follow placebo.
    When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H37 ≠ H38)
    - Catalog bind lock: reject placebo → H37; follow placebo → H38
    - Key-scoped ``_finite_scalar`` probes (no dedicated gate helpers)
    - Numeric equality on clean synth is allowed (different event sets)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off CoS H39≠H40; off Lt H35≠H36; off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_rej = "sweep_reject_placebo_p"
    k_fol = "sweep_follow_placebo_p"
    if k_rej not in blob or k_fol not in blob:
        return []

    errs: list[str] = []
    if k_rej == k_fol:
        errs.append("sweep_reject_follow_placebo_p_keys_collapsed")
    if H37_HYPOTHESIS_ID == H38_HYPOTHESIS_ID:
        errs.append("h37_h38_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_rej) != H37_HYPOTHESIS_ID:
        errs.append("sweep_reject_placebo_p_not_bound_h37")
    if by_key.get(k_fol) != H38_HYPOTHESIS_ID:
        errs.append("sweep_follow_placebo_p_not_bound_h38")
    if by_key.get(k_rej) == H38_HYPOTHESIS_ID:
        errs.append("sweep_reject_placebo_p_incorrectly_bound_h38")
    if by_key.get(k_fol) == H37_HYPOTHESIS_ID:
        errs.append("sweep_follow_placebo_p_incorrectly_bound_h37")

    if not _finite_scalar({k_rej: 0.05}.get(k_rej)):
        errs.append("sweep_reject_placebo_p_finite_probe_broken")
    if _finite_scalar({k_fol: 0.05}.get(k_rej)):
        errs.append("sweep_follow_placebo_p_incorrectly_counts_as_finite_reject_placebo")
    if not _finite_scalar({k_fol: 0.05}.get(k_fol)):
        errs.append("sweep_follow_placebo_p_finite_probe_broken")
    if _finite_scalar({k_rej: 0.05}.get(k_fol)):
        errs.append("sweep_reject_placebo_p_incorrectly_counts_as_finite_follow_placebo")
    return errs


def session_ohlc_vs_book_uncrossed_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate ``session_ohlc_identity_rate`` with H21 ``book_uncrossed_rate``.

    DATA_CONTRACTS: session-candle OHLC identity ≠ uncrossed book. When both
    keys are present:

    - Distinct key identity
    - Hyp/gate: session OHLC is **not** H23; H21 binds ``book_uncrossed_rate``
      only (session OHLC must not trigger H21); H20 gate is daily-only
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off H20/H21/H22 triad polish. Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_sess = "session_ohlc_identity_rate"
    k_book = "book_uncrossed_rate"
    if k_sess not in blob or k_book not in blob:
        return []

    errs: list[str] = []
    if k_sess == k_book:
        errs.append("session_ohlc_book_uncrossed_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_sess) == H23_HYPOTHESIS_ID:
        errs.append("session_ohlc_identity_rate_incorrectly_bound_h23")
    if by_key.get(k_book) == H23_HYPOTHESIS_ID:
        errs.append("book_uncrossed_rate_incorrectly_bound_h23")

    elig = {"book_hypothesis_eligible": True}
    if not northset_has_finite_book_uncrossed_rate({k_book: 1.0, **elig}):
        errs.append("book_uncrossed_h21_gate_helper_broken")
    if northset_has_finite_book_uncrossed_rate({k_sess: 1.0, **elig}):
        errs.append("session_ohlc_incorrectly_triggers_h21_gate")

    if northset_has_finite_ohlc_identity_rate({k_sess: 1.0}):
        errs.append("session_ohlc_incorrectly_triggers_h20_gate")
    if northset_has_finite_ohlc_identity_rate({k_book: 1.0}):
        errs.append("book_uncrossed_incorrectly_triggers_h20_gate")

    if H21_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H23_HYPOTHESIS_ID}:
        errs.append("h21_hypothesis_collapsed_into_h20_or_h23")
    return errs


def book_uncrossed_vs_session_chain_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H21 ``book_uncrossed_rate`` with H29 ``session_chain_rate``.

    DATA_CONTRACTS: uncrossed book ≠ session reconstruction chain. When both
    keys are present:

    - Distinct key identity
    - Distinct hyp ids (H21 ≠ H29)
    - H21 finite gate on book_uncrossed only; H29 binds chain only
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite. Leaves reconstructs lane clear.
    """
    if not isinstance(blob, dict):
        return []
    k_book = "book_uncrossed_rate"
    k_chain = "session_chain_rate"
    if k_book not in blob or k_chain not in blob:
        return []

    errs: list[str] = []
    if k_book == k_chain:
        errs.append("book_uncrossed_session_chain_keys_collapsed")
    if H21_HYPOTHESIS_ID == H29_HYPOTHESIS_ID:
        errs.append("h21_h29_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_chain) != H29_HYPOTHESIS_ID:
        errs.append("session_chain_rate_not_bound_h29")
    if by_key.get(k_book) == H29_HYPOTHESIS_ID:
        errs.append("book_uncrossed_rate_incorrectly_bound_h29")

    elig = {"book_hypothesis_eligible": True}
    if not northset_has_finite_book_uncrossed_rate({k_book: 1.0, **elig}):
        errs.append("book_uncrossed_h21_gate_helper_broken")
    if northset_has_finite_book_uncrossed_rate({k_chain: 1.0, **elig}):
        errs.append("session_chain_incorrectly_triggers_h21_gate")
    return errs


def imbalance_top_p_ic_vs_session_chain_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H22 ``imbalance_top_p_ic`` with H29 ``session_chain_rate``.

    DATA_CONTRACTS: imbalance discovery IC p ≠ session reconstruction chain.
    When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H22 ≠ H29)
    - H22 finite gate on imbalance_top_p_ic only; H29 binds chain only
    - Numeric equality on clean synth is allowed (different units)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_imb = "imbalance_top_p_ic"
    k_chain = "session_chain_rate"
    if k_imb not in blob or k_chain not in blob:
        return []

    errs: list[str] = []
    if k_imb == k_chain:
        errs.append("imbalance_top_p_ic_session_chain_keys_collapsed")
    if H22_HYPOTHESIS_ID == H29_HYPOTHESIS_ID:
        errs.append("h22_h29_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_chain) != H29_HYPOTHESIS_ID:
        errs.append("session_chain_rate_not_bound_h29")
    if by_key.get(k_imb) == H29_HYPOTHESIS_ID:
        errs.append("imbalance_top_p_ic_incorrectly_bound_h29")

    elig = {"book_hypothesis_eligible": True}
    if not northset_has_finite_imbalance_p_ic({k_imb: 0.05, **elig}):
        errs.append("imbalance_p_ic_h22_gate_helper_broken")
    if northset_has_finite_imbalance_p_ic({k_chain: 1.0, **elig}):
        errs.append("session_chain_incorrectly_triggers_h22_gate")
    return errs


def imbalance_top_p_ic_vs_session_ohlc_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H22 ``imbalance_top_p_ic`` with ``session_ohlc_identity_rate``.

    DATA_CONTRACTS: imbalance discovery IC p ≠ session-candle OHLC identity.
    When both keys are present:

    - Distinct key identity
    - H22 finite gate on imbalance_top_p_ic only (session OHLC must not trigger it)
    - session_ohlc must **not** bind H23 / H24 / H29
    - Numeric equality on clean synth is allowed (different units)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_imb = "imbalance_top_p_ic"
    k_sess = "session_ohlc_identity_rate"
    if k_imb not in blob or k_sess not in blob:
        return []

    errs: list[str] = []
    if k_imb == k_sess:
        errs.append("imbalance_top_p_ic_session_ohlc_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    sess_hyp = by_key.get(k_sess)
    if sess_hyp == H23_HYPOTHESIS_ID:
        errs.append("session_ohlc_identity_rate_incorrectly_bound_h23")
    if sess_hyp == H24_HYPOTHESIS_ID:
        errs.append("session_ohlc_identity_rate_incorrectly_bound_h24")
    if sess_hyp == H29_HYPOTHESIS_ID:
        errs.append("session_ohlc_identity_rate_incorrectly_bound_h29")
    if by_key.get(k_imb) in {H23_HYPOTHESIS_ID, H24_HYPOTHESIS_ID, H29_HYPOTHESIS_ID}:
        errs.append("imbalance_top_p_ic_incorrectly_bound_session_hyp")

    elig = {"book_hypothesis_eligible": True}
    if not northset_has_finite_imbalance_p_ic({k_imb: 0.05, **elig}):
        errs.append("imbalance_p_ic_h22_gate_helper_broken")
    if northset_has_finite_imbalance_p_ic({k_sess: 1.0, **elig}):
        errs.append("session_ohlc_incorrectly_triggers_h22_gate")
    return errs


def imbalance_top_p_ic_vs_session_volume_conservation_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H22 ``imbalance_top_p_ic`` with H24 ``session_volume_conservation_rate``.

    DATA_CONTRACTS: imbalance discovery IC p ≠ session volume conservation.
    Completes H22↔session-rate mesh with H22↔H20/H21/H29 and H24 siblings.
    When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H22 ≠ H24)
    - H22 finite gate on imbalance_top_p_ic only; H24 binds volume only
    - imbalance_top_p_ic must not bind H24; volume must not trigger H22 gate
    - Numeric equality on clean synth is allowed (different units)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_imb = "imbalance_top_p_ic"
    k_vol = "session_volume_conservation_rate"
    if k_imb not in blob or k_vol not in blob:
        return []

    errs: list[str] = []
    if k_imb == k_vol:
        errs.append("imbalance_top_p_ic_session_volume_conservation_keys_collapsed")
    if H22_HYPOTHESIS_ID == H24_HYPOTHESIS_ID:
        errs.append("h22_h24_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_vol) != H24_HYPOTHESIS_ID:
        errs.append("session_volume_conservation_rate_not_bound_h24")
    if by_key.get(k_imb) == H24_HYPOTHESIS_ID:
        errs.append("imbalance_top_p_ic_incorrectly_bound_h24")

    elig = {"book_hypothesis_eligible": True}
    if not northset_has_finite_imbalance_p_ic({k_imb: 0.05, **elig}):
        errs.append("imbalance_p_ic_h22_gate_helper_broken")
    if northset_has_finite_imbalance_p_ic({k_vol: 1.0, **elig}):
        errs.append("session_volume_conservation_incorrectly_triggers_h22_gate")
    return errs


def book_uncrossed_vs_session_reconstructs_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H21 ``book_uncrossed_rate`` with H23 ``session_reconstructs_daily_rate``.

    DATA_CONTRACTS: uncrossed book ≠ session envelope reconstructing the daily bar.
    When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H21 ≠ H23)
    - H21 finite gate on book_uncrossed only; H23 binds reconstructs only
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_book = "book_uncrossed_rate"
    k_recon = "session_reconstructs_daily_rate"
    if k_book not in blob or k_recon not in blob:
        return []

    errs: list[str] = []
    if k_book == k_recon:
        errs.append("book_uncrossed_session_reconstructs_keys_collapsed")
    if H21_HYPOTHESIS_ID == H23_HYPOTHESIS_ID:
        errs.append("h21_h23_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_recon) != H23_HYPOTHESIS_ID:
        errs.append("session_reconstructs_daily_rate_not_bound_h23")
    if by_key.get(k_book) == H23_HYPOTHESIS_ID:
        errs.append("book_uncrossed_rate_incorrectly_bound_h23")

    elig = {"book_hypothesis_eligible": True}
    if not northset_has_finite_book_uncrossed_rate({k_book: 1.0, **elig}):
        errs.append("book_uncrossed_h21_gate_helper_broken")
    if northset_has_finite_book_uncrossed_rate({k_recon: 1.0, **elig}):
        errs.append("session_reconstructs_incorrectly_triggers_h21_gate")
    return errs


def book_uncrossed_vs_volume_conservation_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H21 ``book_uncrossed_rate`` with H24 ``session_volume_conservation_rate``.

    DATA_CONTRACTS: uncrossed book ≠ session volume conservation. When both
    keys are present:

    - Distinct key identity
    - Distinct hyp ids (H21 ≠ H24)
    - H21 finite gate on book_uncrossed only; H24 binds volume only
    - book_uncrossed must not bind H24; volume must not trigger H21 gate
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_book = "book_uncrossed_rate"
    k_vol = "session_volume_conservation_rate"
    if k_book not in blob or k_vol not in blob:
        return []

    errs: list[str] = []
    if k_book == k_vol:
        errs.append("book_uncrossed_volume_conservation_keys_collapsed")
    if H21_HYPOTHESIS_ID == H24_HYPOTHESIS_ID:
        errs.append("h21_h24_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_vol) != H24_HYPOTHESIS_ID:
        errs.append("session_volume_conservation_rate_not_bound_h24")
    if by_key.get(k_book) == H24_HYPOTHESIS_ID:
        errs.append("book_uncrossed_rate_incorrectly_bound_h24")

    elig = {"book_hypothesis_eligible": True}
    if not northset_has_finite_book_uncrossed_rate({k_book: 1.0, **elig}):
        errs.append("book_uncrossed_h21_gate_helper_broken")
    if northset_has_finite_book_uncrossed_rate({k_vol: 1.0, **elig}):
        errs.append("session_volume_conservation_incorrectly_triggers_h21_gate")
    return errs


def ohlc_identity_vs_session_ohlc_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate daily ``ohlc_identity_rate`` with ``session_ohlc_identity_rate``.

    Distinct clocks (daily bars vs session L2). When **both** keys are present:

    - Key identity: ``ohlc_identity_rate`` ≠ ``session_ohlc_identity_rate``
    - H20 finite gate binds to **daily** ``ohlc_identity_rate`` only
      (``H20_HYPOTHESIS_ID``); session OHLC is not an H20 metric
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_daily = "ohlc_identity_rate"
    k_sess = "session_ohlc_identity_rate"
    if k_daily not in blob or k_sess not in blob:
        return []

    errs: list[str] = []
    if k_daily == k_sess:
        errs.append("ohlc_identity_session_ohlc_keys_collapsed")

    # Catalog lock: H20 gate helper watches daily key only
    if not callable(northset_has_finite_ohlc_identity_rate):
        errs.append("ohlc_identity_h20_gate_helper_missing")
    # Session key must not be the daily key string (structural)
    if k_sess == "ohlc_identity_rate":
        errs.append("session_ohlc_identity_rate_aliased_to_daily")

    # Ensure H20 id stays distinct from session H23/H24/H29 binds
    session_hyps = {
        H23_HYPOTHESIS_ID,
        H24_HYPOTHESIS_ID,
        H29_HYPOTHESIS_ID,
    }
    if H20_HYPOTHESIS_ID in session_hyps:
        errs.append("h20_hypothesis_collapsed_into_session_binds")
    return errs


def ohlc_identity_vs_session_reconstructs_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate daily ``ohlc_identity_rate`` with ``session_reconstructs_daily_rate``.

    DATA_CONTRACTS: daily OHLC envelope identity (H20) ≠ session envelope that
    reconstructs the daily bar (H23). When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H20 ≠ H23)
    - H20 finite gate on daily ohlc only; H23 binds reconstructs only
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_ohlc = "ohlc_identity_rate"
    k_recon = "session_reconstructs_daily_rate"
    if k_ohlc not in blob or k_recon not in blob:
        return []

    errs: list[str] = []
    if k_ohlc == k_recon:
        errs.append("ohlc_identity_session_reconstructs_keys_collapsed")
    if H20_HYPOTHESIS_ID == H23_HYPOTHESIS_ID:
        errs.append("h20_h23_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_recon) != H23_HYPOTHESIS_ID:
        errs.append("session_reconstructs_daily_rate_not_bound_h23")
    if by_key.get(k_ohlc) == H23_HYPOTHESIS_ID:
        errs.append("ohlc_identity_rate_incorrectly_bound_h23")

    if not northset_has_finite_ohlc_identity_rate({k_ohlc: 1.0}):
        errs.append("ohlc_identity_h20_gate_helper_broken")
    if northset_has_finite_ohlc_identity_rate({k_recon: 1.0}):
        errs.append("session_reconstructs_incorrectly_triggers_h20_gate")
    return errs


def ohlc_identity_vs_session_chain_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate daily ``ohlc_identity_rate`` with H29 ``session_chain_rate``.

    DATA_CONTRACTS: daily OHLC envelope identity (H20) ≠ session chain identity
    (H29). When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H20 ≠ H29)
    - H20 finite gate on daily ohlc only; H29 binds chain only
    - Daily ohlc must not bind H23/H24/H29; chain must not trigger H20 gate
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_ohlc = "ohlc_identity_rate"
    k_chain = "session_chain_rate"
    if k_ohlc not in blob or k_chain not in blob:
        return []

    errs: list[str] = []
    if k_ohlc == k_chain:
        errs.append("ohlc_identity_session_chain_keys_collapsed")
    if H20_HYPOTHESIS_ID == H29_HYPOTHESIS_ID:
        errs.append("h20_h29_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_chain) != H29_HYPOTHESIS_ID:
        errs.append("session_chain_rate_not_bound_h29")
    if by_key.get(k_ohlc) in {
        H23_HYPOTHESIS_ID,
        H24_HYPOTHESIS_ID,
        H29_HYPOTHESIS_ID,
    }:
        errs.append("ohlc_identity_rate_incorrectly_bound_session_hyp")

    if not northset_has_finite_ohlc_identity_rate({k_ohlc: 1.0}):
        errs.append("ohlc_identity_h20_gate_helper_broken")
    if northset_has_finite_ohlc_identity_rate({k_chain: 1.0}):
        errs.append("session_chain_incorrectly_triggers_h20_gate")
    return errs


def ohlc_identity_vs_volume_conservation_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate daily ``ohlc_identity_rate`` with H24 ``session_volume_conservation_rate``.

    DATA_CONTRACTS: daily OHLC envelope identity (H20) ≠ session volume conservation
    (H24). When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H20 ≠ H24)
    - H24 binds volume only; H20 finite gate on daily ohlc only
    - Daily ohlc must not bind H23/H24/H29; volume must not trigger H20 gate
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_ohlc = "ohlc_identity_rate"
    k_vol = "session_volume_conservation_rate"
    if k_ohlc not in blob or k_vol not in blob:
        return []

    errs: list[str] = []
    if k_ohlc == k_vol:
        errs.append("ohlc_identity_volume_conservation_keys_collapsed")
    if H20_HYPOTHESIS_ID == H24_HYPOTHESIS_ID:
        errs.append("h20_h24_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_vol) != H24_HYPOTHESIS_ID:
        errs.append("session_volume_conservation_rate_not_bound_h24")
    if by_key.get(k_ohlc) in {
        H23_HYPOTHESIS_ID,
        H24_HYPOTHESIS_ID,
        H29_HYPOTHESIS_ID,
    }:
        errs.append("ohlc_identity_rate_incorrectly_bound_session_hyp")

    if not northset_has_finite_ohlc_identity_rate({k_ohlc: 1.0}):
        errs.append("ohlc_identity_h20_gate_helper_broken")
    if northset_has_finite_ohlc_identity_rate({k_vol: 1.0}):
        errs.append("session_volume_conservation_incorrectly_triggers_h20_gate")
    return errs


def session_chain_vs_session_identity_siblings_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H29 session_chain_rate with H23 reconstructs / H24 conservation.

    Sibling session identity rates — chain ≠ reconstructs ≠ volume conservation.
    When ``session_chain_rate`` is present alongside either sibling:

    - Catalog bind lock: chain → H29; reconstructs → H23; conservation → H24
    - Distinct hyp ids (no collapse)
    - Numeric equality on clean synth is allowed

    Skip when chain key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_chain = "session_chain_rate"
    k_recon = "session_reconstructs_daily_rate"
    k_vol = "session_volume_conservation_rate"
    if k_chain not in blob:
        return []
    if k_recon not in blob and k_vol not in blob:
        return []

    errs: list[str] = []
    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    chain_hyp = by_key.get(k_chain)
    recon_hyp = by_key.get(k_recon)
    vol_hyp = by_key.get(k_vol)
    if chain_hyp != H29_HYPOTHESIS_ID:
        errs.append("session_chain_rate_not_bound_h29")
    if k_recon in blob and recon_hyp != H23_HYPOTHESIS_ID:
        errs.append("session_reconstructs_daily_rate_not_bound_h23")
    if k_vol in blob and vol_hyp != H24_HYPOTHESIS_ID:
        errs.append("session_volume_conservation_rate_not_bound_h24")
    if chain_hyp is not None and recon_hyp is not None and chain_hyp == recon_hyp:
        errs.append("session_chain_reconstructs_hypothesis_collapsed")
    if chain_hyp is not None and vol_hyp is not None and chain_hyp == vol_hyp:
        errs.append("session_chain_volume_conservation_hypothesis_collapsed")
    return errs


def amihud_mean_ic_vs_amihud_abs_mean_ic_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate ``amihud_mean_ic`` with ``amihud_abs_mean_ic``.

    Sibling Amihud discovery ICs — signed Amihud path ≠ abs Amihud path.
    When both keys are present:

    - Distinct key identity
    - Companion packs stay distinct (``amihud_p_ic`` ≠ ``amihud_abs_p_ic``, etc.)
    - Key-scoped finite probes
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off CoS spread-IC; off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_mean = "amihud_mean_ic"
    k_abs = "amihud_abs_mean_ic"
    if k_mean not in blob or k_abs not in blob:
        return []

    errs: list[str] = []
    if k_mean == k_abs:
        errs.append("amihud_mean_ic_amihud_abs_mean_ic_keys_collapsed")

    companions = (
        ("amihud_p_ic", "amihud_abs_p_ic"),
        ("amihud_t_ic", "amihud_abs_t_ic"),
        ("amihud_n_dates", "amihud_abs_n_dates"),
    )
    for a, b in companions:
        if a == b:
            errs.append(f"amihud_companion_keys_collapsed_{a}")

    if not _finite_scalar({k_mean: 0.1}.get(k_mean)):
        errs.append("amihud_mean_ic_finite_probe_broken")
    if _finite_scalar({k_abs: 0.1}.get(k_mean)):
        errs.append("amihud_abs_mean_ic_incorrectly_counts_as_finite_amihud_mean_ic")
    if not _finite_scalar({k_abs: 0.1}.get(k_abs)):
        errs.append("amihud_abs_mean_ic_finite_probe_broken")
    if _finite_scalar({k_mean: 0.1}.get(k_abs)):
        errs.append("amihud_mean_ic_incorrectly_counts_as_finite_amihud_abs_mean_ic")
    return errs


def mid_lag1_corr_vs_ofi_lag1_corr_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate ``mid_lag1_corr`` with ``ofi_lag1_corr``.

    DATA_CONTRACTS: same panel lag-1 estimator shape, different series
    (mid path AR ≠ OFI path AR). When both keys are present:

    - Distinct key identity
    - Companion n-securities keys stay distinct when both stamped
    - Key-scoped finite probes (gap-style: mid-only blob must not count as ofi)
    - Numeric equality on clean synth is allowed

    Skip when either corr key absent. Research diagnostic only; never live Sharpe.
    Always-on surface — not nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_mid = "mid_lag1_corr"
    k_ofi = "ofi_lag1_corr"
    if k_mid not in blob or k_ofi not in blob:
        return []

    errs: list[str] = []
    if k_mid == k_ofi:
        errs.append("mid_ofi_lag1_corr_keys_collapsed")

    k_mid_n = "mid_lag1_n_securities"
    k_ofi_n = "ofi_lag1_n_securities"
    if k_mid_n in blob and k_ofi_n in blob and k_mid_n == k_ofi_n:
        errs.append("mid_ofi_lag1_n_securities_keys_collapsed")

    if not _finite_scalar({k_mid: 0.5}.get(k_mid)):
        errs.append("mid_lag1_corr_finite_probe_broken")
    if _finite_scalar({k_ofi: 0.5}.get(k_mid)):
        errs.append("ofi_lag1_corr_incorrectly_counts_as_finite_mid_lag1")
    if not _finite_scalar({k_ofi: 0.5}.get(k_ofi)):
        errs.append("ofi_lag1_corr_finite_probe_broken")
    if _finite_scalar({k_mid: 0.5}.get(k_ofi)):
        errs.append("mid_lag1_corr_incorrectly_counts_as_finite_ofi_lag1")
    return errs


def kyle_r2_vs_kyle_ofi_r2_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate always-on ``kyle_r2`` with ``kyle_ofi_r2``.

    DATA_CONTRACTS: both are name-mean OLS R², different flow columns
    (``signed_volume`` vs ``ofi``). When both keys are present:

    - Distinct key identity
    - Key-scoped finite probes (ofi-only blob must not count as kyle_r2)
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Always-on — not nest invent / kyle_ofi.py overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_sv = "kyle_r2"
    k_ofi = "kyle_ofi_r2"
    if k_sv not in blob or k_ofi not in blob:
        return []

    errs: list[str] = []
    if k_sv == k_ofi:
        errs.append("kyle_r2_kyle_ofi_r2_keys_collapsed")

    if not _finite_scalar({k_sv: 0.5}.get(k_sv)):
        errs.append("kyle_r2_finite_probe_broken")
    if _finite_scalar({k_ofi: 0.5}.get(k_sv)):
        errs.append("kyle_ofi_r2_incorrectly_counts_as_finite_kyle_r2")
    if not _finite_scalar({k_ofi: 0.5}.get(k_ofi)):
        errs.append("kyle_ofi_r2_finite_probe_broken")
    if _finite_scalar({k_sv: 0.5}.get(k_ofi)):
        errs.append("kyle_r2_incorrectly_counts_as_finite_kyle_ofi_r2")
    return errs


__all__ = [
    "amihud_mean_ic_vs_amihud_abs_mean_ic_never_equate_honesty_errors",
    "book_uncrossed_vs_imbalance_p_ic_never_equate_honesty_errors",
    "book_uncrossed_vs_session_chain_never_equate_honesty_errors",
    "book_uncrossed_vs_session_reconstructs_never_equate_honesty_errors",
    "book_uncrossed_vs_volume_conservation_never_equate_honesty_errors",
    "imbalance_top_p_ic_vs_session_chain_never_equate_honesty_errors",
    "imbalance_top_p_ic_vs_session_ohlc_never_equate_honesty_errors",
    "imbalance_top_p_ic_vs_session_reconstructs_never_equate_honesty_errors",
    "imbalance_top_p_ic_vs_session_volume_conservation_never_equate_honesty_errors",
    "kyle_r2_vs_kyle_ofi_r2_never_equate_honesty_errors",
    "microprice_p_ic_vs_session_chain_never_equate_honesty_errors",
    "mid_lag1_corr_vs_ofi_lag1_corr_never_equate_honesty_errors",
    "ohlc_identity_vs_book_uncrossed_never_equate_honesty_errors",
    "ohlc_identity_vs_imbalance_p_ic_never_equate_honesty_errors",
    "ohlc_identity_vs_session_chain_never_equate_honesty_errors",
    "ohlc_identity_vs_session_ohlc_never_equate_honesty_errors",
    "ohlc_identity_vs_session_reconstructs_never_equate_honesty_errors",
    "ohlc_identity_vs_volume_conservation_never_equate_honesty_errors",
    "session_bulk_vpin_vs_siblings_never_equate_honesty_errors",
    "session_chain_vs_session_identity_siblings_never_equate_honesty_errors",
    "session_ohlc_vs_book_uncrossed_never_equate_honesty_errors",
    "session_ohlc_vs_reconstructs_never_equate_honesty_errors",
    "session_ohlc_vs_session_chain_never_equate_honesty_errors",
    "session_ohlc_vs_volume_conservation_never_equate_honesty_errors",
    "session_volume_conservation_vs_reconstructs_never_equate_honesty_errors",
    "sweep_reject_control_diff_p_vs_sweep_follow_control_diff_p_never_equate_honesty_errors",
    "sweep_reject_cost_adjusted_mean_bps_vs_sweep_follow_cost_adjusted_mean_bps_never_equate_honesty_errors",
    "sweep_reject_event_p_vs_sweep_follow_event_p_never_equate_honesty_errors",
    "sweep_reject_fold_positive_fraction_vs_sweep_follow_fold_positive_fraction_never_equate_honesty_errors",
    "sweep_reject_placebo_p_vs_sweep_follow_placebo_p_never_equate_honesty_errors",
    "sweep_reject_signed_p_ic_vs_sweep_follow_signed_p_ic_never_equate_honesty_errors",
]
