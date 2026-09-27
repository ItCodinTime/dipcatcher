"""Hypothesis-vs-receipt consistency checkers (``*_consistency_errors``)."""

from __future__ import annotations

from collections.abc import Callable

from ._helpers import (
    _JP_CV_SCOPE_FAMILIES,
    _finite_scalar,
)
from .candle import (
    candle_structure_finite_rate_covers_companions_honesty_errors,
)
from .constants import (
    COVERAGE_GUARANTEE_SCOPE_MARGINAL,
    H16_H18_EXPECTED_FAMILY,
    H16_H18_PANEL_SPECS,
    NORTHSET_H23_H28_SPECS,
)
from .families import (
    hypotheses_include_h1,
    hypotheses_include_h2,
    hypotheses_include_h3,
    hypotheses_include_h4,
    hypotheses_include_h4b,
    hypotheses_include_h7,
    hypotheses_include_h8,
    hypotheses_include_h9,
    hypotheses_include_h10,
    hypotheses_include_h11,
    hypotheses_include_h12,
    hypotheses_include_h15,
    hypotheses_include_h19,
    hypotheses_include_h20,
    hypotheses_include_h21,
    hypotheses_include_h22,
    hypotheses_include_h43,
    hypotheses_include_h99,
    hypotheses_include_id,
    jp_cv_blob_requires_marginal_coverage_scope,
    rankers_oracle_raw,
)
from .kyle import (
    kyle_r2_vs_kyle_ofi_r2_never_equate_honesty_errors,
)
from .northset import (
    northset_all_finite_rate_unit_honesty_errors,
    northset_all_floor_unit_honesty_errors,
    northset_all_fraction_unit_honesty_errors,
    northset_all_mean_ic_finite_honesty_errors,
    northset_all_mean_rank_ic_unit_honesty_errors,
    northset_all_n_dates_nonneg_honesty_errors,
    northset_all_p_ic_unit_interval_honesty_errors,
    northset_all_rate_unit_honesty_errors,
    northset_all_share_unit_honesty_errors,
    northset_all_t_ic_finite_honesty_errors,
    northset_amihud_ic_pack_honesty_errors,
    northset_bid_log_size_slope_ic_pack_honesty_errors,
    northset_book_shape_finite_rates_honesty_errors,
    northset_candle_body_ret_ic_pack_honesty_errors,
    northset_clv_ic_pack_honesty_errors,
    northset_component_sources_honesty_errors,
    northset_depth_honesty_errors,
    northset_dm_park_honesty_errors,
    northset_family_book_source_honesty_errors,
    northset_half_spread_honesty_errors,
    northset_imbalance_depth_ic_pack_honesty_errors,
    northset_imbalance_top_ic_pack_honesty_errors,
    northset_impact_proxy_warning_honesty_errors,
    northset_include_kyle_ofi_nest_presence_honesty_errors,
    northset_kyle_r2_unit_honesty_errors,
    northset_lag_corr_and_sweep_count_honesty_errors,
    northset_log_size_slope_honesty_errors,
    northset_microprice_bps_ic_pack_honesty_errors,
    northset_microprice_weight_balance_honesty_errors,
    northset_n_bars_scored_honesty_errors,
    northset_ofi_lag_ic_honesty_errors,
    northset_overnight_rv_semi_honesty_errors,
    northset_price_return_basis_honesty_errors,
    northset_product_stamp_honesty_errors,
    northset_qlike_means_honesty_errors,
    northset_queue_imbalance_ic_honesty_errors,
    northset_queue_imbalance_mean_alias_honesty_errors,
    northset_queue_priority_bid_ask_pair_honesty_errors,
    northset_queue_priority_le_size_concentration_honesty_errors,
    northset_queue_sweep_ofi_honesty_errors,
    northset_range_spread_honesty_errors,
    northset_receipt_bool_flags_honesty_errors,
    northset_receipt_dgp_data_source_honesty_errors,
    northset_receipt_string_enum_honesty_errors,
    northset_session_book_snaps_honesty_errors,
    northset_session_book_vpin_ic_pack_honesty_errors,
    northset_session_book_vpin_mean_honesty_errors,
    northset_session_close_depth_honesty_errors,
    northset_session_close_ic_packs_honesty_errors,
    northset_session_close_imbalance_honesty_errors,
    northset_session_close_micro_bps_honesty_errors,
    northset_session_close_mid_honesty_errors,
    northset_session_close_mid_micro_pair_honesty_errors,
    northset_session_close_spread_bps_honesty_errors,
    northset_session_identity_rates_honesty_errors,
    northset_session_imbalance_mean_honesty_errors,
    northset_session_imbalance_std_honesty_errors,
    northset_session_l2_enforced_identity_rates_present_honesty_errors,
    northset_session_ofi_abs_dominates_sum_honesty_errors,
    northset_session_ofi_abs_sum_honesty_errors,
    northset_session_ofi_sum_ic_honesty_errors,
    northset_session_ofi_sum_mean_honesty_errors,
    northset_session_spread_bps_mean_honesty_errors,
    northset_shape_columns_ensured_book_panel_path_honesty_errors,
    northset_spread_bps_honesty_errors,
    northset_spread_means_honesty_errors,
    northset_spread_receipt_honesty_errors,
    northset_structure_finite_rate_distinct_from_candle_honesty_errors,
    northset_sweep_control_sample_adequate_honesty_errors,
    northset_sweep_evidence_blob_honesty_errors,
    northset_sweep_evidence_scope_honesty_errors,
    northset_sweep_fold_positive_rates_honesty_errors,
    northset_sweep_signed_ic_packs_honesty_errors,
    northset_volume_over_range_ic_packs_honesty_errors,
    northset_vpin_ic_pack_honesty_errors,
    northset_vpin_sweep_fold_honesty_errors,
    northset_wick_skew_ic_pack_honesty_errors,
)
from .predicates import (
    aci_has_finite_kupiec_p,
    conformal_aci_blob,
    conformal_mondrian_aci_blob,
    conformal_rank_has_finite_fdr,
    crc_has_finite_kupiec_p,
    cv_plus_has_finite_coverage_and_floor,
    data_snooping_has_finite_spa_p,
    evalues_has_finite_e_sup,
    jackknife_plus_has_finite_coverage,
    mondrian_aci_has_finite_high_x_kupiec_p,
    northset_has_finite_book_uncrossed_rate,
    northset_has_finite_imbalance_p_ic,
    northset_has_finite_ohlc_identity_rate,
    northset_has_finite_session_book_vpin_p_ic,
    oracle_has_finite_ls_p,
    oracle_has_finite_p_ic,
    panel_family_has_finite_kupiec_p,
    ranking_data_snooping_blob,
    tail_has_finite_christoffersen_cc_p,
    tail_has_finite_kupiec_p,
    volatility_has_finite_dm_p,
    weighted_conformal_has_finite_kupiec_p,
)
from .session import (
    amihud_mean_honesty_errors,
    amihud_mean_ic_vs_amihud_abs_mean_ic_never_equate_honesty_errors,
    book_age_seconds_honesty_errors,
    book_hypothesis_eligible_honesty_errors,
    book_uncrossed_rate_honesty_errors,
    book_uncrossed_vs_imbalance_p_ic_never_equate_honesty_errors,
    book_uncrossed_vs_session_chain_never_equate_honesty_errors,
    book_uncrossed_vs_session_reconstructs_never_equate_honesty_errors,
    book_uncrossed_vs_volume_conservation_never_equate_honesty_errors,
    close_location_value_clv_alias_identity_honesty_errors,
    clv_p_ic_honesty_errors,
    clv_p_ic_vs_gap_finite_never_equate_honesty_errors,
    depth_shape_finite_rate_honesty_errors,
    dm_gk_vs_park_p_vs_gap_finite_never_equate_honesty_errors,
    dm_split_vs_park_p_vs_gap_finite_never_equate_honesty_errors,
    gap_finite_rate_honesty_errors,
    gap_finite_rate_vs_book_uncrossed_never_equate_honesty_errors,
    gap_finite_rate_vs_session_chain_never_equate_honesty_errors,
    gap_finite_rate_vs_session_reconstructs_never_equate_honesty_errors,
    gap_finite_rate_vs_session_volume_conservation_never_equate_honesty_errors,
    imbalance_top_mean_ic_honesty_errors,
    imbalance_top_p_ic_vs_gap_finite_never_equate_honesty_errors,
    imbalance_top_p_ic_vs_session_chain_never_equate_honesty_errors,
    imbalance_top_p_ic_vs_session_ohlc_never_equate_honesty_errors,
    imbalance_top_p_ic_vs_session_reconstructs_never_equate_honesty_errors,
    imbalance_top_p_ic_vs_session_volume_conservation_never_equate_honesty_errors,
    mean_book_age_seconds_honesty_errors,
    mean_microprice_minus_mid_honesty_errors,
    microprice_p_ic_honesty_errors,
    microprice_p_ic_vs_gap_finite_never_equate_honesty_errors,
    microprice_p_ic_vs_session_chain_never_equate_honesty_errors,
    mid_lag1_corr_vs_ofi_lag1_corr_never_equate_honesty_errors,
    ofi_mean_ic_honesty_errors,
    ofi_p_ic_honesty_errors,
    ofi_p_ic_vs_gap_finite_never_equate_honesty_errors,
    ohlc_identity_rate_honesty_errors,
    ohlc_identity_vs_book_uncrossed_never_equate_honesty_errors,
    ohlc_identity_vs_gap_finite_never_equate_honesty_errors,
    ohlc_identity_vs_imbalance_p_ic_never_equate_honesty_errors,
    ohlc_identity_vs_session_chain_never_equate_honesty_errors,
    ohlc_identity_vs_session_ohlc_never_equate_honesty_errors,
    ohlc_identity_vs_session_reconstructs_never_equate_honesty_errors,
    ohlc_identity_vs_volume_conservation_never_equate_honesty_errors,
    session_book_vpin_p_ic_vs_gap_finite_never_equate_honesty_errors,
    session_bulk_vpin_honesty_errors,
    session_bulk_vpin_vs_siblings_never_equate_honesty_errors,
    session_chain_rate_honesty_errors,
    session_chain_vs_session_identity_siblings_never_equate_honesty_errors,
    session_mean_jump_ratio_honesty_errors,
    session_ohlc_vs_book_uncrossed_never_equate_honesty_errors,
    session_ohlc_vs_gap_finite_never_equate_honesty_errors,
    session_ohlc_vs_reconstructs_never_equate_honesty_errors,
    session_ohlc_vs_session_chain_never_equate_honesty_errors,
    session_ohlc_vs_volume_conservation_never_equate_honesty_errors,
    session_reconstructs_daily_rate_honesty_errors,
    session_volume_conservation_rate_honesty_errors,
    session_volume_conservation_vs_reconstructs_never_equate_honesty_errors,
    sweep_follow_control_diff_p_vs_gap_finite_never_equate_honesty_errors,
    sweep_follow_cost_adjusted_mean_bps_honesty_errors,
    sweep_follow_cost_adjusted_mean_bps_vs_gap_finite_never_equate_honesty_errors,
    sweep_follow_event_mean_bps_honesty_errors,
    sweep_follow_event_p_vs_gap_finite_never_equate_honesty_errors,
    sweep_follow_fold_positive_fraction_vs_gap_finite_never_equate_honesty_errors,
    sweep_follow_placebo_p_vs_gap_finite_never_equate_honesty_errors,
    sweep_follow_signed_mean_ic_honesty_errors,
    sweep_follow_signed_p_ic_vs_gap_finite_never_equate_honesty_errors,
    sweep_reject_control_diff_p_vs_gap_finite_never_equate_honesty_errors,
    sweep_reject_control_diff_p_vs_sweep_follow_control_diff_p_never_equate_honesty_errors,
    sweep_reject_cost_adjusted_mean_bps_honesty_errors,
    sweep_reject_cost_adjusted_mean_bps_vs_gap_finite_never_equate_honesty_errors,
    sweep_reject_cost_adjusted_mean_bps_vs_sweep_follow_cost_adjusted_mean_bps_never_equate_honesty_errors,
    sweep_reject_event_mean_bps_honesty_errors,
    sweep_reject_event_p_vs_gap_finite_never_equate_honesty_errors,
    sweep_reject_event_p_vs_sweep_follow_event_p_never_equate_honesty_errors,
    sweep_reject_fold_positive_fraction_vs_gap_finite_never_equate_honesty_errors,
    sweep_reject_fold_positive_fraction_vs_sweep_follow_fold_positive_fraction_never_equate_honesty_errors,
    sweep_reject_placebo_p_vs_gap_finite_never_equate_honesty_errors,
    sweep_reject_placebo_p_vs_sweep_follow_placebo_p_never_equate_honesty_errors,
    sweep_reject_signed_mean_ic_honesty_errors,
    sweep_reject_signed_p_ic_vs_gap_finite_never_equate_honesty_errors,
    sweep_reject_signed_p_ic_vs_sweep_follow_signed_p_ic_never_equate_honesty_errors,
    vpin_mean_honesty_errors,
    vpin_p_ic_vs_gap_finite_never_equate_honesty_errors,
    wick_skew_p_ic_vs_gap_finite_never_equate_honesty_errors,
)


def h4b_hypothesis_consistency_errors(tail: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H4b notebook consistency (Day Wave 26).

    Contract (research diagnostic only):
    - Non-dict / missing / non-finite ``christoffersen_cc_p`` → no errors (skip).
    - Finite ``christoffersen_cc_p`` without ``H4b_var_christoffersen_cc``
      (calibration) → ``hypothesis_h4b_missing_despite_finite_christoffersen_cc_p``.
    """
    if not tail_has_finite_christoffersen_cc_p(tail):
        return []
    if hypotheses_include_h4b(hypotheses, require_calibration=True):
        return []
    return ["hypothesis_h4b_missing_despite_finite_christoffersen_cc_p"]


def h4_hypothesis_consistency_errors(tail: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H4 notebook consistency (Day Wave 29).

    Contract (research diagnostic only):
    - Non-dict / missing / non-finite ``kupiec_p`` → no errors (skip).
    - Finite ``kupiec_p`` without ``H4_var_kupiec``
      (calibration) → ``hypothesis_h4_missing_despite_finite_kupiec_p``.
    """
    if not tail_has_finite_kupiec_p(tail):
        return []
    if hypotheses_include_h4(hypotheses, require_calibration=True):
        return []
    return ["hypothesis_h4_missing_despite_finite_kupiec_p"]


def h3_hypothesis_consistency_errors(volatility: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H3 notebook consistency (Day Wave 30).

    Contract (research diagnostic only):
    - Non-dict / missing / non-finite ``dm_p`` → no errors (skip).
    - Finite ``dm_p`` without ``H3_vol_dm``
      (discovery) → ``hypothesis_h3_missing_despite_finite_dm_p``.
    """
    if not volatility_has_finite_dm_p(volatility):
        return []
    if hypotheses_include_h3(hypotheses, require_discovery=True):
        return []
    return ["hypothesis_h3_missing_despite_finite_dm_p"]


def h1_hypothesis_consistency_errors(rankers: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H1 notebook consistency (Day Wave 31).

    Contract (research diagnostic only):
    - No ``oracle_raw`` / non-dict / missing / non-finite ``p_ic`` → no errors (skip).
    - Finite ``p_ic`` without ``H1_ranking_oracle``
      (discovery) → ``hypothesis_h1_missing_despite_finite_p_ic``.
    """
    oracle = rankers_oracle_raw(rankers)
    if oracle is None or not oracle_has_finite_p_ic(oracle):
        return []
    if hypotheses_include_h1(hypotheses, require_discovery=True):
        return []
    return ["hypothesis_h1_missing_despite_finite_p_ic"]


def h2_hypothesis_consistency_errors(rankers: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H2 notebook consistency (Day Wave 31).

    Contract (research diagnostic only):
    - No ``oracle_raw`` / non-dict / missing / non-finite ``ls_p`` → no errors (skip).
    - Finite ``ls_p`` without ``H2_decile_mono``
      (discovery) → ``hypothesis_h2_missing_despite_finite_ls_p``.
    """
    oracle = rankers_oracle_raw(rankers)
    if oracle is None or not oracle_has_finite_ls_p(oracle):
        return []
    if hypotheses_include_h2(hypotheses, require_discovery=True):
        return []
    return ["hypothesis_h2_missing_despite_finite_ls_p"]


def h99_hypothesis_consistency_errors(ranking: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H46 data-snooping notebook consistency.

    Contract (research diagnostic only):
    - No ``data_snooping`` block / non-dict / non-finite ``spa_p_consistent``
      → no errors (skip).
    - Finite consistent SPA p without ``H46_ranking_data_snooping`` (discovery)
      → ``hypothesis_h99_missing_despite_finite_spa_p``.
    """
    blob = ranking_data_snooping_blob(ranking)
    if blob is None or not data_snooping_has_finite_spa_p(blob):
        return []
    if hypotheses_include_h99(hypotheses, require_discovery=True):
        return []
    return ["hypothesis_h99_missing_despite_finite_spa_p"]


def h7_hypothesis_consistency_errors(conformal: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H7 notebook consistency (Day Wave 32).

    Contract (research diagnostic only):
    - No ``aci`` / non-dict / missing / non-finite ``kupiec_p`` → no errors (skip).
    - Finite ACI ``kupiec_p`` without ``H7_aci_coverage``
      (calibration) → ``hypothesis_h7_missing_despite_finite_aci_kupiec_p``.
    """
    aci = conformal_aci_blob(conformal)
    if aci is None or not aci_has_finite_kupiec_p(aci):
        return []
    if hypotheses_include_h7(hypotheses, require_calibration=True):
        return []
    return ["hypothesis_h7_missing_despite_finite_aci_kupiec_p"]


def h8_hypothesis_consistency_errors(conformal: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H8 notebook consistency (Day Wave 33).

    Contract (research diagnostic only):
    - No ``mondrian_aci`` / non-dict / missing / non-finite ``high_x_kupiec_p`` → no errors (skip).
    - Finite Mondrian ``high_x_kupiec_p`` without ``H8_mondrian_high_vol``
      (calibration) → ``hypothesis_h8_missing_despite_finite_high_x_kupiec_p``.
    """
    mond = conformal_mondrian_aci_blob(conformal)
    if mond is None or not mondrian_aci_has_finite_high_x_kupiec_p(mond):
        return []
    if hypotheses_include_h8(hypotheses, require_calibration=True):
        return []
    return ["hypothesis_h8_missing_despite_finite_high_x_kupiec_p"]


def h11_hypothesis_consistency_errors(crc: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H11 notebook consistency (Day Wave 34).

    Contract (research diagnostic only):
    - Non-dict / missing / non-finite ``kupiec_p`` → no errors (skip).
    - Finite CRC ``kupiec_p`` without ``H11_crc_var`` (calibration) →
      ``hypothesis_h11_missing_despite_finite_crc_kupiec_p``.
    """
    if not crc_has_finite_kupiec_p(crc):
        return []
    if hypotheses_include_h11(hypotheses, require_calibration=True):
        return []
    return ["hypothesis_h11_missing_despite_finite_crc_kupiec_p"]


def h12_hypothesis_consistency_errors(wcqr: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H12 notebook consistency (Day Wave 35).

    Contract (research diagnostic only):
    - Non-dict / missing / non-finite ``kupiec_p`` → no errors (skip).
    - Finite weighted_conformal ``kupiec_p`` without ``H12_weighted_cqr``
      (calibration) → ``hypothesis_h12_missing_despite_finite_wcqr_kupiec_p``.
    """
    if not weighted_conformal_has_finite_kupiec_p(wcqr):
        return []
    if hypotheses_include_h12(hypotheses, require_calibration=True):
        return []
    return ["hypothesis_h12_missing_despite_finite_wcqr_kupiec_p"]


def h9_hypothesis_consistency_errors(evalues: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H9 notebook consistency (Day Wave 36).

    Contract (research diagnostic only):
    - Non-dict / missing / non-finite ``e_sup`` → no errors (skip).
    - Finite evalues ``e_sup`` without ``H9_eprocess_aci`` (calibration) →
      ``hypothesis_h9_missing_despite_finite_e_sup``.
    """
    if not evalues_has_finite_e_sup(evalues):
        return []
    if hypotheses_include_h9(hypotheses, require_calibration=True):
        return []
    return ["hypothesis_h9_missing_despite_finite_e_sup"]


def h10_hypothesis_consistency_errors(jp: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H10 notebook consistency (Day Wave 37).

    Contract (research diagnostic only):
    - Non-dict / missing / non-finite ``coverage`` → no errors (skip).
    - Finite jackknife_plus ``coverage`` without ``H10_jackknife_coverage``
      (bound) → ``hypothesis_h10_missing_despite_finite_coverage``.
    """
    if not jackknife_plus_has_finite_coverage(jp):
        return []
    if hypotheses_include_h10(hypotheses, require_bound=True):
        return []
    return ["hypothesis_h10_missing_despite_finite_coverage"]


def h15_hypothesis_consistency_errors(cvp: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H15 notebook consistency (Day Wave 38).

    Contract (research diagnostic only):
    - Non-dict / missing / non-finite ``coverage`` or ``coverage_floor`` → no errors (skip).
    - Finite cv_plus ``coverage`` + ``coverage_floor`` without ``H15_cv_plus_floor``
      (bound) → ``hypothesis_h15_missing_despite_finite_coverage_and_floor``.
    """
    if not cv_plus_has_finite_coverage_and_floor(cvp):
        return []
    if hypotheses_include_h15(hypotheses, require_bound=True):
        return []
    return ["hypothesis_h15_missing_despite_finite_coverage_and_floor"]


def h16_h18_panel_kupiec_consistency_errors(families: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H16–H18 panel Kupiec notebook consistency (Day Wave 39).

    Contract (research diagnostic only):
    - Non-dict families / missing family / fixture ``dgp`` / non-finite ``kupiec_p`` → skip.
    - Finite panel ``kupiec_p`` without matching H16/H17/H18 (calibration) → error token.
    """
    if not isinstance(families, dict):
        return []
    errors: list[str] = []
    for family_key, hyp_id, err_token in H16_H18_PANEL_SPECS:
        blob = families.get(family_key)
        if not panel_family_has_finite_kupiec_p(blob):
            continue
        if hypotheses_include_id(hypotheses, hyp_id, require_family=H16_H18_EXPECTED_FAMILY):
            continue
        errors.append(err_token)
    return errors


def h19_hypothesis_consistency_errors(topk: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H19 notebook consistency (Day Wave 40).

    Contract (research diagnostic only):
    - Non-dict / missing / fixture ``dgp`` / non-finite ``fdr`` → no errors (skip).
    - Finite conformal_rank ``fdr`` (non-fixture) without ``H19_conformal_rank``
      (bound) → ``hypothesis_h19_missing_despite_finite_fdr``.
    """
    if not conformal_rank_has_finite_fdr(topk):
        return []
    if hypotheses_include_h19(hypotheses, require_bound=True):
        return []
    return ["hypothesis_h19_missing_despite_finite_fdr"]


def h20_hypothesis_consistency_errors(blob: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H20 Northset OHLC identity consistency."""
    if not northset_has_finite_ohlc_identity_rate(blob):
        return []
    if hypotheses_include_h20(hypotheses, require_bound=True):
        return []
    return ["hypothesis_h20_missing_despite_finite_ohlc_identity_rate"]


def h21_hypothesis_consistency_errors(blob: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H21 Northset uncrossed-book consistency."""
    if not northset_has_finite_book_uncrossed_rate(blob):
        return []
    if hypotheses_include_h21(hypotheses, require_bound=True):
        return []
    return ["hypothesis_h21_missing_despite_finite_book_uncrossed_rate"]


def h22_hypothesis_consistency_errors(blob: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H22 Northset imbalance IC consistency."""
    if not northset_has_finite_imbalance_p_ic(blob):
        return []
    if hypotheses_include_h22(hypotheses, require_discovery=True):
        return []
    return ["hypothesis_h22_missing_despite_finite_imbalance_p_ic"]


def h43_hypothesis_consistency_errors(blob: object, hypotheses: object) -> list[str]:
    """Soft-verify: finite session_book_vpin_p_ic → H43 discovery row present."""
    if not northset_has_finite_session_book_vpin_p_ic(blob):
        return []
    if hypotheses_include_h43(hypotheses, require_discovery=True):
        return []
    return ["hypothesis_h43_missing_despite_finite_session_book_vpin_p_ic"]


def northset_h23_h28_consistency_errors(blob: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for Northset H23–H51 notebook consistency."""
    if not isinstance(blob, dict):
        return []
    errors: list[str] = []
    book_metrics = {"microprice_p_ic", "ofi_p_ic", "vpin_p_ic"}
    for key, hyp_id, family, token in NORTHSET_H23_H28_SPECS:
        if key in book_metrics and blob.get("book_hypothesis_eligible", True) is False:
            continue
        if not _finite_scalar(blob.get(key)):
            continue
        if hypotheses_include_id(hypotheses, hyp_id, require_family=family):
            continue
        errors.append(token)
    return errors


def coverage_guarantee_scope_consistency_errors(families: object) -> list[str]:
    """Return soft-verify errors for Jackknife+/CV+ marginal scope honesty (Day Wave 42).

    Contract (research diagnostic only):
    - Non-dict families / empty blob / neither ``coverage`` nor ``coverage_floor`` → skip.
    - Nonempty jp/cv blob with coverage keys but missing ``coverage_guarantee_scope``
      → ``coverage_guarantee_scope_missing:<fam>``.
    - Key present but value != ``marginal_exchangeable``
      → ``coverage_guarantee_scope_invalid:<fam>``.
    - NaN coverage/floor still requires scope (key presence only).
    """
    if not isinstance(families, dict):
        return []
    errors: list[str] = []
    for fam in _JP_CV_SCOPE_FAMILIES:
        blob = families.get(fam)
        if not jp_cv_blob_requires_marginal_coverage_scope(blob):
            continue
        assert isinstance(blob, dict)
        if "coverage_guarantee_scope" not in blob:
            errors.append(f"coverage_guarantee_scope_missing:{fam}")
        elif blob.get("coverage_guarantee_scope") != COVERAGE_GUARANTEE_SCOPE_MARGINAL:
            errors.append(f"coverage_guarantee_scope_invalid:{fam}")
    return errors


def northset_use_session_l2_gate_consistency_errors(blob: object) -> list[str]:
    """Soft-verify ``use_session_l2`` ↔ ``session_l2_identity_gate`` when both present.

    Stamp contract from ``bench_northset``:
    - ``use_session_l2 is True`` ⇔ gate ``"enforced"``
    - ``use_session_l2 is False`` ⇔ gate ``"skipped"``

    One present without the other → skip (pair incomplete). Invalid gate string
    still caught by floor/string-enum helpers. Research diagnostic only; never
    live Sharpe. Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    if "use_session_l2" not in blob or "session_l2_identity_gate" not in blob:
        return []
    use = blob.get("use_session_l2")
    gate = blob.get("session_l2_identity_gate")
    if type(use) is not bool:
        return []  # bool-flags helper owns type
    if gate not in ("enforced", "skipped"):
        return []  # enum helper owns invalid gate
    if use and gate != "enforced":
        return ["use_session_l2_true_gate_not_enforced"]
    if (not use) and gate != "skipped":
        return ["use_session_l2_false_gate_not_skipped"]
    return []


def northset_session_book_snaps_n_session_consistency_errors(blob: object) -> list[str]:
    """Soft-verify mean_session_book_snaps vs n_session_book_rows / n_session_candles.

    Fail-closed when both sides of the session-L2 coverage pair are present and
    disagree:

    - Finite ``mean_session_book_snaps`` (path on) → ``n_session_book_rows`` must
      be present and ``> 0``; if ``n_session_candles`` is also present, rows must
      equal candles (1:1 session book vs session candle panel totals).
    - ``n_session_book_rows > 0`` with ``mean_session_book_snaps`` present → mean
      must be finite and ``> 0``.

    Skip when mean is NaN/absent and rows are 0/absent (session L2 off).
    Never equate ``mean_session_book_snaps`` to ``n_session_candles`` (per-parent
    mean ≠ panel row count). Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []

    has_mean = "mean_session_book_snaps" in blob
    has_rows = "n_session_book_rows" in blob and blob.get("n_session_book_rows") is not None
    has_candles = "n_session_candles" in blob and blob.get("n_session_candles") is not None

    mean_x: float | None = None
    if has_mean:
        raw_mean = blob.get("mean_session_book_snaps")
        if raw_mean is None:
            mean_x = None
        elif isinstance(raw_mean, bool) or not isinstance(raw_mean, (int, float)):
            # Present-but-malformed must not read as absent/NaN.
            return ["mean_session_book_snaps_non_numeric"]
        else:
            mean_x = float(raw_mean)

    # Parse through float first: int(float("inf")) raises OverflowError, which
    # a JSON receipt with Infinity counts would otherwise crash on.
    rows_n: int | None = None
    if has_rows:
        try:
            rows_f = float(blob.get("n_session_book_rows"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return ["n_session_book_rows_non_integer"]
        if rows_f != rows_f or abs(rows_f) == float("inf") or rows_f != int(rows_f):
            return ["n_session_book_rows_non_integer"]
        rows_n = int(rows_f)

    candles_n: int | None = None
    if has_candles:
        try:
            candles_f = float(blob.get("n_session_candles"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return ["n_session_candles_non_integer"]
        if candles_f != candles_f or abs(candles_f) == float("inf") or candles_f != int(candles_f):
            return ["n_session_candles_non_integer"]
        candles_n = int(candles_f)

    mean_finite = mean_x is not None and mean_x == mean_x and abs(mean_x) != float("inf")
    mean_active = mean_finite and mean_x > 0.0  # type: ignore[operator]
    rows_pos = rows_n is not None and rows_n > 0

    # Session L2 off: NaN/absent mean + zero/absent rows → skip
    if not mean_active and not rows_pos:
        return []

    errs: list[str] = []

    if mean_active:
        if not has_rows:
            errs.append("n_session_book_rows_missing_despite_mean_session_book_snaps")
        elif rows_n is not None and rows_n <= 0:
            errs.append("n_session_book_rows_non_positive_despite_mean_session_book_snaps")
        elif rows_n is not None and candles_n is not None and rows_n != candles_n:
            errs.append("n_session_book_rows_ne_n_session_candles")

    if rows_pos and has_mean and (not mean_finite or not (mean_x > 0.0)):  # type: ignore[operator]
        errs.append("mean_session_book_snaps_non_positive_despite_n_session_book_rows")

    return errs


NORTHSET_SESSION_MEANS_HONESTY_HELPERS = (
    northset_session_imbalance_mean_honesty_errors,
    northset_session_close_micro_bps_honesty_errors,
    northset_session_close_depth_honesty_errors,
    northset_session_imbalance_std_honesty_errors,
    northset_session_close_imbalance_honesty_errors,
    northset_session_close_mid_honesty_errors,
    northset_session_close_mid_micro_pair_honesty_errors,
    northset_session_close_spread_bps_honesty_errors,
    northset_session_spread_bps_mean_honesty_errors,
    northset_session_book_snaps_honesty_errors,
    northset_session_book_snaps_n_session_consistency_errors,
    northset_session_ofi_sum_mean_honesty_errors,
    northset_session_ofi_sum_ic_honesty_errors,
    northset_session_ofi_abs_sum_honesty_errors,
    northset_session_ofi_abs_dominates_sum_honesty_errors,
    northset_session_book_vpin_mean_honesty_errors,
)


def northset_session_means_honesty_errors(blob: object) -> list[str]:
    """Dispatcher: fan into all session-mean soft-verify helpers.

    Keeps individual helpers for unit tests; verify.py can call this once.
    Research diagnostic only; never live Sharpe / promotion.
    """
    errors: list[str] = []
    for fn in NORTHSET_SESSION_MEANS_HONESTY_HELPERS:
        errors.extend(fn(blob))
    return errors


NORTHSET_RECEIPT_HONESTY_HELPERS: tuple[Callable[..., list[str]], ...] = (
    mean_book_age_seconds_honesty_errors,
    book_age_seconds_honesty_errors,
    mean_microprice_minus_mid_honesty_errors,
    northset_log_size_slope_honesty_errors,
    northset_qlike_means_honesty_errors,
    northset_range_spread_honesty_errors,
    amihud_mean_honesty_errors,
    northset_queue_sweep_ofi_honesty_errors,
    northset_queue_priority_le_size_concentration_honesty_errors,
    northset_queue_priority_bid_ask_pair_honesty_errors,
    depth_shape_finite_rate_honesty_errors,
    northset_structure_finite_rate_distinct_from_candle_honesty_errors,
    candle_structure_finite_rate_covers_companions_honesty_errors,
    northset_overnight_rv_semi_honesty_errors,
    northset_book_shape_finite_rates_honesty_errors,
    session_volume_conservation_rate_honesty_errors,
    session_reconstructs_daily_rate_honesty_errors,
    session_ohlc_vs_reconstructs_never_equate_honesty_errors,
    session_ohlc_vs_volume_conservation_never_equate_honesty_errors,
    session_ohlc_vs_session_chain_never_equate_honesty_errors,
    session_volume_conservation_vs_reconstructs_never_equate_honesty_errors,
    session_chain_vs_session_identity_siblings_never_equate_honesty_errors,
    book_uncrossed_rate_honesty_errors,
    ohlc_identity_rate_honesty_errors,
    book_uncrossed_vs_imbalance_p_ic_never_equate_honesty_errors,
    ohlc_identity_vs_imbalance_p_ic_never_equate_honesty_errors,
    session_bulk_vpin_vs_siblings_never_equate_honesty_errors,
    ohlc_identity_vs_book_uncrossed_never_equate_honesty_errors,
    ohlc_identity_vs_gap_finite_never_equate_honesty_errors,
    gap_finite_rate_vs_book_uncrossed_never_equate_honesty_errors,
    session_ohlc_vs_gap_finite_never_equate_honesty_errors,
    session_ohlc_vs_book_uncrossed_never_equate_honesty_errors,
    book_uncrossed_vs_session_chain_never_equate_honesty_errors,
    imbalance_top_p_ic_vs_session_chain_never_equate_honesty_errors,
    imbalance_top_p_ic_vs_session_ohlc_never_equate_honesty_errors,
    imbalance_top_p_ic_vs_session_volume_conservation_never_equate_honesty_errors,
    book_uncrossed_vs_session_reconstructs_never_equate_honesty_errors,
    book_uncrossed_vs_volume_conservation_never_equate_honesty_errors,
    gap_finite_rate_vs_session_chain_never_equate_honesty_errors,
    gap_finite_rate_vs_session_volume_conservation_never_equate_honesty_errors,
    gap_finite_rate_vs_session_reconstructs_never_equate_honesty_errors,
    imbalance_top_p_ic_vs_gap_finite_never_equate_honesty_errors,
    imbalance_top_p_ic_vs_session_reconstructs_never_equate_honesty_errors,
    microprice_p_ic_vs_gap_finite_never_equate_honesty_errors,
    microprice_p_ic_vs_session_chain_never_equate_honesty_errors,
    clv_p_ic_vs_gap_finite_never_equate_honesty_errors,
    dm_split_vs_park_p_vs_gap_finite_never_equate_honesty_errors,
    sweep_follow_event_p_vs_gap_finite_never_equate_honesty_errors,
    sweep_reject_cost_adjusted_mean_bps_vs_gap_finite_never_equate_honesty_errors,
    sweep_reject_control_diff_p_vs_gap_finite_never_equate_honesty_errors,
    wick_skew_p_ic_vs_gap_finite_never_equate_honesty_errors,
    ofi_p_ic_vs_gap_finite_never_equate_honesty_errors,
    dm_gk_vs_park_p_vs_gap_finite_never_equate_honesty_errors,
    sweep_reject_event_p_vs_gap_finite_never_equate_honesty_errors,
    sweep_follow_placebo_p_vs_gap_finite_never_equate_honesty_errors,
    sweep_follow_fold_positive_fraction_vs_gap_finite_never_equate_honesty_errors,
    vpin_p_ic_vs_gap_finite_never_equate_honesty_errors,
    sweep_reject_signed_p_ic_vs_gap_finite_never_equate_honesty_errors,
    sweep_follow_signed_p_ic_vs_gap_finite_never_equate_honesty_errors,
    sweep_reject_control_diff_p_vs_sweep_follow_control_diff_p_never_equate_honesty_errors,
    sweep_reject_fold_positive_fraction_vs_sweep_follow_fold_positive_fraction_never_equate_honesty_errors,
    sweep_reject_event_p_vs_sweep_follow_event_p_never_equate_honesty_errors,
    sweep_reject_signed_p_ic_vs_sweep_follow_signed_p_ic_never_equate_honesty_errors,
    sweep_reject_cost_adjusted_mean_bps_vs_sweep_follow_cost_adjusted_mean_bps_never_equate_honesty_errors,
    sweep_reject_placebo_p_vs_sweep_follow_placebo_p_never_equate_honesty_errors,
    sweep_reject_placebo_p_vs_gap_finite_never_equate_honesty_errors,
    sweep_follow_cost_adjusted_mean_bps_vs_gap_finite_never_equate_honesty_errors,
    sweep_reject_fold_positive_fraction_vs_gap_finite_never_equate_honesty_errors,
    sweep_follow_control_diff_p_vs_gap_finite_never_equate_honesty_errors,
    session_book_vpin_p_ic_vs_gap_finite_never_equate_honesty_errors,
    ohlc_identity_vs_session_ohlc_never_equate_honesty_errors,
    ohlc_identity_vs_session_reconstructs_never_equate_honesty_errors,
    ohlc_identity_vs_session_chain_never_equate_honesty_errors,
    ohlc_identity_vs_volume_conservation_never_equate_honesty_errors,
    session_chain_rate_honesty_errors,
    session_mean_jump_ratio_honesty_errors,
    sweep_follow_signed_mean_ic_honesty_errors,
    sweep_reject_signed_mean_ic_honesty_errors,
    sweep_reject_event_mean_bps_honesty_errors,
    sweep_follow_event_mean_bps_honesty_errors,
    sweep_follow_cost_adjusted_mean_bps_honesty_errors,
    sweep_reject_cost_adjusted_mean_bps_honesty_errors,
    northset_n_bars_scored_honesty_errors,
    ofi_p_ic_honesty_errors,
    northset_session_ofi_sum_ic_honesty_errors,
    northset_vpin_sweep_fold_honesty_errors,
    microprice_p_ic_honesty_errors,
    clv_p_ic_honesty_errors,
    northset_product_stamp_honesty_errors,
    northset_impact_proxy_warning_honesty_errors,
    close_location_value_clv_alias_identity_honesty_errors,
    ofi_mean_ic_honesty_errors,
    northset_ofi_lag_ic_honesty_errors,
    northset_vpin_ic_pack_honesty_errors,
    northset_queue_imbalance_ic_honesty_errors,
    imbalance_top_mean_ic_honesty_errors,
    northset_microprice_bps_ic_pack_honesty_errors,
    northset_clv_ic_pack_honesty_errors,
    northset_imbalance_top_ic_pack_honesty_errors,
    northset_session_book_vpin_ic_pack_honesty_errors,
    northset_candle_body_ret_ic_pack_honesty_errors,
    northset_wick_skew_ic_pack_honesty_errors,
    northset_bid_log_size_slope_ic_pack_honesty_errors,
    northset_imbalance_depth_ic_pack_honesty_errors,
    northset_amihud_ic_pack_honesty_errors,
    amihud_mean_ic_vs_amihud_abs_mean_ic_never_equate_honesty_errors,
    northset_volume_over_range_ic_packs_honesty_errors,
    northset_sweep_signed_ic_packs_honesty_errors,
    northset_session_close_ic_packs_honesty_errors,
    northset_all_n_dates_nonneg_honesty_errors,
    northset_all_p_ic_unit_interval_honesty_errors,
    northset_receipt_dgp_data_source_honesty_errors,
    northset_receipt_string_enum_honesty_errors,
    northset_sweep_evidence_scope_honesty_errors,
    northset_dm_park_honesty_errors,
    northset_sweep_evidence_blob_honesty_errors,
    northset_all_rate_unit_honesty_errors,
    northset_all_share_unit_honesty_errors,
    northset_all_fraction_unit_honesty_errors,
    northset_sweep_fold_positive_rates_honesty_errors,
    northset_queue_imbalance_mean_alias_honesty_errors,
    northset_all_floor_unit_honesty_errors,
    northset_all_finite_rate_unit_honesty_errors,
    northset_all_mean_rank_ic_unit_honesty_errors,
    northset_all_mean_ic_finite_honesty_errors,
    northset_all_t_ic_finite_honesty_errors,
    book_hypothesis_eligible_honesty_errors,
    northset_family_book_source_honesty_errors,
    northset_price_return_basis_honesty_errors,
    northset_depth_honesty_errors,
    northset_component_sources_honesty_errors,
    northset_use_session_l2_gate_consistency_errors,
    northset_include_kyle_ofi_nest_presence_honesty_errors,
    northset_sweep_control_sample_adequate_honesty_errors,
    northset_shape_columns_ensured_book_panel_path_honesty_errors,
    northset_receipt_bool_flags_honesty_errors,
    northset_lag_corr_and_sweep_count_honesty_errors,
    mid_lag1_corr_vs_ofi_lag1_corr_never_equate_honesty_errors,
    northset_kyle_r2_unit_honesty_errors,
    kyle_r2_vs_kyle_ofi_r2_never_equate_honesty_errors,
    gap_finite_rate_honesty_errors,
    vpin_mean_honesty_errors,
    session_bulk_vpin_honesty_errors,
    northset_spread_means_honesty_errors,
    northset_session_identity_rates_honesty_errors,
    northset_session_l2_enforced_identity_rates_present_honesty_errors,
    northset_half_spread_honesty_errors,
    northset_microprice_weight_balance_honesty_errors,
    northset_spread_bps_honesty_errors,
    northset_spread_receipt_honesty_errors,
)


def northset_receipt_honesty_errors(blob: object) -> list[str]:
    """Dispatcher: northset receipt soft-verify helpers not covered by session means.

    Fans out to rate/IC/spread/shape/sweep companions. Session path means stay in
    northset_session_means_honesty_errors. Kyle nest stays in kyle_ofi_nest_*.
    Research diagnostic only; never live Sharpe.
    """
    errs: list[str] = []
    for fn in NORTHSET_RECEIPT_HONESTY_HELPERS:
        errs.extend(fn(blob))
    return errs
