"""Authoritative catalog of Dipcatcher research benchmark families.

Dual honesty catalogs (research scorecard vs paper analytics export)
-------------------------------------------------------------------
- **Research family / scorecard blobs** (bench outputs, H-table payloads):
  ``FORBIDDEN_RESEARCH_METRIC_KEYS`` / ``family_blob_forbidden_metrics_absent``
  enforce headline hygiene — no sharpe / sortino / calmar / pnl / nav *key tokens*.
- **Paper/backtest ``analytics_export``** (see ``metrics.analytics``): equity /
  stress diagnostics may intentionally carry ``pnl`` / ``nav`` keys, but
  ``live_pnl_claim`` must stay false and ``validate_analytics_export`` fails closed
  on ``live_pnl_claim=true``. Do **not** apply the research forbidden-key scanner
  to the full export blob — different catalogs, different contracts.
"""

from __future__ import annotations

from ._helpers import (
    _CANDLE_FEATURE_IC_METHOD_ALLOWED as _CANDLE_FEATURE_IC_METHOD_ALLOWED,
)
from ._helpers import (
    _DATA_SNOOPING_P_KEYS as _DATA_SNOOPING_P_KEYS,
)
from ._helpers import (
    _JP_CV_SCOPE_FAMILIES as _JP_CV_SCOPE_FAMILIES,
)
from ._helpers import (
    _KYLE_FORBIDDEN_TOKENS as _KYLE_FORBIDDEN_TOKENS,
)
from ._helpers import (
    _KYLE_OFI_IC_METHOD_ALLOWED as _KYLE_OFI_IC_METHOD_ALLOWED,
)
from ._helpers import (
    _KYLE_RESIDUAL_SPEARMAN_KEYS as _KYLE_RESIDUAL_SPEARMAN_KEYS,
)
from ._helpers import (
    _NORTHSET_CORWIN_SCHULTZ_PAIR_SCOPE_ALLOWED as _NORTHSET_CORWIN_SCHULTZ_PAIR_SCOPE_ALLOWED,
)
from ._helpers import (
    _NORTHSET_IMPACT_ESTIMATOR_SCOPE_ALLOWED as _NORTHSET_IMPACT_ESTIMATOR_SCOPE_ALLOWED,
)
from ._helpers import (
    _NORTHSET_OVERNIGHT_GAP_METHOD_ALLOWED as _NORTHSET_OVERNIGHT_GAP_METHOD_ALLOWED,
)
from ._helpers import (
    _NORTHSET_PRICE_BASIS_ALLOWED as _NORTHSET_PRICE_BASIS_ALLOWED,
)
from ._helpers import (
    _NORTHSET_RETURN_BASIS_ALLOWED as _NORTHSET_RETURN_BASIS_ALLOWED,
)
from ._helpers import (
    _NORTHSET_SPREAD_ABS_TOL as _NORTHSET_SPREAD_ABS_TOL,
)
from ._helpers import (
    _NORTHSET_SPREAD_REL_TOL as _NORTHSET_SPREAD_REL_TOL,
)
from ._helpers import (
    _NORTHSET_SWEEP_EVIDENCE_SCOPE_ALLOWED as _NORTHSET_SWEEP_EVIDENCE_SCOPE_ALLOWED,
)
from ._helpers import (
    _NORTHSET_TWO_WAY_INFERENCE_INDEX_ALLOWED as _NORTHSET_TWO_WAY_INFERENCE_INDEX_ALLOWED,
)
from ._helpers import (
    _NORTHSET_VPIN_METHOD_ALLOWED as _NORTHSET_VPIN_METHOD_ALLOWED,
)
from ._helpers import (
    _NORTHSET_YANG_ZHANG_QLIKE_SCOPE_ALLOWED as _NORTHSET_YANG_ZHANG_QLIKE_SCOPE_ALLOWED,
)
from ._helpers import (
    _candle_has_feature_ic_marker as _candle_has_feature_ic_marker,
)
from ._helpers import (
    _finite_pair as _finite_pair,
)
from ._helpers import (
    _finite_scalar as _finite_scalar,
)
from ._helpers import (
    _ic_pack_honesty_errors as _ic_pack_honesty_errors,
)
from ._helpers import (
    _iter_mapping_keys as _iter_mapping_keys,
)
from ._helpers import (
    _kyle_ofi_blob as _kyle_ofi_blob,
)
from ._helpers import (
    _kyle_ofi_has_nest_diagnostic_marker as _kyle_ofi_has_nest_diagnostic_marker,
)
from .candle import (
    candle_all_finite_rate_prefix_honesty_errors as candle_all_finite_rate_prefix_honesty_errors,
)
from .candle import (
    candle_all_ic_n_dates_nonneg_honesty_errors as candle_all_ic_n_dates_nonneg_honesty_errors,
)
from .candle import (
    candle_all_ic_p_unit_honesty_errors as candle_all_ic_p_unit_honesty_errors,
)
from .candle import (
    candle_all_ic_pearson_unit_honesty_errors as candle_all_ic_pearson_unit_honesty_errors,
)
from .candle import (
    candle_all_ic_t_finite_honesty_errors as candle_all_ic_t_finite_honesty_errors,
)
from .candle import (
    candle_depth_imbalance_ic_implies_mean_honesty_errors as candle_depth_imbalance_ic_implies_mean_honesty_errors,
)
from .candle import (
    candle_direction_mean_honesty_errors as candle_direction_mean_honesty_errors,
)
from .candle import (
    candle_feature_cols_ic_completeness_honesty_errors as candle_feature_cols_ic_completeness_honesty_errors,
)
from .candle import (
    candle_feature_cols_ic_honesty_errors as candle_feature_cols_ic_honesty_errors,
)
from .candle import (
    candle_feature_cols_ic_implies_mean_honesty_errors as candle_feature_cols_ic_implies_mean_honesty_errors,
)
from .candle import (
    candle_feature_ofi_finite_honesty_errors as candle_feature_ofi_finite_honesty_errors,
)
from .candle import (
    candle_frac_and_spread_x_honesty_errors as candle_frac_and_spread_x_honesty_errors,
)
from .candle import (
    candle_join_coverage_and_chain_honesty_errors as candle_join_coverage_and_chain_honesty_errors,
)
from .candle import (
    candle_log_slopes_finite_honesty_errors as candle_log_slopes_finite_honesty_errors,
)
from .candle import (
    candle_log_tick_spacing_finite_honesty_errors as candle_log_tick_spacing_finite_honesty_errors,
)
from .candle import (
    candle_microprice_minus_mid_finite_pack_honesty_errors as candle_microprice_minus_mid_finite_pack_honesty_errors,
)
from .candle import (
    candle_microprice_weight_balance_ic_honesty_errors as candle_microprice_weight_balance_ic_honesty_errors,
)
from .candle import (
    candle_mwb_scored_implies_mean_unit_honesty_errors as candle_mwb_scored_implies_mean_unit_honesty_errors,
)
from .candle import (
    candle_notional_imbalance_ic_implies_mean_honesty_errors as candle_notional_imbalance_ic_implies_mean_honesty_errors,
)
from .candle import (
    candle_ofi_and_queue_imbalance_means_honesty_errors as candle_ofi_and_queue_imbalance_means_honesty_errors,
)
from .candle import (
    candle_ofi_qp_slope_ic_implies_mean_honesty_errors as candle_ofi_qp_slope_ic_implies_mean_honesty_errors,
)
from .candle import (
    candle_order_book_claim_honesty_errors as candle_order_book_claim_honesty_errors,
)
from .candle import (
    candle_order_book_dgp_data_source_honesty_errors as candle_order_book_dgp_data_source_honesty_errors,
)
from .candle import (
    candle_order_book_family_provenance_honesty_errors as candle_order_book_family_provenance_honesty_errors,
)
from .candle import (
    candle_order_book_ic_method_honesty_errors as candle_order_book_ic_method_honesty_errors,
)
from .candle import (
    candle_order_book_sizing_honesty_errors as candle_order_book_sizing_honesty_errors,
)
from .candle import (
    candle_signed_vol_x_imbalance_mean_honesty_errors as candle_signed_vol_x_imbalance_mean_honesty_errors,
)
from .candle import (
    candle_spread_bps_ic_matches_spread_over_mid_ic_honesty_errors as candle_spread_bps_ic_matches_spread_over_mid_ic_honesty_errors,
)
from .candle import (
    candle_spread_bps_nonneg_honesty_errors as candle_spread_bps_nonneg_honesty_errors,
)
from .candle import (
    candle_spread_over_mid_ic_implies_mean_honesty_errors as candle_spread_over_mid_ic_implies_mean_honesty_errors,
)
from .candle import (
    candle_structure_finite_rate_covers_companions_honesty_errors as candle_structure_finite_rate_covers_companions_honesty_errors,
)
from .candle import (
    candle_structure_ic_implies_mean_honesty_errors as candle_structure_ic_implies_mean_honesty_errors,
)
from .candle import (
    candle_wick_skew_and_body_ret_means_honesty_errors as candle_wick_skew_and_body_ret_means_honesty_errors,
)
from .consistency import (
    NORTHSET_RECEIPT_HONESTY_HELPERS as NORTHSET_RECEIPT_HONESTY_HELPERS,
)
from .consistency import (
    NORTHSET_SESSION_MEANS_HONESTY_HELPERS as NORTHSET_SESSION_MEANS_HONESTY_HELPERS,
)
from .consistency import (
    coverage_guarantee_scope_consistency_errors as coverage_guarantee_scope_consistency_errors,
)
from .consistency import (
    h1_hypothesis_consistency_errors as h1_hypothesis_consistency_errors,
)
from .consistency import (
    h2_hypothesis_consistency_errors as h2_hypothesis_consistency_errors,
)
from .consistency import (
    h3_hypothesis_consistency_errors as h3_hypothesis_consistency_errors,
)
from .consistency import (
    h4_hypothesis_consistency_errors as h4_hypothesis_consistency_errors,
)
from .consistency import (
    h4b_hypothesis_consistency_errors as h4b_hypothesis_consistency_errors,
)
from .consistency import (
    h7_hypothesis_consistency_errors as h7_hypothesis_consistency_errors,
)
from .consistency import (
    h8_hypothesis_consistency_errors as h8_hypothesis_consistency_errors,
)
from .consistency import (
    h9_hypothesis_consistency_errors as h9_hypothesis_consistency_errors,
)
from .consistency import (
    h10_hypothesis_consistency_errors as h10_hypothesis_consistency_errors,
)
from .consistency import (
    h11_hypothesis_consistency_errors as h11_hypothesis_consistency_errors,
)
from .consistency import (
    h12_hypothesis_consistency_errors as h12_hypothesis_consistency_errors,
)
from .consistency import (
    h15_hypothesis_consistency_errors as h15_hypothesis_consistency_errors,
)
from .consistency import (
    h16_h18_panel_kupiec_consistency_errors as h16_h18_panel_kupiec_consistency_errors,
)
from .consistency import (
    h19_hypothesis_consistency_errors as h19_hypothesis_consistency_errors,
)
from .consistency import (
    h20_hypothesis_consistency_errors as h20_hypothesis_consistency_errors,
)
from .consistency import (
    h21_hypothesis_consistency_errors as h21_hypothesis_consistency_errors,
)
from .consistency import (
    h22_hypothesis_consistency_errors as h22_hypothesis_consistency_errors,
)
from .consistency import (
    h43_hypothesis_consistency_errors as h43_hypothesis_consistency_errors,
)
from .consistency import (
    h99_hypothesis_consistency_errors as h99_hypothesis_consistency_errors,
)
from .consistency import (
    northset_h23_h28_consistency_errors as northset_h23_h28_consistency_errors,
)
from .consistency import (
    northset_receipt_honesty_errors as northset_receipt_honesty_errors,
)
from .consistency import (
    northset_session_book_snaps_n_session_consistency_errors as northset_session_book_snaps_n_session_consistency_errors,
)
from .consistency import (
    northset_session_means_honesty_errors as northset_session_means_honesty_errors,
)
from .consistency import (
    northset_use_session_l2_gate_consistency_errors as northset_use_session_l2_gate_consistency_errors,
)
from .constants import (
    BENCHMARK_CATALOG_VERSION as BENCHMARK_CATALOG_VERSION,
)
from .constants import (
    BENCHMARK_FAMILY_ORDER as BENCHMARK_FAMILY_ORDER,
)
from .constants import (
    COVERAGE_GUARANTEE_SCOPE_MARGINAL as COVERAGE_GUARANTEE_SCOPE_MARGINAL,
)
from .constants import (
    DIST_CRPS_EPROCESS_REQUIRED_WHEN_DM as DIST_CRPS_EPROCESS_REQUIRED_WHEN_DM,
)
from .constants import (
    DIST_CRPS_SCALED_EPROCESS_REQUIRED_WHEN_DM as DIST_CRPS_SCALED_EPROCESS_REQUIRED_WHEN_DM,
)
from .constants import (
    DM_CRPS_MARKER_KEY as DM_CRPS_MARKER_KEY,
)
from .constants import (
    DM_CRPS_SCALED_MARKER_KEY as DM_CRPS_SCALED_MARKER_KEY,
)
from .constants import (
    ES_MARKER_KEYS as ES_MARKER_KEYS,
)
from .constants import (
    FORBIDDEN_RESEARCH_METRIC_KEYS as FORBIDDEN_RESEARCH_METRIC_KEYS,
)
from .constants import (
    H1_EXPECTED_FAMILY as H1_EXPECTED_FAMILY,
)
from .constants import (
    H1_HYPOTHESIS_ID as H1_HYPOTHESIS_ID,
)
from .constants import (
    H2_EXPECTED_FAMILY as H2_EXPECTED_FAMILY,
)
from .constants import (
    H2_HYPOTHESIS_ID as H2_HYPOTHESIS_ID,
)
from .constants import (
    H3_EXPECTED_FAMILY as H3_EXPECTED_FAMILY,
)
from .constants import (
    H3_HYPOTHESIS_ID as H3_HYPOTHESIS_ID,
)
from .constants import (
    H4_EXPECTED_FAMILY as H4_EXPECTED_FAMILY,
)
from .constants import (
    H4_HYPOTHESIS_ID as H4_HYPOTHESIS_ID,
)
from .constants import (
    H4B_EXPECTED_FAMILY as H4B_EXPECTED_FAMILY,
)
from .constants import (
    H4B_HYPOTHESIS_ID as H4B_HYPOTHESIS_ID,
)
from .constants import (
    H7_EXPECTED_FAMILY as H7_EXPECTED_FAMILY,
)
from .constants import (
    H7_HYPOTHESIS_ID as H7_HYPOTHESIS_ID,
)
from .constants import (
    H8_EXPECTED_FAMILY as H8_EXPECTED_FAMILY,
)
from .constants import (
    H8_HYPOTHESIS_ID as H8_HYPOTHESIS_ID,
)
from .constants import (
    H9_EXPECTED_FAMILY as H9_EXPECTED_FAMILY,
)
from .constants import (
    H9_HYPOTHESIS_ID as H9_HYPOTHESIS_ID,
)
from .constants import (
    H10_EXPECTED_FAMILY as H10_EXPECTED_FAMILY,
)
from .constants import (
    H10_HYPOTHESIS_ID as H10_HYPOTHESIS_ID,
)
from .constants import (
    H11_EXPECTED_FAMILY as H11_EXPECTED_FAMILY,
)
from .constants import (
    H11_HYPOTHESIS_ID as H11_HYPOTHESIS_ID,
)
from .constants import (
    H12_EXPECTED_FAMILY as H12_EXPECTED_FAMILY,
)
from .constants import (
    H12_HYPOTHESIS_ID as H12_HYPOTHESIS_ID,
)
from .constants import (
    H15_EXPECTED_FAMILY as H15_EXPECTED_FAMILY,
)
from .constants import (
    H15_HYPOTHESIS_ID as H15_HYPOTHESIS_ID,
)
from .constants import (
    H16_H18_EXPECTED_FAMILY as H16_H18_EXPECTED_FAMILY,
)
from .constants import (
    H16_H18_PANEL_SPECS as H16_H18_PANEL_SPECS,
)
from .constants import (
    H16_HYPOTHESIS_ID as H16_HYPOTHESIS_ID,
)
from .constants import (
    H17_HYPOTHESIS_ID as H17_HYPOTHESIS_ID,
)
from .constants import (
    H18_HYPOTHESIS_ID as H18_HYPOTHESIS_ID,
)
from .constants import (
    H19_EXPECTED_FAMILY as H19_EXPECTED_FAMILY,
)
from .constants import (
    H19_HYPOTHESIS_ID as H19_HYPOTHESIS_ID,
)
from .constants import (
    H20_EXPECTED_FAMILY as H20_EXPECTED_FAMILY,
)
from .constants import (
    H20_HYPOTHESIS_ID as H20_HYPOTHESIS_ID,
)
from .constants import (
    H21_EXPECTED_FAMILY as H21_EXPECTED_FAMILY,
)
from .constants import (
    H21_HYPOTHESIS_ID as H21_HYPOTHESIS_ID,
)
from .constants import (
    H22_EXPECTED_FAMILY as H22_EXPECTED_FAMILY,
)
from .constants import (
    H22_HYPOTHESIS_ID as H22_HYPOTHESIS_ID,
)
from .constants import (
    H23_HYPOTHESIS_ID as H23_HYPOTHESIS_ID,
)
from .constants import (
    H24_HYPOTHESIS_ID as H24_HYPOTHESIS_ID,
)
from .constants import (
    H25_HYPOTHESIS_ID as H25_HYPOTHESIS_ID,
)
from .constants import (
    H26_HYPOTHESIS_ID as H26_HYPOTHESIS_ID,
)
from .constants import (
    H27_HYPOTHESIS_ID as H27_HYPOTHESIS_ID,
)
from .constants import (
    H28_HYPOTHESIS_ID as H28_HYPOTHESIS_ID,
)
from .constants import (
    H29_HYPOTHESIS_ID as H29_HYPOTHESIS_ID,
)
from .constants import (
    H30_HYPOTHESIS_ID as H30_HYPOTHESIS_ID,
)
from .constants import (
    H31_HYPOTHESIS_ID as H31_HYPOTHESIS_ID,
)
from .constants import (
    H32_HYPOTHESIS_ID as H32_HYPOTHESIS_ID,
)
from .constants import (
    H33_HYPOTHESIS_ID as H33_HYPOTHESIS_ID,
)
from .constants import (
    H34_HYPOTHESIS_ID as H34_HYPOTHESIS_ID,
)
from .constants import (
    H35_HYPOTHESIS_ID as H35_HYPOTHESIS_ID,
)
from .constants import (
    H36_HYPOTHESIS_ID as H36_HYPOTHESIS_ID,
)
from .constants import (
    H37_HYPOTHESIS_ID as H37_HYPOTHESIS_ID,
)
from .constants import (
    H38_HYPOTHESIS_ID as H38_HYPOTHESIS_ID,
)
from .constants import (
    H39_HYPOTHESIS_ID as H39_HYPOTHESIS_ID,
)
from .constants import (
    H40_HYPOTHESIS_ID as H40_HYPOTHESIS_ID,
)
from .constants import (
    H41_HYPOTHESIS_ID as H41_HYPOTHESIS_ID,
)
from .constants import (
    H42_HYPOTHESIS_ID as H42_HYPOTHESIS_ID,
)
from .constants import (
    H43_EXPECTED_FAMILY as H43_EXPECTED_FAMILY,
)
from .constants import (
    H43_HYPOTHESIS_ID as H43_HYPOTHESIS_ID,
)
from .constants import (
    H44_HYPOTHESIS_ID as H44_HYPOTHESIS_ID,
)
from .constants import (
    H45_HYPOTHESIS_ID as H45_HYPOTHESIS_ID,
)
from .constants import (
    H46_HYPOTHESIS_ID as H46_HYPOTHESIS_ID,
)
from .constants import (
    H47_HYPOTHESIS_ID as H47_HYPOTHESIS_ID,
)
from .constants import (
    H48_HYPOTHESIS_ID as H48_HYPOTHESIS_ID,
)
from .constants import (
    H49_HYPOTHESIS_ID as H49_HYPOTHESIS_ID,
)
from .constants import (
    H50_HYPOTHESIS_ID as H50_HYPOTHESIS_ID,
)
from .constants import (
    H51_HYPOTHESIS_ID as H51_HYPOTHESIS_ID,
)
from .constants import (
    H99_EXPECTED_FAMILY as H99_EXPECTED_FAMILY,
)
from .constants import (
    H99_HYPOTHESIS_ID as H99_HYPOTHESIS_ID,
)
from .constants import (
    KUPIEC_MARKER_KEYS as KUPIEC_MARKER_KEYS,
)
from .constants import (
    NORTHSET_H23_H28_SPECS as NORTHSET_H23_H28_SPECS,
)
from .constants import (
    OPTIONAL_BENCHMARK_FAMILIES as OPTIONAL_BENCHMARK_FAMILIES,
)
from .constants import (
    PREFERRED_CHRISTOFFERSEN_IND_KEYS as PREFERRED_CHRISTOFFERSEN_IND_KEYS,
)
from .constants import (
    REQUIRED_BENCHMARK_FAMILIES as REQUIRED_BENCHMARK_FAMILIES,
)
from .constants import (
    REQUIRED_CHRISTOFFERSEN_CC_KEYS as REQUIRED_CHRISTOFFERSEN_CC_KEYS,
)
from .constants import (
    RESEARCH_RECEIPT_SCHEMA_VERSION as RESEARCH_RECEIPT_SCHEMA_VERSION,
)
from .constants import (
    RESEARCH_RECEIPT_SCHEMA_VERSIONS_ACCEPTED as RESEARCH_RECEIPT_SCHEMA_VERSIONS_ACCEPTED,
)
from .constants import (
    TAIL_ES_BATTERY_REQUIRED_WHEN_ES_MARKERS as TAIL_ES_BATTERY_REQUIRED_WHEN_ES_MARKERS,
)
from .constants import (
    TAIL_VAR_BATTERY_REQUIRED_WHEN_KUPIEC as TAIL_VAR_BATTERY_REQUIRED_WHEN_KUPIEC,
)
from .families import (
    conformal_aci_payload as conformal_aci_payload,
)
from .families import (
    conformal_mondrian_aci_payload as conformal_mondrian_aci_payload,
)
from .families import (
    coverage_guarantee_scope_is_marginal as coverage_guarantee_scope_is_marginal,
)
from .families import (
    dist_crps_eprocess_keys_present as dist_crps_eprocess_keys_present,
)
from .families import (
    dist_crps_eprocess_missing_keys as dist_crps_eprocess_missing_keys,
)
from .families import (
    family_blob_executed as family_blob_executed,
)
from .families import (
    family_blob_forbidden_metrics_absent as family_blob_forbidden_metrics_absent,
)
from .families import (
    family_blob_has_finite_observation as family_blob_has_finite_observation,
)
from .families import (
    family_blob_nonempty as family_blob_nonempty,
)
from .families import (
    hypotheses_include_h1 as hypotheses_include_h1,
)
from .families import (
    hypotheses_include_h2 as hypotheses_include_h2,
)
from .families import (
    hypotheses_include_h3 as hypotheses_include_h3,
)
from .families import (
    hypotheses_include_h4 as hypotheses_include_h4,
)
from .families import (
    hypotheses_include_h4b as hypotheses_include_h4b,
)
from .families import (
    hypotheses_include_h7 as hypotheses_include_h7,
)
from .families import (
    hypotheses_include_h8 as hypotheses_include_h8,
)
from .families import (
    hypotheses_include_h9 as hypotheses_include_h9,
)
from .families import (
    hypotheses_include_h10 as hypotheses_include_h10,
)
from .families import (
    hypotheses_include_h11 as hypotheses_include_h11,
)
from .families import (
    hypotheses_include_h12 as hypotheses_include_h12,
)
from .families import (
    hypotheses_include_h15 as hypotheses_include_h15,
)
from .families import (
    hypotheses_include_h16 as hypotheses_include_h16,
)
from .families import (
    hypotheses_include_h17 as hypotheses_include_h17,
)
from .families import (
    hypotheses_include_h18 as hypotheses_include_h18,
)
from .families import (
    hypotheses_include_h19 as hypotheses_include_h19,
)
from .families import (
    hypotheses_include_h20 as hypotheses_include_h20,
)
from .families import (
    hypotheses_include_h21 as hypotheses_include_h21,
)
from .families import (
    hypotheses_include_h22 as hypotheses_include_h22,
)
from .families import (
    hypotheses_include_h43 as hypotheses_include_h43,
)
from .families import (
    hypotheses_include_h99 as hypotheses_include_h99,
)
from .families import (
    hypotheses_include_id as hypotheses_include_id,
)
from .families import (
    jp_cv_blob_requires_marginal_coverage_scope as jp_cv_blob_requires_marginal_coverage_scope,
)
from .families import (
    rankers_oracle_raw as rankers_oracle_raw,
)
from .families import (
    tail_es_battery_keys_present as tail_es_battery_keys_present,
)
from .families import (
    tail_es_battery_missing_keys as tail_es_battery_missing_keys,
)
from .families import (
    tail_var_battery_keys_present as tail_var_battery_keys_present,
)
from .families import (
    tail_var_battery_missing_keys as tail_var_battery_missing_keys,
)
from .kyle import (
    kyle_lambda_date_series_honesty_errors as kyle_lambda_date_series_honesty_errors,
)
from .kyle import (
    kyle_lambda_dispersion_honesty_errors as kyle_lambda_dispersion_honesty_errors,
)
from .kyle import (
    kyle_lambda_ofi_depth_corr_honesty_errors as kyle_lambda_ofi_depth_corr_honesty_errors,
)
from .kyle import (
    kyle_ofi_book_panel_path_honesty_errors as kyle_ofi_book_panel_path_honesty_errors,
)
from .kyle import (
    kyle_ofi_date_series_counts_honesty_errors as kyle_ofi_date_series_counts_honesty_errors,
)
from .kyle import (
    kyle_ofi_diagnostic_string_honesty_errors as kyle_ofi_diagnostic_string_honesty_errors,
)
from .kyle import (
    kyle_ofi_dispersion_window_honesty_errors as kyle_ofi_dispersion_window_honesty_errors,
)
from .kyle import (
    kyle_ofi_family_honesty_errors as kyle_ofi_family_honesty_errors,
)
from .kyle import (
    kyle_ofi_hac_lags_honesty_errors as kyle_ofi_hac_lags_honesty_errors,
)
from .kyle import (
    kyle_ofi_ic_method_honesty_errors as kyle_ofi_ic_method_honesty_errors,
)
from .kyle import (
    kyle_ofi_join_coverage_honesty_errors as kyle_ofi_join_coverage_honesty_errors,
)
from .kyle import (
    kyle_ofi_label_nonempty_honesty_errors as kyle_ofi_label_nonempty_honesty_errors,
)
from .kyle import (
    kyle_ofi_label_synthetic_honesty_errors as kyle_ofi_label_synthetic_honesty_errors,
)
from .kyle import (
    kyle_ofi_lambda_decile_order_honesty_errors as kyle_ofi_lambda_decile_order_honesty_errors,
)
from .kyle import (
    kyle_ofi_min_join_coverage_pair_honesty_errors as kyle_ofi_min_join_coverage_pair_honesty_errors,
)
from .kyle import (
    kyle_ofi_min_names_honesty_errors as kyle_ofi_min_names_honesty_errors,
)
from .kyle import (
    kyle_ofi_n_dates_companion_honesty_errors as kyle_ofi_n_dates_companion_honesty_errors,
)
from .kyle import (
    kyle_ofi_n_fused_scored_honesty_errors as kyle_ofi_n_fused_scored_honesty_errors,
)
from .kyle import (
    kyle_ofi_nest_claim_honesty_errors as kyle_ofi_nest_claim_honesty_errors,
)
from .kyle import (
    kyle_ofi_nest_honesty_errors as kyle_ofi_nest_honesty_errors,
)
from .kyle import (
    kyle_ofi_pvalue_honesty_errors as kyle_ofi_pvalue_honesty_errors,
)
from .kyle import (
    kyle_ofi_rolling_mean_hac_band_honesty_errors as kyle_ofi_rolling_mean_hac_band_honesty_errors,
)
from .kyle import (
    kyle_ofi_spearman_honesty_errors as kyle_ofi_spearman_honesty_errors,
)
from .kyle import (
    kyle_ofi_std_iqr_honesty_errors as kyle_ofi_std_iqr_honesty_errors,
)
from .kyle import (
    kyle_ofi_synthetic_source_honesty_errors as kyle_ofi_synthetic_source_honesty_errors,
)
from .kyle import (
    kyle_ofi_tstat_honesty_errors as kyle_ofi_tstat_honesty_errors,
)
from .kyle import (
    kyle_r2_vs_kyle_ofi_r2_never_equate_honesty_errors as kyle_r2_vs_kyle_ofi_r2_never_equate_honesty_errors,
)
from .kyle import (
    kyle_residual_flow_honesty_errors as kyle_residual_flow_honesty_errors,
)
from .northset import (
    candle_spread_alias_honesty_errors as candle_spread_alias_honesty_errors,
)
from .northset import (
    northset_all_finite_rate_unit_honesty_errors as northset_all_finite_rate_unit_honesty_errors,
)
from .northset import (
    northset_all_floor_unit_honesty_errors as northset_all_floor_unit_honesty_errors,
)
from .northset import (
    northset_all_fraction_unit_honesty_errors as northset_all_fraction_unit_honesty_errors,
)
from .northset import (
    northset_all_mean_ic_finite_honesty_errors as northset_all_mean_ic_finite_honesty_errors,
)
from .northset import (
    northset_all_mean_rank_ic_unit_honesty_errors as northset_all_mean_rank_ic_unit_honesty_errors,
)
from .northset import (
    northset_all_n_dates_nonneg_honesty_errors as northset_all_n_dates_nonneg_honesty_errors,
)
from .northset import (
    northset_all_p_ic_unit_interval_honesty_errors as northset_all_p_ic_unit_interval_honesty_errors,
)
from .northset import (
    northset_all_rate_unit_honesty_errors as northset_all_rate_unit_honesty_errors,
)
from .northset import (
    northset_all_share_unit_honesty_errors as northset_all_share_unit_honesty_errors,
)
from .northset import (
    northset_all_t_ic_finite_honesty_errors as northset_all_t_ic_finite_honesty_errors,
)
from .northset import (
    northset_amihud_ic_pack_honesty_errors as northset_amihud_ic_pack_honesty_errors,
)
from .northset import (
    northset_bid_log_size_slope_ic_pack_honesty_errors as northset_bid_log_size_slope_ic_pack_honesty_errors,
)
from .northset import (
    northset_book_shape_finite_rates_honesty_errors as northset_book_shape_finite_rates_honesty_errors,
)
from .northset import (
    northset_candle_body_ret_ic_pack_honesty_errors as northset_candle_body_ret_ic_pack_honesty_errors,
)
from .northset import (
    northset_clv_ic_pack_honesty_errors as northset_clv_ic_pack_honesty_errors,
)
from .northset import (
    northset_component_sources_honesty_errors as northset_component_sources_honesty_errors,
)
from .northset import (
    northset_depth_honesty_errors as northset_depth_honesty_errors,
)
from .northset import (
    northset_depth_notional_spread_over_mid_honesty_errors as northset_depth_notional_spread_over_mid_honesty_errors,
)
from .northset import (
    northset_dm_park_honesty_errors as northset_dm_park_honesty_errors,
)
from .northset import (
    northset_family_book_source_honesty_errors as northset_family_book_source_honesty_errors,
)
from .northset import (
    northset_fwd_ret_after_sweep_honesty_errors as northset_fwd_ret_after_sweep_honesty_errors,
)
from .northset import (
    northset_half_spread_honesty_errors as northset_half_spread_honesty_errors,
)
from .northset import (
    northset_imbalance_depth_ic_pack_honesty_errors as northset_imbalance_depth_ic_pack_honesty_errors,
)
from .northset import (
    northset_imbalance_top_ic_pack_honesty_errors as northset_imbalance_top_ic_pack_honesty_errors,
)
from .northset import (
    northset_impact_proxy_warning_honesty_errors as northset_impact_proxy_warning_honesty_errors,
)
from .northset import (
    northset_include_kyle_ofi_nest_presence_honesty_errors as northset_include_kyle_ofi_nest_presence_honesty_errors,
)
from .northset import (
    northset_kyle_r2_unit_honesty_errors as northset_kyle_r2_unit_honesty_errors,
)
from .northset import (
    northset_lag_corr_and_sweep_count_honesty_errors as northset_lag_corr_and_sweep_count_honesty_errors,
)
from .northset import (
    northset_log_size_slope_honesty_errors as northset_log_size_slope_honesty_errors,
)
from .northset import (
    northset_metrics_required_finite_ok_rates_honesty_errors as northset_metrics_required_finite_ok_rates_honesty_errors,
)
from .northset import (
    northset_metrics_required_keys_finite_when_present_honesty_errors as northset_metrics_required_keys_finite_when_present_honesty_errors,
)
from .northset import (
    northset_microprice_bps_ic_pack_honesty_errors as northset_microprice_bps_ic_pack_honesty_errors,
)
from .northset import (
    northset_microprice_weight_balance_honesty_errors as northset_microprice_weight_balance_honesty_errors,
)
from .northset import (
    northset_n_bars_scored_honesty_errors as northset_n_bars_scored_honesty_errors,
)
from .northset import (
    northset_ofi_lag_ic_honesty_errors as northset_ofi_lag_ic_honesty_errors,
)
from .northset import (
    northset_overnight_rv_semi_honesty_errors as northset_overnight_rv_semi_honesty_errors,
)
from .northset import (
    northset_price_return_basis_honesty_errors as northset_price_return_basis_honesty_errors,
)
from .northset import (
    northset_price_slope_tick_top_levels_honesty_errors as northset_price_slope_tick_top_levels_honesty_errors,
)
from .northset import (
    northset_product_stamp_honesty_errors as northset_product_stamp_honesty_errors,
)
from .northset import (
    northset_qlike_means_honesty_errors as northset_qlike_means_honesty_errors,
)
from .northset import (
    northset_queue_imbalance_ic_honesty_errors as northset_queue_imbalance_ic_honesty_errors,
)
from .northset import (
    northset_queue_imbalance_mean_alias_honesty_errors as northset_queue_imbalance_mean_alias_honesty_errors,
)
from .northset import (
    northset_queue_priority_bid_ask_pair_honesty_errors as northset_queue_priority_bid_ask_pair_honesty_errors,
)
from .northset import (
    northset_queue_priority_le_size_concentration_honesty_errors as northset_queue_priority_le_size_concentration_honesty_errors,
)
from .northset import (
    northset_queue_sweep_ofi_honesty_errors as northset_queue_sweep_ofi_honesty_errors,
)
from .northset import (
    northset_range_spread_honesty_errors as northset_range_spread_honesty_errors,
)
from .northset import (
    northset_receipt_bool_flags_honesty_errors as northset_receipt_bool_flags_honesty_errors,
)
from .northset import (
    northset_receipt_dgp_data_source_honesty_errors as northset_receipt_dgp_data_source_honesty_errors,
)
from .northset import (
    northset_receipt_string_enum_honesty_errors as northset_receipt_string_enum_honesty_errors,
)
from .northset import (
    northset_session_book_snaps_honesty_errors as northset_session_book_snaps_honesty_errors,
)
from .northset import (
    northset_session_book_vpin_ic_pack_honesty_errors as northset_session_book_vpin_ic_pack_honesty_errors,
)
from .northset import (
    northset_session_book_vpin_mean_honesty_errors as northset_session_book_vpin_mean_honesty_errors,
)
from .northset import (
    northset_session_close_depth_honesty_errors as northset_session_close_depth_honesty_errors,
)
from .northset import (
    northset_session_close_ic_packs_honesty_errors as northset_session_close_ic_packs_honesty_errors,
)
from .northset import (
    northset_session_close_imbalance_honesty_errors as northset_session_close_imbalance_honesty_errors,
)
from .northset import (
    northset_session_close_micro_bps_honesty_errors as northset_session_close_micro_bps_honesty_errors,
)
from .northset import (
    northset_session_close_mid_honesty_errors as northset_session_close_mid_honesty_errors,
)
from .northset import (
    northset_session_close_mid_micro_pair_honesty_errors as northset_session_close_mid_micro_pair_honesty_errors,
)
from .northset import (
    northset_session_close_spread_bps_honesty_errors as northset_session_close_spread_bps_honesty_errors,
)
from .northset import (
    northset_session_identity_rates_honesty_errors as northset_session_identity_rates_honesty_errors,
)
from .northset import (
    northset_session_imbalance_mean_honesty_errors as northset_session_imbalance_mean_honesty_errors,
)
from .northset import (
    northset_session_imbalance_std_honesty_errors as northset_session_imbalance_std_honesty_errors,
)
from .northset import (
    northset_session_l2_enforced_identity_rates_present_honesty_errors as northset_session_l2_enforced_identity_rates_present_honesty_errors,
)
from .northset import (
    northset_session_ofi_abs_dominates_sum_honesty_errors as northset_session_ofi_abs_dominates_sum_honesty_errors,
)
from .northset import (
    northset_session_ofi_abs_sum_honesty_errors as northset_session_ofi_abs_sum_honesty_errors,
)
from .northset import (
    northset_session_ofi_sum_ic_honesty_errors as northset_session_ofi_sum_ic_honesty_errors,
)
from .northset import (
    northset_session_ofi_sum_mean_honesty_errors as northset_session_ofi_sum_mean_honesty_errors,
)
from .northset import (
    northset_session_spread_bps_mean_honesty_errors as northset_session_spread_bps_mean_honesty_errors,
)
from .northset import (
    northset_shape_and_session_l2_floors_honesty_errors as northset_shape_and_session_l2_floors_honesty_errors,
)
from .northset import (
    northset_shape_columns_ensured_book_panel_path_honesty_errors as northset_shape_columns_ensured_book_panel_path_honesty_errors,
)
from .northset import (
    northset_shape_columns_ensured_rates_honesty_errors as northset_shape_columns_ensured_rates_honesty_errors,
)
from .northset import (
    northset_spread_bps_honesty_errors as northset_spread_bps_honesty_errors,
)
from .northset import (
    northset_spread_means_honesty_errors as northset_spread_means_honesty_errors,
)
from .northset import (
    northset_spread_receipt_honesty_errors as northset_spread_receipt_honesty_errors,
)
from .northset import (
    northset_structure_finite_rate_distinct_from_candle_honesty_errors as northset_structure_finite_rate_distinct_from_candle_honesty_errors,
)
from .northset import (
    northset_sweep_control_sample_adequate_honesty_errors as northset_sweep_control_sample_adequate_honesty_errors,
)
from .northset import (
    northset_sweep_evidence_blob_honesty_errors as northset_sweep_evidence_blob_honesty_errors,
)
from .northset import (
    northset_sweep_evidence_scope_honesty_errors as northset_sweep_evidence_scope_honesty_errors,
)
from .northset import (
    northset_sweep_fold_positive_rates_honesty_errors as northset_sweep_fold_positive_rates_honesty_errors,
)
from .northset import (
    northset_sweep_signed_ic_packs_honesty_errors as northset_sweep_signed_ic_packs_honesty_errors,
)
from .northset import (
    northset_top_level_claim_honesty_errors as northset_top_level_claim_honesty_errors,
)
from .northset import (
    northset_volume_over_range_ic_packs_honesty_errors as northset_volume_over_range_ic_packs_honesty_errors,
)
from .northset import (
    northset_vpin_ic_pack_honesty_errors as northset_vpin_ic_pack_honesty_errors,
)
from .northset import (
    northset_vpin_sweep_fold_honesty_errors as northset_vpin_sweep_fold_honesty_errors,
)
from .northset import (
    northset_wick_skew_ic_pack_honesty_errors as northset_wick_skew_ic_pack_honesty_errors,
)
from .predicates import (
    aci_has_finite_kupiec_p as aci_has_finite_kupiec_p,
)
from .predicates import (
    conformal_aci_blob as conformal_aci_blob,
)
from .predicates import (
    conformal_mondrian_aci_blob as conformal_mondrian_aci_blob,
)
from .predicates import (
    conformal_rank_has_finite_fdr as conformal_rank_has_finite_fdr,
)
from .predicates import (
    crc_has_finite_kupiec_p as crc_has_finite_kupiec_p,
)
from .predicates import (
    cv_plus_has_finite_coverage_and_floor as cv_plus_has_finite_coverage_and_floor,
)
from .predicates import (
    data_snooping_has_finite_spa_p as data_snooping_has_finite_spa_p,
)
from .predicates import (
    evalues_has_finite_e_sup as evalues_has_finite_e_sup,
)
from .predicates import (
    jackknife_plus_has_finite_coverage as jackknife_plus_has_finite_coverage,
)
from .predicates import (
    mondrian_aci_has_finite_high_x_kupiec_p as mondrian_aci_has_finite_high_x_kupiec_p,
)
from .predicates import (
    mondrian_has_finite_high_x_kupiec_p as mondrian_has_finite_high_x_kupiec_p,
)
from .predicates import (
    northset_has_finite_book_uncrossed_rate as northset_has_finite_book_uncrossed_rate,
)
from .predicates import (
    northset_has_finite_imbalance_p_ic as northset_has_finite_imbalance_p_ic,
)
from .predicates import (
    northset_has_finite_ohlc_identity_rate as northset_has_finite_ohlc_identity_rate,
)
from .predicates import (
    northset_has_finite_session_book_vpin_p_ic as northset_has_finite_session_book_vpin_p_ic,
)
from .predicates import (
    oracle_has_finite_ls_p as oracle_has_finite_ls_p,
)
from .predicates import (
    oracle_has_finite_p_ic as oracle_has_finite_p_ic,
)
from .predicates import (
    panel_family_has_finite_kupiec_p as panel_family_has_finite_kupiec_p,
)
from .predicates import (
    ranking_data_snooping_blob as ranking_data_snooping_blob,
)
from .predicates import (
    tail_has_finite_christoffersen_cc_p as tail_has_finite_christoffersen_cc_p,
)
from .predicates import (
    tail_has_finite_kupiec_p as tail_has_finite_kupiec_p,
)
from .predicates import (
    volatility_has_finite_dm_p as volatility_has_finite_dm_p,
)
from .predicates import (
    weighted_conformal_has_finite_kupiec_p as weighted_conformal_has_finite_kupiec_p,
)
from .session import (
    amihud_mean_honesty_errors as amihud_mean_honesty_errors,
)
from .session import (
    amihud_mean_ic_vs_amihud_abs_mean_ic_never_equate_honesty_errors as amihud_mean_ic_vs_amihud_abs_mean_ic_never_equate_honesty_errors,
)
from .session import (
    book_age_seconds_honesty_errors as book_age_seconds_honesty_errors,
)
from .session import (
    book_hypothesis_eligible_honesty_errors as book_hypothesis_eligible_honesty_errors,
)
from .session import (
    book_uncrossed_rate_honesty_errors as book_uncrossed_rate_honesty_errors,
)
from .session import (
    book_uncrossed_vs_imbalance_p_ic_never_equate_honesty_errors as book_uncrossed_vs_imbalance_p_ic_never_equate_honesty_errors,
)
from .session import (
    book_uncrossed_vs_session_chain_never_equate_honesty_errors as book_uncrossed_vs_session_chain_never_equate_honesty_errors,
)
from .session import (
    book_uncrossed_vs_session_reconstructs_never_equate_honesty_errors as book_uncrossed_vs_session_reconstructs_never_equate_honesty_errors,
)
from .session import (
    book_uncrossed_vs_volume_conservation_never_equate_honesty_errors as book_uncrossed_vs_volume_conservation_never_equate_honesty_errors,
)
from .session import (
    close_location_value_clv_alias_identity_honesty_errors as close_location_value_clv_alias_identity_honesty_errors,
)
from .session import (
    clv_p_ic_honesty_errors as clv_p_ic_honesty_errors,
)
from .session import (
    clv_p_ic_vs_gap_finite_never_equate_honesty_errors as clv_p_ic_vs_gap_finite_never_equate_honesty_errors,
)
from .session import (
    depth_shape_finite_rate_honesty_errors as depth_shape_finite_rate_honesty_errors,
)
from .session import (
    dm_gk_vs_park_p_vs_gap_finite_never_equate_honesty_errors as dm_gk_vs_park_p_vs_gap_finite_never_equate_honesty_errors,
)
from .session import (
    dm_split_vs_park_p_vs_gap_finite_never_equate_honesty_errors as dm_split_vs_park_p_vs_gap_finite_never_equate_honesty_errors,
)
from .session import (
    gap_finite_rate_honesty_errors as gap_finite_rate_honesty_errors,
)
from .session import (
    gap_finite_rate_vs_book_uncrossed_never_equate_honesty_errors as gap_finite_rate_vs_book_uncrossed_never_equate_honesty_errors,
)
from .session import (
    gap_finite_rate_vs_session_chain_never_equate_honesty_errors as gap_finite_rate_vs_session_chain_never_equate_honesty_errors,
)
from .session import (
    gap_finite_rate_vs_session_reconstructs_never_equate_honesty_errors as gap_finite_rate_vs_session_reconstructs_never_equate_honesty_errors,
)
from .session import (
    gap_finite_rate_vs_session_volume_conservation_never_equate_honesty_errors as gap_finite_rate_vs_session_volume_conservation_never_equate_honesty_errors,
)
from .session import (
    imbalance_top_mean_ic_honesty_errors as imbalance_top_mean_ic_honesty_errors,
)
from .session import (
    imbalance_top_p_ic_vs_gap_finite_never_equate_honesty_errors as imbalance_top_p_ic_vs_gap_finite_never_equate_honesty_errors,
)
from .session import (
    imbalance_top_p_ic_vs_session_chain_never_equate_honesty_errors as imbalance_top_p_ic_vs_session_chain_never_equate_honesty_errors,
)
from .session import (
    imbalance_top_p_ic_vs_session_ohlc_never_equate_honesty_errors as imbalance_top_p_ic_vs_session_ohlc_never_equate_honesty_errors,
)
from .session import (
    imbalance_top_p_ic_vs_session_reconstructs_never_equate_honesty_errors as imbalance_top_p_ic_vs_session_reconstructs_never_equate_honesty_errors,
)
from .session import (
    imbalance_top_p_ic_vs_session_volume_conservation_never_equate_honesty_errors as imbalance_top_p_ic_vs_session_volume_conservation_never_equate_honesty_errors,
)
from .session import (
    join_coverage_honesty_errors as join_coverage_honesty_errors,
)
from .session import (
    mean_book_age_seconds_honesty_errors as mean_book_age_seconds_honesty_errors,
)
from .session import (
    mean_candle_dir_x_imbalance_honesty_errors as mean_candle_dir_x_imbalance_honesty_errors,
)
from .session import (
    mean_close_mid_abs_rel_honesty_errors as mean_close_mid_abs_rel_honesty_errors,
)
from .session import (
    mean_depth_imbalance_abs_honesty_errors as mean_depth_imbalance_abs_honesty_errors,
)
from .session import (
    mean_depth_imbalance_honesty_errors as mean_depth_imbalance_honesty_errors,
)
from .session import (
    mean_imbalance_top_honesty_errors as mean_imbalance_top_honesty_errors,
)
from .session import (
    mean_microprice_minus_mid_honesty_errors as mean_microprice_minus_mid_honesty_errors,
)
from .session import (
    mean_microprice_weight_balance_honesty_errors as mean_microprice_weight_balance_honesty_errors,
)
from .session import (
    mean_notional_imbalance_honesty_errors as mean_notional_imbalance_honesty_errors,
)
from .session import (
    mean_queue_priority_honesty_errors as mean_queue_priority_honesty_errors,
)
from .session import (
    mean_tob_notional_share_honesty_errors as mean_tob_notional_share_honesty_errors,
)
from .session import (
    mean_tob_size_share_honesty_errors as mean_tob_size_share_honesty_errors,
)
from .session import (
    microprice_p_ic_honesty_errors as microprice_p_ic_honesty_errors,
)
from .session import (
    microprice_p_ic_vs_gap_finite_never_equate_honesty_errors as microprice_p_ic_vs_gap_finite_never_equate_honesty_errors,
)
from .session import (
    microprice_p_ic_vs_session_chain_never_equate_honesty_errors as microprice_p_ic_vs_session_chain_never_equate_honesty_errors,
)
from .session import (
    mid_lag1_corr_vs_ofi_lag1_corr_never_equate_honesty_errors as mid_lag1_corr_vs_ofi_lag1_corr_never_equate_honesty_errors,
)
from .session import (
    ofi_mean_ic_honesty_errors as ofi_mean_ic_honesty_errors,
)
from .session import (
    ofi_p_ic_honesty_errors as ofi_p_ic_honesty_errors,
)
from .session import (
    ofi_p_ic_vs_gap_finite_never_equate_honesty_errors as ofi_p_ic_vs_gap_finite_never_equate_honesty_errors,
)
from .session import (
    ohlc_identity_rate_honesty_errors as ohlc_identity_rate_honesty_errors,
)
from .session import (
    ohlc_identity_vs_book_uncrossed_never_equate_honesty_errors as ohlc_identity_vs_book_uncrossed_never_equate_honesty_errors,
)
from .session import (
    ohlc_identity_vs_gap_finite_never_equate_honesty_errors as ohlc_identity_vs_gap_finite_never_equate_honesty_errors,
)
from .session import (
    ohlc_identity_vs_imbalance_p_ic_never_equate_honesty_errors as ohlc_identity_vs_imbalance_p_ic_never_equate_honesty_errors,
)
from .session import (
    ohlc_identity_vs_session_chain_never_equate_honesty_errors as ohlc_identity_vs_session_chain_never_equate_honesty_errors,
)
from .session import (
    ohlc_identity_vs_session_ohlc_never_equate_honesty_errors as ohlc_identity_vs_session_ohlc_never_equate_honesty_errors,
)
from .session import (
    ohlc_identity_vs_session_reconstructs_never_equate_honesty_errors as ohlc_identity_vs_session_reconstructs_never_equate_honesty_errors,
)
from .session import (
    ohlc_identity_vs_volume_conservation_never_equate_honesty_errors as ohlc_identity_vs_volume_conservation_never_equate_honesty_errors,
)
from .session import (
    ranking_data_snooping_honesty_errors as ranking_data_snooping_honesty_errors,
)
from .session import (
    robinhood_plus_claim_honesty_errors as robinhood_plus_claim_honesty_errors,
)
from .session import (
    session_book_vpin_p_ic_vs_gap_finite_never_equate_honesty_errors as session_book_vpin_p_ic_vs_gap_finite_never_equate_honesty_errors,
)
from .session import (
    session_bulk_vpin_honesty_errors as session_bulk_vpin_honesty_errors,
)
from .session import (
    session_bulk_vpin_vs_siblings_never_equate_honesty_errors as session_bulk_vpin_vs_siblings_never_equate_honesty_errors,
)
from .session import (
    session_chain_rate_honesty_errors as session_chain_rate_honesty_errors,
)
from .session import (
    session_chain_vs_session_identity_siblings_never_equate_honesty_errors as session_chain_vs_session_identity_siblings_never_equate_honesty_errors,
)
from .session import (
    session_mean_jump_ratio_honesty_errors as session_mean_jump_ratio_honesty_errors,
)
from .session import (
    session_ohlc_vs_book_uncrossed_never_equate_honesty_errors as session_ohlc_vs_book_uncrossed_never_equate_honesty_errors,
)
from .session import (
    session_ohlc_vs_gap_finite_never_equate_honesty_errors as session_ohlc_vs_gap_finite_never_equate_honesty_errors,
)
from .session import (
    session_ohlc_vs_reconstructs_never_equate_honesty_errors as session_ohlc_vs_reconstructs_never_equate_honesty_errors,
)
from .session import (
    session_ohlc_vs_session_chain_never_equate_honesty_errors as session_ohlc_vs_session_chain_never_equate_honesty_errors,
)
from .session import (
    session_ohlc_vs_volume_conservation_never_equate_honesty_errors as session_ohlc_vs_volume_conservation_never_equate_honesty_errors,
)
from .session import (
    session_reconstructs_daily_rate_honesty_errors as session_reconstructs_daily_rate_honesty_errors,
)
from .session import (
    session_volume_conservation_rate_honesty_errors as session_volume_conservation_rate_honesty_errors,
)
from .session import (
    session_volume_conservation_vs_reconstructs_never_equate_honesty_errors as session_volume_conservation_vs_reconstructs_never_equate_honesty_errors,
)
from .session import (
    size_concentration_top_honesty_errors as size_concentration_top_honesty_errors,
)
from .session import (
    structure_finite_rate_honesty_errors as structure_finite_rate_honesty_errors,
)
from .session import (
    sweep_follow_control_diff_p_vs_gap_finite_never_equate_honesty_errors as sweep_follow_control_diff_p_vs_gap_finite_never_equate_honesty_errors,
)
from .session import (
    sweep_follow_cost_adjusted_mean_bps_honesty_errors as sweep_follow_cost_adjusted_mean_bps_honesty_errors,
)
from .session import (
    sweep_follow_cost_adjusted_mean_bps_vs_gap_finite_never_equate_honesty_errors as sweep_follow_cost_adjusted_mean_bps_vs_gap_finite_never_equate_honesty_errors,
)
from .session import (
    sweep_follow_event_mean_bps_honesty_errors as sweep_follow_event_mean_bps_honesty_errors,
)
from .session import (
    sweep_follow_event_p_vs_gap_finite_never_equate_honesty_errors as sweep_follow_event_p_vs_gap_finite_never_equate_honesty_errors,
)
from .session import (
    sweep_follow_fold_positive_fraction_vs_gap_finite_never_equate_honesty_errors as sweep_follow_fold_positive_fraction_vs_gap_finite_never_equate_honesty_errors,
)
from .session import (
    sweep_follow_placebo_p_vs_gap_finite_never_equate_honesty_errors as sweep_follow_placebo_p_vs_gap_finite_never_equate_honesty_errors,
)
from .session import (
    sweep_follow_signed_mean_ic_honesty_errors as sweep_follow_signed_mean_ic_honesty_errors,
)
from .session import (
    sweep_follow_signed_p_ic_vs_gap_finite_never_equate_honesty_errors as sweep_follow_signed_p_ic_vs_gap_finite_never_equate_honesty_errors,
)
from .session import (
    sweep_reject_control_diff_p_vs_gap_finite_never_equate_honesty_errors as sweep_reject_control_diff_p_vs_gap_finite_never_equate_honesty_errors,
)
from .session import (
    sweep_reject_control_diff_p_vs_sweep_follow_control_diff_p_never_equate_honesty_errors as sweep_reject_control_diff_p_vs_sweep_follow_control_diff_p_never_equate_honesty_errors,
)
from .session import (
    sweep_reject_cost_adjusted_mean_bps_honesty_errors as sweep_reject_cost_adjusted_mean_bps_honesty_errors,
)
from .session import (
    sweep_reject_cost_adjusted_mean_bps_vs_gap_finite_never_equate_honesty_errors as sweep_reject_cost_adjusted_mean_bps_vs_gap_finite_never_equate_honesty_errors,
)
from .session import (
    sweep_reject_cost_adjusted_mean_bps_vs_sweep_follow_cost_adjusted_mean_bps_never_equate_honesty_errors as sweep_reject_cost_adjusted_mean_bps_vs_sweep_follow_cost_adjusted_mean_bps_never_equate_honesty_errors,
)
from .session import (
    sweep_reject_event_mean_bps_honesty_errors as sweep_reject_event_mean_bps_honesty_errors,
)
from .session import (
    sweep_reject_event_p_vs_gap_finite_never_equate_honesty_errors as sweep_reject_event_p_vs_gap_finite_never_equate_honesty_errors,
)
from .session import (
    sweep_reject_event_p_vs_sweep_follow_event_p_never_equate_honesty_errors as sweep_reject_event_p_vs_sweep_follow_event_p_never_equate_honesty_errors,
)
from .session import (
    sweep_reject_fold_positive_fraction_vs_gap_finite_never_equate_honesty_errors as sweep_reject_fold_positive_fraction_vs_gap_finite_never_equate_honesty_errors,
)
from .session import (
    sweep_reject_fold_positive_fraction_vs_sweep_follow_fold_positive_fraction_never_equate_honesty_errors as sweep_reject_fold_positive_fraction_vs_sweep_follow_fold_positive_fraction_never_equate_honesty_errors,
)
from .session import (
    sweep_reject_placebo_p_vs_gap_finite_never_equate_honesty_errors as sweep_reject_placebo_p_vs_gap_finite_never_equate_honesty_errors,
)
from .session import (
    sweep_reject_placebo_p_vs_sweep_follow_placebo_p_never_equate_honesty_errors as sweep_reject_placebo_p_vs_sweep_follow_placebo_p_never_equate_honesty_errors,
)
from .session import (
    sweep_reject_signed_mean_ic_honesty_errors as sweep_reject_signed_mean_ic_honesty_errors,
)
from .session import (
    sweep_reject_signed_p_ic_vs_gap_finite_never_equate_honesty_errors as sweep_reject_signed_p_ic_vs_gap_finite_never_equate_honesty_errors,
)
from .session import (
    sweep_reject_signed_p_ic_vs_sweep_follow_signed_p_ic_never_equate_honesty_errors as sweep_reject_signed_p_ic_vs_sweep_follow_signed_p_ic_never_equate_honesty_errors,
)
from .session import (
    vpin_mean_honesty_errors as vpin_mean_honesty_errors,
)
from .session import (
    vpin_p_ic_vs_gap_finite_never_equate_honesty_errors as vpin_p_ic_vs_gap_finite_never_equate_honesty_errors,
)
from .session import (
    wick_skew_p_ic_vs_gap_finite_never_equate_honesty_errors as wick_skew_p_ic_vs_gap_finite_never_equate_honesty_errors,
)
