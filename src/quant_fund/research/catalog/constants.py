"""Named hypothesis ids, specs, and policy constants for the research catalog."""

from __future__ import annotations

BENCHMARK_CATALOG_VERSION = 2


RESEARCH_RECEIPT_SCHEMA_VERSION = 1


REQUIRED_BENCHMARK_FAMILIES = frozenset(
    {
        "ranking",
        "alpha",
        "volatility",
        "distribution",
        "regime",
        "tail",
        "drawdown",
        "liquidity",
        "reinforcement",
        "conformal",
        "evalues",
        "jackknife_plus",
        "crc",
        "weighted_conformal",
        "interval_risk",
        "quantile_bandit",
        "cv_plus",
        "cpcv",
        "localized_conformal",
        "conformal_rank",
        "online_crc",
        "portfolio_conformal",
        "northset",
    }
)


OPTIONAL_BENCHMARK_FAMILIES = frozenset(
    {
        "candle_order_book",
        "robinhood_plus",
        # Descriptive proper-score diagnostics of the SYNTHETIC return panel
        # (see research/benches_extra.py); never live-P&L or headline ratios.
        "complexity",
        "roughness",
        "serial_randomness",
    }
)


BENCHMARK_FAMILY_ORDER = (
    "ranking",
    "alpha",
    "volatility",
    "distribution",
    "regime",
    "tail",
    "drawdown",
    "liquidity",
    "reinforcement",
    "conformal",
    "evalues",
    "jackknife_plus",
    "crc",
    "weighted_conformal",
    "interval_risk",
    "quantile_bandit",
    "cv_plus",
    "cpcv",
    "localized_conformal",
    "conformal_rank",
    "online_crc",
    "portfolio_conformal",
    "northset",
)


FORBIDDEN_RESEARCH_METRIC_KEYS = frozenset(
    {
        "sharpe",
        "sortino",
        "calmar",
        "pnl",
        "nav",
    }
)


KUPIEC_MARKER_KEYS = frozenset({"kupiec_p", "kupiec_lr"})


TAIL_VAR_BATTERY_REQUIRED_WHEN_KUPIEC = (
    "christoffersen_cc_p",
    "christoffersen_cc_lr",
    "christoffersen_ind_p",
    "christoffersen_ind_lr",
)


REQUIRED_CHRISTOFFERSEN_CC_KEYS = frozenset(
    {
        "christoffersen_cc_p",
        "christoffersen_cc_lr",
    }
)


PREFERRED_CHRISTOFFERSEN_IND_KEYS = frozenset(
    {
        "christoffersen_ind_p",
        "christoffersen_ind_lr",
    }
)


DM_CRPS_MARKER_KEY = "dm_crps_p"


DM_CRPS_SCALED_MARKER_KEY = "dm_crps_scaled_p"


DIST_CRPS_EPROCESS_REQUIRED_WHEN_DM = (
    "e_dm_crps_final",
    "e_dm_crps_reject",
    "e_dm_crps_n",
)


DIST_CRPS_SCALED_EPROCESS_REQUIRED_WHEN_DM = (
    "e_dm_crps_scaled_final",
    "e_dm_crps_scaled_reject",
    "e_dm_crps_scaled_n",
)


ES_MARKER_KEYS = frozenset({"es_95", "realized_es", "var_95"})


TAIL_ES_BATTERY_REQUIRED_WHEN_ES_MARKERS = (
    "acerbi_szekely_z1",
    "acerbi_szekely_z2",
    "fissler_ziegel_mean",
    "es_hit_count",
)


H4B_HYPOTHESIS_ID = "H4b_var_christoffersen_cc"


H4B_EXPECTED_FAMILY = "calibration"


H4_HYPOTHESIS_ID = "H4_var_kupiec"


H4_EXPECTED_FAMILY = "calibration"


H3_HYPOTHESIS_ID = "H3_vol_dm"


H3_EXPECTED_FAMILY = "discovery"


H1_HYPOTHESIS_ID = "H1_ranking_oracle"


H1_EXPECTED_FAMILY = "discovery"


H2_HYPOTHESIS_ID = "H2_decile_mono"


H2_EXPECTED_FAMILY = "discovery"


H99_HYPOTHESIS_ID = "H99_ranking_data_snooping"


H99_EXPECTED_FAMILY = "discovery"


H7_HYPOTHESIS_ID = "H7_aci_coverage"


H7_EXPECTED_FAMILY = "calibration"


H8_HYPOTHESIS_ID = "H8_mondrian_high_vol"


H8_EXPECTED_FAMILY = "calibration"


H11_HYPOTHESIS_ID = "H11_crc_var"


H11_EXPECTED_FAMILY = "calibration"


H12_HYPOTHESIS_ID = "H12_weighted_cqr"


H12_EXPECTED_FAMILY = "calibration"


H9_HYPOTHESIS_ID = "H9_eprocess_aci"


H9_EXPECTED_FAMILY = "calibration"


H10_HYPOTHESIS_ID = "H10_jackknife_coverage"


H10_EXPECTED_FAMILY = "bound"


H15_HYPOTHESIS_ID = "H15_cv_plus_floor"


H15_EXPECTED_FAMILY = "bound"


H16_H18_EXPECTED_FAMILY = "calibration"


H16_H18_PANEL_SPECS: tuple[tuple[str, str, str], ...] = (
    (
        "localized_conformal",
        "H16_localized_cqr",
        "hypothesis_h16_missing_despite_finite_kupiec_p",
    ),
    (
        "online_crc",
        "H17_online_crc",
        "hypothesis_h17_missing_despite_finite_kupiec_p",
    ),
    (
        "portfolio_conformal",
        "H18_portfolio_conformal",
        "hypothesis_h18_missing_despite_finite_kupiec_p",
    ),
)


H16_HYPOTHESIS_ID = "H16_localized_cqr"


H17_HYPOTHESIS_ID = "H17_online_crc"


H18_HYPOTHESIS_ID = "H18_portfolio_conformal"


H19_HYPOTHESIS_ID = "H19_conformal_rank"


H19_EXPECTED_FAMILY = "bound"


H20_HYPOTHESIS_ID = "H20_northset_ohlc"


H20_EXPECTED_FAMILY = "bound"


H21_HYPOTHESIS_ID = "H21_northset_book"


H21_EXPECTED_FAMILY = "bound"


H22_HYPOTHESIS_ID = "H22_northset_imbalance"


H22_EXPECTED_FAMILY = "discovery"


H23_HYPOTHESIS_ID = "H23_northset_session"


H24_HYPOTHESIS_ID = "H24_northset_volume"


H25_HYPOTHESIS_ID = "H25_northset_microprice"


H26_HYPOTHESIS_ID = "H26_northset_wick"


H27_HYPOTHESIS_ID = "H27_northset_ofi"


H28_HYPOTHESIS_ID = "H28_northset_gk"


H29_HYPOTHESIS_ID = "H29_northset_chain"


H30_HYPOTHESIS_ID = "H30_northset_clv"


H31_HYPOTHESIS_ID = "H31_northset_overnight"


H32_HYPOTHESIS_ID = "H32_northset_vpin"


H33_HYPOTHESIS_ID = "H33_northset_sweep_reject"


H34_HYPOTHESIS_ID = "H34_northset_sweep_follow"


H35_HYPOTHESIS_ID = "H35_northset_reject_event"


H36_HYPOTHESIS_ID = "H36_northset_follow_event"


H37_HYPOTHESIS_ID = "H37_northset_reject_placebo"


H38_HYPOTHESIS_ID = "H38_northset_follow_placebo"


H39_HYPOTHESIS_ID = "H39_northset_reject_cost"


H40_HYPOTHESIS_ID = "H40_northset_follow_cost"


H41_HYPOTHESIS_ID = "H41_northset_reject_stability"


H42_HYPOTHESIS_ID = "H42_northset_follow_stability"


H44_HYPOTHESIS_ID = "H44_northset_reject_control"


H45_HYPOTHESIS_ID = "H45_northset_follow_control"


H46_HYPOTHESIS_ID = "H46_northset_reject_liq_control"


H47_HYPOTHESIS_ID = "H47_northset_follow_liq_control"


H48_HYPOTHESIS_ID = "H48_northset_follow_oot"


H49_HYPOTHESIS_ID = "H49_northset_follow_name_cluster"


H50_HYPOTHESIS_ID = "H50_northset_follow_two_way_cluster"


H51_HYPOTHESIS_ID = "H51_northset_follow_overnight_gap"


NORTHSET_H23_H28_SPECS: tuple[tuple[str, str, str, str], ...] = (
    (
        "session_reconstructs_daily_rate",
        H23_HYPOTHESIS_ID,
        "bound",
        "hypothesis_h23_missing_despite_finite_session_reconstructs_daily_rate",
    ),
    (
        "session_volume_conservation_rate",
        H24_HYPOTHESIS_ID,
        "bound",
        "hypothesis_h24_missing_despite_finite_session_volume_conservation_rate",
    ),
    (
        "microprice_p_ic",
        H25_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h25_missing_despite_finite_microprice_p_ic",
    ),
    (
        "wick_skew_p_ic",
        H26_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h26_missing_despite_finite_wick_skew_p_ic",
    ),
    (
        "ofi_p_ic",
        H27_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h27_missing_despite_finite_ofi_p_ic",
    ),
    (
        "dm_gk_vs_park_p",
        H28_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h28_missing_despite_finite_dm_gk_vs_park_p",
    ),
    (
        "session_chain_rate",
        H29_HYPOTHESIS_ID,
        "bound",
        "hypothesis_h29_missing_despite_finite_session_chain_rate",
    ),
    (
        "clv_p_ic",
        H30_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h30_missing_despite_finite_clv_p_ic",
    ),
    (
        "dm_split_vs_park_p",
        H31_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h31_missing_despite_finite_dm_split_vs_park_p",
    ),
    (
        "vpin_p_ic",
        H32_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h32_missing_despite_finite_vpin_p_ic",
    ),
    (
        "sweep_reject_signed_p_ic",
        H33_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h33_missing_despite_finite_sweep_reject_signed_p_ic",
    ),
    (
        "sweep_follow_signed_p_ic",
        H34_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h34_missing_despite_finite_sweep_follow_signed_p_ic",
    ),
    (
        "sweep_reject_event_p",
        H35_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h35_missing_despite_finite_sweep_reject_event_p",
    ),
    (
        "sweep_follow_event_p",
        H36_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h36_missing_despite_finite_sweep_follow_event_p",
    ),
    (
        "sweep_reject_placebo_p",
        H37_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h37_missing_despite_finite_sweep_reject_placebo_p",
    ),
    (
        "sweep_follow_placebo_p",
        H38_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h38_missing_despite_finite_sweep_follow_placebo_p",
    ),
    (
        "sweep_reject_cost_adjusted_mean_bps",
        H39_HYPOTHESIS_ID,
        "bound",
        "hypothesis_h39_missing_despite_finite_sweep_reject_cost_adjusted_mean_bps",
    ),
    (
        "sweep_follow_cost_adjusted_mean_bps",
        H40_HYPOTHESIS_ID,
        "bound",
        "hypothesis_h40_missing_despite_finite_sweep_follow_cost_adjusted_mean_bps",
    ),
    (
        "sweep_reject_fold_positive_fraction",
        H41_HYPOTHESIS_ID,
        "bound",
        "hypothesis_h41_missing_despite_finite_sweep_reject_fold_positive_fraction",
    ),
    (
        "sweep_follow_fold_positive_fraction",
        H42_HYPOTHESIS_ID,
        "bound",
        "hypothesis_h42_missing_despite_finite_sweep_follow_fold_positive_fraction",
    ),
    (
        "sweep_reject_control_diff_p",
        H44_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h44_missing_despite_finite_sweep_reject_control_diff_p",
    ),
    (
        "sweep_follow_control_diff_p",
        H45_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h45_missing_despite_finite_sweep_follow_control_diff_p",
    ),
    (
        "sweep_reject_liq_control_diff_p",
        H46_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h46_missing_despite_finite_sweep_reject_liq_control_diff_p",
    ),
    (
        "sweep_follow_liq_control_diff_p",
        H47_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h47_missing_despite_finite_sweep_follow_liq_control_diff_p",
    ),
    (
        "sweep_follow_oot_holdout_mean_bps",
        H48_HYPOTHESIS_ID,
        "bound",
        "hypothesis_h48_missing_despite_finite_sweep_follow_oot_holdout_mean_bps",
    ),
    (
        "sweep_follow_name_cluster_p",
        H49_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h49_missing_despite_finite_sweep_follow_name_cluster_p",
    ),
    (
        "sweep_follow_two_way_cluster_p",
        H50_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h50_missing_despite_finite_sweep_follow_two_way_cluster_p",
    ),
    (
        "sweep_follow_overnight_gap_p",
        H51_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h51_missing_despite_finite_sweep_follow_overnight_gap_p",
    ),
)


H43_HYPOTHESIS_ID = "H43_northset_session_book_vpin"


H43_EXPECTED_FAMILY = "discovery"


COVERAGE_GUARANTEE_SCOPE_MARGINAL = "marginal_exchangeable"
