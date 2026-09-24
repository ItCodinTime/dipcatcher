# suite durations — 2026-09-24 (macOS arm64, `uv run pytest -q --durations=40`)

```
============================= slowest 40 durations =============================
60.22s call     tests/unit/test_session_receipt_keys_cli_echo.py::test_research_cli_echoes_every_session_receipt_key
26.49s call     tests/unit/test_caviar.py::test_as_and_ig_specs
26.02s call     tests/unit/test_mean_spread_bps_honesty.py::test_cli_echoes_mean_spread_bps
23.70s call     tests/unit/test_microprice_weight_balance_fuse.py::test_northset_cli_echoes_mean
23.41s call     tests/unit/test_half_spread_cli_echo.py::test_northset_cli_echoes_mean_half_spread
23.31s call     tests/unit/test_mean_session_spread_bps_mean.py::test_northset_cli_echoes_mean_session_spread_bps_mean
23.27s call     tests/unit/test_side_notional_northset.py::test_northset_cli_echoes_side_notional
23.10s call     tests/unit/test_mean_session_close_micro_bps.py::test_northset_cli_echoes_mean_session_close_micro_bps
23.01s call     tests/unit/test_session_receipt_keys_cli_echo.py::test_northset_cli_echoes_every_session_receipt_key
22.71s call     tests/unit/test_mean_session_close_spread_bps.py::test_northset_cli_echoes_mean_session_close_spread_bps
22.51s call     tests/unit/test_mean_session_close_mid.py::test_northset_cli_echoes_mean_session_close_mid
22.48s call     tests/unit/test_mean_session_close_depths.py::test_northset_cli_echoes_mean_session_close_depths
22.32s call     tests/unit/test_northset_cli_shape_rates.py::test_northset_cli_echoes_shape_rates
21.99s call     tests/unit/test_mean_session_close_imbalance.py::test_northset_cli_echoes_mean_session_close_imbalance
21.98s call     tests/unit/test_tob_size_share_northset.py::test_doctor_and_northset_cli_echo_tob
21.93s call     tests/unit/test_research_agent.py::test_research_recovers_synthetic_oracle
21.69s call     tests/unit/test_mean_session_ofi_abs_sum.py::test_dispatcher_includes_helper_and_cli_echoes
21.32s call     tests/unit/test_spread_alias_receipt_parity.py::test_northset_cli_echoes_half_means
21.21s call     tests/unit/test_mean_session_imbalance_std.py::test_northset_cli_echoes_mean_session_imbalance_std
21.19s call     tests/unit/test_northset_cli_shape_rates.py::test_northset_cli_floor_flags_echo
9.22s call     tests/unit/test_crps_closed_form.py::test_crps_empirical_gaussian_ensemble_near_closed_form
8.76s call     tests/unit/test_garch_ext.py::TestFIGARCH::test_fit_recovers_d
8.73s call     tests/unit/test_bench_dm_wired.py::test_bench_volatility_exposes_diebold_mariano
7.96s setup    tests/unit/test_bench_crps_closed.py::test_bench_distribution_closed_form_crps_finite
7.92s setup    tests/unit/test_bench_forbidden_metrics.py::test_bench_sample_forbidden_metrics_absent
7.10s call     tests/unit/test_arma.py::test_aic_detects_signal
6.64s setup    tests/unit/test_bench_distribution_eprocess.py::test_bench_distribution_exposes_eprocess_dm_crps
6.45s call     tests/unit/test_quantile_signals.py::test_sim_live_end_to_end_synthetic
6.07s call     tests/unit/test_causal_weights.py::test_early_and_final_weights_differ_when_alphas_evolve
5.42s call     tests/end_to_end/test_synthetic_pipeline.py::test_ingest_train_optimize_backtest
5.32s setup    tests/unit/test_bench_volatility_eprocess.py::test_bench_volatility_exposes_eprocess_dm
5.05s call     tests/unit/test_amihud_mean_ic_vs_amihud_abs_mean_ic_never_equate.py::test_synth_both_stamped_never_equate_clean
4.09s call     tests/unit/test_paper_long_stress.py::test_paper_200_step_resume_analytics_export
3.97s call     tests/unit/test_realized_garch_risk_gate.py::test_backtest_rgarch_overlay_ignores_ohlc_on_or_after_origin
3.85s call     tests/unit/test_deep_rl.py::test_policy_gradient_panel_is_finite_and_reproducible
3.80s call     tests/unit/test_feature_cols_spearman_pearson_and_lag1_n_soft_verify.py::test_completeness_requires_pearson_and_t_pack
3.57s call     tests/unit/test_jackknife_plus.py::test_jackknife_plus_synthetic_panel_meets_floor
3.49s setup    tests/unit/test_bench_tail_es_diagnostics.py::test_bench_tail_es_diagnostics_keys_and_honesty
3.18s call     tests/unit/test_robinhood_plus.py::test_compare_ridge_vs_robinhood_plus_receipt
3.07s call     tests/unit/test_hedge_lab.py::test_run_hedge_lab_synthetic_smoke
```

Top ~19 are cli_echo subprocess tests (~20-60s each, ≈7min total): each spawns a full CLI run. Nothing pathological; batching or a lighter entry point would cut suite wall time ~25%. Long tail <10s/test is healthy.
