"""Institutional evidence controls for Northset liquidity sweeps."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from quant_fund.config.models import AppConfig, NorthsetConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.metrics.inference import mean_tstat
from quant_fund.northset.sweep_research import (
    PRIMARY_EXECUTABLE_TEST,
    _embed_on_calendar,
    _event_daily_frame,
    _panel_calendar,
    estimated_round_trip_cost_bps,
    event_adv_participation,
    event_coverage_stats,
    lead_lag_diagnostics,
    liquidity_matched_control_difference,
    matched_control_difference,
    name_clustered_event_mean,
    out_of_time_holdout,
    overnight_gap_event_mean,
    parameter_sensitivity_grid,
    sweep_evidence_battery,
    sweep_forward_frame,
    two_way_clustered_event_mean,
    within_date_permutation_test,
)
from quant_fund.northset.sweeps import liquidity_sweep_frame
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent


def _bars(n_assets: int = 10, n_days: int = 70, seed: int = 41):
    return SyntheticMarketProvider(n_assets=n_assets, n_days=n_days, seed=seed).get_bars()


def _config() -> AppConfig:
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.min_names = 3
    cfg.northset.sweep_lookback = 10
    cfg.northset.sweep_horizons = [1, 3, 5]
    cfg.northset.sweep_vol_lookback = 10
    cfg.northset.sweep_n_boot = 50
    cfg.northset.sweep_n_permutations = 50
    cfg.northset.sweep_n_folds = 3
    cfg.northset.sweep_sensitivity_lookbacks = [5, 10]
    return cfg


def test_forward_frame_uses_next_open_and_market_control() -> None:
    sweep = liquidity_sweep_frame(_bars(), lookback=10)
    frame = sweep_forward_frame(sweep, horizons=(1, 3), vol_lookback=10)
    assert "sweep_exec_ret_1" in frame.columns
    assert "sweep_excess_ret_3" in frame.columns
    assert "sweep_overnight_ret" in frame.columns
    assert "sweep_overnight_excess" in frame.columns
    # Excess returns are cross-sectionally demeaned on every eligible date.
    means = (
        frame.group_by("event_time")
        .agg(pl.col("sweep_excess_ret_1").mean())["sweep_excess_ret_1"]
        .drop_nulls()
        .to_numpy()
    )
    assert np.max(np.abs(means)) < 1e-12


def test_sweep_evidence_has_controls_fdr_stability_and_costs() -> None:
    cfg = _config()
    sweep = liquidity_sweep_frame(_bars(), lookback=cfg.northset.sweep_lookback)
    evidence = sweep_evidence_battery(sweep, cfg)
    assert evidence["timing_contract"] == "event_close_then_next_open"
    assert evidence["control_contract"] == "same_date_cross_sectional_mean"
    assert evidence["inference_index"] == "calendar_including_idle_zeros"
    assert evidence["horizons"] == [1, 3, 5]
    assert len(evidence["event_studies"]) == 6
    assert len(evidence["volatility_regimes"]) == 4
    assert set(evidence["permutation_placebos"]) == {
        "sweep_reject_signed",
        "sweep_follow_signed",
    }
    for row in evidence["event_studies"]:
        assert row["entry"] == "next_open"
        assert row["control"] == "same_date_cross_sectional_mean"
        assert row["inference_index"] == "calendar_including_idle_zeros"
        assert int(row["n_calendar_dates"]) >= int(row["n_dates"])
        assert row["n_folds"] in {0, 3}
        assert isinstance(row["reject_fdr"], bool)
        assert "cost_adjusted_mean_bps" in row
        assert "bootstrap_lo_bps" in row
    assert family_blob_forbidden_metrics_absent(evidence) is True


def test_permutation_placebo_is_reproducible_and_bounded() -> None:
    sweep = liquidity_sweep_frame(_bars(), lookback=10)
    frame = sweep_forward_frame(sweep, horizons=(1,), vol_lookback=10)
    kwargs = {
        "score": "sweep_follow_signed",
        "target": "sweep_excess_ret_1",
        "min_names": 3,
        "n_permutations": 50,
        "seed": 17,
    }
    a = within_date_permutation_test(frame, **kwargs)
    b = within_date_permutation_test(frame, **kwargs)
    assert a == b
    assert a["n_permutations"] == 50
    p_value = float(a["placebo_p_value"])
    assert math.isnan(p_value) or 0.0 < p_value <= 1.0


def test_round_trip_hurdle_increases_with_costs() -> None:
    cfg = _config()
    sweep = liquidity_sweep_frame(_bars(), lookback=10)
    low = estimated_round_trip_cost_bps(sweep, cfg)
    cfg.costs.half_spread_bps += 10.0
    high = estimated_round_trip_cost_bps(sweep, cfg)
    assert high == pytest.approx(low + 20.0)


def test_matched_controls_leads_sensitivity_and_coverage_present() -> None:
    cfg = _config()
    sweep = liquidity_sweep_frame(_bars(), lookback=cfg.northset.sweep_lookback)
    evidence = sweep_evidence_battery(sweep, cfg)
    controls = evidence["matched_controls"]
    assert set(controls) == {"sweep_reject_signed", "sweep_follow_signed"}
    for blob in controls.values():
        assert blob["control_design"] == "same_date_eligible_non_swept"
        assert blob["n_control_rows"] > 0
        assert isinstance(blob["sample_adequate"], bool)
    leads = evidence["lead_diagnostics"]
    for blob in leads.values():
        assert blob["diagnostic"] == "signal_vs_prior_bar_excess_return"
        assert blob["n_events"] > 0
    grid = evidence["parameter_sensitivity"]
    assert grid["lookbacks"] == [5, 10]
    assert grid["n_parameter_trials"] == 4
    assert grid["disclosure"] == "every_cell_counted_as_a_trial"
    assert set(grid["sign_stable_by_signal"]) == {
        "sweep_reject_signed",
        "sweep_follow_signed",
    }
    coverage = evidence["coverage"]
    assert coverage["n_unique_event_securities"] > 0
    assert 0.0 < coverage["top_security_event_share"] <= 1.0
    assert coverage["n_event_dates"] <= coverage["n_calendar_dates"]
    liq = evidence["liquidity_matched_controls"]
    for blob in liq.values():
        assert blob["control_design"] == "same_date_eligible_non_swept_lagged_dvol_quartile"
    clustered = evidence["name_clustered"]
    for blob in clustered.values():
        assert blob["cluster"] == "security_id"
        assert blob["se_kernel"] == "iid_across_names"
    two_way = evidence["two_way_clustered"]
    for blob in two_way.values():
        assert blob["cluster"] == "event_time_and_security_id"
        assert blob["se_kernel"] == "cameron_gelbach_miller"
        assert blob["inference_index"] == "event_rows_not_calendar_zeros"
    gaps = evidence["overnight_gaps"]
    for blob in gaps.values():
        assert blob["method"] == "event_close_to_next_open"
        assert blob["inference_index"] == "calendar_including_idle_zeros"
    oot = evidence["oot_holdouts"]["sweep_follow_signed"]
    assert oot["split"] == "last_fold_holdout"
    assert evidence["primary_test"] == PRIMARY_EXECUTABLE_TEST
    ledger = evidence["trial_ledger"]
    assert ledger["fdr_family"] == "northset_sweep_event_studies_only"
    assert ledger["n_counted_trials"] == (
        ledger["n_event_study_cells"]
        + ledger["n_regime_cells"]
        + ledger["n_placebos"]
        + ledger["n_matched_controls"]
        + ledger["n_liquidity_matched_controls"]
        + ledger["n_name_clustered"]
        + ledger["n_two_way_clustered"]
        + ledger["n_overnight_gaps"]
        + ledger["n_oot_holdouts"]
        + ledger["n_lead_diagnostics"]
        + ledger["n_parameter_trials"]
    )
    assert ledger["n_counted_trials"] > ledger["n_parameter_trials"]
    adv = evidence["adv_participation"]
    assert adv["n_events"] > 0
    assert math.isfinite(float(adv["median_participation"]))


def test_matched_control_difference_fails_closed_below_floor() -> None:
    sweep = liquidity_sweep_frame(_bars(n_assets=4, n_days=30), lookback=10)
    frame = sweep_forward_frame(sweep, horizons=(1,), vol_lookback=10)
    result = matched_control_difference(
        frame,
        signal="sweep_follow_signed",
        target="sweep_excess_ret_1",
        min_events=100_000,
        min_dates=5,
    )
    assert result["sample_adequate"] is False
    assert math.isnan(float(result["p_value"]))


def test_matched_control_difference_detects_planted_event_edge() -> None:
    n_days, n_names = 60, 8
    rows = []
    for d in range(n_days):
        for i in range(n_names):
            rows.append((f"S{i}", d, i))
    frame = pl.DataFrame(
        {
            "security_id": [r[0] for r in rows],
            "day": [r[1] for r in rows],
            "name_idx": [r[2] for r in rows],
        }
    ).with_columns(
        (pl.datetime(2020, 1, 1) + pl.duration(days=pl.col("day"))).alias("event_time"),
        pl.lit(True).alias("sweep_eligible"),
        # name 0 sweeps high (long direction) each day; the rest are controls
        (pl.col("name_idx") == 0).cast(pl.Float64).alias("sweep_high"),
        pl.lit(0.0).alias("sweep_low"),
        (pl.col("name_idx") == 0).cast(pl.Float64).alias("sweep_follow_signed"),
        # planted: event name earns +50 bps excess vs controls at 0
        pl.when(pl.col("name_idx") == 0).then(0.005).otherwise(0.0).alias("sweep_excess_ret_1"),
    )
    result = matched_control_difference(
        frame,
        signal="sweep_follow_signed",
        target="sweep_excess_ret_1",
        min_events=30,
        min_dates=20,
    )
    assert result["sample_adequate"] is True
    assert result["mean_diff_bps"] == pytest.approx(50.0)
    assert float(result["p_value"]) < 1e-6


def test_lead_diagnostic_is_finite_on_synthetic_panel() -> None:
    sweep = liquidity_sweep_frame(_bars(), lookback=10)
    frame = sweep_forward_frame(sweep, horizons=(1,), vol_lookback=10)
    lead = lead_lag_diagnostics(frame, signal="sweep_follow_signed")
    assert lead["n_dates"] > 0
    assert math.isfinite(float(lead["mean_lead_bps"]))


def test_parameter_sensitivity_counts_every_cell() -> None:
    cfg = _config()
    sweep = liquidity_sweep_frame(_bars(), lookback=cfg.northset.sweep_lookback)
    grid = parameter_sensitivity_grid(sweep, cfg)
    assert len(grid["cells"]) == grid["n_parameter_trials"] == 4
    for cell in grid["cells"]:
        assert cell["lookback"] in (5, 10)
        assert "p_value" in cell and "sample_adequate" in cell


def test_event_coverage_counts_double_sweeps() -> None:
    sweep = liquidity_sweep_frame(_bars(), lookback=10)
    coverage = event_coverage_stats(sweep)
    assert coverage["n_high_events"] + coverage["n_low_events"] > 0
    assert coverage["n_both_ambiguous"] >= 0


def test_event_inference_fails_closed_below_sample_floor() -> None:
    cfg = _config()
    cfg.northset.sweep_min_events = 100_000
    sweep = liquidity_sweep_frame(_bars(n_assets=4, n_days=30), lookback=10)
    evidence = sweep_evidence_battery(sweep, cfg)
    for row in evidence["event_studies"]:
        assert row["sample_adequate"] is False
        assert math.isnan(float(row["p_value"]))
        assert math.isnan(float(row["bootstrap_lo_bps"]))


def test_sweep_evidence_config_fails_closed() -> None:
    with pytest.raises(ValueError, match="sweep_horizons"):
        NorthsetConfig(sweep_horizons=[])
    with pytest.raises(ValueError, match="sweep_n_boot"):
        NorthsetConfig(sweep_n_boot=49)
    with pytest.raises(ValueError, match="sweep_n_permutations"):
        NorthsetConfig(sweep_n_permutations=49)
    with pytest.raises(ValueError, match="sweep_min_events"):
        NorthsetConfig(sweep_min_events=9)
    with pytest.raises(ValueError, match="sweep_min_dates"):
        NorthsetConfig(sweep_min_dates=4)
    with pytest.raises(ValueError, match="sweep_min_fold_positive_fraction"):
        NorthsetConfig(sweep_min_fold_positive_fraction=1.1)
    with pytest.raises(ValueError, match="sweep_sensitivity_lookbacks"):
        NorthsetConfig(sweep_sensitivity_lookbacks=[1])
    with pytest.raises(ValueError, match="sweep_sensitivity_lookbacks"):
        NorthsetConfig(sweep_sensitivity_lookbacks=[10, 10])


def test_liquidity_matched_controls_ignore_other_quartile_drift() -> None:
    n_days, n_names = 60, 8
    rows = [(d, i) for d in range(n_days) for i in range(n_names)]
    frame = pl.DataFrame(
        {
            "security_id": [f"S{i}" for _d, i in rows],
            "day": [d for d, _i in rows],
            "name_idx": [i for _d, i in rows],
        }
    ).with_columns(
        (pl.datetime(2020, 1, 1) + pl.duration(days=pl.col("day"))).alias("event_time"),
        pl.lit(True).alias("sweep_eligible"),
        (pl.col("name_idx") == 0).cast(pl.Float64).alias("sweep_high"),
        pl.lit(0.0).alias("sweep_low"),
        pl.lit(1.0).alias("close"),
        pl.when(pl.col("name_idx") <= 1).then(1000.0).otherwise(1.0).alias("volume"),
        pl.when(pl.col("name_idx") == 0).then(1.0).otherwise(0.0).alias("sweep_follow_signed"),
        pl.when(pl.col("name_idx") == 0)
        .then(0.005)
        .when(pl.col("name_idx") == 1)
        .then(0.0)
        .otherwise(0.01)
        .alias("sweep_excess_ret_1"),
    )
    unmatched = matched_control_difference(
        frame,
        signal="sweep_follow_signed",
        target="sweep_excess_ret_1",
        min_events=30,
        min_dates=20,
    )
    matched = liquidity_matched_control_difference(
        frame,
        signal="sweep_follow_signed",
        target="sweep_excess_ret_1",
        min_events=30,
        min_dates=20,
    )
    assert unmatched["sample_adequate"] is True
    assert matched["sample_adequate"] is True
    assert float(unmatched["mean_diff_bps"]) < 0.0
    assert matched["mean_diff_bps"] == pytest.approx(50.0, abs=1e-6)
    assert float(matched["p_value"]) < 1e-6


def test_name_clustered_t_detects_per_security_edge() -> None:
    n_days, n_names = 40, 8
    rows = [(d, i) for d in range(n_days) for i in range(n_names)]
    frame = pl.DataFrame(
        {
            "security_id": [f"S{i}" for _d, i in rows],
            "day": [d for d, _i in rows],
            "name_idx": [i for _d, i in rows],
        }
    ).with_columns(
        (pl.datetime(2020, 1, 1) + pl.duration(days=pl.col("day"))).alias("event_time"),
        pl.lit(1.0).alias("sweep_follow_signed"),
        (0.005 + 1e-6 * pl.col("name_idx")).alias("sweep_excess_ret_1"),
    )
    result = name_clustered_event_mean(
        frame,
        signal="sweep_follow_signed",
        target="sweep_excess_ret_1",
        min_names=5,
        min_events=30,
    )
    assert result["sample_adequate"] is True
    assert result["n_names"] == n_names
    assert result["mean_excess_bps"] == pytest.approx(50.0, abs=0.05)
    assert float(result["p_value"]) < 1e-6


def test_two_way_clustered_t_detects_event_edge() -> None:
    n_days, n_names = 40, 8
    rows = [(d, i) for d in range(n_days) for i in range(n_names)]
    frame = pl.DataFrame(
        {
            "security_id": [f"S{i}" for _d, i in rows],
            "day": [d for d, _i in rows],
            "name_idx": [i for _d, i in rows],
        }
    ).with_columns(
        (pl.datetime(2020, 1, 1) + pl.duration(days=pl.col("day"))).alias("event_time"),
        pl.lit(1.0).alias("sweep_follow_signed"),
        (0.005 + 1e-6 * pl.col("name_idx")).alias("sweep_excess_ret_1"),
    )
    result = two_way_clustered_event_mean(
        frame,
        signal="sweep_follow_signed",
        target="sweep_excess_ret_1",
        min_names=5,
        min_events=30,
        n_boot=39,
        seed=7,
    )
    assert result["sample_adequate"] is True
    assert result["n_names"] == n_names
    assert result["n_dates"] == n_days
    assert result["mean_excess_bps"] == pytest.approx(50.0, abs=0.05)
    assert float(result["p_value"]) < 1e-6
    assert result["se_kernel"] == "cameron_gelbach_miller"
    assert result["inference_index"] == "event_rows_not_calendar_zeros"
    wild_p = float(result["wild_bootstrap_p"])
    assert 0.0 < wild_p <= 1.0


def test_overnight_gap_detects_close_to_next_open_jump() -> None:
    n_days, n_names = 40, 6
    rows = [(d, i) for d in range(n_days) for i in range(n_names)]
    frame = pl.DataFrame(
        {
            "security_id": [f"S{i}" for _d, i in rows],
            "day": [d for d, _i in rows],
            "name_idx": [i for _d, i in rows],
        }
    ).with_columns(
        (pl.datetime(2020, 1, 1) + pl.duration(days=pl.col("day"))).alias("event_time"),
        pl.lit(1.0).alias("sweep_follow_signed"),
        pl.when(pl.col("name_idx") == 0).then(0.01).otherwise(0.0).alias("sweep_overnight_excess"),
    )
    result = overnight_gap_event_mean(
        frame,
        signal="sweep_follow_signed",
        min_events=20,
        min_dates=10,
    )
    assert result["sample_adequate"] is True
    assert result["method"] == "event_close_to_next_open"
    assert result["mean_gap_bps"] == pytest.approx(100.0 / n_names, abs=0.05)
    assert float(result["p_value"]) < 0.05


def test_out_of_time_holdout_same_sign_and_flip() -> None:
    n_days, n_names = 60, 6
    rows = [(d, i) for d in range(n_days) for i in range(n_names)]
    base = pl.DataFrame(
        {
            "security_id": [f"S{i}" for _d, i in rows],
            "day": [d for d, _i in rows],
        }
    ).with_columns(
        (pl.datetime(2020, 1, 1) + pl.duration(days=pl.col("day"))).alias("event_time"),
        pl.lit(1.0).alias("sweep_follow_signed"),
    )
    same = base.with_columns(pl.lit(0.005).alias("sweep_excess_ret_1"))
    stable = out_of_time_holdout(
        same,
        signal="sweep_follow_signed",
        target="sweep_excess_ret_1",
        n_folds=3,
        min_events=30,
        min_dates=10,
    )
    assert stable["sample_adequate"] is True
    assert float(stable["same_sign"]) == 1.0
    flipped = base.with_columns(
        pl.when(pl.col("day") < 40).then(0.005).otherwise(-0.005).alias("sweep_excess_ret_1")
    )
    unstable = out_of_time_holdout(
        flipped,
        signal="sweep_follow_signed",
        target="sweep_excess_ret_1",
        n_folds=3,
        min_events=30,
        min_dates=10,
    )
    assert unstable["sample_adequate"] is True
    assert float(unstable["same_sign"]) == 0.0


def test_event_adv_participation_finite_on_synthetic() -> None:
    cfg = _config()
    sweep = liquidity_sweep_frame(_bars(), lookback=10)
    frame = sweep_forward_frame(sweep, horizons=(1,), vol_lookback=10)
    blob = event_adv_participation(frame, cfg)
    assert blob["n_events"] > 0
    assert math.isfinite(float(blob["median_participation"]))
    assert float(blob["median_participation"]) >= 0.0
    assert 0.0 <= float(blob["share_above_ten_pct_adv"]) <= 1.0


def test_calendar_embed_is_more_conservative_than_compressed_event_dates() -> None:
    n_days, n_names = 80, 6
    rows = [(d, i) for d in range(n_days) for i in range(n_names)]
    frame = pl.DataFrame(
        {
            "security_id": [f"S{i}" for _d, i in rows],
            "day": [d for d, _i in rows],
            "name_idx": [i for _d, i in rows],
        }
    ).with_columns(
        (pl.datetime(2020, 1, 1) + pl.duration(days=pl.col("day"))).alias("event_time"),
        pl.when(pl.col("day") % 10 == 0).then(1.0).otherwise(0.0).alias("sweep_follow_signed"),
        (0.004 + 0.0003 * pl.col("name_idx") + 0.0002 * (pl.col("day") % 7)).alias(
            "sweep_excess_ret_1"
        ),
        pl.lit(5.0).alias("sweep_round_trip_cost_bps"),
    )
    daily, n_events = _event_daily_frame(
        frame, signal="sweep_follow_signed", target="sweep_excess_ret_1"
    )
    event = daily["_daily"].to_numpy().astype(float)
    cal, _costs = _embed_on_calendar(daily, _panel_calendar(frame))
    _mu_e, t_event, _p_e = mean_tstat(event, lags=1)
    _mu_c, t_cal, _p_c = mean_tstat(cal, lags=1)
    assert n_events > 0
    assert cal.size > event.size
    assert math.isfinite(t_event) and math.isfinite(t_cal)
    assert abs(t_cal) < abs(t_event)


def test_invalid_ohlc_is_quarantined_from_sweep_detection() -> None:
    bars = _bars(n_assets=4, n_days=40, seed=3)
    smashed = bars.with_row_index("_i").with_columns(
        pl.when(pl.col("_i") == 15)
        .then(pl.col("close") * 0.5)
        .otherwise(pl.col("high"))
        .alias("high")
    )
    victim = smashed.filter(pl.col("_i") == 15).select("security_id", "event_time")
    smashed = smashed.drop("_i")
    out = liquidity_sweep_frame(smashed, lookback=10)
    row = out.join(victim, on=["security_id", "event_time"], how="inner")
    assert row.height == 1
    assert bool(row["ohlc_ok"][0]) is False
    assert bool(row["sweep_eligible"][0]) is False
    assert float(row["sweep_high"].fill_null(0.0)[0]) == 0.0
    assert float(row["sweep_low"].fill_null(0.0)[0]) == 0.0
    coverage = event_coverage_stats(out)
    assert coverage["n_ohlc_quarantined"] >= 1
    # Later bars still form a PIT extreme; identity failure must not wipe the window.
    later = out.filter(pl.col("ohlc_ok") & pl.col("prior_high").is_not_null())
    assert later.height > 0
