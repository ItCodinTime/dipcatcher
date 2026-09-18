import math

import numpy as np
import pytest

from quant_fund.metrics.inference import mean_difference_t, two_proportion_test
from quant_fund.portfolio.interval_risk import equal_weight_per_date
from quant_fund.research.agent import _build_hypotheses


def _families(**overrides: object) -> dict:
    base: dict = {
        "volatility": {"dm_p": 0.02, "dm_stat": 2.1, "dm_preferred": "ewma"},
        "tail": {"kupiec_p": 0.40, "kupiec_lr": 0.7},
        "drawdown": {
            "brier": 0.18,
            "brier_base_rate": 0.22,
            "n_dates": 40,
        },
        "reinforcement": {
            "mean_advantage_vs_ridge": 0.04,
            "n_dates": 50,
        },
        "conformal": {
            "aci": {"kupiec_p": 0.30, "kupiec_lr": 1.0},
            "mondrian_aci": {"high_x_kupiec_p": 0.45, "high_x_kupiec_lr": 0.8},
        },
        "evalues": {"e_sup": 2.0},
        "jackknife_plus": {"coverage": 0.85, "coverage_floor": 0.8},
        "crc": {"kupiec_p": 0.55, "kupiec_lr": 0.4},
        "weighted_conformal": {"kupiec_p": 0.60, "kupiec_lr": 0.3},
        "interval_risk": {
            "bind_wide": 0.55,
            "bind_tight": 0.25,
            "weight_rule": "equal_weight_per_date",
            "bind_gap_by_date": [0.2, 0.3, 0.25, 0.4, 0.15, 0.35],
            "n_wide": 80,
            "n_tight": 80,
            "n_bind_wide": 44,
            "n_bind_tight": 20,
            "n_dates": 6,
        },
        "quantile_bandit": {
            "mean_advantage_vs_ridge": 0.03,
            "n_dates": 50,
        },
    }
    base.update(overrides)
    return base


def _rankers() -> list[dict]:
    return [
        {
            "name": "oracle_raw",
            "t_ic": 4.0,
            "p_ic": 1e-4,
            "ls_t": 3.0,
            "ls_p": 0.003,
        }
    ]


def test_mean_difference_t_is_not_dummy_01() -> None:
    t, p = mean_difference_t(0.12, 40)
    assert math.isfinite(t) and math.isfinite(p)
    assert p not in (0.0, 1.0)
    t0, p0 = mean_difference_t(0.0, 40)
    assert t0 == 0.0
    assert p0 == 1.0  # legitimate t=0, not dummy encoding of a sign


@pytest.mark.parametrize("sd", [0.0, -1.0, float("nan"), float("inf")])
def test_mean_difference_t_rejects_explicit_invalid_scale(sd: float) -> None:
    t, p = mean_difference_t(0.12, 40, sd=sd)
    assert math.isnan(t) and math.isnan(p)


def test_mean_difference_t_rejects_unknown_alternative() -> None:
    with pytest.raises(ValueError, match="alternative"):
        mean_difference_t(0.12, 40, alternative="one-sided")


def test_two_proportion_detects_rate_gap() -> None:
    z, p = two_proportion_test(40, 80, 16, 80, alternative="greater")
    assert z > 0
    assert 0.0 < p < 0.05


def test_two_proportion_rejects_invalid_counts_and_alternative() -> None:
    t, p = two_proportion_test(81, 80, 16, 80)
    assert math.isnan(t) and math.isnan(p)
    with pytest.raises(ValueError, match="alternative"):
        two_proportion_test(40, 80, 16, 80, alternative="one-sided")


def test_equal_weight_per_date_is_one_over_n_not_median_cap() -> None:
    dates = ["d1", "d1", "d2", "d2", "d2"]
    w = equal_weight_per_date(dates)
    assert np.allclose(w[:2], 0.5)
    assert np.allclose(w[2:], 1.0 / 3.0)
    assert not np.allclose(w, np.median(w))


def test_dummy_p0_not_used_for_h5_h6_h10_h13_h14() -> None:
    hyps = _build_hypotheses(_families(), _rankers())
    by = {h.id: h for h in hyps}
    for hid in (
        "H5_drawdown_brier",
        "H6_linucb_vs_uniform",
        "H13_interval_caps",
        "H14_quantile_thompson",
    ):
        h = by[hid]
        assert math.isfinite(h.p_value)
        assert h.p_value not in (0.0, 1.0)
        assert "H5-style" not in h.test
        assert h.test != "Brier score comparison"
    h10 = by["H10_jackknife_coverage"]
    assert h10.family == "bound"
    assert h10.meets_floor is True
    assert h10.reject_fdr is False
    assert h10.reject_raw is False
    assert not math.isfinite(h10.p_value)


def test_h5_uses_diebold_mariano_when_date_brier_series_present() -> None:
    rng = np.random.default_rng(0)
    clf = 0.10 + 0.02 * rng.normal(size=30)
    base = 0.20 + 0.02 * rng.normal(size=30)
    fam = _families(
        drawdown={
            "brier": float(np.mean(clf)),
            "brier_base_rate": float(np.mean(base)),
            "brier_by_date": clf.tolist(),
            "brier_base_by_date": base.tolist(),
        }
    )
    h5 = next(h for h in _build_hypotheses(fam, _rankers()) if h.id == "H5_drawdown_brier")
    assert "Diebold" in h5.test
    assert 0.0 < h5.p_value < 1.0


def test_h6_h14_use_hac_when_reward_series_present() -> None:
    rng = np.random.default_rng(1)
    pol = 0.05 + 0.01 * rng.normal(size=40)
    rid = 0.01 + 0.01 * rng.normal(size=40)
    fam = _families(
        reinforcement={
            "mean_advantage_vs_ridge": float(np.mean(pol - rid)),
            "n_dates": 40,
            "policy_reward": pol.tolist(),
            "ridge_reward": rid.tolist(),
        },
        quantile_bandit={
            "mean_advantage_vs_ridge": float(np.mean(pol - rid)),
            "n_dates": 40,
            "policy_reward": pol.tolist(),
            "ridge_reward": rid.tolist(),
        },
    )
    by = {h.id: h for h in _build_hypotheses(fam, _rankers())}
    assert "HAC" in by["H6_linucb_vs_uniform"].test
    assert "HAC" in by["H14_quantile_thompson"].test
    assert by["H6_linucb_vs_uniform"].p_value not in (0.0, 1.0)
    assert by["H14_quantile_thompson"].p_value not in (0.0, 1.0)


def test_calibration_and_discovery_fdr_lists_are_disjoint() -> None:
    hyps = _build_hypotheses(_families(), _rankers())
    cal = {h.id for h in hyps if h.family == "calibration"}
    disc = {h.id for h in hyps if h.family == "discovery"}
    bound = {h.id for h in hyps if h.family == "bound"}
    assert cal.isdisjoint(disc)
    assert cal.isdisjoint(bound)
    assert disc.isdisjoint(bound)
    assert cal == {
        "H4_var_kupiec",
        "H7_aci_coverage",
        "H8_mondrian_high_vol",
        "H9_eprocess_aci",
        "H11_crc_var",
        "H12_weighted_cqr",
    }
    assert {
        "H1_ranking_oracle",
        "H2_decile_mono",
        "H3_vol_dm",
        "H5_drawdown_brier",
        "H6_linucb_vs_uniform",
        "H13_interval_caps",
        "H14_quantile_thompson",
    } <= disc
    assert bound == {"H10_jackknife_coverage"}
    cal_fdr = {h.id for h in hyps if h.family == "calibration" and h.reject_fdr}
    disc_fdr = {h.id for h in hyps if h.family == "discovery" and h.reject_fdr}
    assert cal_fdr.isdisjoint(disc_fdr)
    assert all(not h.reject_fdr for h in hyps if h.family == "bound")


def test_h13_equal_weight_claim_matches_weight_rule() -> None:
    hyps = _build_hypotheses(_families(), _rankers())
    h13 = next(h for h in hyps if h.id == "H13_interval_caps")
    assert "equal-weight" in h13.statement.lower()
    fam = _families(
        interval_risk={
            "bind_wide": 0.6,
            "bind_tight": 0.2,
            "weight_rule": "median_cap",
            "n_wide": 40,
            "n_tight": 40,
            "n_bind_wide": 24,
            "n_bind_tight": 8,
        }
    )
    h13_other = next(h for h in _build_hypotheses(fam, _rankers()) if h.id == "H13_interval_caps")
    assert "equal-weight" not in h13_other.statement.lower()
    assert "equal weight" not in h13_other.statement.lower()


def test_h10_floor_miss_is_not_a_fdr_discovery() -> None:
    fam = _families(jackknife_plus={"coverage": 0.70, "coverage_floor": 0.8})
    h10 = next(h for h in _build_hypotheses(fam, _rankers()) if h.id == "H10_jackknife_coverage")
    assert h10.meets_floor is False
    assert h10.family == "bound"
    assert h10.reject_fdr is False
    assert not math.isfinite(h10.p_value)


def test_bh_fdr_is_family_split_not_pooled() -> None:
    hyps = _build_hypotheses(_families(), _rankers())
    cal = [h for h in hyps if h.family == "calibration"]
    disc = [h for h in hyps if h.family == "discovery"]
    bound = [h for h in hyps if h.family == "bound"]
    assert cal and disc
    # Bound checks are never FDR-adjusted
    assert all(h.reject_fdr is False for h in bound)
    # At least one discovery hyp should have reject_fdr populated (True or False, but field set)
    assert all(isinstance(h.reject_fdr, bool) for h in disc + cal)


def test_h16_h18_panel_kupiec_not_descriptive_fixture() -> None:
    """Panel rows get Kupiec calibration hyps; fixture dgp is excluded."""
    fam = _families(
        localized_conformal={
            "dgp": "panel",
            "coverage": 0.91,
            "alpha": 0.10,
            "kupiec_lr": 0.5,
            "kupiec_p": 0.48,
        },
        online_crc={
            "dgp": "panel",
            "mean_risk": 0.04,
            "nominal": 0.05,
            "kupiec_lr": 0.2,
            "kupiec_p": 0.65,
        },
        portfolio_conformal={
            "dgp": "panel",
            "coverage": 0.90,
            "alpha": 0.10,
            "kupiec_lr": 0.1,
            "kupiec_p": 0.75,
        },
        conformal_rank={"dgp": "panel", "fdr": 0.12, "alpha": 0.20},
    )
    by = {h.id: h for h in _build_hypotheses(fam, _rankers())}
    for hid in ("H16_localized_cqr", "H17_online_crc", "H18_portfolio_conformal"):
        h = by[hid]
        assert h.family == "calibration"
        assert "Kupiec" in h.test
        assert math.isfinite(h.p_value)
        assert "Descriptive" not in h.decision
    h19 = by["H19_conformal_rank"]
    assert h19.family == "bound"
    assert h19.meets_floor is True
    assert not math.isfinite(h19.p_value)
    assert "at or below" in h19.decision.lower()
    # fixture dgp must not enter the H-table
    fam_fix = _families(
        localized_conformal={
            "dgp": "fixture",
            "coverage": 0.9,
            "alpha": 0.1,
            "kupiec_p": 0.5,
            "kupiec_lr": 1.0,
        },
        conformal_rank={"dgp": "fixture", "fdr": 0.05, "alpha": 0.2},
    )
    ids = {h.id for h in _build_hypotheses(fam_fix, _rankers())}
    assert "H16_localized_cqr" not in ids
    assert "H19_conformal_rank" not in ids


def test_h16_h18_nan_kupiec_p_skips_not_success() -> None:
    """NaN kupiec_p must not mint H16–H18 (same as empty / missing)."""
    fam = _families(
        localized_conformal={
            "dgp": "panel",
            "kupiec_p": float("nan"),
            "kupiec_lr": float("nan"),
        },
        online_crc={"dgp": "panel", "kupiec_p": float("nan"), "kupiec_lr": float("nan")},
        portfolio_conformal={
            "dgp": "panel",
            "kupiec_p": float("nan"),
            "kupiec_lr": float("nan"),
        },
    )
    ids = {h.id for h in _build_hypotheses(fam, _rankers())}
    for hid in ("H16_localized_cqr", "H17_online_crc", "H18_portfolio_conformal"):
        assert hid not in ids


def test_h16_h18_empty_blob_skips() -> None:
    """Empty panel family blobs skip H16–H18 (regression)."""
    fam = _families(
        localized_conformal={},
        online_crc={},
        portfolio_conformal={},
    )
    ids = {h.id for h in _build_hypotheses(fam, _rankers())}
    for hid in ("H16_localized_cqr", "H17_online_crc", "H18_portfolio_conformal"):
        assert hid not in ids


def test_h16_finite_kupiec_p_still_present() -> None:
    """Finite kupiec_p on panel still mints H16 (existing behavior)."""
    fam = _families(
        localized_conformal={
            "dgp": "panel",
            "coverage": 0.91,
            "alpha": 0.10,
            "kupiec_lr": 0.5,
            "kupiec_p": 0.48,
        },
    )
    by = {h.id: h for h in _build_hypotheses(fam, _rankers())}
    h = by["H16_localized_cqr"]
    assert h.family == "calibration"
    assert math.isfinite(h.p_value)
    assert "consistent" in h.decision.lower()


def test_other_kupiec_hyps_skip_nan_p() -> None:
    """H4/H7/H8/H11/H12 mirror finite-p gate (no success mint on NaN)."""
    fam = _families(
        tail={"kupiec_p": float("nan"), "kupiec_lr": float("nan")},
        conformal={
            "aci": {"kupiec_p": float("nan"), "kupiec_lr": float("nan")},
            "mondrian_aci": {
                "high_x_kupiec_p": float("nan"),
                "high_x_kupiec_lr": float("nan"),
            },
        },
        crc={"kupiec_p": float("nan"), "kupiec_lr": float("nan")},
        weighted_conformal={"kupiec_p": float("nan"), "kupiec_lr": float("nan")},
    )
    ids = {h.id for h in _build_hypotheses(fam, _rankers())}
    for hid in (
        "H4_var_kupiec",
        "H7_aci_coverage",
        "H8_mondrian_high_vol",
        "H11_crc_var",
        "H12_weighted_cqr",
    ):
        assert hid not in ids


def test_h19_finite_fdr_not_unavailable() -> None:
    fam = _families(conformal_rank={"dgp": "panel", "fdr": 0.35, "alpha": 0.20})
    h19 = next(h for h in _build_hypotheses(fam, _rankers()) if h.id == "H19_conformal_rank")
    assert "unavailable" not in h19.decision.lower()
    assert "exceeds" in h19.decision.lower()
    assert h19.meets_floor is False


def test_h9_nan_e_sup_skips_not_success() -> None:
    """NaN e_sup must not mint H9 — max(nan, 1.0)→1 would fake p=1 calibration success."""
    fam = _families(evalues={"e_sup": float("nan")})
    ids = {h.id for h in _build_hypotheses(fam, _rankers())}
    assert "H9_eprocess_aci" not in ids
    fam_inf = _families(evalues={"e_sup": float("inf")})
    assert "H9_eprocess_aci" not in {h.id for h in _build_hypotheses(fam_inf, _rankers())}


def test_h9_finite_e_sup_still_present() -> None:
    fam = _families(evalues={"e_sup": 2.0})
    by = {h.id: h for h in _build_hypotheses(fam, _rankers())}
    h = by["H9_eprocess_aci"]
    assert h.family == "calibration"
    assert math.isfinite(h.p_value)
    assert "consistent" in h.decision.lower()


def test_h10_h15_nan_coverage_skips_not_below_floor() -> None:
    """NaN coverage must not mint bound rows claiming below-floor failure."""
    fam = _families(
        jackknife_plus={"coverage": float("nan"), "coverage_floor": 0.8},
        cv_plus={
            "coverage": float("nan"),
            "coverage_floor": 0.8,
            "coverage_identity": "mean",
        },
    )
    ids = {h.id for h in _build_hypotheses(fam, _rankers())}
    assert "H10_jackknife_coverage" not in ids
    assert "H15_cv_plus_floor" not in ids
    fam2 = _families(
        cv_plus={
            "coverage": 0.9,
            "coverage_floor": float("nan"),
            "coverage_identity": "mean",
        },
    )
    assert "H15_cv_plus_floor" not in {h.id for h in _build_hypotheses(fam2, _rankers())}


def test_h19_nan_fdr_skips_not_unavailable() -> None:
    """NaN FDR must skip H19 (no 'unavailable' bound mint); finite still present."""
    fam = _families(conformal_rank={"dgp": "panel", "fdr": float("nan"), "alpha": 0.20})
    ids = {h.id for h in _build_hypotheses(fam, _rankers())}
    assert "H19_conformal_rank" not in ids
    fam_ok = _families(conformal_rank={"dgp": "panel", "fdr": 0.12, "alpha": 0.20})
    h19 = next(h for h in _build_hypotheses(fam_ok, _rankers()) if h.id == "H19_conformal_rank")
    assert h19.meets_floor is True
    assert "unavailable" not in h19.decision.lower()


def test_h3_nan_dm_p_skips() -> None:
    """NaN vol dm_p must skip H3 (not mint unavailable discovery row)."""
    fam = _families(volatility={"dm_p": float("nan"), "dm_stat": 2.1, "dm_preferred": "ewma"})
    ids = {h.id for h in _build_hypotheses(fam, _rankers())}
    assert "H3_vol_dm" not in ids
    fam_inf = _families(volatility={"dm_p": float("inf"), "dm_stat": 2.1, "dm_preferred": "ewma"})
    assert "H3_vol_dm" not in {h.id for h in _build_hypotheses(fam_inf, _rankers())}


def test_h3_finite_dm_p_still_present() -> None:
    fam = _families(volatility={"dm_p": 0.02, "dm_stat": 2.1, "dm_preferred": "ewma"})
    by = {h.id: h for h in _build_hypotheses(fam, _rankers())}
    h = by["H3_vol_dm"]
    assert h.family == "discovery"
    assert math.isfinite(h.p_value)
    assert h.p_value == pytest.approx(0.02)


def test_pairwise_nan_p_value_skips_rank_dm() -> None:
    """NaN pairwise DM p_value must not mint H_rank_dm_* (is not None was insufficient)."""
    rankers = _rankers() + [
        {
            "name": "_pairwise_dm_summary",
            "pairwise_dm_all": [
                {
                    "a": "ridge",
                    "b": "oracle_raw",
                    "statistic": 1.2,
                    "p_value": float("nan"),
                    "preferred": "ridge",
                }
            ],
        }
    ]
    ids = {h.id for h in _build_hypotheses(_families(), rankers)}
    assert not any(i.startswith("H_rank_dm_") for i in ids)


def test_pairwise_finite_p_value_mints_rank_dm() -> None:
    rankers = _rankers() + [
        {
            "name": "_pairwise_dm_summary",
            "pairwise_dm_all": [
                {
                    "a": "ridge",
                    "b": "oracle_raw",
                    "statistic": 1.2,
                    "p_value": 0.03,
                    "preferred": "ridge",
                }
            ],
        }
    ]
    by = {h.id: h for h in _build_hypotheses(_families(), rankers)}
    h = by["H_rank_dm_ridge_vs_oracle_raw"]
    assert h.family == "discovery"
    assert math.isfinite(h.p_value)
    assert h.p_value == pytest.approx(0.03)


def test_oracle_nan_p_ic_skips_h1_independent_h2() -> None:
    """Non-finite p_ic skips H1 only; finite ls_p still mints H2 (and vice versa)."""
    rankers_nan_ic = [
        {
            "name": "oracle_raw",
            "t_ic": 4.0,
            "p_ic": float("nan"),
            "ls_t": 3.0,
            "ls_p": 0.003,
        }
    ]
    by = {h.id: h for h in _build_hypotheses(_families(), rankers_nan_ic)}
    assert "H1_ranking_oracle" not in by
    assert "H2_decile_mono" in by
    assert math.isfinite(by["H2_decile_mono"].p_value)

    rankers_nan_ls = [
        {
            "name": "oracle_raw",
            "t_ic": 4.0,
            "p_ic": 1e-4,
            "ls_t": 3.0,
            "ls_p": float("nan"),
        }
    ]
    by2 = {h.id: h for h in _build_hypotheses(_families(), rankers_nan_ls)}
    assert "H1_ranking_oracle" in by2
    assert math.isfinite(by2["H1_ranking_oracle"].p_value)
    assert "H2_decile_mono" not in by2

    rankers_ok = _rankers()
    by3 = {h.id: h for h in _build_hypotheses(_families(), rankers_ok)}
    assert "H1_ranking_oracle" in by3 and "H2_decile_mono" in by3


def test_contrast_nan_p_skips_h5_h6_h13_h14() -> None:
    """Non-finite contrast p must skip H5/H6/H13/H14 (no unavailable discovery mint)."""
    # n_dates < 3 → mean_difference_t / contrast path returns NaN p
    fam = _families(
        drawdown={"brier": 0.18, "brier_base_rate": 0.22, "n_dates": 1},
        reinforcement={"mean_advantage_vs_ridge": 0.04, "n_dates": 1},
        interval_risk={"bind_wide": 0.55, "bind_tight": 0.25, "n_dates": 1},
        quantile_bandit={"mean_advantage_vs_ridge": 0.03, "n_dates": 1},
    )
    ids = {h.id for h in _build_hypotheses(fam, _rankers())}
    assert "H5_drawdown_brier" not in ids
    assert "H6_linucb_vs_uniform" not in ids
    assert "H13_interval_caps" not in ids
    assert "H14_quantile_thompson" not in ids


def test_h5_nan_brier_skips() -> None:
    """NaN brier must not enter the H5 gate (finite-input contract)."""
    fam = _families(
        drawdown={"brier": float("nan"), "brier_base_rate": 0.22, "n_dates": 40},
    )
    ids = {h.id for h in _build_hypotheses(fam, _rankers())}
    assert "H5_drawdown_brier" not in ids


def test_h5_finite_scalars_no_series_nan_p_skips() -> None:
    """Finite brier pair but no date series / n → non-finite p → omit H5."""
    fam = _families(drawdown={"brier": 0.18, "brier_base_rate": 0.22})
    ids = {h.id for h in _build_hypotheses(fam, _rankers())}
    assert "H5_drawdown_brier" not in ids


def test_h5_finite_path_still_mints() -> None:
    """Finite brier + n_dates (or series) still mints H5 when p is finite."""
    by = {h.id: h for h in _build_hypotheses(_families(), _rankers())}
    assert "H5_drawdown_brier" in by
    assert math.isfinite(by["H5_drawdown_brier"].p_value)
    assert "unavailable" not in by["H5_drawdown_brier"].decision.lower()


def test_h6_nan_mean_advantage_skips() -> None:
    """NaN mean_advantage_vs_ridge must not enter the H6 gate."""
    fam = _families(reinforcement={"mean_advantage_vs_ridge": float("nan"), "n_dates": 50})
    ids = {h.id for h in _build_hypotheses(fam, _rankers())}
    assert "H6_linucb_vs_uniform" not in ids


def test_h6_finite_advantage_without_series_skips() -> None:
    """Finite advantage but no reward series / n → non-finite p → omit H6."""
    fam = _families(reinforcement={"mean_advantage_vs_ridge": 0.04})
    ids = {h.id for h in _build_hypotheses(fam, _rankers())}
    assert "H6_linucb_vs_uniform" not in ids


def test_h6_finite_with_series_still_present() -> None:
    """Finite advantage + reward series still mints H6 (HAC path)."""
    rng = np.random.default_rng(1)
    pol = 0.05 + 0.01 * rng.normal(size=40)
    rid = 0.01 + 0.01 * rng.normal(size=40)
    fam = _families(
        reinforcement={
            "mean_advantage_vs_ridge": float(np.mean(pol - rid)),
            "policy_reward": pol.tolist(),
            "ridge_reward": rid.tolist(),
        },
    )
    by = {h.id: h for h in _build_hypotheses(fam, _rankers())}
    assert "H6_linucb_vs_uniform" in by
    assert math.isfinite(by["H6_linucb_vs_uniform"].p_value)
    assert "HAC" in by["H6_linucb_vs_uniform"].test
    assert "unavailable" not in by["H6_linucb_vs_uniform"].decision.lower()


def test_h13_h14_nan_inputs_skip() -> None:
    """Sibling contrast hyps: non-finite input scalars skip before contrast."""
    fam = _families(
        interval_risk={"bind_wide": float("nan"), "bind_tight": 0.25, "n_dates": 40},
        quantile_bandit={"mean_advantage_vs_ridge": float("inf"), "n_dates": 50},
    )
    ids = {h.id for h in _build_hypotheses(fam, _rankers())}
    assert "H13_interval_caps" not in ids
    assert "H14_quantile_thompson" not in ids


def test_contrast_finite_p_still_mints_h5_h6_h13_h14() -> None:
    by = {h.id: h for h in _build_hypotheses(_families(), _rankers())}
    for hid in (
        "H5_drawdown_brier",
        "H6_linucb_vs_uniform",
        "H13_interval_caps",
        "H14_quantile_thompson",
    ):
        assert hid in by
        assert math.isfinite(by[hid].p_value)
        assert "unavailable" not in by[hid].decision.lower()


def test_h4b_nan_christoffersen_cc_p_skips() -> None:
    """NaN/inf christoffersen_cc_p must not mint H4b (same Kupiec skip contract)."""
    fam = _families(
        tail={
            "kupiec_p": 0.40,
            "kupiec_lr": 0.7,
            "christoffersen_cc_p": float("nan"),
            "christoffersen_cc_lr": float("nan"),
        }
    )
    ids = {h.id for h in _build_hypotheses(fam, _rankers())}
    assert "H4_var_kupiec" in ids  # Kupiec still mints independently
    assert "H4b_var_christoffersen_cc" not in ids
    fam_inf = _families(
        tail={
            "kupiec_p": 0.40,
            "kupiec_lr": 0.7,
            "christoffersen_cc_p": float("inf"),
            "christoffersen_cc_lr": 1.0,
        }
    )
    assert "H4b_var_christoffersen_cc" not in {h.id for h in _build_hypotheses(fam_inf, _rankers())}
    # Missing key also skips (default _families has no cc_p)
    assert "H4b_var_christoffersen_cc" not in {
        h.id for h in _build_hypotheses(_families(), _rankers())
    }


def test_h4b_finite_christoffersen_cc_p_mints_calibration() -> None:
    """Finite christoffersen_cc_p mints H4b in calibration family (CC only, not ind)."""
    fam = _families(
        tail={
            "kupiec_p": 0.40,
            "kupiec_lr": 0.7,
            "christoffersen_cc_p": 0.35,
            "christoffersen_cc_lr": 1.2,
            "christoffersen_ind_p": 0.55,
            "christoffersen_ind_lr": 0.4,
        }
    )
    by = {h.id: h for h in _build_hypotheses(fam, _rankers())}
    assert "H4b_var_christoffersen_cc" in by
    h = by["H4b_var_christoffersen_cc"]
    assert h.family == "calibration"
    assert math.isclose(h.p_value, 0.35)
    assert math.isclose(h.statistic, 1.2)
    assert "Christoffersen CC" in h.test
    assert "consistent" in h.decision.lower() or "rejected" in h.decision.lower()
    # Prefer CC only — no separate ind hyp (avoid BH double-count with CC)
    assert not any("christoffersen_ind" in hid or hid.endswith("_ind") for hid in by)
    assert "H_var_christoffersen_ind" not in by
    assert "H4b_var_christoffersen_ind" not in by
