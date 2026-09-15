import math

import numpy as np

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


def test_two_proportion_detects_rate_gap() -> None:
    z, p = two_proportion_test(40, 80, 16, 80, alternative="greater")
    assert z > 0
    assert 0.0 < p < 0.05


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
    h13_other = next(
        h for h in _build_hypotheses(fam, _rankers()) if h.id == "H13_interval_caps"
    )
    assert "equal-weight" not in h13_other.statement.lower()
    assert "equal weight" not in h13_other.statement.lower()


def test_h10_floor_miss_is_not_a_fdr_discovery() -> None:
    fam = _families(jackknife_plus={"coverage": 0.70, "coverage_floor": 0.8})
    h10 = next(
        h for h in _build_hypotheses(fam, _rankers()) if h.id == "H10_jackknife_coverage"
    )
    assert h10.meets_floor is False
    assert h10.family == "bound"
    assert h10.reject_fdr is False
    assert not math.isfinite(h10.p_value)
