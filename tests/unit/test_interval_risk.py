import numpy as np

from quant_fund.portfolio.interval_risk import (
    apply_interval_caps,
    bench_interval_caps,
    cap_from_interval,
    downside,
    equal_weight_per_date,
    interval_refs,
    interval_width,
)

REFS = {"max_weight": 0.02, "width_ref": 0.10, "downside_ref": 0.05}


def test_identical_intervals_identical_caps() -> None:
    lo = np.array([-0.03, -0.03, -0.03])
    hi = np.array([0.04, 0.04, 0.04])
    caps = cap_from_interval(lo, hi, **REFS)
    assert np.allclose(caps, caps[0])
    assert caps[0] == cap_from_interval(lo[:1], hi[:1], **REFS)[0]


def test_twice_the_width_strictly_smaller_cap() -> None:
    lo = np.array([-0.02, -0.02])
    hi = np.array([0.02, 0.06])  # widths 0.04 and 0.08; same downside
    caps = cap_from_interval(lo, hi, **REFS)
    assert caps[1] < caps[0]
    assert caps[1] > 0.0


def test_twice_width_stays_at_floor_when_already_zero() -> None:
    lo = np.array([np.nan])
    hi = np.array([np.nan])
    c0 = cap_from_interval(lo, hi, **REFS)
    c1 = cap_from_interval(lo, hi * 2.0, **REFS)
    assert c0[0] == 0.0
    assert c1[0] == 0.0


def test_nan_interval_fail_closed() -> None:
    lo = np.array([np.nan, -0.02, -0.02])
    hi = np.array([0.02, np.nan, 0.02])
    caps = cap_from_interval(lo, hi, **REFS)
    assert caps[0] == 0.0
    assert caps[1] == 0.0
    assert caps[2] > 0.0


def test_inverted_and_inf_fail_closed() -> None:
    lo = np.array([0.10, -np.inf, -0.02])
    hi = np.array([0.01, 0.02, np.inf])
    caps = cap_from_interval(lo, hi, **REFS)
    assert caps[0] == 0.0
    assert caps[1] == 0.0
    assert caps[2] == 0.0


def test_capped_l1_le_original_l1() -> None:
    lo = np.array([-0.20, -0.02, 0.01])
    hi = np.array([0.20, 0.03, 0.04])
    w = np.array([0.08, -0.05, 0.01])
    capped, _caps = apply_interval_caps(w, lo, hi, **REFS)
    assert np.sum(np.abs(capped)) <= np.sum(np.abs(w)) + 1e-12


def test_preserves_sign_no_silent_renormalize() -> None:
    lo = np.array([-0.15, -0.15])
    hi = np.array([0.15, 0.15])
    w = np.array([0.08, -0.08])
    capped, caps = apply_interval_caps(w, lo, hi, **REFS)
    assert capped[0] > 0.0
    assert capped[1] < 0.0
    assert np.all(np.abs(capped) <= caps + 1e-15)
    assert np.sum(np.abs(capped)) < np.sum(np.abs(w)) - 1e-12
    assert not np.isclose(np.sum(np.abs(capped)), 1.0)


def test_worse_downside_strictly_smaller_cap() -> None:
    # same width 0.08; second interval sits deeper in the left tail
    lo = np.array([-0.02, -0.06])
    hi = np.array([0.06, 0.02])
    assert np.allclose(interval_width(lo, hi), 0.08)
    caps = cap_from_interval(lo, hi, **REFS)
    assert caps[1] < caps[0]


def test_caps_in_unit_box() -> None:
    lo = np.array([-0.01, -0.50, 0.02, 0.0])
    hi = np.array([0.01, 0.50, 0.03, 0.0])
    caps = cap_from_interval(lo, hi, **REFS)
    assert np.all(caps >= 0.0)
    assert np.all(caps <= REFS["max_weight"] + 1e-15)


def test_point_interval_at_zero_is_full_cap() -> None:
    caps = cap_from_interval(np.array([0.0]), np.array([0.0]), **REFS)
    assert np.isclose(caps[0], REFS["max_weight"])


def test_deterministic() -> None:
    lo = np.array([-0.04, -0.01])
    hi = np.array([0.05, 0.02])
    a = cap_from_interval(lo, hi, **REFS)
    b = cap_from_interval(lo, hi, **REFS)
    assert np.array_equal(a, b)


def test_downside_and_width_helpers() -> None:
    lo = np.array([-0.05, 0.02])
    hi = np.array([0.03, 0.04])
    assert np.allclose(interval_width(lo, hi), [0.08, 0.02])
    assert np.allclose(downside(lo), [0.05, 0.0])


def test_nan_weight_fail_closed() -> None:
    lo = np.array([-0.02])
    hi = np.array([0.02])
    capped, _caps = apply_interval_caps(np.array([np.nan]), lo, hi, **REFS)
    assert capped[0] == 0.0


def test_bench_geometry_only_no_sharpe() -> None:
    lo = np.array([-0.02, -0.10])
    hi = np.array([0.02, 0.10])
    w = np.array([0.03, 0.001])
    out = bench_interval_caps(lo, hi, w, **REFS)
    assert {"mean_cap", "frac_binding", "mean_width"} <= set(out)
    assert out["frac_binding"] == 0.5
    assert np.isclose(out["mean_width"], 0.12)
    forbidden = ("sharpe", "pnl", "information_ratio", "ir", "sortino")
    assert not any(any(tok in k.lower() for tok in forbidden) for k in out)


def test_numeric_width_to_cap_example() -> None:
    # width 0.04, downside 0.02 → 0.02 * 0.10/0.14 * 0.05/0.07
    lo = np.array([-0.02])
    hi = np.array([0.02])
    cap = cap_from_interval(lo, hi, **REFS)[0]
    expected = 0.02 * (0.10 / 0.14) * (0.05 / 0.07)
    assert np.isclose(cap, expected)
    assert np.isclose(cap, 0.02 * (5.0 / 7.0) * (5.0 / 7.0))


def test_interval_refs_from_calibration_median() -> None:
    lo = np.array([-0.02, -0.02, -0.10])
    hi = np.array([0.02, 0.02, 0.10])
    wr, dr = interval_refs(lo, hi, multiple=2.0)
    assert wr > 0.0 and dr > 0.0
    # median width is 0.04 → ref 0.08
    assert np.isclose(wr, 0.08)


def test_equal_weight_not_median_cap_and_not_all_bind() -> None:
    """1/n ≈ 0.028 < name_max=0.05, so binding is interval-driven, not a tautology."""
    n_names, n_dates = 36, 4
    dates = np.repeat([f"d{i}" for i in range(n_dates)], n_names)
    refs = {"max_weight": 0.05, "width_ref": 0.10, "downside_ref": 0.05}
    w = np.minimum(equal_weight_per_date(dates), refs["max_weight"])
    n = n_names * n_dates
    lo = np.empty(n)
    hi = np.empty(n)
    # Half tight (cap > 1/n), half wide (cap < 1/n)
    tight = np.arange(n) % 2 == 0
    lo[tight] = -0.01
    hi[tight] = 0.01
    lo[~tight] = -0.20
    hi[~tight] = 0.20
    caps = cap_from_interval(lo, hi, **refs)
    bind = np.abs(w) > caps
    assert 0.0 < float(np.mean(bind)) < 1.0
    assert not np.allclose(w, np.median(caps))
    width = hi - lo
    med = float(np.median(width))
    wide = width >= med
    tite = width < med
    assert float(np.mean(bind[wide])) > float(np.mean(bind[tite]))


def test_equal_weight_per_date_same_on_a_date() -> None:
    dates = np.array(["2020-01-02"] * 4 + ["2020-01-03"] * 2)
    w = equal_weight_per_date(dates)
    assert np.allclose(w[:4], 0.25)
    assert np.allclose(w[4:], 0.5)
    assert w.size == 6
