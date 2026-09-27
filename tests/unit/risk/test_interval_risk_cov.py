"""Coverage tests for portfolio/interval_risk.py: the fail-closed input guards
(shape alignment, non-positive calibration knobs) and the degenerate-input edges
(empty date list, no valid calibration intervals, ref floors at 1e-8,
threshold-equality binding)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.portfolio.interval_risk import (
    DEFAULT_DOWNSIDE_REF,
    DEFAULT_WIDTH_REF,
    apply_interval_caps,
    bench_interval_caps,
    cap_from_interval,
    equal_weight_per_date,
    interval_refs,
    interval_width,
)

REFS = {"max_weight": 0.02, "width_ref": 0.10, "downside_ref": 0.05}


# --- shape alignment (lo/hi must broadcast pairwise) -------------------------


def test_interval_width_rejects_mismatched_shapes() -> None:
    lo = np.array([-0.02, -0.02])
    hi = np.array([0.02])
    with pytest.raises(ValueError, match="same shape"):
        interval_width(lo, hi)


def test_shape_guard_propagates_through_public_api() -> None:
    lo = np.array([-0.02, -0.02])
    hi = np.array([0.02, 0.03, 0.04])
    with pytest.raises(ValueError, match="same shape"):
        cap_from_interval(lo, hi, **REFS)
    with pytest.raises(ValueError, match="same shape"):
        interval_refs(lo, hi)
    with pytest.raises(ValueError, match="same shape"):
        bench_interval_caps(lo, hi, np.array([0.01, 0.01]), **REFS)


def test_scalar_endpoints_still_align() -> None:
    """Scalar lo/hi share shape () → the guard admits them."""
    assert np.isclose(interval_width(-0.02, 0.03), 0.05)
    cap = cap_from_interval(-0.02, 0.03, **REFS)
    assert np.ndim(cap) == 0
    assert 0.0 < float(cap) <= REFS["max_weight"]


# --- equal_weight_per_date empty input ----------------------------------------


def test_equal_weight_per_date_empty_is_empty_not_nan() -> None:
    w = equal_weight_per_date([])
    assert w.dtype == np.float64
    assert w.shape == (0,)


# --- interval_refs calibration edges ------------------------------------------


@pytest.mark.parametrize("multiple", [0.0, -1.0, -0.5])
def test_interval_refs_rejects_nonpositive_multiple(multiple: float) -> None:
    with pytest.raises(ValueError, match="multiple"):
        interval_refs(np.array([-0.02]), np.array([0.02]), multiple=multiple)


def test_interval_refs_no_valid_intervals_returns_defaults() -> None:
    """All-invalid calibration (NaN, inverted, ±inf) falls back to the defaults."""
    lo = np.array([np.nan, 0.10, -np.inf, -0.02])
    hi = np.array([0.02, 0.05, 0.02, np.nan])
    wr, dr = interval_refs(lo, hi)
    assert wr == DEFAULT_WIDTH_REF
    assert dr == DEFAULT_DOWNSIDE_REF


def test_interval_refs_only_valid_rows_enter_the_median() -> None:
    """An invalid row must not shift the median of the valid ones."""
    lo = np.array([-0.02, -0.02, 0.10])  # third is inverted
    hi = np.array([0.02, 0.02, 0.01])
    wr, dr = interval_refs(lo, hi, multiple=2.0)
    # valid widths are both 0.04 → ref 2*0.04; if the inverted row leaked
    # in, its negative width would drag the median under 0.04.
    assert np.isclose(wr, 0.08)
    assert np.isclose(dr, 2.0 * 0.02)


def test_interval_refs_degenerate_calibration_floors_at_1e_8() -> None:
    """A book of point intervals at zero → median width/downside are 0, so the
    refs clamp to the 1e-8 floor instead of dividing by zero downstream."""
    wr, dr = interval_refs(np.array([0.0, 0.0]), np.array([0.0, 0.0]))
    assert wr == pytest.approx(1e-8)
    assert dr == pytest.approx(1e-8)


# --- cap_from_interval parameter guards ----------------------------------------


def test_cap_from_interval_rejects_negative_max_weight() -> None:
    with pytest.raises(ValueError, match="max_weight"):
        cap_from_interval(
            np.array([-0.02]),
            np.array([0.02]),
            max_weight=-0.01,
            width_ref=0.10,
            downside_ref=0.05,
        )


def test_cap_from_interval_zero_max_weight_caps_everything() -> None:
    """Threshold equality: max_weight == 0 is admitted (< 0 is the guard) and
    collapses every cap to exactly 0."""
    caps = cap_from_interval(
        np.array([-0.02, -0.02]),
        np.array([0.02, 0.02]),
        max_weight=0.0,
        width_ref=0.10,
        downside_ref=0.05,
    )
    assert np.all(caps == 0.0)


@pytest.mark.parametrize(
    ("width_ref", "downside_ref"),
    [(0.0, 0.05), (-0.10, 0.05), (0.10, 0.0), (0.10, -0.05), (0.0, 0.0)],
)
def test_cap_from_interval_rejects_nonpositive_refs(width_ref: float, downside_ref: float) -> None:
    """Both halves of the `or` guard must trip independently."""
    with pytest.raises(ValueError, match="width_ref and downside_ref"):
        cap_from_interval(
            np.array([-0.02]),
            np.array([0.02]),
            max_weight=0.02,
            width_ref=width_ref,
            downside_ref=downside_ref,
        )


# --- apply_interval_caps alignment and edges -----------------------------------


def test_apply_interval_caps_rejects_weight_shape_mismatch() -> None:
    lo = np.array([-0.02, -0.02])
    hi = np.array([0.02, 0.02])
    with pytest.raises(ValueError, match="same shape"):
        apply_interval_caps(np.array([0.01]), lo, hi, **REFS)
    with pytest.raises(ValueError, match="same shape"):
        apply_interval_caps(np.array([0.01] * 3), lo, hi, **REFS)


def test_zero_weight_stays_zero_but_cap_is_positive() -> None:
    lo = np.array([-0.02])
    hi = np.array([0.02])
    capped, caps = apply_interval_caps(np.array([0.0]), lo, hi, **REFS)
    assert capped[0] == 0.0
    assert caps[0] > 0.0


def test_inf_weight_fails_closed_to_zero() -> None:
    lo = np.array([-0.02])
    hi = np.array([0.02])
    capped, _ = apply_interval_caps(np.array([np.inf, -np.inf][:1]), lo, hi, **REFS)
    assert capped[0] == 0.0


def test_weight_exactly_at_cap_is_not_binding() -> None:
    """Threshold equality: |w| == cap is clipped but does NOT count as binding
    (bench uses strict >)."""
    lo = np.array([-0.02])
    hi = np.array([0.02])
    _capped, caps = apply_interval_caps(np.array([0.5]), lo, hi, **REFS)
    at_cap = caps.copy()
    out = bench_interval_caps(lo, hi, at_cap, **REFS)
    assert out["frac_binding"] == 0.0
    assert out["n"] == 1.0


def test_apply_interval_caps_does_not_renormalize() -> None:
    """Gross after capping is whatever the caps leave — no silent rescale."""
    lo = np.array([-0.20, -0.20])
    hi = np.array([0.20, 0.20])
    w = np.array([0.5, -0.5])
    capped, caps = apply_interval_caps(w, lo, hi, **REFS)
    assert np.abs(capped).sum() == pytest.approx(caps.sum())
    assert capped[0] > 0.0 and capped[1] < 0.0
