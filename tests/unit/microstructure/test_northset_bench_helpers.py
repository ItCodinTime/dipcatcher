"""Unit tests for pure Northset bench helpers (McCabe extractions)."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from quant_fund.northset.bench_helpers import (
    book_structure_mean_fields,
    cond_fwd_mean,
    enforce_structure_floors,
    evidence_provenance,
    float_col_or_empty,
    join_age_metrics,
    metrics_required_finite_ok,
    nanmean_col,
    nanmean_finite,
    rate_or_nan,
    structure_finite_rates,
    sweep_evidence_receipt_fields,
)


def test_nanmean_finite_empty_and_all_nan() -> None:
    assert math.isnan(nanmean_finite([]))
    assert math.isnan(nanmean_finite([float("nan"), float("nan")]))
    assert nanmean_finite([1.0, float("nan"), 3.0]) == pytest.approx(2.0)


def test_nanmean_col_missing_vs_present() -> None:
    frame = pl.DataFrame({"a": [1.0, 3.0, float("nan")]})
    assert math.isnan(nanmean_col(frame, "missing"))
    assert nanmean_col(frame, "a") == pytest.approx(2.0)


def test_float_col_or_empty() -> None:
    frame = pl.DataFrame({"x": [1.5, 2.5]})
    assert float_col_or_empty(frame, "missing").size == 0
    np.testing.assert_allclose(float_col_or_empty(frame, "x"), [1.5, 2.5])


def test_evidence_provenance_paths() -> None:
    assert evidence_provenance(
        bar_source="synthetic",
        book_source="synthetic_lob",
        book_dgp="synthetic_lob",
        session_l2_enabled=True,
    ) == ("SYNTHETIC", "synthetic_lob")
    label, dgp = evidence_provenance(
        bar_source="synthetic",
        book_source="vendor_x",
        book_dgp="vendor_panel:vendor_x",
        session_l2_enabled=True,
    )
    assert label == "SYNTHETIC"
    assert dgp == "vendor_panel:vendor_x"
    label, dgp = evidence_provenance(
        bar_source="polygon",
        book_source="synthetic_lob",
        book_dgp="synthetic_lob",
        session_l2_enabled=True,
    )
    assert label == "MIXED_SYNTHETIC_DERIVED"
    assert dgp == "mixed_sources"
    label, dgp = evidence_provenance(
        bar_source="polygon",
        book_source="vendor_x",
        book_dgp="vendor_panel:vendor_x",
        session_l2_enabled=False,
    )
    assert label == "polygon"
    assert dgp == "vendor_panel:vendor_x"


def test_join_age_metrics_missing_and_present() -> None:
    empty = pl.DataFrame({"security_id": []})
    cov, mean_age, max_age = join_age_metrics(empty)
    assert math.isnan(cov) and math.isnan(mean_age) and math.isnan(max_age)
    fused = pl.DataFrame(
        {
            "join_coverage": [0.75, 0.75],
            "book_age_seconds": [1.0, 3.0],
        }
    )
    cov, mean_age, max_age = join_age_metrics(fused)
    assert cov == pytest.approx(0.75)
    assert mean_age == pytest.approx(2.0)
    assert max_age == pytest.approx(3.0)


def test_enforce_structure_floors_fail_closed() -> None:
    rates = {"depth_shape_finite_rate": 0.5}
    enforce_structure_floors(rates, {"depth_shape_finite_rate": None})
    with pytest.raises(ValueError, match="below floor"):
        enforce_structure_floors(rates, {"depth_shape_finite_rate": 0.9})


def test_rate_or_nan_missing_columns() -> None:
    def _rate(_rows: list[dict[str, float]]) -> float:
        return 1.0

    assert math.isnan(rate_or_nan({"a"}, ("a", "b"), [], _rate))
    assert rate_or_nan({"a", "b"}, ("a", "b"), [], _rate) == 1.0


def test_metrics_required_finite_ok_empty() -> None:
    assert metrics_required_finite_ok([]) is False


def test_cond_fwd_mean() -> None:
    scored = pl.DataFrame(
        {
            "fwd_ret_1": [0.1, -0.2, 0.3],
            "sweep_high_reclaim": [1.0, 0.0, 1.0],
        }
    )
    assert math.isnan(cond_fwd_mean(scored, "missing"))
    assert cond_fwd_mean(scored, "sweep_high_reclaim") == pytest.approx(0.2)


def test_book_structure_mean_fields_missing() -> None:
    book = pl.DataFrame({"security_id": ["a"]})
    fields = book_structure_mean_fields(book)
    assert "mean_tob_size_share" in fields
    assert math.isnan(fields["mean_tob_size_share"])


def test_structure_finite_rates_missing_columns_are_nan() -> None:
    book = pl.DataFrame({"security_id": ["a"]})
    rates = structure_finite_rates(book, [])
    assert all(math.isnan(v) for v in rates.values())


def test_sweep_evidence_receipt_fields_primary_default() -> None:
    evidence = {
        "event_studies": [
            {
                "horizon": 2,
                "signal": "sweep_follow_signed",
                "p_value": 0.1,
                "hac_t": 1.0,
                "mean_excess_bps": 1.0,
                "cost_adjusted_mean_bps": 0.5,
                "cost_adjusted_p_greater": 0.2,
                "positive_fraction": 0.6,
            },
            {
                "horizon": 1,
                "signal": "sweep_reject_signed",
                "p_value": 0.01,
                "hac_t": 2.0,
                "mean_excess_bps": 3.0,
                "cost_adjusted_mean_bps": 2.0,
                "cost_adjusted_p_greater": 0.05,
                "positive_fraction": 0.7,
            },
        ],
        "permutation_placebos": {
            "sweep_reject_signed": {"placebo_p_value": 0.4, "observed_mean_ic": 0.1},
            "sweep_follow_signed": {"placebo_p_value": 0.5, "observed_mean_ic": 0.2},
        },
        "matched_controls": {},
    }
    out = sweep_evidence_receipt_fields(evidence, primary_test_id_default="PRIMARY")
    assert out["sweep_reject_event_p"] == 0.01
    assert "sweep_follow_event_p" not in out  # horizon 2 skipped
    assert out["sweep_primary_test_id"] == "PRIMARY"
    assert out["sweep_reject_placebo_p"] == 0.4
