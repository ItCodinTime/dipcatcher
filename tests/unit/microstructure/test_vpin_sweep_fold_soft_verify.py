"""Soft-verify vpin_p/t_ic, n_sweep_low, fold_positive fractions."""

from __future__ import annotations

from quant_fund.research.catalog import northset_vpin_sweep_fold_honesty_errors


def test_vpin_sweep_fold_ok() -> None:
    assert (
        northset_vpin_sweep_fold_honesty_errors(
            {
                "vpin_p_ic": 0.2,
                "vpin_t_ic": -1.5,
                "n_sweep_low": 0,
                "sweep_reject_fold_positive_fraction": 0.0,
                "sweep_follow_fold_positive_fraction": 1.0,
            }
        )
        == []
    )
    assert northset_vpin_sweep_fold_honesty_errors({}) == []


def test_vpin_ic_and_n_sweep_low() -> None:
    assert "vpin_t_ic_non_finite_fail_closed" in northset_vpin_sweep_fold_honesty_errors(
        {"vpin_t_ic": float("inf")}
    )
    assert "n_sweep_low_negative" in northset_vpin_sweep_fold_honesty_errors({"n_sweep_low": -1})


def test_fold_positive_bounds() -> None:
    assert "sweep_reject_fold_positive_fraction_out_of_unit_interval" in (
        northset_vpin_sweep_fold_honesty_errors({"sweep_reject_fold_positive_fraction": -0.01})
    )
    assert (
        northset_vpin_sweep_fold_honesty_errors(
            {"sweep_follow_fold_positive_fraction": float("nan")}
        )
        == []
    )
