"""Soft-verify queue_imbalance_mean, sweep_any_rate, ofi_mean_ic."""

from __future__ import annotations

from quant_fund.research.catalog import northset_queue_sweep_ofi_honesty_errors


def test_queue_sweep_ofi_ok() -> None:
    assert (
        northset_queue_sweep_ofi_honesty_errors(
            {
                "queue_imbalance_mean": -0.5,
                "sweep_any_rate": 0.2,
                "ofi_mean_ic": -0.03,
            }
        )
        == []
    )
    assert northset_queue_sweep_ofi_honesty_errors({}) == []


def test_queue_and_sweep_bounds() -> None:
    assert "queue_imbalance_mean_out_of_unit_interval" in northset_queue_sweep_ofi_honesty_errors(
        {"queue_imbalance_mean": 1.01}
    )
    assert "sweep_any_rate_out_of_unit_interval" in northset_queue_sweep_ofi_honesty_errors(
        {"sweep_any_rate": 1.5}
    )


def test_ofi_mean_ic_finite_when_present() -> None:
    assert northset_queue_sweep_ofi_honesty_errors({"ofi_mean_ic": float("nan")}) == []
    assert "ofi_mean_ic_non_finite" in northset_queue_sweep_ofi_honesty_errors(
        {"ofi_mean_ic": float("inf")}
    )
