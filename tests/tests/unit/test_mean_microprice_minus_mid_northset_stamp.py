"""Northset stamps mean_microprice_minus_mid[+_bps]; soft-verify both."""

from __future__ import annotations

import inspect

from quant_fund.northset import benches
from quant_fund.research.catalog import mean_microprice_minus_mid_honesty_errors


def test_bench_stamps_microprice_minus_mid_pair() -> None:
    src = inspect.getsource(benches.bench_northset)
    assert '"mean_microprice_minus_mid"' in src
    assert '"mean_microprice_minus_mid_bps"' in src


def test_microprice_minus_mid_honesty_covers_bps() -> None:
    assert (
        mean_microprice_minus_mid_honesty_errors(
            {"mean_microprice_minus_mid": -0.01, "mean_microprice_minus_mid_bps": 1.2}
        )
        == []
    )
    assert "mean_microprice_minus_mid_bps_non_finite_fail_closed" in (
        mean_microprice_minus_mid_honesty_errors({"mean_microprice_minus_mid_bps": float("inf")})
    )


def test_microprice_minus_mid_nan_skipped() -> None:
    assert (
        mean_microprice_minus_mid_honesty_errors({"mean_microprice_minus_mid": float("nan")}) == []
    )
