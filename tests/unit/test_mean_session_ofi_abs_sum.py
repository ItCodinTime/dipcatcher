"""mean_session_ofi_abs_sum receipt + soft-verify ≥0; VPIN denom companion."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import SESSION_RECEIPT_KEYS, bench_northset
from quant_fund.research.catalog import (
    NORTHSET_SESSION_MEANS_HONESTY_HELPERS,
    northset_session_ofi_abs_sum_honesty_errors,
)


def test_stamps_and_in_receipt_keys_when_session_l2_on() -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=24, seed=43).get_bars()
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = True
    receipt = bench_northset(bars, cfg)
    assert "mean_session_ofi_abs_sum" in SESSION_RECEIPT_KEYS
    val = float(receipt["mean_session_ofi_abs_sum"])
    assert math.isfinite(val) and val >= 0.0
    assert northset_session_ofi_abs_sum_honesty_errors(receipt) == []


def test_nan_when_session_l2_off_and_soft_verify_skips() -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=24, seed=43).get_bars()
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = False
    receipt = bench_northset(bars, cfg)
    assert math.isnan(float(receipt["mean_session_ofi_abs_sum"]))
    assert northset_session_ofi_abs_sum_honesty_errors(receipt) == []
    assert northset_session_ofi_abs_sum_honesty_errors({"mean_session_ofi_abs_sum": -1.0}) == [
        "mean_session_ofi_abs_sum_negative_or_non_finite"
    ]


def test_dispatcher_includes_helper_and_cli_echoes() -> None:
    names = {fn.__name__ for fn in NORTHSET_SESSION_MEANS_HONESTY_HELPERS}
    assert "northset_session_ofi_abs_sum_honesty_errors" in names
    config = Path("configs/research.yaml")
    if not config.exists():
        pytest.skip("configs/research.yaml missing")
    result = CliRunner().invoke(app, ["northset", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert "mean_session_ofi_abs_sum=" in result.output
