"""Session L2 enforced identity rates present + notional IC→mean honesty."""

from __future__ import annotations

import math
from pathlib import Path

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.bench import bench_candle_order_book
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    candle_notional_imbalance_ic_implies_mean_honesty_errors,
    mean_notional_imbalance_honesty_errors,
    northset_session_l2_enforced_identity_rates_present_honesty_errors,
)


def _northset(*, session_l2: bool):
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = session_l2
    return bench_northset(SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars(), cfg)


def test_enforced_gate_requires_session_identity_rate_keys() -> None:
    assert northset_session_l2_enforced_identity_rates_present_honesty_errors(
        {"session_l2_identity_gate": "enforced"}
    ) == [
        "session_ohlc_identity_rate_missing_while_session_l2_identity_gate_enforced",
        "session_reconstructs_daily_rate_missing_while_session_l2_identity_gate_enforced",
        "session_volume_conservation_rate_missing_while_session_l2_identity_gate_enforced",
        "session_chain_rate_missing_while_session_l2_identity_gate_enforced",
    ]
    assert (
        northset_session_l2_enforced_identity_rates_present_honesty_errors(
            {"session_l2_identity_gate": "skipped"}
        )
        == []
    )


def test_synth_session_l2_on_enforced_identity_rates_present() -> None:
    receipt = _northset(session_l2=True)
    assert receipt.get("session_l2_identity_gate") == "enforced"
    assert northset_session_l2_enforced_identity_rates_present_honesty_errors(receipt) == []
    assert northset_session_l2_enforced_identity_rates_present_honesty_errors in (
        NORTHSET_RECEIPT_HONESTY_HELPERS
    )


def test_candle_stamps_mean_notional_and_ic_implies_mean() -> None:
    receipt = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars()
    )
    assert "ic_notional_imbalance" in receipt
    mean = float(receipt["mean_notional_imbalance"])
    assert math.isfinite(mean) and -1.0 <= mean <= 1.0
    assert candle_notional_imbalance_ic_implies_mean_honesty_errors(receipt) == []
    assert mean_notional_imbalance_honesty_errors(receipt) == []


def test_notional_ic_honesty_flags_missing_mean() -> None:
    assert candle_notional_imbalance_ic_implies_mean_honesty_errors(
        {"family": "candle_order_book", "ic_notional_imbalance": 0.1}
    ) == ["mean_notional_imbalance_missing_while_notional_ic_scored"]


def test_verify_wires_notional_and_helpers_exist() -> None:
    src = Path("src/quant_fund/research/verify.py").read_text(encoding="utf-8")
    assert "candle_notional_imbalance_ic_implies_mean_honesty_errors" in src
    assert "mean_notional_imbalance_honesty_errors" in src
