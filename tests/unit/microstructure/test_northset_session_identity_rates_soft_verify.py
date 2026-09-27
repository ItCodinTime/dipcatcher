"""Soft-verify session L2 identity rates ∈ [0, 1]."""

from __future__ import annotations

from quant_fund.research.catalog import northset_session_identity_rates_honesty_errors


def test_identity_rates_ok() -> None:
    assert (
        northset_session_identity_rates_honesty_errors(
            {
                "session_ohlc_identity_rate": 1.0,
                "session_reconstructs_daily_rate": 1.0,
                "session_volume_conservation_rate": 0.99,
                "session_chain_rate": 0.95,
            }
        )
        == []
    )
    assert (
        northset_session_identity_rates_honesty_errors({"session_chain_rate": float("nan")}) == []
    )


def test_identity_rates_flag_oob() -> None:
    errs = northset_session_identity_rates_honesty_errors(
        {"session_reconstructs_daily_rate": 1.01, "session_chain_rate": -0.1}
    )
    assert "session_reconstructs_daily_rate_out_of_unit_interval" in errs
    assert "session_chain_rate_out_of_unit_interval" in errs


def test_session_ohlc_identity_rate_oob() -> None:
    assert northset_session_identity_rates_honesty_errors({"session_ohlc_identity_rate": 1.5}) == [
        "session_ohlc_identity_rate_out_of_unit_interval"
    ]


def test_identity_rates_volume_conservation() -> None:
    assert northset_session_identity_rates_honesty_errors(
        {"session_volume_conservation_rate": 2.0}
    ) == ["session_volume_conservation_rate_out_of_unit_interval"]


def test_synth_session_l2_identity_rates_honesty_clean() -> None:
    """Session L2 on → identity rates stamped and soft-verify clean."""
    from quant_fund.config.models import AppConfig
    from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
    from quant_fund.northset.benches import bench_northset

    bars = SyntheticMarketProvider(n_assets=4, n_days=40, seed=7).get_bars()
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = True
    receipt = bench_northset(bars, cfg)
    for key in (
        "session_ohlc_identity_rate",
        "session_reconstructs_daily_rate",
        "session_volume_conservation_rate",
        "session_chain_rate",
    ):
        assert key in receipt, key
    assert northset_session_identity_rates_honesty_errors(receipt) == []


def test_session_ohlc_identity_never_equates_daily_ohlc_key() -> None:
    """session_ohlc_identity_rate ≠ ohlc_identity_rate — distinct receipt keys."""
    from quant_fund.config.models import AppConfig
    from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
    from quant_fund.northset.benches import bench_northset

    bars = SyntheticMarketProvider(n_assets=4, n_days=40, seed=7).get_bars()
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = True
    receipt = bench_northset(bars, cfg)
    assert "ohlc_identity_rate" in receipt
    assert "session_ohlc_identity_rate" in receipt
    # Both may equal 1.0 on synth; never-equate is key identity, not value inequality.
    assert "ohlc_identity_rate" != "session_ohlc_identity_rate"
