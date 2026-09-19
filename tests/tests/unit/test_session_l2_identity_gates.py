"""Fail-closed session-L2 identity gates in bench_northset."""

from __future__ import annotations

import pytest

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import (
    bench_northset,
    enforce_session_l2_identity_floors,
)


def _bars(n_assets: int = 6, n_days: int = 28, seed: int = 13):
    return SyntheticMarketProvider(n_assets=n_assets, n_days=n_days, seed=seed).get_bars()


def test_enforce_passes_at_unity() -> None:
    rates = enforce_session_l2_identity_floors(
        ohlc_identity_rate=1.0,
        session_ohlc_identity_rate=1.0,
        session_reconstructs_daily_rate=1.0,
        session_volume_conservation_rate=1.0,
        session_chain_rate=1.0,
        book_uncrossed_rate=1.0,
        floor=0.99,
    )
    assert rates["session_chain_rate"] == 1.0


def test_enforce_fails_below_floor() -> None:
    with pytest.raises(ValueError, match="session-L2 identity gate failed"):
        enforce_session_l2_identity_floors(
            ohlc_identity_rate=1.0,
            session_ohlc_identity_rate=1.0,
            session_reconstructs_daily_rate=1.0,
            session_volume_conservation_rate=1.0,
            session_chain_rate=0.5,
            book_uncrossed_rate=1.0,
            floor=0.99,
        )


def test_enforce_fails_on_nan() -> None:
    with pytest.raises(ValueError, match="nan"):
        enforce_session_l2_identity_floors(
            ohlc_identity_rate=1.0,
            session_ohlc_identity_rate=float("nan"),
            session_reconstructs_daily_rate=1.0,
            session_volume_conservation_rate=1.0,
            session_chain_rate=1.0,
            book_uncrossed_rate=1.0,
            floor=0.99,
        )


def test_bench_northset_session_l2_enforces_and_stamps() -> None:
    bars = _bars()
    cfg = AppConfig()
    cfg.northset.require_adjusted_ohlc = False
    cfg.data.source = "synthetic"
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = True
    cfg.northset.session_l2_identity_floor = 0.99
    receipt = bench_northset(bars, cfg)
    assert receipt["session_l2_identity_gate"] == "enforced"
    assert receipt["session_l2_identity_floor"] == 0.99
    assert receipt["session_chain_rate"] >= 0.99
    assert receipt["session_volume_conservation_rate"] >= 0.99
    assert receipt["ohlc_identity_rate"] >= 0.99
    assert receipt["book_uncrossed_rate"] >= 0.99
    assert receipt["session_reconstructs_daily_rate"] >= 0.99


def test_bench_skips_gate_when_session_l2_off() -> None:
    bars = _bars(n_assets=4, n_days=20, seed=2)
    cfg = AppConfig()
    cfg.northset.require_adjusted_ohlc = False
    cfg.data.source = "synthetic"
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = False
    receipt = bench_northset(bars, cfg)
    assert receipt["session_l2_identity_gate"] == "skipped"
    assert receipt["n_session_book_rows"] == 0


def test_enforce_rejects_bad_floor() -> None:
    with pytest.raises(ValueError, match="identity floor"):
        enforce_session_l2_identity_floors(
            ohlc_identity_rate=1.0,
            session_ohlc_identity_rate=1.0,
            session_reconstructs_daily_rate=1.0,
            session_volume_conservation_rate=1.0,
            session_chain_rate=1.0,
            book_uncrossed_rate=1.0,
            floor=1.5,
        )
