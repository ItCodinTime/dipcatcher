"""Determinism, PIT membership, and oracle-slope invariants for the labeled panel."""

from __future__ import annotations

import random

import numpy as np
import polars as pl

from quant_fund.data.adapters.synthetic import SyntheticMarketProvider


def test_provider_frames_are_fully_deterministic() -> None:
    first = SyntheticMarketProvider(n_assets=4, n_days=40, seed=11)
    second = SyntheticMarketProvider(n_assets=4, n_days=40, seed=11)
    assert first.get_bars().equals(second.get_bars())
    assert first.get_corporate_actions().equals(second.get_corporate_actions())
    assert first.get_security_master().equals(second.get_security_master())


def test_construction_does_not_disturb_global_random_state() -> None:
    np.random.seed(123)
    random.seed(456)
    expected_np = np.random.normal()
    expected_py = random.random()
    np.random.seed(123)
    random.seed(456)
    SyntheticMarketProvider(n_assets=4, n_days=20, seed=11)
    assert np.random.normal() == expected_np
    assert random.random() == expected_py


def test_delisted_asset_stays_member_through_its_last_bar() -> None:
    provider = SyntheticMarketProvider(n_assets=4, n_days=40, seed=3)
    bars = provider.get_bars()
    master = provider.get_security_master()
    delisted = master.filter(master["security_id"] == "SEC_0003")
    last_bar = bars.filter(bars["security_id"] == "SEC_0003")["event_time"].max()
    valid_to = delisted["valid_to"][0]
    assert valid_to is not None
    assert valid_to > last_bar
    delist = provider.get_corporate_actions().filter(pl.col("action_type") == "delist")
    assert delist.height == 1
    assert valid_to > delist["event_time"][0]


def test_planted_signal_recovers_the_documented_oracle() -> None:
    """log-return_{t+1} on planted_signal_t slopes to oracle_beta * vol regime."""
    provider = SyntheticMarketProvider(n_assets=12, n_days=320, seed=7)
    bars = provider.get_bars().sort("security_id", "event_time")
    closes = bars.pivot(index="event_time", on="security_id", values="close").drop("event_time")
    signals = bars.pivot(index="event_time", on="security_id", values="planted_signal").drop(
        "event_time"
    )
    keep = [
        i
        for i, name in enumerate(closes.columns)
        if name not in ("SEC_MKT", "SEC_0001", "SEC_0011")
    ]
    returns = np.log(closes.to_numpy()[1:, keep] / closes.to_numpy()[:-1, keep])
    x = signals.to_numpy()[:-1, keep]
    n = returns.shape[0]

    def slope(y: np.ndarray, x_: np.ndarray) -> float:
        mask = np.isfinite(y) & np.isfinite(x_)
        return float((x_[mask] * y[mask]).sum() / (x_[mask] * x_[mask]).sum())

    low_regime = slope(returns[: n // 2], x[: n // 2])
    high_regime = slope(returns[n // 2 :], x[n // 2 :])
    assert abs(low_regime - provider.oracle_beta) < 0.004
    assert abs(high_regime - 2.0 * provider.oracle_beta) < 0.008


def test_every_frame_is_labeled_synthetic() -> None:
    provider = SyntheticMarketProvider(n_assets=4, n_days=20, seed=5)
    for frame in (
        provider.get_bars(),
        provider.get_corporate_actions(),
        provider.get_security_master(),
    ):
        assert frame["source"].unique().to_list() == ["synthetic"]
        assert frame["revision_id"].unique().to_list() == ["SYNTHETIC"]
