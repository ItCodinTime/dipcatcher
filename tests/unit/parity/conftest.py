"""Shared SYNTHETIC tapes for parity tests. Not market evidence."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from quant_fund.config.loader import load_config
from quant_fund.config.models import AppConfig
from quant_fund.parity.session import Bar, MarketSession
from quant_fund.parity.strategy import FixedWeightStrategy


def make_session(n_days: int = 4) -> MarketSession:
    bars: list[Bar] = []
    for day in range(n_days):
        stamp = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=day)
        for sid, px0 in (("A", 100.0), ("B", 50.0)):
            px = px0 + day
            bars.append(
                Bar(
                    security_id=sid,
                    event_time=stamp,
                    open=px,
                    high=px,
                    low=px,
                    close=px,
                    volume=1_000_000.0,
                    source="synthetic",
                    revision_id="r0",
                    available_time=stamp,
                    adv=100_000_000.0,
                    vol_20=0.02,
                )
            )
    return MarketSession(bars)


def paper_config(tmp_path: Path) -> AppConfig:
    cfg = load_config("configs/paper.yaml")
    cfg.data.root = tmp_path
    return cfg


@pytest.fixture
def session() -> MarketSession:
    return make_session()


@pytest.fixture
def config(tmp_path: Path) -> AppConfig:
    return paper_config(tmp_path)


@pytest.fixture
def strategy() -> FixedWeightStrategy:
    return FixedWeightStrategy({"A": 0.05, "B": -0.02})
