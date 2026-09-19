"""Wave 33: SyntheticMarketProvider edge fixtures (labeled SYNTHETIC / research-only)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from quant_fund.data.adapters.synthetic import REVISION, SOURCE, SyntheticMarketProvider

RESEARCH_ONLY = True
LIVE_PNL_CLAIM = False


def test_synthetic_rejects_empty_universe_and_short_panel() -> None:
    with pytest.raises(ValueError, match="n_assets must be >= 2"):
        SyntheticMarketProvider(n_assets=0, n_days=40, seed=1)
    with pytest.raises(ValueError, match="n_assets must be >= 2"):
        SyntheticMarketProvider(n_assets=1, n_days=40, seed=1)
    with pytest.raises(ValueError, match="n_days must be >= 16"):
        SyntheticMarketProvider(n_assets=4, n_days=0, seed=1)
    with pytest.raises(ValueError, match="n_days must be >= 16"):
        SyntheticMarketProvider(n_assets=4, n_days=15, seed=1)


def test_synthetic_rejects_bad_seed() -> None:
    with pytest.raises(ValueError, match="seed must be between"):
        SyntheticMarketProvider(n_assets=4, n_days=20, seed=-1)
    with pytest.raises(ValueError, match="seed must be between"):
        SyntheticMarketProvider(n_assets=4, n_days=20, seed=2**32)


def test_synthetic_pit_columns_and_labels_present() -> None:
    p = SyntheticMarketProvider(n_assets=4, n_days=20, seed=11)
    bars = p.get_bars()
    for c in (
        "event_time",
        "available_time",
        "ingested_time",
        "source",
        "revision_id",
        "security_id",
        "planted_signal",
    ):
        assert c in bars.columns
    assert bars["source"].unique().to_list() == [SOURCE]
    assert bars["revision_id"].unique().to_list() == [REVISION]
    assert bars.height > 0
    # available_time == event_time for synthetic close bars (no lookahead lag)
    assert (bars["available_time"] == bars["event_time"]).all()


def test_synthetic_filter_empty_and_seed_reproducible() -> None:
    p = SyntheticMarketProvider(n_assets=4, n_days=20, seed=42)
    assert p.get_bars(security_ids=["NO_SUCH_SEC"]).height == 0
    assert p.get_bars(start=datetime(2099, 1, 1, tzinfo=UTC)).height == 0
    assert p.get_corporate_actions(end=datetime(2000, 1, 1, tzinfo=UTC)).height == 0

    a = SyntheticMarketProvider(n_assets=4, n_days=20, seed=42).get_bars()["close"].to_list()
    b = SyntheticMarketProvider(n_assets=4, n_days=20, seed=42).get_bars()["close"].to_list()
    c = SyntheticMarketProvider(n_assets=4, n_days=20, seed=43).get_bars()["close"].to_list()
    assert a == b
    assert a != c

    master = p.get_security_master()
    assert "SEC_MKT" in master["security_id"].to_list()
    assert master.height == 4
    for column in (
        "valid_from",
        "valid_to",
        "available_time",
        "ingested_time",
        "source",
        "revision_id",
        "ticker",
    ):
        assert column in master.columns
    assert (master["available_time"] == master["valid_from"]).all()
    assert master["source"].unique().to_list() == [SOURCE]
