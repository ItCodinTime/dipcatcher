"""Wave 37: universe membership / ADV / assert_universe_not_from_future edges.

Complements test_data_pit.test_universe_does_not_use_future_bars (happy path).
Research/infrastructure only — no live broker / vendor MD.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quant_fund.config.models import UniverseConfig
from quant_fund.data.universe import (
    assert_universe_not_from_future,
    build_membership_panel,
    membership_asof,
    trailing_adv,
)
from quant_fund.schemas.errors import LeakageError

RESEARCH_ONLY = True
LIVE_PNL_CLAIM = False


def _bars(n: int = 25, *, vol_a: float = 2e5, vol_b: float = 1e3) -> pl.DataFrame:
    times = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n)]
    return pl.DataFrame(
        {
            "security_id": ["A"] * n + ["B"] * n,
            "event_time": times + times,
            "open": [10.0] * (2 * n),
            "high": [11.0] * (2 * n),
            "low": [9.0] * (2 * n),
            "close": [10.0] * (2 * n),
            "volume": [vol_a] * n + [vol_b] * n,
        }
    )


def _master() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "security_id": ["A", "B", "C"],
            "exchange": ["XNYS", "XNYS", "XNYS"],
            "sector": ["t", "t", "t"],
            "industry": ["i", "i", "i"],
            "security_type": ["common_stock", "common_stock", "common_stock"],
            "ticker": ["AAA", "BBB", "CCC"],
        }
    )


def test_membership_asof_empty_when_no_history() -> None:
    """asof before any bar → cleared empty frame (not an exception)."""
    bars = _bars(5)
    cfg = UniverseConfig(
        min_price=1.0,
        min_adv=0.0,
        min_history_bars=1,
        top_n_adv=None,
        exchanges=["XNYS"],
        security_types=["common_stock"],
    )
    when = datetime(2019, 1, 1, tzinfo=UTC)
    mem = membership_asof(bars, _master(), when, cfg)
    assert mem.is_empty()


def test_membership_filters_adv_and_history() -> None:
    """ADV floor and min_history_bars exclude thin / short names."""
    bars = _bars(25)
    asof = bars["event_time"].unique().sort()[-1]
    # A: ADV = 10 * 2e5 = 2e6; B: ADV = 10 * 1e3 = 1e4
    cfg_adv = UniverseConfig(
        min_price=1.0,
        min_adv=1_000_000.0,
        min_history_bars=10,
        top_n_adv=None,
        exchanges=["XNYS"],
        security_types=["common_stock"],
    )
    mem = membership_asof(bars, _master(), asof, cfg_adv)
    assert mem["security_id"].to_list() == ["A"]
    assert mem["adv"][0] == pytest.approx(2e6)
    assert mem["symbol"][0] == "AAA"

    cfg_hist = UniverseConfig(
        min_price=1.0,
        min_adv=0.0,
        min_history_bars=100,
        top_n_adv=None,
        exchanges=["XNYS"],
        security_types=["common_stock"],
    )
    assert membership_asof(bars, _master(), asof, cfg_hist).is_empty()


def test_membership_asof_excludes_late_available_bars() -> None:
    late_event = datetime(2020, 1, 20, tzinfo=UTC)
    bars = _bars(21, vol_a=1e3).with_columns(
        pl.when((pl.col("security_id") == "A") & (pl.col("event_time") == late_event))
        .then(1e8)
        .otherwise(pl.col("volume"))
        .alias("volume"),
        pl.when((pl.col("security_id") == "A") & (pl.col("event_time") == late_event))
        .then(datetime(2020, 1, 22, tzinfo=UTC))
        .otherwise(pl.col("event_time"))
        .alias("available_time"),
    )
    asof = datetime(2020, 1, 21, tzinfo=UTC)
    cfg = UniverseConfig(
        min_price=1.0,
        min_adv=1_000_000.0,
        min_history_bars=20,
        top_n_adv=None,
        exchanges=["XNYS"],
        security_types=["common_stock"],
    )

    membership = membership_asof(bars, _master(), asof, cfg)

    assert membership.is_empty()


def test_membership_top_n_adv_truncates() -> None:
    bars = _bars(25)
    asof = bars["event_time"].unique().sort()[-1]
    cfg = UniverseConfig(
        min_price=1.0,
        min_adv=0.0,
        min_history_bars=5,
        top_n_adv=1,
        exchanges=["XNYS"],
        security_types=["common_stock"],
    )
    mem = membership_asof(bars, _master(), asof, cfg)
    assert mem.height == 1
    assert mem["security_id"][0] == "A"


def test_trailing_adv_lookback_null_until_window_full() -> None:
    """rolling ADV(20) is null for the first lookback-1 bars per security."""
    bars = _bars(25)
    enriched = trailing_adv(bars.filter(pl.col("security_id") == "A"), lookback=20)
    assert enriched.filter(pl.col("adv").is_null()).height == 19
    assert enriched.filter(pl.col("adv").is_not_null()).height == 6
    last_adv = enriched.sort("event_time")["adv"][-1]
    assert last_adv == pytest.approx(10.0 * 2e5)


def test_build_membership_panel_empty_timestamps() -> None:
    assert build_membership_panel(_bars(5), _master(), [], UniverseConfig()).is_empty()


def test_membership_asof_ignores_late_master_restatement() -> None:
    """Late-available sector restatement cannot move the as-of universe join."""
    bars = _bars(25)
    asof = datetime(2020, 1, 21, tzinfo=UTC)
    master = pl.DataFrame(
        {
            "security_id": ["A", "A", "B"],
            "exchange": ["XNYS", "XNYS", "XNYS"],
            "sector": ["tech", "health", "t"],
            "industry": ["i", "i", "i"],
            "security_type": ["common_stock", "common_stock", "common_stock"],
            "ticker": ["AAA", "AAA", "BBB"],
            "valid_from": [
                datetime(2020, 1, 1, tzinfo=UTC),
                datetime(2020, 1, 10, tzinfo=UTC),
                datetime(2020, 1, 1, tzinfo=UTC),
            ],
            "valid_to": [None, None, None],
            "available_time": [
                datetime(2020, 1, 1, tzinfo=UTC),
                datetime(2020, 2, 1, tzinfo=UTC),
                datetime(2020, 1, 1, tzinfo=UTC),
            ],
        }
    )
    cfg = UniverseConfig(
        min_price=1.0,
        min_adv=0.0,
        min_history_bars=5,
        top_n_adv=None,
        exchanges=["XNYS"],
        security_types=["common_stock"],
    )
    mem = membership_asof(bars, master, asof, cfg).sort("security_id")
    by_id = dict(zip(mem["security_id"].to_list(), mem["sector"].to_list(), strict=True))
    assert by_id["A"] == "tech"


def test_assert_universe_not_from_future_empty_membership_noop() -> None:
    bars = _bars(10)
    asof = bars["event_time"].unique().sort()[5]
    assert_universe_not_from_future(pl.DataFrame(), bars, asof)


def test_assert_universe_not_from_future_no_future_bars_noop() -> None:
    bars = _bars(10)
    asof = bars["event_time"].unique().sort()[-1]
    hist = bars.filter(pl.col("event_time") <= asof)
    cfg = UniverseConfig(
        min_price=1.0,
        min_adv=0.0,
        min_history_bars=5,
        top_n_adv=None,
        exchanges=["XNYS"],
        security_types=["common_stock"],
    )
    mem = membership_asof(hist, _master(), asof, cfg)
    assert_universe_not_from_future(mem, hist, asof)


def test_assert_universe_not_from_future_pit_consistent_panel() -> None:
    """Backward rolling ADV at asof matches PIT slice — guard must not raise."""
    bars = _bars(30)
    times = bars["event_time"].unique().sort().to_list()
    asof = times[20]
    cfg = UniverseConfig(
        min_price=1.0,
        min_adv=0.0,
        min_history_bars=5,
        top_n_adv=None,
        exchanges=["XNYS"],
        security_types=["common_stock"],
    )
    panel = build_membership_panel(bars, _master(), [asof, times[25]], cfg)
    assert "asof" in panel.columns
    assert_universe_not_from_future(panel, bars, asof)


def test_assert_universe_detects_late_available_bar() -> None:
    late_event = datetime(2020, 1, 20, tzinfo=UTC)
    bars = _bars(22, vol_a=1e3).with_columns(
        pl.when((pl.col("security_id") == "A") & (pl.col("event_time") == late_event))
        .then(1e8)
        .otherwise(pl.col("volume"))
        .alias("volume"),
        pl.when((pl.col("security_id") == "A") & (pl.col("event_time") == late_event))
        .then(datetime(2020, 1, 22, tzinfo=UTC))
        .otherwise(pl.col("event_time"))
        .alias("available_time"),
    )
    asof = datetime(2020, 1, 21, tzinfo=UTC)
    membership = pl.DataFrame({"security_id": ["A"], "asof": [asof], "adv": [50_000_000.0]})

    with pytest.raises(LeakageError, match="Universe ADV"):
        assert_universe_not_from_future(membership, bars, asof)


def test_assert_universe_leakage_raises_when_adv_diverges(monkeypatch: pytest.MonkeyPatch) -> None:
    """If full-panel ADV at asof differs from PIT ADV, LeakageError fires."""
    bars = _bars(25)
    asof = bars["event_time"].unique().sort()[15]
    mem = pl.DataFrame({"security_id": ["A"], "asof": [asof], "adv": [1.0]})

    calls = {"n": 0}

    def _fake_trailing(frame: pl.DataFrame, lookback: int) -> pl.DataFrame:  # noqa: ARG001
        calls["n"] += 1
        # First call = full panel (leaked); second = PIT
        adv = 99.0 if calls["n"] == 1 else 1.0
        row = frame.filter(pl.col("event_time") == asof)
        if row.is_empty():
            return row
        return row.with_columns(pl.lit(adv).alias("adv"))

    monkeypatch.setattr("quant_fund.data.universe.trailing_adv", _fake_trailing)
    with pytest.raises(LeakageError, match="Universe ADV"):
        assert_universe_not_from_future(mem, bars, asof)


def _listing_cfg() -> UniverseConfig:
    return UniverseConfig(
        min_price=1.0,
        min_adv=0.0,
        min_history_bars=5,
        top_n_adv=None,
        exchanges=["XNYS"],
        security_types=["common_stock"],
    )


def test_membership_asof_drops_name_after_visible_delist() -> None:
    bars = _bars(25)
    times = bars["event_time"].unique().sort().to_list()
    delist_on = times[20]
    after = times[21]
    actions = pl.DataFrame(
        {
            "security_id": ["A"],
            "event_time": [delist_on],
            "available_time": [delist_on],
            "action_type": ["delist"],
        }
    )
    cfg = _listing_cfg()
    on_event = membership_asof(bars, _master(), delist_on, cfg, actions=actions)
    after_event = membership_asof(bars, _master(), after, cfg, actions=actions)
    assert "A" in on_event["security_id"].to_list()
    assert "A" not in after_event["security_id"].to_list()
    assert "B" in after_event["security_id"].to_list()


def test_membership_asof_late_delist_cannot_rewrite_preavailability() -> None:
    bars = _bars(25)
    asof = datetime(2020, 1, 21, tzinfo=UTC)
    actions = pl.DataFrame(
        {
            "security_id": ["A"],
            "event_time": [datetime(2020, 1, 10, tzinfo=UTC)],
            "available_time": [datetime(2020, 2, 1, tzinfo=UTC)],
            "action_type": ["delist"],
        }
    )
    mem = membership_asof(bars, _master(), asof, _listing_cfg(), actions=actions)
    assert "A" in mem["security_id"].to_list()


def test_membership_asof_ticker_change_overrides_symbol() -> None:
    bars = _bars(25)
    asof = datetime(2020, 1, 21, tzinfo=UTC)
    actions = pl.DataFrame(
        {
            "security_id": ["A"],
            "event_time": [datetime(2020, 1, 10, tzinfo=UTC)],
            "available_time": [datetime(2020, 1, 10, tzinfo=UTC)],
            "action_type": ["ticker_change"],
            "new_ticker": ["AA2"],
        }
    )
    mem = membership_asof(bars, _master(), asof, _listing_cfg(), actions=actions)
    by_id = dict(zip(mem["security_id"].to_list(), mem["symbol"].to_list(), strict=True))
    assert by_id["A"] == "AA2"
    assert by_id["B"] == "BBB"


def test_membership_asof_announced_delist_respects_include_delisted_flag() -> None:
    bars = _bars(25)
    asof = datetime(2020, 1, 15, tzinfo=UTC)
    actions = pl.DataFrame(
        {
            "security_id": ["A"],
            "event_time": [datetime(2020, 1, 25, tzinfo=UTC)],
            "available_time": [datetime(2020, 1, 10, tzinfo=UTC)],
            "action_type": ["delist"],
        }
    )
    keep = membership_asof(bars, _master(), asof, _listing_cfg(), actions=actions)
    drop_cfg = _listing_cfg().model_copy(update={"include_delisted": False})
    drop = membership_asof(bars, _master(), asof, drop_cfg, actions=actions)
    assert "A" in keep["security_id"].to_list()
    assert "A" not in drop["security_id"].to_list()
