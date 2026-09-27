"""Catalog honesty and loss accounting.

Published factor shocks and public-domain paths stay in separate columns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from quant_fund.stress.bundle import (
    bundle_path,
    eurchf_levels,
    load_bundle,
    max_simple_drawdown,
    max_yield_increase_pp,
    simple_return,
    window_series,
)
from quant_fund.stress.catalog import (
    CRISIS_CATALOG,
    FactorShock,
    crisis_by_id,
    require_historical_episodes,
)
from quant_fund.stress.replay import _apply_shock, replay_crisis, replay_portfolio
from quant_fund.stress.strategy import (
    Position,
    ResearchStrategy,
    load_strategy,
    portfolio_returns,
    strategy_from_mapping,
)

_ROOT = Path(__file__).resolve().parents[3]
_REQUIRED = (
    "crash_1987",
    "dotcom_2000_02",
    "gfc_2008",
    "flash_2010",
    "chf_2015",
    "etf_2015",
    "volmageddon_2018",
    "covid_2020",
    "rates_2022",
)

# Primary simple-return shocks read from the cited documents.
_PRIMARY = {
    "crash_1987": {"us_equity": -0.226},
    "gfc_2008": {"us_equity": -0.57},
    "flash_2010": {"us_equity": -0.10},
    "chf_2015": {"eurchf": -0.30},
    "etf_2015": {"us_equity": -0.078},
    "volmageddon_2018": {"us_equity": -0.042, "inverse_vol_etp": -0.84},
    "dfast_2025_severely_adverse": {"us_equity": -0.50, "cre_price": -0.30},
}


def _book() -> ResearchStrategy:
    return load_strategy(_ROOT / "configs" / "stress_research.yaml")


def _primary_map(crisis_id: str) -> dict[str, float]:
    crisis = crisis_by_id(crisis_id)
    out: dict[str, float] = {}
    for shock in crisis.shocks:
        if shock.role == "primary" and shock.unit == "simple_return":
            out[shock.factor] = shock.value
    return out


def test_required_episodes_are_historical_and_cited() -> None:
    require_historical_episodes()
    ids = [crisis.crisis_id for crisis in CRISIS_CATALOG]
    assert ids[:9] == list(_REQUIRED)
    for crisis_id in _REQUIRED:
        crisis = crisis_by_id(crisis_id)
        assert crisis.historical is True
        assert crisis.summary
        assert crisis.limitations
        for shock in crisis.shocks:
            assert shock.citation
            assert shock.source_url.startswith("https://")
            assert shock.licence
            assert shock.role in {"primary", "context"}
    dfast = crisis_by_id("dfast_2025_severely_adverse")
    assert dfast.historical is False
    assert dfast.window_id is None
    with pytest.raises(KeyError):
        crisis_by_id("not_a_crisis")


def test_omitted_equity_scalars_stay_omitted() -> None:
    assert crisis_by_id("dotcom_2000_02").shocks == ()
    for crisis_id in ("covid_2020", "rates_2022"):
        factors = {shock.factor for shock in crisis_by_id(crisis_id).shocks}
        assert "us_equity" not in factors
        assert all(shock.role == "context" for shock in crisis_by_id(crisis_id).shocks)
    for crisis_id, expected in _PRIMARY.items():
        got = _primary_map(crisis_id)
        for factor, value in expected.items():
            assert got[factor] == pytest.approx(value)


def test_context_shocks_are_not_primary() -> None:
    crash = crisis_by_id("crash_1987")
    roles = {(shock.value, shock.role) for shock in crash.shocks if shock.factor == "us_equity"}
    assert (-0.226, "primary") in roles
    assert (-0.31, "context") in roles
    etf = crisis_by_id("etf_2015")
    primary = [shock.value for shock in etf.shocks if shock.role == "primary"]
    assert primary == [-0.078]
    assert any(
        shock.factor == "etp_share_down_at_least_20pct" and shock.unit == "fraction"
        for shock in etf.shocks
    )
    vol = crisis_by_id("volmageddon_2018")
    assert any(
        shock.factor == "vix" and shock.unit == "level_change" and shock.role == "context"
        for shock in vol.shocks
    )


def test_bundle_is_a_small_public_domain_extract() -> None:
    path = bundle_path()
    assert path.is_file()
    assert path.stat().st_size < 500_000
    bundle = load_bundle()
    assert "Public domain" in bundle["licence"]
    assert "H.15" in bundle["licence"]
    assert "H.10" in bundle["licence"]
    assert bundle["retrieved_utc"] == "2026-09-27T09:35:00Z"
    counted = 0
    for block in bundle["series"].values():
        for window in block["windows"].values():
            counted += len(window["observations"])
    assert counted == bundle["n_observations"] == 2468


def test_verified_fred_prints() -> None:
    dgs10_2022 = dict(window_series("DGS10", "rates_2022"))
    assert dgs10_2022["2022-01-03"] == pytest.approx(1.63)
    assert dgs10_2022["2022-10-24"] == pytest.approx(4.25)
    assert dict(window_series("DFF", "rates_2022"))["2022-12-30"] == pytest.approx(4.33)
    assert dict(window_series("DGS10", "flash_2010"))["2010-05-06"] == pytest.approx(3.41)
    assert dict(window_series("DGS10", "covid_2020"))["2020-03-09"] == pytest.approx(0.54)
    assert dict(window_series("DGS10", "crash_1987"))["1987-10-19"] == pytest.approx(10.15)
    chf = dict(window_series("DEXSZUS", "chf_2015"))
    usd = dict(window_series("DEXUSEU", "chf_2015"))
    assert chf["2015-01-15"] == pytest.approx(0.893)
    assert usd["2015-01-15"] == pytest.approx(1.1598)
    levels = dict(eurchf_levels("chf_2015"))
    assert levels["2015-01-15"] == pytest.approx(0.893 * 1.1598)
    drawdown = max_simple_drawdown(np.asarray(list(levels.values()), dtype=float))
    # Noon buying rates do not reach the published −30% day-low.
    assert -0.30 < drawdown < -0.05
    increase = max_yield_increase_pp(np.asarray(list(dgs10_2022.values()), dtype=float))
    assert increase + 1e-12 >= 4.25 - 1.63


def test_path_helpers_fail_closed() -> None:
    assert max_yield_increase_pp(np.asarray([1.0, 2.0, 0.5, 1.5])) == pytest.approx(1.0)
    assert max_yield_increase_pp(np.asarray([5.0, 4.0, 3.0])) == pytest.approx(0.0)
    assert max_simple_drawdown(np.asarray([100.0, 80.0, 90.0, 70.0])) == pytest.approx(-0.30)
    assert max_simple_drawdown(np.asarray([100.0, 110.0, 121.0])) == pytest.approx(0.0)
    assert simple_return(1.20, 0.84) == pytest.approx(-0.30)
    with pytest.raises(ValueError):
        max_yield_increase_pp(np.asarray([1.0]))
    with pytest.raises(ValueError):
        max_simple_drawdown(np.asarray([1.0, 0.0]))
    with pytest.raises(ValueError):
        simple_return(0.0, 1.0)
    with pytest.raises(KeyError):
        window_series("NOT_A_SERIES", "crash_1987")


def test_shock_mapping_does_not_invent_a_zero() -> None:
    position = Position("ust_10y", 7.0, "duration")
    shock = FactorShock(
        factor="ust_10y",
        value=0.01,
        unit="yield_change",
        role="primary",
        qualifier="test",
        citation="unit test mapping, not a market print",
        source_url="https://example.invalid",
        licence="test",
    )
    applied = _apply_shock(position, shock)
    assert applied is not None
    assert applied.portfolio_return == pytest.approx(-0.07)
    assert applied.loss == pytest.approx(0.07)
    assert _apply_shock(Position("ust_10y", 7.0, "return"), shock) is None
    context = FactorShock(
        "us_equity",
        -0.31,
        "simple_return",
        "context",
        "test",
        "citation",
        "https://example.invalid",
        "test",
    )
    assert _apply_shock(Position("us_equity", 1.0, "return"), context) is None


def test_smoke_book_keeps_factor_and_path_losses_apart() -> None:
    strategy = _book()
    assert strategy.research_only is True
    replays = {item.crisis_id: item for item in replay_portfolio(strategy)}
    assert set(replays) == {crisis.crisis_id for crisis in CRISIS_CATALOG}

    crash = replays["crash_1987"]
    assert crash.factor_loss.loss == pytest.approx(0.60 * 0.226)
    assert crash.path_loss.loss is not None
    assert crash.path_loss.note.startswith("H.15/H.10")
    assert "DGS10_max_increase_pp" in crash.path_stats

    assert replays["dotcom_2000_02"].factor_loss.loss is None
    assert replays["gfc_2008"].factor_loss.loss == pytest.approx(0.60 * 0.57)
    assert replays["flash_2010"].factor_loss.loss == pytest.approx(0.60 * 0.10)
    assert replays["etf_2015"].factor_loss.loss == pytest.approx(0.60 * 0.078)
    assert replays["volmageddon_2018"].factor_loss.loss == pytest.approx(0.60 * 0.042 + 0.05 * 0.84)
    assert replays["covid_2020"].factor_loss.loss is None
    assert replays["rates_2022"].factor_loss.loss is None

    chf = replays["chf_2015"]
    assert chf.factor_loss.loss == pytest.approx(0.10 * 0.30)
    assert chf.path_loss.loss is not None
    assert chf.path_loss.loss == pytest.approx(-0.10 * chf.path_stats["eurchf_max_drawdown"])
    assert chf.factor_loss.loss != pytest.approx(chf.path_loss.loss)

    rates = replays["rates_2022"]
    increase = rates.path_stats["DGS10_max_increase_pp"]
    assert rates.path_loss.loss == pytest.approx(7.0 * increase / 100.0)
    assert "DGS2_max_increase_pp" in rates.path_stats
    assert all(item.factor == "ust_10y" for item in rates.path_loss.contributions)

    dfast = replays["dfast_2025_severely_adverse"]
    assert dfast.historical is False
    assert dfast.factor_loss.loss == pytest.approx(0.60 * 0.50)
    assert dfast.path_loss.loss is None

    equity_only = ResearchStrategy("unit", (Position("us_equity", 1.0, "return"),))
    one_day = replay_crisis(equity_only, crisis_by_id("crash_1987"))
    assert one_day.factor_loss.loss == pytest.approx(0.226)
    assert one_day.path_loss.loss is None


def test_historical_only_skips_the_supervisory_scenario() -> None:
    strategy = _book()
    ids = {item.crisis_id for item in replay_portfolio(strategy, historical_only=True)}
    assert "dfast_2025_severely_adverse" not in ids
    assert "gfc_2008" in ids


def test_strategy_loader_rejects_bad_payloads(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        strategy_from_mapping(
            {
                "name": "live",
                "research_only": False,
                "positions": [{"factor": "us_equity", "weight": 1.0, "mapping": "return"}],
            }
        )
    with pytest.raises(ValueError):
        strategy_from_mapping(
            {
                "name": "dup",
                "positions": [
                    {"factor": "us_equity", "weight": 1.0, "mapping": "return"},
                    {"factor": "us_equity", "weight": 0.2, "mapping": "return"},
                ],
            }
        )
    with pytest.raises(ValueError):
        strategy_from_mapping(
            {
                "name": "bad-map",
                "positions": [{"factor": "us_equity", "weight": 1.0, "mapping": "beta"}],
            }
        )
    refused = ResearchStrategy("no", (Position("us_equity", 1.0, "return"),), research_only=False)
    with pytest.raises(ValueError):
        replay_crisis(refused, crisis_by_id("crash_1987"))
    with pytest.raises(ValueError):
        strategy_from_mapping({"name": "  ", "positions": [{"factor": "us_equity", "weight": 1.0}]})
    with pytest.raises(ValueError):
        strategy_from_mapping({"name": "empty", "positions": []})
    with pytest.raises(ValueError):
        strategy_from_mapping({"name": "row", "positions": ["us_equity"]})
    with pytest.raises(ValueError):
        strategy_from_mapping(
            {"name": "nan", "positions": [{"factor": "us_equity", "weight": float("nan")}]}
        )
    with pytest.raises(ValueError):
        strategy_from_mapping(
            {
                "name": "assets",
                "positions": [{"factor": "us_equity", "weight": 1.0}],
                "asset_weights": ["nope"],
            }
        )
    with pytest.raises(ValueError):
        strategy_from_mapping(
            {
                "name": "assets",
                "positions": [{"factor": "us_equity", "weight": 1.0}],
                "asset_weights": {"a": float("inf")},
            }
        )
    relative = strategy_from_mapping(
        {
            "name": "cached",
            "positions": [{"factor": "us_equity", "weight": 1.0}],
            "returns_csv": "panel.csv",
        },
        base_dir=tmp_path,
    )
    assert relative.returns_csv == str((tmp_path / "panel.csv").resolve())
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("- not-a-mapping\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_strategy(bad_yaml)
    with pytest.raises(ValueError):
        portfolio_returns(("a", "b"), np.ones((12, 2)), (("a", 1.0),))
