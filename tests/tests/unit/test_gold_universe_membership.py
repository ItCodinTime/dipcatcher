"""Day Wave 108/109: gold/CS consume PIT universe membership.

Research/infrastructure only — no live broker / vendor MD / live_pnl_claim.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from quant_fund.config import load_config
from quant_fund.config.models import AppConfig, HorizonConfig
from quant_fund.data.universe import (
    attach_membership_flag,
    require_panel_keys_in_membership,
    require_valid_membership_panel,
    restrict_to_membership,
)
from quant_fund.features.cross_sectional import apply_cross_sectional
from quant_fund.features.engine import build_features
from quant_fund.features.metadata import FEATURE_SET_VERSION
from quant_fund.labels.engine import build_labels
from quant_fund.pipeline.dataset import build_gold, clear_panel_cache, panel
from quant_fund.schemas.errors import PointInTimeError

RESEARCH_ONLY = True
LIVE_PNL_CLAIM = False


def _times(n: int) -> list[datetime]:
    return [datetime(2020, 1, 2, 16, 0, tzinfo=UTC) + timedelta(days=i) for i in range(n)]


def _membership(security_ids: list[str], asof: list[datetime]) -> pl.DataFrame:
    return pl.DataFrame({"security_id": security_ids, "asof": asof})


def test_require_valid_membership_panel_fail_closed_edges() -> None:
    with pytest.raises(PointInTimeError, match="missing required columns"):
        require_valid_membership_panel(pl.DataFrame({"security_id": ["A"]}))
    t0 = datetime(2020, 1, 2, tzinfo=UTC)
    with pytest.raises(PointInTimeError, match="blank security_id"):
        require_valid_membership_panel(pl.DataFrame({"security_id": ["  "], "asof": [t0]}))
    with pytest.raises(PointInTimeError, match="null asof"):
        require_valid_membership_panel(pl.DataFrame({"security_id": ["A"], "asof": [None]}))
    with pytest.raises(PointInTimeError, match="duplicate"):
        require_valid_membership_panel(pl.DataFrame({"security_id": ["A", "A"], "asof": [t0, t0]}))


def test_restrict_to_membership_empty_universe_fail_closed() -> None:
    t0 = datetime(2020, 1, 2, tzinfo=UTC)
    frame = pl.DataFrame({"security_id": ["A"], "event_time": [t0], "x": [1.0]})
    empty = pl.DataFrame(schema={"security_id": pl.String, "asof": pl.Datetime(time_zone="UTC")})
    with pytest.raises(PointInTimeError, match="empty"):
        restrict_to_membership(frame, empty)
    assert restrict_to_membership(frame.clear(), empty).is_empty()


def test_late_membership_asof_cannot_mark_earlier_bar() -> None:
    times = _times(2)
    frame = pl.DataFrame(
        {
            "security_id": ["A", "A"],
            "event_time": times,
            "x": [1.0, 2.0],
        }
    )
    membership = _membership(["A"], [times[1]])
    flagged = attach_membership_flag(frame, membership).sort("event_time")
    assert flagged["_in_universe"].to_list() == [False, True]
    kept = restrict_to_membership(frame, membership)
    assert kept["event_time"].to_list() == [times[1]]


def test_cs_transform_excludes_non_universe_names() -> None:
    t0 = datetime(2020, 1, 2, tzinfo=UTC)
    df = pl.DataFrame(
        {
            "event_time": [t0, t0],
            "security_id": ["keep", "drop"],
            "x": [1.0, 100.0],
            "_in_universe": [True, False],
        }
    )
    out = apply_cross_sectional(df, ["x"], 0.0).sort("security_id")
    by_id = dict(zip(out["security_id"].to_list(), out["cs_pct_x"].to_list(), strict=True))
    assert by_id["keep"] == pytest.approx(0.5)
    assert by_id["drop"] is None


def _ohlc_bars(n: int = 25) -> pl.DataFrame:
    times = _times(n)
    rows: list[dict[str, object]] = []
    for sid, px0, volume in (("A", 10.0, 2e5), ("B", 10.0, 1e2)):
        px = px0
        for i, ts in enumerate(times):
            px = px0 * (1.01 if sid == "B" else 1.0)
            if sid == "B":
                px = px0 * (1.0 + 0.05 * (i + 1))
            rows.append(
                {
                    "security_id": sid,
                    "event_time": ts,
                    "available_time": ts,
                    "close": float(px),
                    "close_total_return": float(px),
                    "volume": volume,
                    "open_split_adjusted": float(px),
                    "close_split_adjusted": float(px),
                    "high_split_adjusted": float(px) * 1.01,
                    "low_split_adjusted": float(px) * 0.99,
                }
            )
    return pl.DataFrame(rows)


def test_build_features_membership_keeps_history_and_drops_ineligible() -> None:
    bars = _ohlc_bars(25)
    times = bars["event_time"].unique().sort().to_list()
    asof = times[-1]
    membership = _membership(["A"], [asof])
    feats = build_features(bars, AppConfig(), membership=membership)
    assert feats["security_id"].to_list() == ["A"]
    assert feats["event_time"].to_list() == [asof]
    # Rolling vol still uses the pre-membership history on A, not a 1-bar window.
    assert feats["vol_20"][0] is not None
    assert feats["vol_20"][0] == pytest.approx(0.0, abs=1e-12)


def test_ineligible_name_cannot_move_cs_mean() -> None:
    bars = _ohlc_bars(25)
    times = bars["event_time"].unique().sort().to_list()
    asof = times[-1]
    cfg = AppConfig()
    all_names = build_features(bars, cfg)
    members = build_features(bars, cfg, membership=_membership(["A"], [asof]))
    full_row = all_names.filter((pl.col("security_id") == "A") & (pl.col("event_time") == asof))
    member_row = members.filter((pl.col("security_id") == "A") & (pl.col("event_time") == asof))
    assert full_row["cs_mean_ret"][0] != pytest.approx(member_row["cs_mean_ret"][0])
    assert member_row["cs_mean_ret"][0] == pytest.approx(member_row["ret_1"][0])
    assert "B" not in members["security_id"].to_list()


def test_build_labels_idio_mean_ignores_ineligible_names() -> None:
    times = _times(3)
    bars = pl.DataFrame(
        {
            "security_id": ["A"] * 3 + ["B"] * 3 + ["SEC_MKT"] * 3,
            "event_time": times * 3,
            "close_total_return": [100.0, 110.0, 121.0, 100.0, 200.0, 400.0, 100.0, 101.0, 102.0],
        }
    )
    cfg = AppConfig(horizons=HorizonConfig(bars=[1], names=["1d"]))
    membership = _membership(["A"], [times[0]])
    labels = build_labels(bars, cfg, membership=membership)
    assert labels["security_id"].to_list() == ["A"]
    a = labels.filter(pl.col("security_id") == "A")
    # Cross-section mean at t0 uses only members; B's +100% cannot pull A's idio.
    assert a["future_idio_return_1"][0] == pytest.approx(0.0, abs=1e-12)


def test_build_features_empty_membership_fail_closed() -> None:
    bars = _ohlc_bars(5)
    empty = pl.DataFrame(schema={"security_id": pl.String, "asof": pl.Datetime(time_zone="UTC")})
    with pytest.raises(PointInTimeError, match="empty"):
        build_features(bars, AppConfig(), membership=empty)


@pytest.mark.synthetic
def test_build_gold_persists_only_membership_keys(tmp_path: Path) -> None:
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.data.synthetic_n_assets = 6
    cfg.data.synthetic_n_days = 40
    cfg.universe.min_history_bars = 5
    cfg.universe.min_adv = 0.0
    cfg.universe.top_n_adv = 2
    feats, labs = build_gold(cfg)
    universe = pl.read_parquet(tmp_path / "silver" / "universe.parquet")
    gold_keys = set(zip(feats["security_id"].to_list(), feats["event_time"].to_list(), strict=True))
    lab_keys = set(zip(labs["security_id"].to_list(), labs["event_time"].to_list(), strict=True))
    member_keys = set(
        zip(universe["security_id"].to_list(), universe["asof"].to_list(), strict=True)
    )
    assert gold_keys <= member_keys
    assert lab_keys <= member_keys
    assert feats.height > 0
    n_names = feats.filter(pl.col("event_time") == feats["event_time"].max())[
        "security_id"
    ].n_unique()
    assert n_names == 2


def test_require_panel_keys_in_membership_fail_closed_on_extra_name() -> None:
    t0 = datetime(2020, 1, 2, tzinfo=UTC)
    frame = pl.DataFrame(
        {"security_id": ["A", "drop"], "event_time": [t0, t0], "ret_1": [0.1, 0.2]}
    )
    with pytest.raises(PointInTimeError, match="outside PIT universe"):
        require_panel_keys_in_membership(frame, _membership(["A"], [t0]))


def test_require_panel_keys_in_membership_allows_gold_subset() -> None:
    t0 = datetime(2020, 1, 2, tzinfo=UTC)
    t1 = datetime(2020, 1, 3, tzinfo=UTC)
    frame = pl.DataFrame({"security_id": ["A"], "event_time": [t0], "ret_1": [0.1]})
    require_panel_keys_in_membership(frame, _membership(["A", "B"], [t0, t1]))


def test_require_panel_keys_in_membership_empty_universe_fail_closed() -> None:
    t0 = datetime(2020, 1, 2, tzinfo=UTC)
    frame = pl.DataFrame({"security_id": ["A"], "event_time": [t0], "ret_1": [0.1]})
    empty = pl.DataFrame(schema={"security_id": pl.String, "asof": pl.Datetime(time_zone="UTC")})
    with pytest.raises(PointInTimeError, match="empty"):
        require_panel_keys_in_membership(frame, empty)


def _write_cached_gold(root: Path, times: list[datetime], security_ids: list[str]) -> None:
    (root / "gold").mkdir(parents=True, exist_ok=True)
    (root / "silver").mkdir(parents=True, exist_ok=True)
    n = len(security_ids)
    features = pl.DataFrame(
        {
            "security_id": security_ids,
            "event_time": times,
            "available_time": times,
            "feature_set_version": [FEATURE_SET_VERSION] * n,
            "ret_1": [0.1] * n,
        }
    )
    labels = pl.DataFrame(
        {
            "security_id": security_ids,
            "event_time": times,
            "future_ret_1": [0.0] * n,
        }
    )
    features.write_parquet(root / "gold" / "features.parquet")
    labels.write_parquet(root / "gold" / "labels.parquet")


def test_panel_rejects_stale_gold_after_universe_shrinks(tmp_path: Path) -> None:
    times = _times(2)
    root = tmp_path / "lake"
    _write_cached_gold(root, [times[0], times[0]], ["keep", "drop"])
    _membership(["keep", "drop"], [times[0], times[0]]).write_parquet(
        root / "silver" / "universe.parquet"
    )
    clear_panel_cache()
    cfg = AppConfig.model_validate({"data": {"root": str(root)}})
    loaded = panel(cfg)
    assert set(loaded["security_id"].to_list()) == {"keep", "drop"}
    _membership(["keep"], [times[0]]).write_parquet(root / "silver" / "universe.parquet")
    clear_panel_cache()
    with pytest.raises(PointInTimeError, match="outside PIT universe"):
        panel(cfg)


def test_panel_rejects_gold_when_universe_artifact_missing(tmp_path: Path) -> None:
    t0 = datetime(2020, 1, 2, tzinfo=UTC)
    root = tmp_path / "lake"
    _write_cached_gold(root, [t0], ["A"])
    clear_panel_cache()
    cfg = AppConfig.model_validate({"data": {"root": str(root)}})
    with pytest.raises(PointInTimeError, match="universe.parquet is required"):
        panel(cfg)


def test_late_universe_asof_cannot_validate_earlier_gold_bar(tmp_path: Path) -> None:
    times = _times(2)
    root = tmp_path / "lake"
    _write_cached_gold(root, times, ["A", "A"])
    _membership(["A"], [times[1]]).write_parquet(root / "silver" / "universe.parquet")
    clear_panel_cache()
    cfg = AppConfig.model_validate({"data": {"root": str(root)}})
    with pytest.raises(PointInTimeError, match="outside PIT universe"):
        panel(cfg)
