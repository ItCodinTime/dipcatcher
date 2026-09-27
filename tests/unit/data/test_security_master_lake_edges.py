"""Wave 38: FrameSecurityMaster + Lake edge fixtures — empty/missing fail-closed.

Research/infrastructure only — no live broker / vendor MD.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from quant_fund.data.lake import Lake
from quant_fund.data.security_master import (
    FrameSecurityMaster,
    attach_master_attributes,
    snapshot_asof,
)
from quant_fund.schemas.errors import PointInTimeError

RESEARCH_ONLY = True
LIVE_PNL_CLAIM = False


def _master_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "security_id": ["SEC_A", "SEC_A", "SEC_B"],
            "ticker": ["AAA", "AAA", "BBB"],
            "valid_from": [
                datetime(2020, 1, 1, tzinfo=UTC),
                datetime(2021, 6, 1, tzinfo=UTC),
                datetime(2020, 1, 1, tzinfo=UTC),
            ],
            "valid_to": [
                datetime(2021, 6, 1, tzinfo=UTC),
                None,
                None,
            ],
            "sector": ["tech", "tech", "health"],
        }
    )


def test_empty_master_asof_and_record_none() -> None:
    empty = pl.DataFrame(
        schema={
            "security_id": pl.String,
            "ticker": pl.String,
            "valid_from": pl.Datetime("us", "UTC"),
            "valid_to": pl.Datetime("us", "UTC"),
        }
    )
    sm = FrameSecurityMaster(empty)
    when = datetime(2020, 6, 1, tzinfo=UTC)
    assert sm.asof(when, "AAA") is None
    assert sm.record("SEC_A", when) is None
    assert sm.frame().is_empty()


def test_missing_ticker_and_security_id_return_none() -> None:
    sm = FrameSecurityMaster(_master_frame())
    when = datetime(2020, 6, 1, tzinfo=UTC)
    assert sm.asof(when, "ZZZ") is None
    assert sm.record("SEC_Z", when) is None


def test_asof_respects_valid_window() -> None:
    sm = FrameSecurityMaster(_master_frame())
    # Before first window
    assert sm.asof(datetime(2019, 12, 31, tzinfo=UTC), "AAA") is None
    # Inside first window (valid_to exclusive)
    assert sm.asof(datetime(2020, 6, 1, tzinfo=UTC), "AAA") == "SEC_A"
    # At boundary valid_to → second open-ended row
    assert sm.asof(datetime(2021, 6, 1, tzinfo=UTC), "AAA") == "SEC_A"
    # Open-ended BBB
    assert sm.asof(datetime(2025, 1, 1, tzinfo=UTC), "BBB") == "SEC_B"


def test_asof_ignores_late_available_restatement() -> None:
    """A restated mapping cannot rewrite ticker identity before available_time."""
    sm = FrameSecurityMaster(
        pl.DataFrame(
            {
                "security_id": ["SEC_OLD", "SEC_NEW"],
                "ticker": ["AAA", "AAA"],
                "valid_from": [
                    datetime(2020, 1, 1, tzinfo=UTC),
                    datetime(2020, 6, 1, tzinfo=UTC),
                ],
                "valid_to": [None, None],
                "available_time": [
                    datetime(2020, 1, 1, tzinfo=UTC),
                    datetime(2021, 1, 1, tzinfo=UTC),
                ],
            }
        )
    )
    assert sm.asof(datetime(2020, 8, 1, tzinfo=UTC), "AAA") == "SEC_OLD"
    assert sm.asof(datetime(2021, 6, 1, tzinfo=UTC), "AAA") == "SEC_NEW"
    assert sm.record("SEC_NEW", datetime(2020, 8, 1, tzinfo=UTC)) is None
    hit = sm.record("SEC_NEW", datetime(2021, 6, 1, tzinfo=UTC))
    assert hit is not None
    assert hit["security_id"] == "SEC_NEW"


def test_snapshot_asof_excludes_unobservable_vintages() -> None:
    master = pl.DataFrame(
        {
            "security_id": ["A", "A"],
            "ticker": ["AAA", "AAA"],
            "valid_from": [
                datetime(2020, 1, 1, tzinfo=UTC),
                datetime(2020, 6, 1, tzinfo=UTC),
            ],
            "valid_to": [None, None],
            "available_time": [
                datetime(2020, 1, 1, tzinfo=UTC),
                datetime(2021, 1, 1, tzinfo=UTC),
            ],
            "sector": ["tech", "health"],
        }
    )
    early = snapshot_asof(master, datetime(2020, 8, 1, tzinfo=UTC))
    assert early.height == 1
    assert early["sector"].to_list() == ["tech"]
    late = snapshot_asof(master, datetime(2021, 6, 1, tzinfo=UTC))
    assert late.height == 1
    assert late["sector"].to_list() == ["health"]


def test_attach_master_attributes_is_pit_and_does_not_explode() -> None:
    t0 = datetime(2020, 1, 2, tzinfo=UTC)
    t1 = datetime(2020, 8, 1, tzinfo=UTC)
    t2 = datetime(2021, 6, 1, tzinfo=UTC)
    bars = pl.DataFrame(
        {
            "security_id": ["A", "A", "A"],
            "event_time": [t0, t1, t2],
            "available_time": [t0, t1, t2],
            "close": [10.0, 11.0, 12.0],
        }
    )
    master = pl.DataFrame(
        {
            "security_id": ["A", "A"],
            "ticker": ["AAA", "AAA"],
            "valid_from": [
                datetime(2020, 1, 1, tzinfo=UTC),
                datetime(2020, 6, 1, tzinfo=UTC),
            ],
            "valid_to": [None, None],
            "available_time": [
                datetime(2020, 1, 1, tzinfo=UTC),
                datetime(2021, 1, 1, tzinfo=UTC),
            ],
            "sector": ["tech", "health"],
            "industry": ["software", "biotech"],
            "exchange": ["XNYS", "XNYS"],
        }
    )
    out = attach_master_attributes(bars, master).sort("event_time")
    assert out.height == 3
    assert out["sector"].to_list() == ["tech", "tech", "health"]


def test_attach_master_attributes_duplicate_identity_fail_closed() -> None:
    bars = pl.DataFrame(
        {
            "security_id": ["A"],
            "event_time": [datetime(2020, 1, 2, tzinfo=UTC)],
        }
    )
    master = pl.DataFrame(
        {
            "security_id": ["A", "A"],
            "sector": ["tech", "health"],
            "industry": ["i", "i"],
            "exchange": ["XNYS", "XNYS"],
        }
    )
    with pytest.raises(PointInTimeError, match="duplicate security_id"):
        attach_master_attributes(bars, master)


def test_record_returns_row_dict_or_none() -> None:
    sm = FrameSecurityMaster(_master_frame())
    hit = sm.record("SEC_B", datetime(2020, 6, 1, tzinfo=UTC))
    assert hit is not None
    assert hit["security_id"] == "SEC_B"
    assert hit["ticker"] == "BBB"
    assert hit["sector"] == "health"
    # After first AAA window closed
    assert sm.record("SEC_A", datetime(2021, 7, 1, tzinfo=UTC)) is not None
    # Before valid_from
    assert sm.record("SEC_B", datetime(2019, 1, 1, tzinfo=UTC)) is None


def test_lake_creates_layer_dirs(tmp_path: Path) -> None:
    lake = Lake(tmp_path / "lake")
    for part in ("raw", "bronze", "silver", "gold", "metadata"):
        assert (lake.root / part).is_dir()


def test_lake_write_read_roundtrip(tmp_path: Path) -> None:
    lake = Lake(tmp_path / "lake")
    frame = pl.DataFrame({"security_id": ["A", "B"], "x": [1.0, 2.0]})
    path = lake.write_parquet(frame, "bronze/demo.parquet")
    assert path.is_file()
    assert lake.exists("bronze/demo.parquet")
    got = lake.read_parquet("bronze/demo.parquet")
    assert got.height == 2
    assert got["security_id"].to_list() == ["A", "B"]


def test_lake_read_missing_fail_closed(tmp_path: Path) -> None:
    lake = Lake(tmp_path / "lake")
    assert not lake.exists("bronze/missing.parquet")
    with pytest.raises(FileNotFoundError):
        lake.read_parquet("bronze/missing.parquet")


def test_lake_nested_write_creates_parent(tmp_path: Path) -> None:
    lake = Lake(tmp_path / "lake")
    path = lake.write_parquet(
        pl.DataFrame({"a": [1]}),
        "silver/nested/deep/bars.parquet",
    )
    assert path.is_file()
    assert lake.exists("silver/nested/deep/bars.parquet")
