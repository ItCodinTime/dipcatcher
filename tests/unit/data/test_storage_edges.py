"""``data.sources.storage`` edge paths: empty time ranges and torn-write cleanup."""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from quant_fund.data.sources.storage import _time_range, write_source_frame

pytestmark = pytest.mark.synthetic


def test_time_range_missing_column_returns_none() -> None:
    frame = pl.DataFrame({"other": ["2026-01-01"]})
    assert _time_range(frame, "event_time") is None


def test_time_range_empty_frame_returns_none() -> None:
    frame = pl.DataFrame({"event_time": []}).cast({"event_time": pl.String})
    assert _time_range(frame, "event_time") is None


def test_time_range_all_nulls_returns_none() -> None:
    frame = pl.DataFrame({"event_time": [None, None]})
    assert _time_range(frame, "event_time") is None


@pytest.mark.parametrize("source", ["", ".", "..", "a/b", "a\\b"])
def test_unsafe_source_label_rejected(source: str, tmp_path: Any) -> None:
    with pytest.raises(Exception, match="source"):
        write_source_frame(pl.DataFrame({"x": [1]}), tmp_path, source)


def test_absolute_filename_rejected(tmp_path: Any) -> None:
    with pytest.raises(Exception, match="relative"):
        write_source_frame(pl.DataFrame({"x": [1]}), tmp_path, "src", filename="/etc/x.parquet")


def test_filename_escaping_root_rejected(tmp_path: Any) -> None:
    with pytest.raises(Exception, match="escapes"):
        write_source_frame(pl.DataFrame({"x": [1]}), tmp_path, "src", filename="../x.parquet")


def test_non_parquet_suffix_rejected(tmp_path: Any) -> None:
    with pytest.raises(Exception, match="parquet"):
        write_source_frame(pl.DataFrame({"x": [1]}), tmp_path, "src", filename="x.csv")


def test_torn_write_removes_tmp_and_raises(tmp_path: Any) -> None:
    destination = tmp_path / "raw" / "sources" / "badsource.parquet"

    class _ExplodingFrame:
        columns: list[str] = []

        def is_empty(self) -> bool:
            return True

        def write_parquet(self, path: Any) -> None:
            path.write_bytes(b"partial")
            raise OSError("simulated write failure")

    with pytest.raises(OSError, match="simulated write failure"):
        write_source_frame(_ExplodingFrame(), tmp_path, "badsource")  # type: ignore[arg-type]
    assert not destination.exists()
    assert not destination.with_name("badsource.parquet.tmp").exists()
