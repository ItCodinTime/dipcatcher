"""Lake write-path hardening (P6.3 follow-up).

Same defect class as pooled_stream: the destination path must never exist
in truncated form, concurrent writers to one rel must not share a tmp file,
and ``rel`` must not escape the lake root.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
import pytest

from quant_fund.data.lake import Lake


def _frame(tag: int) -> pl.DataFrame:
    return pl.DataFrame({"security_id": [f"S{tag}"], "close": [float(tag)]})


def test_write_parquet_rejects_path_traversal(tmp_path: Path) -> None:
    lake = Lake(tmp_path)
    for bad in ("../outside.parquet", "/abs/x.parquet", "a/../b.parquet", "x\\y.parquet", ""):
        with pytest.raises(ValueError, match="clean relative path"):
            lake.write_parquet(_frame(1), bad)
    assert not (tmp_path / "outside.parquet").exists()


def test_read_and_exists_reject_traversal(tmp_path: Path) -> None:
    lake = Lake(tmp_path)
    with pytest.raises(ValueError):
        lake.read_parquet("../secrets.parquet")
    with pytest.raises(ValueError):
        lake.exists("..")


def test_write_parquet_leaves_no_tmp_litter(tmp_path: Path) -> None:
    lake = Lake(tmp_path)
    lake.write_parquet(_frame(1), "silver/bars.parquet")
    litter = list((tmp_path / "silver").glob("*.tmp")) + list((tmp_path / "silver").glob(".*.tmp"))
    assert litter == []
    back = lake.read_parquet("silver/bars.parquet")
    assert back["close"].to_list() == [1.0]


def test_write_parquet_failed_write_leaves_prior_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lake = Lake(tmp_path)
    lake.write_parquet(_frame(7), "silver/bars.parquet")

    def crash_write(self: pl.DataFrame, target: Path) -> None:
        # Partial bytes at the tmp path, then die — the torn tmp must be
        # cleaned up and must never reach the canonical path.
        Path(target).write_bytes(b"\x50\x41\x52\x31truncated")
        raise OSError("simulated mid-write crash")

    monkeypatch.setattr(pl.DataFrame, "write_parquet", crash_write)
    with pytest.raises(OSError, match="simulated"):
        lake.write_parquet(_frame(9), "silver/bars.parquet")
    assert lake.read_parquet("silver/bars.parquet")["close"].to_list() == [7.0]
    assert not list((tmp_path / "silver").glob("*.tmp"))


def test_concurrent_writers_same_rel_never_share_tmp(tmp_path: Path) -> None:
    """Two threads racing one destination must each write a private tmp file —
    a shared `<name>.tmp` lets one writer rename the other's torn output."""
    lake = Lake(tmp_path)
    rel = "silver/raced.parquet"
    barrier = threading.Barrier(8)

    def write(tag: int) -> None:
        barrier.wait()
        lake.write_parquet(_frame(tag), rel)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, range(8)))

    # Final file must be one complete frame — never a mix of two tmp writes.
    result = lake.read_parquet(rel)
    assert result.height == 1
    assert int(result["close"][0]) in range(8)
    assert result["security_id"][0] == f"S{int(result['close'][0])}"
    assert not list((tmp_path / "silver").glob("*.tmp"))
