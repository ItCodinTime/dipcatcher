"""Unit tests for bitemporal restatement semantics (DESIGN.md §4.2, §12 W1).

Corrections are appends with a new known_at; old versions are retained
forever; a restatement is inert before its publication time.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quant_fund.pit import PitVault, RestatementPolicy, VaultError
from quant_fund.pit.corrections import prepare_correction
from quant_fund.pit.manifest import sha256_file

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _seed(vault: PitVault) -> None:
    vault.append(
        "silver/bars",
        pl.DataFrame(
            {
                "security_id": ["A", "B"],
                "event_time": [T0, T0],
                "known_at": [T0, T0],
                "close": [100.0, 50.0],
            }
        ),
    )


def test_restatement_inert_before_publication(tmp_path) -> None:
    vault = PitVault(tmp_path / "pit")
    vault.create_dataset("silver/bars")
    _seed(vault)
    before = vault.asof("silver/bars", T0 + timedelta(days=1))
    vault.restate(
        "silver/bars",
        pl.DataFrame({"security_id": ["A"], "event_time": [T0], "close": [999.0]}),
        known_at=T0 + timedelta(days=10),
    )
    after = vault.asof("silver/bars", T0 + timedelta(days=1))
    # Identical content, byte-for-byte identical canonical hash.
    assert after.content_sha256 == before.content_sha256
    assert after.frame.equals(before.frame)
    published = vault.asof("silver/bars", T0 + timedelta(days=10))
    assert published.frame.filter(pl.col("security_id") == "A")["close"].to_list() == [999.0]
    # The other key is untouched by the restatement.
    assert published.frame.filter(pl.col("security_id") == "B")["close"].to_list() == [50.0]


def test_restate_keeps_old_parts_untouched(tmp_path) -> None:
    vault = PitVault(tmp_path / "pit")
    vault.create_dataset("silver/bars")
    _seed(vault)
    part1 = vault.root / "silver/bars/parts/r0000001.parquet"
    sha_before = sha256_file(part1)
    vault.restate(
        "silver/bars",
        pl.DataFrame({"security_id": ["A"], "event_time": [T0], "close": [999.0]}),
        known_at=T0 + timedelta(days=10),
    )
    assert sha256_file(part1) == sha_before
    assert (vault.root / "silver/bars/parts/r0000002.parquet").is_file()
    hist = vault.history("silver/bars", ("A", T0))
    assert hist["close"].to_list() == [100.0, 999.0]
    assert vault.verify("silver/bars") == []


def test_restate_naive_known_at_refused(tmp_path) -> None:
    vault = PitVault(tmp_path / "pit")
    vault.create_dataset("silver/bars")
    _seed(vault)
    with pytest.raises(VaultError, match="timezone-aware"):
        vault.restate(
            "silver/bars",
            pl.DataFrame({"security_id": ["A"], "event_time": [T0], "close": [1.0]}),
            known_at=datetime(2024, 1, 2),
        )


def test_prepare_correction_overrides_known_at_column() -> None:
    frame = pl.DataFrame(
        {
            "security_id": ["A"],
            "event_time": [T0],
            "known_at": [T0],
            "close": [1.0],
        }
    )
    stamped = prepare_correction(frame, known_at=T0 + timedelta(days=3))
    assert stamped["known_at"].to_list() == [T0 + timedelta(days=3)]


def test_strict_first_proves_restatement_inert(tmp_path) -> None:
    """STRICT_FIRST reads the earliest version even after publication (§4.2)."""
    vault = PitVault(tmp_path / "pit")
    vault.create_dataset("silver/bars")
    _seed(vault)
    vault.restate(
        "silver/bars",
        pl.DataFrame({"security_id": ["A"], "event_time": [T0], "close": [999.0]}),
        known_at=T0 + timedelta(days=10),
    )
    strict = vault.asof(
        "silver/bars", T0 + timedelta(days=30), policy=RestatementPolicy.STRICT_FIRST
    )
    assert strict.frame.filter(pl.col("security_id") == "A")["close"].to_list() == [100.0]
    latest = vault.asof("silver/bars", T0 + timedelta(days=30))
    assert latest.frame.filter(pl.col("security_id") == "A")["close"].to_list() == [999.0]


def test_multiple_restatements_chain_visibility(tmp_path) -> None:
    vault = PitVault(tmp_path / "pit")
    vault.create_dataset("silver/bars")
    _seed(vault)
    for day, value in ((5, 101.0), (10, 102.0), (15, 103.0)):
        vault.restate(
            "silver/bars",
            pl.DataFrame({"security_id": ["A"], "event_time": [T0], "close": [value]}),
            known_at=T0 + timedelta(days=day),
        )
    expectations = {1: 100.0, 5: 101.0, 7: 101.0, 12: 102.0, 30: 103.0}
    for day, expected in expectations.items():
        out = vault.asof("silver/bars", T0 + timedelta(days=day))
        got = out.frame.filter(pl.col("security_id") == "A")["close"].to_list()
        assert got == [expected], f"day {day}: {got} != [{expected}]"
    assert vault.history("silver/bars", ("A", T0)).height == 4
