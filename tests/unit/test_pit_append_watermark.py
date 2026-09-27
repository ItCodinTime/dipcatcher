"""PIT append backdated-known_at watermark policy tests (ADVERSARIAL §1b-W4).

Default posture is warn-only; datasets created with
``monotonic_known_at=True`` reject backdated appends (fail closed).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quant_fund.pit import PitVault, VaultError

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _frame(known_at: datetime) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "event_time": [known_at - timedelta(hours=16)],
            "known_at": [known_at],
            "security_id": ["AAA"],
            "close": [100.0],
        }
    ).with_columns(
        pl.col("event_time").cast(pl.Datetime("us", "UTC")),
        pl.col("known_at").cast(pl.Datetime("us", "UTC")),
    )


def test_backdated_append_warns_by_default(tmp_path, caplog) -> None:
    vault = PitVault(tmp_path)
    vault.create_dataset("silver/bars")
    vault.append("silver/bars", _frame(T0 + timedelta(days=10)))
    with caplog.at_level(logging.WARNING, logger="quant_fund.pit.vault"):
        vault.append("silver/bars", _frame(T0))  # backdated known_at
    assert any("backdated append" in record.message for record in caplog.records)
    # warn-only: the append landed (two distinct event_time keys visible)
    assert len(vault.list_datasets()) == 1
    frame = vault.asof("silver/bars", T0 + timedelta(days=11)).frame
    assert frame.height == 2


def test_backdated_append_rejected_when_opted_in(tmp_path, caplog) -> None:
    vault = PitVault(tmp_path)
    vault.create_dataset("silver/bars", monotonic_known_at=True)
    vault.append("silver/bars", _frame(T0 + timedelta(days=10)))
    with pytest.raises(VaultError, match="backdated append"):
        vault.append("silver/bars", _frame(T0))
    # rejected appends leave no record behind
    assert not any("backdated" in record.message for record in caplog.records)


def test_monotonic_append_passes_with_flag(tmp_path) -> None:
    vault = PitVault(tmp_path)
    vault.create_dataset("silver/bars", monotonic_known_at=True)
    vault.append("silver/bars", _frame(T0))
    vault.append("silver/bars", _frame(T0 + timedelta(days=1)))  # not backdated
    assert vault.asof("silver/bars", T0 + timedelta(days=2)).frame.height == 2
