"""Unit tests for the ``quant pit`` CLI (DESIGN.md §4.5, §12 W1)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from typer.testing import CliRunner

from quant_fund.pit import PitVault
from quant_fund.pit.cli import pit_app

T0 = datetime(2024, 1, 1, tzinfo=UTC)
runner = CliRunner()


def _seed(root) -> None:
    vault = PitVault(root)
    vault.create_dataset("silver/bars")
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


def test_pit_init(tmp_path) -> None:
    root = tmp_path / "pit"
    result = runner.invoke(pit_app, ["init", "--root", str(root)])
    assert result.exit_code == 0
    assert root.is_dir()


def test_pit_verify_clean(tmp_path) -> None:
    root = tmp_path / "pit"
    _seed(root)
    result = runner.invoke(pit_app, ["verify", "--root", str(root)])
    assert result.exit_code == 0, result.output
    assert "OK silver/bars" in result.output


def test_pit_verify_corrupt_exits_1(tmp_path) -> None:
    root = tmp_path / "pit"
    _seed(root)
    part = root / "silver/bars/parts/r0000001.parquet"
    with open(part, "ab") as handle:
        handle.write(b"bitrot")
    result = runner.invoke(pit_app, ["verify", "--root", str(root)])
    assert result.exit_code == 1
    assert "VIOLATION silver/bars" in result.output


def test_pit_verify_dataset_filter(tmp_path) -> None:
    root = tmp_path / "pit"
    _seed(root)
    result = runner.invoke(pit_app, ["verify", "--root", str(root), "--dataset", "silver/bars"])
    assert result.exit_code == 0, result.output


def test_pit_verify_empty_root_exits_1(tmp_path) -> None:
    root = tmp_path / "pit"
    root.mkdir()
    result = runner.invoke(pit_app, ["verify", "--root", str(root)])
    assert result.exit_code == 1


def test_pit_stats(tmp_path) -> None:
    root = tmp_path / "pit"
    _seed(root)
    result = runner.invoke(pit_app, ["stats", "--root", str(root)])
    assert result.exit_code == 0, result.output
    assert "silver/bars: rows=2 revisions=1 files=1" in result.output
    assert "status=ok" in result.output


def test_pit_restate(tmp_path) -> None:
    root = tmp_path / "pit"
    _seed(root)
    correction = tmp_path / "correction.parquet"
    pl.DataFrame({"security_id": ["A"], "event_time": [T0], "close": [101.5]}).write_parquet(
        correction
    )
    known_at = (T0 + timedelta(days=5)).isoformat()
    result = runner.invoke(
        pit_app,
        [
            "restate",
            "--root",
            str(root),
            "--dataset",
            "silver/bars",
            "--parquet",
            str(correction),
            "--known-at",
            known_at,
        ],
    )
    assert result.exit_code == 0, result.output
    assert "revision=2" in result.output
    out = PitVault(root).asof("silver/bars", T0 + timedelta(days=6))
    assert out.frame.filter(pl.col("security_id") == "A")["close"].to_list() == [101.5]


def test_pit_restate_bad_timestamp_exits_2(tmp_path) -> None:
    root = tmp_path / "pit"
    _seed(root)
    correction = tmp_path / "correction.parquet"
    pl.DataFrame({"security_id": ["A"], "event_time": [T0], "close": [101.5]}).write_parquet(
        correction
    )
    result = runner.invoke(
        pit_app,
        [
            "restate",
            "--root",
            str(root),
            "--dataset",
            "silver/bars",
            "--parquet",
            str(correction),
            "--known-at",
            "not-a-date",
        ],
    )
    assert result.exit_code == 2


def test_pit_app_commands_registered() -> None:
    """The sub-typer itself carries the §4.5 command set."""
    names = {cmd.name for cmd in pit_app.registered_commands}
    assert names == {"init", "verify", "stats", "restate"}


def test_quant_app_mounts_pit() -> None:
    """The pit sub-typer is mounted on the main quant app (§9.2 glue).

    The main app eagerly imports the legacy CLI stack (pandas/sklearn via
    quant_models); in the minimal PROOFCORE dev env this test skips, in the
    full repo env it runs for real.
    """
    try:
        from quant_fund.cli._app import app
    except ModuleNotFoundError as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"full CLI stack unavailable: {exc}")
    names = {group.name for group in app.registered_groups}
    assert "pit" in names
