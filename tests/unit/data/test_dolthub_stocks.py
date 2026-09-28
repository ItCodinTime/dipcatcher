"""Offline tests for the DoltHub as-traded loader and membership coverage."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.config.models import DataConfig
from quant_fund.data.adapters.dolthub_stocks import (
    REVISION_ID,
    SOURCE_NAME,
    DolthubSqlClient,
    DolthubStocksSource,
    cache_paths,
    fetch_ohlcv_on_dates,
    load_ohlcv_on_dates,
    normalize_dolthub_ohlcv,
    panel_fingerprint,
    parse_symbols,
    read_cached_symbol,
    session_close,
    write_cached_symbol,
)
from quant_fund.data.index_membership import (
    load_membership,
    members_asof,
    membership_price_coverage,
    merge_bar_panels,
)
from quant_fund.data.sources.base import SourceError
from quant_fund.data.sources.registry import get_source, source_names
from quant_fund.proofcore.contracts import sha256_hex_bytes


class FakeSqlHttp:
    """Records SQL URLs and returns queued DoltHub-shaped payloads."""

    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = list(payloads)
        self.urls: list[str] = []

    def get_json(self, url: str, **_: object) -> dict:
        self.urls.append(url)
        if not self.payloads:
            raise AssertionError(f"unexpected SQL request: {url}")
        return self.payloads.pop(0)


def _success(rows: list[dict]) -> dict:
    return {
        "query_execution_status": "Success",
        "query_execution_message": "",
        "rows": rows,
    }


def test_parse_symbols_keeps_share_class_dot() -> None:
    assert parse_symbols("brk.b,BF.B, meta") == ["BRK.B", "BF.B", "META"]
    with pytest.raises(ValueError, match="unsafe"):
        parse_symbols("AA/BB")


def test_session_close_is_1600_et() -> None:
    stamp = session_close(date(2020, 1, 2))
    assert stamp == datetime(2020, 1, 2, 21, 0, tzinfo=UTC)


def test_normalize_maps_as_traded_rows() -> None:
    rows = [
        {
            "security_id": "SIVB",
            "symbol": "SIVB",
            "event_time": session_close(date(2020, 1, 2)),
            "available_time": session_close(date(2020, 1, 2)),
            "ingested_time": datetime(2026, 1, 1, tzinfo=UTC),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000.0,
        }
    ]
    frame = normalize_dolthub_ohlcv(rows)
    assert frame.height == 1
    assert frame["source"][0] == SOURCE_NAME
    assert frame["revision_id"][0] == REVISION_ID
    assert frame["session"][0] == "rth"


def test_cache_round_trip_checks_provenance(tmp_path: Path) -> None:
    rows = [
        {
            "security_id": "SIVB",
            "symbol": "SIVB",
            "event_time": session_close(date(2020, 1, 2)),
            "available_time": session_close(date(2020, 1, 2)),
            "ingested_time": datetime(2026, 1, 1, tzinfo=UTC),
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.5,
            "volume": 50.0,
        }
    ]
    frame = normalize_dolthub_ohlcv(rows)
    paths = write_cached_symbol(
        frame,
        cache_root=tmp_path,
        commit="abc123",
        symbol="SIVB",
        provenance={"test": True},
    )
    assert paths.bars.is_file() and paths.receipt.is_file()
    receipt = json.loads(paths.receipt.read_text(encoding="utf-8"))
    assert receipt["sha256"] == sha256_hex_bytes(paths.bars.read_bytes())
    assert receipt["dolt_commit"] == "abc123"
    assert receipt["provenance"]["test"] is True
    loaded = read_cached_symbol(tmp_path, "abc123", "SIVB")
    assert loaded is not None
    assert loaded.height == 1
    # Tamper with bytes → provenance fail-closed.
    paths.bars.write_bytes(b"not-parquet")
    with pytest.raises(SourceError, match="provenance mismatch"):
        read_cached_symbol(tmp_path, "abc123", "SIVB")


def test_fetch_on_dates_batches_sql() -> None:
    client = FakeSqlHttp(
        [
            _success(
                [
                    {
                        "date": "2020-01-02",
                        "act_symbol": "SIVB",
                        "open": "1",
                        "high": "2",
                        "low": "0.5",
                        "close": "1.5",
                        "volume": "10",
                    },
                    {
                        "date": "2020-01-02",
                        "act_symbol": "FRC",
                        "open": "3",
                        "high": "4",
                        "low": "2",
                        "close": "3.5",
                        "volume": "20",
                    },
                ]
            )
        ]
    )
    sql = DolthubSqlClient(client=client)  # type: ignore[arg-type]
    rows = fetch_ohlcv_on_dates(sql, ["SIVB", "FRC"], [date(2020, 1, 2)])
    assert {row["security_id"] for row in rows} == {"SIVB", "FRC"}
    from urllib.parse import unquote_plus

    decoded = unquote_plus(client.urls[0])
    assert "act_symbol IN ('SIVB', 'FRC')" in decoded
    assert "date IN ('2020-01-02')" in decoded


def test_load_ohlcv_on_dates_uses_cache(tmp_path: Path) -> None:
    rows = [
        {
            "security_id": "TWTR",
            "symbol": "TWTR",
            "event_time": session_close(date(2020, 1, 2)),
            "available_time": session_close(date(2020, 1, 2)),
            "ingested_time": datetime(2026, 1, 1, tzinfo=UTC),
            "open": 40.0,
            "high": 41.0,
            "low": 39.0,
            "close": 40.5,
            "volume": 100.0,
        }
    ]
    frame = normalize_dolthub_ohlcv(rows)
    write_cached_symbol(
        frame,
        cache_root=tmp_path,
        commit="cafebabe",
        symbol="TWTR",
        provenance={},
    )

    class Boom:
        def get_json(self, url: str, **_: object) -> dict:
            raise AssertionError(f"network should not be hit: {url}")

    sql = DolthubSqlClient(client=Boom())  # type: ignore[arg-type]
    result = load_ohlcv_on_dates(
        ["TWTR"],
        ["2020-01-02"],
        cache_dir=tmp_path,
        sql_client=sql,
        commit="cafebabe",
        allow_download=False,
    )
    assert result.frame.height == 1
    assert result.dolt_commit == "cafebabe"
    assert panel_fingerprint(result.frame) == panel_fingerprint(frame)


def test_source_registered_and_config_accepts() -> None:
    assert "dolthub_stocks" in source_names()
    assert get_source("dolthub").name == SOURCE_NAME
    assert get_source("post-no-preference").name == SOURCE_NAME
    assert DataConfig.model_validate({"source": "dolthub_stocks"}).source == "dolthub_stocks"


def test_members_asof_and_coverage_report() -> None:
    membership = {
        "current": ["AAA", "BBB"],
        "changes": [
            {"date": "2020-01-03", "added": "BBB", "removed": "CCC"},
            {"date": "2020-01-01", "added": "AAA", "removed": ""},
        ],
    }
    assert members_asof(membership, "2020-01-02") == {"AAA", "CCC"}
    bars = pl.DataFrame(
        {
            "security_id": ["AAA", "CCC"],
            "event_time": [
                datetime(2020, 1, 2, 21, tzinfo=UTC),
                datetime(2020, 1, 2, 21, tzinfo=UTC),
            ],
            "close": [1.0, 2.0],
        }
    )
    report = membership_price_coverage(bars, membership, ["2020-01-02"])
    assert report[0]["members"] == 2
    assert report[0]["with_bar"] == 2
    assert report[0]["coverage"] == pytest.approx(1.0)
    assert report[0]["missing"] == []


def test_coverage_counts_partial_panel() -> None:
    membership = {"current": ["AAA", "BBB", "CCC"], "changes": []}
    bars = pl.DataFrame(
        {
            "security_id": ["AAA", "BBB"],
            "event_time": [datetime(2023, 1, 3, 21, tzinfo=UTC)] * 2,
            "close": [1.0, 2.0],
        }
    )
    report = membership_price_coverage(bars, membership, ["2023-01-03"])[0]
    assert report["with_bar"] == 2
    assert report["members"] == 3
    assert report["coverage"] == pytest.approx(2 / 3)
    assert report["missing"] == ["CCC"]


def test_merge_bar_panels_dedupes() -> None:
    left = pl.DataFrame(
        {
            "security_id": ["AAA"],
            "event_time": [datetime(2020, 1, 2, 21, tzinfo=UTC)],
            "close": [1.0],
            "source": ["yahoo"],
        }
    )
    right = pl.DataFrame(
        {
            "security_id": ["AAA"],
            "event_time": [datetime(2020, 1, 2, 21, tzinfo=UTC)],
            "close": [9.0],
            "source": ["dolthub_stocks"],
        }
    )
    merged = merge_bar_panels([left, right])
    assert merged.height == 1
    assert float(merged["close"][0]) == 9.0


def test_membership_coverage_cli(tmp_path: Path) -> None:
    membership = {
        "current": ["AAA", "BBB"],
        "changes": [],
    }
    member_path = tmp_path / "membership.json"
    member_path.write_text(json.dumps(membership), encoding="utf-8")
    bars = pl.DataFrame(
        {
            "security_id": ["AAA"],
            "event_time": [datetime(2016, 1, 4, 21, tzinfo=UTC)],
            "close": [1.0],
        }
    )
    bars_path = tmp_path / "bars.parquet"
    bars.write_parquet(bars_path)
    out = tmp_path / "coverage.json"
    result = CliRunner().invoke(
        app,
        [
            "membership-coverage",
            "--membership",
            str(member_path),
            "--bars",
            str(bars_path),
            "--anchor",
            "2016-01-04",
            "--label",
            "yahoo_only",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "coverage" in result.output
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["coverage"][0]["with_bar"] == 1
    assert payload["coverage"][0]["members"] == 2
    assert payload["sources"] == ["yahoo_only"]


def test_vendored_membership_loads_when_present() -> None:
    path = Path("research/reality/survivorship/membership.json")
    if not path.is_file():
        pytest.skip("membership snapshot not vendored in this checkout")
    data = load_membership(path)
    assert "SIVB" in members_asof(data, "2023-03-14")
    assert "SIVB" not in members_asof(data, "2023-03-15")


def test_source_adapter_dates_path(tmp_path: Path) -> None:
    http = FakeSqlHttp(
        [
            _success([{"h": "deadbeef"}]),
            _success(
                [
                    {
                        "date": "2020-01-02",
                        "act_symbol": "SIVB",
                        "open": "1",
                        "high": "1",
                        "low": "1",
                        "close": "1",
                        "volume": "1",
                    }
                ]
            ),
        ]
    )
    source = DolthubStocksSource(client=http)  # type: ignore[arg-type]
    frame = source.fetch(
        symbols="SIVB",
        dates="2020-01-02",
        cache_dir=tmp_path,
        allow_download=True,
    )
    assert frame.height == 1
    assert cache_paths(tmp_path, "deadbeef", "SIVB").receipt.is_file()
