from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import UTC, date, datetime, time
from pathlib import Path

import pytest
from quant_fund.data.adapters.binance import (
    BINANCE_SYMBOLS,
    BinanceArchiveSpec,
    normalize_binance_archives,
)


def _archive(root: Path, symbol: str, day_text: str, *, close: str = "101.0") -> BinanceArchiveSpec:
    filename = f"{symbol}-1d-{day_text}.zip"
    archive = root / filename
    csv_name = filename.removesuffix(".zip") + ".csv"
    day = date.fromisoformat(day_text)
    opened_ms = int(datetime.combine(day, time.min, tzinfo=UTC).timestamp() * 1000)
    closed_ms = int(
        datetime.combine(day, time.max, tzinfo=UTC).replace(microsecond=999000).timestamp() * 1000
    )
    row = f"{opened_ms},100.0,102.0,99.0,{close},12.5,{closed_ms},1250.0,10,6.0,600.0,0\n"
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr(csv_name, row)
    archive.write_bytes(payload.getvalue())
    checksum = root / f"{filename}.CHECKSUM"
    checksum.write_text(
        f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  {filename}\n", encoding="utf-8"
    )
    return BinanceArchiveSpec(
        symbol=symbol,
        date=day_text,
        archive_path=archive,
        checksum_path=checksum,
        url=f"https://data.binance.vision/data/spot/daily/klines/{symbol}/1d/{filename}",
        revision_id="current-2024-01-01",
    )


def test_normalize_binance_archives_builds_canonical_pit_bars(tmp_path: Path) -> None:
    specs = [_archive(tmp_path, symbol, "2024-01-01") for symbol in BINANCE_SYMBOLS]

    frame = normalize_binance_archives(
        specs,
        ingested_time=datetime(2024, 1, 2, 1, tzinfo=UTC),
    )

    assert frame.columns == [
        "security_id",
        "symbol",
        "event_time",
        "available_time",
        "ingested_time",
        "source",
        "revision_id",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "currency",
        "session",
    ]
    assert frame.height == 5
    assert set(frame["symbol"].to_list()) == set(BINANCE_SYMBOLS)
    assert frame["event_time"][0] == datetime(2024, 1, 1, 23, 59, 59, 999000, tzinfo=UTC)
    assert frame["available_time"][0] == datetime(2024, 1, 2, tzinfo=UTC)
    assert frame["source"][0] == "binance-public-data"
    assert frame["currency"][0] == "USDT"


def test_normalize_uses_archive_date_for_daily_timestamps(tmp_path: Path) -> None:
    specs = [_archive(tmp_path, symbol, "2024-02-29") for symbol in BINANCE_SYMBOLS]

    frame = normalize_binance_archives(
        specs,
        ingested_time=datetime(2024, 3, 1, 1, tzinfo=UTC),
    )

    assert frame["event_time"][0] == datetime(2024, 2, 29, 23, 59, 59, 999000, tzinfo=UTC)
    assert frame["available_time"][0] == datetime(2024, 3, 1, tzinfo=UTC)


def test_normalize_rejects_same_day_availability(tmp_path: Path) -> None:
    specs = [_archive(tmp_path, symbol, "2024-01-01") for symbol in BINANCE_SYMBOLS]

    with pytest.raises(ValueError, match="ingested_time"):
        normalize_binance_archives(
            specs,
            ingested_time=datetime(2024, 1, 1, 23, tzinfo=UTC),
        )


def test_normalize_rejects_archive_checksum_mismatch(tmp_path: Path) -> None:
    specs = [_archive(tmp_path, symbol, "2024-01-01") for symbol in BINANCE_SYMBOLS]
    specs[0].archive_path.write_bytes(specs[0].archive_path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="sha256 mismatch"):
        normalize_binance_archives(
            specs,
            ingested_time=datetime(2024, 1, 2, 1, tzinfo=UTC),
        )


def test_normalize_rejects_malformed_checksum(tmp_path: Path) -> None:
    specs = [_archive(tmp_path, symbol, "2024-01-01") for symbol in BINANCE_SYMBOLS]
    specs[0].checksum_path.write_text("not-a-checksum\n", encoding="utf-8")

    with pytest.raises(ValueError, match="malformed Binance checksum"):
        normalize_binance_archives(
            specs,
            ingested_time=datetime(2024, 1, 2, 1, tzinfo=UTC),
        )


def test_normalize_rejects_unrepresentable_timestamp(tmp_path: Path) -> None:
    specs = [_archive(tmp_path, symbol, "2024-01-01") for symbol in BINANCE_SYMBOLS]
    spec = specs[0]
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr(
            "BTCUSDT-1d-2024-01-01.csv",
            "999999999999999999999,100.0,102.0,99.0,101.0,12.5,1704153599999,1250.0,10,6.0,600.0,0\n",
        )
    spec.archive_path.write_bytes(payload.getvalue())
    spec.checksum_path.write_text(
        f"{hashlib.sha256(spec.archive_path.read_bytes()).hexdigest()}  {spec.archive_path.name}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="timestamp"):
        normalize_binance_archives(
            specs,
            ingested_time=datetime(2024, 1, 2, 1, tzinfo=UTC),
        )


def test_normalize_rejects_nonfinite_auxiliary_volume(tmp_path: Path) -> None:
    specs = [_archive(tmp_path, symbol, "2024-01-01") for symbol in BINANCE_SYMBOLS]
    spec = specs[0]
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr(
            "BTCUSDT-1d-2024-01-01.csv",
            "1704067200000,100.0,102.0,99.0,101.0,12.5,1704153599999,nan,10,6.0,600.0,0\n",
        )
    spec.archive_path.write_bytes(payload.getvalue())
    spec.checksum_path.write_text(
        f"{hashlib.sha256(spec.archive_path.read_bytes()).hexdigest()}  {spec.archive_path.name}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="auxiliary Binance kline"):
        normalize_binance_archives(
            specs,
            ingested_time=datetime(2024, 1, 2, 1, tzinfo=UTC),
        )


def test_normalize_rejects_non_utf8_archive_row(tmp_path: Path) -> None:
    specs = [_archive(tmp_path, symbol, "2024-01-01") for symbol in BINANCE_SYMBOLS]
    spec = specs[0]
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("BTCUSDT-1d-2024-01-01.csv", b"\xff\xfe")
    spec.archive_path.write_bytes(payload.getvalue())
    spec.checksum_path.write_text(
        f"{hashlib.sha256(spec.archive_path.read_bytes()).hexdigest()}  {spec.archive_path.name}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="UTF-8"):
        normalize_binance_archives(
            specs,
            ingested_time=datetime(2024, 1, 2, 1, tzinfo=UTC),
        )


def test_normalize_rejects_noncanonical_symbol_input(tmp_path: Path) -> None:
    specs = [_archive(tmp_path, symbol, "2024-01-01") for symbol in BINANCE_SYMBOLS]
    specs[0] = BinanceArchiveSpec(
        symbol=" btcusdt ",
        date=specs[0].date,
        archive_path=specs[0].archive_path,
        checksum_path=specs[0].checksum_path,
        url=specs[0].url,
        revision_id=specs[0].revision_id,
    )

    with pytest.raises(ValueError, match="canonical Binance symbol"):
        normalize_binance_archives(
            specs,
            ingested_time=datetime(2024, 1, 2, 1, tzinfo=UTC),
        )


def test_normalize_rejects_oversized_zip_member(tmp_path: Path) -> None:
    specs = [_archive(tmp_path, symbol, "2024-01-01") for symbol in BINANCE_SYMBOLS]
    spec = specs[0]
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("BTCUSDT-1d-2024-01-01.csv", "x" * (2 * 1024 * 1024))
    spec.archive_path.write_bytes(payload.getvalue())
    spec.checksum_path.write_text(
        f"{hashlib.sha256(spec.archive_path.read_bytes()).hexdigest()}  {spec.archive_path.name}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="member exceeds maximum size"):
        normalize_binance_archives(
            specs,
            ingested_time=datetime(2024, 1, 2, 1, tzinfo=UTC),
        )


def test_normalize_rejects_csv_field_limit_errors(tmp_path: Path) -> None:
    specs = [_archive(tmp_path, symbol, "2024-01-01") for symbol in BINANCE_SYMBOLS]
    spec = specs[0]
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_STORED) as handle:
        handle.writestr("BTCUSDT-1d-2024-01-01.csv", "x" * 200_000)
    spec.archive_path.write_bytes(payload.getvalue())
    spec.checksum_path.write_text(
        f"{hashlib.sha256(spec.archive_path.read_bytes()).hexdigest()}  {spec.archive_path.name}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="malformed CSV"):
        normalize_binance_archives(
            specs,
            ingested_time=datetime(2024, 1, 2, 1, tzinfo=UTC),
        )


def test_normalize_rejects_mixed_revision_ids(tmp_path: Path) -> None:
    specs = [_archive(tmp_path, symbol, "2024-01-01") for symbol in BINANCE_SYMBOLS]
    specs[1] = BinanceArchiveSpec(
        symbol=specs[1].symbol,
        date=specs[1].date,
        archive_path=specs[1].archive_path,
        checksum_path=specs[1].checksum_path,
        url=specs[1].url,
        revision_id="different-revision",
    )

    with pytest.raises(ValueError, match="same revision_id"):
        normalize_binance_archives(
            specs,
            ingested_time=datetime(2024, 1, 2, 1, tzinfo=UTC),
        )
