"""Contracts for reality_sweep.fetch_yahoo_panel + prepare_bars.

The fetch path must fail closed: HTTP/parse errors abort the run rather than
substituting synthetic prices; insufficient survivors abort; duplicates abort.
Adapters are stubbed — no network.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest

import quant_fund.data.adapters.yahoo_eod as yahoo
from quant_fund.research.reality_sweep import fetch_yahoo_panel, prepare_bars

_EARLY = date(2014, 6, 30)
_LATE = date(2024, 6, 28)


def _frame(symbol: str, first: date, last: date, source: str = "test") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "security_id": [symbol, symbol],
            "event_time": [
                datetime(first.year, first.month, first.day, tzinfo=UTC),
                datetime(last.year, last.month, last.day, tzinfo=UTC),
            ],
            "open": [10.0, 11.0],
            "high": [10.5, 11.5],
            "low": [9.5, 10.5],
            "close": [10.2, 11.2],
            "volume": [1_000.0, 1_200.0],
            "source": [source, source],
        }
    )


def _spec(symbols: list[str]) -> dict[str, Any]:
    return {
        "data": {
            "fetch_start_utc": "2014-01-01T00:00:00",
            "fetch_end_utc": "2024-12-31T00:00:00",
            "symbols": symbols,
        }
    }


def _install_stub(monkeypatch: pytest.MonkeyPatch, frames: dict[str, pl.DataFrame]) -> None:
    def fake_fetch(symbol: str, **kwargs: Any) -> dict[str, Any]:
        if symbol not in frames:
            raise ConnectionError(f"no coverage for {symbol}")
        return {"symbol": symbol}

    def fake_parse(payload: dict[str, Any], security_id: str, yahoo_symbol: str) -> pl.DataFrame:
        return frames[security_id]

    monkeypatch.setattr(yahoo, "fetch_yahoo_chart", fake_fetch)
    monkeypatch.setattr(yahoo, "parse_yahoo_chart", fake_parse)
    # The real fetch sleeps 0.35s between symbols.
    monkeypatch.setattr("time.sleep", lambda *_a: None)


def _good_frames(n: int = 8) -> dict[str, pl.DataFrame]:
    return {f"SYM{i}": _frame(f"SYM{i}", date(2010, 1, 4), date(2024, 12, 30)) for i in range(n)}


def test_happy_path_seals_parquet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frames = _good_frames(8)
    _install_stub(monkeypatch, frames)
    panel, meta = fetch_yahoo_panel(_spec(sorted(frames)), tmp_path / "cache")
    assert meta["symbols_included"] == sorted(frames)
    assert meta["symbols_excluded"] == []
    assert meta["n_bars"] == panel.height
    parquet = Path(meta["parquet_path"])
    assert parquet.is_file()
    import hashlib

    assert meta["dataset_sha256"] == hashlib.sha256(parquet.read_bytes()).hexdigest()


def test_short_history_symbol_excluded_but_run_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = _good_frames(8)
    frames["LATE"] = _frame("LATE", date(2020, 1, 1), date(2024, 12, 30))
    _install_stub(monkeypatch, frames)
    panel, meta = fetch_yahoo_panel(_spec(sorted(frames)), tmp_path / "cache")
    excluded = [row["symbol"] for row in meta["symbols_excluded"]]
    assert excluded == ["LATE"]
    assert "LATE" not in set(panel["security_id"].unique())


def test_fewer_than_eight_survivors_aborts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frames = _good_frames(7)
    _install_stub(monkeypatch, frames)
    with pytest.raises(RuntimeError, match="fewer than 8"):
        fetch_yahoo_panel(_spec(sorted(frames)), tmp_path / "cache")


def test_fetch_errors_abort_without_synthetic_substitute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = _good_frames(8)
    frames.pop("SYM7")  # fetch will raise for it
    _install_stub(monkeypatch, frames)
    with pytest.raises(RuntimeError, match="synthetic substitute"):
        fetch_yahoo_panel(_spec(sorted(frames) + ["SYM7"]), tmp_path / "cache")


def test_duplicate_bar_aborts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frames = _good_frames(8)
    dup = _frame("SYM0", date(2010, 1, 4), date(2024, 12, 30))
    frames["SYM0"] = pl.concat([dup, dup.head(1)])
    _install_stub(monkeypatch, frames)
    with pytest.raises(RuntimeError, match="duplicate"):
        fetch_yahoo_panel(_spec(sorted(frames)), tmp_path / "cache")


def test_prepare_bars_is_causal() -> None:
    bars = pl.DataFrame(
        {
            "security_id": ["A"] * 30,
            "event_time": [datetime(2020, 1, 1, tzinfo=UTC) for _ in range(30)],
            "open": [10.0 + i for i in range(30)],
            "high": [10.5 + i for i in range(30)],
            "low": [9.5 + i for i in range(30)],
            "close": [10.0 + i for i in range(30)],
            "volume": [100.0 + 10 * i for i in range(30)],
            "source": ["test"] * 30,
        }
    )
    out = prepare_bars(bars)
    assert {"security_id", "event_time", "adv", "vol_20", "close_total_return"} <= set(out.columns)
    # adv at row t is the mean dollar volume over rows < t only (shift(1)).
    adv = out["adv"].to_list()
    dvs = [c * v for c, v in zip(bars["close"].to_list(), bars["volume"].to_list(), strict=True)]
    assert adv[5] == pytest.approx(sum(dvs[:5]) / 5)
    # First row has no history: backfilled with its own dollar volume.
    assert adv[0] == pytest.approx(dvs[0])
