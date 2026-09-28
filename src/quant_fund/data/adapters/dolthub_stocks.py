"""DoltHub ``post-no-preference/stocks`` as-traded daily bars.

Public SQL API, no API key. The upstream database keeps delisted names
(SIVB, FRC, TWTR, …) with as-traded OHLCV. Bars are cached under
``data/dolthub_stocks/<commit>/`` with a per-file provenance receipt
(sha256 of the parquet bytes + the Dolt commit hash).

``available_time`` is the US RTH session close (16:00 America/New_York),
matching Stooq/Yahoo file-tape convention. Prices are stamped
``revision_id=DOLTHUB_AS_TRADED`` — as received from the public DB, not a
CRSP total-return vintage. Research plumbing only; no broker path.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from quant_fund.data.sources.base import (
    HttpClient,
    SourceAdapter,
    SourceError,
    query_url,
    utc_now,
)
from quant_fund.data.sources.normalize import normalize_ohlcv
from quant_fund.proofcore.contracts import sha256_hex_bytes

OWNER = "post-no-preference"
REPO = "stocks"
DEFAULT_REF = "master"
SOURCE_NAME = "dolthub_stocks"
REVISION_ID = "DOLTHUB_AS_TRADED"
API_ROOT = "https://www.dolthub.com/api/v1alpha1"
USER_AGENT = "dipcatcher-research/1.0 (dolthub as-traded cache; not a live trading system)"
US_CLOSE = ZoneInfo("America/New_York")
# DoltHub's hosted SQL deadline is ~54s. Week windows stay well under it.
_WINDOW_DAYS = 7
_SYMBOL_BATCH = 40
_DATE_BATCH = 20
_MAX_RESPONSE_BYTES = 40_000_000


@dataclass(frozen=True)
class CachePaths:
    """Parquet + provenance receipt for one symbol at one Dolt commit."""

    bars: Path
    receipt: Path


def default_cache_dir() -> Path:
    """Local as-traded cache. Override with ``DOLTHUB_STOCKS_CACHE``."""
    return Path(os.environ.get("DOLTHUB_STOCKS_CACHE", "data/dolthub_stocks"))


def session_close(day: date) -> datetime:
    """Bar close as available_time: 16:00 America/New_York → UTC."""
    local = datetime(day.year, day.month, day.day, 16, 0, tzinfo=US_CLOSE)
    return local.astimezone(UTC)


def parse_symbols(value: str | Sequence[str] | None) -> list[str]:
    """Uppercase, de-duplicated ticker list. Dots (``BRK.B``) are kept."""
    if value is None:
        return []
    parts = value.split(",") if isinstance(value, str) else list(value)
    out: list[str] = []
    for part in parts:
        symbol = str(part).strip().upper()
        if not symbol:
            continue
        if any(ch in symbol for ch in "/\\") or any(ch.isspace() for ch in symbol):
            raise ValueError(f"unsafe ticker {part!r}")
        out.append(symbol)
    return list(dict.fromkeys(out))


def parse_bound(value: date | datetime | str, *, role: str) -> date:
    """Inclusive calendar bound for SQL date filters."""
    if isinstance(value, datetime):
        return value.astimezone(UTC).date() if value.tzinfo else value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    try:
        if "T" in text:
            return parse_bound(datetime.fromisoformat(text.replace("Z", "+00:00")), role=role)
        return date.fromisoformat(text[:10])
    except ValueError as exc:
        raise ValueError(f"invalid {role} date {value!r}") from exc


def _sql_string_list(values: Sequence[str]) -> str:
    escaped = []
    for value in values:
        if "'" in value or "\\" in value or "\x00" in value:
            raise SourceError(f"refusing SQL-unsafe ticker {value!r}")
        escaped.append(f"'{value}'")
    return ", ".join(escaped)


def _chunked(items: Sequence[str], size: int) -> list[list[str]]:
    if size < 1:
        raise ValueError("chunk size must be positive")
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def _date_windows(start: date, end: date, *, days: int = _WINDOW_DAYS) -> list[tuple[date, date]]:
    """Half-open ``[lo, hi)`` windows covering ``start..end`` inclusive."""
    if end < start:
        raise ValueError("end date must be on or after start date")
    if days < 1:
        raise ValueError("window days must be positive")
    windows: list[tuple[date, date]] = []
    cursor = start
    stop = end + timedelta(days=1)
    while cursor < stop:
        nxt = min(cursor + timedelta(days=days), stop)
        windows.append((cursor, nxt))
        cursor = nxt
    return windows


class DolthubSqlClient:
    """Thin wrapper around DoltHub's public v1alpha1 SQL endpoint."""

    def __init__(
        self,
        client: HttpClient | None = None,
        *,
        owner: str = OWNER,
        repo: str = REPO,
        ref: str = DEFAULT_REF,
    ) -> None:
        self.client = client or HttpClient(
            timeout=60.0,
            retries=2,
            max_bytes=_MAX_RESPONSE_BYTES,
            user_agent=USER_AGENT,
        )
        self.owner = owner
        self.repo = repo
        self.ref = ref

    def endpoint(self) -> str:
        return f"{API_ROOT}/{self.owner}/{self.repo}/{self.ref}"

    def query(self, sql: str) -> dict[str, Any]:
        url = query_url(self.endpoint(), {"q": sql})
        payload = self.client.get_json(url)
        if not isinstance(payload, dict):
            raise SourceError("DoltHub SQL response is not an object")
        status = str(payload.get("query_execution_status") or "")
        if status != "Success":
            message = str(payload.get("query_execution_message") or status or "unknown error")
            raise SourceError(f"DoltHub SQL failed: {message}")
        rows = payload.get("rows")
        if rows is None:
            payload = {**payload, "rows": []}
        elif not isinstance(rows, list):
            raise SourceError("DoltHub SQL rows are not a list")
        return payload

    def head_commit(self) -> str:
        payload = self.query("SELECT HASHOF('HEAD') AS h")
        rows = payload.get("rows") or []
        if not rows or not rows[0].get("h"):
            raise SourceError("DoltHub HASHOF('HEAD') returned no commit")
        digest = str(rows[0]["h"]).strip()
        if not digest or any(ch in digest for ch in "/\\"):
            raise SourceError(f"unsafe Dolt commit ref {digest!r}")
        return digest


def _rows_to_ohlcv(rows: Iterable[dict[str, Any]], *, ingested: datetime) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in rows:
        symbol = str(raw.get("act_symbol") or raw.get("security_id") or "").strip().upper()
        day_s = str(raw.get("date") or "").strip()
        if not symbol or not day_s:
            continue
        try:
            day = date.fromisoformat(day_s[:10])
        except ValueError as exc:
            raise SourceError(f"DoltHub row has bad date {day_s!r}") from exc
        close_ts = session_close(day)
        out.append(
            {
                "security_id": symbol,
                "symbol": symbol,
                "event_time": close_ts,
                "available_time": close_ts,
                "ingested_time": ingested,
                "open": raw.get("open"),
                "high": raw.get("high"),
                "low": raw.get("low"),
                "close": raw.get("close"),
                "volume": raw.get("volume") if raw.get("volume") is not None else 0.0,
            }
        )
    return out


def fetch_ohlcv_rows(
    sql_client: DolthubSqlClient,
    symbols: Sequence[str],
    *,
    start: date,
    end: date,
) -> list[dict[str, Any]]:
    """Fetch as-traded rows for ``symbols`` over ``[start, end]`` (calendar).

    Uses short date windows and symbol batches so hosted SQL stays under the
    public deadline. Empty symbol lists return no rows.
    """
    tickers = parse_symbols(symbols)
    if not tickers:
        return []
    if end < start:
        raise ValueError("end date must be on or after start date")
    ingested = utc_now()
    collected: list[dict[str, Any]] = []
    for lo, hi in _date_windows(start, end):
        hi_inclusive = hi - timedelta(days=1)
        for batch in _chunked(tickers, _SYMBOL_BATCH):
            sql = (
                "SELECT date, act_symbol, open, high, low, close, volume "
                "FROM ohlcv WHERE act_symbol IN ("
                f"{_sql_string_list(batch)}) "
                f"AND date >= '{lo.isoformat()}' "
                f"AND date <= '{hi_inclusive.isoformat()}'"
            )
            payload = sql_client.query(sql)
            collected.extend(_rows_to_ohlcv(payload.get("rows") or [], ingested=ingested))
    return collected


def fetch_ohlcv_on_dates(
    sql_client: DolthubSqlClient,
    symbols: Sequence[str],
    days: Sequence[date | str],
) -> list[dict[str, Any]]:
    """Fetch as-traded rows for ``symbols`` on explicit calendar ``days``."""
    tickers = parse_symbols(symbols)
    dates = [parse_bound(day, role="session") for day in days]
    dates = list(dict.fromkeys(dates))
    if not tickers or not dates:
        return []
    ingested = utc_now()
    collected: list[dict[str, Any]] = []
    for date_batch in _chunked([d.isoformat() for d in dates], _DATE_BATCH):
        for sym_batch in _chunked(tickers, _SYMBOL_BATCH):
            sql = (
                "SELECT date, act_symbol, open, high, low, close, volume "
                "FROM ohlcv WHERE act_symbol IN ("
                f"{_sql_string_list(sym_batch)}) "
                f"AND date IN ({_sql_string_list(date_batch)})"
            )
            payload = sql_client.query(sql)
            collected.extend(_rows_to_ohlcv(payload.get("rows") or [], ingested=ingested))
    return collected


_Schema = dict[str, pl.DataType | type[pl.DataType]]

_EMPTY_BAR_SCHEMA: _Schema = {
    "security_id": pl.String,
    "symbol": pl.String,
    "event_time": pl.Datetime("us", "UTC"),
    "available_time": pl.Datetime("us", "UTC"),
    "ingested_time": pl.Datetime("us", "UTC"),
    "source": pl.String,
    "revision_id": pl.String,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "currency": pl.String,
    "session": pl.String,
}


def empty_bars() -> pl.DataFrame:
    """Typed empty panel so empty cache writes stay schema-stable."""
    return pl.DataFrame(schema=_EMPTY_BAR_SCHEMA)


def normalize_dolthub_ohlcv(rows: Iterable[dict[str, Any]]) -> pl.DataFrame:
    """Map raw DoltHub rows onto the bronze OHLCV contract."""
    material = list(rows)
    if not material:
        return empty_bars()
    frame = normalize_ohlcv(material, source=SOURCE_NAME, revision_id=REVISION_ID)
    return (
        frame.with_columns(
            pl.lit("USD").alias("currency"),
            pl.lit("rth").alias("session"),
        )
        .unique(subset=["security_id", "event_time"], keep="last")
        .sort(["security_id", "event_time"])
    )


def cache_paths(cache_root: Path, commit: str, symbol: str) -> CachePaths:
    safe = symbol.strip().upper()
    if not safe or any(ch in safe for ch in "/\\") or any(ch.isspace() for ch in safe):
        raise ValueError(f"unsafe cache symbol {symbol!r}")
    if not commit or any(ch in commit for ch in "/\\"):
        raise ValueError(f"unsafe commit {commit!r}")
    base = cache_root / commit
    return CachePaths(bars=base / f"{safe}.parquet", receipt=base / f"{safe}.json")


def write_cached_symbol(
    frame: pl.DataFrame,
    *,
    cache_root: Path,
    commit: str,
    symbol: str,
    provenance: dict[str, Any],
) -> CachePaths:
    """Persist one symbol's bars and an immutable sha256 receipt."""
    paths = cache_paths(cache_root, commit, symbol)
    paths.bars.parent.mkdir(parents=True, exist_ok=True)
    subset = frame.filter(pl.col("security_id") == symbol.strip().upper())
    tmp = paths.bars.with_name(paths.bars.name + ".tmp")
    try:
        subset.write_parquet(tmp)
        tmp.replace(paths.bars)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    digest = sha256_hex_bytes(paths.bars.read_bytes())
    receipt = {
        "schema_version": 1,
        "source": SOURCE_NAME,
        "revision_id": REVISION_ID,
        "owner": OWNER,
        "repo": REPO,
        "ref": DEFAULT_REF,
        "dolt_commit": commit,
        "symbol": symbol.strip().upper(),
        "path": str(paths.bars),
        "sha256": digest,
        "rows": subset.height,
        "columns": sorted(subset.columns),
        "retrieved_at": datetime.now(UTC).isoformat(),
        "provenance": provenance,
    }
    receipt_tmp = paths.receipt.with_name(paths.receipt.name + ".tmp")
    receipt_tmp.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    receipt_tmp.replace(paths.receipt)
    return paths


def read_cached_symbol(cache_root: Path, commit: str, symbol: str) -> pl.DataFrame | None:
    """Return cached bars when the parquet and matching receipt both exist."""
    paths = cache_paths(cache_root, commit, symbol)
    if not paths.bars.is_file() or not paths.receipt.is_file():
        return None
    receipt = json.loads(paths.receipt.read_text(encoding="utf-8"))
    digest = sha256_hex_bytes(paths.bars.read_bytes())
    if digest != str(receipt.get("sha256") or ""):
        raise SourceError(
            f"cache provenance mismatch for {symbol}@{commit}: "
            f"file sha256 {digest} != receipt {receipt.get('sha256')}"
        )
    if str(receipt.get("dolt_commit") or "") != commit:
        raise SourceError(f"cache receipt commit mismatch for {symbol}")
    return pl.read_parquet(paths.bars)


def panel_fingerprint(frame: pl.DataFrame) -> str:
    """Stable content hash of a bar panel (sorted rows, canonical CSV bytes)."""
    if frame.is_empty():
        return hashlib.sha256(b"").hexdigest()
    ordered = frame.sort(["security_id", "event_time"])
    # CSV keeps the hash independent of parquet encoding noise.
    payload = ordered.write_csv().encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class FetchResult:
    """Normalized panel plus the Dolt commit and per-symbol cache paths."""

    frame: pl.DataFrame
    dolt_commit: str
    cached: tuple[CachePaths, ...]
    provenance: dict[str, Any]


def load_ohlcv(
    symbols: Sequence[str],
    *,
    start: date | datetime | str,
    end: date | datetime | str,
    cache_dir: Path | None = None,
    sql_client: DolthubSqlClient | None = None,
    commit: str | None = None,
    allow_download: bool = True,
) -> FetchResult:
    """Load as-traded bars from cache, fetching missing symbols when allowed."""
    tickers = parse_symbols(symbols)
    start_d = parse_bound(start, role="start")
    end_d = parse_bound(end, role="end")
    root = cache_dir or default_cache_dir()
    client = sql_client or DolthubSqlClient()
    resolved = commit or (client.head_commit() if allow_download else None)
    if not resolved:
        raise SourceError("dolt commit is required when allow_download is false")

    frames: list[pl.DataFrame] = []
    cached_paths: list[CachePaths] = []
    missing: list[str] = []
    for symbol in tickers:
        hit = read_cached_symbol(root, resolved, symbol)
        if hit is None or hit.is_empty():
            missing.append(symbol)
            continue
        # Restrict to the requested window without re-fetching.
        clipped = hit.filter(
            (pl.col("event_time").dt.date() >= start_d) & (pl.col("event_time").dt.date() <= end_d)
        )
        frames.append(clipped)
        cached_paths.append(cache_paths(root, resolved, symbol))

    if missing:
        if not allow_download:
            raise SourceError(
                f"dolthub cache miss for {len(missing)} symbols under commit {resolved}; "
                "pass allow_download=True or pre-warm the cache"
            )
        rows = fetch_ohlcv_rows(client, missing, start=start_d, end=end_d)
        fetched = normalize_dolthub_ohlcv(rows)
        provenance_base = {
            "owner": OWNER,
            "repo": REPO,
            "ref": client.ref,
            "dolt_commit": resolved,
            "start": start_d.isoformat(),
            "end": end_d.isoformat(),
            "window_days": _WINDOW_DAYS,
            "api": client.endpoint(),
        }
        for symbol in missing:
            paths = write_cached_symbol(
                fetched if not fetched.is_empty() else empty_bars(),
                cache_root=root,
                commit=resolved,
                symbol=symbol,
                provenance={
                    **provenance_base,
                    "symbol": symbol,
                    "panel_sha256": panel_fingerprint(
                        fetched.filter(pl.col("security_id") == symbol)
                        if not fetched.is_empty()
                        else empty_bars()
                    ),
                },
            )
            cached_paths.append(paths)
            if not fetched.is_empty():
                part = fetched.filter(pl.col("security_id") == symbol)
                if not part.is_empty():
                    frames.append(part)

    if not frames:
        panel = pl.DataFrame()
    else:
        panel = (
            pl.concat(frames, how="diagonal_relaxed")
            .unique(subset=["security_id", "event_time"], keep="last")
            .sort(["security_id", "event_time"])
        )
    return FetchResult(
        frame=panel,
        dolt_commit=resolved,
        cached=tuple(cached_paths),
        provenance={
            "owner": OWNER,
            "repo": REPO,
            "ref": client.ref,
            "dolt_commit": resolved,
            "start": start_d.isoformat(),
            "end": end_d.isoformat(),
            "symbols": tickers,
            "panel_sha256": panel_fingerprint(panel),
            "n_cached_receipts": len(cached_paths),
        },
    )


def load_ohlcv_on_dates(
    symbols: Sequence[str],
    days: Sequence[date | str],
    *,
    cache_dir: Path | None = None,
    sql_client: DolthubSqlClient | None = None,
    commit: str | None = None,
    allow_download: bool = True,
) -> FetchResult:
    """Load as-traded bars on specific dates (coverage / spot checks).

    Uses point ``date IN (...)`` SQL, which stays under DoltHub's hosted
    deadline, then writes per-symbol cache receipts for provenance.
    """
    tickers = parse_symbols(symbols)
    dates = [parse_bound(day, role="session") for day in days]
    dates = list(dict.fromkeys(dates))
    root = cache_dir or default_cache_dir()
    client = sql_client or DolthubSqlClient()
    if not tickers or not dates:
        return FetchResult(
            frame=pl.DataFrame(),
            dolt_commit=commit or "",
            cached=(),
            provenance={"symbols": tickers, "dates": [d.isoformat() for d in dates]},
        )
    resolved = commit or (client.head_commit() if allow_download else None)
    if not resolved:
        raise SourceError("dolt commit is required when allow_download is false")

    frames: list[pl.DataFrame] = []
    cached_paths: list[CachePaths] = []
    missing: list[str] = []
    wanted = set(dates)
    for symbol in tickers:
        hit = read_cached_symbol(root, resolved, symbol)
        if hit is None:
            missing.append(symbol)
            continue
        clipped = hit.filter(pl.col("event_time").dt.date().is_in(sorted(wanted)))
        # Treat as miss when none of the requested dates are present and download
        # is allowed — the range cache may be for a different window.
        if clipped.is_empty() and allow_download:
            missing.append(symbol)
            continue
        frames.append(clipped)
        cached_paths.append(cache_paths(root, resolved, symbol))

    if missing and not allow_download:
        raise SourceError(f"dolthub cache miss for {len(missing)} symbols under commit {resolved}")
    if missing:
        rows = fetch_ohlcv_on_dates(client, missing, dates)
        fetched = normalize_dolthub_ohlcv(rows)
        provenance_base = {
            "owner": OWNER,
            "repo": REPO,
            "ref": client.ref,
            "dolt_commit": resolved,
            "dates": [d.isoformat() for d in dates],
            "mode": "dates",
            "api": client.endpoint(),
        }
        for symbol in missing:
            paths = write_cached_symbol(
                fetched,
                cache_root=root,
                commit=resolved,
                symbol=symbol,
                provenance={
                    **provenance_base,
                    "symbol": symbol,
                    "panel_sha256": panel_fingerprint(
                        fetched.filter(pl.col("security_id") == symbol)
                    ),
                },
            )
            cached_paths.append(paths)
            part = fetched.filter(pl.col("security_id") == symbol)
            if not part.is_empty():
                frames.append(part)

    if not frames:
        panel = pl.DataFrame()
    else:
        panel = (
            pl.concat(frames, how="diagonal_relaxed")
            .unique(subset=["security_id", "event_time"], keep="last")
            .sort(["security_id", "event_time"])
        )
    return FetchResult(
        frame=panel,
        dolt_commit=resolved,
        cached=tuple(cached_paths),
        provenance={
            "owner": OWNER,
            "repo": REPO,
            "ref": client.ref,
            "dolt_commit": resolved,
            "dates": [d.isoformat() for d in dates],
            "symbols": tickers,
            "panel_sha256": panel_fingerprint(panel),
            "n_cached_receipts": len(cached_paths),
            "mode": "dates",
        },
    )


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"expected a boolean, got {value!r}")


class DolthubStocksSource(SourceAdapter):
    """Registered public source: DoltHub post-no-preference/stocks as-traded."""

    name = SOURCE_NAME

    def __init__(self, client: HttpClient | None = None) -> None:
        super().__init__(
            client
            or HttpClient(
                timeout=60.0,
                retries=2,
                max_bytes=_MAX_RESPONSE_BYTES,
                user_agent=USER_AGENT,
            )
        )

    def fetch(self, **kwargs: object) -> pl.DataFrame:
        """Fetch/cached as-traded bars. ``dates`` (comma list) uses point SQL."""
        allowed = {
            "symbols",
            "start",
            "end",
            "cache_dir",
            "commit",
            "allow_download",
            "dates",
        }
        unknown = sorted(set(kwargs) - allowed)
        if unknown:
            raise ValueError(f"unknown dolthub_stocks fetch kwargs: {', '.join(unknown)}")
        symbols = kwargs.get("symbols")
        if symbols is None:
            raise ValueError("symbols is required")
        if not isinstance(symbols, (str, list, tuple)):
            raise ValueError("symbols must be a string or sequence")
        cache_raw = kwargs.get("cache_dir")
        root = Path(str(cache_raw)) if cache_raw is not None else default_cache_dir()
        commit_raw = kwargs.get("commit")
        commit = str(commit_raw) if commit_raw is not None else None
        allow = _as_bool(kwargs.get("allow_download", True))
        sql = DolthubSqlClient(client=self.client)
        dates = kwargs.get("dates")
        if dates is not None:
            if isinstance(dates, str):
                day_list: list[str] = [part.strip() for part in dates.split(",") if part.strip()]
            elif isinstance(dates, (list, tuple)):
                day_list = [str(item) for item in dates]
            else:
                raise ValueError("dates must be a comma string or sequence")
            return load_ohlcv_on_dates(
                symbols,
                day_list,
                cache_dir=root,
                sql_client=sql,
                commit=commit,
                allow_download=allow,
            ).frame
        start = kwargs.get("start")
        end = kwargs.get("end")
        if start is None or end is None:
            raise ValueError("start and end are required unless dates= is set")
        return load_ohlcv(
            symbols,
            start=str(start) if not isinstance(start, (date, datetime)) else start,
            end=str(end) if not isinstance(end, (date, datetime)) else end,
            cache_dir=root,
            sql_client=sql,
            commit=commit,
            allow_download=allow,
        ).frame
