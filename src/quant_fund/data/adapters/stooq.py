"""Public daily file tape from Stooq CSVs. Session-close PIT, not SIP vintages.

Stooq EOD is vendor split-adjusted. Bronze stores those prints as received and
stamps ``revision_id=STOOQ_VENDOR_ADJ``. Silver split factors are identity
unless a separate corporate-action file is supplied. ``available_time`` equals
the session close (16:00 America/New_York for ``*.us``, 16:30 Europe/London
for ``*.uk``). That is a close-published convention, not a SIP as-of vintage.

Stooq EOD is preferred when it returns CSV. If Stooq serves a JavaScript
challenge page, Dipcatcher falls back to Yahoo v8 daily charts with the same
PIT stamps and ``revision_id=YAHOO_VENDOR_ADJ``.
"""

from __future__ import annotations

import csv
import io
import time
import urllib.error
import urllib.request
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

SOURCE = "stooq"
REVISION = "STOOQ_VENDOR_ADJ"
STOQ_URL = "https://stooq.com/q/d/l/?s={ticker}&i=d"
USER_AGENT = "dipcatcher-research/1.0 (local file tape; not a live trading system)"

US_CLOSE = ZoneInfo("America/New_York")
UK_CLOSE = ZoneInfo("Europe/London")

US_LIQUID: tuple[tuple[str, str], ...] = (
    ("SPY", "spy.us"),
    ("AAPL", "aapl.us"),
    ("MSFT", "msft.us"),
    ("GOOGL", "googl.us"),
    ("AMZN", "amzn.us"),
    ("META", "meta.us"),
    ("NVDA", "nvda.us"),
    ("JPM", "jpm.us"),
    ("JNJ", "jnj.us"),
    ("XOM", "xom.us"),
    ("UNH", "unh.us"),
    ("V", "v.us"),
    ("PG", "pg.us"),
    ("HD", "hd.us"),
    ("MA", "ma.us"),
    ("KO", "ko.us"),
)

UK_LIQUID: tuple[tuple[str, str], ...] = (
    ("VOD", "vod.uk"),
    ("BP", "bp.uk"),
    ("HSBA", "hsba.uk"),
    ("AZN", "azn.uk"),
    ("ULVR", "ulvr.uk"),
    ("SHEL", "shel.uk"),
)


def session_close(day: date, suffix: str) -> datetime:
    """Bar close as available_time. ``*.us`` 16:00 ET, ``*.uk`` 16:30 London."""
    if suffix.endswith(".uk"):
        local = datetime(day.year, day.month, day.day, 16, 30, tzinfo=UK_CLOSE)
    else:
        local = datetime(day.year, day.month, day.day, 16, 0, tzinfo=US_CLOSE)
    return local.astimezone(UTC)


def parse_stooq_csv(text: str, *, security_id: str, stooq_symbol: str) -> pl.DataFrame:
    """Parse Stooq ``Date,Open,High,Low,Close,Volume`` into PIT bars."""
    reader = csv.DictReader(io.StringIO(text))
    ingested = datetime.now(tz=UTC)
    rows: list[dict[str, object]] = []
    for raw in reader:
        date_s = (raw.get("Date") or "").strip()
        if not date_s:
            continue
        try:
            day = datetime.strptime(date_s, "%Y-%m-%d").date()
        except ValueError:
            continue
        try:
            opn = float(raw["Open"])
            high = float(raw["High"])
            low = float(raw["Low"])
            close = float(raw["Close"])
            volume = float(raw.get("Volume") or 0.0)
        except (KeyError, TypeError, ValueError):
            continue
        if not all(np_finite(v) and v > 0 for v in (opn, high, low, close)):
            continue
        if volume < 0 or not np_finite(volume):
            continue
        # Envelope repair: Stooq prints occasionally invert on adjusted days.
        lo = min(opn, high, low, close)
        hi = max(opn, high, low, close)
        ts = session_close(day, stooq_symbol)
        rows.append(
            {
                "security_id": security_id,
                "symbol": security_id,
                "event_time": ts,
                "available_time": ts,
                "ingested_time": ingested,
                "source": SOURCE,
                "revision_id": REVISION,
                "open": float(opn),
                "high": float(hi),
                "low": float(lo),
                "close": float(close),
                "volume": float(volume),
                "currency": "GBP" if stooq_symbol.endswith(".uk") else "USD",
                "session": "rth",
            }
        )
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows)


def np_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def fetch_stooq_csv(stooq_symbol: str, *, timeout: float = 30.0) -> str:
    """HTTP GET one Stooq daily CSV. Caller owns rate limits."""
    url = STOQ_URL.format(ticker=stooq_symbol)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — explicit public CSV
        body = resp.read()
    return body.decode("utf-8", errors="replace")


def _master_row(
    security_id: str,
    first_ts: datetime,
    ingested: datetime,
    *,
    exchange: str,
    currency: str,
    sector: str = "Unknown",
    industry: str = "Unknown",
) -> dict[str, object]:
    return {
        "security_id": security_id,
        "ticker": security_id,
        "name": security_id,
        "exchange": exchange,
        "currency": currency,
        "sector": sector,
        "industry": industry,
        "security_type": "common_stock",
        "valid_from": first_ts,
        "valid_to": None,
        "available_time": first_ts,
        "ingested_time": ingested,
        "source": SOURCE,
        "revision_id": REVISION,
    }


def write_file_lake(
    bars: pl.DataFrame,
    root: Path,
    *,
    exchange: str = "XNAS",
    sectors: dict[str, str] | None = None,
) -> dict[str, Path]:
    """Write bars + security_master parquet. Empty corporate_actions on purpose.

    ``sectors`` is a static current-classification map (GICS-style sector per
    ``security_id``). It is applied to the whole history, so it is not a
    point-in-time classification vintage; names outside the map stay
    ``Unknown``.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if bars.is_empty():
        raise ValueError("refusing to write an empty file tape")
    ingested = datetime.now(tz=UTC)
    master_rows = []
    sector_map = sectors or {}
    for sid in bars["security_id"].unique().sort().to_list():
        name_bars = bars.filter(pl.col("security_id") == sid)
        first_raw = name_bars["event_time"].min()
        if not isinstance(first_raw, datetime):
            continue
        currency = str(name_bars["currency"][0]) if "currency" in name_bars.columns else "USD"
        exch = "XLON" if currency == "GBP" else exchange
        sector = str(sector_map.get(str(sid), "Unknown"))
        master_rows.append(
            _master_row(
                str(sid),
                first_raw,
                ingested,
                exchange=exch,
                currency=currency,
                sector=sector,
                industry=sector,
            )
        )
    master = pl.DataFrame(master_rows)
    bars_path = root / "bars.parquet"
    master_path = root / "security_master.parquet"
    actions_path = root / "corporate_actions.parquet"
    bars.write_parquet(bars_path)
    master.write_parquet(master_path)
    # Empty actions: Stooq EOD is already vendor-adjusted; identity silver splits.
    pl.DataFrame(
        schema={
            "security_id": pl.String,
            "event_time": pl.Datetime(time_zone="UTC"),
            "available_time": pl.Datetime(time_zone="UTC"),
            "ingested_time": pl.Datetime(time_zone="UTC"),
            "source": pl.String,
            "revision_id": pl.String,
            "action_type": pl.String,
            "factor": pl.Float64,
            "amount": pl.Float64,
            "new_ticker": pl.String,
        }
    ).write_parquet(actions_path)
    return {"bars": bars_path, "master": master_path, "actions": actions_path}


def download_stooq_universe(
    root: Path,
    names: tuple[tuple[str, str], ...] = US_LIQUID,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    pause_s: float = 0.4,
) -> dict[str, object]:
    """Download a liquid public universe into ``root`` as a PIT-shaped file tape."""
    frames: list[pl.DataFrame] = []
    errors: dict[str, str] = {}
    for security_id, symbol in names:
        try:
            text = fetch_stooq_csv(symbol)
            frame = parse_stooq_csv(text, security_id=security_id, stooq_symbol=symbol)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            errors[security_id] = str(exc)
            time.sleep(pause_s)
            continue
        if start is not None and not frame.is_empty():
            frame = frame.filter(pl.col("event_time") >= start)
        if end is not None and not frame.is_empty():
            frame = frame.filter(pl.col("event_time") <= end)
        if not frame.is_empty():
            frames.append(frame)
        time.sleep(pause_s)
    if not frames:
        return {
            "status": "empty",
            "n_names": 0,
            "errors": errors,
            "pit_convention": "session_close_equals_available",
            "vendor_adjusted": True,
            "sip_vintage": False,
        }
    bars = pl.concat(frames, how="diagonal_relaxed").sort(["event_time", "security_id"])
    paths = write_file_lake(bars, root)
    return {
        "status": "ok",
        "n_names": int(bars["security_id"].n_unique()),
        "n_bars": int(bars.height),
        "paths": {k: str(v) for k, v in paths.items()},
        "errors": errors,
        "source": SOURCE,
        "revision_id": REVISION,
        "pit_convention": "session_close_equals_available",
        "vendor_adjusted": True,
        "sip_vintage": False,
        "champion_alias": False,
        "research_only": True,
    }
