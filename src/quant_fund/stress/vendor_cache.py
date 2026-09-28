"""Optional Kenneth French daily-factor cache.

The daily factor file is not committed. Kenneth French's library is free for
academic and research use; download it from the source below and keep the
cache out of git. This parser converts the published percent units to decimal
excess returns.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from quant_fund.data.sources.base import HttpClient
from quant_fund.utils.atomicio import atomic_write_text

FRENCH_DAILY_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
FRENCH_LICENCE = (
    "Kenneth R. French data library. Free for academic and research use; "
    "download from the Tuck URL. Not redistributed in this repository."
)

Array = NDArray[np.float64]


def parse_french_daily_factors(text: str) -> tuple[tuple[str, ...], Array]:
    """Parse the daily factor CSV text.

    Returns ISO dates and a ``(n, 4)`` array of decimal ``Mkt-RF, SMB, HML, RF``.
    The file stores percent. Parsing stops at the first blank line after the
    daily block, which is how the library separates later annual tables.
    """
    lines = text.splitlines()
    header_at: int | None = None
    for i, line in enumerate(lines):
        if "Mkt-RF" in line and "SMB" in line and "RF" in line:
            header_at = i
            break
    if header_at is None:
        raise ValueError("French daily file has no Mkt-RF header")
    dates: list[str] = []
    rows: list[list[float]] = []
    for line in lines[header_at + 1 :]:
        stripped = line.strip()
        if not stripped:
            if rows:
                break
            continue
        cells = [cell.strip() for cell in stripped.split(",")]
        if len(cells) < 5 or not cells[0].isdigit() or len(cells[0]) != 8:
            if rows:
                break
            continue
        date = f"{cells[0][:4]}-{cells[0][4:6]}-{cells[0][6:8]}"
        try:
            values = [float(cells[j]) / 100.0 for j in range(1, 5)]
        except ValueError as exc:
            raise ValueError(f"non-numeric French factor row {cells[0]}") from exc
        if not np.isfinite(values).all():
            continue
        dates.append(date)
        rows.append(values)
    if len(rows) < 10:
        raise ValueError("French daily parse produced fewer than 10 rows")
    return tuple(dates), np.asarray(rows, dtype=np.float64)


def slice_dates(
    dates: tuple[str, ...],
    values: Array,
    start: str,
    end: str,
) -> tuple[tuple[str, ...], Array]:
    """Inclusive date slice. Dates are ISO strings."""
    keep = [i for i, date in enumerate(dates) if start <= date <= end]
    if len(keep) < 2:
        raise ValueError(f"no French rows in [{start}, {end}]")
    index = np.asarray(keep, dtype=int)
    return tuple(dates[i] for i in keep), np.asarray(values[index], dtype=np.float64)


def fetch_french_daily_factors(dest: Path, client: HttpClient | None = None) -> Path:
    """Download the daily factor zip into ``dest`` as a CSV of decimal factors.

    Network. The caller chooses ``dest``; this function does not write inside
    the repository tree unless asked.
    """
    http = client or HttpClient(timeout=60.0, max_bytes=20_000_000)
    payload = http.get_bytes(FRENCH_DAILY_URL)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not names:
            raise ValueError("French zip did not contain a CSV")
        text = archive.read(names[0]).decode("utf-8", errors="replace")
    dates, values = parse_french_daily_factors(text)
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines = ["date,mkt_rf,smb,hml,rf"]
    for date, row in zip(dates, values, strict=True):
        lines.append(",".join([date, *(f"{float(v):.8f}" for v in row)]))
    atomic_write_text(dest, "\n".join(lines) + "\n")
    return dest
