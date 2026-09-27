"""Public-domain H.15 / H.10 crisis windows.

The JSON bundle is a small extract retrieved on 2026-09-27. It is not a vendor
equity tape. See the ``licence`` field inside the file.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

_BUNDLE_PATH = Path(__file__).resolve().parent / "public_domain" / "fred_h10_h15.json"

# H.15 yield series that map to duration factors. Values are in percent.
YIELD_FACTORS: dict[str, str] = {
    "DGS10": "ust_10y",
    "DGS2": "ust_2y",
    "DFF": "fed_funds",
}


def bundle_path() -> Path:
    return _BUNDLE_PATH


@lru_cache(maxsize=1)
def load_bundle() -> dict[str, Any]:
    """Load the committed H.15/H.10 extract. Missing or malformed files fail closed."""
    path = bundle_path()
    if not path.is_file():
        raise FileNotFoundError(f"public-domain stress bundle is missing: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "series" not in raw:
        raise ValueError("public-domain stress bundle is malformed")
    return cast(dict[str, Any], raw)


def window_series(series_id: str, window_id: str) -> tuple[tuple[str, float], ...]:
    """Return ``(date, value)`` pairs for one bundled series window."""
    bundle = load_bundle()
    series = bundle.get("series", {})
    if not isinstance(series, dict) or series_id not in series:
        raise KeyError(f"series {series_id!r} is not in the public-domain bundle")
    block = series[series_id]
    windows = block.get("windows", {}) if isinstance(block, dict) else {}
    if window_id not in windows:
        raise KeyError(f"window {window_id!r} is not stored for {series_id}")
    obs = windows[window_id].get("observations", [])
    out: list[tuple[str, float]] = []
    for row in obs:
        if not isinstance(row, dict):
            raise ValueError(f"malformed observation in {series_id}/{window_id}")
        date = str(row["date"])
        value = float(row["value"])
        if not np.isfinite(value):
            raise ValueError(f"non-finite observation {series_id} {date}")
        out.append((date, value))
    if not out:
        raise ValueError(f"empty window {series_id}/{window_id}")
    return tuple(out)


def series_for_window(window_id: str) -> tuple[str, ...]:
    """Series ids that contain ``window_id``."""
    bundle = load_bundle()
    series = bundle.get("series", {})
    found: list[str] = []
    if not isinstance(series, dict):
        return ()
    for series_id, block in series.items():
        windows = block.get("windows", {}) if isinstance(block, dict) else {}
        if isinstance(windows, dict) and window_id in windows:
            found.append(str(series_id))
    return tuple(found)


def eurchf_levels(window_id: str) -> tuple[tuple[str, float], ...]:
    """CHF per EUR from H.10: DEXSZUS (CHF per USD) times DEXUSEU (USD per EUR)."""
    chf = dict(window_series("DEXSZUS", window_id))
    usd = dict(window_series("DEXUSEU", window_id))
    dates = sorted(set(chf) & set(usd))
    if not dates:
        raise ValueError(f"no overlapping H.10 dates for {window_id}")
    return tuple((date, chf[date] * usd[date]) for date in dates)


def max_yield_increase_pp(values: NDArray[np.float64]) -> float:
    """Largest rise in a yield path, in percentage points: max_{j>i} y_j - y_i."""
    y = np.asarray(values, dtype=float).reshape(-1)
    if y.size < 2 or not np.isfinite(y).all():
        raise ValueError("yield path must contain at least two finite points")
    running_min = np.minimum.accumulate(y)
    return float(np.max(y - running_min))


def max_simple_drawdown(levels: NDArray[np.float64]) -> float:
    """Most negative simple return from a running peak. Zero if the path never falls."""
    y = np.asarray(levels, dtype=float).reshape(-1)
    if y.size < 2 or not np.isfinite(y).all() or np.any(y <= 0.0):
        raise ValueError("price path must contain at least two finite positive levels")
    running_max = np.maximum.accumulate(y)
    return float(np.min(y / running_max - 1.0))


def simple_return(start: float, end: float) -> float:
    if not np.isfinite(start) or not np.isfinite(end) or start == 0.0:
        raise ValueError("simple return needs finite levels and a non-zero start")
    return float(end / start - 1.0)
