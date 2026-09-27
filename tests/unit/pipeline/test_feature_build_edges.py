"""Wave 25: compute_base_features empty / missing OHLCV fail-closed."""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from quant_fund.config.models import AppConfig
from quant_fund.features.engine import compute_base_features


def test_compute_base_features_empty_bars_raise() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        compute_base_features(pl.DataFrame(), AppConfig())


def test_compute_base_features_missing_ohlcv_raise() -> None:
    bars = pl.DataFrame(
        {
            "security_id": ["A"],
            "event_time": [datetime(2024, 1, 2)],
        }
    )
    with pytest.raises(ValueError, match="missing required OHLCV"):
        compute_base_features(bars, AppConfig())


def test_compute_base_features_minimal_row_runs() -> None:
    """One complete bar row is accepted (features may be null from lookbacks)."""
    bars = pl.DataFrame(
        {
            "security_id": ["A"],
            "event_time": [datetime(2024, 1, 2)],
            "close_total_return": [100.0],
            "close": [100.0],
            "volume": [1_000.0],
            "open_split_adjusted": [99.0],
            "close_split_adjusted": [100.0],
            "high_split_adjusted": [101.0],
            "low_split_adjusted": [98.0],
        }
    )
    out = compute_base_features(bars, AppConfig())
    assert out.height == 1
    assert "ret_1" in out.columns
    assert "vol_parkinson" in out.columns
