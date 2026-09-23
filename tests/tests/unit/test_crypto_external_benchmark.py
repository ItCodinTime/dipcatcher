from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from quant_fund.research.crypto_external_benchmark import (
    ExternalBenchmarkProtocol,
    build_external_protocol,
    select_external_model,
    validate_external_panel,
)


def _panel(source: str = "binance-public-data") -> pl.DataFrame:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows = []
    for day in range(40):
        event = start + timedelta(days=day)
        for symbol in ("BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT"):
            rows.append({
                "security_id": symbol, "symbol": symbol, "event_time": event,
                "available_time": event + timedelta(days=1),
                "ingested_time": event + timedelta(days=2), "source": source,
                "revision_id": "receipt-v1", "open": 100.0, "high": 101.0,
                "low": 99.0, "close": 100.5 + day, "volume": 10.0,
                "currency": "USDT", "session": "24x7",
            })
    return pl.DataFrame(rows)


def test_protocol_is_frozen_and_candidate_only() -> None:
    protocol = build_external_protocol(
        input_id="fixture", input_sha256="a" * 64, n_dates=40
    )

    assert isinstance(protocol, ExternalBenchmarkProtocol)
    assert protocol.symbols == ("BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT")
    assert protocol.interval == "1d"
    assert protocol.timezone == "UTC"
    assert protocol.candidate_only is True
    assert protocol.proof_status == "not_proof"
    assert protocol.sota_proven is False


def test_validate_external_panel_accepts_complete_pit_panel() -> None:
    protocol = build_external_protocol(input_id="fixture", input_sha256="a" * 64, n_dates=40)

    result = validate_external_panel(_panel(), protocol)

    assert result["ok"] is True
    assert result["source"] == "binance-public-data"
    assert result["rows"] == 200
    assert result["symbols"] == list(protocol.symbols)
    assert result["candidate_only"] is True
    assert result["proof_status"] == "not_proof"


def test_validate_external_panel_rejects_synthetic_or_mixed_sources() -> None:
    protocol = build_external_protocol(input_id="fixture", input_sha256="a" * 64, n_dates=40)

    with pytest.raises(ValueError, match="non-synthetic"):
        validate_external_panel(_panel("synthetic"), protocol)

    mixed = _panel().with_columns(
        pl.when(pl.col("symbol") == "BTCUSDT").then(pl.lit("other")).otherwise(pl.col("source")).alias("source")
    )
    with pytest.raises(ValueError, match="one external source"):
        validate_external_panel(mixed, protocol)


def test_validate_external_panel_rejects_future_availability_and_duplicates() -> None:
    protocol = build_external_protocol(input_id="fixture", input_sha256="a" * 64, n_dates=40)
    bad_time = _panel().with_columns(
        pl.when(pl.col("symbol") == "BTCUSDT")
        .then(pl.col("event_time") - pl.duration(days=1))
        .otherwise(pl.col("available_time"))
        .alias("available_time")
    )
    with pytest.raises(ValueError, match="available_time"):
        validate_external_panel(bad_time, protocol)

    duplicate = pl.concat([_panel(), _panel().head(1)])
    with pytest.raises(ValueError, match="duplicate"):
        validate_external_panel(duplicate, protocol)


def test_protocol_metadata_rejects_invalid_digest_and_date_count() -> None:
    with pytest.raises(ValueError, match="metadata"):
        build_external_protocol(input_id="fixture", input_sha256="z" * 64, n_dates=40)
    with pytest.raises(ValueError, match="positive integer"):
        build_external_protocol(input_id="fixture", input_sha256="a" * 64, n_dates=0)


def test_validate_external_panel_rejects_incomplete_or_mixed_revision_panel() -> None:
    protocol = build_external_protocol(input_id="fixture", input_sha256="a" * 64, n_dates=40)
    incomplete = _panel().filter(~((pl.col("symbol") == "ADAUSDT") & (pl.col("event_time").dt.day() == 5)))
    with pytest.raises(ValueError, match="complete"):
        validate_external_panel(incomplete, protocol)
    mixed = _panel().with_columns(
        pl.when(pl.col("symbol") == "XRPUSDT").then(pl.lit("receipt-v2")).otherwise(pl.col("revision_id")).alias("revision_id")
    )
    with pytest.raises(ValueError, match="revision_id"):
        validate_external_panel(mixed, protocol)


def test_variance_scale_qlike_is_not_project_log_scale() -> None:
    from quant_fund.research.crypto_external_benchmark import variance_scale_qlike

    assert variance_scale_qlike(4.0, 2.0) == pytest.approx(2.0 - 1.0 - 0.6931471805599453)
    with pytest.raises(ValueError, match="positive"):
        variance_scale_qlike(0.0, 2.0)


def test_build_external_origins_is_deterministic_and_disjoint() -> None:
    from quant_fund.research.crypto_external_benchmark import build_external_origins

    validation, test = build_external_origins(
        n_dates=200, h=5, min_history=40, stride=5, n_validation=4, n_test=5
    )
    assert validation.tolist() == [40, 45, 50, 55]
    assert test.tolist() == [65, 70, 75, 80, 85]
    assert int(validation[-1]) + 5 - 1 < int(test[0])


def test_validate_external_panel_rejects_naive_or_mixed_timezone_timestamps() -> None:
    protocol = build_external_protocol(input_id="fixture", input_sha256="a" * 64, n_dates=40)
    naive = _panel().with_columns(
        pl.col("event_time").dt.replace_time_zone(None).alias("event_time")
    )
    with pytest.raises(ValueError, match="timezone"):
        validate_external_panel(naive, protocol)


def test_select_external_model_uses_validation_only_and_rejects_invalid_losses() -> None:
    import numpy as np

    selected = select_external_model({
        "candidate": np.array([1.0, 2.0, 1.5]),
        "baseline": np.array([2.0, 2.0, 2.0]),
    })
    assert selected == "candidate"
    with pytest.raises(ValueError, match="finite"):
        select_external_model({"candidate": np.array([1.0, np.nan])})


def test_summarize_external_losses_reports_hac_dm_and_fail_closed_status() -> None:
    import numpy as np
    from quant_fund.research.crypto_external_benchmark import summarize_external_losses

    validation = {
        "candidate": np.full(30, 0.5),
        "rolling": np.full(30, 0.8),
        "ewma": np.full(30, 0.9),
    }
    test = {
        "candidate": np.full(30, 0.5),
        "rolling": np.full(30, 0.8),
        "ewma": np.full(30, 0.9),
    }
    result = summarize_external_losses(validation, test, horizon=5, source="preserved")
    assert result["selected"] == "candidate"
    assert result["candidate_only"] is True
    assert result["proof_status"] == "not_proof"
    assert result["sota_proven"] is False
    assert result["comparisons"]["rolling"]["hac_lags"] == 4
    assert result["comparisons"]["rolling"]["p_two_sided_bonferroni"] is not None
    assert result["comparisons"]["rolling"]["mean_loss_difference"] < 0
