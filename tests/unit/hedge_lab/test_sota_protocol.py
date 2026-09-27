"""SOTA protocol: public-feature G1, path RankIC, calibration gate, file tape."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quant_fund.metrics.path_rankic import channel_rankic, mean_path_rankic
from quant_fund.models.dlinear import dlinear_forecast
from quant_fund.models.ranking import PUBLIC_FEATURES, drop_oracle_columns
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent
from quant_fund.research.sota_protocol import (
    load_sota_protocol,
    promotion_decision,
    protocol_sha256,
)


def test_drop_oracle_columns_removes_planted() -> None:
    frame = pl.DataFrame(
        {
            "cs_z_mom_20": [0.1],
            "planted_signal": [1.0],
            "cs_z_planted_signal": [2.0],
        }
    )
    public = drop_oracle_columns(frame)
    assert "planted_signal" not in public.columns
    assert "cs_z_planted_signal" not in public.columns
    assert "cs_z_mom_20" in public.columns


def test_dlinear_forecast_is_causal_and_finite() -> None:
    rng = np.random.default_rng(0)
    hist = np.cumsum(rng.normal(0.001, 0.01, size=(40, 4)), axis=0) + 100.0
    pred = dlinear_forecast(hist, 5)
    assert pred.shape == (5, 4)
    assert np.isfinite(pred).all()


def test_path_rankic_perfect_series() -> None:
    t = np.arange(8, dtype=float)
    pred = np.column_stack([t, t + 1, t - 1, t * 1.1])
    realized = pred + 0.05 * np.arange(8)[:, None]
    row = channel_rankic(pred, realized)
    assert row["mean"] == pytest.approx(1.0, abs=1e-9)
    pooled = mean_path_rankic([row, row])
    assert pooled["n"] == 2
    assert pooled["mean"] == pytest.approx(1.0, abs=1e-9)


def test_sota_protocol_yaml_freezes() -> None:
    proto = load_sota_protocol("configs/sota_protocol.yaml")
    assert proto.frozen is True
    assert proto.synthetic_promotable is False
    assert proto.champion_alias_on_synthetic is False
    assert proto.horizons == [1, 5]
    assert proto.embargo_bars == 5
    assert proto.sample_count == 4
    assert proto.variant == "mini"
    digest = protocol_sha256(proto)
    assert len(digest) == 64
    assert protocol_sha256(proto) == digest


def test_promotion_stays_zero_on_synthetic() -> None:
    g1 = {
        "ridge": {"mean_ic": 0.1},
        "kronos_mini": {
            "mean_ic": 0.9,
            "diebold_mariano_crps": {"preferred": "robinhood_plus"},
        },
    }
    calib = {"pass": True}
    decision = promotion_decision(data_source="SYNTHETIC", g1=g1, calibration=calib)
    assert decision["blend_weight"] == 0.0
    assert decision["sizes_book"] is False
    assert decision["champion_alias"] is False


def test_stooq_parse_fixture_session_close() -> None:
    from quant_fund.data.adapters.stooq import parse_stooq_csv

    csv = (
        "Date,Open,High,Low,Close,Volume\n"
        "2024-01-02,100,101,99,100.5,1000000\n"
        "2024-01-03,100.5,102,100,101,1100000\n"
    )
    bars = parse_stooq_csv(csv, security_id="AAPL", stooq_symbol="aapl.us")
    assert bars.height == 2
    assert {"event_time", "available_time", "ingested_time", "source"} <= set(bars.columns)
    assert (bars["event_time"] == bars["available_time"]).all()
    first = bars["event_time"][0]
    assert isinstance(first, datetime)
    assert first.tzinfo is not None
    assert first.astimezone(UTC).hour in {21, 20}


@pytest.mark.synthetic
def test_g1_public_ridge_receipt(tmp_path: Path) -> None:
    from quant_fund.config import load_config
    from quant_fund.models.robinhood_plus.compare import compare_g1_public_ridge_vs_kronos
    from quant_fund.models.robinhood_plus.constants import ENGINE_NAME
    from quant_fund.pipeline.dataset import build_gold, panel
    from quant_fund.research.catalog import robinhood_plus_claim_honesty_errors

    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.data.synthetic_n_assets = 8
    cfg.data.synthetic_n_days = 90
    cfg.universe.min_history_bars = 5
    cfg.universe.min_adv = 0.0
    cfg.fusion.skip_intervals = True
    cfg.validation.train_bars = 40
    cfg.validation.val_bars = 10
    cfg.validation.test_bars = 10
    cfg.robinhood_plus.lookback = 12
    cfg.robinhood_plus.pred_len = 5
    cfg.robinhood_plus.sample_count = 2
    build_gold(cfg)
    gold = panel(cfg)
    assert "cs_z_planted_signal" in gold.columns
    receipt = compare_g1_public_ridge_vs_kronos(cfg, gold, n_asofs=4, include_torch=False)
    assert receipt["family"] == ENGINE_NAME
    assert receipt["champion"] == "public_ridge"
    assert receipt["oracle_columns_dropped"] is True
    assert receipt["synthetic_not_promotable"] is True
    assert receipt["champion_alias"] is False
    assert receipt["sizes_book"] is False
    assert receipt["default_blend_weight"] == 0.0
    train = receipt["ridge_train"]
    if isinstance(train, dict) and train.get("features"):
        assert "cs_z_planted_signal" not in train["features"]
        assert set(train["features"]) <= set(PUBLIC_FEATURES)
    assert family_blob_forbidden_metrics_absent(receipt)
    assert robinhood_plus_claim_honesty_errors(receipt) == []
    assert "sharpe" not in str(receipt).lower()


@pytest.mark.synthetic
def test_sota_protocol_run_skips_torch(tmp_path: Path) -> None:
    from quant_fund.config import load_config
    from quant_fund.pipeline.dataset import build_gold, panel
    from quant_fund.research.sota_protocol import load_sota_protocol, run_sota_protocol

    cfg = load_config("configs/sota_g1.yaml")
    cfg.data.root = tmp_path
    cfg.data.synthetic_n_assets = 8
    cfg.data.synthetic_n_days = 90
    cfg.universe.min_history_bars = 5
    cfg.universe.min_adv = 0.0
    cfg.validation.train_bars = 40
    cfg.validation.val_bars = 10
    cfg.validation.test_bars = 10
    proto = load_sota_protocol("configs/sota_protocol.yaml")
    proto = proto.model_copy(
        update={"n_asofs": 3, "lookback": 12, "sample_count": 2, "pred_len": 5}
    )
    build_gold(cfg)
    receipt = run_sota_protocol(
        cfg,
        proto,
        panel(cfg),
        include_torch=False,
        include_path_rankic=True,
        include_calibration=True,
    )
    assert receipt["frozen"] is True
    assert receipt["protocol_id"] == "dipcatcher.sota.v1"
    assert receipt["champion_alias"] is False
    assert receipt["blend_weight"] == 0.0
    assert receipt["g1"]["synthetic_not_promotable"] is True
    assert "dlinear" in receipt["path_rankic"]
    assert "PatchTST" in receipt["path_rankic"]["tsfm_skipped_no_local_weights"]
    assert receipt["calibration"]["family"] == "calibration_gate"
    assert family_blob_forbidden_metrics_absent(receipt)
    assert (tmp_path / "metadata" / "sota_receipt.json").is_file()


def test_yahoo_chart_parse_fixture() -> None:
    from quant_fund.data.adapters.yahoo_eod import parse_yahoo_chart

    payload = {
        "chart": {
            "result": [
                {
                    "timestamp": [1704153600, 1704240000],
                    "indicators": {
                        "quote": [
                            {
                                "open": [100.0, 101.0],
                                "high": [102.0, 103.0],
                                "low": [99.0, 100.5],
                                "close": [101.0, 102.5],
                                "volume": [1_000_000, 1_100_000],
                            }
                        ]
                    },
                }
            ]
        }
    }
    bars = parse_yahoo_chart(payload, security_id="AAPL", yahoo_symbol="AAPL")
    assert bars.height == 2
    assert (bars["event_time"] == bars["available_time"]).all()
    assert bars["source"][0] == "yahoo"


def test_stooq_write_lake_roundtrip(tmp_path: Path) -> None:
    from quant_fund.data.adapters.parquet import ParquetMarketProvider
    from quant_fund.data.adapters.stooq import parse_stooq_csv, write_file_lake

    csv = (
        "Date,Open,High,Low,Close,Volume\n"
        "2024-06-03,10,11,9,10.5,500000\n"
        "2024-06-04,10.5,11.2,10.1,10.8,520000\n"
    )
    bars = parse_stooq_csv(csv, security_id="MSFT", stooq_symbol="msft.us")
    paths = write_file_lake(bars, tmp_path)
    provider = ParquetMarketProvider(tmp_path)
    loaded = provider.get_bars()
    assert loaded.height == 2
    master = provider.get_security_master()
    assert master.height == 1
    assert paths["bars"].is_file()
