"""robinhood+ Kronos-derived K-line engine. Research-only — no Sharpe."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from quant_fund.config.models import AppConfig, RobinhoodPlusConfig
from quant_fund.models.robinhood_plus.autoregress import nucleus_sample
from quant_fund.models.robinhood_plus.bench import bench_robinhood_plus
from quant_fund.models.robinhood_plus.constants import ENGINE_NAME, PRICE_COLS, VOLUME_COL
from quant_fund.models.robinhood_plus.engine import (
    extract_kline,
    forecast_robinhood_plus_cross_section,
)
from quant_fund.models.robinhood_plus.predictor import RobinhoodPlusPredictor
from quant_fund.models.robinhood_plus.tokenizer import (
    HierarchicalBSQTokenizer,
    bits_to_indices,
    indices_to_bits,
    repair_ohlc,
)
from quant_fund.pipeline.dataset import build_gold
from quant_fund.pipeline.forecast import forecast_asof
from quant_fund.research.catalog import (
    OPTIONAL_BENCHMARK_FAMILIES,
    family_blob_forbidden_metrics_absent,
    robinhood_plus_claim_honesty_errors,
)


def _kline(n: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    open_px = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_px, close) * (1.0 + rng.uniform(0.0, 0.004, n))
    low = np.minimum(open_px, close) * (1.0 - rng.uniform(0.0, 0.004, n))
    volume = rng.uniform(1_000.0, 5_000.0, n)
    amount = volume * close
    return np.column_stack([open_px, high, low, close, volume, amount])


def test_bits_round_trip() -> None:
    rng = np.random.default_rng(1)
    idx = rng.integers(0, 32, size=16)
    bits = indices_to_bits(idx, 5)
    assert bits_to_indices(bits).tolist() == idx.tolist()


def test_repair_ohlc_enforces_identity() -> None:
    broken = np.array([[10.0, 9.0, 11.0, 10.5, -1.0, 0.0]])
    fixed = repair_ohlc(broken)
    assert fixed[0, 1] >= max(fixed[0, 0], fixed[0, 3])
    assert fixed[0, 2] <= min(fixed[0, 0], fixed[0, 3])
    assert fixed[0, 4] >= 0.0


def test_tokenizer_encode_decode_finite() -> None:
    tok = HierarchicalBSQTokenizer(s1_bits=5, s2_bits=5, seed=3)
    x = _kline()
    tokens = tok.encode(x)
    decoded = tok.decode(tokens.s1, tokens.s2, tokens.mean, tokens.std)
    assert decoded.shape == x.shape
    assert np.isfinite(decoded).all()
    assert (decoded[:, 1] + 1e-12 >= np.maximum(decoded[:, 0], decoded[:, 3])).all()


def test_transformer_decoder_emits_finite_paths() -> None:
    pred = RobinhoodPlusPredictor(
        lookback=16,
        pred_len=3,
        sample_count=2,
        seed=8,
        decoder="transformer",
        horizons=(1,),
    )
    out = pred.predict_kline(_kline(20, seed=8))
    assert out.status == "ok"
    assert out.paths.shape == (2, 3, 6)
    assert np.isfinite(out.paths).all()


def test_predictor_is_deterministic() -> None:
    x = _kline(40, seed=4)
    a = RobinhoodPlusPredictor(lookback=24, pred_len=5, sample_count=4, seed=11)
    b = RobinhoodPlusPredictor(lookback=24, pred_len=5, sample_count=4, seed=11)
    fa = a.predict_kline(x)
    fb = b.predict_kline(x)
    assert fa.status == "ok"
    assert fa.expected_returns["5d"] == pytest.approx(fb.expected_returns["5d"])
    assert np.allclose(fa.paths, fb.paths)
    assert 0.0 <= fa.probability_positive["5d"] <= 1.0
    assert fa.quantiles["5d"][0.05] <= fa.quantiles["5d"][0.95]


def test_predictor_rejects_future_horizon_keys() -> None:
    pred = RobinhoodPlusPredictor(lookback=16, pred_len=5, sample_count=3, horizons=(1, 5, 20))
    out = pred.predict_kline(_kline(20))
    assert "1d" in out.expected_returns
    assert "5d" in out.expected_returns
    assert "20d" not in out.expected_returns


def test_nucleus_sample_top_p_keeps_mass() -> None:
    rng = np.random.default_rng(0)
    counts = np.log(np.array([100.0, 1.0, 1.0, 1.0]))
    draws = [nucleus_sample(counts, rng, temperature=0.2, top_p=0.5) for _ in range(40)]
    assert set(draws).issubset({0, 1, 2, 3})
    assert draws.count(0) >= 20


def test_future_available_time_is_excluded() -> None:
    start = datetime(2020, 1, 2, 16, 0, 0)
    rows = []
    x = _kline(10, seed=5)
    for i in range(10):
        rows.append(
            {
                "event_time": start + timedelta(days=i),
                "available_time": start + timedelta(days=i + (5 if i == 9 else 0)),
                "security_id": "S0",
                PRICE_COLS[0]: float(x[i, 0]),
                PRICE_COLS[1]: float(x[i, 1]),
                PRICE_COLS[2]: float(x[i, 2]),
                PRICE_COLS[3]: float(x[i, 3]),
                VOLUME_COL: float(x[i, 4]),
            }
        )
    frame = pl.DataFrame(rows)
    asof = start + timedelta(days=9)
    out = forecast_robinhood_plus_cross_section(
        frame, asof, lookback=10, pred_len=3, sample_count=2, security_ids=["S0"]
    )
    assert "S0" in out
    used = extract_kline(
        frame.filter(
            (pl.col("event_time") <= asof) & (pl.col("available_time") <= asof)
        )
    )
    assert used.shape[0] == 9


def test_missing_ohlc_returns_empty_map() -> None:
    frame = pl.DataFrame(
        {
            "event_time": [datetime(2020, 1, 2)],
            "security_id": ["S0"],
            "close": [10.0],
        }
    )
    out = forecast_robinhood_plus_cross_section(frame, datetime(2020, 1, 2))
    assert out == {}


def test_config_rejects_bad_blend_and_bits() -> None:
    with pytest.raises(ValidationError):
        RobinhoodPlusConfig.model_validate({"blend_weight": 1.5})
    with pytest.raises(ValidationError):
        RobinhoodPlusConfig.model_validate({"s1_bits": 12, "s2_bits": 12})
    with pytest.raises(ValidationError):
        RobinhoodPlusConfig.model_validate({"lookback": 1})


def test_catalog_honesty_and_forbidden_keys() -> None:
    assert ENGINE_NAME in OPTIONAL_BENCHMARK_FAMILIES
    good = {
        "family": "robinhood_plus",
        "research_only": True,
        "execution_claim": "research_only",
        "claim": "research_metric_only",
        "backend": "numpy",
        "mean_ic": 0.01,
    }
    assert robinhood_plus_claim_honesty_errors(good) == []
    assert family_blob_forbidden_metrics_absent(good)
    bad = dict(good)
    bad["execution_claim"] = "live"
    assert "robinhood_plus_execution_claim_invalid" in robinhood_plus_claim_honesty_errors(bad)
    dirty = dict(good)
    dirty["sharpe"] = 1.2
    assert family_blob_forbidden_metrics_absent(dirty) is False
    steal = dict(good)
    steal["sizes_book"] = True
    steal["mean_ic_challenger"] = 0.01
    steal["mean_ic_champion"] = 0.05
    assert "robinhood_plus_sizes_book_without_ic_win" in robinhood_plus_claim_honesty_errors(steal)


@pytest.mark.synthetic
def test_forecast_asof_stamps_n_ok_and_n_fallback(tmp_path: Path) -> None:
    from quant_fund.config import load_config

    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.data.synthetic_n_assets = 6
    cfg.data.synthetic_n_days = 80
    cfg.universe.min_history_bars = 5
    cfg.universe.min_adv = 0.0
    cfg.fusion.skip_intervals = True
    cfg.robinhood_plus.lookback = 16
    cfg.robinhood_plus.pred_len = 5
    cfg.robinhood_plus.sample_count = 3
    cfg.robinhood_plus.blend_weight = 0.0
    build_gold(cfg)
    state = forecast_asof(cfg)
    assert any(n.startswith("robinhood_plus_n_ok=") for n in state.notes)
    assert any(n.startswith("robinhood_plus_n_fallback=") for n in state.notes)
    assert "robinhood_plus_challenger" in state.notes
    row = state.forecasts[0]
    assert "robinhood_plus_n_ok" in row.diagnostics
    assert "robinhood_plus_n_fallback" in row.diagnostics
    assert row.diagnostics.get("core_engine") == "ridge"
    dumped = state.model_dump_json()
    assert "sharpe" not in dumped.lower()


@pytest.mark.synthetic
def test_forecast_asof_stamps_robinhood_plus_core_engine(tmp_path: Path) -> None:
    from quant_fund.config import load_config

    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.data.synthetic_n_assets = 6
    cfg.data.synthetic_n_days = 80
    cfg.universe.min_history_bars = 5
    cfg.universe.min_adv = 0.0
    cfg.fusion.skip_intervals = True
    cfg.robinhood_plus.lookback = 16
    cfg.robinhood_plus.pred_len = 5
    cfg.robinhood_plus.sample_count = 3
    cfg.robinhood_plus.blend_weight = 1.0
    build_gold(cfg)
    state = forecast_asof(cfg)
    assert ENGINE_NAME in state.notes
    assert any(n.startswith("robinhood_plus_n_ok=") for n in state.notes)
    assert any(n.startswith("robinhood_plus_n_fallback=") for n in state.notes)
    assert any(f.diagnostics.get("core_engine") == ENGINE_NAME for f in state.forecasts)
    row = next(f for f in state.forecasts if f.diagnostics.get("core_engine") == ENGINE_NAME)
    assert np.isfinite(row.alpha["5d"])
    assert "5d" in row.expected_returns
    assert 0.0 <= row.probability_positive["5d"] <= 1.0
    dumped = state.model_dump_json()
    assert "sharpe" not in dumped.lower()


@pytest.mark.synthetic
def test_bench_receipt_is_research_only(tmp_path: Path) -> None:
    from quant_fund.config import load_config

    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.data.synthetic_n_assets = 6
    cfg.data.synthetic_n_days = 90
    cfg.universe.min_history_bars = 5
    cfg.universe.min_adv = 0.0
    cfg.robinhood_plus.lookback = 16
    cfg.robinhood_plus.pred_len = 5
    cfg.robinhood_plus.sample_count = 3
    build_gold(cfg)
    from quant_fund.pipeline.dataset import panel

    receipt = bench_robinhood_plus(panel(cfg), cfg)
    assert receipt["family"] == ENGINE_NAME
    assert receipt["research_only"] is True
    assert receipt["execution_claim"] == "research_only"
    assert robinhood_plus_claim_honesty_errors(receipt) == []
    assert family_blob_forbidden_metrics_absent(receipt)
    assert "sharpe" not in str(receipt).lower()


def test_torch_backend_fails_closed_without_local_weights() -> None:
    from quant_fund.models.robinhood_plus.torch_backend import (
        RobinhoodPlusTorchError,
        load_pretrained_predictor,
    )

    with pytest.raises(RobinhoodPlusTorchError, match="allow_network=false"):
        load_pretrained_predictor("mini", allow_network=False)


def test_forecast_asof_torch_backend_fails_closed() -> None:
    from quant_fund.config.models import RobinhoodPlusBackend
    from quant_fund.models.robinhood_plus.torch_backend import RobinhoodPlusTorchError
    from quant_fund.pipeline.forecast import _robinhood_plus_asof

    cfg = AppConfig()
    cfg.robinhood_plus.enabled = True
    cfg.robinhood_plus.blend_weight = 1.0
    cfg.robinhood_plus.backend = RobinhoodPlusBackend.TORCH
    frame = pl.DataFrame(
        {
            "event_time": [datetime(2020, 1, 2)],
            "security_id": ["S0"],
            **{c: [10.0] for c in PRICE_COLS},
            VOLUME_COL: [100.0],
        }
    )
    with pytest.raises(RobinhoodPlusTorchError, match="allow_network=false"):
        _robinhood_plus_asof(frame, datetime(2020, 1, 2), cfg, ["S0"])


def test_poison_next_bar_does_not_move_forecast() -> None:
    start = datetime(2020, 1, 2, 16, 0, 0)
    x = _kline(24, seed=9)
    rows = []
    for i in range(24):
        rows.append(
            {
                "event_time": start + timedelta(days=i),
                "available_time": start + timedelta(days=i),
                "security_id": "S0",
                PRICE_COLS[0]: float(x[i, 0]),
                PRICE_COLS[1]: float(x[i, 1]),
                PRICE_COLS[2]: float(x[i, 2]),
                PRICE_COLS[3]: float(x[i, 3]),
                VOLUME_COL: float(x[i, 4]),
            }
        )
    frame = pl.DataFrame(rows)
    asof = start + timedelta(days=20)
    clean = forecast_robinhood_plus_cross_section(
        frame, asof, lookback=16, pred_len=4, sample_count=3, security_ids=["S0"], seed=3
    )
    poisoned = frame.with_columns(
        pl.when(pl.col("event_time") > asof)
        .then(pl.lit(1.0e9))
        .otherwise(pl.col(PRICE_COLS[3]))
        .alias(PRICE_COLS[3])
    )
    dirty = forecast_robinhood_plus_cross_section(
        poisoned, asof, lookback=16, pred_len=4, sample_count=3, security_ids=["S0"], seed=3
    )
    assert clean["S0"].forecast.status == "ok"
    assert dirty["S0"].forecast.status == "ok"
    assert clean["S0"].forecast.rank_score == pytest.approx(dirty["S0"].forecast.rank_score)
    assert np.allclose(clean["S0"].forecast.paths, dirty["S0"].forecast.paths)


def test_poison_unpublished_available_time_does_not_enter_lookback() -> None:
    start = datetime(2020, 1, 2, 16, 0, 0)
    x = _kline(16, seed=2)
    rows = []
    for i in range(16):
        rows.append(
            {
                "event_time": start + timedelta(days=i),
                "available_time": start + timedelta(days=i),
                "security_id": "S0",
                PRICE_COLS[0]: float(x[i, 0]),
                PRICE_COLS[1]: float(x[i, 1]),
                PRICE_COLS[2]: float(x[i, 2]),
                PRICE_COLS[3]: float(x[i, 3]),
                VOLUME_COL: float(x[i, 4]),
            }
        )
    frame = pl.DataFrame(rows)
    asof = start + timedelta(days=15)
    dropped = frame.filter(pl.col("event_time") != start + timedelta(days=10))
    delayed = frame.with_columns(
        pl.when(pl.col("event_time") == start + timedelta(days=10))
        .then(pl.lit(asof + timedelta(days=1)))
        .otherwise(pl.col("available_time"))
        .alias("available_time")
    ).with_columns(
        pl.when(pl.col("event_time") == start + timedelta(days=10))
        .then(pl.lit(1.0e9))
        .otherwise(pl.col(PRICE_COLS[3]))
        .alias(PRICE_COLS[3])
    )
    a = forecast_robinhood_plus_cross_section(
        dropped, asof, lookback=16, pred_len=3, sample_count=2, security_ids=["S0"], seed=4
    )
    b = forecast_robinhood_plus_cross_section(
        delayed, asof, lookback=16, pred_len=3, sample_count=2, security_ids=["S0"], seed=4
    )
    assert a["S0"].forecast.rank_score == pytest.approx(b["S0"].forecast.rank_score)


@pytest.mark.synthetic
def test_compare_ridge_vs_robinhood_plus_receipt(tmp_path: Path) -> None:
    from quant_fund.config import load_config
    from quant_fund.models.robinhood_plus.compare import compare_ridge_vs_robinhood_plus
    from quant_fund.pipeline.dataset import panel

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
    receipt = compare_ridge_vs_robinhood_plus(cfg, panel(cfg), n_asofs=4, train_ridge=True)
    assert receipt["family"] == ENGINE_NAME
    assert receipt["research_only"] is True
    assert receipt["execution_claim"] == "research_only"
    assert "sharpe" not in str(receipt).lower()
    assert "sizes_book" in receipt
    assert robinhood_plus_claim_honesty_errors(receipt) == []
    assert family_blob_forbidden_metrics_absent(receipt)
    if receipt["status"] == "ok":
        assert "mean_ic_champion" in receipt
        assert "mean_ic_challenger" in receipt
        assert "diebold_mariano_crps" in receipt
        if receipt["sizes_book"] is True:
            assert receipt["mean_ic_challenger"] > receipt["mean_ic_champion"]
        else:
            assert cfg.robinhood_plus.blend_weight == 0.0


def test_default_blend_weight_does_not_size_the_book() -> None:
    cfg = AppConfig()
    assert cfg.robinhood_plus.enabled is True
    assert cfg.robinhood_plus.blend_weight == 0.0


def test_kronos_mini_three_way_skips_without_local_weights() -> None:
    from quant_fund.models.robinhood_plus.compare import compare_numpy_vs_kronos_mini_vs_ridge

    cfg = AppConfig()
    empty = pl.DataFrame({"event_time": [], "security_id": []})
    receipt = compare_numpy_vs_kronos_mini_vs_ridge(cfg, empty, n_asofs=2)
    assert receipt["research_only"] is True
    assert receipt["numpy_markov"]["status"] == "empty_panel"
    assert receipt["kronos_mini"]["status"] in {
        "skipped_no_local_weights",
        "empty_panel",
        "insufficient_history",
        "insufficient_scored",
    }
    assert receipt["kronos_mini"].get("sizes_book") is not True
    assert "sharpe" not in str(receipt).lower()


def _local_kronos_mini_dirs() -> tuple[Path, Path] | None:
    repo = Path(__file__).resolve().parents[2]
    tok = repo / "third_party" / "kronos_weights" / "Kronos-Tokenizer-2k"
    model = repo / "third_party" / "kronos_weights" / "Kronos-mini"
    if tok.is_dir() and model.is_dir() and (model / "model.safetensors").is_file():
        return tok, model
    return None


def test_local_kronos_mini_is_wired_when_checkpoints_exist() -> None:
    from quant_fund.config.models import RobinhoodPlusBackend
    from quant_fund.models.robinhood_plus.torch_backend import torch_available
    from quant_fund.pipeline.forecast import _robinhood_plus_asof

    paths = _local_kronos_mini_dirs()
    if not torch_available() or paths is None:
        pytest.skip("local Kronos-mini checkpoints or torch extra not present")
    tok, model = paths
    cfg = AppConfig()
    cfg.robinhood_plus.enabled = True
    cfg.robinhood_plus.blend_weight = 1.0
    cfg.robinhood_plus.backend = RobinhoodPlusBackend.TORCH
    cfg.robinhood_plus.allow_network = False
    cfg.robinhood_plus.tokenizer_path = str(tok)
    cfg.robinhood_plus.model_path = str(model)
    cfg.robinhood_plus.lookback = 16
    cfg.robinhood_plus.pred_len = 5
    cfg.robinhood_plus.sample_count = 1
    start = datetime(2020, 1, 2, 16, 0, 0)
    x = _kline(24, seed=1)
    rows = []
    for i in range(24):
        rows.append(
            {
                "event_time": start + timedelta(days=i),
                "available_time": start + timedelta(days=i),
                "security_id": "S0",
                PRICE_COLS[0]: float(x[i, 0]),
                PRICE_COLS[1]: float(x[i, 1]),
                PRICE_COLS[2]: float(x[i, 2]),
                PRICE_COLS[3]: float(x[i, 3]),
                VOLUME_COL: float(x[i, 4]),
            }
        )
    frame = pl.DataFrame(rows)
    out = _robinhood_plus_asof(frame, start + timedelta(days=20), cfg, ["S0"])
    hit = out["S0"].forecast
    assert hit.status == "ok"
    assert hit.diagnostics.get("backend") == "torch"
    assert np.isfinite(hit.rank_score)
