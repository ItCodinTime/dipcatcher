"""vol_bench: SYNTHETIC vol shards, proper-score bench runs, sealed receipts.

All data here is SYNTHETIC — correctness evidence, never market evidence.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from quant_fund.research.catalog import (
    FORBIDDEN_RESEARCH_METRIC_KEYS,
    family_blob_forbidden_metrics_absent,
)
from quant_fund.research.vol_bench import (
    VOL_SHARD_GENERATORS,
    VolShard,
    break_vol,
    garch_vol,
    resolve_vol_models,
    resolve_vol_shard_generators,
    rough_vol,
    run_vol_bench,
    write_vol_bench_receipt,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes


def _assert_shard_sane(shard: VolShard, n: int) -> None:
    assert isinstance(shard, VolShard)
    for series in (shard.returns, shard.rv, shard.parkinson):
        assert series.shape == (n,)
        assert np.isfinite(series).all()
    assert (shard.rv > 0.0).all()
    assert (shard.parkinson > 0.0).all()
    assert shard.config["data_label"] == "SYNTHETIC"


def _rv_lag1_autocorr(rv: np.ndarray) -> float:
    return float(np.corrcoef(rv[:-1], rv[1:])[0, 1])


def test_garch_vol_shard_planted_clustering() -> None:
    shard = garch_vol(2000, 1)
    _assert_shard_sane(shard, 2000)
    # GJR-GARCH latent vol => realized variance is autocorrelated.
    assert _rv_lag1_autocorr(shard.rv) > 0.1
    assert 0.0 < shard.config["persistence"] < 1.0


def test_rough_vol_shard_planted_clustering() -> None:
    shard = rough_vol(2000, 2)
    _assert_shard_sane(shard, 2000)
    assert _rv_lag1_autocorr(shard.rv) > 0.05
    assert shard.config["hurst"] == 0.1


def test_break_vol_shard_planted_regimes() -> None:
    shard = break_vol(1500, 3)
    _assert_shard_sane(shard, 1500)
    b1, b2 = shard.config["break_bars"]
    lo, crisis, settled = np.split(shard.rv, [b1, b2])
    # Realized variance tracks the planted levels (up to proxy noise).
    assert crisis.mean() > 3.0 * lo.mean()
    assert crisis.mean() > 2.0 * settled.mean()


def test_vol_shards_deterministic_under_seed() -> None:
    for name, gen in VOL_SHARD_GENERATORS.items():
        a, b = gen(64, 11), gen(64, 11)
        assert np.array_equal(a.returns, b.returns), name
        assert np.array_equal(a.rv, b.rv), name
        assert np.array_equal(a.parkinson, b.parkinson), name


def test_vol_shard_generators_fail_closed_on_size() -> None:
    for gen in VOL_SHARD_GENERATORS.values():
        with pytest.raises(ValueError):
            gen(0, 0)
        with pytest.raises(ValueError):
            gen(-5, 0)
        with pytest.raises(ValueError):
            gen(1.5, 0)


def test_resolve_vol_registries() -> None:
    assert set(resolve_vol_models()) == {
        "rv_roll",
        "rv_ewma",
        "har",
        "realized_garch",
        "dip_garch_t",
    }
    assert set(resolve_vol_models(["har"])) == {"har"}
    with pytest.raises(ValueError, match="unknown vol model"):
        resolve_vol_models(["not_a_model"])
    with pytest.raises(ValueError):
        resolve_vol_models([])
    assert set(resolve_vol_shard_generators()) == set(VOL_SHARD_GENERATORS)
    with pytest.raises(ValueError, match="unknown vol shard"):
        resolve_vol_shard_generators(["not_a_shard"])
    with pytest.raises(ValueError):
        resolve_vol_shard_generators([])


def _const_shard(n: int, seed: int) -> VolShard:
    """Constant-variance shard: rv = parkinson = 1e-3 for every bar."""
    del seed
    return VolShard(
        "const_vol",
        np.zeros(n, dtype=float),
        np.full(n, 1e-3, dtype=float),
        np.full(n, 1e-3, dtype=float),
        {"data_label": "SYNTHETIC", "seed": 0},
    )


def _persist(rets: np.ndarray, rv: np.ndarray, park: np.ndarray, h: int, seed: int) -> float:
    return h * float(rv[-1])


def _double(rets: np.ndarray, rv: np.ndarray, park: np.ndarray, h: int, seed: int) -> float:
    return 2.0 * h * float(rv[-1])


# On a constant-rv shard, "persist" is the exact oracle and "double" is a
# 2x-biased forecast: r = x/f = 0.5 -> qlike = ln(2) - 0.5, mse = (h*1e-3)^2.
QLIKE_DOUBLE = math.log(2.0) - 0.5


def test_run_vol_bench_known_answer_qlike_mse() -> None:
    frame, receipt = run_vol_bench(
        {"persist": _persist, "double": _double, "har": _persist},
        shards={"const_vol": _const_shard},
        horizons=(1, 3),
        min_history=40,
        n_origins=10,
        stride=3,
        seed=5,
    )
    assert frame.height == 6  # 3 models x 2 horizons
    assert (frame["status"] == "ok").all()
    for h in (1, 3):
        persist = frame.filter(shard="const_vol", model="persist", horizon=h)
        assert persist["qlike"].to_list() == [0.0]
        assert persist["mse"].to_list() == [0.0]
        double = frame.filter(shard="const_vol", model="double", horizon=h)
        assert double["qlike"].to_list() == pytest.approx([QLIKE_DOUBLE], abs=1e-12)
        assert double["mse"].to_list() == pytest.approx([(h * 1e-3) ** 2], rel=1e-9)
        # DM vs the "har" reference: oracle ties har, double loses by the
        # known-answer margin.
        assert persist["dm_qlike_mean"].to_list() == [0.0]
        assert double["dm_qlike_mean"].to_list() == pytest.approx([QLIKE_DOUBLE], abs=1e-12)
        har_row = frame.filter(shard="const_vol", model="har", horizon=h)
        assert har_row["dm_qlike_mean"].null_count() == 1
    assert receipt["n_rows"] == 6
    assert receipt["n_error_rows"] == 0


def test_run_vol_bench_records_model_failure() -> None:
    def _broken(rets: np.ndarray, rv: np.ndarray, park: np.ndarray, h: int, seed: int) -> float:
        raise ValueError("planted fit failure")

    frame, receipt = run_vol_bench(
        {"persist": _persist, "broken": _broken},
        shards={"const_vol": _const_shard},
        horizons=(1,),
        min_history=40,
        n_origins=10,
        stride=2,
        seed=0,
    )
    assert frame.height == 2
    broken = frame.filter(model="broken")
    assert broken["status"].to_list() == ["error"]
    assert "planted fit failure" in broken["error"].to_list()[0]
    assert broken["qlike"].null_count() == 1
    assert frame.filter(model="persist")["status"].to_list() == ["ok"]
    assert receipt["n_error_rows"] == 1


def test_run_vol_bench_rejects_nonpositive_forecast_as_error() -> None:
    def _zero(rets: np.ndarray, rv: np.ndarray, park: np.ndarray, h: int, seed: int) -> float:
        return 0.0

    frame, _ = run_vol_bench(
        {"zero": _zero},
        shards={"const_vol": _const_shard},
        horizons=(1,),
        min_history=40,
        n_origins=10,
        stride=2,
    )
    assert frame["status"].to_list() == ["error"]
    assert "finite/positive" in frame["error"].to_list()[0]


def test_run_vol_bench_fail_closed_arguments() -> None:
    models = {"persist": _persist}
    shards = {"const_vol": _const_shard}
    with pytest.raises(ValueError):
        run_vol_bench({}, shards=shards)
    with pytest.raises(ValueError):
        run_vol_bench(models, shards=shards, horizons=())
    with pytest.raises(ValueError):
        run_vol_bench(models, shards=shards, horizons=(0,))
    with pytest.raises(ValueError):
        run_vol_bench(models, shards=shards, horizons=(1.5,))
    with pytest.raises(ValueError):
        run_vol_bench(models, shards=shards, n_origins=9)
    with pytest.raises(ValueError):
        run_vol_bench(models, shards=shards, n_origins=True)
    with pytest.raises(ValueError):
        run_vol_bench(models, shards=shards, min_history=10)
    with pytest.raises(ValueError):
        run_vol_bench(models, shards=shards, n_bars=20)


def test_run_vol_bench_rejects_mislabeled_custom_shard() -> None:
    def mislabeled(n: int, seed: int) -> VolShard:
        shard = _const_shard(n, seed)
        return VolShard(
            shard.name, shard.returns, shard.rv, shard.parkinson, {"data_label": "REAL"}
        )

    with pytest.raises(ValueError, match="SYNTHETIC label"):
        run_vol_bench(
            {"persist": _persist},
            shards={"const_vol": mislabeled},
            horizons=(1,),
            min_history=40,
            n_origins=10,
            stride=2,
        )


def test_real_models_score_finite_losses() -> None:
    """All registry models produce finite proper losses on a GARCH shard."""
    frame, receipt = run_vol_bench(
        resolve_vol_models(),
        shards=["garch_vol"],
        horizons=(1,),
        min_history=128,
        n_origins=10,
        stride=1,
        seed=9,
    )
    assert set(frame["model"].unique()) == {
        "rv_roll",
        "rv_ewma",
        "har",
        "realized_garch",
        "dip_garch_t",
    }
    assert (frame["status"] == "ok").all(), frame["error"].to_list()
    assert frame["qlike"].is_finite().all()
    assert (frame["qlike"] >= 0.0).all()
    assert (frame["mse"] >= 0.0).all()
    # DM differentials vs the har reference are populated for non-har models.
    har_row = frame.filter(model="har")
    assert har_row["dm_qlike_mean"].null_count() == 1
    for model in ("rv_roll", "rv_ewma", "realized_garch", "dip_garch_t"):
        row = frame.filter(model=model)
        assert row["dm_qlike_mean"].is_finite().all(), model
    assert receipt["n_error_rows"] == 0


def test_receipt_round_trip(tmp_path: Path) -> None:
    _, receipt = run_vol_bench(
        {"persist": _persist, "double": _double},
        shards={"const_vol": _const_shard, "garch_vol": garch_vol},
        horizons=(1, 2),
        min_history=40,
        n_origins=10,
        stride=2,
        seed=13,
    )
    path = write_vol_bench_receipt(receipt, tmp_path)
    assert path.name.startswith("vol_bench_") and path.suffix == ".json"
    payload = json.loads(path.read_text())
    assert payload["schema"] == "vol_bench.v1"
    assert payload["kind"] == "vol_bench"
    assert payload["data_label"] == "SYNTHETIC"
    assert payload["live_pnl_claim"] is False
    assert payload["seed"] == 13
    assert payload["horizons"] == [1, 2]
    assert payload["models"] == ["double", "persist"]
    assert payload["dm_reference"] == "har"
    assert len(payload["inputs_sha256"]) == 64
    blob = payload["shards"]["garch_vol"]
    assert len(blob["rv_sha256"]) == 64
    assert len(blob["returns_sha256"]) == 64
    assert len(blob["parkinson_sha256"]) == 64
    assert blob["config"]["data_label"] == "SYNTHETIC"
    # Seal check: receipt_sha256 covers the payload (fleet_eval convention).
    sealed = payload.pop("receipt_sha256")
    assert sealed == hash_bytes(canonical_json_bytes(payload))
    assert write_vol_bench_receipt(receipt, tmp_path) == path
    path.write_text("tampered\n")
    with pytest.raises(FileExistsError, match="different content"):
        write_vol_bench_receipt(receipt, tmp_path)


def test_receipt_rejects_non_synthetic_label(tmp_path: Path) -> None:
    _, receipt = run_vol_bench(
        {"persist": _persist},
        shards={"const_vol": _const_shard},
        horizons=(1,),
        min_history=40,
        n_origins=10,
        stride=2,
    )
    with pytest.raises(ValueError, match="synthetic research contract"):
        write_vol_bench_receipt({**receipt, "data_label": "REAL"}, tmp_path)
    receipt["shards"]["const_vol"]["config"]["paper_pnl"] = 1.0
    with pytest.raises(ValueError, match="synthetic research contract"):
        write_vol_bench_receipt(receipt, tmp_path)


def test_no_forbidden_metric_keys(tmp_path: Path) -> None:
    """Receipt + frame columns carry proper-score keys only."""
    frame, receipt = run_vol_bench(
        resolve_vol_models(["rv_roll", "rv_ewma", "har"]),
        shards=["garch_vol"],
        horizons=(1, 2),
        min_history=64,
        n_origins=10,
        stride=2,
        seed=0,
    )
    path = write_vol_bench_receipt(receipt, tmp_path)
    payload = json.loads(path.read_text())
    assert payload["live_pnl_claim"] is False
    assert family_blob_forbidden_metrics_absent(payload["results"]) is True
    for row in payload["results"]:
        assert family_blob_forbidden_metrics_absent(row) is True
    for meta in payload["shards"].values():
        assert family_blob_forbidden_metrics_absent(meta) is True

    def _walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == "live_pnl_claim":
                    continue  # envelope flag asserted False above
                parts = str(key).lower().replace("-", "_").split("_")
                assert not any(tok in FORBIDDEN_RESEARCH_METRIC_KEYS for tok in parts if tok), key
                _walk(value)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(payload)
    for col in frame.columns:
        parts = col.lower().replace("-", "_").split("_")
        assert not any(tok in FORBIDDEN_RESEARCH_METRIC_KEYS for tok in parts if tok), col


def test_vol_bench_cli_writes_receipt(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from quant_fund.cli.main import app

    result = CliRunner().invoke(
        app,
        [
            "vol-bench",
            "--config",
            "configs/research.yaml",
            "--models",
            "rv_roll,rv_ewma",
            "--shards",
            "garch_vol",
            "--horizons",
            "1",
            "--min-history",
            "64",
            "--n-origins",
            "10",
            "--stride",
            "1",
            "--seed",
            "3",
            "--out-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "DATA_LABEL=SYNTHETIC" in result.output
    receipts = list(tmp_path.glob("vol_bench_*.json"))
    assert len(receipts) == 1
    payload = json.loads(receipts[0].read_text())
    assert payload["kind"] == "vol_bench"
    assert payload["seed"] == 3
