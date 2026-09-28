"""Tests for the mixture-stability lane (P3.8)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quant_fund.research.fleet_eval import SyntheticShard
from quant_fund.research.mixture_stability import (
    format_mixture_stability_table,
    mixture_stability_contract_errors,
    run_mixture_stability,
    run_mixture_stability_eval,
    write_mixture_stability_receipt,
)

TAUS = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
N_TRAIN, N_EVAL, N_BOOT = 192, 96, 40


class _GoodHead:
    """Empirical in-sample quantiles + a constant additive bias."""

    def __init__(self, bias: float = 0.0) -> None:
        self._bias = bias
        self._qs: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self._qs = np.quantile(np.asarray(y, dtype=float).reshape(-1), TAUS)

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert self._qs is not None
        return np.tile(self._qs + self._bias, (np.asarray(x).shape[0], 1))


class _LagHead(_GoodHead):
    fleet_lagged_predict = True


def _shards() -> dict:
    rng = np.random.default_rng(7)
    n = N_TRAIN + N_EVAL
    y = rng.normal(size=n)
    x = np.column_stack([y[:-1], rng.normal(size=n - 1)])
    x = np.vstack([np.zeros(2), x])
    shard = SyntheticShard(
        "iid_gaussian", x, y, {"data_label": "SYNTHETIC", "serial_dependence": False}
    )
    return {"iid_gaussian": lambda n_shard, seed: shard}


def _factories() -> dict:
    return {"good_a": _GoodHead, "good_b": lambda: _GoodHead(0.01), "bad": lambda: _GoodHead(4.0)}


def _run(**kw):
    return run_mixture_stability(
        _factories(), _shards(), N_TRAIN, N_EVAL, seed=3, taus=TAUS, n_boot=N_BOOT, **kw
    )


def test_emission_shapes() -> None:
    frame, receipt = _run()
    ok = frame.filter(pl.col("status") == "ok")
    # 3 mixers x 3 heads on one shard.
    assert ok.height == 9
    assert set(ok["mixer"].unique()) == {"uniform", "ewa", "fixed_share"}
    assert receipt["n_boot"] == N_BOOT
    assert receipt["shards"][0]["bound_violations"] == 0


def test_bad_head_never_leads() -> None:
    frame, _ = _run()
    ok = frame.filter((pl.col("status") == "ok") & (pl.col("head") == "bad"))
    for row in ok.iter_rows(named=True):
        if row["mixer"] == "uniform":
            continue
        assert row["w_obs"] < 0.05
        assert row["w_hi"] < 0.2
        assert row["p_leader"] == pytest.approx(0.0)


def test_uniform_has_no_leader() -> None:
    frame, _ = _run()
    uni = frame.filter((pl.col("status") == "ok") & (pl.col("mixer") == "uniform"))
    assert uni["p_leader"].null_count() == uni.height
    assert uni["is_leader"].sum() == 0
    assert uni["w_obs"].to_list() == pytest.approx([1 / 3] * 3)


def test_weight_bands_contain_observed() -> None:
    frame, _ = _run()
    ok = frame.filter((pl.col("status") == "ok") & (pl.col("mixer") == "ewa"))
    for row in ok.iter_rows(named=True):
        assert row["w_lo"] <= row["w_hi"]
        assert 0.0 <= row["w_lo"] <= 1.0
        assert 0.0 <= row["w_hi"] <= 1.0


def test_leader_stability_bounded() -> None:
    _, receipt = _run()
    report = receipt["shards"][0]
    for m, rep in report["mixers"].items():
        if m == "uniform":
            assert rep["leader"] is None
            assert rep["p_leader"] is None
            continue
        assert 0.0 <= rep["leader_stability"] <= 1.0
        assert sum(rep["p_leader"].values()) == pytest.approx(1.0, abs=0.02)
    assert report["bound_violations"] == 0


def test_determinism() -> None:
    f1, r1 = _run()
    f2, r2 = _run()
    assert f1.equals(f2)
    assert r1 == r2


def test_write_and_contract(tmp_path: Path) -> None:
    _, receipt = run_mixture_stability_eval(
        seed=0,
        n_train=N_TRAIN,
        n_eval=N_EVAL,
        n_boot=N_BOOT,
        head_names=["empirical", "gaussian"],
        shard_names=["iid_gaussian"],
    )
    assert mixture_stability_contract_errors(receipt) == []
    path = write_mixture_stability_receipt(receipt, tmp_path)
    assert path.name.startswith("mixture_stability_")
    raw = json.loads(path.read_text())
    assert raw["receipt_sha256"][:16] in path.name
    bad = dict(receipt)
    bad["kind"] = "other"
    assert "kind" in mixture_stability_contract_errors(bad)


def test_fail_closed_validation() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        run_mixture_stability({}, _shards(), N_TRAIN, N_EVAL, taus=TAUS)
    with pytest.raises(ValueError, match="alpha"):
        run_mixture_stability(_factories(), _shards(), N_TRAIN, N_EVAL, taus=TAUS, alpha=1.5)
    with pytest.raises(ValueError, match="n_boot"):
        run_mixture_stability(_factories(), _shards(), N_TRAIN, N_EVAL, taus=TAUS, n_boot=1)
    with pytest.raises(ValueError, match="block"):
        run_mixture_stability(
            _factories(), _shards(), N_TRAIN, N_EVAL, taus=TAUS, n_boot=10, block=-1
        )


def test_non_synthetic_rejected() -> None:
    shard = SyntheticShard("real", np.zeros((N_TRAIN + N_EVAL, 2)), np.zeros(N_TRAIN + N_EVAL), {})
    with pytest.raises(ValueError, match="SYNTHETIC"):
        run_mixture_stability(
            _factories(), {"iid_gaussian": lambda n, s: shard}, N_TRAIN, N_EVAL, taus=TAUS
        )


def test_registry_heads_and_eval_entrypoint() -> None:
    frame, receipt = run_mixture_stability_eval(
        seed=0,
        n_train=96,
        n_eval=48,
        n_boot=20,
        head_names=["empirical", "gaussian"],
        shard_names=["iid_gaussian"],
    )
    ok = frame.filter(pl.col("status") == "ok")
    assert ok.height == 2 * 3  # 2 heads x 3 mixers
    assert receipt["verdict"] == "pass"
    assert mixture_stability_contract_errors(receipt) == []


def test_format_table_runs() -> None:
    frame, _ = _run()
    text = format_mixture_stability_table(frame)
    assert "iid_gaussian" in text
    assert "fixed_share" in text
