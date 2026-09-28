"""Tests for the selection-concordance lane (P3.9)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quant_fund.research.concordance import (
    CONCORDANCE_SCHEMA,
    _dm_eliminated,
    _jaccard,
    _kendall,
    concordance_contract_errors,
    format_concordance_table,
    run_concordance,
    run_concordance_eval,
    write_concordance_receipt,
)
from quant_fund.research.fleet_eval import SyntheticShard

TAUS = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
N_TRAIN, N_EVAL, N_BOOT = 160, 80, 60


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


def _shard(bias_ar: float = 0.0) -> SyntheticShard:
    rng = np.random.default_rng(11)
    n = N_TRAIN + N_EVAL
    eps = rng.normal(size=n)
    y = np.empty(n)
    y[0] = eps[0]
    for i in range(1, n):
        y[i] = 0.3 * y[i - 1] + eps[i]
    x = np.column_stack([np.roll(y, 1), rng.normal(size=n)])
    x[0] = 0.0
    return SyntheticShard(
        "iid_gaussian", x, y, {"data_label": "SYNTHETIC", "serial_dependence": True}
    )


def _shards() -> dict:
    shard = _shard()
    return {"iid_gaussian": lambda n, s: shard}


def _factories(dominated: bool = True) -> dict:
    f: dict = {"good_a": _GoodHead, "good_b": lambda: _GoodHead(0.005)}
    if dominated:
        f["bad"] = lambda: _GoodHead(6.0)
    return f


def _run(**kw):
    return run_concordance(
        _factories(), _shards(), N_TRAIN, N_EVAL, seed=5, taus=TAUS, n_boot=N_BOOT, **kw
    )


def test_emission_shapes() -> None:
    frame, payload = _run()
    ok = frame.filter(pl.col("status") == "ok")
    assert ok.height == 3
    assert set(ok["head"]) == {"bad", "good_a", "good_b"}
    assert payload["n_boot"] == N_BOOT
    report = payload["shards"][0]
    assert set(report["eliminated"]) == {"mcs", "stepm", "dm"}
    assert "concordant" in report


def test_dominated_head_eliminated_everywhere() -> None:
    frame, payload = _run()
    bad = frame.filter(pl.col("head") == "bad").row(0, named=True)
    assert bad["eliminated_all"]
    report = payload["shards"][0]
    assert "bad" in report["eliminated_intersection"]
    assert report["best"] in {"good_a", "good_b"}
    # With a clear loser, the winner is decisively ahead.
    assert report["decisive"]


def test_identical_heads_no_elimination() -> None:
    # Two identical heads cannot separate each other; a near-identical third.
    factories = {
        "a": _GoodHead,
        "b": _GoodHead,
        "c": lambda: _GoodHead(0.02),
    }
    frame, payload = run_concordance(
        factories, _shards(), N_TRAIN, N_EVAL, seed=5, taus=TAUS, n_boot=N_BOOT
    )
    assert frame.filter(pl.col("status") == "ok").height == 3
    assert payload["shards"][0]["eliminated"]["dm"] == []


def test_dm_eliminated_unit() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(size=200) * 0.01
    losses = {
        "good": np.abs(rng.normal(size=200)) * 0.1 + 0.05 + base,
        "bad": np.abs(rng.normal(size=200)) * 0.1 + 5.0 + base,
        "identical_to_good": np.abs(rng.normal(size=200)) * 0.1 + 0.05 + base,
    }
    losses["identical_to_good"] = losses["good"].copy()
    res = _dm_eliminated(losses, 0.10)
    assert res["bad"]["eliminated"]
    assert "good" in res["bad"]["beaten_by"]
    # Identical series cannot eliminate each other.
    assert not res["good"]["eliminated"]
    assert res["good"]["confidence"] is not None


def test_kendall_and_jaccard_helpers() -> None:
    assert _jaccard(set(), set()) == 1.0
    assert _jaccard({"a"}, {"a", "b"}) == pytest.approx(0.5)
    assert _kendall([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert _kendall([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert _kendall([1, None, 3], [1, 2, 3]) is not None
    assert _kendall([1], [1]) is None
    assert _kendall([1, 1], [1, 1]) is None  # constant vector


def test_determinism() -> None:
    f1, p1 = _run()
    f2, p2 = _run()
    assert f1.equals(f2)
    assert p1 == p2


def test_write_and_contract(tmp_path: Path) -> None:
    _, receipt = run_concordance_eval(
        seed=0,
        n_train=N_TRAIN,
        n_eval=N_EVAL,
        n_boot=30,
        head_names=["empirical", "gaussian", "qar"],
        shard_names=["iid_gaussian"],
    )
    assert concordance_contract_errors(receipt) == []
    path = write_concordance_receipt(receipt, tmp_path)
    assert path.name.startswith("concordance_")
    raw = json.loads(path.read_text())
    assert raw["receipt_sha256"][:16] in path.name
    assert raw["payload"]["schema"] == CONCORDANCE_SCHEMA
    bad = dict(receipt)
    bad["kind"] = "other"
    assert "kind" in concordance_contract_errors(bad)


def test_fail_closed_validation() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        run_concordance({}, _shards(), N_TRAIN, N_EVAL, taus=TAUS)
    with pytest.raises(ValueError, match="alpha"):
        run_concordance(_factories(), _shards(), N_TRAIN, N_EVAL, taus=TAUS, alpha=1.5)
    with pytest.raises(ValueError, match="n_eval"):
        run_concordance(_factories(), _shards(), N_TRAIN, 5, taus=TAUS)
    with pytest.raises(ValueError, match="block"):
        run_concordance(_factories(), _shards(), N_TRAIN, N_EVAL, taus=TAUS, block=-1)


def test_error_head_recorded_not_silent() -> None:
    class _Broken:
        def fit(self, x, y):
            raise RuntimeError("no fit")

        def predict(self, x):
            raise RuntimeError("no predict")

    factories = {"good_a": _GoodHead, "good_b": lambda: _GoodHead(0.01), "broken": _Broken}
    frame, payload = run_concordance(
        factories, _shards(), N_TRAIN, N_EVAL, seed=5, taus=TAUS, n_boot=N_BOOT
    )
    err = frame.filter(pl.col("status") == "error")
    assert err.height == 1
    assert err["head"].to_list() == ["broken"]
    assert "no fit" in err["error"].to_list()[0]


def test_format_table_runs() -> None:
    frame, _ = _run()
    text = format_concordance_table(frame)
    assert "iid_gaussian" in text
    assert "bad" in text
