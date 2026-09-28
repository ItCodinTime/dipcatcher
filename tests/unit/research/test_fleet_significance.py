"""Fleet significance lane: DM matrix + Hansen MCS over per-row proper scores."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from quant_fund.research.fleet_eval import (
    SHARD_GENERATORS,
    SyntheticShard,
    fleet_head_factories,
)
from quant_fund.research.fleet_significance import (
    FLEET_SIG_SCHEMA,
    fleet_significance_contract_errors,
    run_fleet_significance,
    write_fleet_significance_receipt,
)

TAUS = (0.1, 0.5, 0.9)
N_TRAIN = 128
N_EVAL = 64


class _GoodHead:
    """Empirical-quantile head: in-sample quantiles, no structure."""

    fleet_lagged_predict = False

    def __init__(self, bias: float = 0.0) -> None:
        self._bias = bias

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> _GoodHead:
        self._qs = np.quantile(np.asarray(y, dtype=float), list(TAUS))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        n = np.asarray(x).shape[0]
        return np.tile(self._qs + self._bias, (n, 1))

    def metadata(self) -> Any:
        class _Meta:
            family = "test"
            name = "good"
            version = "0"

        return _Meta()


def _factories(n_good: int = 2, biased: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {
        f"good_{i}": (lambda i=i: _GoodHead(bias=0.01 * i)) for i in range(n_good)
    }
    if biased:
        out["dominated"] = lambda: _GoodHead(bias=5.0)
    return out


def _shards() -> dict[str, Any]:
    return {k: SHARD_GENERATORS[k] for k in ("iid_gaussian", "heavy_tail")}


def _run(**kwargs: Any) -> tuple[Any, dict[str, Any]]:
    params = dict(shards=_shards(), n_train=N_TRAIN, n_eval=N_EVAL, seed=0, taus=TAUS, n_boot=100)
    params.update(kwargs)
    return run_fleet_significance(
        _factories(**{k: params.pop(k) for k in ("n_good", "biased") if k in params}), **params
    )


def test_emits_rows_and_matrices() -> None:
    frame, receipt = _run()
    assert frame.height == 4  # 2 heads x 2 shards
    payload = receipt["payload"]
    assert payload["schema"] == FLEET_SIG_SCHEMA
    assert payload["kind"] == "fleet_significance_eval"
    assert payload["data_label"] == "SYNTHETIC"
    assert payload["live_pnl_claim"] is False
    assert len(payload["shard_results"]) == 2
    sr = payload["shard_results"][0]
    assert set(sr["dm"]) == set(sr["models"])
    assert all(set(row) == set(sr["models"]) for row in sr["dm"].values())
    assert sr["mcs"]["models"] == sr["models"]
    assert payload["pooled"] is not None


def test_identical_heads_are_equivalent() -> None:
    factories = {"a": lambda: _GoodHead(0.0), "b": lambda: _GoodHead(0.0)}
    frame, receipt = run_fleet_significance(
        factories, _shards(), N_TRAIN, N_EVAL, seed=0, taus=TAUS, n_boot=50
    )
    sr = receipt["payload"]["shard_results"][0]
    for i in ("a", "b"):
        for j in ("a", "b"):
            assert sr["dm"][i][j]["t"] == pytest.approx(0.0)
            assert sr["dm"][i][j]["p"] == pytest.approx(1.0)
    assert sr["mcs"]["included"] == {"a": True, "b": True}


def test_dominated_head_is_eliminated() -> None:
    frame, receipt = _run(biased=True)
    sr = receipt["payload"]["shard_results"][0]
    assert sr["mcs"]["included"]["dominated"] is False
    assert sr["mcs"]["p_values"]["dominated"] is not None
    assert sr["mcs"]["p_values"]["dominated"] < sr["mcs"]["alpha"]
    # The dominated head's DM row is positive against the good heads.
    for other in ("good_0", "good_1"):
        assert sr["dm"]["dominated"][other]["t"] is not None
        assert sr["dm"]["dominated"][other]["t"] > 0.0
    row = frame.filter((frame["shard"] == sr["shard"]) & (frame["model"] == "dominated"))
    assert row["mcs_included"].to_list() == [False]


def test_dm_sign_convention() -> None:
    # l_dominated - l_good > 0 → t > 0; the reverse ordering flips the sign.
    _, receipt = _run(biased=True)
    dm = receipt["payload"]["shard_results"][0]["dm"]
    assert dm["dominated"]["good_0"]["t"] > 0.0
    assert dm["good_0"]["dominated"]["t"] < 0.0


def test_crps_loss_matches_riemann_sum() -> None:
    from quant_fund.metrics.scoring import pinball_loss
    from quant_fund.research.fleet_significance import _loss_series

    y = np.array([0.1, -0.2, 0.05, 0.3])
    q = np.tile(np.array([-0.1, 0.0, 0.1]), (4, 1))
    taus = np.array(TAUS)
    got = _loss_series(q, y, taus, "crps")
    dt = np.diff(np.concatenate([[0.0], taus]))
    expected = sum(2.0 * pinball_loss(y, q[:, j], float(taus[j])) * dt[j] for j in range(taus.size))
    np.testing.assert_allclose(got, expected)


def test_determinism() -> None:
    frame_a, receipt_a = _run()
    frame_b, receipt_b = _run()
    assert frame_a.equals(frame_b)
    pa, pb = dict(receipt_a), dict(receipt_b)
    pa.pop("generated_at")
    pb.pop("generated_at")
    assert pa == pb


def test_write_and_contract(tmp_path: Path) -> None:
    _, receipt = _run()
    assert fleet_significance_contract_errors(receipt) == []
    path = write_fleet_significance_receipt(receipt, tmp_path)
    assert path.name.startswith("fleet_significance_")
    sealed = json.loads(path.read_text())
    assert sealed["receipt_sha256"]
    # Rewriting identical content is idempotent; different content fails.
    write_fleet_significance_receipt(receipt, tmp_path)
    tampered = dict(receipt)
    tampered["verdict"] = "blocked"
    tampered["payload"] = {**receipt["payload"], "data_label": "REAL"}
    with pytest.raises(ValueError):
        write_fleet_significance_receipt(tampered, tmp_path)


def test_failed_head_visible_and_excluded() -> None:
    class _Broken:
        def fit(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> _Broken:
            return self

        def predict(self, x: np.ndarray) -> np.ndarray:
            return np.full((np.asarray(x).shape[0], len(TAUS)), np.nan)

        def metadata(self) -> Any:
            class _Meta:
                family = "test"

            return _Meta()

    factories = {"good_0": lambda: _GoodHead(), "good_1": lambda: _GoodHead(0.01), "bad": _Broken}
    frame, receipt = run_fleet_significance(
        factories, _shards(), N_TRAIN, N_EVAL, seed=0, taus=TAUS, n_boot=50
    )
    assert receipt["payload"]["n_error_rows"] == 2
    assert receipt["verdict"] == "fail"
    assert "bad" in receipt["payload"]["shard_results"][0]["excluded"]
    bad = frame.filter(frame["model"] == "bad")
    assert bad["status"].to_list() == ["error", "error"]


def test_validation_fail_closed() -> None:
    with pytest.raises(ValueError):
        run_fleet_significance({}, _shards(), N_TRAIN, N_EVAL)
    with pytest.raises(ValueError):
        _run(n_eval=5)  # below MCS MIN_OBS
    with pytest.raises(ValueError):
        _run(loss="sharpe")
    with pytest.raises(ValueError):
        _run(n_boot=0)
    with pytest.raises(ValueError):
        _run(alpha=1.5)


def test_non_synthetic_shard_rejected() -> None:
    def fake(n: int, seed: int) -> SyntheticShard:
        return SyntheticShard(
            "iid_gaussian",
            np.zeros((n, 1)),
            np.zeros(n),
            {"data_label": "REAL"},
        )

    with pytest.raises(ValueError, match="SYNTHETIC"):
        run_fleet_significance(
            _factories(), {"iid_gaussian": fake}, N_TRAIN, N_EVAL, taus=TAUS, n_boot=10
        )


def test_registry_heads_run() -> None:
    factories = fleet_head_factories(TAUS, 0, ["empirical", "gaussian"])
    frame, receipt = run_fleet_significance(
        factories, _shards(), N_TRAIN, N_EVAL, seed=0, taus=TAUS, n_boot=50
    )
    assert frame.height == 4
    assert receipt["payload"]["shard_results"][0]["mcs"]["n_obs"] == N_EVAL
