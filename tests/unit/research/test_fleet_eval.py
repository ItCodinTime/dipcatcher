"""fleet_eval: SYNTHETIC shard generators, proper-score fleet runs, receipts.

All data here is SYNTHETIC — correctness evidence, never market evidence.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from scipy import stats as sps

from quant_fund.models.distribution import (
    EmpiricalDistribution,
    GaussianDistribution,
)
from quant_fund.research.catalog import (
    FORBIDDEN_RESEARCH_METRIC_KEYS,
    family_blob_forbidden_metrics_absent,
)
from quant_fund.research.fleet_eval import (
    SHARD_GENERATORS,
    SyntheticShard,
    ar1_lagged_x,
    bimodal_mixture,
    fleet_head_factories,
    garch_cluster,
    gjr_leverage,
    heavy_tail,
    iid_gaussian,
    left_skew,
    regime_switch,
    resolve_shard_generators,
    run_distribution_fleet,
    vol_break,
    write_fleet_receipt,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

TAUS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


def _assert_shard_sane(shard: SyntheticShard, n: int) -> None:
    assert isinstance(shard, SyntheticShard)
    assert shard.x.shape == (n, 1)
    assert shard.y.shape == (n,)
    assert np.isfinite(shard.x).all()
    assert np.isfinite(shard.y).all()
    assert shard.config["data_label"] == "SYNTHETIC"


def _lag1_sq_autocorr(y: np.ndarray) -> float:
    sq = y * y
    return float(np.corrcoef(sq[:-1], sq[1:])[0, 1])


def test_iid_gaussian_shard() -> None:
    shard = iid_gaussian(400, 1)
    _assert_shard_sane(shard, 400)
    assert abs(sps.skew(shard.y)) < 0.5
    assert abs(sps.kurtosis(shard.y, fisher=True)) < 1.0
    assert shard.config["sigma"] == 0.02


def test_bimodal_mixture_planted_modes() -> None:
    shard = bimodal_mixture(4000, 2)
    _assert_shard_sane(shard, 4000)
    # Two separated Gaussian components → platykurtic, hollow center.
    assert sps.kurtosis(shard.y, fisher=True) < -0.3
    center = 0.65 * shard.config["mu"][0] + 0.35 * shard.config["mu"][1]
    band = np.abs(shard.y - center) < 0.008
    assert float(np.mean(band)) < 0.2


def test_heavy_tail_planted_kurtosis() -> None:
    shard = heavy_tail(4000, 3)
    _assert_shard_sane(shard, 4000)
    # t(3) has infinite kurtosis; the sampled excess must be clearly positive.
    assert sps.kurtosis(shard.y, fisher=True) > 2.0


def test_left_skew_planted_sign() -> None:
    shard = left_skew(3000, 4)
    _assert_shard_sane(shard, 3000)
    assert sps.skew(shard.y) < -0.3
    assert shard.config["lam"] < 0.0


def test_regime_switch_planted_vol_clustering() -> None:
    shard = regime_switch(4000, 5)
    _assert_shard_sane(shard, 4000)
    # Two persistent vol states → squared returns are autocorrelated.
    assert _lag1_sq_autocorr(shard.y) > 0.05
    assert 0 < shard.config["n_state1"] < 4000


def test_garch_cluster_planted_vol_clustering() -> None:
    shard = garch_cluster(4000, 6)
    _assert_shard_sane(shard, 4000)
    assert _lag1_sq_autocorr(shard.y) > 0.05
    assert 0.0 < shard.config["persistence"] < 1.0


def test_vol_break_planted_regime_shift() -> None:
    shard = vol_break(4000, 7)
    assert shard.x.shape == (4000, 1)
    assert shard.y.shape == (4000,)
    assert shard.config["data_label"] == "SYNTHETIC"
    n_post = int(shard.config["n_post_break"])
    pre, post = shard.y[:-n_post], shard.y[-n_post:]
    lo, hi = shard.config["sigma"]
    # Planted break: post-break vol is ~5x pre-break.
    assert float(post.std(ddof=1)) > 3.0 * float(pre.std(ddof=1))
    assert abs(float(pre.std(ddof=1)) - lo) / lo < 0.2
    assert abs(float(post.std(ddof=1)) - hi) / hi < 0.2
    assert shard.config["serial_dependence"] is True


def test_gjr_leverage_planted_asymmetry() -> None:
    shard = gjr_leverage(4000, 8)
    _assert_shard_sane(shard, 4000)
    # Negative-skew innovations + GJR gamma -> left-skewed, clustered returns.
    assert sps.skew(shard.y) < -0.2
    assert _lag1_sq_autocorr(shard.y) > 0.03
    assert 0.0 < shard.config["persistence"] < 1.0
    assert shard.config["serial_dependence"] is True


def test_ar1_lagged_x_causal_features() -> None:
    shard = ar1_lagged_x(2000, 9)
    assert shard.x.shape == (2000, 2)
    assert shard.y.shape == (2000,)
    assert np.isfinite(shard.x).all() and np.isfinite(shard.y).all()
    # x[:, 0] is the previous row's return — causal at each origin.
    assert np.allclose(shard.x[1:, 0], shard.y[:-1])
    rho = float(shard.config["rho"])
    y, x = shard.y[1:], shard.x[1:, 0]
    assert abs(float(np.corrcoef(x, y)[0, 1]) - rho) < 0.1
    assert shard.config["serial_dependence"] is True


def test_shards_deterministic_under_seed() -> None:
    for name, gen in SHARD_GENERATORS.items():
        a, b = gen(64, 11), gen(64, 11)
        assert np.array_equal(a.y, b.y), name


def test_generators_fail_closed_on_size() -> None:
    for gen in SHARD_GENERATORS.values():
        with pytest.raises(ValueError):
            gen(0, 0)
        with pytest.raises(ValueError):
            gen(-5, 0)
        with pytest.raises(ValueError):
            gen(1.5, 0)


def test_resolve_shard_generators() -> None:
    assert set(resolve_shard_generators()) == set(SHARD_GENERATORS)
    assert set(resolve_shard_generators(["heavy_tail"])) == {"heavy_tail"}
    with pytest.raises(ValueError, match="unknown fleet shard"):
        resolve_shard_generators(["not_a_shard"])
    with pytest.raises(ValueError):
        resolve_shard_generators([])


def _two_head_factories() -> dict[str, Any]:
    return {
        "empirical": lambda: EmpiricalDistribution(list(TAUS)),
        "gaussian": lambda: GaussianDistribution(list(TAUS)),
    }


EXPECTED_COLUMNS = {
    "shard",
    "model",
    "family",
    "status",
    "error",
    "n_train",
    "n_eval",
    "seed",
    "crps",
    "pit_ks",
    "pit_ks_p",
    "coverage_80",
    "coverage_90",
    *{f"pinball_{t:g}" for t in TAUS},
}


def test_fleet_run_schema() -> None:
    frame, receipt = run_distribution_fleet(
        _two_head_factories(),
        shards=["iid_gaussian", "bimodal_mixture"],
        n_train=256,
        n_eval=128,
        seed=7,
        taus=TAUS,
    )
    assert frame.height == 4
    assert set(frame.columns) == EXPECTED_COLUMNS
    assert set(frame["shard"].unique()) == {"iid_gaussian", "bimodal_mixture"}
    assert set(frame["model"].unique()) == {"empirical", "gaussian"}
    assert (frame["status"] == "ok").all()
    assert frame["error"].null_count() == frame.height
    for col in ("crps", "pit_ks", "pit_ks_p", "coverage_80", "coverage_90"):
        assert frame[col].drop_nulls().is_finite().all()
    for col in frame.columns:
        if col.startswith("pinball_"):
            assert (frame[col] >= 0.0).all()
    assert (frame["crps"] >= 0.0).all()
    assert ((frame["coverage_80"] >= 0.0) & (frame["coverage_80"] <= 1.0)).all()
    assert ((frame["coverage_90"] >= 0.0) & (frame["coverage_90"] <= 1.0)).all()
    assert receipt["data_label"] == "SYNTHETIC"
    assert receipt["live_pnl_claim"] is False
    assert receipt["seed"] == 7


def test_fleet_records_head_failure() -> None:
    def bad_factory() -> Any:
        class _Bad:
            def fit(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> Any:
                raise ValueError("planted fit failure")

            def predict(self, x: np.ndarray) -> np.ndarray:
                raise RuntimeError("unreachable")

            def metadata(self) -> Any:
                return None

        return _Bad()

    frame, receipt = run_distribution_fleet(
        {**_two_head_factories(), "broken": bad_factory},
        shards=["iid_gaussian"],
        n_train=128,
        n_eval=64,
        seed=0,
        taus=TAUS,
    )
    assert frame.height == 3
    broken = frame.filter(model="broken")
    assert broken["status"].to_list() == ["error"]
    assert "planted fit failure" in broken["error"].to_list()[0]
    assert broken["crps"].null_count() == 1
    assert (frame.filter(model="gaussian")["status"] == "ok").all()
    assert receipt["n_error_rows"] == 1


def test_fleet_rejects_crossing_forecast_as_error_row() -> None:
    class _Crossed:
        def fit(self, x: np.ndarray, y: np.ndarray) -> _Crossed:
            return self

        def predict(self, x: np.ndarray) -> np.ndarray:
            return np.tile(np.linspace(1.0, -1.0, len(TAUS)), (x.shape[0], 1))

        def metadata(self) -> SimpleNamespace:
            return SimpleNamespace(family="distribution")

    frame, receipt = run_distribution_fleet(
        {"crossed": _Crossed}, shards=["iid_gaussian"], n_train=64, n_eval=32
    )
    assert frame["status"].to_list() == ["error"]
    assert "crossing quantiles" in frame["error"].to_list()[0]
    assert receipt["n_error_rows"] == 1


def test_fleet_rejects_mislabeled_custom_shard() -> None:
    def mislabeled(n: int, seed: int) -> SyntheticShard:
        shard = iid_gaussian(n, seed)
        return SyntheticShard(shard.name, shard.x, shard.y, {"data_label": "REAL"})

    with pytest.raises(ValueError, match="SYNTHETIC label"):
        run_distribution_fleet(
            _two_head_factories(), shards={"iid_gaussian": mislabeled}, n_train=64, n_eval=32
        )


def test_fleet_fail_closed_arguments() -> None:
    with pytest.raises(ValueError):
        run_distribution_fleet({}, n_train=64, n_eval=32)
    with pytest.raises(ValueError):
        run_distribution_fleet(_two_head_factories(), n_train=0, n_eval=32)
    with pytest.raises(ValueError):
        run_distribution_fleet(_two_head_factories(), n_train=64, n_eval=0)
    with pytest.raises(ValueError):
        run_distribution_fleet(_two_head_factories(), n_train=1.5, n_eval=32)
    with pytest.raises(ValueError):
        run_distribution_fleet(_two_head_factories(), n_train=64, n_eval=True)
    with pytest.raises(ValueError):
        run_distribution_fleet(_two_head_factories(), n_train=64, n_eval=32, taus=(0.5, 0.1))


def test_receipt_round_trip(tmp_path: Any) -> None:
    _, receipt = run_distribution_fleet(
        _two_head_factories(),
        shards=["iid_gaussian", "heavy_tail"],
        n_train=128,
        n_eval=64,
        seed=13,
        taus=TAUS,
    )
    path = write_fleet_receipt(receipt, tmp_path)
    assert path.name.startswith("fleet_eval_") and path.suffix == ".json"
    payload = json.loads(path.read_text())
    assert payload["schema"] == "fleet_eval.v1"
    assert payload["data_label"] == "SYNTHETIC"
    assert payload["seed"] == 13
    assert payload["n_train"] == 128 and payload["n_eval"] == 64
    assert payload["taus"] == list(TAUS)
    assert payload["models"] == ["empirical", "gaussian"]
    digest = payload["inputs_sha256"]
    assert isinstance(digest, str) and len(digest) == 64
    for name in ("iid_gaussian", "heavy_tail"):
        blob = payload["shards"][name]
        assert len(blob["y_sha256"]) == 64 and len(blob["x_sha256"]) == 64
        assert blob["config"]["data_label"] == "SYNTHETIC"
    # Seal check: receipt_sha256 covers the payload (real_benchmark convention).
    sealed = payload.pop("receipt_sha256")
    assert sealed == hash_bytes(canonical_json_bytes(payload))
    assert write_fleet_receipt(receipt, tmp_path) == path
    path.write_text("tampered\n")
    with pytest.raises(FileExistsError, match="different content"):
        write_fleet_receipt(receipt, tmp_path)


def test_receipt_rejects_non_synthetic_label(tmp_path: Any) -> None:
    _, receipt = run_distribution_fleet(
        _two_head_factories(), shards=["iid_gaussian"], n_train=64, n_eval=32
    )
    with pytest.raises(ValueError, match="synthetic research contract"):
        write_fleet_receipt({**receipt, "data_label": "REAL"}, tmp_path)
    receipt["shards"]["iid_gaussian"]["config"]["paper_pnl"] = 1.0
    with pytest.raises(ValueError, match="synthetic research contract"):
        write_fleet_receipt(receipt, tmp_path)


def test_dependent_shard_has_no_iid_ks_p_value() -> None:
    frame, _ = run_distribution_fleet(
        {"gaussian": lambda: GaussianDistribution(list(TAUS))},
        shards=["regime_switch"],
        n_train=128,
        n_eval=64,
    )
    assert frame["status"].to_list() == ["ok"]
    assert frame["pit_ks"].is_finite().all()
    assert frame["pit_ks_p"].null_count() == 1


def test_no_forbidden_metric_keys(tmp_path: Any) -> None:
    """Receipt + frame columns carry proper-score keys only."""
    frame, receipt = run_distribution_fleet(
        fleet_head_factories(TAUS, 0),
        shards=["iid_gaussian"],
        n_train=256,
        n_eval=128,
        seed=0,
        taus=TAUS,
    )
    path = write_fleet_receipt(receipt, tmp_path)
    payload = json.loads(path.read_text())
    # Mandated honesty flag (receipt envelope convention), not a metric key.
    assert payload["live_pnl_claim"] is False
    # Metric blobs pass the catalog gate.
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


def test_fleet_registry_covers_default_heads() -> None:
    factories = fleet_head_factories(TAUS, 0)
    assert set(factories) == {
        "empirical",
        "gaussian",
        "skew_t",
        "gmm",
        "isotonic",
        "stack",
        "qar",
        "regime",
        "fhs_skew",
        "lgbm_q2",
        "conf_t",
        "hstep_t",
        "hstep_emp",
        "nbeats",
        "nhits",
        "moirai2",
    }
    for factory in factories.values():
        assert factory().metadata().family == "distribution"
    with pytest.raises(ValueError, match="unknown fleet head"):
        fleet_head_factories(TAUS, 0, ["nope"])


def test_fleet_scores_conditional_heads() -> None:
    """The landed conditional/series heads all score on their shards."""
    heads = ["qar", "regime", "fhs_skew", "lgbm_q2", "conf_t", "hstep_t", "hstep_emp"]
    frame, receipt = run_distribution_fleet(
        fleet_head_factories(TAUS, 0, heads),
        shards=["regime_switch", "vol_break", "gjr_leverage", "ar1_lagged_x"],
        n_train=256,
        n_eval=128,
        seed=3,
        taus=TAUS,
    )
    assert frame.height == len(heads) * 4
    assert (frame["status"] == "ok").all(), frame.filter(status="error")
    for col in ("crps", "pit_ks", "coverage_80", "coverage_90"):
        assert frame[col].drop_nulls().is_finite().all()
    # Serially-dependent shards keep the KS statistic but suppress the
    # iid-assumption p-value.
    assert frame["pit_ks_p"].null_count() == frame.height
    assert receipt["model_versions"]["qar"]["version"] == "v1"
    assert receipt["model_versions"]["hstep_t"]["head"] == "hstep"


def test_qar_uses_observed_lag_not_lookahead() -> None:
    """On ar1_lagged_x, QAR's conditional map must beat the unconditional grid."""
    frame, _ = run_distribution_fleet(
        fleet_head_factories(TAUS, 0, ["qar", "empirical"]),
        shards=["ar1_lagged_x"],
        n_train=256,
        n_eval=128,
        seed=5,
        taus=TAUS,
    )
    assert (frame["status"] == "ok").all()
    scores = {row["model"]: row["crps"] for row in frame.iter_rows(named=True)}
    # rho=0.35 AR(1): conditioning on the observed lag must tighten the
    # 1-step distribution relative to the unconditional empirical head.
    assert scores["qar"] < scores["empirical"]


def test_receipt_embeds_head_versions() -> None:
    _, receipt = run_distribution_fleet(
        _two_head_factories(), shards=["iid_gaussian"], n_train=128, n_eval=64, seed=0
    )
    versions = receipt["model_versions"]
    assert set(versions) == {"empirical", "gaussian"}
    assert versions["empirical"] == {"head": "empirical", "version": "v1"}
