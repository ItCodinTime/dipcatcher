"""calibration_eval: PIT histogram, coverage, calibration slopes, receipts.

All data here is SYNTHETIC — correctness evidence, never market evidence.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.models.distribution import (
    EmpiricalDistribution,
    GaussianDistribution,
)
from quant_fund.research.calibration_eval import (
    CALIBRATION_EVAL_KIND,
    calibration_contract_errors,
    median_mz_regression,
    pit_histogram_block,
    quantile_hit_rates,
    reliability_regression,
    run_calibration_eval,
    write_calibration_receipt,
)
from quant_fund.research.fleet_eval import (
    SyntheticShard,
    fleet_head_factories,
    iid_gaussian,
)
from quant_fund.research.receipt_v2 import seal_receipt, verify_receipt_payload

TAUS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


def _two_head_factories() -> dict[str, Any]:
    return {
        "empirical": lambda: EmpiricalDistribution(list(TAUS)),
        "gaussian": lambda: GaussianDistribution(list(TAUS)),
    }


def _small_receipt(seed: int = 5) -> dict[str, Any]:
    _, receipt = run_calibration_eval(
        _two_head_factories(),
        shards=["iid_gaussian", "heavy_tail"],
        n_train=128,
        n_eval=64,
        seed=seed,
        taus=TAUS,
    )
    return receipt


def _volatile_keys() -> set[str]:
    return {"generated_at", "receipt_sha256", "environment"}


# --- Statistic KATs -----------------------------------------------------


def test_pit_histogram_uniform_is_flat() -> None:
    # 100 equispaced PITs in (0,1): 10 bins get 10 each — chi2 = 0, p = 1.
    pits = np.linspace(0.01, 0.99, 100)
    block = pit_histogram_block(pits, n_bins=10)
    assert block["pit_bins"] == 10
    assert block["pit_shares"] == pytest.approx([0.1] * 10)
    assert block["pit_chi2"] == pytest.approx(0.0)
    assert block["pit_chi2_p"] == pytest.approx(1.0)
    assert block["pit_boundary_frac"] == 0.0


def test_pit_histogram_point_mass_chi2() -> None:
    # All 100 PITs in the first of 10 bins: (90^2 + 9*10^2)/10 = 900.
    block = pit_histogram_block(np.full(100, 0.05), n_bins=10)
    assert block["pit_chi2"] == pytest.approx(900.0)
    assert block["pit_shares"][0] == pytest.approx(1.0)
    assert block["pit_chi2_p"] == pytest.approx(0.0)


def test_pit_histogram_boundary_mass_is_clipped_and_counted() -> None:
    pits = np.concatenate([[0.0, 1.0], np.linspace(0.1, 0.9, 98)])
    block = pit_histogram_block(pits, n_bins=10)
    assert block["pit_boundary_frac"] == pytest.approx(0.02)
    # Clipped endpoints land in the first/last bins, not dropped.
    assert pytest.approx(sum(block["pit_shares"])) == 1.0


def test_pit_histogram_fails_closed() -> None:
    with pytest.raises(ValueError):
        pit_histogram_block(np.full(49, 0.5))  # below the 50-draw minimum
    with pytest.raises(ValueError):
        pit_histogram_block(np.concatenate([np.full(60, 0.5), [np.nan]]))
    with pytest.raises(ValueError):
        pit_histogram_block(np.linspace(0.01, 0.99, 100), n_bins=51)


def test_quantile_hit_rates_hand_computed() -> None:
    y = np.array([0.1, 0.2, 0.3, 0.4])
    q = np.array(
        [
            [0.15, 0.35],
            [0.25, 0.45],
            [0.05, 0.15],
            [0.35, 0.55],
        ]
    )
    taus = np.array([0.25, 0.75])
    hits = quantile_hit_rates(y, q, taus)
    # col 0: [T,T,F,F] -> 0.5 ; col 1: [T,T,F,T] -> 0.75
    assert hits == pytest.approx([0.5, 0.75])


def test_quantile_hit_rates_shape_checks() -> None:
    y = np.array([0.1, 0.2])
    q = np.array([[0.15, 0.35], [0.25, 0.45]])
    with pytest.raises(ValueError):
        quantile_hit_rates(y, q[:, :1], np.array([0.25, 0.75]))
    with pytest.raises(ValueError):
        quantile_hit_rates(y, np.array([[0.15, np.nan], [0.25, 0.45]]), np.array([0.25, 0.75]))
    with pytest.raises(ValueError):
        quantile_hit_rates(np.array([]), q, np.array([0.25, 0.75]))


def test_reliability_regression_perfect_calibration() -> None:
    taus = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    slope, intercept = reliability_regression(taus.copy(), taus)
    assert slope == pytest.approx(1.0)
    assert intercept == pytest.approx(0.0, abs=1e-9)


def test_reliability_regression_shifted_and_steep() -> None:
    taus = np.array([0.1, 0.5, 0.9])
    slope, intercept = reliability_regression(taus + 0.1, taus)
    assert slope == pytest.approx(1.0)
    assert intercept == pytest.approx(0.1)
    slope, intercept = reliability_regression(0.2 + 0.8 * taus, taus)
    assert slope == pytest.approx(0.8)
    assert intercept == pytest.approx(0.2)


def test_reliability_regression_fails_closed() -> None:
    with pytest.raises(ValueError, match="two tau levels"):
        reliability_regression(np.array([0.5]), np.array([0.5]))
    with pytest.raises(ValueError, match="same length"):
        reliability_regression(np.array([0.5, 0.6]), np.array([0.5]))
    with pytest.raises(ValueError, match="strictly increasing"):
        reliability_regression(np.array([0.5, 0.6]), np.array([0.9, 0.1]))


def test_median_mz_regression_hand_computed() -> None:
    q = np.linspace(0.0, 1.0, 11)
    slope, intercept = median_mz_regression(2.0 + 3.0 * q, q)
    assert slope == pytest.approx(3.0)
    assert intercept == pytest.approx(2.0)


def test_median_mz_regression_constant_predictor_is_none() -> None:
    assert median_mz_regression(np.linspace(0.0, 1.0, 5), np.full(5, 0.4)) is None


# --- Lane behavior ------------------------------------------------------


EXPECTED_COLUMNS = {
    "shard",
    "model",
    "family",
    "status",
    "error",
    "n_train",
    "n_eval",
    "seed",
    "pit_bins",
    "pit_shares",
    "pit_chi2",
    "pit_chi2_p",
    "pit_boundary_frac",
    "coverage_80",
    "coverage_90",
    "calibration_slope",
    "calibration_intercept",
    "mz_slope",
    "mz_intercept",
    *{f"hit_rate_{t:g}" for t in TAUS},
}


def test_calibration_run_schema() -> None:
    frame, receipt = run_calibration_eval(
        _two_head_factories(),
        shards=["iid_gaussian", "bimodal_mixture"],
        n_train=256,
        n_eval=128,
        seed=7,
        taus=TAUS,
    )
    assert frame.height == 4
    assert set(frame.columns) == EXPECTED_COLUMNS
    assert (frame["status"] == "ok").all()
    assert frame["error"].null_count() == frame.height
    assert frame["pit_shares"].list.len().unique().to_list() == [10]
    for col in (
        "pit_chi2",
        "pit_chi2_p",
        "pit_boundary_frac",
        "coverage_80",
        "coverage_90",
        "calibration_slope",
        "calibration_intercept",
    ):
        assert frame[col].drop_nulls().is_finite().all(), col
    for row in frame.iter_rows(named=True):
        shares = np.asarray(row["pit_shares"], dtype=float)
        assert shares.shape == (10,)
        assert pytest.approx(float(shares.sum())) == 1.0
        for tau in TAUS:
            assert 0.0 <= row[f"hit_rate_{tau:g}"] <= 1.0
    assert receipt["schema"] == "calibration_eval.v1"
    assert receipt["kind"] == "calibration_eval"
    assert receipt["data_label"] == "SYNTHETIC"
    assert receipt["live_pnl_claim"] is False
    assert receipt["n_rows"] == 4 and receipt["n_error_rows"] == 0
    assert receipt["coverage_levels"] == [0.8, 0.9]
    assert receipt["pit_bins"] == 10


def test_calibration_records_head_failure() -> None:
    def bad_factory() -> Any:
        class _Bad:
            def fit(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> Any:
                raise ValueError("planted fit failure")

            def predict(self, x: np.ndarray) -> np.ndarray:
                raise RuntimeError("unreachable")

            def metadata(self) -> Any:
                return None

        return _Bad()

    frame, receipt = run_calibration_eval(
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
    assert broken["pit_chi2"].null_count() == 1
    assert receipt["n_error_rows"] == 1


def test_calibration_fail_closed_on_short_eval() -> None:
    # n_eval < 50 leaves no honest PIT histogram support — error rows, not NaN.
    frame, receipt = run_calibration_eval(
        _two_head_factories(),
        shards=["iid_gaussian"],
        n_train=128,
        n_eval=32,
        seed=0,
        taus=TAUS,
    )
    assert frame["status"].to_list() == ["error", "error"]
    assert receipt["n_error_rows"] == 2


def test_calibration_fail_closed_arguments() -> None:
    with pytest.raises(ValueError):
        run_calibration_eval({}, n_train=64, n_eval=32)
    with pytest.raises(ValueError):
        run_calibration_eval(_two_head_factories(), n_train=0, n_eval=64)
    with pytest.raises(ValueError):
        run_calibration_eval(_two_head_factories(), n_train=64, n_eval=0)
    with pytest.raises(ValueError):
        run_calibration_eval(_two_head_factories(), n_train=64, n_eval=True)
    with pytest.raises(ValueError):
        run_calibration_eval(_two_head_factories(), n_train=64, n_eval=64, taus=(0.5,))
    with pytest.raises(ValueError):
        run_calibration_eval(_two_head_factories(), n_train=64, n_eval=64, taus=(0.5, 0.1))
    with pytest.raises(ValueError):
        run_calibration_eval(_two_head_factories(), n_train=64, n_eval=64, pit_bins=2.5)


def test_calibration_rejects_mislabeled_shard() -> None:
    def mislabeled(n: int, seed: int) -> SyntheticShard:
        shard = iid_gaussian(n, seed)
        return SyntheticShard(shard.name, shard.x, shard.y, {"data_label": "REAL"})

    with pytest.raises(ValueError, match="SYNTHETIC label"):
        run_calibration_eval(
            _two_head_factories(), shards={"iid_gaussian": mislabeled}, n_train=64, n_eval=64
        )


def test_dependent_shard_suppresses_chi2_p_value() -> None:
    frame, _ = run_calibration_eval(
        {"gaussian": lambda: GaussianDistribution(list(TAUS))},
        shards=["regime_switch", "iid_gaussian"],
        n_train=128,
        n_eval=64,
        taus=TAUS,
    )
    dep = frame.filter(shard="regime_switch")
    iid = frame.filter(shard="iid_gaussian")
    assert dep["status"].to_list() == ["ok"]
    assert dep["pit_chi2"].is_finite().all()
    assert dep["pit_chi2_p"].null_count() == 1
    assert iid["pit_chi2_p"].is_finite().all()


def test_qar_uses_observed_lag_not_lookahead() -> None:
    """Same origin protocol as fleet_eval: qar consumes the observed lag-1."""
    frame, receipt = run_calibration_eval(
        fleet_head_factories(TAUS, 0, ["qar", "empirical"]),
        shards=["ar1_lagged_x"],
        n_train=256,
        n_eval=128,
        seed=5,
        taus=TAUS,
    )
    assert (frame["status"] == "ok").all()
    assert receipt["n_error_rows"] == 0


def test_calibrated_head_scores_reasonable() -> None:
    """Gaussian head on a Gaussian shard: slope near 1, coverage near nominal."""
    frame, _ = run_calibration_eval(
        {"gaussian": lambda: GaussianDistribution(list(TAUS))},
        shards=["iid_gaussian"],
        n_train=512,
        n_eval=256,
        seed=1,
        taus=TAUS,
    )
    row = frame.row(0, named=True)
    assert row["status"] == "ok"
    assert 0.5 < row["calibration_slope"] < 1.5
    assert 0.6 < row["coverage_80"] < 1.0
    assert 0.7 < row["coverage_90"] <= 1.0


# --- Receipt contract and seal ------------------------------------------


def test_contract_accepts_valid_receipt() -> None:
    assert calibration_contract_errors(_small_receipt()) == []


def test_contract_rejects_bad_envelope_fields() -> None:
    receipt = _small_receipt()
    for key, bad in (
        ("schema", "fleet_eval.v1"),
        ("kind", "distribution_fleet_eval"),
        ("data_label", "REAL"),
        ("live_pnl_claim", True),
    ):
        tampered = {**receipt, key: bad}
        assert calibration_contract_errors(tampered) != []
    assert "results_missing_or_empty" in calibration_contract_errors({**receipt, "results": []})


def test_contract_rederives_row_counts() -> None:
    receipt = _small_receipt()
    tampered = {**receipt, "n_rows": 99}
    assert "n_rows_mismatch" in calibration_contract_errors(tampered)
    tampered = {**receipt, "n_error_rows": 99}
    assert "n_error_rows_mismatch" in calibration_contract_errors(tampered)


def test_v1_receipt_verifies_and_contract_fires() -> None:
    receipt = _small_receipt()
    result = verify_receipt_payload(seal_receipt(receipt))
    assert result["valid"] is True, result["errors"]
    # A tampered-but-resignable n_rows survives the seal but not the contract.
    body = {**receipt, "n_rows": 1}
    result = verify_receipt_payload(seal_receipt(body))
    assert result["valid"] is False
    assert "n_rows_mismatch" in result["errors"]


def test_tampered_coverage_breaks_the_seal() -> None:
    receipt = _small_receipt()
    sealed = seal_receipt(receipt)
    tampered = json.loads(json.dumps(sealed))
    tampered["results"][0]["coverage_80"] = 0.0
    result = verify_receipt_payload(tampered)
    assert result["valid"] is False
    assert "receipt_sha256_mismatch" in result["errors"]


def test_receipt_round_trip(tmp_path: Any) -> None:
    receipt = _small_receipt(seed=13)
    path = write_calibration_receipt(receipt, tmp_path)
    assert path.name.startswith("calibration_eval_") and path.suffix == ".json"
    payload = json.loads(path.read_text())
    assert payload["schema"] == "calibration_eval.v1"
    assert payload["seed"] == 13
    for name in ("iid_gaussian", "heavy_tail"):
        blob = payload["shards"][name]
        assert len(blob["y_sha256"]) == 64 and len(blob["x_sha256"]) == 64
        assert blob["config"]["data_label"] == "SYNTHETIC"
    assert write_calibration_receipt(receipt, tmp_path) == path
    path.write_text("tampered\n")
    with pytest.raises(FileExistsError, match="different content"):
        write_calibration_receipt(receipt, tmp_path)


def test_write_receipt_rejects_bad_version(tmp_path: Any) -> None:
    receipt = _small_receipt()
    with pytest.raises(ValueError, match="receipt_version"):
        write_calibration_receipt(receipt, tmp_path, receipt_version=3)


def test_write_receipt_validates_contract_before_sealing(tmp_path: Any) -> None:
    receipt = _small_receipt()
    with pytest.raises(ValueError, match="synthetic research contract"):
        write_calibration_receipt({**receipt, "data_label": "REAL"}, tmp_path)
    receipt["shards"]["iid_gaussian"]["config"]["paper_pnl"] = 1.0
    with pytest.raises(ValueError, match="synthetic research contract"):
        write_calibration_receipt(receipt, tmp_path)


# --- receipt.v2 envelope ------------------------------------------------


def test_v2_receipt_verifies(tmp_path: Any) -> None:
    receipt = _small_receipt()
    path = write_calibration_receipt(receipt, tmp_path, receipt_version=2)
    payload = json.loads(path.read_text())
    assert payload["schema"] == "receipt.v2"
    assert payload["kind"] == CALIBRATION_EVAL_KIND
    assert payload["verdict"] == "pass"
    result = verify_receipt_payload(payload)
    assert result["valid"] is True, result["errors"]


def test_v2_consistency_catches_payload_tamper(tmp_path: Any) -> None:
    receipt = _small_receipt()
    path = write_calibration_receipt(receipt, tmp_path, receipt_version=2)
    payload = json.loads(path.read_text())
    # Re-seal a tampered payload: the bound digests re-derive the lie.
    tampered = json.loads(json.dumps(payload))
    tampered["payload"]["n_rows"] = 1
    result = verify_receipt_payload(seal_receipt(tampered))
    assert result["valid"] is False
    assert "payload_n_rows_mismatch" in result["errors"]

    tampered = json.loads(json.dumps(payload))
    tampered["payload"]["shards"]["iid_gaussian"]["y_sha256"] = "0" * 64
    result = verify_receipt_payload(seal_receipt(tampered))
    assert result["valid"] is False
    assert "dataset_hash_mismatch" in result["errors"]

    tampered = json.loads(json.dumps(payload))
    tampered["verdict"] = "fail"
    result = verify_receipt_payload(seal_receipt(tampered))
    assert result["valid"] is False
    assert "verdict_mismatch" in result["errors"]


def test_determinism_same_seed() -> None:
    a, b = _small_receipt(seed=11), _small_receipt(seed=11)
    strip = _volatile_keys()
    sa = {k: v for k, v in seal_receipt(a).items() if k not in strip}
    sb = {k: v for k, v in seal_receipt(b).items() if k not in strip}
    assert sa == sb


def test_no_forbidden_metric_keys() -> None:
    frame, receipt = run_calibration_eval(
        _two_head_factories(), shards=["iid_gaussian"], n_train=128, n_eval=64
    )
    sealed = seal_receipt(receipt)

    def _walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                parts = str(key).lower().replace("-", "_").split("_")
                assert not any(
                    tok in {"sharpe", "sortino", "calmar", "pnl", "nav"} for tok in parts
                ), key
                _walk(value)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk({k: v for k, v in sealed.items() if k != "live_pnl_claim"})
    for col in frame.columns:
        parts = col.lower().replace("-", "_").split("_")
        assert not any(tok in {"sharpe", "sortino", "calmar", "pnl", "nav"} for tok in parts)


# --- CLI ----------------------------------------------------------------


def test_cli_calibration_eval(tmp_path: Any) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "calibration-eval",
            "--models",
            "empirical,gaussian",
            "--shards",
            "iid_gaussian",
            "--n-train",
            "96",
            "--n-eval",
            "64",
            "--out-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "DATA_LABEL=SYNTHETIC" in result.output
    receipts = list(tmp_path.glob("calibration_eval_*.json"))
    assert len(receipts) == 1
    payload = json.loads(receipts[0].read_text())
    assert verify_receipt_payload(payload)["valid"] is True


def test_cli_calibration_eval_bad_version(tmp_path: Any) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "calibration-eval",
            "--models",
            "empirical",
            "--shards",
            "iid_gaussian",
            "--n-train",
            "96",
            "--n-eval",
            "64",
            "--out-dir",
            str(tmp_path),
            "--receipt-version",
            "3",
        ],
    )
    assert result.exit_code != 0
