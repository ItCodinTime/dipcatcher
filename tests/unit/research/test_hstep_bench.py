"""Tests for the multi-horizon distribution bench (P3.2-lite)."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.research.hstep_bench import (
    HSTEP_MODELS,
    hstep_bench_receipt_v2,
    hstep_bench_report,
    hstep_bench_v1_contract_errors,
    hstep_bench_v2_consistency_errors,
    resolve_hstep_models,
    run_hstep_bench,
    write_hstep_bench_receipt,
)
from quant_fund.research.receipt_v2 import verify_receipt_file


@pytest.fixture()
def small_run():
    return run_hstep_bench(
        shards=["iid_gaussian", "garch_cluster"],
        n_train=256,
        n_eval=128,
        horizons=(1, 5),
        seed=3,
    )


def test_known_answer_gaussian_iid_on_iid_shard():
    """On the iid_gaussian shard the true h-step law IS gaussian_iid: it must
    sit near the top of the CRPS ranking and be roughly calibrated."""
    frame, receipt = run_hstep_bench(
        models={"gaussian_iid": HSTEP_MODELS["gaussian_iid"], "ewma_iid": HSTEP_MODELS["ewma_iid"]},
        shards=["iid_gaussian"],
        n_train=512,
        n_eval=256,
        horizons=(1,),
        seed=7,
    )
    row = next(r for r in frame.iter_rows(named=True) if r["model"] == "gaussian_iid")
    assert row["status"] == "ok"
    # True model: coverage near nominal (non-overlapping 1-step targets).
    assert abs(row["coverage_80"] - 0.8) < 0.15
    assert abs(row["coverage_90"] - 0.9) < 0.12
    # Gaussian truth ⇒ gaussian_iid pinball beats the EWMA lag at h=1 is not
    # guaranteed pointwise, but CRPS must be within noise of each other.
    ewma = next(r for r in frame.iter_rows(named=True) if r["model"] == "ewma_iid")
    assert abs(row["crps"] - ewma["crps"]) / row["crps"] < 0.2
    assert receipt["n_error_rows"] == 0


def test_ewma_beats_gaussian_on_clustered_shard():
    """GARCH clustering makes the unconditional Gaussian undercover at h=1;
    the vol-timing EWMA baseline must cover closer to nominal."""
    frame, _ = run_hstep_bench(
        shards=["garch_cluster"],
        n_train=512,
        n_eval=256,
        horizons=(1,),
        seed=11,
    )
    rows = {r["model"]: r for r in frame.iter_rows(named=True) if r["status"] == "ok"}
    assert rows["gaussian_iid"]["coverage_80"] < 0.75
    assert rows["ewma_iid"]["coverage_80"] > rows["gaussian_iid"]["coverage_80"]


def test_hstep_blocks_reach_the_bench(small_run):
    """The multi-horizon head's >1 cells are now honestly scored."""
    frame, receipt = small_run
    models = {r["model"] for r in frame.iter_rows(named=True)}
    assert {"hstep_t", "hstep_emp", "gaussian_iid", "ewma_iid"} <= models
    horizons = {r["horizon"] for r in frame.iter_rows(named=True)}
    assert horizons == {1, 5}
    # n_targets is the disjoint-block count: 128 // h.
    for r in frame.iter_rows(named=True):
        assert r["n_targets"] == 128 // r["horizon"]
    assert receipt["n_error_rows"] == 0


def test_determinism():
    _, r1 = run_hstep_bench(shards=["left_skew"], n_train=200, n_eval=64, horizons=(1, 2), seed=5)
    _, r2 = run_hstep_bench(shards=["left_skew"], n_train=200, n_eval=64, horizons=(1, 2), seed=5)
    assert r1["inputs_sha256"] == r2["inputs_sha256"]
    assert r1["results"] == r2["results"]


def test_fail_closed_args():
    with pytest.raises(ValueError):
        run_hstep_bench(shards=["bogus_shard"], n_train=64, n_eval=64)
    with pytest.raises(ValueError):
        run_hstep_bench(models={}, shards=["iid_gaussian"], n_train=64, n_eval=64)
    with pytest.raises(ValueError):
        run_hstep_bench(shards=["iid_gaussian"], n_train=64, n_eval=64, horizons=(0,))
    with pytest.raises(ValueError):
        run_hstep_bench(shards=["iid_gaussian"], n_train=64, n_eval=4, horizons=(1, 20))
    with pytest.raises(ValueError):
        resolve_hstep_models(["bogus_model"])


def test_error_rows_visible_not_silent():
    """A broken model produces error rows (verdict fail), never silence."""

    def _broken(y, horizons, taus):
        raise RuntimeError("planted failure")

    frame, receipt = run_hstep_bench(
        models={"broken": _broken},
        shards=["iid_gaussian"],
        n_train=128,
        n_eval=64,
        horizons=(1, 2),
        seed=1,
    )
    assert frame["status"].to_list() == ["error", "error"]
    assert "planted failure" in frame["error"][0]
    assert receipt["n_error_rows"] == 2
    # Error rows still seal — the receipt is honest evidence of the failure,
    # and a v2 envelope carries verdict=fail.
    assert hstep_bench_receipt_v2(receipt)["verdict"] == "fail"


def test_contract_and_v2_consistency(small_run):
    _, receipt = small_run
    assert hstep_bench_v1_contract_errors(receipt) == []
    envelope = hstep_bench_receipt_v2(receipt)
    assert envelope["kind"] == "hstep_bench"
    assert hstep_bench_v2_consistency_errors(envelope) == []

    # Tamper: flip a scored cell -> verdict_mismatch.
    tampered = json.loads(json.dumps(envelope))
    tampered["payload"]["results"][0]["status"] = "error"
    tampered["payload"]["n_error_rows"] = 1
    errs = hstep_bench_v2_consistency_errors(tampered)
    assert "verdict_mismatch" in errs

    # Tamper params.
    tampered2 = json.loads(json.dumps(envelope))
    tampered2["payload"]["seed"] = 999
    assert "params_hash_mismatch" in hstep_bench_v2_consistency_errors(tampered2)


def test_write_and_verify_round_trip(tmp_path, small_run):
    _, receipt = small_run
    p1 = write_hstep_bench_receipt(receipt, tmp_path)
    res1 = verify_receipt_file(p1)
    assert res1["valid"] is True and res1["errors"] == [], res1

    p2 = write_hstep_bench_receipt(receipt, tmp_path, receipt_version=2)
    res2 = verify_receipt_file(p2)
    assert res2["valid"] is True and res2["errors"] == [], res2

    # Immutable evidence: rewriting the same receipt is a no-op; different
    # content at the same path raises.
    again = write_hstep_bench_receipt(receipt, tmp_path)
    assert again == p1

    with pytest.raises(ValueError):
        write_hstep_bench_receipt(receipt, tmp_path, receipt_version=3)


def test_cli_end_to_end(tmp_path):
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "hstep-bench",
            "--shards",
            "iid_gaussian",
            "--n-train",
            "200",
            "--n-eval",
            "96",
            "--horizons",
            "1,4",
            "--out-dir",
            str(tmp_path),
        ],
    )
    assert res.exit_code == 0, res.output
    assert "DATA_LABEL=SYNTHETIC" in res.output
    assert "receipt=" in res.output
    receipts = list(tmp_path.glob("hstep_bench_*.json"))
    assert len(receipts) == 1
    assert verify_receipt_file(receipts[0])["errors"] == []

    # Bad model name is a clean Invalid value, not a traceback.
    res = runner.invoke(app, ["hstep-bench", "--models", "bogus"])
    assert res.exit_code == 2
    assert "unknown hstep model" in res.output


def test_report_renders(small_run):
    frame, _ = small_run
    text = hstep_bench_report(frame)
    assert "hstep_t" in text and "garch_cluster" in text and "cov90" in text
