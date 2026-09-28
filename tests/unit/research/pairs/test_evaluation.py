"""Eval lane: planted detection truth, receipt contract, table formatting."""

from __future__ import annotations

import json
import math
from pathlib import Path

import polars as pl
import pytest

from quant_fund.research.pairs.evaluation import (
    PAIRS_SCHEMA,
    format_pairs_table,
    run_pairs_eval,
    write_pairs_receipt,
)


@pytest.fixture(scope="module")
def eval_result():
    return run_pairs_eval(seed=0)


class TestPlantedDetection:
    def test_planted_pair_detected_rank_one(self, eval_result):
        frame, receipt = eval_result
        assert isinstance(frame, pl.DataFrame)
        planted = receipt["planted"]
        assert planted["pair"] == [0, 1]
        assert planted["rank_by_p_bh"] == 1
        assert planted["detected"] is True

    def test_alignment_statistics_finite(self, eval_result):
        _, receipt = eval_result
        al = receipt["alignment"]
        assert al["n_obs"] > 100
        # Planted AR(1) spread reverts: -z should correlate positively with
        # the forward spread change (the reversion direction).
        assert al["spearman_ic"] > 0.0
        assert al["pearson_ic"] > 0.0
        assert math.isfinite(al["band_hit_rate"])

    def test_descriptive_stats(self, eval_result):
        _, receipt = eval_result
        d = receipt["descriptive"]
        assert d["n_pairs_tested"] >= 1
        assert d["true_half_life"] == pytest.approx(6.5788, rel=1e-3)
        assert d["est_half_life"] == pytest.approx(d["true_half_life"], rel=0.6)
        assert 0.0 <= d["share_in_position"] <= 1.0
        assert d["n_entries"] >= 1


class TestReceipt:
    def test_contract_fields(self, eval_result):
        _, receipt = eval_result
        assert receipt["schema"] == PAIRS_SCHEMA
        assert receipt["data_label"] == "SYNTHETIC"
        assert receipt["live_pnl_claim"] is False
        assert receipt["kind"] == "stat_arb_pairs_eval"
        assert isinstance(receipt["screen"], list) and receipt["screen"]
        assert receipt["notes"]

    def test_write_and_reread(self, eval_result, tmp_path: Path):
        _, receipt = eval_result
        path = write_pairs_receipt(receipt, tmp_path)
        assert path.name.startswith("stat_arb_pairs_eval_")
        payload = json.loads(path.read_text())
        assert payload["schema"] == PAIRS_SCHEMA
        assert payload["receipt_sha256"][:16] in path.name
        # Idempotent rewrite of identical content is allowed.
        assert write_pairs_receipt(receipt, tmp_path) == path

    def test_reject_tampered(self, eval_result, tmp_path: Path):
        _, receipt = eval_result
        bad = dict(receipt)
        bad["live_pnl_claim"] = True
        with pytest.raises(ValueError):
            write_pairs_receipt(bad, tmp_path)
        bad2 = dict(receipt)
        bad2["data_label"] = "REAL"
        with pytest.raises(ValueError):
            write_pairs_receipt(bad2, tmp_path)
        bad3 = dict(receipt)
        bad3["screen"] = []
        with pytest.raises(ValueError):
            write_pairs_receipt(bad3, tmp_path)
        # Forbidden-metric key anywhere in the blob fails closed.
        bad4 = dict(receipt)
        bad4["descriptive"] = {**receipt["descriptive"], "sharpe_annualized": 1.0}
        with pytest.raises(ValueError):
            write_pairs_receipt(bad4, tmp_path)


def test_format_table(eval_result):
    frame, _ = eval_result
    text = format_pairs_table(frame)
    assert "p_bh" in text.splitlines()[0]
    assert len(text.splitlines()) == min(15, frame.height) + 1


def test_determinism():
    _, r0 = run_pairs_eval(seed=0)
    _, r1 = run_pairs_eval(seed=0)
    assert r0["inputs_sha256"] == r1["inputs_sha256"]
    assert r0["planted"]["rank_by_p_bh"] == r1["planted"]["rank_by_p_bh"]
