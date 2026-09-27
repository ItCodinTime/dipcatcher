"""dipcatcher.sota.v2 slate helpers + gate stack.

Pure-function coverage for the lane plumbing: slate parsing, receipts,
window/drawdown math, and the sealed DM+RC+SPA+StepM gate stack. Lane
runners that need gold data are exercised through monkeypatched seams.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl
import pytest
import yaml

import quant_fund.hedge_lab.v2_slate as v2


def _cfg(tmp_path, *, source: str = "file") -> Any:
    return SimpleNamespace(
        data=SimpleNamespace(root=str(tmp_path), source=source, benchmark_id="SPY"),
        validation=SimpleNamespace(
            scheme="walk_forward", train_bars=200, val_bars=50, test_bars=50, embargo_bars=5
        ),
        embargo_bars=lambda: 5,
    )


def _write_slate(tmp_path, *, parent_exists: bool = True, **eval_over) -> Any:
    parent = tmp_path / "parent.yaml"
    if parent_exists:
        parent.write_text(yaml.safe_dump({"protocol_id": "sota_v2"}), encoding="utf-8")
    slate = {
        "protocol_id": "sota_v2_lanes",
        "slate_id": "lanes_v2",
        "parent_slate": str(parent),
        "evaluation": {
            "config": "configs/hedge_lab.yaml",
            "wide_config": "configs/hedge_lab_wide.yaml",
            "benchmark": "ridge",
            "one_way_cost": 0.001,
            "lane1_receipt": str(tmp_path / "ml_lane.json"),
            **eval_over,
        },
        "lanes": [
            {
                "lane": 2,
                "name": "horizon",
                "receipt": str(tmp_path / "lane2.json"),
                "challengers": ["gbrt"],
                "labels": ["future_idio_return_5"],
            },
            {
                "lane": 3,
                "name": "hmm_gate",
                "receipt": str(tmp_path / "lane3.json"),
                "book": "topk5_12_1",
                "gate": {
                    "fit_through": "2024-12-31",
                    "n_states": 3,
                    "seed": 7,
                    "lag_bars": 1,
                    "switch_cost": 0.001,
                },
            },
            {
                "lane": 4,
                "name": "bandit",
                "receipt": str(tmp_path / "lane4.json"),
                "actions": ["cash", "spy", "topk5"],
                "method": "ridge_q",
                "train_through": "2024-12-31",
                "state": {
                    "lag_bars": 1,
                    "columns": ["spy", "sv", "bv", "bdd", "sdd", "cm", "cv", "cs"],
                },
            },
            {
                "lane": 5,
                "name": "pit_universe_rebuild",
                "receipt": str(tmp_path / "lane5.json"),
                "label": "future_idio_return_1",
                "horizon_bars": 1,
                "embargo_bars": 1,
                "tape": "wide",
                "universe": "adv",
                "pit_rebuild": False,
            },
        ],
        "promotion": {"alpha": 0.05},
    }
    path = tmp_path / "slate.yaml"
    path.write_text(yaml.safe_dump(slate), encoding="utf-8")
    return str(path)


def test_sha256_file_matches_stdlib(tmp_path) -> None:
    f = tmp_path / "x.bin"
    f.write_bytes(b"hello slate")
    assert v2._sha256_file(f) == hashlib.sha256(b"hello slate").hexdigest()
    f.write_bytes(b"hello slate!")
    assert v2._sha256_file(f) != hashlib.sha256(b"hello slate").hexdigest()


def test_load_slate_real_config() -> None:
    slate, slate_sha, parent_sha = v2.load_slate()
    for key in ("slate_id", "parent_slate", "evaluation", "lanes", "promotion"):
        assert key in slate
    assert len(slate_sha) == 64 and len(parent_sha) == 64
    assert {int(lane["lane"]) for lane in slate["lanes"]} >= {2, 3, 4, 5}


def test_load_slate_rejects_non_mapping(tmp_path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        v2.load_slate(p)


def test_load_slate_missing_key(tmp_path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({"slate_id": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="parent_slate"):
        v2.load_slate(p)


def test_load_slate_missing_parent(tmp_path) -> None:
    path = _write_slate(tmp_path, parent_exists=False)
    with pytest.raises(FileNotFoundError, match="parent slate"):
        v2.load_slate(path)


def test_lane_spec_lookup_and_miss() -> None:
    slate = {"lanes": [{"lane": 3, "name": "g"}, {"lane": "5", "name": "w"}]}
    assert v2._lane_spec(slate, 3)["name"] == "g"
    assert v2._lane_spec(slate, 5)["name"] == "w"  # int() coercion
    with pytest.raises(ValueError, match="lane 9"):
        v2._lane_spec(slate, 9)


def test_fingerprint_deterministic_and_sensitive() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(512, 4))
    y = rng.normal(size=512)
    kw = {"scheme": "wf", "train_bars": 100, "val_bars": 20, "test_bars": 20, "embargo_bars": 5}
    fp1 = v2._fingerprint(x, y, ["a", "b"], "lbl", **kw)
    fp2 = v2._fingerprint(x, y, ["a", "b"], "lbl", **kw)
    assert fp1 == fp2 and len(fp1) == 16
    assert v2._fingerprint(x, y, ["a"], "lbl", **kw) != fp1
    assert v2._fingerprint(x, y, ["a", "b"], "other", **kw) != fp1
    assert v2._fingerprint(x, y, ["a", "b"], "lbl", **{**kw, "embargo_bars": 1}) != fp1
    assert v2._fingerprint(x + 1e-9, y, ["a", "b"], "lbl", **kw) != fp1


def test_window_mask_inclusive_bounds() -> None:
    import datetime as dt

    dates = [dt.date(2024, 12, 30), dt.date(2024, 12, 31), dt.date(2025, 1, 2)]
    mask = v2._window_mask(dates, start="2024-12-31", end="2025-01-02")
    assert mask.tolist() == [False, True, True]
    assert v2._window_mask(dates, start=None, end=None).all()
    assert not v2._window_mask(dates, start="2030-01-01", end=None).any()


def test_drawdown_known_path() -> None:
    r = np.array([0.10, -0.50, 0.10])
    dd = v2._drawdown(r)
    # wealth 1.10 -> 0.55 -> 0.605; peak 1.10
    assert dd[0] == pytest.approx(0.0)
    assert dd[1] == pytest.approx(-0.5)
    assert dd[2] == pytest.approx(0.605 / 1.10 - 1.0)
    # NaN treated as flat, not poison.
    dd2 = v2._drawdown(np.array([0.1, np.nan, -0.1]))
    assert np.isfinite(dd2).all()


def test_rolling_vol_shape_and_value() -> None:
    r = np.arange(30, dtype=float) * 0.01
    v = v2._rolling_vol(r, 5)
    assert np.isnan(v[:5]).all()
    assert v[5] == pytest.approx(float(np.std(r[:5], ddof=1)))
    assert v.shape == r.shape


def test_write_receipt_dual_writes(tmp_path) -> None:
    artifact = tmp_path / "artifacts" / "lane.json"
    out = v2._write_receipt({"a": 1}, str(artifact), tmp_path)
    assert json.loads(artifact.read_text()) == {"a": 1}
    meta = tmp_path / "metadata" / "lane.json"
    assert json.loads(meta.read_text()) == {"a": 1}
    assert out["artifact_path"] == str(artifact)
    assert out["metadata_path"] == str(meta)


def test_base_receipt_marks_synthetic(tmp_path) -> None:
    slate, sha, psha = v2.load_slate(_write_slate(tmp_path))
    lane = v2._lane_spec(slate, 3)
    rec = v2._base_receipt(slate, sha, psha, lane, _cfg(tmp_path, source="synthetic"), 100)
    assert rec["synthetic_not_promotable"] is True
    assert rec["blend_weight"] == 0.0 and rec["live_pnl_claim"] is False
    rec2 = v2._base_receipt(slate, sha, psha, lane, _cfg(tmp_path), 100)
    assert rec2["synthetic_not_promotable"] is False
    assert rec2["embargo_bars"] == 5


def _ic(card_means: dict[str, float], n: int = 40) -> dict[str, Any]:
    rng = np.random.default_rng(3)
    return {k: rng.normal(m, 0.01, n) for k, m in card_means.items()}


def test_lane_gates_missing_benchmark() -> None:
    with pytest.raises(ValueError, match="benchmark"):
        v2._lane_gates({"gbrt": np.ones(20)}, benchmark="ridge", n_boot=50, lags=None)


def test_lane_gates_short_series_fails_closed() -> None:
    out = v2._lane_gates(
        {"ridge": np.zeros(5), "gbrt": np.ones(5)}, benchmark="ridge", n_boot=50, lags=None
    )
    assert out["promote"] is False
    assert np.isnan(out["reality_check_p"])
    assert out["dm"] == {}


def test_lane_gates_dominant_challenger_promotes() -> None:
    aligned = _ic({"ridge": 0.0, "gbrt": 0.10}, n=60)
    out = v2._lane_gates(aligned, benchmark="ridge", n_boot=200, lags=1)
    assert out["promote"] is True
    assert out["cleared"] == ["gbrt"]
    assert out["dm"]["gbrt"]["preferred"] == "gbrt"
    assert out["dm"]["gbrt"]["p_value"] < 0.05
    assert "gbrt" in out["stepm_rejected"]


def test_lane_gates_worse_challenger_does_not_promote() -> None:
    aligned = _ic({"ridge": 0.0, "gbrt": -0.10}, n=60)
    out = v2._lane_gates(aligned, benchmark="ridge", n_boot=200, lags=1)
    assert out["promote"] is False
    assert out["cleared"] == []


def test_lane_gates_tie_not_promoted() -> None:
    rng = np.random.default_rng(11)
    base = rng.normal(0.0, 0.02, 60)
    out = v2._lane_gates(
        {"ridge": base, "gbrt": base.copy()}, benchmark="ridge", n_boot=100, lags=None
    )
    assert out["promote"] is False


def _synthetic_blob(tmp_path, engines, *, n_dates: int = 240, n_names: int = 8) -> dict:
    """Scores blob mimicking ``_oos_scores``: challenger dominates benchmark.

    Dates span both the selection window (through 2024-12-31) and the frozen
    holdout (from 2025-01-02) so the windowed gates each see >10 days.
    """
    rng = np.random.default_rng(5)
    dates = np.repeat(
        [
            d.strftime("%Y-%m-%d")
            for d in [
                __import__("datetime").date(2024, 8, 1) + __import__("datetime").timedelta(days=i)
                for i in range(n_dates)
            ]
        ],
        n_names,
    )
    ids = np.tile([f"S{i}" for i in range(n_names)], n_dates)
    signal = rng.normal(size=dates.size)
    y = signal + rng.normal(scale=0.5, size=dates.size)
    scores = {
        "ridge": rng.normal(size=dates.size),
        "gbrt": signal + rng.normal(scale=0.05, size=dates.size),
    }
    return {
        "x": np.zeros((dates.size, 2)),
        "y": y,
        "dates": dates,
        "ids": ids,
        "used": ["f1"],
        "horizon": 5,
        "fingerprint": "fp",
        "n_dates": n_dates,
        "scores": {e: scores.get(e, scores["gbrt"]) for e in engines},
    }


def test_run_lane2_end_to_end(tmp_path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(v2, "load_config", lambda path: cfg)
    monkeypatch.setattr(
        v2,
        "_oos_scores",
        lambda c, label, engines, use_cache=True: _synthetic_blob(tmp_path, engines),
    )
    out = v2.run_lane2(_write_slate(tmp_path), n_boot=100)
    assert out["catalog"] == "hedge_lab_analytics"
    assert out["engines"] == ["ridge", "gbrt"]
    res = out["results"]["future_idio_return_5"]
    assert res["embargo_bars"] == 5
    for w in ("full", "selection", "holdout"):
        assert w in res["windows"]
        assert res["windows"][w]["gates"]["dm"]
    # Dominant challenger clears selection+holdout on a file-source tape.
    assert out["promote"] is True
    assert (tmp_path / "lane2.json").is_file()
    assert (tmp_path / "metadata" / "lane2.json").is_file()


def test_run_lane2_synthetic_source_blocks_promote(tmp_path, monkeypatch) -> None:
    cfg = _cfg(tmp_path, source="synthetic")
    monkeypatch.setattr(v2, "load_config", lambda path: cfg)
    monkeypatch.setattr(
        v2,
        "_oos_scores",
        lambda c, label, engines, use_cache=True: _synthetic_blob(tmp_path, engines),
    )
    out = v2.run_lane2(_write_slate(tmp_path), n_boot=100)
    assert out["promote"] is False  # synthetic tapes can never promote


def test_run_lane1_calibration_and_anded(tmp_path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(v2, "load_config", lambda path: cfg)
    monkeypatch.setattr(v2, "_calibration", lambda c: {"family": "calibration_gate", "pass": True})
    (tmp_path / "metadata").mkdir()
    lane1 = tmp_path / "ml_lane.json"
    lane1.write_text(json.dumps({"promote": True, "data_source": "file"}), encoding="utf-8")
    out = v2.run_lane1_calibration(_write_slate(tmp_path))
    assert out["promote"] is True
    assert out["promote_statistical_gates"] is True
    assert out["calibration_required_for_blend"] is True
    assert out["calibration_pass"] is True
    meta = tmp_path / "metadata" / "ml_lane.json"
    assert json.loads(meta.read_text())["promote"] is True


def test_run_lane1_calibration_failing_gate_blocks(tmp_path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(v2, "load_config", lambda path: cfg)
    monkeypatch.setattr(v2, "_calibration", lambda c: {"family": "calibration_gate", "pass": False})
    (tmp_path / "metadata").mkdir()
    lane1 = tmp_path / "ml_lane.json"
    lane1.write_text(json.dumps({"promote": True, "data_source": "file"}), encoding="utf-8")
    out = v2.run_lane1_calibration(_write_slate(tmp_path))
    assert out["promote"] is False
    assert out["promote_statistical_gates"] is True


def test_run_lane1_calibration_missing_receipt(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(v2, "load_config", lambda path: _cfg(tmp_path))
    with pytest.raises(FileNotFoundError, match="lane-1 receipt"):
        v2.run_lane1_calibration(_write_slate(tmp_path))


def _write_labels(tmp_path, n: int = 320, names: int = 8) -> None:
    """Gold labels parquet with close prices incl. SPY, spanning the holdout."""
    rng = np.random.default_rng(2)
    start = __import__("datetime").date(2024, 6, 3)
    dates = [start + __import__("datetime").timedelta(days=i) for i in range(n)]
    tickers = ["SPY", *[f"S{i}" for i in range(names - 1)]]
    rows_t, rows_s, rows_c = [], [], []
    for t in tickers:
        px = 100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.01, n))
        for d, c in zip(dates, px, strict=True):
            rows_t.append(d)
            rows_s.append(t)
            rows_c.append(float(c))
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"event_time": rows_t, "security_id": rows_s, "close": rows_c}).write_parquet(
        gold_dir / "labels.parquet"
    )


def test_book_inputs_pivots_labels(tmp_path, monkeypatch) -> None:
    _write_labels(tmp_path)
    dates, closes = v2._book_inputs(_cfg(tmp_path))
    assert "SPY" in closes
    assert len(dates) == 320
    assert all(v.shape == (320,) for v in closes.values())


def test_run_lane3_hmm_gate(tmp_path, monkeypatch) -> None:
    pytest.importorskip("hmmlearn")
    _write_labels(tmp_path)
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(v2, "load_config", lambda path: cfg)
    out = v2.run_lane3(_write_slate(tmp_path), n_boot=100)
    assert out["promote"] is False
    assert out["affects_blend_weight"] is False
    assert out["blend_weight"] == 0.0
    assert "stress" in out["hmm_labels"].values()
    assert 0.0 <= out["multiplier_holdout_mean"] <= 1.0
    for w in ("selection", "holdout", "full"):
        assert out["windows"][w]["n_returns"] > 0
        assert "hmm_gated_topk5" in out["windows"][w]["cards"]
    card = out["windows"]["holdout"]["cards"]["hmm_gated_topk5"]
    assert card["research_only"] is True
    assert (tmp_path / "lane3.json").is_file()


def test_run_lane3_requires_spy(tmp_path, monkeypatch) -> None:
    pytest.importorskip("hmmlearn")
    _write_labels(tmp_path)
    # Rewrite labels without SPY.
    gold = tmp_path / "gold" / "labels.parquet"
    frame = pl.read_parquet(gold).filter(pl.col("security_id") != "SPY")
    frame.write_parquet(gold)
    monkeypatch.setattr(v2, "load_config", lambda path: _cfg(tmp_path))
    with pytest.raises(ValueError, match="SPY"):
        v2.run_lane3(_write_slate(tmp_path), n_boot=50)


def test_run_lane4_bandit(tmp_path, monkeypatch) -> None:
    _write_labels(tmp_path)
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(v2, "load_config", lambda path: cfg)
    (tmp_path / "ml_lane.json").write_text("{}", encoding="utf-8")

    def fake_panel(c):
        labels = pl.read_parquet(tmp_path / "gold" / "labels.parquet")
        rng = np.random.default_rng(4)
        return labels.with_columns(
            pl.Series("cs_z_mom_12_1", rng.normal(size=labels.height)),
            pl.Series("cs_z_vol_20", rng.normal(size=labels.height)),
        )

    monkeypatch.setattr(v2, "panel", fake_panel)
    out = v2.run_lane4(_write_slate(tmp_path), n_boot=100)
    assert out["promote"] is False
    assert out["affects_blend_weight"] is False
    assert out["actions"] == ["cash", "spy", "topk5"]
    assert out["n_train_dates"] > 0
    total = sum(out["holdout_action_counts"].values())
    assert total == out["windows"]["holdout"]["n_returns"]
    assert (tmp_path / "lane4.json").is_file()


def test_run_lane4_requires_lane1(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(v2, "load_config", lambda path: _cfg(tmp_path))
    with pytest.raises(FileNotFoundError, match="lane-1 receipt"):
        v2.run_lane4(_write_slate(tmp_path), n_boot=50)
