"""v2_slate seams: score cache, fail-closed calibration, lane 5, CLI dispatch."""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl
import pytest
import yaml
from numpy.testing import assert_allclose

import quant_fund.hedge_lab.v2_slate as v2

N_NAMES = 8
DATES = [date(2024, 11, 4) + timedelta(days=i) for i in range(80)]


def _cfg(tmp_path: Path, *, source: str = "file") -> Any:
    return SimpleNamespace(
        data=SimpleNamespace(root=str(tmp_path), source=source, benchmark_id="SPY"),
        validation=SimpleNamespace(
            scheme="expanding",
            train_bars=15,
            val_bars=4,
            test_bars=5,
            embargo_bars=1,
        ),
        embargo_bars=lambda: 1,
    )


def _blob(tmp_path: Path, engines: list[str]) -> dict[str, Any]:
    """Deterministic scores blob; gbrt/nautica dominate, ridge is noise."""
    rng = np.random.default_rng(9)
    n_dates = len(DATES)
    dates = np.repeat(DATES, N_NAMES)
    ids = np.tile([f"S{i}" for i in range(N_NAMES)], n_dates)
    signal = rng.normal(size=dates.size)
    y = signal + rng.normal(scale=0.4, size=dates.size)
    scores = {
        name: (y + rng.normal(scale=0.1, size=y.size))
        if name in {"gbrt", "nautica", "tsmom"}
        else rng.normal(size=y.size)
        for name in engines
    }
    return {
        "x": np.column_stack([signal, rng.normal(size=dates.size)]),
        "y": y,
        "dates": dates,
        "ids": ids,
        "used": ["f_signal"],
        "horizon": 1,
        "fingerprint": "fp",
        "n_dates": n_dates,
        "scores": scores,
    }


def _write_slate(tmp_path: Path) -> str:
    parent = tmp_path / "parent.yaml"
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
        },
        "lanes": [
            {
                "lane": 5,
                "name": "pit_universe_rebuild",
                "receipt": str(tmp_path / "lane5.json"),
                "label": "future_idio_return_1",
                "horizon_bars": 1,
                "embargo_bars": 1,
                "tape": "wide",
                "universe": "adv",
                "pit_rebuild": True,
            }
        ],
        "promotion": {"alpha": 0.05},
    }
    path = tmp_path / "slate.yaml"
    path.write_text(yaml.safe_dump(slate), encoding="utf-8")
    return str(path)


class TestOosScoresCache:
    def test_fit_then_cached_roundtrip(self, tmp_path, monkeypatch) -> None:
        cfg = _cfg(tmp_path)
        rng = np.random.default_rng(3)
        dates = np.repeat(DATES[:20], 2)
        x = rng.normal(size=(dates.size, 2))
        y = rng.normal(size=dates.size)
        ids = np.tile(["A", "B"], 20)
        gold = pl.DataFrame(
            {
                "event_time": dates.tolist(),
                "security_id": ids.tolist(),
                "close": np.ones(dates.size),
            }
        )
        calls: list[str] = []

        def fake_oos(engine, c, x_, y_, d_, i_, *, horizon_bars, feature_names):
            calls.append(engine)
            return rng.normal(size=y_.size)

        monkeypatch.setattr(v2, "panel", lambda c: gold)
        monkeypatch.setattr(v2, "design_matrix", lambda *a, **k: (x, y, dates, ["f1"], ids))
        monkeypatch.setattr(v2, "oos_rank_scores", fake_oos)

        blob1 = v2._oos_scores(cfg, "future_idio_return_1", ["ridge"])
        assert calls == ["ridge"]
        cache_dir = tmp_path / "metadata" / "oos_scores_v2_future_idio_return_1"
        assert list(cache_dir.glob("ridge_*.npy"))

        # Second call hits the cache — engine is not refit.
        blob2 = v2._oos_scores(cfg, "future_idio_return_1", ["ridge"])
        assert calls == ["ridge"]
        assert_allclose(blob1["scores"]["ridge"], blob2["scores"]["ridge"])
        assert blob2["fingerprint"] == blob1["fingerprint"]

    def test_use_cache_false_skips_disk(self, tmp_path, monkeypatch) -> None:
        cfg = _cfg(tmp_path)
        dates = np.repeat(DATES[:10], 2)
        x = np.ones((dates.size, 1))
        y = np.ones(dates.size)
        ids = np.tile(["A", "B"], 10)
        gold = pl.DataFrame(
            {
                "event_time": dates.tolist(),
                "security_id": ids.tolist(),
                "close": np.ones(dates.size),
            }
        )
        monkeypatch.setattr(v2, "panel", lambda c: gold)
        monkeypatch.setattr(v2, "design_matrix", lambda *a, **k: (x, y, dates, ["f1"], ids))
        monkeypatch.setattr(
            v2,
            "oos_rank_scores",
            lambda *a, **k: np.ones(y.size),
        )
        blob = v2._oos_scores(cfg, "future_idio_return_1", ["ridge"], use_cache=False)
        assert_allclose(blob["scores"]["ridge"], np.ones(dates.size))


class TestCalibrationFailClosed:
    def test_panel_error_returns_blocked_blob(self, tmp_path, monkeypatch) -> None:
        def boom(_cfg):
            raise OSError("lake offline")

        monkeypatch.setattr(v2, "panel", boom)
        out = v2._calibration(_cfg(tmp_path))
        assert out["family"] == "calibration_gate"
        assert out["pass"] is False
        assert out["recorded"] is False
        assert "lake offline" in out["error"]


class TestLaneGatesEdges:
    def test_all_challengers_short_fails_closed(self) -> None:
        out = v2._lane_gates(
            {"ridge": np.ones(20), "champ": np.ones(5)},
            benchmark="ridge",
            n_boot=50,
            lags=None,
        )
        assert out["promote"] is False
        assert out["dm"] == {}
        assert np.isnan(out["reality_check_p"])


class TestRunLane5:
    def test_pit_rebuild_receipt(self, tmp_path, monkeypatch) -> None:
        cfg = _cfg(tmp_path)
        monkeypatch.setattr(v2, "load_config", lambda path: cfg)
        blob = _blob(tmp_path, ["ridge", "gbrt", "lambdarank", "xgboost"])
        monkeypatch.setattr(v2, "_oos_scores", lambda c, label, engines, use_cache=True: blob)
        feats = pl.DataFrame({"feature_set_version": ["v2-pit"]})
        labs = pl.DataFrame({"x": [1.0]})
        monkeypatch.setattr(v2, "build_gold", lambda c: (feats, labs))
        (tmp_path / "ml_lane.json").write_text("{}", encoding="utf-8")

        out = v2.run_lane5(_write_slate(tmp_path), n_boot=60)
        assert out["best_tree"] == "gbrt"  # top selection-window IC
        assert set(out["engines"]) == {"ridge", "gbrt"}
        assert out["tape"] == "wide"
        assert out["decision_window"] == "holdout"
        assert out["pit_rebuild"]["gold_rebuilt"] is True
        assert out["pit_rebuild"]["feature_set_version"] == ["v2-pit"]
        assert set(out["windows"]) == {"full", "selection", "holdout"}
        # The promote bit mirrors exactly the holdout gate on a file tape.
        assert out["promote"] == bool(out["windows"]["holdout"]["gates"]["promote"])
        assert (tmp_path / "lane5.json").is_file()
        assert (tmp_path / "metadata" / "lane5.json").is_file()

    def test_requires_lane1_receipt(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(v2, "load_config", lambda p: _cfg(tmp_path))
        with pytest.raises(FileNotFoundError, match="lane-1 receipt"):
            v2.run_lane5(_write_slate(tmp_path), n_boot=50)


class TestCliDispatch:
    def test_cal_lane_prints_json(self, tmp_path, monkeypatch, capsys) -> None:
        monkeypatch.setattr(
            v2,
            "run_lane1_calibration",
            lambda slate: {"promote": True, "calibration_pass": True},
        )
        monkeypatch.setattr(
            sys, "argv", ["v2", "--lane", "cal", "--slate", str(tmp_path / "s.yaml")]
        )
        v2.main()
        out = json.loads(capsys.readouterr().out)
        assert out == {"promote": True, "calibration_pass": True}

    def test_numbered_lane_dispatch(self, tmp_path, monkeypatch, capsys) -> None:
        monkeypatch.setattr(
            v2,
            "run_lane5",
            lambda slate, **kw: {"promote": False, "artifact_path": "/x/lane5.json"},
        )
        monkeypatch.setattr(
            sys,
            "argv",
            ["v2", "--lane", "5", "--n-boot", "7", "--no-cache"],
        )
        v2.main()
        out = json.loads(capsys.readouterr().out)
        assert out == {"promote": False, "artifact": "/x/lane5.json"}
