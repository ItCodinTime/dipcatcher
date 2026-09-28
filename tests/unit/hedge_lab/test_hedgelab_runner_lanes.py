"""End-to-end hedge_lab receipt writers over a synthetic tape.

``load_config`` / ``panel`` / ``design_matrix`` / ``oos_rank_scores`` are
monkeypatched seams; every gate, book, and receipt write is real.
"""

from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl
import pytest

# The runners import these inside the function, so patch the source modules.
import quant_fund.config as config_mod
import quant_fund.hedge_lab.gated_race as gated_race
import quant_fund.hedge_lab.lightspeed_book as lightspeed_book
import quant_fund.hedge_lab.mirror as mirror_mod
import quant_fund.pipeline.dataset as dataset_mod
import quant_fund.research.benches as benches_mod

# Calendar spans the frozen cut: ~40 selection dates, ~20 holdout dates.
N_NAMES = 8
DATES = [date(2024, 11, 4) + timedelta(days=i) for i in range(80)]
_NAMES = [f"S{i}" for i in range(N_NAMES)]


def _cfg(tmp_path: Path, *, source: str = "file") -> Any:
    return SimpleNamespace(
        data=SimpleNamespace(root=str(tmp_path), source=source, benchmark_id="SPY"),
        validation=SimpleNamespace(
            scheme="expanding", train_bars=15, val_bars=4, test_bars=5, embargo_bars=1
        ),
        embargo_bars=lambda: 1,
    )


def _matrix() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Design matrix spanning the frozen holdout cut (2025-01-02)."""
    dates = np.repeat(DATES, N_NAMES)
    ids = np.tile(_NAMES, len(DATES))
    rng = np.random.default_rng(11)
    signal = rng.normal(size=dates.size)
    x = np.column_stack([signal, rng.normal(size=dates.size)])
    y = signal + rng.normal(scale=0.5, size=dates.size)
    return x, y, dates, ["f_signal", "f_noise"], ids


def _install_cs_mocks(monkeypatch, tmp_path: Path, *, source: str = "file") -> Any:
    cfg = _cfg(tmp_path, source=source)
    monkeypatch.setattr(config_mod, "load_config", lambda path: cfg)
    x, y, dates, used, ids = _matrix()
    rng = np.random.default_rng(7)
    # challenger scores track the label; ridge is near noise.
    score_map: dict[str, np.ndarray] = {}

    def fake_scores(name, c, x_, y_, dates_, ids_, *, horizon_bars, feature_names):
        if name not in score_map:
            if name in {"nautica", "tsmom", "gbrt"}:
                score_map[name] = y_ + rng.normal(scale=0.1, size=y_.size)
            else:
                score_map[name] = rng.normal(size=y_.size)
        return score_map[name]

    gold = pl.DataFrame(
        {
            "event_time": dates.tolist() + dates.tolist(),
            "security_id": ids.tolist() + ["SPY"] * dates.size,
            "close": np.abs(rng.normal(100.0, 5.0, dates.size * 2)).tolist(),
        }
    )
    monkeypatch.setattr(dataset_mod, "panel", lambda c: gold)
    monkeypatch.setattr(dataset_mod, "design_matrix", lambda *a, **k: (x, y, dates, used, ids))
    monkeypatch.setattr(benches_mod, "oos_rank_scores", fake_scores)
    return cfg


class TestGatedRace:
    def test_receipt_shape_and_promotion(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        _install_cs_mocks(monkeypatch, tmp_path)
        out = gated_race.run_gated_race(engines=("ridge", "nautica"), n_boot=60, cpu_fraction=0.5)
        assert out["catalog"] == "hedge_lab_analytics"
        assert out["engines"] == ["ridge", "nautica"]
        assert out["champion"] == "ridge"
        assert out["champion_alias"] is False
        assert out["blend_weight"] == 0.0
        assert out["live_pnl_claim"] is False
        assert out["synthetic_not_promotable"] is False
        assert set(out["gated_ls"]["nautica"]) >= {
            "raw",
            "riskstack",
            "stepm_stack",
            "riskstack_gate",
            "stepm_gate",
        }
        assert "selection_gates" in out and "holdout_gates" in out
        # A dominant challenger that clears both windows promotes.
        assert out["gates"]["promote"] is True
        assert "nautica" in out["gates"]["cleared"]
        assert out["promote"] is True
        assert "nautica" in out["cleared_both"]
        assert out["best_dd_safe"] is not None
        receipt = tmp_path / "metadata" / "gated_race.json"
        assert json.loads(receipt.read_text())["promote"] is True
        assert (tmp_path / "artifacts" / "hedge_lab" / "gated_race.json").is_file()

    def test_synthetic_source_blocks_promote(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        _install_cs_mocks(monkeypatch, tmp_path, source="synthetic")
        out = gated_race.run_gated_race(
            engines=("ridge", "nautica"),
            n_boot=50,
            artifact_name="gated_race_syn.json",
        )
        assert out["promote"] is False
        assert out["synthetic_not_promotable"] is True
        assert out["data_source"] == "SYNTHETIC"

    def test_no_strong_challenger_stays_flat(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        _install_cs_mocks(monkeypatch, tmp_path)
        out = gated_race.run_gated_race(engines=("ridge", "classic"), n_boot=50)
        assert out["cleared_both"] == []
        assert out["promote"] is False


class TestHoldoutConfirm:
    def test_three_windows(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        _install_cs_mocks(monkeypatch, tmp_path)
        out = gated_race.run_holdout_confirm(n_boot=50)
        assert out["engines"] == list(gated_race.CONFIRM_ENGINES)
        for window in ("full", "selection", "holdout"):
            blob = out[window]
            assert "cs_ic" in blob and "gates" in blob
            assert set(blob["cs_ls"]) == set(gated_race.CONFIRM_ENGINES)
            slim = blob["cs_ic"][0]
            assert "ic_series" not in slim
        assert out["selection"]["window_end"] == "2024-12-31"
        assert out["holdout"]["window_start"] == "2025-01-02"
        assert out["holdout"]["cs_ic"][0]["n_dates"] < out["full"]["cs_ic"][0]["n_dates"]
        assert out["promote"] is False
        assert out["champion_alias"] is False
        assert (tmp_path / "metadata" / "holdout_confirm.json").is_file()
        assert (tmp_path / "artifacts" / "hedge_lab" / "holdout_confirm.json").is_file()


def _price_gold() -> pl.DataFrame:
    """Price panel covering the momentum universe + SPY/QQQ, 320 sessions."""
    rng = np.random.default_rng(21)
    start = date(2024, 1, 2)
    days = [start + timedelta(days=i) for i in range(440)]
    days = [d for d in days if d.weekday() < 5][:320]
    tickers = (
        "SPY",
        "QQQ",
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "GOOGL",
        "META",
        "AVGO",
        "TSLA",
        "XLK",
        "PLTR",
        "MU",
    )
    rows_t, rows_s, rows_c = [], [], []
    for t in tickers:
        drift = 0.0004 if t not in {"SPY"} else 0.0003
        px = 100.0 * np.cumprod(1.0 + rng.normal(drift, 0.012, len(days)))
        for d, c in zip(days, px, strict=True):
            rows_t.append(d)
            rows_s.append(t)
            rows_c.append(float(c))
    return pl.DataFrame({"event_time": rows_t, "security_id": rows_s, "close": rows_c})


def _install_book_mocks(monkeypatch, tmp_path: Path) -> Any:
    """CS mocks + a real price panel for the momentum/rotation books."""
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(config_mod, "load_config", lambda path: cfg)
    x, y, dates, used, ids = _matrix()
    rng = np.random.default_rng(13)
    score_map: dict[str, np.ndarray] = {}

    def fake_scores(name, c, x_, y_, dates_, ids_, *, horizon_bars, feature_names):
        if name not in score_map:
            if name == "nautica":
                score_map[name] = y_ + rng.normal(scale=0.1, size=y_.size)
            else:
                score_map[name] = rng.normal(size=y_.size)
        return score_map[name]

    gold = _price_gold()
    monkeypatch.setattr(dataset_mod, "panel", lambda c: gold)
    monkeypatch.setattr(dataset_mod, "design_matrix", lambda *a, **k: (x, y, dates, used, ids))
    monkeypatch.setattr(benches_mod, "oos_rank_scores", fake_scores)
    return cfg


class TestLightspeedFileBook:
    def test_full_receipt_with_books_and_rotation(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        _install_book_mocks(monkeypatch, tmp_path)
        out = lightspeed_book.run_lightspeed_file_book(n_boot=60)
        assert out["catalog"] == "hedge_lab_analytics"
        assert out["champion"] == "ridge"
        assert out["champion_alias"] is False
        assert out["blend_weight"] == 0.0
        assert out["broker"] is None
        assert out["live_pnl_claim"] is False
        assert len(out["cs_ic"]) == 3
        # Both momentum families ran on the taped names.
        assert "stock-momentum-v1" in out["momentum_books"]
        book = out["momentum_books"]["stock-momentum-v1"]
        assert book["n_risk_names"] >= 5  # tape ∩ the 19-name frozen universe
        for w in ("full", "is", "holdout"):
            assert book["windows"][w]["research_only"] is True
        assert book["spy_buyhold_windows"] is not None
        # TQQQ rotation: reconstructed 3x + metalabel gate ran on QQQ.
        rot = out["tqqq_rotation"]
        assert rot is not None
        assert rot["tqqq"] == "reconstructed_3x_qqq_daily"
        assert rot["metalabel_windows"]["full"]["research_only"] is True
        assert math.isfinite(rot["metalabel_last_multiplier"])
        assert (tmp_path / "metadata" / "lightspeed_book.json").is_file()
        assert (tmp_path / "artifacts" / "hedge_lab" / "lightspeed_book.json").is_file()

    def test_receipt_writes_when_no_benchmarks(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        _install_book_mocks(monkeypatch, tmp_path)
        gold = _price_gold().filter(~pl.col("security_id").is_in(["SPY", "QQQ"]))
        monkeypatch.setattr(dataset_mod, "panel", lambda c: gold)
        out = lightspeed_book.run_lightspeed_file_book(n_boot=40)
        assert out["tqqq_rotation"] is None
        for book in out["momentum_books"].values():
            assert book["spy_buyhold_windows"] is None
        assert out["receipt_path"].endswith("lightspeed_book.json")


class TestFileTapeMirror:
    def test_receipt_pairs_and_rows(self, tmp_path, monkeypatch) -> None:
        # Public artifact write is cwd-relative; sandbox it under tmp_path.
        monkeypatch.chdir(tmp_path)
        _install_book_mocks(monkeypatch, tmp_path)
        out = mirror_mod.run_file_tape_mirror()
        assert out["catalog"] == "hedge_lab_analytics"
        assert out["champion_alias"] is False
        assert out["blend_weight"] == 0.0
        assert out["live_pnl_claim"] is False
        engines = {"reversal", "classic_st", "ridge", "combo_ic", "anti_univ", "random"}
        named = {row["name"].rsplit("_c", 1)[0] for row in out["rows"]}
        assert engines == named
        # Two cost levels per engine.
        assert len(out["rows"]) == 2 * len(engines)
        for row in out["rows"]:
            assert math.isfinite(row["book_sharpe"])
            assert row["live_pnl_claim"] is False
            assert row["sharpe_sum"] == pytest.approx(row["book_sharpe"] + row["mirror_sharpe"])
        assert (tmp_path / "metadata" / "mirror_anti_future_idio_return_1.json").is_file()
        assert (tmp_path / "artifacts" / "hedge_lab" / "mirror_latest.json").is_file()


@pytest.mark.synthetic
def test_run_hedge_lab_mirror_branch(tmp_path: Path) -> None:
    """mirror=True exercises the sign-flip book block + mirror artifact."""
    from quant_fund.config import load_config
    from quant_fund.hedge_lab.runner import HedgeLabProtocol, run_hedge_lab

    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.data.synthetic_n_assets = 6
    cfg.data.synthetic_n_days = 90
    cfg.universe.min_history_bars = 8
    cfg.fusion.skip_intervals = True
    cfg.validation.train_bars = 30
    cfg.validation.val_bars = 10
    cfg.validation.test_bars = 10
    cfg.risk_gate.stale_price_bars = 63
    proto = HedgeLabProtocol(
        rebalance_every=30,
        lookback_bars=15,
        bootstrap=False,
        claim_ram=False,
        mirror=True,
    )
    receipt = run_hedge_lab(cfg, proto, claim_ram=False)
    assert receipt["synthetic_not_promotable"] is True
    mirror = receipt["mirror"]
    assert mirror["enabled"] is True
    assert mirror["status"] == "ok"
    assert math.isfinite(mirror["economic"]["sharpe"])
    assert "backtest" in mirror
    assert (tmp_path / "artifacts" / "hedge_lab" / "equity_mirror.parquet").is_file()
    assert mirror["bootstrap"]["status"] == "skipped"


@pytest.mark.synthetic
def test_run_hedge_lab_risk_diag_and_refresh(tmp_path: Path) -> None:
    """refresh_gold=True hits the rebuild branch; >=20 returns runs VaR/Kupiec."""
    from quant_fund.config import load_config
    from quant_fund.hedge_lab.runner import HedgeLabProtocol, run_hedge_lab

    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.data.synthetic_n_assets = 6
    cfg.data.synthetic_n_days = 120
    cfg.universe.min_history_bars = 8
    cfg.fusion.skip_intervals = True
    cfg.validation.train_bars = 30
    cfg.validation.val_bars = 10
    cfg.validation.test_bars = 10
    cfg.risk_gate.stale_price_bars = 63
    proto = HedgeLabProtocol(
        rebalance_every=10,
        lookback_bars=10,
        bootstrap=False,
        claim_ram=False,
    )
    receipt = run_hedge_lab(cfg, proto, claim_ram=False, refresh_gold=True)
    assert receipt["synthetic_not_promotable"] is True
    assert receipt["economic"]["n_returns"] >= 20  # precondition for risk diag
    diag = receipt["market_risk"]
    for key in (
        "empirical_var",
        "parametric_var",
        "ewma_var",
        "ewma_es",
        "kupiec",
        "christoffersen",
    ):
        assert key in diag
    assert diag["research_only"] is True
    assert diag["execution_claim"] == "paper_backtest"
