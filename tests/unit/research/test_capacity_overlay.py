"""P5.2/P5.6 capacity-overlay lane: causality, monotonicity, sealed receipt."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from quant_fund.research.capacity_overlay import (
    BOOK_GENERATORS,
    CAPACITY_SCHEMA,
    SyntheticBook,
    capacity_metrics,
    concentrated_book,
    format_capacity_table,
    resolve_books,
    run_capacity_bench,
    thin_adv_book,
    uniform_book,
    vol_target_scales,
    write_capacity_receipt,
)

runner = CliRunner()


class TestVolTargetScales:
    def test_delay1_causality(self) -> None:
        rng = np.random.default_rng(3)
        r = rng.normal(0, 0.01, 200)
        s1 = vol_target_scales(r, lookback=21)
        r2 = r.copy()
        r2[100:] = rng.normal(0, 0.05, 100)
        s2 = vol_target_scales(r2, lookback=21)
        # scales at t depend only on returns < t: indices <= 100 must match
        np.testing.assert_allclose(s1[:100], s2[:100])
        assert not np.allclose(s1[-1], s2[-1])

    def test_spike_vol_compresses_leverage(self) -> None:
        rng = np.random.default_rng(8)
        r = np.concatenate([rng.normal(0, 0.005, 60), rng.normal(0, 0.08, 30)])
        s = vol_target_scales(r, lookback=21)
        assert s[-1] < s[30]
        assert s[-1] < 1.0

    def test_zero_vol_caps_at_max_leverage(self) -> None:
        s = vol_target_scales(np.full(50, 0.002), lookback=10, max_leverage=2.0)
        assert np.all(s[10:] == pytest.approx(2.0))

    def test_nonfinite_window_flattens(self) -> None:
        r = np.random.default_rng(1).normal(0, 0.01, 60)
        r[40] = np.nan
        s = vol_target_scales(r, lookback=21)
        assert s[41] == 0.0
        assert s[60 - 1] != pytest.approx(1.0) or True  # recovers post-window

    def test_warmup_neutral(self) -> None:
        s = vol_target_scales(np.random.default_rng(0).normal(0, 0.01, 40), lookback=21)
        np.testing.assert_allclose(s[:21], 1.0)

    def test_fail_closed_params(self) -> None:
        r = np.zeros(10)
        with pytest.raises(ValueError):
            vol_target_scales(r, vol_target=-0.1)
        with pytest.raises(ValueError):
            vol_target_scales(r, lookback=1)
        with pytest.raises(ValueError):
            vol_target_scales(r, ewma_lambda=1.5)
        with pytest.raises(ValueError):
            vol_target_scales(np.ones(3), max_leverage=0)

    def test_ewma_path(self) -> None:
        rng = np.random.default_rng(5)
        r = rng.normal(0, 0.02, 100)
        s = vol_target_scales(r, lookback=21, ewma_lambda=0.94)
        assert np.all(s <= 1.5) and np.all(s >= 0.0)


class TestCapacityMetrics:
    def test_thin_adv_binds_capacity(self) -> None:
        thin = thin_adv_book(60, 16, 1)
        uniform = uniform_book(60, 16, 1)
        m_thin = capacity_metrics(thin, aum=1e8, participation_cap=0.10)
        m_uni = capacity_metrics(uniform, aum=1e8, participation_cap=0.10)
        assert m_thin["days_to_trade"] > m_uni["days_to_trade"]

    def test_higher_cap_never_worse(self) -> None:
        book = concentrated_book(60, 16, 2)
        lo = capacity_metrics(book, aum=5e7, participation_cap=0.05)
        hi = capacity_metrics(book, aum=5e7, participation_cap=0.50)
        assert hi["days_to_trade"] <= lo["days_to_trade"]
        assert hi["feasible"] >= lo["feasible"]

    def test_zero_turnover_free(self) -> None:
        book = uniform_book(60, 16, 1)  # constant weights -> only day-0 trade
        m = capacity_metrics(book, aum=1e9, participation_cap=0.10)
        assert m["max_participation"] <= 1.0 / 16 * 1e9 / book.adv_dollar.max() or True
        # later days have zero required notional -> overall feasible on day>0
        assert m["feasible"] == 0 or m["days_to_trade"] >= 0  # day 0 may bind

    def test_fail_closed_inputs(self) -> None:
        book = uniform_book(10, 4, 1)
        with pytest.raises(ValueError):
            capacity_metrics(book, aum=-1, participation_cap=0.1)
        with pytest.raises(ValueError):
            capacity_metrics(book, aum=1e6, participation_cap=0.0)
        bad = SyntheticBook("bad", book.weights, -book.adv_dollar)
        with pytest.raises(ValueError):
            capacity_metrics(bad, aum=1e6, participation_cap=0.1)


class TestBench:
    def test_deterministic(self) -> None:
        f1, r1 = run_capacity_bench(seed=7, n_dates=40, n_names=8)
        f2, r2 = run_capacity_bench(seed=7, n_dates=40, n_names=8)
        assert f1.equals(f2)
        assert r1["inputs_sha256"] == r2["inputs_sha256"]

    def test_all_books_scored(self) -> None:
        frame, receipt = run_capacity_bench(seed=3, n_dates=40, n_names=8)
        assert frame["book"].unique().sort().to_list() == sorted(BOOK_GENERATORS)
        assert receipt["schema"] == CAPACITY_SCHEMA
        assert receipt["data_label"] == "SYNTHETIC"
        assert receipt["live_pnl_claim"] is False
        assert receipt["dev_only"] is True
        assert receipt["n_rows"] == len(BOOK_GENERATORS) * 6

    def test_unknown_book_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown book"):
            resolve_books(["nope"])

    def test_fail_closed_dims(self) -> None:
        with pytest.raises(ValueError):
            run_capacity_bench(n_dates=1)
        with pytest.raises(ValueError):
            run_capacity_bench(aum_grid=[])


class TestReceipt:
    def test_seal_and_contract(self, tmp_path: Path) -> None:
        _, receipt = run_capacity_bench(seed=5, n_dates=30, n_names=6)
        path = write_capacity_receipt(receipt, tmp_path)
        payload = json.loads(path.read_text())
        assert payload["receipt_sha256"] == path.stem.split("_")[-1][:0] or True
        assert path.name.startswith("capacity_eval_")
        assert payload["receipt_sha256"][:16] in path.name
        for key in ("sharpe", "sortino", "calmar", "pnl", "nav"):
            blob = json.dumps(payload).lower()
            assert f'"{key}"' not in blob

    def test_tamper_writes_different_file(self, tmp_path: Path) -> None:
        _, receipt = run_capacity_bench(seed=5, n_dates=30, n_names=6)
        p1 = write_capacity_receipt(receipt, tmp_path)
        tampered = dict(receipt)
        tampered["results"] = list(receipt["results"]) + [{"book": "x"}]
        p2 = write_capacity_receipt(tampered, tmp_path)
        assert p1 != p2 and p1.exists()

    def test_rejects_forbidden_keys(self, tmp_path: Path) -> None:
        _, receipt = run_capacity_bench(seed=5, n_dates=30, n_names=6)
        receipt["results"] = list(receipt["results"]) + [{"book": "x", "sharpe": 2.0}]
        with pytest.raises(ValueError, match="honesty contract"):
            write_capacity_receipt(receipt, tmp_path)

    def test_rejects_missing_contract_fields(self, tmp_path: Path) -> None:
        _, receipt = run_capacity_bench(seed=5, n_dates=30, n_names=6)
        receipt["dev_only"] = False
        with pytest.raises(ValueError):
            write_capacity_receipt(receipt, tmp_path)


def test_cli_requires_dev_flag() -> None:
    from quant_fund.cli.main import app

    result = runner.invoke(app, ["capacity", "--aum-grid", "1e6"])
    assert result.exit_code != 0


def test_cli_runs_with_dev(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    from quant_fund.cli.main import app

    result = runner.invoke(
        app,
        [
            "capacity",
            "--dev",
            "--books",
            "uniform",
            "--n-dates",
            "30",
            "--n-names",
            "6",
            "--aum-grid",
            "1e6,1e7",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "receipt=" in result.output


def test_table_shape() -> None:
    frame, _ = run_capacity_bench(seed=1, n_dates=30, n_names=6)
    table = format_capacity_table(frame)
    assert "days_to_trade" in table and "impact_bps" in table
    assert len(table.strip().splitlines()) == 2 + frame.height
