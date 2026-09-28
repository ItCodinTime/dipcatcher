"""Unit coverage for hedge_lab helpers: tape fetch, char books, mirror edges."""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

from quant_fund.hedge_lab import tape
from quant_fund.hedge_lab.char_books import (
    binary_timing,
    lagged_characteristic_book,
    price_above_sma_book,
)
from quant_fund.hedge_lab.mirror import (
    dollar_neutral_weights,
    long_short_path,
    negate_target_weights,
    nested_worst_univariate_scores,
    pair_book_and_mirror,
    summarize_pair,
)
from quant_fund.hedge_lab.promotion import (
    clears_book_overlay,
    clears_cs_promotion,
    names_clearing_both,
    positive_mean_ic,
)


class TestFetchTape:
    def test_narrow_tape_uses_55_name_universe(self, tmp_path: Path, monkeypatch) -> None:
        captured: dict = {}

        def fake_download(dest, names, *, start, end, pause_s, sectors):
            captured["dest"] = dest
            captured["names"] = names
            captured["sectors"] = sectors
            return {"rows": len(names)}

        monkeypatch.setattr(tape, "download_yahoo_universe", fake_download)
        monkeypatch.setattr(tape, "assert_disk_budget", lambda **kw: {"ok": True})
        monkeypatch.setattr(tape, "lab_root", lambda: tmp_path)

        out = tape.fetch_hedge_lab_tape()
        assert captured["dest"] == tmp_path / "data" / "file_us" / "raw"
        assert captured["dest"].is_dir()
        assert len(captured["names"]) == len(tape.HEDGE_LAB_US)
        assert captured["sectors"] is None
        assert out["rows"] == len(tape.HEDGE_LAB_US)
        assert out["lab_universe"] == "HEDGE_LAB_US"
        assert out["n_requested"] == len(tape.HEDGE_LAB_US)
        assert out["survivorship"] == "current_constituents_applied_to_history"
        assert out["champion_alias"] is False
        assert out["sip_vintage"] is False

    def test_wide_tape_carries_sector_map(self, tmp_path: Path, monkeypatch) -> None:
        captured: dict = {}

        def fake_download(dest, names, *, start, end, pause_s, sectors):
            captured["names"] = names
            captured["sectors"] = sectors
            captured["dest"] = dest
            return {}

        monkeypatch.setattr(tape, "download_yahoo_universe", fake_download)
        monkeypatch.setattr(tape, "assert_disk_budget", lambda **kw: {"ok": True})
        monkeypatch.setattr(tape, "lab_root", lambda: tmp_path)

        out = tape.fetch_hedge_lab_tape(wide=True)
        names, sectors = tape.wide_universe()
        assert captured["names"] == names
        assert captured["sectors"] == sectors
        assert captured["sectors"]["SPY"] == "Benchmark"
        assert captured["dest"] == tmp_path / "data" / "file_us_wide" / "raw"
        assert out["lab_universe"] == "HEDGE_LAB_US_WIDE"
        assert out["n_requested"] == len(names)

    def test_explicit_root_wins(self, tmp_path: Path, monkeypatch) -> None:
        captured: dict = {}

        def fake_download(dest, names, **kw):
            captured["dest"] = dest
            return {}

        monkeypatch.setattr(tape, "download_yahoo_universe", fake_download)
        monkeypatch.setattr(tape, "assert_disk_budget", lambda **kw: {"ok": True})
        monkeypatch.setattr(tape, "lab_root", lambda: pytest.fail("lab_root must not be consulted"))

        dest = tmp_path / "custom"
        tape.fetch_hedge_lab_tape(root=dest)
        assert captured["dest"] == dest
        assert dest.is_dir()


def _char_frame() -> pl.DataFrame:
    # Three dates, six names. Scores monotone in name order; returns differ.
    dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
    rows_t, rows_s, rows_sco, rows_r = [], [], [], []
    for d in dates:
        for i in range(6):
            rows_t.append(d)
            rows_s.append(f"S{i}")
            rows_sco.append(float(i))
            rows_r.append(0.01 * (i - 2))
    return pl.DataFrame(
        {
            "event_time": rows_t,
            "security_id": rows_s,
            "score": rows_sco,
            "ret": rows_r,
        }
    )


class TestLaggedCharacteristicBook:
    def test_first_date_is_cash_and_lagged_scores(self) -> None:
        dates, pnl = lagged_characteristic_book(
            _char_frame(), "score", prefer_high=True, k_frac=1 / 3, one_way_cost=0.0
        )
        assert dates == ["2024-01-02", "2024-01-03", "2024-01-04"]
        assert pnl[0] == 0.0
        # Day 2 holds prior-date top k=2 names: S4, S5 -> ret mean (0.02+0.03)/2
        assert pnl[1] == pytest.approx(0.025)
        assert pnl[2] == pytest.approx(0.025)

    def test_prefer_low_picks_bottom_tail(self) -> None:
        _, pnl = lagged_characteristic_book(
            _char_frame(), "score", prefer_high=False, k_frac=1 / 3, one_way_cost=0.0
        )
        # Bottom names S0, S1 -> (-0.02 + -0.01) / 2
        assert pnl[1] == pytest.approx(-0.015)

    def test_turnover_cost_hits_entry(self) -> None:
        _, pnl = lagged_characteristic_book(
            _char_frame(), "score", prefer_high=True, k_frac=1 / 3, one_way_cost=0.01
        )
        # Day 2: gross 0.025 minus 0.01 * turn. From cash, turn = sum|w| = 1.0.
        assert pnl[1] == pytest.approx(0.025 - 0.01)

    def test_thin_prior_date_skips_and_resets(self) -> None:
        # Four dates; the third has only 4 names (<5), so day four cannot
        # trade on it and the book reports flat — positions do not leak.
        dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
        rows_t, rows_s, rows_sco, rows_r = [], [], [], []
        for d in dates:
            # The thin date keeps the names day-2's book selects (S4,S5) so
            # day-3 pnl is nonzero; its 4-name group still blocks day 4.
            members = (0, 1, 4, 5) if d == "2024-01-04" else range(6)
            for i in members:
                rows_t.append(d)
                rows_s.append(f"S{i}")
                rows_sco.append(float(i))
                rows_r.append(0.01 * (i - 2))
        frame = pl.DataFrame(
            {
                "event_time": rows_t,
                "security_id": rows_s,
                "score": rows_sco,
                "ret": rows_r,
            }
        )
        _, pnl = lagged_characteristic_book(
            frame, "score", prefer_high=True, k_frac=1 / 3, one_way_cost=0.0
        )
        assert pnl[0] == 0.0
        assert pnl[1] == pytest.approx(0.025)  # day-1 top-2 S4,S5 held on day 2
        assert pnl[2] == pytest.approx(0.025)  # day-2 top-2 held on day 3
        assert pnl[3] == 0.0  # prior date too thin -> flat, prev reset

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="k_frac"):
            lagged_characteristic_book(_char_frame(), "score", prefer_high=True, k_frac=0.0)
        with pytest.raises(ValueError, match="k_frac"):
            lagged_characteristic_book(_char_frame(), "score", prefer_high=True, k_frac=1.5)
        with pytest.raises(ValueError, match="missing"):
            lagged_characteristic_book(_char_frame().drop("ret"), "score", prefer_high=True)
        with pytest.raises(ValueError, match="missing"):
            lagged_characteristic_book(_char_frame().drop("score"), "score", prefer_high=True)

    def test_string_dates_sort_and_group(self) -> None:
        # String event_time exercises _date_key's non-isoformat branch.
        dates, pnl = lagged_characteristic_book(
            _char_frame(), "score", prefer_high=True, k_frac=0.5, one_way_cost=0.0
        )
        assert dates[0] < dates[-1]
        assert pnl.shape == (3,)


class TestPriceAboveSma:
    def test_uptrend_holds_and_flat_names_stay_cash(self) -> None:
        t_len, n = 12, 3
        px = np.ones((t_len, n))
        px[:, 0] = np.linspace(1.0, 1.2, t_len)  # rising -> above SMA
        px[:, 1] = np.linspace(1.2, 1.0, t_len)  # falling -> below SMA
        pnl = price_above_sma_book(px, window=4, one_way_cost=0.0)
        assert pnl.shape == (t_len,)
        assert np.all(pnl[:4] == 0.0)
        # After warm-up the book is long S0 only, so pnl equals S0's return.
        ret0 = px[5, 0] / px[4, 0] - 1.0
        assert pnl[5] == pytest.approx(ret0)

    def test_entry_cost_and_flat_second_day(self) -> None:
        t_len, n = 8, 2
        px = np.ones((t_len, n))
        px[:, 0] = np.linspace(1.0, 1.1, t_len)
        px[:, 1] = np.linspace(1.0, 0.9, t_len)
        pnl = price_above_sma_book(px, window=3, one_way_cost=0.01)
        # First invested day: turn=1.0 (from cash); cost deducted once.
        assert pnl[3] < px[3, 0] / px[2, 0] - 1.0

    def test_ndim_guard(self) -> None:
        with pytest.raises(ValueError, match="T, N"):
            price_above_sma_book(np.ones(5))

    def test_nan_prices_are_flat_not_poisonous(self) -> None:
        px = np.ones((10, 2))
        px[:, 0] = np.linspace(1.0, 1.1, 10)
        px[5, 0] = np.nan  # a missing close mid-path
        pnl = price_above_sma_book(px, window=3)
        assert np.isfinite(pnl).all()


class TestBinaryTiming:
    def test_holds_asset_on_prior_true(self) -> None:
        signal = np.array([True, True, False, False])
        asset = np.array([0.01, 0.02, 0.03, 0.04])
        pnl = binary_timing(signal, asset, one_way_cost=0.0)
        assert pnl[0] == 0.0
        assert pnl[1] == pytest.approx(0.02)  # sig[0]=True
        assert pnl[2] == pytest.approx(0.03)  # sig[1]=True
        assert pnl[3] == pytest.approx(0.0)  # sig[2]=False -> flat

    def test_toggle_cost_charged_once_per_flip(self) -> None:
        signal = np.array([False, True, False])
        asset = np.full(3, 0.01)
        pnl = binary_timing(signal, asset, one_way_cost=0.005)
        # t=1: sig[0]=False -> flat, but turn toggles prev False->False? no cost
        # t=2: sig[1]=True -> on; prev was False at t=1 -> cost charged
        assert pnl[2] == pytest.approx(0.01 - 0.005)

    def test_alignment_guard(self) -> None:
        with pytest.raises(ValueError, match="align"):
            binary_timing(np.array([True]), np.ones(2))


class TestMirrorEdges:
    def test_negate_requires_target_weight(self) -> None:
        with pytest.raises(ValueError, match="target_weight"):
            negate_target_weights(pl.DataFrame({"w": [1.0]}))

    def test_dollar_neutral_guards(self) -> None:
        with pytest.raises(ValueError, match="k_frac"):
            dollar_neutral_weights(np.ones(8), k_frac=0.0)
        with pytest.raises(ValueError, match="k_frac"):
            dollar_neutral_weights(np.ones(8), k_frac=0.6)
        with pytest.raises(ValueError, match="k_frac"):
            dollar_neutral_weights(np.ones(8), k_frac=math.nan)
        # Fewer than 4 finite scores -> flat.
        w = dollar_neutral_weights(np.array([1.0, 2.0, np.nan, np.nan]))
        assert_allclose(w, np.zeros(4))

    def test_long_short_alignment_errors(self) -> None:
        scores = np.ones(8)
        y = np.ones(8)
        dates = np.array(["2024-01-02"] * 8)
        with pytest.raises(ValueError, match="non-negative"):
            long_short_path(scores, y, dates, one_way_cost=-0.1)
        with pytest.raises(ValueError, match="align"):
            long_short_path(np.ones(4), y, dates)
        with pytest.raises(ValueError, match="align"):
            long_short_path(scores, y, dates[:4])
        with pytest.raises(ValueError, match="align"):
            long_short_path(scores, y, dates, np.array(["x"] * 4))

    def test_long_short_id_aligned_turnover(self) -> None:
        # Two dates, 6 names each; scores flip membership across days.
        dates = np.array(["2024-01-02"] * 6 + ["2024-01-03"] * 6)
        ids = np.array([f"S{i}" for i in range(6)] * 2)
        scores = np.concatenate(
            [np.arange(6.0), np.arange(6.0)[::-1]]  # day 2 inverts the ranking
        )
        y = np.zeros(12)
        out = long_short_path(scores, y, dates, ids, k_frac=1 / 3, one_way_cost=0.01)
        assert out["turnover"][0] == pytest.approx(1.0)
        # Day 2: longs/shorts swap fully -> |Δw| sums to 2.0.
        assert out["turnover"][1] == pytest.approx(2.0)
        assert out["returns"][1] == pytest.approx(-0.02)

    def test_long_short_positional_turnover_without_ids(self) -> None:
        dates = np.array(["2024-01-02"] * 6 + ["2024-01-03"] * 6)
        scores = np.concatenate([np.arange(6.0), np.arange(6.0)[::-1]])
        out = long_short_path(scores, np.zeros(12), dates, None, k_frac=1 / 3, one_way_cost=0.0)
        assert out["economic"]["n_dates"] == 2

    def test_thin_group_is_skipped(self) -> None:
        # First date has only 3 names (<4 finite) -> skipped entirely.
        dates = np.array(["2024-01-02"] * 3 + ["2024-01-03"] * 6)
        scores = np.arange(9.0)
        out = long_short_path(scores, np.zeros(9), dates, None, k_frac=0.4)
        assert out["economic"]["n_dates"] == 1
        assert out["dates"] == ["2024-01-03"]

    def test_isoformat_and_text_dates(self) -> None:
        day = datetime(2024, 1, 2)
        dates = np.array([day] * 6 + ["2024-01-03"] * 6, dtype=object)
        scores = np.arange(12.0)
        out = long_short_path(scores, np.zeros(12), dates)
        assert out["dates"] == ["2024-01-02", "2024-01-03"]

    def test_summarize_pair_fields(self) -> None:
        rng = np.random.default_rng(0)
        n_dates, n_names = 30, 8
        dates = np.repeat([f"2024-01-{d:02d}" for d in range(1, n_dates + 1)], n_names)
        scores = rng.normal(size=dates.size)
        y = rng.normal(size=dates.size)
        pair = pair_book_and_mirror(scores, y, dates)
        summary = summarize_pair("demo", pair)
        assert summary["name"] == "demo"
        assert summary["sharpe_sum"] == pytest.approx(
            summary["book_sharpe"] + summary["mirror_sharpe"]
        )
        assert summary["live_pnl_claim"] is False
        assert summary["blend_weight"] == 0.0
        assert "book_total_return" in summary

    def test_nested_worst_fallback_when_no_folds(self) -> None:
        from types import SimpleNamespace

        n = 30
        rng = np.random.default_rng(1)
        x = rng.normal(size=(n, 3))
        # Column 1 is anti-correlated with y -> lowest train date-IC.
        y = -x[:, 1] * 2.0 + rng.normal(scale=0.1, size=n)
        # Five names per date — the date-IC estimator's minimum group size.
        dates = np.repeat([f"2024-01-{d:02d}" for d in range(1, 7)], 5)
        cfg = SimpleNamespace(
            validation=SimpleNamespace(
                scheme="expanding", train_bars=10_000, val_bars=10, test_bars=10
            ),
            embargo_bars=lambda: 1,
        )
        pred = nested_worst_univariate_scores(x, y, dates, cfg, horizon_bars=1)
        assert_allclose(pred, x[:, 1])

    def test_nested_worst_walk_forward_path(self) -> None:
        from types import SimpleNamespace

        n_dates, n_names = 40, 5
        base = date(2024, 2, 1)
        dates = np.repeat([(base + timedelta(days=i)).isoformat() for i in range(n_dates)], n_names)
        rng = np.random.default_rng(2)
        x = rng.normal(size=(dates.size, 3))
        y = -x[:, 0] + rng.normal(scale=0.2, size=dates.size)
        cfg = SimpleNamespace(
            validation=SimpleNamespace(scheme="expanding", train_bars=15, val_bars=3, test_bars=5),
            embargo_bars=lambda: 1,
        )
        pred = nested_worst_univariate_scores(x, y, dates, cfg, horizon_bars=1)
        # Some rows get a walk-forward prediction, early rows stay NaN.
        assert np.isfinite(pred).sum() > 0
        assert np.isnan(pred).sum() > 0


class TestPromotionGates:
    def test_positive_mean_ic_types(self) -> None:
        assert positive_mean_ic(0.01) is True
        assert positive_mean_ic(0.0) is False
        assert positive_mean_ic(-0.5) is False
        assert positive_mean_ic(math.nan) is False
        assert positive_mean_ic("oops") is False

    def test_cs_promotion_all_gates_required(self) -> None:
        base = dict(
            name="champ",
            mean_ic=0.05,
            dm_preferred="champ",
            dm_p=0.01,
            stepm_rejected=["champ"],
            rc_p=0.01,
            spa_p=0.01,
        )
        assert clears_cs_promotion(**base) is True
        assert clears_cs_promotion(**{**base, "mean_ic": -0.01}) is False
        assert clears_cs_promotion(**{**base, "dm_preferred": "ridge"}) is False
        assert clears_cs_promotion(**{**base, "stepm_rejected": []}) is False
        assert clears_cs_promotion(**{**base, "rc_p": 0.5}) is False
        assert clears_cs_promotion(**{**base, "spa_p": 0.5}) is False
        assert clears_cs_promotion(**{**base, "dm_p": math.nan}) is False
        assert clears_cs_promotion(**{**base, "dm_p": "bad"}) is False

    def test_names_clearing_both_intersects(self) -> None:
        assert names_clearing_both(["a", "b"], ["b", "c"]) == ["b"]
        assert names_clearing_both([], ["b"]) == []

    def test_book_overlay_gate(self) -> None:
        assert clears_book_overlay(excess_means=[0.01, 0.02], rc_p=0.01, spa_p=0.01) is True
        assert clears_book_overlay(excess_means=[0.01, -0.001], rc_p=0.01, spa_p=0.01) is False
        assert clears_book_overlay(excess_means=[], rc_p=0.01, spa_p=0.01) is False
        assert clears_book_overlay(excess_means=[0.01], rc_p=math.inf, spa_p=0.01) is False
        assert clears_book_overlay(excess_means=[0.01], rc_p="bad", spa_p=0.01) is False
