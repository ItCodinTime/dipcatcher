"""SYNTHETIC unit tests for Kyle λ / OFI→Δmid Northset diagnostics.

research_only — no live tape, no live P&L claim. Uses synthetic bars+L2 only.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.northset.kyle_ofi import (
    bench_kyle_ofi_fused,
    cont_ofi_series,
    fuse_bars_l2_kyle_frame,
    kyle_lambda_by_date,
    kyle_lambda_ols,
    ofi_delta_mid_date_ic,
)


def _synthetic_bars_and_book(
    n_assets: int = 6,
    n_days: int = 24,
    seed: int = 11,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build labeled SYNTHETIC bars + aligned L2 top-of-book panel."""
    rng = np.random.default_rng(seed)
    t0 = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    bar_rows: list[dict[str, object]] = []
    book_rows: list[dict[str, object]] = []
    for a in range(n_assets):
        sid = f"S{a}"
        px = 100.0 + a
        for d in range(n_days):
            ts = t0 + timedelta(days=d)
            ret = float(rng.normal(0.0, 0.01))
            opn = px
            close = px * (1.0 + ret)
            high = max(opn, close) * (1.0 + abs(float(rng.normal(0.0, 0.002))))
            low = min(opn, close) * (1.0 - abs(float(rng.normal(0.0, 0.002))))
            vol = float(rng.uniform(1e5, 5e5))
            bar_rows.append(
                {
                    "security_id": sid,
                    "symbol": sid,
                    "event_time": ts,
                    "available_time": ts,
                    "open": opn,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": vol,
                    "source": "synthetic",
                    "revision_id": "SYNTHETIC",
                }
            )
            # SYNTHETIC L2: tilt sizes with bar direction; mid near close
            direction = np.sign(close - opn)
            half = max(close * 2e-4, 0.01)
            best_bid = close - half
            best_ask = close + half
            mid = 0.5 * (best_bid + best_ask)
            top_total = max(vol * 0.02, 1.0)
            imb = float(np.clip(0.5 * direction + rng.normal(0.0, 0.1), -0.8, 0.8))
            top_bid = top_total * (0.5 + 0.5 * imb)
            top_ask = top_total * (0.5 - 0.5 * imb)
            book_rows.append(
                {
                    "security_id": sid,
                    "event_time": ts,
                    "available_time": ts,
                    "source": "synthetic",
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "mid": mid,
                    "top_bid_size": top_bid,
                    "top_ask_size": top_ask,
                    "bid_depth": top_bid * 2.5,
                    "ask_depth": top_ask * 2.5,
                }
            )
            px = close
    return pl.DataFrame(bar_rows), pl.DataFrame(book_rows)


def test_kyle_lambda_ols_recovers_known_slope() -> None:
    """Closed-form: Δmid = 0.02 * flow + noise → λ ≈ 0.02."""
    rng = np.random.default_rng(0)
    flow = rng.normal(0.0, 1.0, size=200)
    delta = 0.02 * flow + rng.normal(0.0, 0.001, size=200)
    lam = kyle_lambda_ols(delta, flow)
    assert lam == pytest.approx(0.02, rel=0.05, abs=0.002)


def test_cont_ofi_first_zero_and_finite() -> None:
    bp = np.array([10.0, 10.0, 10.1, 10.0])
    bs = np.array([5.0, 6.0, 4.0, 5.0])
    ap = np.array([10.1, 10.1, 10.2, 10.1])
    az = np.array([5.0, 4.0, 5.0, 6.0])
    ofi = cont_ofi_series(bp, bs, ap, az)
    assert ofi[0] == 0.0
    assert np.all(np.isfinite(ofi))


def test_fuse_adds_signed_depth_ofi_delta_mid() -> None:
    bars, book = _synthetic_bars_and_book()
    fused = fuse_bars_l2_kyle_frame(bars, book)
    for col in ("signed_depth", "ofi", "delta_mid", "mid"):
        assert col in fused.columns
    assert fused.height == bars.height
    # last day per name has null delta_mid
    nulls = fused.filter(pl.col("delta_mid").is_null()).height
    assert nulls == bars["security_id"].n_unique()


def test_kyle_lambda_by_date_uses_hac_keys() -> None:
    bars, book = _synthetic_bars_and_book(n_assets=8, n_days=30, seed=3)
    fused = fuse_bars_l2_kyle_frame(bars, book)
    receipt = kyle_lambda_by_date(fused, flow="signed_depth", min_names=3)
    assert receipt["research_only"] is True
    assert receipt["label"] == "SYNTHETIC"
    assert receipt["ic_method"] == "date_level_spearman_hac"
    assert receipt["n_dates_lambda"] >= 3
    assert "kyle_lambda_t" in receipt
    assert "kyle_lambda_p" in receipt
    assert "flow_delta_mid_n_dates" in receipt
    assert float(receipt["flow_delta_mid_n_dates"]) >= 3.0
    assert "sharpe" not in {k.lower() for k in receipt}
    assert "live_pnl_claim" not in receipt


def test_ofi_delta_mid_date_ic_not_pooled() -> None:
    bars, book = _synthetic_bars_and_book(n_assets=8, n_days=28, seed=7)
    fused = fuse_bars_l2_kyle_frame(bars, book)
    receipt = ofi_delta_mid_date_ic(fused, min_names=3, lag=0)
    assert receipt["ic_method"] == "date_level_spearman_hac"
    assert receipt["diagnostic"] == "ofi_to_delta_mid"
    assert receipt["research_only"] is True
    assert float(receipt["n_dates"]) >= 3.0
    assert "t_spearman" in receipt and "p_spearman" in receipt


def test_bench_kyle_ofi_fused_honesty() -> None:
    bars, book = _synthetic_bars_and_book(n_assets=8, n_days=32, seed=11)
    receipt = bench_kyle_ofi_fused(bars, book, min_names=3, label="SYNTHETIC")
    assert receipt["family"] == "kyle_ofi"
    assert receipt["research_only"] is True
    assert receipt["claim"] == "research_diagnostic_only"
    assert receipt["ic_method"] == "date_level_spearman_hac"
    assert receipt["dgp"] == "synthetic_lob"
    for key in (
        "kyle_lambda_depth_mean",
        "kyle_lambda_depth_t",
        "kyle_lambda_ofi_mean",
        "ofi_delta_mid_lag0_mean_spearman",
        "ofi_delta_mid_lag0_t",
        "ofi_delta_mid_lag0_n_dates",
        "ofi_delta_mid_lag1_mean_spearman",
    ):
        assert key in receipt
    assert float(receipt["ofi_delta_mid_lag0_n_dates"]) >= 3.0
    forbidden = ("sharpe", "sortino", "calmar", "pnl", "nav", "live_pnl_claim")
    assert not any(any(tok in k.lower() for tok in forbidden) for k in receipt)


def test_kyle_lambda_ofi_depth_corr_rejects_misaligned_series(monkeypatch) -> None:
    import quant_fund.northset.kyle_ofi as kyle_ofi

    calls = iter(
        [
            {"kyle_lambda_dates": ["2020-01-01", "2020-01-02"], "kyle_lambda_series": [1.0]},
            {"kyle_lambda_dates": ["2020-01-01"], "kyle_lambda_series": [1.0]},
        ]
    )
    monkeypatch.setattr(kyle_ofi, "kyle_lambda_by_date", lambda *args, **kwargs: next(calls))

    with pytest.raises(ValueError, match="kyle_lambda dates/series length mismatch"):
        kyle_ofi.kyle_lambda_ofi_depth_corr(pl.DataFrame())


def test_bench_rejects_empty_bars() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        fuse_bars_l2_kyle_frame(pl.DataFrame({"security_id": [], "event_time": []}))


def test_fuse_requires_source_on_external_book() -> None:
    bars, book = _synthetic_bars_and_book(n_assets=4, n_days=10, seed=2)
    bare = book.drop("source")
    with pytest.raises(ValueError, match="source"):
        fuse_bars_l2_kyle_frame(bars, bare)


def test_fuse_rejects_mixed_sources() -> None:
    bars, book = _synthetic_bars_and_book(n_assets=4, n_days=10, seed=2)
    mixed = book.with_columns(
        pl.when(pl.col("security_id") == "S0")
        .then(pl.lit("alpaca"))
        .otherwise(pl.lit("polygon"))
        .alias("source")
    )
    with pytest.raises(ValueError, match="mixes sources"):
        fuse_bars_l2_kyle_frame(bars, mixed)


def test_fuse_fail_closed_empty_join() -> None:
    bars, book = _synthetic_bars_and_book(n_assets=4, n_days=10, seed=2)
    shifted = book.with_columns(pl.col("event_time") + pl.duration(days=400))
    with pytest.raises(ValueError, match="empty frame"):
        fuse_bars_l2_kyle_frame(bars, shifted)


def test_bench_vendor_fixture_stamps_book_source() -> None:
    fx = Path(__file__).resolve().parents[1] / "fixtures" / "northset"
    bars = pl.read_parquet(fx / "bars_aligned.parquet")
    book = pl.read_parquet(fx / "alpaca_remapped_panel.parquet")
    # Drop benchmark-only names if any mismatch leaves sparse cross-section
    receipt = bench_kyle_ofi_fused(
        bars,
        book,
        min_names=2,
        label="SYNTHETIC",
        book_panel_path=str(fx / "alpaca_remapped_panel.parquet"),
    )
    assert receipt["book_source"] == "alpaca"
    assert receipt["book_dgp"] == "vendor_panel:alpaca"
    assert receipt["dgp"] == "vendor_panel:alpaca"
    assert receipt["research_only"] is True
    assert receipt["ic_method"] == "date_level_spearman_hac"
    assert float(receipt["join_coverage"]) >= 0.5
    assert "signed_depth_delta_mid_lag1_mean_spearman" in receipt
    assert receipt["book_panel_path"] is not None


def test_bench_polygon_fixture_remap_path() -> None:
    from quant_fund.microstructure.vendor_book_map import remap_vendor_quotes_to_panel

    fx = Path(__file__).resolve().parents[1] / "fixtures" / "northset"
    bars = pl.read_parquet(fx / "bars_aligned.parquet")
    raw = pl.read_parquet(fx / "polygon_quotes.parquet")
    book = remap_vendor_quotes_to_panel(raw, vendor="polygon")
    receipt = bench_kyle_ofi_fused(bars, book, min_names=2, label="SYNTHETIC")
    assert receipt["book_source"] == "polygon"
    assert receipt["book_dgp"] == "vendor_panel:polygon"
    assert receipt["claim"] == "research_diagnostic_only"


def test_signed_depth_lag1_date_ic() -> None:
    from quant_fund.northset.kyle_ofi import flow_delta_mid_date_ic

    bars, book = _synthetic_bars_and_book(n_assets=8, n_days=28, seed=9)
    fused = fuse_bars_l2_kyle_frame(bars, book)
    receipt = flow_delta_mid_date_ic(fused, flow="signed_depth", lag=1, min_names=3)
    assert receipt["diagnostic"] == "signed_depth_to_delta_mid"
    assert receipt["flow_lag"] == 1
    assert receipt["ic_method"] == "date_level_spearman_hac"
    assert float(receipt["n_dates"]) >= 3.0


def test_fuse_stamps_explicit_fwd_targets() -> None:
    bars, book = _synthetic_bars_and_book(n_assets=4, n_days=12, seed=5)
    fused = fuse_bars_l2_kyle_frame(bars, book)
    for col in ("fwd_delta_mid", "fwd_ret_1", "fwd_ret_2", "fwd_ret_3", "fwd_close_ret_1"):
        assert col in fused.columns
    # fwd_delta_mid matches delta_mid; last bar nulls
    both = fused.select("delta_mid", "fwd_delta_mid").drop_nulls()
    assert both.height > 0
    assert (both["delta_mid"] - both["fwd_delta_mid"]).abs().max() < 1e-12


def test_predictive_fwd_targets_date_ic() -> None:
    from quant_fund.northset.kyle_ofi import predictive_flow_fwd_date_ic

    bars, book = _synthetic_bars_and_book(n_assets=8, n_days=30, seed=13)
    fused = fuse_bars_l2_kyle_frame(bars, book)
    for flow, target in (
        ("ofi", "fwd_delta_mid"),
        ("ofi", "fwd_ret_1"),
        ("signed_depth", "fwd_delta_mid"),
        ("signed_depth", "fwd_ret_1"),
    ):
        receipt = predictive_flow_fwd_date_ic(fused, flow=flow, target=target, min_names=3)
        assert receipt["feature_lag"] == 0
        assert receipt["target"] == target
        assert receipt["ic_method"] == "date_level_spearman_hac"
        assert receipt["research_only"] is True
        assert float(receipt["n_dates"]) >= 3.0


def test_kyle_lambda_stability_and_fwd_on_bench() -> None:
    bars, book = _synthetic_bars_and_book(n_assets=8, n_days=32, seed=17)
    receipt = bench_kyle_ofi_fused(bars, book, min_names=3, label="SYNTHETIC")
    assert "kyle_lambda_depth_std" in receipt
    assert "kyle_lambda_depth_iqr" in receipt
    assert "kyle_lambda_ofi_std" in receipt
    assert "kyle_lambda_ofi_iqr" in receipt
    assert receipt["kyle_lambda_depth_std"] == receipt["kyle_lambda_depth_std"]  # finite or nan ok
    for key in (
        "ofi_fwd_delta_mid_mean_spearman",
        "ofi_fwd_ret_1_mean_spearman",
        "signed_depth_fwd_delta_mid_mean_spearman",
        "signed_depth_fwd_ret_1_mean_spearman",
        "ofi_fwd_delta_mid_t",
        "ofi_fwd_delta_mid_n_dates",
    ):
        assert key in receipt
    assert float(receipt["ofi_fwd_delta_mid_n_dates"]) >= 3.0
    forbidden = ("sharpe", "sortino", "calmar", "pnl", "nav", "live_pnl_claim")
    assert not any(any(tok in k.lower() for tok in forbidden) for k in receipt)


def test_kyle_lambda_fwd_ret_1_and_horizon2() -> None:
    bars, book = _synthetic_bars_and_book(n_assets=8, n_days=36, seed=19)
    fused = fuse_bars_l2_kyle_frame(bars, book)
    assert "fwd_ret_2" in fused.columns
    from quant_fund.northset.kyle_ofi import kyle_lambda_by_date, predictive_flow_fwd_date_ic

    lam = kyle_lambda_by_date(fused, flow="signed_depth", target="fwd_ret_1", min_names=3)
    assert lam["target"] == "fwd_ret_1"
    assert "kyle_lambda_std" in lam
    assert lam["research_only"] is True
    h2 = predictive_flow_fwd_date_ic(fused, flow="ofi", target="fwd_ret_2", min_names=3)
    assert h2["target"] == "fwd_ret_2"
    assert float(h2["n_dates"]) >= 3.0
    receipt = bench_kyle_ofi_fused(bars, book, min_names=3)
    assert "kyle_lambda_depth_fwd_ret_1_mean" in receipt
    assert "ofi_fwd_ret_2_mean_spearman" in receipt
    assert "signed_depth_fwd_ret_2_mean_spearman" in receipt
    forbidden = ("sharpe", "sortino", "calmar", "pnl", "nav", "live_pnl_claim")
    assert not any(any(tok in k.lower() for tok in forbidden) for k in receipt)


def test_kyle_lambda_fwd_delta_mid_and_ret3() -> None:
    from quant_fund.northset.kyle_ofi import kyle_lambda_by_date, predictive_flow_fwd_date_ic

    bars, book = _synthetic_bars_and_book(n_assets=8, n_days=40, seed=23)
    fused = fuse_bars_l2_kyle_frame(bars, book)
    assert "fwd_ret_3" in fused.columns
    lam = kyle_lambda_by_date(fused, flow="ofi", target="fwd_delta_mid", min_names=3)
    assert lam["target"] == "fwd_delta_mid"
    assert lam["claim"] == "research_diagnostic_only"
    h3 = predictive_flow_fwd_date_ic(fused, flow="signed_depth", target="fwd_ret_3", min_names=3)
    assert h3["target"] == "fwd_ret_3"
    assert float(h3["n_dates"]) >= 3.0
    receipt = bench_kyle_ofi_fused(bars, book, min_names=3)
    for key in (
        "kyle_lambda_depth_fwd_delta_mid_mean",
        "kyle_lambda_ofi_fwd_delta_mid_mean",
        "kyle_lambda_ofi_fwd_delta_mid_std",
        "ofi_fwd_ret_3_mean_spearman",
        "signed_depth_fwd_ret_3_mean_spearman",
        "ofi_fwd_ret_3_n_dates",
    ):
        assert key in receipt
    forbidden = ("sharpe", "sortino", "calmar", "pnl", "nav", "live_pnl_claim")
    assert not any(any(tok in k.lower() for tok in forbidden) for k in receipt)


def test_kyle_lambda_dispersion_deciles_and_hac_band() -> None:
    from quant_fund.northset.kyle_ofi import kyle_lambda_by_date, kyle_lambda_dispersion

    series = np.linspace(0.01, 0.05, 30)
    disp = kyle_lambda_dispersion(series, window=10)
    assert disp["kyle_lambda_p10"] < disp["kyle_lambda_p50"] < disp["kyle_lambda_p90"]
    assert disp["kyle_lambda_rolling_mean"] == disp["kyle_lambda_rolling_mean"]
    assert "kyle_lambda_rolling_hac_lo" in disp
    assert disp["claim"] == "research_diagnostic_only"

    bars, book = _synthetic_bars_and_book(n_assets=8, n_days=40, seed=29)
    fused = fuse_bars_l2_kyle_frame(bars, book)
    lam = kyle_lambda_by_date(fused, flow="signed_depth", min_names=3)
    assert "kyle_lambda_p50" in lam
    assert "kyle_lambda_rolling_hac_hi" in lam
    receipt = bench_kyle_ofi_fused(bars, book, min_names=3)
    for key in (
        "kyle_lambda_depth_p10",
        "kyle_lambda_depth_p50",
        "kyle_lambda_depth_p90",
        "kyle_lambda_ofi_p50",
        "kyle_lambda_depth_rolling_mean",
        "kyle_lambda_depth_rolling_hac_lo",
        "kyle_lambda_depth_rolling_hac_hi",
        "kyle_lambda_ofi_rolling_hac_t",
        "kyle_lambda_dispersion_window",
    ):
        assert key in receipt
    forbidden = ("sharpe", "sortino", "calmar", "pnl", "nav", "live_pnl_claim")
    assert not any(any(tok in k.lower() for tok in forbidden) for k in receipt)


def test_kyle_lambda_date_series_and_ofi_depth_corr() -> None:
    from quant_fund.northset.kyle_ofi import (
        kyle_lambda_date_series_frame,
        kyle_lambda_ofi_depth_corr,
    )

    bars, book = _synthetic_bars_and_book(n_assets=8, n_days=40, seed=31)
    fused = fuse_bars_l2_kyle_frame(bars, book)
    frame = kyle_lambda_date_series_frame(
        fused, flow="signed_depth", target="delta_mid", min_names=3
    )
    assert frame.height >= 3
    for col in ("event_time", "kyle_lambda", "flow", "target", "research_only", "claim"):
        assert col in frame.columns
    assert frame["research_only"].all()
    assert set(frame["claim"].unique().to_list()) == {"research_diagnostic_only"}

    corr = kyle_lambda_ofi_depth_corr(fused, target="delta_mid", min_names=3)
    assert corr["diagnostic"] == "kyle_lambda_ofi_vs_signed_depth_corr"
    assert corr["research_only"] is True
    assert float(corr["n_dates_aligned"]) >= 3.0
    assert "kyle_lambda_ofi_depth_spearman" in corr
    assert "kyle_lambda_ofi_depth_pearson" in corr

    receipt = bench_kyle_ofi_fused(bars, book, min_names=3)
    for key in (
        "kyle_lambda_ofi_depth_spearman",
        "kyle_lambda_ofi_depth_pearson",
        "kyle_lambda_ofi_depth_n_dates",
        "kyle_lambda_date_series_n_depth",
        "kyle_lambda_date_series_n_ofi",
    ):
        assert key in receipt
    forbidden = ("sharpe", "sortino", "calmar", "pnl", "nav", "live_pnl_claim")
    assert not any(any(tok in k.lower() for tok in forbidden) for k in receipt)


def test_cli_dump_lambda_series_help_mentions_research_only() -> None:

    result = CliRunner().invoke(app, ["kyle-ofi", "--help"])
    assert result.exit_code == 0
    assert "--dump-lambda-series" in result.stdout
    assert (
        "research_only" in result.stdout.lower() or "research_diagnostic" in result.stdout.lower()
    )


def test_residual_depth_fwd_ret_1_soft_verify() -> None:
    """residual_depth→fwd_ret_1 date IC + bench/CLI keys; research_only parity."""
    from quant_fund.northset.kyle_ofi import residual_flow_date_ic

    bars, book = _synthetic_bars_and_book(n_assets=8, n_days=40, seed=43)
    fused = fuse_bars_l2_kyle_frame(bars, book)
    resid = residual_flow_date_ic(
        fused, flow="signed_depth", control="ofi", target="fwd_ret_1", min_names=3
    )
    assert resid["target"] == "fwd_ret_1"
    assert resid["flow"] == "signed_depth"
    assert resid["control"] == "ofi"
    assert resid["research_only"] is True
    assert resid["claim"] == "research_diagnostic_only"
    assert resid["ic_method"] == "date_level_spearman_hac"
    assert float(resid["n_dates"]) >= 3.0
    forbidden = ("sharpe", "sortino", "calmar", "pnl", "nav", "live_pnl_claim")
    assert not any(any(tok in k.lower() for tok in forbidden) for k in resid)

    receipt = bench_kyle_ofi_fused(bars, book, min_names=3)
    for key in (
        "residual_depth_ex_ofi_fwd_ret_1_mean_spearman",
        "residual_depth_ex_ofi_fwd_ret_1_t",
        "residual_depth_ex_ofi_fwd_ret_1_n_dates",
        "residual_ofi_ex_depth_fwd_ret_1_mean_spearman",
        "residual_ofi_ex_depth_fwd_ret_1_t",
    ):
        assert key in receipt
    assert receipt["research_only"] is True
    assert receipt["claim"] == "research_diagnostic_only"
    assert not any(any(tok in k.lower() for tok in forbidden) for k in receipt)


def test_residual_flow_and_dump_lambda_series(tmp_path: Path) -> None:
    from quant_fund.northset.kyle_ofi import (
        kyle_lambda_date_series_frame,
        residual_flow_date_ic,
    )

    bars, book = _synthetic_bars_and_book(n_assets=8, n_days=36, seed=41)
    fused = fuse_bars_l2_kyle_frame(bars, book)
    resid = residual_flow_date_ic(
        fused, flow="ofi", control="signed_depth", target="fwd_delta_mid", min_names=3
    )
    assert resid["claim"] == "research_diagnostic_only"
    assert resid["research_only"] is True
    assert float(resid["n_dates"]) >= 3.0
    assert "mean_spearman" in resid

    frame = kyle_lambda_date_series_frame(fused, flow="ofi", min_names=3)
    assert frame.height >= 3
    assert frame["research_only"].all()
    out = tmp_path / "kyle_lambda_series.parquet"
    pl.concat(
        [
            kyle_lambda_date_series_frame(fused, flow="signed_depth", min_names=3),
            frame,
        ]
    ).write_parquet(out)
    loaded = pl.read_parquet(out)
    assert loaded.height >= 6
    assert "kyle_lambda" in loaded.columns
    assert "sharpe" not in loaded.columns

    receipt = bench_kyle_ofi_fused(bars, book, min_names=3)
    assert "residual_ofi_ex_depth_fwd_delta_mid_mean_spearman" in receipt
    assert "residual_depth_ex_ofi_fwd_delta_mid_t" in receipt
    forbidden = ("sharpe", "sortino", "calmar", "pnl", "nav", "live_pnl_claim")
    assert not any(any(tok in k.lower() for tok in forbidden) for k in receipt)


def test_residual_flow_soft_verify_honesty() -> None:
    """Soft-verify residual IC keys: research_only, claim, no Sharpe/pnl leak."""
    from quant_fund.northset.kyle_ofi import residual_flow_date_ic

    bars, book = _synthetic_bars_and_book(n_assets=8, n_days=36, seed=41)
    fused = fuse_bars_l2_kyle_frame(bars, book)
    resid = residual_flow_date_ic(
        fused, flow="ofi", control="signed_depth", target="fwd_delta_mid", min_names=3
    )
    assert resid["research_only"] is True
    assert resid["claim"] == "research_diagnostic_only"
    assert resid["ic_method"] == "date_level_spearman_hac"
    assert "residual_" in str(resid["diagnostic"])
    for key in (
        "mean_spearman",
        "t_spearman",
        "p_spearman",
        "n_dates",
        "flow",
        "control",
        "target",
    ):
        assert key in resid
    forbidden = ("sharpe", "sortino", "calmar", "pnl", "nav", "live_pnl_claim")
    assert not any(any(tok in k.lower() for tok in forbidden) for k in resid)
    # string values must not smuggle forbidden tokens either
    for v in resid.values():
        if isinstance(v, str):
            low = v.lower()
            assert "sharpe" not in low and "live_pnl" not in low

    receipt = bench_kyle_ofi_fused(bars, book, min_names=3)
    assert receipt["research_only"] is True
    assert receipt["claim"] == "research_diagnostic_only"
    assert "residual_ofi_ex_depth_fwd_delta_mid_mean_spearman" in receipt
    assert "residual_ofi_ex_depth_fwd_delta_mid_t" in receipt
    assert "residual_ofi_ex_depth_fwd_delta_mid_n_dates" in receipt
    assert not any(any(tok in k.lower() for tok in forbidden) for k in receipt)


def test_residual_flow_soft_verify_no_sharpe_leak() -> None:
    """Soft-verify residual_flow_date_ic + bench residual keys stay research_only."""
    from quant_fund.northset.kyle_ofi import residual_flow_date_ic

    bars, book = _synthetic_bars_and_book(n_assets=8, n_days=36, seed=41)
    fused = fuse_bars_l2_kyle_frame(bars, book)
    resid = residual_flow_date_ic(
        fused, flow="ofi", control="signed_depth", target="fwd_delta_mid", min_names=3
    )
    assert resid["research_only"] is True
    assert resid["claim"] == "research_diagnostic_only"
    assert resid["ic_method"] == "date_level_spearman_hac"
    assert "mean_spearman" in resid and "t_spearman" in resid and "n_dates" in resid
    assert float(resid["n_dates"]) >= 3.0
    forbidden = ("sharpe", "sortino", "calmar", "pnl", "nav", "live_pnl_claim")
    assert not any(any(tok in k.lower() for tok in forbidden) for k in resid)
    assert not any(
        isinstance(v, str) and any(tok in v.lower() for tok in ("sharpe", "live_pnl"))
        for v in resid.values()
    )

    receipt = bench_kyle_ofi_fused(bars, book, min_names=3)
    assert receipt["research_only"] is True
    assert receipt["claim"] == "research_diagnostic_only"
    for key in (
        "residual_ofi_ex_depth_fwd_delta_mid_mean_spearman",
        "residual_ofi_ex_depth_fwd_delta_mid_t",
        "residual_ofi_ex_depth_fwd_delta_mid_n_dates",
    ):
        assert key in receipt
    assert not any(any(tok in k.lower() for tok in forbidden) for k in receipt)


def test_residual_fwd_ret_2_soft_verify() -> None:
    from quant_fund.northset.kyle_ofi import residual_flow_date_ic

    bars, book = _synthetic_bars_and_book(n_assets=8, n_days=42, seed=47)
    fused = fuse_bars_l2_kyle_frame(bars, book)
    resid = residual_flow_date_ic(
        fused, flow="signed_depth", control="ofi", target="fwd_ret_2", min_names=3
    )
    assert resid["target"] == "fwd_ret_2"
    assert resid["research_only"] is True
    assert resid["claim"] == "research_diagnostic_only"
    forbidden = ("sharpe", "sortino", "calmar", "pnl", "nav", "live_pnl_claim")
    assert not any(any(tok in k.lower() for tok in forbidden) for k in resid)
    receipt = bench_kyle_ofi_fused(bars, book, min_names=3)
    for key in (
        "residual_depth_ex_ofi_fwd_ret_2_mean_spearman",
        "residual_depth_ex_ofi_fwd_ret_2_t",
        "residual_ofi_ex_depth_fwd_ret_2_mean_spearman",
        "residual_ofi_ex_depth_fwd_ret_2_n_dates",
    ):
        assert key in receipt
    assert receipt["research_only"] is True
    assert not any(any(tok in k.lower() for tok in forbidden) for k in receipt)


def test_residual_fwd_ret_3_soft_verify() -> None:
    from quant_fund.northset.kyle_ofi import residual_flow_date_ic

    bars, book = _synthetic_bars_and_book(n_assets=8, n_days=44, seed=53)
    fused = fuse_bars_l2_kyle_frame(bars, book)
    resid = residual_flow_date_ic(
        fused, flow="signed_depth", control="ofi", target="fwd_ret_3", min_names=3
    )
    assert resid["target"] == "fwd_ret_3"
    assert resid["research_only"] is True
    assert resid["claim"] == "research_diagnostic_only"
    forbidden = ("sharpe", "sortino", "calmar", "pnl", "nav", "live_pnl_claim")
    assert not any(any(tok in k.lower() for tok in forbidden) for k in resid)
    receipt = bench_kyle_ofi_fused(bars, book, min_names=3)
    for key in (
        "residual_depth_ex_ofi_fwd_ret_3_mean_spearman",
        "residual_depth_ex_ofi_fwd_ret_3_t",
        "residual_ofi_ex_depth_fwd_ret_3_mean_spearman",
        "residual_ofi_ex_depth_fwd_ret_3_n_dates",
    ):
        assert key in receipt
    assert not any(any(tok in k.lower() for tok in forbidden) for k in receipt)


def test_kyle_residual_flow_catalog_soft_verify() -> None:
    """Catalog soft-verify: residual keys imply research_only / claim / companions."""
    from quant_fund.research.catalog import (
        kyle_lambda_dispersion_honesty_errors,
        kyle_ofi_nest_honesty_errors,
        kyle_residual_flow_honesty_errors,
    )

    bare = {"family": "kyle_ofi", "research_only": True, "claim": "research_diagnostic_only"}
    assert kyle_residual_flow_honesty_errors(bare) == []

    good = {
        "label": "SYNTHETIC",
        "ic_method": "date_level_spearman_hac",
        "family": "kyle_ofi",
        "research_only": True,
        "claim": "research_diagnostic_only",
        "residual_ofi_ex_depth_fwd_delta_mid_mean_spearman": 0.1,
        "residual_ofi_ex_depth_fwd_delta_mid_t": 1.2,
        "residual_ofi_ex_depth_fwd_delta_mid_n_dates": 10.0,
    }
    assert kyle_residual_flow_honesty_errors(good) == []
    assert kyle_ofi_nest_honesty_errors({"kyle_ofi": good}) == []

    bad_claim = dict(good, claim="live")
    assert "kyle_ofi_residual_claim_invalid" in kyle_residual_flow_honesty_errors(bad_claim)

    bad_missing_t = {
        "family": "kyle_ofi",
        "research_only": True,
        "claim": "research_diagnostic_only",
        "residual_ofi_ex_depth_fwd_delta_mid_mean_spearman": 0.2,
        "residual_ofi_ex_depth_fwd_delta_mid_n_dates": 5.0,
    }
    errs = kyle_residual_flow_honesty_errors(bad_missing_t)
    assert any("missing_t" in e for e in errs)

    leak = dict(good, sharpe_ratio=1.5)
    assert "kyle_ofi_residual_forbidden_metric_keys" in kyle_residual_flow_honesty_errors(leak)

    # Dispersion soft-verify when p50 present
    disp = {
        "family": "kyle_ofi",
        "research_only": True,
        "claim": "research_diagnostic_only",
        "kyle_lambda_depth_p50": 0.01,
    }
    assert kyle_lambda_dispersion_honesty_errors(disp) == []
    assert "kyle_ofi_dispersion_claim_invalid" in kyle_lambda_dispersion_honesty_errors(
        dict(disp, claim="alpha")
    )

    # Live nested bench path (if residual keys present)
    bars, book = _synthetic_bars_and_book(n_assets=6, n_days=28, seed=11)
    receipt = bench_kyle_ofi_fused(bars, book, min_names=3)
    nest_errs = kyle_ofi_nest_honesty_errors({"kyle_ofi": receipt})
    assert nest_errs == [], nest_errs


def test_kyle_ofi_depth_corr_catalog_soft_verify() -> None:
    """Catalog soft-verify: ofi_depth corr keys imply research_only / claim / companions."""
    from quant_fund.research.catalog import (
        kyle_lambda_ofi_depth_corr_honesty_errors,
        kyle_ofi_nest_honesty_errors,
    )

    bare = {"family": "kyle_ofi", "research_only": True, "claim": "research_diagnostic_only"}
    assert kyle_lambda_ofi_depth_corr_honesty_errors(bare) == []

    good = {
        "label": "SYNTHETIC",
        "ic_method": "date_level_spearman_hac",
        "family": "kyle_ofi",
        "research_only": True,
        "claim": "research_diagnostic_only",
        "kyle_lambda_ofi_depth_spearman": 0.5,
        "kyle_lambda_ofi_depth_pearson": 0.4,
        "kyle_lambda_ofi_depth_n_dates": 10,
        "kyle_lambda_ofi_depth_prod_hac_t": 1.2,
        "kyle_lambda_ofi_depth_prod_hac_p": 0.2,
    }
    assert kyle_lambda_ofi_depth_corr_honesty_errors(good) == []
    assert kyle_ofi_nest_honesty_errors({"kyle_ofi": good}) == []

    bad_claim = dict(good, claim="live")
    assert "kyle_ofi_depth_corr_claim_invalid" in kyle_lambda_ofi_depth_corr_honesty_errors(
        bad_claim
    )
    missing = {k: v for k, v in good.items() if k != "kyle_lambda_ofi_depth_n_dates"}
    assert "kyle_ofi_depth_corr_missing:kyle_lambda_ofi_depth_n_dates" in (
        kyle_lambda_ofi_depth_corr_honesty_errors(missing)
    )
    leaked = dict(good, sharpe_ratio=1.0)
    assert "kyle_ofi_depth_corr_forbidden_metric_keys" in kyle_lambda_ofi_depth_corr_honesty_errors(
        leaked
    )


def test_kyle_residual_flow_soft_verify_covers_fwd_ret_2_3() -> None:
    from quant_fund.research.catalog import kyle_residual_flow_honesty_errors

    good = {
        "family": "kyle_ofi",
        "research_only": True,
        "claim": "research_diagnostic_only",
        "residual_ofi_ex_depth_fwd_ret_3_mean_spearman": 0.1,
        "residual_ofi_ex_depth_fwd_ret_3_t": 1.0,
        "residual_ofi_ex_depth_fwd_ret_3_n_dates": 5,
    }
    assert kyle_residual_flow_honesty_errors(good) == []
    missing_t = {
        "family": "kyle_ofi",
        "research_only": True,
        "claim": "research_diagnostic_only",
        "residual_depth_ex_ofi_fwd_ret_2_mean_spearman": 0.2,
        "residual_depth_ex_ofi_fwd_ret_2_n_dates": 4,
    }
    assert "kyle_ofi_residual_missing_t:residual_depth_ex_ofi_fwd_ret_2" in (
        kyle_residual_flow_honesty_errors(missing_t)
    )


def test_kyle_ofi_join_coverage_catalog_soft_verify() -> None:
    """Nest join_coverage finite → (0,1] (or floor) + nonempty book_source."""
    from quant_fund.research.catalog import (
        kyle_ofi_join_coverage_honesty_errors,
        kyle_ofi_nest_honesty_errors,
    )

    bare = {"family": "kyle_ofi", "research_only": True, "claim": "research_diagnostic_only"}
    assert kyle_ofi_join_coverage_honesty_errors(bare) == []

    good = {
        "label": "SYNTHETIC",
        "ic_method": "date_level_spearman_hac",
        "family": "kyle_ofi",
        "join_coverage": 0.95,
        "book_source": "synthetic_lob",
        "research_only": True,
        "claim": "research_diagnostic_only",
    }
    assert kyle_ofi_join_coverage_honesty_errors(good) == []
    assert kyle_ofi_nest_honesty_errors({"kyle_ofi": good}) == []

    bad_cov = dict(good, join_coverage=0.0)
    assert "join_coverage_outside_open_unit_interval_fail_closed" in (
        kyle_ofi_join_coverage_honesty_errors(bad_cov)
    )
    floored = dict(good, join_coverage=0.5, min_join_coverage=0.5)
    assert kyle_ofi_join_coverage_honesty_errors(floored) == []
    no_src = dict(good, book_source="")
    assert "kyle_ofi_join_coverage_book_source_missing" in (
        kyle_ofi_join_coverage_honesty_errors(no_src)
    )
    missing_src = {k: v for k, v in good.items() if k != "book_source"}
    assert "kyle_ofi_join_coverage_book_source_missing" in (
        kyle_ofi_join_coverage_honesty_errors(missing_src)
    )


def test_kyle_lambda_date_series_catalog_soft_verify() -> None:
    """dump/date-series n_* stamps imply research_only + research_diagnostic_only."""
    from quant_fund.research.catalog import (
        kyle_lambda_date_series_honesty_errors,
        kyle_ofi_nest_honesty_errors,
    )

    bare = {"family": "kyle_ofi", "research_only": True, "claim": "research_diagnostic_only"}
    assert kyle_lambda_date_series_honesty_errors(bare) == []

    good = {
        "label": "SYNTHETIC",
        "ic_method": "date_level_spearman_hac",
        "family": "kyle_ofi",
        "research_only": True,
        "claim": "research_diagnostic_only",
        "kyle_lambda_date_series_n_depth": 10,
        "kyle_lambda_date_series_n_ofi": 10,
    }
    assert kyle_lambda_date_series_honesty_errors(good) == []
    assert kyle_ofi_nest_honesty_errors({"kyle_ofi": good}) == []

    bad = dict(good, claim="alpha")
    assert "kyle_ofi_date_series_claim_invalid" in kyle_lambda_date_series_honesty_errors(bad)
    leaked = dict(good, live_pnl_claim=True)
    assert "kyle_ofi_date_series_forbidden_metric_keys" in (
        kyle_lambda_date_series_honesty_errors(leaked)
    )


def test_kyle_ofi_synthetic_source_catalog_soft_verify() -> None:
    """book_dgp/dgp synthetic_lob ⇔ data_source SYNTHETIC; dgp fields match."""
    from quant_fund.research.catalog import (
        kyle_ofi_nest_honesty_errors,
        kyle_ofi_synthetic_source_honesty_errors,
    )

    bare = {"family": "kyle_ofi", "research_only": True, "claim": "research_diagnostic_only"}
    assert kyle_ofi_synthetic_source_honesty_errors(bare) == []

    good = {
        "label": "SYNTHETIC",
        "ic_method": "date_level_spearman_hac",
        "family": "kyle_ofi",
        "book_dgp": "synthetic_lob",
        "dgp": "synthetic_lob",
        "data_source": "SYNTHETIC",
        "book_source": "synthetic_lob",
        "research_only": True,
        "claim": "research_diagnostic_only",
    }
    assert kyle_ofi_synthetic_source_honesty_errors(good) == []
    assert kyle_ofi_nest_honesty_errors({"kyle_ofi": good}) == []

    bad_ds = dict(good, data_source="vendor_x")
    assert "kyle_ofi_synthetic_dgp_data_source_not_SYNTHETIC" in (
        kyle_ofi_synthetic_source_honesty_errors(bad_ds)
    )
    bad_dgp = dict(good, book_dgp="vendor_panel:x", dgp="vendor_panel:x", data_source="SYNTHETIC")
    assert "kyle_ofi_SYNTHETIC_data_source_book_dgp_not_synthetic_lob" in (
        kyle_ofi_synthetic_source_honesty_errors(bad_dgp)
    )
    mismatch = dict(good, dgp="other")
    assert "kyle_ofi_dgp_book_dgp_mismatch" in kyle_ofi_synthetic_source_honesty_errors(mismatch)
    external = {
        "family": "kyle_ofi",
        "book_dgp": "vendor_panel:ab",
        "dgp": "vendor_panel:ab",
        "data_source": "ab",
        "book_source": "ab",
    }
    assert kyle_ofi_synthetic_source_honesty_errors(external) == []
    external_bad = dict(external, data_source="SYNTHETIC")
    assert "kyle_ofi_nonsynthetic_dgp_data_source_SYNTHETIC" in (
        kyle_ofi_synthetic_source_honesty_errors(external_bad)
    )


def test_kyle_ofi_label_and_nest_claim_catalog_soft_verify() -> None:
    """SYN* label vs data_source; any nest marker ⇒ research_only + claim."""
    from quant_fund.research.catalog import (
        kyle_ofi_label_synthetic_honesty_errors,
        kyle_ofi_nest_claim_honesty_errors,
        kyle_ofi_nest_honesty_errors,
    )

    bare = {"family": "kyle_ofi"}
    assert kyle_ofi_label_synthetic_honesty_errors(bare) == []
    assert kyle_ofi_nest_claim_honesty_errors(bare) == []

    good = {
        "ic_method": "date_level_spearman_hac",
        "family": "kyle_ofi",
        "label": "SYNTHETIC",
        "book_dgp": "synthetic_lob",
        "dgp": "synthetic_lob",
        "data_source": "SYNTHETIC",
        "book_source": "synthetic_lob",
        "research_only": True,
        "claim": "research_diagnostic_only",
        "join_coverage": 1.0,
    }
    assert kyle_ofi_label_synthetic_honesty_errors(good) == []
    assert kyle_ofi_nest_claim_honesty_errors(good) == []
    assert kyle_ofi_nest_honesty_errors({"kyle_ofi": good}) == []

    bad_label = dict(good, data_source="vendor_x")
    assert "kyle_ofi_SYN_label_data_source_not_SYNTHETIC" in (
        kyle_ofi_label_synthetic_honesty_errors(bad_label)
    )
    no_claim = {
        "family": "kyle_ofi",
        "join_coverage": 0.9,
        "book_source": "synthetic_lob",
        "research_only": True,
    }
    assert "kyle_ofi_nest_claim_invalid" in kyle_ofi_nest_claim_honesty_errors(no_claim)
    no_ro = {
        "family": "kyle_ofi",
        "kyle_lambda_depth_mean": 0.01,
        "claim": "research_diagnostic_only",
    }
    assert "kyle_ofi_nest_research_only_missing" in kyle_ofi_nest_claim_honesty_errors(no_ro)


def test_kyle_ofi_ic_method_catalog_soft_verify() -> None:
    """ic_method must be date_level_spearman_hac when IC markers present."""
    from quant_fund.research.catalog import (
        kyle_ofi_ic_method_honesty_errors,
        kyle_ofi_nest_honesty_errors,
    )

    bare = {"family": "kyle_ofi"}
    assert kyle_ofi_ic_method_honesty_errors(bare) == []

    good = {
        "min_names": 3,
        "n_fused": 10,
        "n_scored": 8,
        "family": "kyle_ofi",
        "ic_method": "date_level_spearman_hac",
        "kyle_lambda_depth_mean": 0.01,
        "research_only": True,
        "claim": "research_diagnostic_only",
        "book_dgp": "synthetic_lob",
        "dgp": "synthetic_lob",
        "data_source": "SYNTHETIC",
        "book_source": "synthetic_lob",
        "label": "SYNTHETIC",
    }
    assert kyle_ofi_ic_method_honesty_errors(good) == []
    assert kyle_ofi_nest_honesty_errors({"kyle_ofi": good}) == []

    bad = dict(good, ic_method="pearson_pool")
    assert "kyle_ofi_ic_method_invalid" in kyle_ofi_ic_method_honesty_errors(bad)
    missing = {k: v for k, v in good.items() if k != "ic_method"}
    assert "kyle_ofi_ic_method_missing" in kyle_ofi_ic_method_honesty_errors(missing)


def test_kyle_ofi_family_and_hac_lags_catalog_soft_verify() -> None:
    """Markers ⇒ family kyle_ofi; hac_lags optional but ≥0 int; HAC lo≤hi."""
    from quant_fund.research.catalog import (
        kyle_ofi_family_honesty_errors,
        kyle_ofi_hac_lags_honesty_errors,
        kyle_ofi_nest_honesty_errors,
    )

    bare = {"family": "kyle_ofi"}
    assert kyle_ofi_family_honesty_errors(bare) == []
    assert kyle_ofi_hac_lags_honesty_errors(bare) == []

    good = {
        "min_names": 3,
        "n_fused": 10,
        "n_scored": 8,
        "family": "kyle_ofi",
        "ic_method": "date_level_spearman_hac",
        "kyle_lambda_depth_mean": 0.01,
        "research_only": True,
        "claim": "research_diagnostic_only",
        "book_dgp": "synthetic_lob",
        "dgp": "synthetic_lob",
        "data_source": "SYNTHETIC",
        "book_source": "synthetic_lob",
        "label": "SYNTHETIC",
        "hac_lags": 5,
        "kyle_lambda_depth_rolling_hac_lo": -0.1,
        "kyle_lambda_depth_rolling_hac_hi": 0.1,
        "kyle_lambda_depth_rolling_mean": 0.0,
    }
    assert kyle_ofi_family_honesty_errors(good) == []
    assert kyle_ofi_hac_lags_honesty_errors(good) == []
    assert kyle_ofi_nest_honesty_errors({"kyle_ofi": good}) == []

    bad_fam = dict(good, family="northset")
    assert "kyle_ofi_family_invalid" in kyle_ofi_family_honesty_errors(bad_fam)
    missing_fam = {k: v for k, v in good.items() if k != "family"}
    assert "kyle_ofi_family_invalid" in kyle_ofi_family_honesty_errors(missing_fam)
    assert "kyle_ofi_hac_lags_negative" in kyle_ofi_hac_lags_honesty_errors(dict(good, hac_lags=-1))
    assert "kyle_ofi_depth_rolling_hac_lo_gt_hi" in kyle_ofi_hac_lags_honesty_errors(
        dict(good, kyle_lambda_depth_rolling_hac_lo=0.2, kyle_lambda_depth_rolling_hac_hi=0.1)
    )


def test_kyle_ofi_min_names_and_n_fused_catalog_soft_verify() -> None:
    """min_names >= 1 int; n_scored <= n_fused when either count present."""
    from quant_fund.research.catalog import (
        kyle_ofi_min_names_honesty_errors,
        kyle_ofi_n_fused_scored_honesty_errors,
        kyle_ofi_nest_honesty_errors,
    )

    bare = {"family": "kyle_ofi"}
    assert kyle_ofi_min_names_honesty_errors(bare) == []
    assert kyle_ofi_n_fused_scored_honesty_errors(bare) == []

    good = {
        "family": "kyle_ofi",
        "min_names": 3,
        "n_fused": 100,
        "n_scored": 90,
        "ic_method": "date_level_spearman_hac",
        "kyle_lambda_depth_mean": 0.01,
        "research_only": True,
        "claim": "research_diagnostic_only",
        "book_dgp": "synthetic_lob",
        "dgp": "synthetic_lob",
        "data_source": "SYNTHETIC",
        "book_source": "synthetic_lob",
        "label": "SYNTHETIC",
    }
    assert kyle_ofi_min_names_honesty_errors(good) == []
    assert kyle_ofi_n_fused_scored_honesty_errors(good) == []
    assert kyle_ofi_nest_honesty_errors({"kyle_ofi": good}) == []

    assert "kyle_ofi_min_names_lt_one" in kyle_ofi_min_names_honesty_errors(dict(good, min_names=0))
    missing_mn = {k: v for k, v in good.items() if k != "min_names"}
    assert "kyle_ofi_min_names_missing" in kyle_ofi_min_names_honesty_errors(missing_mn)
    assert "kyle_ofi_n_scored_gt_n_fused" in kyle_ofi_n_fused_scored_honesty_errors(
        dict(good, n_scored=200)
    )
    assert "kyle_ofi_n_fused_scored_pair_incomplete" in (
        kyle_ofi_n_fused_scored_honesty_errors(dict(good, n_scored=None))
        if False
        else kyle_ofi_n_fused_scored_honesty_errors(
            {k: v for k, v in good.items() if k != "n_scored"}
        )
    )


def test_kyle_ofi_book_panel_path_and_join_floor_catalog_soft_verify() -> None:
    """book_panel_path nonempty when set; min_join_coverage floor vs join_coverage."""
    from quant_fund.research.catalog import (
        kyle_ofi_book_panel_path_honesty_errors,
        kyle_ofi_min_join_coverage_pair_honesty_errors,
        kyle_ofi_nest_honesty_errors,
    )

    bare = {"family": "kyle_ofi"}
    assert kyle_ofi_book_panel_path_honesty_errors(bare) == []
    assert kyle_ofi_min_join_coverage_pair_honesty_errors(bare) == []

    none_path = {"family": "kyle_ofi", "book_panel_path": None}
    assert kyle_ofi_book_panel_path_honesty_errors(none_path) == []
    assert "kyle_ofi_book_panel_path_empty" in kyle_ofi_book_panel_path_honesty_errors(
        {"family": "kyle_ofi", "book_panel_path": "  "}
    )

    good = {
        "family": "kyle_ofi",
        "book_panel_path": "/tmp/book.parquet",
        "join_coverage": 0.8,
        "min_join_coverage": 0.5,
        "min_names": 3,
        "n_fused": 10,
        "n_scored": 8,
        "ic_method": "date_level_spearman_hac",
        "kyle_lambda_depth_mean": 0.01,
        "research_only": True,
        "claim": "research_diagnostic_only",
        "book_dgp": "synthetic_lob",
        "dgp": "synthetic_lob",
        "data_source": "SYNTHETIC",
        "book_source": "synthetic_lob",
        "label": "SYNTHETIC",
    }
    assert kyle_ofi_book_panel_path_honesty_errors(good) == []
    assert kyle_ofi_min_join_coverage_pair_honesty_errors(good) == []
    assert kyle_ofi_nest_honesty_errors({"kyle_ofi": good}) == []

    assert "kyle_ofi_join_coverage_below_min_join_coverage" in (
        kyle_ofi_min_join_coverage_pair_honesty_errors(
            dict(good, join_coverage=0.2, min_join_coverage=0.5)
        )
    )
    assert "kyle_ofi_min_join_coverage_outside_unit_interval" in (
        kyle_ofi_min_join_coverage_pair_honesty_errors(dict(good, min_join_coverage=1.5))
    )


def test_kyle_ofi_dispersion_window_and_diagnostic_catalog_soft_verify() -> None:
    """dispersion_window >=1 int; diagnostic nonempty string when stamped."""
    from quant_fund.research.catalog import (
        kyle_ofi_diagnostic_string_honesty_errors,
        kyle_ofi_dispersion_window_honesty_errors,
        kyle_ofi_nest_honesty_errors,
    )

    bare = {"family": "kyle_ofi"}
    assert kyle_ofi_dispersion_window_honesty_errors(bare) == []
    assert kyle_ofi_diagnostic_string_honesty_errors(bare) == []

    good = {
        "family": "kyle_ofi",
        "kyle_lambda_dispersion_window": 10,
        "kyle_lambda_depth_p50": 0.01,
        "diagnostic": "residual_ofi_ex_signed_depth_to_fwd_delta_mid",
        "min_names": 3,
        "n_fused": 10,
        "n_scored": 8,
        "ic_method": "date_level_spearman_hac",
        "kyle_lambda_depth_mean": 0.01,
        "research_only": True,
        "claim": "research_diagnostic_only",
        "book_dgp": "synthetic_lob",
        "dgp": "synthetic_lob",
        "data_source": "SYNTHETIC",
        "book_source": "synthetic_lob",
        "label": "SYNTHETIC",
    }
    assert kyle_ofi_dispersion_window_honesty_errors(good) == []
    assert kyle_ofi_diagnostic_string_honesty_errors(good) == []
    assert kyle_ofi_nest_honesty_errors({"kyle_ofi": good}) == []

    assert "kyle_ofi_dispersion_window_lt_one" in kyle_ofi_dispersion_window_honesty_errors(
        dict(good, kyle_lambda_dispersion_window=0)
    )
    missing_w = {k: v for k, v in good.items() if k != "kyle_lambda_dispersion_window"}
    assert "kyle_ofi_dispersion_window_missing" in kyle_ofi_dispersion_window_honesty_errors(
        missing_w
    )
    assert "kyle_ofi_diagnostic_empty" in kyle_ofi_diagnostic_string_honesty_errors(
        dict(good, diagnostic="  ")
    )
    # full bench omits diagnostic — still OK
    benchish = {k: v for k, v in good.items() if k != "diagnostic"}
    assert kyle_ofi_diagnostic_string_honesty_errors(benchish) == []


def test_kyle_ofi_lambda_decile_order_catalog_soft_verify() -> None:
    """depth/ofi p10<=p50<=p90 when all three finite."""
    from quant_fund.research.catalog import (
        kyle_ofi_lambda_decile_order_honesty_errors,
        kyle_ofi_nest_honesty_errors,
    )

    bare = {"family": "kyle_ofi"}
    assert kyle_ofi_lambda_decile_order_honesty_errors(bare) == []

    good = {
        "family": "kyle_ofi",
        "kyle_lambda_depth_p10": 0.01,
        "kyle_lambda_depth_p50": 0.02,
        "kyle_lambda_depth_p90": 0.03,
        "kyle_lambda_ofi_p10": -0.01,
        "kyle_lambda_ofi_p50": 0.0,
        "kyle_lambda_ofi_p90": 0.02,
        "kyle_lambda_dispersion_window": 10,
        "min_names": 3,
        "n_fused": 10,
        "n_scored": 8,
        "ic_method": "date_level_spearman_hac",
        "kyle_lambda_depth_mean": 0.02,
        "research_only": True,
        "claim": "research_diagnostic_only",
        "book_dgp": "synthetic_lob",
        "dgp": "synthetic_lob",
        "data_source": "SYNTHETIC",
        "book_source": "synthetic_lob",
        "label": "SYNTHETIC",
    }
    assert kyle_ofi_lambda_decile_order_honesty_errors(good) == []
    assert kyle_ofi_nest_honesty_errors({"kyle_ofi": good}) == []

    bad_depth = dict(good, kyle_lambda_depth_p50=0.05)  # p50 > p90
    assert "kyle_ofi_depth_decile_order_invalid" in (
        kyle_ofi_lambda_decile_order_honesty_errors(bad_depth)
    )
    bad_ofi = dict(good, kyle_lambda_ofi_p10=0.05)  # p10 > p50
    assert "kyle_ofi_ofi_decile_order_invalid" in (
        kyle_ofi_lambda_decile_order_honesty_errors(bad_ofi)
    )
    # partial triple skips
    partial = {"family": "kyle_ofi", "kyle_lambda_depth_p10": 0.01, "kyle_lambda_depth_p50": 0.02}
    assert kyle_ofi_lambda_decile_order_honesty_errors(partial) == []


def test_kyle_ofi_std_iqr_and_rolling_mean_catalog_soft_verify() -> None:
    """std/iqr >=0 when finite; rolling_mean required when HAC band finite."""
    from quant_fund.research.catalog import (
        kyle_ofi_nest_honesty_errors,
        kyle_ofi_rolling_mean_hac_band_honesty_errors,
        kyle_ofi_std_iqr_honesty_errors,
    )

    bare = {"family": "kyle_ofi"}
    assert kyle_ofi_std_iqr_honesty_errors(bare) == []
    assert kyle_ofi_rolling_mean_hac_band_honesty_errors(bare) == []

    good = {
        "family": "kyle_ofi",
        "kyle_lambda_depth_std": 0.01,
        "kyle_lambda_ofi_std": 0.02,
        "kyle_lambda_depth_iqr": 0.015,
        "kyle_lambda_ofi_iqr": 0.025,
        "kyle_lambda_depth_rolling_hac_lo": -0.01,
        "kyle_lambda_depth_rolling_hac_hi": 0.01,
        "kyle_lambda_depth_rolling_mean": 0.0,
        "kyle_lambda_ofi_rolling_hac_lo": -0.02,
        "kyle_lambda_ofi_rolling_hac_hi": 0.02,
        "kyle_lambda_ofi_rolling_mean": 0.001,
        "kyle_lambda_depth_p10": 0.01,
        "kyle_lambda_depth_p50": 0.02,
        "kyle_lambda_depth_p90": 0.03,
        "kyle_lambda_ofi_p10": -0.01,
        "kyle_lambda_ofi_p50": 0.0,
        "kyle_lambda_ofi_p90": 0.02,
        "kyle_lambda_dispersion_window": 10,
        "min_names": 3,
        "n_fused": 10,
        "n_scored": 8,
        "ic_method": "date_level_spearman_hac",
        "kyle_lambda_depth_mean": 0.02,
        "research_only": True,
        "claim": "research_diagnostic_only",
        "book_dgp": "synthetic_lob",
        "dgp": "synthetic_lob",
        "data_source": "SYNTHETIC",
        "book_source": "synthetic_lob",
        "label": "SYNTHETIC",
    }
    assert kyle_ofi_std_iqr_honesty_errors(good) == []
    assert kyle_ofi_rolling_mean_hac_band_honesty_errors(good) == []
    assert kyle_ofi_nest_honesty_errors({"kyle_ofi": good}) == []

    assert "kyle_ofi_kyle_lambda_depth_std_negative" in kyle_ofi_std_iqr_honesty_errors(
        dict(good, kyle_lambda_depth_std=-0.1)
    )
    missing_mean = {k: v for k, v in good.items() if k != "kyle_lambda_depth_rolling_mean"}
    assert "kyle_ofi_depth_rolling_mean_missing_with_hac_band" in (
        kyle_ofi_rolling_mean_hac_band_honesty_errors(missing_mean)
    )


def test_kyle_ofi_n_dates_companion_catalog_soft_verify() -> None:
    """*_n_dates >=1 when companion mean_spearman / t is finite."""
    from quant_fund.research.catalog import (
        kyle_ofi_n_dates_companion_honesty_errors,
        kyle_ofi_nest_honesty_errors,
    )

    bare = {"family": "kyle_ofi"}
    assert kyle_ofi_n_dates_companion_honesty_errors(bare) == []

    good = {
        "family": "kyle_ofi",
        "kyle_lambda_depth_mean": 0.02,
        "kyle_lambda_depth_t": 1.5,
        "kyle_lambda_depth_n_dates": 10,
        "ofi_delta_mid_lag0_mean_spearman": 0.1,
        "ofi_delta_mid_lag0_t": 2.0,
        "ofi_delta_mid_lag0_n_dates": 8,
        "kyle_lambda_depth_std": 0.01,
        "kyle_lambda_ofi_std": 0.01,
        "kyle_lambda_depth_iqr": 0.01,
        "kyle_lambda_ofi_iqr": 0.01,
        "kyle_lambda_depth_p10": 0.01,
        "kyle_lambda_depth_p50": 0.02,
        "kyle_lambda_depth_p90": 0.03,
        "kyle_lambda_ofi_p10": -0.01,
        "kyle_lambda_ofi_p50": 0.0,
        "kyle_lambda_ofi_p90": 0.02,
        "kyle_lambda_dispersion_window": 10,
        "min_names": 3,
        "n_fused": 10,
        "n_scored": 8,
        "ic_method": "date_level_spearman_hac",
        "research_only": True,
        "claim": "research_diagnostic_only",
        "book_dgp": "synthetic_lob",
        "dgp": "synthetic_lob",
        "data_source": "SYNTHETIC",
        "book_source": "synthetic_lob",
        "label": "SYNTHETIC",
    }
    assert kyle_ofi_n_dates_companion_honesty_errors(good) == []
    assert kyle_ofi_nest_honesty_errors({"kyle_ofi": good}) == []

    assert "kyle_ofi_n_dates_missing:kyle_lambda_depth" in (
        kyle_ofi_n_dates_companion_honesty_errors(
            {k: v for k, v in good.items() if k != "kyle_lambda_depth_n_dates"}
        )
    )
    assert "kyle_ofi_n_dates_lt_one:ofi_delta_mid_lag0" in (
        kyle_ofi_n_dates_companion_honesty_errors(dict(good, ofi_delta_mid_lag0_n_dates=0))
    )
    # rolling hac_t must not demand a nonexistent n_dates companion
    hac_only = {
        "family": "kyle_ofi",
        "kyle_lambda_depth_rolling_hac_t": 1.0,
        "kyle_lambda_depth_rolling_hac_lo": -0.1,
        "kyle_lambda_depth_rolling_hac_hi": 0.1,
        "kyle_lambda_depth_rolling_mean": 0.0,
    }
    assert kyle_ofi_n_dates_companion_honesty_errors(hac_only) == []


def test_kyle_ofi_pvalue_catalog_soft_verify() -> None:
    """Finite *_p stamps must lie in [0, 1]."""
    from quant_fund.research.catalog import (
        kyle_ofi_nest_honesty_errors,
        kyle_ofi_pvalue_honesty_errors,
    )

    bare = {"family": "kyle_ofi"}
    assert kyle_ofi_pvalue_honesty_errors(bare) == []

    good = {
        "family": "kyle_ofi",
        "kyle_lambda_depth_p": 0.18,
        "ofi_delta_mid_lag0_p": 0.05,
        "kyle_lambda_ofi_depth_prod_hac_p": 0.001,
        "kyle_lambda_depth_mean": 0.02,
        "kyle_lambda_depth_t": 1.5,
        "kyle_lambda_depth_n_dates": 10,
        "kyle_lambda_depth_std": 0.01,
        "kyle_lambda_ofi_std": 0.01,
        "kyle_lambda_depth_iqr": 0.01,
        "kyle_lambda_ofi_iqr": 0.01,
        "kyle_lambda_depth_p10": 0.01,
        "kyle_lambda_depth_p50": 0.02,
        "kyle_lambda_depth_p90": 0.03,
        "kyle_lambda_ofi_p10": -0.01,
        "kyle_lambda_ofi_p50": 0.0,
        "kyle_lambda_ofi_p90": 0.02,
        "kyle_lambda_dispersion_window": 10,
        "min_names": 3,
        "n_fused": 10,
        "n_scored": 8,
        "ic_method": "date_level_spearman_hac",
        "research_only": True,
        "claim": "research_diagnostic_only",
        "book_dgp": "synthetic_lob",
        "dgp": "synthetic_lob",
        "data_source": "SYNTHETIC",
        "book_source": "synthetic_lob",
        "label": "SYNTHETIC",
    }
    assert kyle_ofi_pvalue_honesty_errors(good) == []
    assert kyle_ofi_nest_honesty_errors({"kyle_ofi": good}) == []

    assert "kyle_ofi_pvalue_outside_unit_interval:kyle_lambda_depth_p" in (
        kyle_ofi_pvalue_honesty_errors(dict(good, kyle_lambda_depth_p=1.5))
    )
    assert "kyle_ofi_pvalue_outside_unit_interval:ofi_delta_mid_lag0_p" in (
        kyle_ofi_pvalue_honesty_errors(dict(good, ofi_delta_mid_lag0_p=-0.01))
    )
    # NaN p skipped
    assert kyle_ofi_pvalue_honesty_errors(dict(good, kyle_lambda_ofi_p=float("nan"))) == []


def test_kyle_ofi_spearman_catalog_soft_verify() -> None:
    """Finite spearman/pearson stamps must lie in [-1, 1]."""
    from quant_fund.research.catalog import (
        kyle_ofi_nest_honesty_errors,
        kyle_ofi_spearman_honesty_errors,
    )

    bare = {"family": "kyle_ofi"}
    assert kyle_ofi_spearman_honesty_errors(bare) == []

    good = {
        "family": "kyle_ofi",
        "ofi_delta_mid_lag0_mean_spearman": 0.4,
        "ofi_delta_mid_lag0_t": 2.0,
        "ofi_delta_mid_lag0_n_dates": 8,
        "kyle_lambda_ofi_depth_spearman": 0.9,
        "kyle_lambda_ofi_depth_pearson": 0.88,
        "kyle_lambda_ofi_depth_n_dates": 10,
        "kyle_lambda_ofi_depth_prod_hac_t": 1.0,
        "kyle_lambda_ofi_depth_prod_hac_p": 0.1,
        "kyle_lambda_depth_mean": 0.02,
        "kyle_lambda_depth_t": 1.5,
        "kyle_lambda_depth_n_dates": 10,
        "kyle_lambda_depth_p": 0.1,
        "kyle_lambda_depth_std": 0.01,
        "kyle_lambda_ofi_std": 0.01,
        "kyle_lambda_depth_iqr": 0.01,
        "kyle_lambda_ofi_iqr": 0.01,
        "kyle_lambda_depth_p10": 0.01,
        "kyle_lambda_depth_p50": 0.02,
        "kyle_lambda_depth_p90": 0.03,
        "kyle_lambda_ofi_p10": -0.01,
        "kyle_lambda_ofi_p50": 0.0,
        "kyle_lambda_ofi_p90": 0.02,
        "kyle_lambda_dispersion_window": 10,
        "min_names": 3,
        "n_fused": 10,
        "n_scored": 8,
        "ic_method": "date_level_spearman_hac",
        "research_only": True,
        "claim": "research_diagnostic_only",
        "book_dgp": "synthetic_lob",
        "dgp": "synthetic_lob",
        "data_source": "SYNTHETIC",
        "book_source": "synthetic_lob",
        "label": "SYNTHETIC",
    }
    assert kyle_ofi_spearman_honesty_errors(good) == []
    assert kyle_ofi_nest_honesty_errors({"kyle_ofi": good}) == []

    assert "kyle_ofi_corr_outside_unit_interval:ofi_delta_mid_lag0_mean_spearman" in (
        kyle_ofi_spearman_honesty_errors(dict(good, ofi_delta_mid_lag0_mean_spearman=1.2))
    )
    assert "kyle_ofi_corr_outside_unit_interval:kyle_lambda_ofi_depth_pearson" in (
        kyle_ofi_spearman_honesty_errors(dict(good, kyle_lambda_ofi_depth_pearson=-1.1))
    )


def test_kyle_ofi_tstat_catalog_soft_verify() -> None:
    """Present *_t must be finite; finite mean_spearman requires companion *_t."""
    from quant_fund.research.catalog import (
        kyle_ofi_nest_honesty_errors,
        kyle_ofi_tstat_honesty_errors,
    )

    bare = {"family": "kyle_ofi"}
    assert kyle_ofi_tstat_honesty_errors(bare) == []

    good = {
        "family": "kyle_ofi",
        "ofi_delta_mid_lag0_mean_spearman": 0.4,
        "ofi_delta_mid_lag0_t": 2.0,
        "ofi_delta_mid_lag0_n_dates": 8,
        "ofi_delta_mid_lag0_p": 0.05,
        "kyle_lambda_depth_t": 1.5,
        "kyle_lambda_depth_mean": 0.02,
        "kyle_lambda_depth_n_dates": 10,
        "kyle_lambda_depth_p": 0.1,
        "kyle_lambda_depth_std": 0.01,
        "kyle_lambda_ofi_std": 0.01,
        "kyle_lambda_depth_iqr": 0.01,
        "kyle_lambda_ofi_iqr": 0.01,
        "kyle_lambda_depth_p10": 0.01,
        "kyle_lambda_depth_p50": 0.02,
        "kyle_lambda_depth_p90": 0.03,
        "kyle_lambda_ofi_p10": -0.01,
        "kyle_lambda_ofi_p50": 0.0,
        "kyle_lambda_ofi_p90": 0.02,
        "kyle_lambda_dispersion_window": 10,
        "min_names": 3,
        "n_fused": 10,
        "n_scored": 8,
        "ic_method": "date_level_spearman_hac",
        "research_only": True,
        "claim": "research_diagnostic_only",
        "book_dgp": "synthetic_lob",
        "dgp": "synthetic_lob",
        "data_source": "SYNTHETIC",
        "book_source": "synthetic_lob",
        "label": "SYNTHETIC",
    }
    assert kyle_ofi_tstat_honesty_errors(good) == []
    assert kyle_ofi_nest_honesty_errors({"kyle_ofi": good}) == []

    assert "kyle_ofi_tstat_non_finite:kyle_lambda_depth_t" in (
        kyle_ofi_tstat_honesty_errors(dict(good, kyle_lambda_depth_t=float("nan")))
    )
    assert "kyle_ofi_tstat_missing:ofi_delta_mid_lag0" in (
        kyle_ofi_tstat_honesty_errors(
            {k: v for k, v in good.items() if k != "ofi_delta_mid_lag0_t"}
        )
    )


def test_kyle_ofi_label_nonempty_and_date_series_counts_catalog_soft_verify() -> None:
    """Markers require nonempty label; date_series n_* >=0 ints when present."""
    from quant_fund.research.catalog import (
        kyle_ofi_date_series_counts_honesty_errors,
        kyle_ofi_label_nonempty_honesty_errors,
        kyle_ofi_nest_honesty_errors,
    )

    bare = {"family": "kyle_ofi"}
    assert kyle_ofi_label_nonempty_honesty_errors(bare) == []
    assert kyle_ofi_date_series_counts_honesty_errors(bare) == []

    good = {
        "family": "kyle_ofi",
        "label": "SYNTHETIC",
        "kyle_lambda_date_series_n_depth": 10,
        "kyle_lambda_date_series_n_ofi": 9,
        "kyle_lambda_depth_mean": 0.02,
        "kyle_lambda_depth_t": 1.5,
        "kyle_lambda_depth_n_dates": 10,
        "kyle_lambda_depth_p": 0.1,
        "kyle_lambda_depth_std": 0.01,
        "kyle_lambda_ofi_std": 0.01,
        "kyle_lambda_depth_iqr": 0.01,
        "kyle_lambda_ofi_iqr": 0.01,
        "kyle_lambda_depth_p10": 0.01,
        "kyle_lambda_depth_p50": 0.02,
        "kyle_lambda_depth_p90": 0.03,
        "kyle_lambda_ofi_p10": -0.01,
        "kyle_lambda_ofi_p50": 0.0,
        "kyle_lambda_ofi_p90": 0.02,
        "kyle_lambda_dispersion_window": 10,
        "min_names": 3,
        "n_fused": 10,
        "n_scored": 8,
        "ic_method": "date_level_spearman_hac",
        "research_only": True,
        "claim": "research_diagnostic_only",
        "book_dgp": "synthetic_lob",
        "dgp": "synthetic_lob",
        "data_source": "SYNTHETIC",
        "book_source": "synthetic_lob",
    }
    assert kyle_ofi_label_nonempty_honesty_errors(good) == []
    assert kyle_ofi_date_series_counts_honesty_errors(good) == []
    assert kyle_ofi_nest_honesty_errors({"kyle_ofi": good}) == []

    assert "kyle_ofi_label_empty_with_markers" in kyle_ofi_label_nonempty_honesty_errors(
        dict(good, label="  ")
    )
    assert "kyle_ofi_kyle_lambda_date_series_n_depth_negative" in (
        kyle_ofi_date_series_counts_honesty_errors(dict(good, kyle_lambda_date_series_n_depth=-1))
    )
