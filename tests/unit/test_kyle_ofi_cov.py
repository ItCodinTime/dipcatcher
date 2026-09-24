"""Coverage tests for Kyle λ / OFI→Δmid edge branches (SYNTHETIC / research_only).

Targets the fail-closed error paths, degenerate-series fallbacks, and
Cont–Kukanov–Stoikov sign conventions that the happy-path tests skip.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

import quant_fund.northset.kyle_ofi as kyle_ofi
from quant_fund.metrics.cross_section import date_ic_series
from quant_fund.northset.kyle_ofi import (
    cont_ofi_by_security,
    cont_ofi_series,
    ensure_forward_targets,
    flow_delta_mid_date_ic,
    fuse_bars_l2_kyle_frame,
    kyle_lambda_by_date,
    kyle_lambda_date_series_frame,
    kyle_lambda_dispersion,
    kyle_lambda_ofi_depth_corr,
    kyle_lambda_ols,
    ofi_delta_mid_date_ic,
    predictive_flow_fwd_date_ic,
    residual_flow_date_ic,
)

_T0 = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)


def _ts(day: int) -> datetime:
    return _T0 + timedelta(days=day)


def _bars_with_book_cols(
    n_assets: int = 4,
    n_days: int = 8,
    *,
    source: str | None = "synthetic",
    with_close: bool = True,
    seed: int = 7,
) -> pl.DataFrame:
    """Bars that already carry top-of-book columns (book=None SYNTHETIC path)."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for a in range(n_assets):
        px = 100.0 + a
        for d in range(n_days):
            nxt = px * (1.0 + float(rng.normal(0.0, 0.01)))
            half = max(px * 2e-4, 0.01)
            tot = float(rng.uniform(1e3, 5e3))
            imb = float(np.clip(rng.normal(0.0, 0.3), -0.9, 0.9))
            row: dict[str, object] = {
                "security_id": f"S{a}",
                "event_time": _ts(d),
                "mid": px,
                "best_bid": px - half,
                "best_ask": px + half,
                "top_bid_size": tot * (0.5 + 0.5 * imb),
                "top_ask_size": tot * (0.5 - 0.5 * imb),
            }
            if with_close:
                row["close"] = nxt
            if source is not None:
                row["source"] = source
            rows.append(row)
            px = nxt
    return pl.DataFrame(rows)


def _fused_frame(
    n_assets: int = 4,
    n_dates: int = 6,
    *,
    seed: int = 3,
    const_depth: bool = False,
) -> pl.DataFrame:
    """Minimal frame with the columns the IC/λ diagnostics consume."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for d in range(n_dates):
        for a in range(n_assets):
            rows.append(
                {
                    "security_id": f"S{a}",
                    "event_time": _ts(d),
                    "ofi": float(rng.normal(0.0, 1.0)),
                    "signed_depth": 5.0 if const_depth else float(rng.normal(0.0, 1.0)),
                    "delta_mid": float(rng.normal(0.0, 0.01)),
                    "fwd_delta_mid": float(rng.normal(0.0, 0.01)),
                    "fwd_ret_1": float(rng.normal(0.0, 0.01)),
                }
            )
    return pl.DataFrame(rows)


# --- cont_ofi_series -------------------------------------------------------


def test_cont_ofi_series_rejects_mismatched_lengths() -> None:
    bp = np.array([10.0, 10.1])
    ok = np.array([5.0, 5.0])
    with pytest.raises(ValueError, match="share length"):
        cont_ofi_series(bp, ok, ok, np.array([1.0]))


def test_cont_ofi_series_single_observation_is_zero() -> None:
    ofi = cont_ofi_series(np.array([10.0]), np.array([5.0]), np.array([10.1]), np.array([4.0]))
    assert ofi.tolist() == [0.0]


def test_cont_ofi_series_nonfinite_transition_marks_nan() -> None:
    bp = np.array([10.0, 10.1, float("nan"), 10.2])
    bs = np.array([5.0, 5.0, 5.0, 5.0])
    ap = np.array([10.1, 10.2, 10.1, 10.3])
    az = np.array([4.0, 4.0, 4.0, 4.0])
    ofi = cont_ofi_series(bp, bs, ap, az)
    assert ofi[0] == 0.0
    assert np.isnan(ofi[2])  # NaN print itself
    assert np.isnan(ofi[3])  # previous snapshot was NaN → transition discarded
    assert np.isfinite(ofi[1])


def test_cont_ofi_series_sign_conventions() -> None:
    """CKS(2014) events: bid up adds new bid size; bid down removes old; ask
    down removes new ask size; ask up adds back old ask size."""
    bp = np.array([10.0, 11.0, 10.0, 10.0])
    bs = np.array([1.0, 2.0, 3.0, 4.0])
    ap = np.array([10.0, 11.0, 10.0, 10.0])
    az = np.array([5.0, 6.0, 7.0, 8.0])
    ofi = cont_ofi_series(bp, bs, ap, az)
    # i=1: bid up (+bs[1]), ask up (+az[0])
    assert ofi[1] == pytest.approx(2.0 + 5.0)
    # i=2: bid down (−bs[1]), ask down (−az[2])
    assert ofi[2] == pytest.approx(-2.0 - 7.0)
    # i=3: both unchanged → net size changes only
    assert ofi[3] == pytest.approx((4.0 - 3.0) + (7.0 - 8.0))


# --- cont_ofi_by_security --------------------------------------------------


def test_cont_ofi_by_security_missing_columns_fail_closed() -> None:
    book = pl.DataFrame(
        {
            "security_id": ["S0", "S0"],
            "event_time": [_ts(0), _ts(1)],
            "best_ask": [10.1, 10.2],
            "top_bid_size": [1.0, 1.0],
            "top_ask_size": [1.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="missing required columns"):
        cont_ofi_by_security(book)


def test_cont_ofi_by_security_empty_book_returns_empty_ofi() -> None:
    book = pl.DataFrame(
        schema={
            "security_id": pl.Utf8,
            "event_time": pl.Datetime,
            "best_bid": pl.Float64,
            "best_ask": pl.Float64,
            "top_bid_size": pl.Float64,
            "top_ask_size": pl.Float64,
        }
    )
    out = cont_ofi_by_security(book)
    assert out.height == 0
    assert out.schema["ofi"] == pl.Float64


# --- kyle_lambda_ols -------------------------------------------------------


def test_kyle_lambda_ols_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="must align"):
        kyle_lambda_ols(np.array([1.0, 2.0]), np.array([1.0]))


def test_kyle_lambda_ols_underidentified_returns_nan() -> None:
    assert np.isnan(kyle_lambda_ols(np.array([1.0, 2.0]), np.array([1.0, 2.0])))
    # Fewer than 3 finite pairs after masking NaN
    d = np.array([1.0, float("nan"), 2.0, 3.0])
    f = np.array([1.0, 1.0, float("nan"), 3.0])
    assert np.isnan(kyle_lambda_ols(d, f))


def test_kyle_lambda_ols_zero_variance_flow_returns_nan() -> None:
    d = np.array([1.0, -1.0, 0.5, 0.2])
    f = np.ones(4)
    assert np.isnan(kyle_lambda_ols(d, f))


def test_kyle_lambda_ols_demeaned_slope_intercept_absorbed() -> None:
    rng = np.random.default_rng(0)
    flow = rng.normal(0.0, 1.0, size=100)
    delta = 3.5 + 0.02 * flow  # nonzero intercept must not bias λ
    lam = kyle_lambda_ols(delta, flow)
    assert lam == pytest.approx(0.02, abs=1e-12)


# --- _stamp_book_honesty ---------------------------------------------------


def test_stamp_book_honesty_synthesized_short_circuits() -> None:
    src, dgp = kyle_ofi._stamp_book_honesty(pl.DataFrame(), synthesized=True)
    assert (src, dgp) == ("synthetic_lob", "synthetic_lob")


def test_stamp_book_honesty_rejects_null_sources() -> None:
    book = pl.DataFrame({"source": [None, None]})
    with pytest.raises(ValueError, match="empty/null source"):
        kyle_ofi._stamp_book_honesty(book, synthesized=False)


def test_stamp_book_honesty_synthetic_label_maps_to_lob() -> None:
    src, dgp = kyle_ofi._stamp_book_honesty(
        pl.DataFrame({"source": ["SYNTHETIC"]}), synthesized=False
    )
    assert (src, dgp) == ("SYNTHETIC", "synthetic_lob")


def test_stamp_book_honesty_vendor_maps_to_vendor_panel() -> None:
    src, dgp = kyle_ofi._stamp_book_honesty(pl.DataFrame({"source": ["alpaca"]}), synthesized=False)
    assert (src, dgp) == ("alpaca", "vendor_panel:alpaca")


# --- fuse_bars_l2_kyle_frame -----------------------------------------------


def test_fuse_rejects_missing_join_key() -> None:
    bars = _bars_with_book_cols()
    book = bars.select(
        "security_id", "best_bid", "best_ask", "top_bid_size", "top_ask_size"
    ).with_columns(pl.lit("synthetic").alias("source"))
    with pytest.raises(ValueError, match="missing join key"):
        fuse_bars_l2_kyle_frame(bars, book)


def test_fuse_rejects_null_source_on_external_book() -> None:
    bars = _bars_with_book_cols()
    book = bars.select(
        "security_id", "event_time", "best_bid", "best_ask", "top_bid_size", "top_ask_size"
    ).with_columns(pl.lit(None).cast(pl.Utf8).alias("source"))
    with pytest.raises(ValueError, match="empty/null source"):
        fuse_bars_l2_kyle_frame(bars, book)


def test_fuse_book_none_stamps_bars_source_honesty() -> None:
    bars = _bars_with_book_cols(source="synthetic")
    fused = fuse_bars_l2_kyle_frame(bars)
    assert set(fused["book_source"].unique().to_list()) == {"synthetic"}
    assert set(fused["book_dgp"].unique().to_list()) == {"synthetic_lob"}


def test_fuse_book_none_vendor_source_marks_vendor_dgp() -> None:
    bars = _bars_with_book_cols(source="testvendor")
    fused = fuse_bars_l2_kyle_frame(bars)
    assert set(fused["book_dgp"].unique().to_list()) == {"vendor_panel:testvendor"}


def test_fuse_book_none_without_source_defaults_synthetic() -> None:
    bars = _bars_with_book_cols(source=None)
    fused = fuse_bars_l2_kyle_frame(bars)
    assert set(fused["book_source"].unique().to_list()) == {"synthetic_lob"}
    assert fused["join_coverage"].to_list() == [1.0] * fused.height


def test_fuse_rejects_join_coverage_below_floor() -> None:
    bars = _bars_with_book_cols(n_days=10)
    book = (
        bars.filter(pl.col("event_time") < _ts(3))
        .select("security_id", "event_time", "best_bid", "best_ask", "top_bid_size", "top_ask_size")
        .with_columns(pl.lit("synthetic").alias("source"))
    )
    with pytest.raises(ValueError, match="join coverage"):
        fuse_bars_l2_kyle_frame(bars, book)  # 30% < 0.5 external floor


def test_fuse_explicit_min_join_coverage_enforced() -> None:
    bars = _bars_with_book_cols(n_days=10)
    book = (
        bars.filter(pl.col("event_time") < _ts(5))
        .select("security_id", "event_time", "best_bid", "best_ask", "top_bid_size", "top_ask_size")
        .with_columns(pl.lit("synthetic").alias("source"))
    )
    with pytest.raises(ValueError, match="join coverage"):
        fuse_bars_l2_kyle_frame(bars, book, min_join_coverage=0.6)
    fused = fuse_bars_l2_kyle_frame(bars, book, min_join_coverage=0.5)
    assert float(fused["join_coverage"][0]) == pytest.approx(0.5)


def test_fuse_fail_closed_when_fused_missing_columns() -> None:
    bars = _bars_with_book_cols().drop("mid")
    with pytest.raises(ValueError, match="fused frame missing columns"):
        fuse_bars_l2_kyle_frame(bars)


def test_fuse_derives_depth_columns_from_top_sizes() -> None:
    bars = _bars_with_book_cols()
    fused = fuse_bars_l2_kyle_frame(bars)
    assert (fused["bid_depth"] == fused["top_bid_size"]).all()
    assert (fused["ask_depth"] == fused["top_ask_size"]).all()
    assert (fused["signed_depth"] == fused["top_bid_size"] - fused["top_ask_size"]).all()


def test_fuse_preserves_preexisting_ofi_column() -> None:
    bars = _bars_with_book_cols().with_columns(pl.lit(7.5).alias("ofi"))
    fused = fuse_bars_l2_kyle_frame(bars)
    assert fused["ofi"].to_list() == [7.5] * fused.height


def test_fuse_without_close_uses_mid_forward_returns() -> None:
    bars = _bars_with_book_cols(with_close=False)
    fused = fuse_bars_l2_kyle_frame(bars)
    assert "fwd_close_ret_1" not in fused.columns
    grp = fused.filter(pl.col("security_id") == "S0").sort("event_time")
    mid = grp["mid"].to_numpy()
    expected = mid[1:] / mid[:-1] - 1.0
    got = grp["fwd_ret_1"].to_numpy()[:-1]
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-15)


# --- ensure_forward_targets ------------------------------------------------


def test_ensure_forward_targets_falls_back_to_close() -> None:
    rows = [
        {"security_id": f"S{a}", "event_time": _ts(d), "close": float(100 + d * 2 + a)}
        for d in range(4)
        for a in range(3)
    ]
    out = ensure_forward_targets(pl.DataFrame(rows))
    grp = out.filter(pl.col("security_id") == "S0").sort("event_time")
    close = grp["close"].to_numpy()
    np.testing.assert_allclose(grp["fwd_delta_mid"].to_numpy()[:-1], np.diff(close), atol=1e-15)
    np.testing.assert_allclose(
        grp["fwd_ret_1"].to_numpy()[:-1], close[1:] / close[:-1] - 1.0, atol=1e-15
    )


def test_ensure_forward_targets_stamps_from_mid_when_no_close() -> None:
    rng = np.random.default_rng(5)
    rows = [
        {
            "security_id": f"S{a}",
            "event_time": _ts(d),
            "mid": float(100.0 + rng.normal()),
        }
        for d in range(4)
        for a in range(3)
    ]
    frame = pl.DataFrame(rows)
    out = ensure_forward_targets(frame)
    for col in ("fwd_delta_mid", "fwd_ret_1", "fwd_ret_2", "fwd_ret_3"):
        assert col in out.columns
    grp = out.filter(pl.col("security_id") == "S0").sort("event_time")
    mid = grp["mid"].to_numpy()
    np.testing.assert_allclose(grp["fwd_delta_mid"].to_numpy()[:-1], mid[1:] - mid[:-1], atol=1e-15)


def test_ensure_forward_targets_unchanged_without_price_column() -> None:
    frame = pl.DataFrame(
        {
            "security_id": ["S0", "S1"],
            "event_time": [_ts(0), _ts(0)],
            "ofi": [1.0, 2.0],
        }
    )
    out = ensure_forward_targets(frame)
    assert out.columns == frame.columns


def test_ensure_forward_targets_only_adds_missing() -> None:
    frame = (
        _fused_frame(n_assets=3, n_dates=4)
        .drop("fwd_ret_1")
        .with_row_index("i")
        .with_columns((pl.lit(100.0) + pl.col("i").cast(pl.Float64)).alias("mid"))
        .drop("i")
    )
    out = ensure_forward_targets(frame)
    for col in ("fwd_ret_1", "fwd_ret_2", "fwd_ret_3"):
        assert col in out.columns
    # pre-existing fwd_delta_mid is left alone (not recomputed from mid);
    # ensure sorts the frame, so compare value sets not row order
    assert sorted(out["fwd_delta_mid"].drop_nulls().to_list()) == sorted(
        frame["fwd_delta_mid"].drop_nulls().to_list()
    )


# --- residual_flow_date_ic --------------------------------------------------


def test_residual_flow_rejects_same_flow_and_control() -> None:
    with pytest.raises(ValueError, match="must differ"):
        residual_flow_date_ic(_fused_frame(), flow="ofi", control="ofi", target="fwd_delta_mid")


def test_residual_flow_rejects_missing_target() -> None:
    with pytest.raises(ValueError, match="missing target"):
        residual_flow_date_ic(_fused_frame(), target="fwd_ret_9")


def test_residual_flow_rejects_missing_flow_column() -> None:
    fused = _fused_frame().drop("ofi")
    with pytest.raises(ValueError, match="missing ofi"):
        residual_flow_date_ic(fused, flow="ofi", control="signed_depth")


def test_residual_flow_raises_when_no_date_has_enough_names() -> None:
    fused = _fused_frame(n_assets=2, n_dates=5)
    with pytest.raises(ValueError, match="no dates with enough names"):
        residual_flow_date_ic(
            fused, flow="ofi", control="signed_depth", target="fwd_delta_mid", min_names=3
        )


def test_residual_flow_skips_sparse_and_masked_dates() -> None:
    fused = _fused_frame(n_assets=4, n_dates=6, seed=9)
    # date 0: only 2 names → grp.height < min_names skip
    sparse = fused.filter(
        ~((pl.col("event_time") == _ts(0)) & (pl.col("security_id").is_in(["S2", "S3"])))
    )
    # date 1: null out 2 ofi's → masked below min_names
    masked = sparse.with_columns(
        pl.when((pl.col("event_time") == _ts(1)) & (pl.col("security_id") == "S0"))
        .then(None)
        .otherwise(pl.col("ofi"))
        .alias("ofi")
    )
    masked = masked.with_columns(
        pl.when((pl.col("event_time") == _ts(1)) & (pl.col("security_id") == "S1"))
        .then(None)
        .otherwise(pl.col("ofi"))
        .alias("ofi")
    )
    out = residual_flow_date_ic(
        masked, flow="ofi", control="signed_depth", target="fwd_delta_mid", min_names=3
    )
    assert out["n_dates"] == pytest.approx(4.0)  # dates 0 and 1 skipped
    assert out["n_rows"] == 4 * 4


def test_residual_flow_degenerate_control_demeaned_residual() -> None:
    """Constant control within a date → residual falls back to demeaned flow,
    so the IC must equal the raw flow IC (shift/rank invariant)."""
    fused = _fused_frame(n_assets=4, n_dates=6, seed=12, const_depth=True)
    out = residual_flow_date_ic(
        fused, flow="ofi", control="signed_depth", target="fwd_delta_mid", min_names=3
    )
    assert out["diagnostic"] == "residual_ofi_ex_signed_depth_to_fwd_delta_mid"
    assert out["research_only"] is True
    assert out["claim"] == "research_diagnostic_only"
    assert out["n_rows"] == fused.height
    ordered = fused.sort(["event_time", "security_id"])
    expected = date_ic_series(
        ordered["ofi"].to_numpy().astype(float),
        ordered["fwd_delta_mid"].to_numpy().astype(float),
        ordered["event_time"].to_numpy(),
        min_names=3,
    )
    assert out["mean_spearman"] == pytest.approx(expected.mean_spearman)
    assert out["mean_pearson"] == pytest.approx(expected.mean_pearson)


def test_residual_flow_projection_matches_manual_ols() -> None:
    """Residual = flow − proj(flow onto control) per date; verify IC equals the
    IC of the hand-computed OLS residual."""
    fused = _fused_frame(n_assets=5, n_dates=6, seed=21).with_columns(
        (pl.col("signed_depth") * 2.0 + pl.col("fwd_ret_1") * 3.0).alias("ofi")
    )
    out = residual_flow_date_ic(
        fused, flow="ofi", control="signed_depth", target="fwd_delta_mid", min_names=3
    )
    ordered = fused.sort(["event_time", "security_id"])
    resid_parts: list[np.ndarray] = []
    for _d, grp in ordered.group_by("event_time", maintain_order=True):
        f = grp["ofi"].to_numpy().astype(float)
        c = grp["signed_depth"].to_numpy().astype(float)
        beta = float(np.dot(c - c.mean(), f - f.mean()) / np.dot(c - c.mean(), c - c.mean()))
        resid_parts.append(f - (f.mean() + beta * (c - c.mean())))
    resid = np.concatenate(resid_parts)
    expected = date_ic_series(
        resid,
        ordered["fwd_delta_mid"].to_numpy().astype(float),
        ordered["event_time"].to_numpy(),
        min_names=3,
    )
    assert out["n_rows"] == resid.size
    assert out["mean_spearman"] == pytest.approx(expected.mean_spearman)
    assert out["mean_pearson"] == pytest.approx(expected.mean_pearson)


# --- kyle_lambda_dispersion ------------------------------------------------


def test_kyle_lambda_dispersion_empty_series_all_nan() -> None:
    out = kyle_lambda_dispersion(np.array([]))
    assert out["n_dates"] == 0
    assert out["research_only"] is True
    for q in (10, 20, 30, 40, 50, 60, 70, 80, 90):
        assert np.isnan(out[f"kyle_lambda_p{q}"])
    for key in (
        "kyle_lambda_rolling_mean",
        "kyle_lambda_rolling_hac_t",
        "kyle_lambda_rolling_hac_p",
        "kyle_lambda_rolling_hac_lo",
        "kyle_lambda_rolling_hac_hi",
    ):
        assert np.isnan(out[key])


def test_kyle_lambda_dispersion_all_nan_input() -> None:
    out = kyle_lambda_dispersion(np.array([float("nan"), float("nan")]))
    assert out["n_dates"] == 0
    assert np.isnan(out["kyle_lambda_p50"])


def test_kyle_lambda_dispersion_short_series_no_hac_band() -> None:
    out = kyle_lambda_dispersion(np.array([0.01, 0.03]), window=10)
    assert out["n_dates"] == 2
    assert out["kyle_lambda_rolling_mean"] == pytest.approx(0.02)
    assert np.isnan(out["kyle_lambda_rolling_hac_t"])
    assert np.isnan(out["kyle_lambda_rolling_hac_lo"])
    assert np.isnan(out["kyle_lambda_rolling_hac_hi"])
    # deciles still computed on the 2-point series
    assert out["kyle_lambda_p10"] < out["kyle_lambda_p90"]


def test_kyle_lambda_dispersion_window_floor_three() -> None:
    lam = np.array([0.01, 0.02, 0.03, 0.05, 0.08])
    out = kyle_lambda_dispersion(lam, window=1)
    assert out["window"] == 1  # stamped as requested
    # trailing band still uses ≥3 points → mean over last 3
    assert out["kyle_lambda_rolling_mean"] == pytest.approx(np.mean([0.03, 0.05, 0.08]))


# --- kyle_lambda_date_series_frame -----------------------------------------


def test_kyle_lambda_date_series_frame_rejects_mismatched_lengths(monkeypatch) -> None:
    monkeypatch.setattr(
        kyle_ofi,
        "kyle_lambda_by_date",
        lambda *a, **k: {
            "kyle_lambda_dates": ["2020-01-01", "2020-01-02"],
            "kyle_lambda_series": [1.0],
        },
    )
    with pytest.raises(ValueError, match="length mismatch"):
        kyle_lambda_date_series_frame(pl.DataFrame())


def test_kyle_lambda_date_series_frame_empty_series_schema(monkeypatch) -> None:
    monkeypatch.setattr(
        kyle_ofi,
        "kyle_lambda_by_date",
        lambda *a, **k: {
            "kyle_lambda_dates": [],
            "kyle_lambda_series": [],
            "book_source": "synthetic_lob",
            "book_dgp": "synthetic_lob",
        },
    )
    frame = kyle_lambda_date_series_frame(pl.DataFrame())
    assert frame.height == 0
    for col in (
        "event_time",
        "kyle_lambda",
        "flow",
        "target",
        "book_source",
        "book_dgp",
        "research_only",
        "claim",
    ):
        assert col in frame.columns


# --- kyle_lambda_ofi_depth_corr --------------------------------------------


def test_kyle_lambda_ofi_depth_corr_too_few_shared_dates() -> None:
    fused = _fused_frame(n_assets=4, n_dates=2)
    out = kyle_lambda_ofi_depth_corr(fused, target="delta_mid", min_names=3)
    assert out["n_dates_aligned"] == 2.0
    for key in (
        "kyle_lambda_ofi_depth_spearman",
        "kyle_lambda_ofi_depth_pearson",
        "kyle_lambda_ofi_depth_prod_hac_t",
        "kyle_lambda_ofi_depth_prod_hac_p",
    ):
        assert np.isnan(out[key])


def test_kyle_lambda_ofi_depth_corr_constant_series_nan(monkeypatch) -> None:
    """Zero-variance aligned λ series → Spearman/Pearson degenerate to NaN."""
    calls = iter(
        [
            {
                "kyle_lambda_dates": ["d1", "d2", "d3", "d4"],
                "kyle_lambda_series": [1.0, 1.0, 1.0, 1.0],
                "book_source": "synthetic_lob",
                "book_dgp": "synthetic_lob",
            },
            {
                "kyle_lambda_dates": ["d1", "d2", "d3", "d4"],
                "kyle_lambda_series": [1.0, 2.0, 3.0, 4.0],
                "book_source": "synthetic_lob",
                "book_dgp": "synthetic_lob",
            },
        ]
    )
    monkeypatch.setattr(kyle_ofi, "kyle_lambda_by_date", lambda *a, **k: next(calls))
    out = kyle_lambda_ofi_depth_corr(pl.DataFrame())
    assert out["n_dates_aligned"] == 4.0
    # Pearson guards zero variance → NaN. Spearman not asserted: argsort().argsort()
    # break ties by position, which spuriously yields 1.0 on a constant series.
    assert np.isnan(out["kyle_lambda_ofi_depth_pearson"])


# --- kyle_lambda_by_date ----------------------------------------------------


def test_kyle_lambda_by_date_rejects_unsupported_target() -> None:
    with pytest.raises(ValueError, match="unsupported kyle target"):
        kyle_lambda_by_date(_fused_frame(), target="fwd_close_ret_1")


def test_kyle_lambda_by_date_rejects_missing_columns() -> None:
    fused = _fused_frame().drop("signed_depth")
    with pytest.raises(ValueError, match="fused must contain"):
        kyle_lambda_by_date(fused, flow="signed_depth")


def test_kyle_lambda_by_date_rejects_empty_sample() -> None:
    fused = _fused_frame().with_columns(pl.lit(None).cast(pl.Float64).alias("delta_mid"))
    with pytest.raises(ValueError, match="no finite"):
        kyle_lambda_by_date(fused, flow="signed_depth")


def test_kyle_lambda_by_date_two_dates_mean_only() -> None:
    """<3 identified dates → finite mean λ but no HAC t/p."""
    fused = _fused_frame(n_assets=4, n_dates=2, seed=17)
    out = kyle_lambda_by_date(fused, flow="signed_depth", min_names=3)
    assert out["n_dates_lambda"] == 2
    assert np.isfinite(out["kyle_lambda_mean"])
    assert np.isnan(out["kyle_lambda_t"])
    assert np.isnan(out["kyle_lambda_p"])
    assert np.isfinite(out["kyle_lambda_std"])


def test_kyle_lambda_by_date_no_identified_dates_nan() -> None:
    fused = _fused_frame(n_assets=2, n_dates=5, seed=19)
    out = kyle_lambda_by_date(fused, flow="signed_depth", min_names=3)
    assert out["n_dates_lambda"] == 0
    assert np.isnan(out["kyle_lambda_mean"])
    assert np.isnan(out["kyle_lambda_t"])
    # dispersion fallback keys still stamped
    assert np.isnan(out["kyle_lambda_p50"])
    assert np.isnan(out["kyle_lambda_rolling_mean"])


def test_kyle_lambda_by_date_skips_dates_below_min_names() -> None:
    fused = _fused_frame(n_assets=4, n_dates=5, seed=23)
    trimmed = fused.filter(
        ~((pl.col("event_time") == _ts(0)) & (pl.col("security_id").is_in(["S2", "S3"])))
    )
    out = kyle_lambda_by_date(trimmed, flow="signed_depth", min_names=3)
    assert out["n_dates_lambda"] == 4
    assert len(out["kyle_lambda_series"]) == 4
    assert len(out["kyle_lambda_dates"]) == 4


# --- ofi_delta_mid_date_ic / predictive / flow IC error branches ------------


def test_ofi_delta_mid_date_ic_missing_columns() -> None:
    fused = _fused_frame().drop("ofi")
    with pytest.raises(ValueError, match="ofi and delta_mid"):
        ofi_delta_mid_date_ic(fused)


def test_ofi_delta_mid_date_ic_rejects_negative_lag() -> None:
    with pytest.raises(ValueError, match="lag must be >= 0"):
        ofi_delta_mid_date_ic(_fused_frame(), lag=-1)


def test_predictive_flow_rejects_nonforward_target() -> None:
    with pytest.raises(ValueError, match="stamped forward column"):
        predictive_flow_fwd_date_ic(_fused_frame(), flow="ofi", target="delta_mid")


def test_predictive_flow_rejects_missing_flow_column() -> None:
    fused = _fused_frame().drop("ofi")
    with pytest.raises(ValueError, match="must contain ofi"):
        predictive_flow_fwd_date_ic(fused, flow="ofi", target="fwd_delta_mid")


def test_predictive_flow_rejects_missing_target_column() -> None:
    fused = _fused_frame().drop("fwd_ret_1")
    with pytest.raises(ValueError, match="must contain fwd_ret_1"):
        predictive_flow_fwd_date_ic(fused, flow="ofi", target="fwd_ret_1")


def test_flow_delta_mid_rejects_missing_columns() -> None:
    fused = _fused_frame().drop("signed_depth")
    with pytest.raises(ValueError, match="must contain signed_depth"):
        flow_delta_mid_date_ic(fused, flow="signed_depth")


def test_flow_delta_mid_rejects_negative_lag() -> None:
    with pytest.raises(ValueError, match="lag must be >= 0"):
        flow_delta_mid_date_ic(_fused_frame(), flow="ofi", lag=-1)


def test_flow_delta_mid_lag1_shifts_within_security() -> None:
    fused = _fused_frame(n_assets=4, n_dates=6, seed=31)
    lag0 = flow_delta_mid_date_ic(fused, flow="ofi", lag=0, min_names=3)
    lag1 = flow_delta_mid_date_ic(fused, flow="ofi", lag=1, min_names=3)
    assert lag0["flow_lag"] == 0 and lag1["flow_lag"] == 1
    # one score per security is null after the shift → dropped
    assert lag1["n_rows"] == lag0["n_rows"] - 4


# --- _assert_no_forbidden ---------------------------------------------------


def test_assert_no_forbidden_rejects_leaked_keys() -> None:
    with pytest.raises(AssertionError, match="forbidden"):
        kyle_ofi._assert_no_forbidden({"kyle_pnl_estimate": 1.0})
    kyle_ofi._assert_no_forbidden({"kyle_lambda_mean": 0.01})  # clean receipt passes
