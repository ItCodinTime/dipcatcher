"""Candle-book and Kyle/OFI commands.

Split out of the original module. Import the parent path; it re-exports these names.
"""

from __future__ import annotations

from pathlib import Path

import typer

from .app import app
from .support import _cfg, format_data_label


@app.command("candle-book")
def candle_book(
    config: Path = typer.Option(Path("configs/research.yaml")),
    depth: int = typer.Option(5, help="L2 depth for SYNTHETIC book"),
    seed: int = typer.Option(7, help="RNG seed for SYNTHETIC L2"),
    book: Path | None = typer.Option(
        None, help="Optional L2 parquet: Northset panel or raw vendor quotes"
    ),
    vendor: str | None = typer.Option(
        None, help="If set, remap raw vendor quotes (alpaca|polygon|generic)"
    ),
    bars_path: Path | None = typer.Option(
        None, help="Optional OHLCV parquet aligned to --book (else SYNTHETIC bars)"
    ),
) -> None:
    """Fused candlestick + order-book research bench (optional family).

    Default: SYNTHETIC bars + SYNTHETIC L2. With ``--book``, fuses external/
    vendor-mapped panels (research-only). Not in REQUIRED_BENCHMARK_FAMILIES.
    """
    import polars as pl

    from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
    from quant_fund.microstructure import bench_candle_order_book
    from quant_fund.microstructure.book_panel import load_book_panel, validate_book_panel
    from quant_fund.microstructure.vendor_book_map import remap_vendor_quotes_to_panel

    cfg = _cfg(config)
    book_df = None
    if book is not None:
        raw = pl.read_parquet(book)
        if vendor:
            book_df = remap_vendor_quotes_to_panel(raw, vendor=str(vendor))
        elif {"best_bid", "best_ask", "security_id", "event_time"} <= set(raw.columns):
            book_df = validate_book_panel(raw) if "source" in raw.columns else load_book_panel(book)
            if "source" not in book_df.columns:
                raise typer.BadParameter("book panel missing source column (honesty)")
        else:
            raise typer.BadParameter("pass --vendor for raw quote schemas, or a Northset panel")
    if bars_path is not None:
        bars = pl.read_parquet(bars_path)
    else:
        n_assets = int(getattr(getattr(cfg, "data", None), "n_assets", 8) or 8)
        n_days = int(getattr(getattr(cfg, "data", None), "n_days", 120) or 120)
        bars = SyntheticMarketProvider(
            n_assets=max(n_assets, 4), n_days=max(n_days, 40), seed=seed
        ).get_bars()
    receipt = bench_candle_order_book(bars, book=book_df, depth=depth, seed=seed, label="SYNTHETIC")
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo("SYNTHETIC")
    typer.echo("candle_order_book — fused candlestick + L2 research (optional family)")
    typer.echo(
        f"family={receipt.get('family')} "
        f"n_bars={receipt['n_bars']} n_scored={receipt['n_scored']} depth={receipt['depth']} "
        f"n_fused={receipt.get('n_fused')} min_names={receipt.get('min_names')} "
        f"book_source={receipt.get('book_source')} book_dgp={receipt.get('book_dgp')} "
        f"join_coverage={receipt.get('join_coverage')} "
        f"mean_book_age_s={receipt.get('mean_book_age_seconds')} "
        f"max_book_age_s={receipt.get('max_book_age_seconds')} "
        f"mean_microprice_minus_mid={receipt.get('mean_microprice_minus_mid')} "
        f"depth_shape_finite_rate={receipt.get('depth_shape_finite_rate')} "
        f"structure_finite_rate={receipt.get('structure_finite_rate')} "
        f"finite_rate_microprice_minus_mid={receipt.get('finite_rate_microprice_minus_mid')} "
        f"finite_rate_bid_size_concentration_top={receipt.get('finite_rate_bid_size_concentration_top')} "
        f"finite_rate_ask_size_concentration_top={receipt.get('finite_rate_ask_size_concentration_top')} "
        f"ic_method={receipt.get('ic_method')}"
    )
    typer.echo(
        f"mean_abs_ic={receipt.get('mean_abs_ic')} "
        f"best={receipt.get('best_feature_ic_key')}@{receipt.get('best_feature_ic')}"
    )
    # Spread-alias means: quoted == effective (book alias); close_mid_abs_rel is
    # the candle 2·|C−mid|/mid diagnostic — never equated with either spread.
    typer.echo(
        f"mean_quoted_spread={receipt.get('mean_quoted_spread')} "
        f"mean_effective_spread={receipt.get('mean_effective_spread')} "
        f"mean_half_spread={receipt.get('mean_half_spread')} "
        f"mean_half_spread_bps={receipt.get('mean_half_spread_bps')} "
        f"mean_spread_bps={receipt.get('mean_spread_bps')} "
        f"mean_close_mid_abs_rel={receipt.get('mean_close_mid_abs_rel')} "
        f"mean_microprice_weight_balance={receipt.get('mean_microprice_weight_balance')} "
    )
    # Residual honesty companions (identity rates / path means / sweep evidence).
    typer.echo(
        f"gap_finite_rate={receipt.get('gap_finite_rate')} "
        f"queue_imbalance_mean={receipt.get('queue_imbalance_mean')} "
        f"mean_bid_log_size_slope={receipt.get('mean_bid_log_size_slope')} "
        f"n_sweep_high={receipt.get('n_sweep_high')} "
        f"overnight_share={receipt.get('overnight_share')} "
        f"session_mean_rv={receipt.get('session_mean_rv')} "
        f"yang_zhang_variance={receipt.get('yang_zhang_variance')} "
        f"sweep_reject_fold_positive_fraction={receipt.get('sweep_reject_fold_positive_fraction')} "
        f"mean_fwd_ret_after_high_reclaim={receipt.get('mean_fwd_ret_after_high_reclaim')} "
        f"book_hypothesis_eligible={receipt.get('book_hypothesis_eligible')}"
    )
    # Full non-IC receipt parity: every bench stamp is echoed (source-level test).
    typer.echo(
        f"best_feature_ic={receipt.get('best_feature_ic')} "
        f"best_feature_ic_key={receipt.get('best_feature_ic_key')} "
        f"book_dgp={receipt.get('book_dgp')} "
        f"book_source={receipt.get('book_source')} "
        f"claim={receipt.get('claim')} "
    )
    typer.echo(
        f"data_source={receipt.get('data_source')} "
        f"depth={receipt.get('depth')} "
        f"depth_shape_finite_rate={receipt.get('depth_shape_finite_rate')} "
        f"dgp={receipt.get('dgp')} "
        f"family={receipt.get('family')} "
    )
    typer.echo(
        f"finite_rate_ask_size_concentration_top={receipt.get('finite_rate_ask_size_concentration_top')} "
        f"finite_rate_bid_size_concentration_top={receipt.get('finite_rate_bid_size_concentration_top')} "
        f"finite_rate_microprice_minus_mid={receipt.get('finite_rate_microprice_minus_mid')} "
        f"join_coverage={receipt.get('join_coverage')} "
        f"label={receipt.get('label')} "
    )
    typer.echo(
        f"max_book_age_seconds={receipt.get('max_book_age_seconds')} "
        f"mean_abs_ic={receipt.get('mean_abs_ic')} "
        f"mean_ask_log_price_slope={receipt.get('mean_ask_log_price_slope')} "
        f"mean_ask_log_size_slope={receipt.get('mean_ask_log_size_slope')} "
        f"mean_ask_mean_log_tick_spacing={receipt.get('mean_ask_mean_log_tick_spacing')} "
    )
    typer.echo(
        f"mean_ask_queue_priority_proxy={receipt.get('mean_ask_queue_priority_proxy')} "
        f"mean_ask_size_concentration_top={receipt.get('mean_ask_size_concentration_top')} "
        f"mean_bid_log_price_slope={receipt.get('mean_bid_log_price_slope')} "
        f"mean_bid_log_size_slope={receipt.get('mean_bid_log_size_slope')} "
        f"mean_bid_mean_log_tick_spacing={receipt.get('mean_bid_mean_log_tick_spacing')} "
    )
    typer.echo(
        f"mean_bid_size_concentration_top={receipt.get('mean_bid_size_concentration_top')} "
        f"mean_book_age_seconds={receipt.get('mean_book_age_seconds')} "
        f"mean_candle_body_frac={receipt.get('mean_candle_body_frac')} "
        f"mean_candle_body_ret={receipt.get('mean_candle_body_ret')} "
        f"mean_candle_dir_x_imbalance={receipt.get('mean_candle_dir_x_imbalance')} "
    )
    typer.echo(
        f"mean_candle_direction={receipt.get('mean_candle_direction')} "
        f"mean_candle_range_frac={receipt.get('mean_candle_range_frac')} "
        f"mean_close_mid_abs_rel={receipt.get('mean_close_mid_abs_rel')} "
        f"mean_depth_imbalance={receipt.get('mean_depth_imbalance')} "
        f"mean_depth_imbalance_abs={receipt.get('mean_depth_imbalance_abs')} "
    )
    typer.echo(
        f"mean_effective_spread={receipt.get('mean_effective_spread')} "
        f"mean_half_spread={receipt.get('mean_half_spread')} "
        f"mean_half_spread_bps={receipt.get('mean_half_spread_bps')} "
        f"mean_imbalance_top={receipt.get('mean_imbalance_top')} "
        f"mean_imbalance_x_body_frac={receipt.get('mean_imbalance_x_body_frac')} "
    )
    typer.echo(
        f"mean_microprice_minus_mid={receipt.get('mean_microprice_minus_mid')} "
        f"mean_microprice_minus_mid_bps={receipt.get('mean_microprice_minus_mid_bps')} "
        f"mean_microprice_weight_balance={receipt.get('mean_microprice_weight_balance')} "
        f"mean_notional_imbalance={receipt.get('mean_notional_imbalance')} "
        f"mean_ofi={receipt.get('mean_ofi')} "
    )
    typer.echo(
        f"mean_queue_imbalance={receipt.get('mean_queue_imbalance')} "
        f"mean_queue_priority_proxy={receipt.get('mean_queue_priority_proxy')} "
        f"mean_quoted_spread={receipt.get('mean_quoted_spread')} "
        f"mean_signed_vol_x_imbalance={receipt.get('mean_signed_vol_x_imbalance')} "
        f"mean_spread_bps={receipt.get('mean_spread_bps')} "
    )
    typer.echo(
        f"mean_spread_over_mid={receipt.get('mean_spread_over_mid')} "
        f"mean_spread_x_range={receipt.get('mean_spread_x_range')} "
        f"mean_tob_size_share={receipt.get('mean_tob_size_share')} "
        f"mean_wick_skew={receipt.get('mean_wick_skew')} "
        f"min_names={receipt.get('min_names')} "
    )
    typer.echo(
        f"n_bars={receipt.get('n_bars')} "
        f"n_fused={receipt.get('n_fused')} "
        f"n_scored={receipt.get('n_scored')} "
        f"structure_finite_rate={receipt.get('structure_finite_rate')} "
    )
    typer.echo(f"research_only={receipt.get('research_only')} claim={receipt.get('claim')}")
    ics: list[tuple[str, float]] = []
    for k, v in receipt.items():
        key = str(k)
        if not key.startswith("ic_") or key.endswith(("_t", "_p", "_n_dates", "_pearson")):
            continue
        try:
            val = float(v)
        except (TypeError, ValueError):
            continue
        if val == val:  # finite
            ics.append((key, val))
    ics.sort(key=lambda kv: abs(kv[1]), reverse=True)
    for key, value in ics[:8]:
        typer.echo(f"{key}={float(value):.4f}")


@app.command("kyle-ofi")
def kyle_ofi(
    config: Path = typer.Option(Path("configs/research.yaml")),
    seed: int = typer.Option(7, help="RNG seed for SYNTHETIC bars when --bars omitted"),
    book: Path | None = typer.Option(
        None, help="Optional L2 parquet: Northset panel or raw vendor quotes"
    ),
    vendor: str | None = typer.Option(
        None, help="If set, remap raw vendor quotes (alpaca|polygon|generic)"
    ),
    bars_path: Path | None = typer.Option(
        None, help="Optional OHLCV parquet aligned to --book (else SYNTHETIC bars)"
    ),
    dump_lambda_series: Path | None = typer.Option(
        None,
        "--dump-lambda-series",
        help="Write research_only Kyle λ date-series parquet (depth+ofi rows; claim=research_diagnostic_only; never Sharpe/pnl)",
    ),
) -> None:
    """Kyle λ / OFI→Δmid research diagnostics (date-level IC + HAC).

    Default: SYNTHETIC bars + SYNTHETIC L2. With ``--book``, fuses external/
    vendor-mapped panels (research-only). Stamps book_source / book_dgp.
    Optional ``--dump-lambda-series`` writes a research_only λ panel parquet.
    """
    import polars as pl

    from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
    from quant_fund.microstructure.book_panel import load_book_panel, validate_book_panel
    from quant_fund.microstructure.vendor_book_map import remap_vendor_quotes_to_panel
    from quant_fund.northset.kyle_ofi import (
        bench_kyle_ofi_fused,
        fuse_bars_l2_kyle_frame,
        kyle_lambda_date_series_frame,
    )

    cfg = _cfg(config)
    book_df = None
    book_panel_path = None
    if book is not None:
        book_panel_path = str(book)
        raw = pl.read_parquet(book)
        if vendor:
            book_df = remap_vendor_quotes_to_panel(raw, vendor=str(vendor))
        elif {"best_bid", "best_ask", "security_id", "event_time"} <= set(raw.columns):
            book_df = validate_book_panel(raw) if "source" in raw.columns else load_book_panel(book)
            if "source" not in book_df.columns:
                raise typer.BadParameter("book panel missing source column (honesty)")
        else:
            raise typer.BadParameter("pass --vendor for raw quote schemas, or a Northset panel")
    if bars_path is not None:
        bars = pl.read_parquet(bars_path)
    else:
        n_assets = int(getattr(getattr(cfg, "data", None), "n_assets", 8) or 8)
        n_days = int(getattr(getattr(cfg, "data", None), "n_days", 120) or 120)
        bars = SyntheticMarketProvider(
            n_assets=max(n_assets, 4), n_days=max(n_days, 40), seed=seed
        ).get_bars()
        # When no external book, synthesize aligned TOB on bars for research path
        if book_df is None:
            import numpy as np

            rng = np.random.default_rng(seed)
            rows = []
            for row in bars.iter_rows(named=True):
                close = float(row["close"])
                half = max(close * 2e-4, 0.01)
                try:
                    vol = float(row["volume"])
                except (KeyError, TypeError, ValueError):
                    raise typer.BadParameter(
                        "kyle-ofi synthetic book requires bar volume; a fabricated "
                        "default would invent order sizes"
                    ) from None
                if not np.isfinite(vol) or vol <= 0.0:
                    raise typer.BadParameter(
                        "kyle-ofi synthetic book requires positive finite bar volume; "
                        "missing/zero volume would fabricate order sizes"
                    )
                imb = float(rng.uniform(-0.3, 0.3))
                top_total = max(vol * 0.02, 1.0)
                rows.append(
                    {
                        "security_id": row["security_id"],
                        "event_time": row["event_time"],
                        "available_time": row.get("available_time", row["event_time"]),
                        "source": "synthetic",
                        "best_bid": close - half,
                        "best_ask": close + half,
                        "mid": close,
                        "top_bid_size": top_total * (0.5 + 0.5 * imb),
                        "top_ask_size": top_total * (0.5 - 0.5 * imb),
                    }
                )
            book_df = pl.DataFrame(rows)
    receipt = bench_kyle_ofi_fused(
        bars, book_df, min_names=3, label="SYNTHETIC", book_panel_path=book_panel_path
    )

    if dump_lambda_series is not None:
        fused = fuse_bars_l2_kyle_frame(bars, book_df)
        depth_s = kyle_lambda_date_series_frame(
            fused, flow="signed_depth", target="delta_mid", min_names=3
        )
        ofi_s = kyle_lambda_date_series_frame(fused, flow="ofi", target="delta_mid", min_names=3)
        out_df = pl.concat([depth_s, ofi_s]).sort(["flow", "event_time"])
        dump_lambda_series.parent.mkdir(parents=True, exist_ok=True)
        out_df.write_parquet(dump_lambda_series)
        typer.echo(
            f"dumped_lambda_series={dump_lambda_series} rows={out_df.height} "
            f"research_only=True claim=research_diagnostic_only"
        )
    synthetic = str(receipt.get("book_dgp", "")).startswith("synthetic")
    typer.echo(format_data_label(synthetic=synthetic, data_source=str(receipt.get("data_source"))))
    typer.echo("SYNTHETIC" if synthetic else str(receipt.get("book_source")))
    typer.echo("kyle_ofi — Kyle λ / OFI→Δmid research (date-level IC + HAC)")
    typer.echo(
        f"n_fused={receipt['n_fused']} n_scored={receipt['n_scored']} "
        f"book_source={receipt.get('book_source')} book_dgp={receipt.get('book_dgp')} "
        f"join_coverage={receipt.get('join_coverage')} "
        f"ic_method={receipt.get('ic_method')}"
    )
    typer.echo(
        f"kyle_λ_depth={receipt.get('kyle_lambda_depth_mean')} "
        f"t={receipt.get('kyle_lambda_depth_t')} "
        f"kyle_λ_ofi={receipt.get('kyle_lambda_ofi_mean')} "
        f"t={receipt.get('kyle_lambda_ofi_t')}"
    )
    typer.echo(
        f"ofi_Δmid_lag0_spearman={receipt.get('ofi_delta_mid_lag0_mean_spearman')} "
        f"lag1={receipt.get('ofi_delta_mid_lag1_mean_spearman')} "
        f"signed_depth_lag1={receipt.get('signed_depth_delta_mid_lag1_mean_spearman')}"
    )
    typer.echo(
        f"kyle_λ_depth_std={receipt.get('kyle_lambda_depth_std')} "
        f"iqr={receipt.get('kyle_lambda_depth_iqr')} "
        f"kyle_λ_ofi_std={receipt.get('kyle_lambda_ofi_std')} "
        f"iqr={receipt.get('kyle_lambda_ofi_iqr')}"
    )
    typer.echo(
        f"ofi_fwd_Δmid={receipt.get('ofi_fwd_delta_mid_mean_spearman')} "
        f"ofi_fwd_ret_1={receipt.get('ofi_fwd_ret_1_mean_spearman')} "
        f"depth_fwd_Δmid={receipt.get('signed_depth_fwd_delta_mid_mean_spearman')} "
        f"depth_fwd_ret_1={receipt.get('signed_depth_fwd_ret_1_mean_spearman')}"
    )
    typer.echo(
        f"kyle_λ_depth_fwd_ret_1={receipt.get('kyle_lambda_depth_fwd_ret_1_mean')} "
        f"kyle_λ_ofi_fwd_ret_1={receipt.get('kyle_lambda_ofi_fwd_ret_1_mean')} "
        f"ofi_fwd_ret_2={receipt.get('ofi_fwd_ret_2_mean_spearman')} "
        f"depth_fwd_ret_2={receipt.get('signed_depth_fwd_ret_2_mean_spearman')}"
    )
    typer.echo(
        f"residual_ofi_ex_depth_fwd_Δmid={receipt.get('residual_ofi_ex_depth_fwd_delta_mid_mean_spearman')} "
        f"t={receipt.get('residual_ofi_ex_depth_fwd_delta_mid_t')} "
        f"residual_depth_ex_ofi={receipt.get('residual_depth_ex_ofi_fwd_delta_mid_mean_spearman')}"
    )
    typer.echo(
        f"residual_depth_ex_ofi_fwd_ret_1={receipt.get('residual_depth_ex_ofi_fwd_ret_1_mean_spearman')} "
        f"t={receipt.get('residual_depth_ex_ofi_fwd_ret_1_t')} "
        f"residual_ofi_ex_depth_fwd_ret_1={receipt.get('residual_ofi_ex_depth_fwd_ret_1_mean_spearman')}"
    )
    typer.echo(
        f"residual_depth_ex_ofi_fwd_ret_2={receipt.get('residual_depth_ex_ofi_fwd_ret_2_mean_spearman')} "
        f"residual_ofi_ex_depth_fwd_ret_2={receipt.get('residual_ofi_ex_depth_fwd_ret_2_mean_spearman')}"
    )
    typer.echo(
        f"residual_depth_ex_ofi_fwd_ret_3={receipt.get('residual_depth_ex_ofi_fwd_ret_3_mean_spearman')} "
        f"residual_ofi_ex_depth_fwd_ret_3={receipt.get('residual_ofi_ex_depth_fwd_ret_3_mean_spearman')}"
    )
    typer.echo(
        f"λ_depth_p10/p50/p90={receipt.get('kyle_lambda_depth_p10')}/"
        f"{receipt.get('kyle_lambda_depth_p50')}/{receipt.get('kyle_lambda_depth_p90')} "
        f"roll_hac=[{receipt.get('kyle_lambda_depth_rolling_hac_lo')},"
        f"{receipt.get('kyle_lambda_depth_rolling_hac_hi')}]"
    )
    typer.echo(f"research_only={receipt.get('research_only')} claim={receipt.get('claim')}")


__all__ = [
    "candle_book",
    "kyle_ofi",
]
