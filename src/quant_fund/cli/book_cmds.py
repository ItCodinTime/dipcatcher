"""Session book, vendor map, book panel, and northset commands.

Split out of the original module. Import the parent path; it re-exports these names.
"""

from __future__ import annotations

from pathlib import Path

import typer

from .app import app
from .support import _cfg, format_data_label


@app.command("session-book")
def session_book_cmd(
    config: Path = typer.Option(Path("configs/research.yaml")),
    out: Path = typer.Option(Path("data/book_panel/session_l2.parquet")),
    daily_out: Path | None = typer.Option(
        None, help="Optional path for daily-aggregated session book features"
    ),
    depth: int = typer.Option(5, help="L2 depth for SYNTHETIC session books"),
    seed: int = typer.Option(7, help="RNG seed"),
    n_session: int | None = typer.Option(None, help="Session candles per day"),
) -> None:
    """Write multi-snapshot session L2 parquet (SYNTHETIC) + optional daily agg.

    One book snapshot per session candle. Research-only plumbing (ADR-021).
    """
    from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
    from quant_fund.microstructure import book_metrics as bm
    from quant_fund.microstructure.synthetic_lob import (
        aggregate_session_book_to_daily,
        ensure_book_panel_shape_columns,
        synthesize_session_l2,
    )
    from quant_fund.northset.identities import (
        book_uncrossed_rate,
        ohlc_identity_rate,
        session_candles_from_daily,
        session_chain_rate,
        session_reconstructs_daily_rate,
        session_volume_conservation_rate,
        validate_session_book_counts,
    )

    cfg = _cfg(config)
    n_assets = int(getattr(getattr(cfg, "data", None), "n_assets", 8) or 8)
    n_days = int(getattr(getattr(cfg, "data", None), "n_days", 120) or 120)
    n_candles = int(n_session or cfg.northset.n_session_candles)
    bars = SyntheticMarketProvider(
        n_assets=max(n_assets, 4), n_days=max(n_days, 40), seed=seed
    ).get_bars()
    session = session_candles_from_daily(bars, n_candles=n_candles, seed=seed)
    session_book = synthesize_session_l2(
        session,
        depth=depth,
        seed=seed,
        base_spread_bps=float(cfg.northset.base_spread_bps),
    )
    # Fail-closed before write: a SYNTHETIC panel missing shape columns is a bug.
    ensure_book_panel_shape_columns(session_book)
    validate_session_book_counts(session_book, n_session_candles=n_candles)
    # Panel validate needs required cols; session keys are extra (ok).
    from quant_fund.microstructure.book_panel import validate_book_panel

    panel = validate_book_panel(
        session_book.drop(
            [c for c in ("parent_event_time", "session_index") if c in session_book.columns]
        )
    )
    # Keep session keys in the written file for path research.
    from quant_fund.utils.atomicio import atomic_write_parquet

    to_write = session_book
    atomic_write_parquet(to_write, out)
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo("SYNTHETIC")
    chain = session_chain_rate(session)
    vol_cons = session_volume_conservation_rate(bars, session)
    reconstruct = session_reconstructs_daily_rate(bars, session)
    book_metrics = session_book.drop(
        [c for c in ("parent_event_time", "session_index") if c in session_book.columns]
    )
    uncrossed = book_uncrossed_rate(book_metrics)
    from quant_fund.northset.benches import enforce_session_l2_identity_floors

    floor = float(getattr(cfg.northset, "session_l2_identity_floor", 0.99))
    ohlc = ohlc_identity_rate(bars)
    session_ohlc = ohlc_identity_rate(session)
    enforce_session_l2_identity_floors(
        ohlc_identity_rate=ohlc,
        session_ohlc_identity_rate=session_ohlc,
        session_reconstructs_daily_rate=reconstruct,
        session_volume_conservation_rate=vol_cons,
        session_chain_rate=chain,
        book_uncrossed_rate=uncrossed,
        floor=floor,
    )
    typer.echo(f"session_book rows={session_book.height} n_session_candles={n_candles} path={out}")
    rate_rows = bm.metric_rows_from_frame(book_metrics)
    typer.echo(
        f"depth_shape_finite_rate={bm.depth_shape_finite_rate(rate_rows)} "
        f"concentration_top_finite_rate={bm.concentration_top_finite_rate(rate_rows)} "
        f"queue_priority_finite_rate={bm.queue_priority_finite_rate(rate_rows)} "
        f"side_notional_finite_rate={bm.side_notional_finite_rate(rate_rows)} "
        f"tob_size_share_finite_rate={bm.tob_size_share_finite_rate(rate_rows)} "
        f"shape_columns_ensured=true"
    )
    typer.echo(
        f"identity ohlc={ohlc:.6g} "
        f"session_ohlc={session_ohlc:.6g} "
        f"reconstruct={reconstruct:.6g} vol_cons={vol_cons:.6g} "
        f"chain={chain:.6g} book_uncrossed={uncrossed:.6g} "
        f"floor={floor} gate=enforced"
    )
    if daily_out is not None:
        daily = aggregate_session_book_to_daily(session_book)
        atomic_write_parquet(daily, daily_out)
        typer.echo(f"session_book_daily rows={daily.height} path={daily_out}")
    del panel  # validated shape for required cols


@app.command("vendor-book-map")
def vendor_book_map_cmd(
    vendor: str = typer.Option("alpaca", help="Preset: alpaca | polygon | generic"),
    columns: str = typer.Option(
        "",
        help="Comma-separated vendor columns (empty → show preset aliases only)",
    ),
    parquet: Path | None = typer.Option(
        None, help="Optional vendor-shaped parquet to remap → panel"
    ),
    out: Path | None = typer.Option(None, help="Where to write remapped Northset panel parquet"),
) -> None:
    """Offline dry-run of vendor quote columns → Northset book panel.

    No network. Proves ADR-021 interchange before a live vendor feed exists.
    """
    import json

    from quant_fund.microstructure.vendor_book_map import (
        VENDOR_PRESETS,
        dry_run_vendor_book_map,
        remap_vendor_quotes_to_panel,
    )

    key = vendor.strip().lower()
    if key not in VENDOR_PRESETS:
        raise typer.BadParameter(f"vendor must be one of {sorted(VENDOR_PRESETS)}")
    if columns.strip():
        cols = [c.strip() for c in columns.split(",") if c.strip()]
    elif parquet is not None:
        import polars as pl

        cols = list(pl.read_parquet(parquet).columns)
    else:
        # Show alias table only
        typer.echo(f"vendor={key}")
        for target, aliases in VENDOR_PRESETS[key].items():
            typer.echo(f"  {target} ← {list(aliases)}")
        return
    report = dry_run_vendor_book_map(cols, vendor=key)
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo(json.dumps(report.as_dict(), indent=2, default=str))
    if parquet is not None:
        import polars as pl

        from quant_fund.microstructure.book_panel import write_book_panel

        raw = pl.read_parquet(parquet)
        panel = remap_vendor_quotes_to_panel(raw, vendor=key)
        dest = out or Path("data/book_panel/vendor_remapped.parquet")
        path = write_book_panel(panel, dest)
        typer.echo(f"remapped rows={panel.height} path={path}")


@app.command("book-panel")
def book_panel_cmd(
    config: Path = typer.Option(Path("configs/research.yaml")),
    out: Path = typer.Option(Path("data/book_panel/synthetic_l2.parquet")),
    depth: int = typer.Option(5, help="L2 depth for SYNTHETIC book"),
    seed: int = typer.Option(7, help="RNG seed for SYNTHETIC L2"),
) -> None:
    """Write a vendor-shaped L2 book panel parquet (SYNTHETIC by default).

    Same columns vendor books must emit (ADR-021). Research-only plumbing.
    """
    from quant_fund.data.adapters.order_book import SyntheticOrderBookProvider
    from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
    from quant_fund.microstructure import book_metrics as bm
    from quant_fund.microstructure.book_panel import write_book_panel
    from quant_fund.microstructure.synthetic_lob import ensure_book_panel_shape_columns

    cfg = _cfg(config)
    n_assets = int(getattr(getattr(cfg, "data", None), "n_assets", 8) or 8)
    n_days = int(getattr(getattr(cfg, "data", None), "n_days", 120) or 120)
    bars = SyntheticMarketProvider(
        n_assets=max(n_assets, 4), n_days=max(n_days, 40), seed=seed
    ).get_bars()
    panel = SyntheticOrderBookProvider(
        depth=depth, seed=seed, base_spread_bps=float(cfg.northset.base_spread_bps)
    ).get_book_panel(bars)
    # Fail-closed before write: a SYNTHETIC panel missing shape columns is a bug.
    ensure_book_panel_shape_columns(panel)
    path = write_book_panel(panel, out)
    rows = bm.metric_rows_from_frame(panel)
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo("SYNTHETIC")
    typer.echo(f"book_panel rows={panel.height} cols={len(panel.columns)} path={path}")
    typer.echo(
        f"depth_shape_finite_rate={bm.depth_shape_finite_rate(rows)} "
        f"concentration_top_finite_rate={bm.concentration_top_finite_rate(rows)} "
        f"queue_priority_finite_rate={bm.queue_priority_finite_rate(rows)} "
        f"side_notional_finite_rate={bm.side_notional_finite_rate(rows)} "
        f"tob_size_share_finite_rate={bm.tob_size_share_finite_rate(rows)} "
        f"shape_columns_ensured=true"
    )


@app.command()
def northset(
    config: Path = typer.Option(Path("configs/research.yaml")),
    book: Path | None = typer.Option(
        None,
        help="L2 parquet: Northset panel OR raw vendor quotes (use --vendor to remap)",
    ),
    vendor: str | None = typer.Option(
        None,
        help="If set (alpaca|polygon|generic), remap --book through vendor-book-map first",
    ),
    depth_shape_floor: float | None = typer.Option(
        None, help="Fail-closed depth-shape finite-rate floor override (SYNTHETIC deep books)."
    ),
    concentration_floor: float | None = typer.Option(
        None, help="Fail-closed concentration-top finite-rate floor override."
    ),
) -> None:
    """Run Northset — order-book and candlestick research inside Dipcatcher."""
    import polars as pl

    from quant_fund.northset.benches import bench_northset
    from quant_fund.pipeline.dataset import ensure_silver

    cfg = _cfg(config)
    if depth_shape_floor is not None:
        cfg.northset.depth_shape_finite_floor = float(depth_shape_floor)
    if concentration_floor is not None:
        cfg.northset.concentration_top_finite_floor = float(concentration_floor)
    if book is not None:
        panel_path = book
        if vendor is not None:
            from quant_fund.microstructure.book_panel import write_book_panel
            from quant_fund.microstructure.vendor_book_map import remap_vendor_quotes_to_panel

            raw = pl.read_parquet(book)
            panel = remap_vendor_quotes_to_panel(raw, vendor=vendor.strip().lower())
            panel_path = book.with_name(book.stem + f"_remapped_{vendor.strip().lower()}.parquet")
            write_book_panel(panel, panel_path)
            typer.echo(f"remapped vendor={vendor} → {panel_path}")
        cfg.northset.book_panel_path = str(panel_path)
    bars = ensure_silver(cfg)
    if "security_id" in bars.columns:
        bars = bars.filter(pl.col("security_id") != cfg.data.benchmark_id)
    blob = bench_northset(bars, cfg)
    typer.echo(
        format_data_label(
            synthetic=cfg.data.source == "synthetic",
            data_source=cfg.data.source,
        )
    )
    if cfg.data.source == "synthetic":
        typer.echo("SYNTHETIC")
    typer.echo("Northset — Dipcatcher order-book and candlestick research")
    typer.echo(
        f"join_coverage={blob.get('join_coverage')} "
        f"mean_book_age_s={blob.get('mean_book_age_seconds')} "
        f"mean_book_age_seconds={blob.get('mean_book_age_seconds')} "
        f"max_book_age_s={blob.get('max_book_age_seconds')} "
        f"book_source={blob.get('book_source')} "
        f"include_kyle_ofi={blob.get('include_kyle_ofi')}"
    )
    # Shape floor tokens only echo when set; rates always echo (NaN on thin/external TOB).
    typer.echo(
        f"depth_shape_finite_rate={blob.get('depth_shape_finite_rate')} "
        f"concentration_top_finite_rate={blob.get('concentration_top_finite_rate')} "
        f"queue_priority_finite_rate={blob.get('queue_priority_finite_rate')} "
        f"side_notional_finite_rate={blob.get('side_notional_finite_rate')} "
        f"tob_size_share_finite_rate={blob.get('tob_size_share_finite_rate')} "
        f"mean_tob_size_share={blob.get('mean_tob_size_share')} "
        f"mean_tob_notional_share={blob.get('mean_tob_notional_share')} "
        f"mean_notional_imbalance={blob.get('mean_notional_imbalance')} "
        f"mean_bid_size_concentration_top={blob.get('mean_bid_size_concentration_top')} "
        f"mean_ask_size_concentration_top={blob.get('mean_ask_size_concentration_top')} "
        f"mean_bid_depth={blob.get('mean_bid_depth')} "
        f"mean_ask_depth={blob.get('mean_ask_depth')} "
        f"mean_side_notional_proxy_bid={blob.get('mean_side_notional_proxy_bid')} "
        f"mean_side_notional_proxy_ask={blob.get('mean_side_notional_proxy_ask')} "
        f"mean_top_of_book_notional_proxy={blob.get('mean_top_of_book_notional_proxy')} "
        f"mean_spread_over_mid={blob.get('mean_spread_over_mid')} "
        f"mean_bid_log_price_slope={blob.get('mean_bid_log_price_slope')} "
        f"mean_ask_log_price_slope={blob.get('mean_ask_log_price_slope')} "
        f"mean_bid_mean_log_tick_spacing={blob.get('mean_bid_mean_log_tick_spacing')} "
        f"mean_ask_mean_log_tick_spacing={blob.get('mean_ask_mean_log_tick_spacing')} "
        f"mean_top_bid_size={blob.get('mean_top_bid_size')} "
        f"mean_top_ask_size={blob.get('mean_top_ask_size')} "
        f"mean_n_bid_levels={blob.get('mean_n_bid_levels')} "
        f"mean_n_ask_levels={blob.get('mean_n_ask_levels')} "
        f"mean_depth_imbalance={blob.get('mean_depth_imbalance')} "
        f"mean_imbalance_top={blob.get('mean_imbalance_top')} "
        f"mean_depth_imbalance_abs={blob.get('mean_depth_imbalance_abs')} "
        f"mean_queue_priority_proxy={blob.get('mean_queue_priority_proxy')} "
        f"mean_ask_queue_priority_proxy={blob.get('mean_ask_queue_priority_proxy')} "
        f"shape_columns_ensured={blob.get('shape_columns_ensured')} "
        f"metrics_required_finite_ok={blob.get('metrics_required_finite_ok')}"
    )
    if depth_shape_floor is not None:
        typer.echo(f"depth_shape_finite_floor={depth_shape_floor}")
    if concentration_floor is not None:
        typer.echo(f"concentration_top_finite_floor={concentration_floor}")
    # Spread means: book quoted/effective/half aliases + the distinct candle
    # close–mid diagnostic (never equate close_mid_abs_rel with quoted spread).
    typer.echo(
        f"mean_quoted_spread={blob.get('mean_quoted_spread')} "
        f"mean_effective_spread={blob.get('mean_effective_spread')} "
        f"mean_half_spread={blob.get('mean_half_spread')} "
        f"mean_half_spread_bps={blob.get('mean_half_spread_bps')} "
        f"amihud_mean={blob.get('amihud_mean')} "
        f"mean_spread_bps={blob.get('mean_spread_bps')} "
        f"mean_close_mid_abs_rel={blob.get('mean_close_mid_abs_rel')} "
        f"mean_microprice_weight_balance={blob.get('mean_microprice_weight_balance')} "
        f"mean_microprice_minus_mid={blob.get('mean_microprice_minus_mid')} "
        f"mean_microprice_minus_mid_bps={blob.get('mean_microprice_minus_mid_bps')} "
    )
    # mean_session_imbalance_mean: session-L2 path mean ≠ daily imbalance_top
    typer.echo(
        f"mean_session_imbalance_mean={blob.get('mean_session_imbalance_mean')} "
        f"mean_session_imbalance_std={blob.get('mean_session_imbalance_std')} "
        f"mean_session_close_imbalance={blob.get('mean_session_close_imbalance')} "
        f"mean_session_close_micro_bps={blob.get('mean_session_close_micro_bps')} "
        f"mean_session_close_mid={blob.get('mean_session_close_mid')} "
        f"mean_session_close_bid_depth={blob.get('mean_session_close_bid_depth')} "
        f"mean_session_close_ask_depth={blob.get('mean_session_close_ask_depth')} "
        f"mean_session_spread_bps_mean={blob.get('mean_session_spread_bps_mean')} "
        f"mean_session_close_spread_bps={blob.get('mean_session_close_spread_bps')} "
        f"session_ofi_sum_mean={blob.get('session_ofi_sum_mean')} "
        f"mean_session_ofi_abs_sum={blob.get('mean_session_ofi_abs_sum')} "
        f"session_book_vpin_mean={blob.get('session_book_vpin_mean')} "
        f"mean_session_book_snaps={blob.get('mean_session_book_snaps')}"
    )
    # Residual honesty companions (CLI polish — rates/IC/sweep/overnight).
    typer.echo(
        f"gap_finite_rate={blob.get('gap_finite_rate')} "
        f"book_hypothesis_eligible={blob.get('book_hypothesis_eligible')} "
        f"session_book_hypothesis_eligible={blob.get('session_book_hypothesis_eligible')} "
        f"session_l2_identity_gate={blob.get('session_l2_identity_gate')} "
        f"queue_imbalance_mean={blob.get('queue_imbalance_mean')} "
        f"mean_bid_log_size_slope={blob.get('mean_bid_log_size_slope')} "
        f"mean_ask_log_size_slope={blob.get('mean_ask_log_size_slope')} "
        f"mean_true_range={blob.get('mean_true_range')} "
        f"mid_lag1_corr={blob.get('mid_lag1_corr')} "
        f"ofi_lag1_corr={blob.get('ofi_lag1_corr')} "
        f"n_sweep_high={blob.get('n_sweep_high')} "
        f"n_sweep_low={blob.get('n_sweep_low')} "
        f"overnight_share={blob.get('overnight_share')} "
        f"session_mean_rv={blob.get('session_mean_rv')} "
        f"session_mean_bv={blob.get('session_mean_bv')} "
        f"semi_up={blob.get('semi_up')} "
        f"semi_down={blob.get('semi_down')} "
        f"yang_zhang_variance={blob.get('yang_zhang_variance')} "
        f"yang_zhang_qlike_vs_cc={blob.get('yang_zhang_qlike_vs_cc')} "
        f"parkinson_qlike_vs_cc={blob.get('parkinson_qlike_vs_cc')} "
        f"garman_klass_qlike_vs_cc={blob.get('garman_klass_qlike_vs_cc')} "
        f"rogers_satchell_qlike_vs_cc={blob.get('rogers_satchell_qlike_vs_cc')} "
        f"overnight_plus_oc_qlike_vs_cc={blob.get('overnight_plus_oc_qlike_vs_cc')} "
        f"session_rv_qlike_vs_cc={blob.get('session_rv_qlike_vs_cc')} "
        f"roll_spread={blob.get('roll_spread')} "
        f"abdi_ranaldo_spread={blob.get('abdi_ranaldo_spread')} "
        f"corwin_schultz_spread={blob.get('corwin_schultz_spread')} "
        f"sweep_reject_fold_positive_fraction={blob.get('sweep_reject_fold_positive_fraction')} "
        f"sweep_follow_fold_positive_fraction={blob.get('sweep_follow_fold_positive_fraction')} "
        f"sweep_reject_cost_adjusted_mean_bps={blob.get('sweep_reject_cost_adjusted_mean_bps')} "
        f"mean_fwd_ret_after_high_reclaim={blob.get('mean_fwd_ret_after_high_reclaim')} "
        f"mean_fwd_ret_after_low_reclaim={blob.get('mean_fwd_ret_after_low_reclaim')} "
        f"mean_fwd_ret_after_high_follow={blob.get('mean_fwd_ret_after_high_follow')} "
        f"mean_fwd_ret_after_low_follow={blob.get('mean_fwd_ret_after_low_follow')} "
        f"sweep_reject_event_mean_bps={blob.get('sweep_reject_event_mean_bps')} "
        f"sweep_follow_event_mean_bps={blob.get('sweep_follow_event_mean_bps')} "
        f"sweep_reject_control_diff_mean_bps={blob.get('sweep_reject_control_diff_mean_bps')} "
        f"sweep_follow_control_diff_mean_bps={blob.get('sweep_follow_control_diff_mean_bps')} "
        f"sweep_reject_liq_control_diff_mean_bps={blob.get('sweep_reject_liq_control_diff_mean_bps')} "
        f"sweep_follow_liq_control_diff_mean_bps={blob.get('sweep_follow_liq_control_diff_mean_bps')} "
        f"sweep_follow_oot_holdout_mean_bps={blob.get('sweep_follow_oot_holdout_mean_bps')} "
        f"sweep_median_event_adv_participation={blob.get('sweep_median_event_adv_participation')} "
        f"sweep_n_counted_trials={blob.get('sweep_n_counted_trials')} "
        f"sweep_primary_test_id={blob.get('sweep_primary_test_id')} "
        f"n_session_book_rows={blob.get('n_session_book_rows')}"
    )
    # Identity / reconstruct rates (stamped on receipt; never equate ohlc vs session_ohlc).
    typer.echo(
        f"ohlc_identity_rate={blob.get('ohlc_identity_rate')} "
        f"session_ohlc_identity_rate={blob.get('session_ohlc_identity_rate')} "
        f"book_uncrossed_rate={blob.get('book_uncrossed_rate')} "
        f"session_chain_rate={blob.get('session_chain_rate')} "
        f"session_reconstructs_daily_rate={blob.get('session_reconstructs_daily_rate')} "
        f"session_volume_conservation_rate={blob.get('session_volume_conservation_rate')} "
        f"structure_finite_rate={blob.get('structure_finite_rate')}"
    )
    typer.echo(blob)


__all__ = [
    "book_panel_cmd",
    "northset",
    "session_book_cmd",
    "vendor_book_map_cmd",
]
