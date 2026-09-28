"""Research command.

Split out of the original module. Import the parent path; it re-exports these names.
"""

from __future__ import annotations

from pathlib import Path

import typer

from .app import app
from .support import _cfg, format_data_label, format_fdr_families


@app.command()
def research(config: Path = typer.Option(Path("configs/research.yaml"))) -> None:
    """Run Dipcatcher's proprietary research benches and write a labeled notebook."""
    from quant_fund.research.agent import format_p_value, run_research

    cfg = _cfg(config)
    nb = run_research(cfg)
    # Always print DATA_LABEL + FDR families (regression-tested).
    typer.echo(format_data_label(synthetic=bool(nb.synthetic), data_source=str(nb.data_source)))
    if nb.synthetic:
        typer.echo("SYNTHETIC")
    typer.echo(f"{nb.product} — {nb.firm}'s proprietary research lab {nb.version}")
    typer.echo(format_fdr_families(list(nb.hypotheses)))
    typer.echo(nb.disclaimer)
    for h in nb.hypotheses:
        typer.echo(f"{h.id}: {h.decision} (p={format_p_value(h.p_value)})")
    for r in nb.rankers:
        if str(r.get("name", "")).startswith("_"):
            dm_all = r.get("pairwise_dm_all") or []
            for d in dm_all:
                typer.echo(
                    f"DM {d.get('a')} vs {d.get('b')}: "
                    f"stat={d.get('statistic')} p={d.get('p_value')} preferred={d.get('preferred')}"
                )
            continue
        typer.echo(
            f"{r['name']}: IC={r['mean_ic']:.4f} RankIC={r.get('mean_rank_ic') or 0:.4f} "
            f"t={r['t_ic']:.2f} mono={r.get('decile_monotonicity') or 0:.3f}"
        )
    vol = nb.families.get("volatility", {})
    if vol:
        typer.echo(
            f"volatility QLIKE ewma={vol.get('qlike_ewma')} rolling={vol.get('qlike_rolling')}"
        )
    rl = nb.families.get("reinforcement", {})
    if rl:
        typer.echo(
            f"linucb reward={rl.get('mean_policy_reward')} "
            f"vs_uniform={rl.get('mean_advantage_vs_uniform')} "
            f"regret={rl.get('mean_regret_vs_oracle')}"
        )
    conf = nb.families.get("conformal", {})
    if conf:
        aci = conf.get("aci", {})
        raw = conf.get("gaussian_raw", {})
        wrap = conf.get("wrappee", "scaled_gaussian")
        typer.echo(
            f"conformal wrappee={wrap} "
            f"raw_cov={raw.get('coverage')} "
            f"cqr_raw_cov={conf.get('cqr_raw', {}).get('coverage')} "
            f"scaled_cov={conf.get('scaled', conf.get('scaled_gaussian', {})).get('coverage')} "
            f"cqr_cov={conf.get('cqr', {}).get('coverage')} "
            f"aci_cov={aci.get('coverage')} "
            f"aci_width={aci.get('mean_width')} "
            f"mondrian_cov={conf.get('mondrian_aci', {}).get('coverage')} "
            f"high_x={conf.get('mondrian_aci', {}).get('high_x_coverage')} "
            f"worst_x={conf.get('mondrian_aci', {}).get('worst_x_coverage')}"
        )
    ev = nb.families.get("evalues", {})
    if ev:
        typer.echo(
            f"evalues cov={ev.get('coverage')} e_final={ev.get('e_final')} "
            f"ever_cross={ev.get('ever_cross')}"
        )
    jp = nb.families.get("jackknife_plus", {})
    if jp:
        typer.echo(
            f"jackknife_plus cov={jp.get('coverage')} width={jp.get('mean_width')} "
            f"floor={jp.get('coverage_floor')}"
        )
    crc = nb.families.get("crc", {})
    if crc:
        typer.echo(
            f"crc wrappee={crc.get('wrappee')} risk={crc.get('risk')} "
            f"crc_stat={crc.get('crc_stat')} lambda={crc.get('lambda_hat')} "
            f"high_vol_bound={crc.get('high_vol_mean_bound')} "
            f"low_vol_bound={crc.get('low_vol_mean_bound')}"
        )
    wcqr = nb.families.get("weighted_conformal", {})
    if wcqr:
        typer.echo(
            f"weighted_conformal cov={wcqr.get('coverage')} "
            f"width={wcqr.get('mean_width')} "
            f"unweighted_cov={wcqr.get('unweighted_coverage')}"
        )
    caps = nb.families.get("interval_risk", {})
    if caps:
        typer.echo(
            f"interval_risk mean_cap={caps.get('mean_cap')} "
            f"frac_binding={caps.get('frac_binding')} "
            f"mean_width={caps.get('mean_width')}"
        )
    qb = nb.families.get("quantile_bandit", {})
    if qb:
        typer.echo(
            f"quantile_bandit reward={qb.get('mean_policy_reward')} "
            f"vs_uniform={qb.get('mean_advantage_vs_uniform')} "
            f"regret={qb.get('mean_regret_vs_oracle')}"
        )
    ns = nb.families.get("northset", {})
    if ns:
        typer.echo(
            f"northset ohlc_ok={ns.get('ohlc_identity_rate')} "
            f"book_ok={ns.get('book_uncrossed_rate')} "
            f"session_ok={ns.get('session_reconstructs_daily_rate')} "
            f"imb_ic={ns.get('imbalance_top_mean_ic')} "
            f"ofi_ic={ns.get('ofi_mean_ic')} "
            f"kyle_lambda={ns.get('kyle_lambda')} "
            f"park_qlike={ns.get('parkinson_qlike_vs_cc')} "
            f"gk_qlike={ns.get('garman_klass_qlike_vs_cc')} "
            f"cs_spread={ns.get('corwin_schultz_spread')} "
            f"chain_ok={ns.get('session_chain_rate')} "
            f"jump={ns.get('session_mean_jump_ratio')} "
            f"vpin={ns.get('vpin_mean')} "
            f"sweep_rate={ns.get('sweep_any_rate')} "
            f"sweep_rej_ic={ns.get('sweep_reject_signed_mean_ic')} "
            f"sweep_rej_event={ns.get('sweep_reject_event_mean_bps')}bps "
            f"sweep_follow_event={ns.get('sweep_follow_event_mean_bps')}bps "
            f"sweep_follow_costed={ns.get('sweep_follow_cost_adjusted_mean_bps')}bps "
            f"mean_session_spread_bps_mean={ns.get('mean_session_spread_bps_mean')} "
            f"mean_session_close_spread_bps={ns.get('mean_session_close_spread_bps')} "
            f"mean_session_close_imbalance={ns.get('mean_session_close_imbalance')} "
            f"mean_session_close_micro_bps={ns.get('mean_session_close_micro_bps')} "
            f"mean_session_close_mid={ns.get('mean_session_close_mid')} "
            f"mean_session_close_bid_depth={ns.get('mean_session_close_bid_depth')} "
            f"mean_session_close_ask_depth={ns.get('mean_session_close_ask_depth')} "
            f"mean_session_imbalance_std={ns.get('mean_session_imbalance_std')} "
            f"mean_session_imbalance_mean={ns.get('mean_session_imbalance_mean')} "
            f"session_ofi_sum_mean={ns.get('session_ofi_sum_mean')} "
            f"mean_session_ofi_abs_sum={ns.get('mean_session_ofi_abs_sum')} "
            f"session_book_vpin_mean={ns.get('session_book_vpin_mean')} "
            f"mean_session_book_snaps={ns.get('mean_session_book_snaps')}"
        )
    typer.echo(f"json={nb.artifacts.get('json')}")
    typer.echo(f"markdown={nb.artifacts.get('markdown')}")


@app.command("execution-sensitivity")
def execution_sensitivity_cmd(
    config: Path = typer.Option(Path("configs/backtest.yaml")),
    max_dates: int | None = typer.Option(
        None,
        help=(
            "Use only the first N decision dates present on both the feature "
            "panel and the causal gold panel."
        ),
    ),
    signal_latencies: str = typer.Option(
        "0,1", help="Comma-separated signal-to-order delays in bars."
    ),
    exchange_latencies: str = typer.Option(
        "0,1", help="Comma-separated order-to-exchange delays in bars."
    ),
    impact: str = typer.Option("1,2", help="Comma-separated square-root impact multipliers."),
    seed: int = typer.Option(0, help="Deterministic simulator seed."),
    initial_nav: float = typer.Option(1_000_000.0),
) -> None:
    """Latency/impact grid for one strategy. Execution diagnostic, not a live P&L claim."""
    import polars as pl

    from quant_fund.backtest.event_sim import execution_sensitivity, format_sensitivity_table
    from quant_fund.features.engine import build_features
    from quant_fund.pipeline.dataset import ensure_silver
    from quant_fund.pipeline.dataset import panel as decision_panel
    from quant_fund.pipeline.forecast import build_causal_weight_panel

    def _ints(raw: str) -> tuple[int, ...]:
        parts = tuple(int(piece.strip()) for piece in raw.split(",") if piece.strip())
        if not parts:
            raise typer.BadParameter("expected a comma-separated list of bar counts")
        return parts

    def _floats(raw: str) -> tuple[float, ...]:
        parts = tuple(float(piece.strip()) for piece in raw.split(",") if piece.strip())
        if not parts:
            raise typer.BadParameter("expected a comma-separated list of multipliers")
        return parts

    cfg = _cfg(config)
    bars = ensure_silver(cfg)
    feat = build_features(bars, cfg)
    dates = feat["event_time"].unique().sort().to_list()
    # Gold drops warmup bars (history / label horizon). optimize_asof refuses
    # a decision date with no panel row, so the grid uses the overlap only.
    on_panel = set(decision_panel(cfg)["event_time"].unique().to_list())
    dates = [day for day in dates if day in on_panel]
    if len(dates) < 2:
        raise typer.BadParameter(
            "need at least 2 decision dates on both the feature panel and the causal gold panel"
        )
    if max_dates is not None:
        if max_dates < 2:
            raise typer.BadParameter("--max-dates must be at least 2")
        dates = dates[:max_dates]
    feat = feat.filter(pl.col("event_time").is_in(dates))
    weights = build_causal_weight_panel(cfg, dates)
    report = execution_sensitivity(
        feat,
        weights,
        cfg,
        signal_latencies=_ints(signal_latencies),
        exchange_latencies=_ints(exchange_latencies),
        impact_multipliers=_floats(impact),
        initial_nav=initial_nav,
        seed=seed,
    )
    rows = report.get("rows")
    source = cfg.data.source
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        source = str(rows[0].get("source", source))
    synthetic = source.upper() == "SYNTHETIC" or "synthetic" in source.lower()
    typer.echo(format_data_label(synthetic=synthetic, data_source=source))
    if synthetic:
        typer.echo("SYNTHETIC")
    typer.echo(format_sensitivity_table(report))


@app.command("verify-identities")
def verify_identities(
    out: Path = typer.Option(
        Path("data/metadata/research/identity_sweep.json"), "--out", help="Receipt JSON path."
    ),
    trials: int = typer.Option(8, "--trials", help="Seeded SYNTHETIC draws per identity."),
    seed: int = typer.Option(7, "--seed", help="Base seed for the synthetic generators."),
) -> None:
    """Prove catalog/northset microstructure identities on SYNTHETIC draws.

    Prints the identity/residual table and writes an immutable receipt.
    Exits non-zero when any identity's max-abs residual exceeds its
    tolerance — fail-closed, CI gate candidate.
    """
    from quant_fund.research.identity_sweep import (
        format_identity_table,
        run_identity_sweep,
        write_identity_receipt,
    )

    receipt = run_identity_sweep(n_trials=int(trials), seed=int(seed))
    write_identity_receipt(out, receipt)
    typer.echo("SYNTHETIC")
    typer.echo(format_identity_table(receipt))
    typer.echo(f"receipt={out}")
    raise typer.Exit(code=0 if receipt["all_passed"] else 1)


@app.command()
def fleet(
    config: Path = typer.Option(Path("configs/research.yaml")),
    models: str | None = typer.Option(
        None, help="Comma-separated head names (default: full fleet registry)."
    ),
    shards: str | None = typer.Option(
        None, help="Comma-separated shard names (default: all synthetic shards)."
    ),
    n_train: int = typer.Option(512, help="Leading fit rows per shard."),
    n_eval: int = typer.Option(256, help="Trailing scored rows per shard."),
    seed: int | None = typer.Option(
        None, help="Base seed (default: train.random_seed from config)."
    ),
    out_dir: Path = typer.Option(Path("receipts"), help="Receipt output directory."),
    receipt_version: int = typer.Option(
        1,
        "--receipt-version",
        help="Receipt schema version: 1 = fleet_eval.v1 (default), 2 = unified receipt.v2 envelope.",
    ),
) -> None:
    """Run the SYNTHETIC distribution-challenger fleet and write a receipt.

    Proper scores only (pinball/CRPS/PIT-KS/coverage) on labeled synthetic
    shards — correctness evidence, never market or live-P&L claims.
    """
    from quant_fund.research.fleet_eval import (
        fleet_head_factories,
        resolve_shard_generators,
        run_distribution_fleet,
        write_fleet_receipt,
    )

    cfg = _cfg(config)
    base_seed = cfg.train.random_seed if seed is None else seed
    try:
        factories = fleet_head_factories(
            cfg.quantiles.levels,
            base_seed,
            None if models is None else models.split(","),
        )
        resolved_shards = resolve_shard_generators(None if shards is None else shards.split(","))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    frame, receipt = run_distribution_fleet(
        factories,
        resolved_shards,
        n_train=n_train,
        n_eval=n_eval,
        seed=base_seed,
        taus=cfg.quantiles.levels,
    )
    if receipt_version not in (1, 2):
        raise typer.BadParameter("--receipt-version must be 1 or 2")
    path = write_fleet_receipt(receipt, out_dir, receipt_version=receipt_version)
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo(frame)
    typer.echo(f"receipt={path}")


@app.command("verify-receipt")
def verify_receipt_cmd(
    path: Path = typer.Argument(..., help="Receipt JSON file to verify."),
) -> None:
    """Verify a sealed receipt: structure plus hash consistency.

    ``receipt.v2`` envelopes are validated against the pydantic schema and
    their sealed digest, environment fingerprint, code-map digest, and (for
    known kinds) dataset/params digests are re-derived. Older v1 receipts get
    a ``receipt_sha256`` seal check (canonical or strict JSON convention);
    ``fleet_eval.v1`` payloads get their writer's contract too. Exits non-zero
    on any violation — fail-closed.
    """
    import json

    from quant_fund.research.receipt_v2 import verify_receipt_file

    result = verify_receipt_file(path)
    typer.echo(json.dumps(result, indent=2))
    raise typer.Exit(code=0 if result["valid"] else 1)


@app.command("vol-bench")
def vol_bench(
    config: Path = typer.Option(Path("configs/research.yaml")),
    models: str | None = typer.Option(
        None,
        help="Comma-separated vol model names (default: rv_roll, rv_ewma, har, "
        "realized_garch, dip_garch_t).",
    ),
    shards: str | None = typer.Option(
        None, help="Comma-separated shard names (default: all SYNTHETIC vol shards)."
    ),
    horizons: str = typer.Option(
        "1,5", "--horizons", help="Comma-separated forecast horizons in bars."
    ),
    min_history: int = typer.Option(300, help="Leading fit bars per origin."),
    n_origins: int = typer.Option(24, help="Scored origins per shard/horizon (>=10)."),
    stride: int | None = typer.Option(None, help="Origin spacing in bars (default: max horizon)."),
    n_bars: int | None = typer.Option(
        None, help="Shard length (default: minimal for the origin schedule)."
    ),
    seed: int | None = typer.Option(
        None, help="Base seed (default: train.random_seed from config)."
    ),
    out_dir: Path = typer.Option(Path("receipts"), help="Receipt output directory."),
) -> None:
    """Run the SYNTHETIC volatility bench and write a sealed receipt.

    QLIKE + MSE on next-bar/h-step cumulative realized variance for HAR,
    realized-GARCH, dip_garch_t and RV baselines over seeded synthetic vol
    shards (GARCH clustering, rough vol, structural breaks). Proper scores
    only — correctness evidence, never market or live-P&L claims.
    """
    from quant_fund.research.vol_bench import (
        resolve_vol_models,
        resolve_vol_shard_generators,
        run_vol_bench,
        write_vol_bench_receipt,
    )

    cfg = _cfg(config)
    base_seed = cfg.train.random_seed if seed is None else seed
    try:
        horizon_set = tuple(int(h.strip()) for h in horizons.split(",") if h.strip())
        forecasters = resolve_vol_models(None if models is None else models.split(","))
        resolved_shards = resolve_vol_shard_generators(
            None if shards is None else shards.split(",")
        )
        frame, receipt = run_vol_bench(
            forecasters,
            resolved_shards,
            horizons=horizon_set,
            min_history=int(min_history),
            n_origins=int(n_origins),
            stride=None if stride is None else int(stride),
            n_bars=None if n_bars is None else int(n_bars),
            seed=int(base_seed),
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    path = write_vol_bench_receipt(receipt, out_dir)
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo(frame)
    typer.echo(f"receipt={path}")


@app.command()
def rankic(
    panels: str | None = typer.Option(
        None, help="Comma-separated panel names (default: all synthetic panels)."
    ),
    challengers: str | None = typer.Option(
        None, help="Comma-separated challenger names (default: all transforms)."
    ),
    n_assets: int = typer.Option(32, help="Assets per date."),
    n_dates: int = typer.Option(96, help="Panel length in dates."),
    horizons: str = typer.Option("1,5,20", help="Comma-separated forward horizons."),
    seed: int = typer.Option(11, help="Base seed for the synthetic panels."),
    out_dir: Path = typer.Option(Path("receipts"), help="Receipt output directory."),
) -> None:
    """Cross-sectional rank-IC bench on SYNTHETIC planted-signal panels (P3.4).

    Per-date Spearman rank-IC between each challenger transform and h-step
    forward returns, summarized with a Newey-West mean-IC t-stat. Proper-score
    framing only — never a P&L or live-trading claim.
    """
    from quant_fund.research.cross_sectional import (
        format_rankic_table,
        resolve_panels,
        run_cross_sectional_bench,
        write_rankic_receipt,
    )

    def _names(raw: str | None) -> list[str] | None:
        if raw is None:
            return None
        return [piece.strip() for piece in raw.split(",") if piece.strip()]

    try:
        resolved = resolve_panels(_names(panels))
        horizon_tuple = tuple(int(piece.strip()) for piece in horizons.split(",") if piece.strip())
        if not horizon_tuple:
            raise typer.BadParameter("--horizons must be nonempty")
        frame, receipt = run_cross_sectional_bench(
            resolved,
            challengers=_names(challengers),
            horizons=horizon_tuple,
            n_assets=n_assets,
            n_dates=n_dates,
            seed=seed,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    path = write_rankic_receipt(receipt, out_dir)
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo("SYNTHETIC")
    typer.echo(format_rankic_table(frame))
    typer.echo(f"receipt={path}")


@app.command()
def capacity(
    books: str | None = typer.Option(
        None, help="Comma-separated synthetic book names (default: all)."
    ),
    n_dates: int = typer.Option(126, help="Dates per synthetic book."),
    n_names: int = typer.Option(32, help="Names per synthetic book."),
    seed: int = typer.Option(11, help="Base seed."),
    aum_grid: str = typer.Option(
        "1e6,1e7,5e7,1e8,5e8,1e9", help="Comma-separated AUM levels in dollars."
    ),
    participation_cap: float = typer.Option(
        0.10, help="Max share of dollar ADV a rebalance may consume per name-day."
    ),
    impact_coeff: float = typer.Option(0.1, help="Square-root impact coefficient."),
    dev: bool = typer.Option(False, "--dev", help="Acknowledge dev-only use; required to run."),
    out_dir: Path = typer.Option(Path("receipts"), help="Receipt output directory."),
) -> None:
    """P5.6 participation-capacity bench on SYNTHETIC books (dev-only).

    Feasibility fractions, days-to-trade, and sqrt-impact cost in bps per
    book x AUM cell. Sealed `capacity_overlay_eval` receipt. Size bounds
    only — never a market or live-P&L claim.
    """
    if not dev:
        raise typer.BadParameter(
            "capacity is dev-only evidence tooling; pass --dev to acknowledge."
        )
    from quant_fund.research.capacity_overlay import (
        format_capacity_table,
        resolve_books,
        run_capacity_bench,
        write_capacity_receipt,
    )

    try:
        grid = tuple(float(x) for x in aum_grid.split(","))
        resolved = resolve_books(None if books is None else books.split(","))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    book_objs = [gen(n_dates, n_names, seed + i) for i, gen in enumerate(resolved.values())]
    frame, receipt = run_capacity_bench(
        book_objs,
        seed=seed,
        aum_grid=grid,
        participation_cap=participation_cap,
        impact_coeff=impact_coeff,
    )
    path = write_capacity_receipt(receipt, out_dir)
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo(format_capacity_table(frame))
    typer.echo(f"receipt={path}")


@app.command("cost-calibration")
def cost_calibration(
    half_spread_bps: float = typer.Option(1.0, help="Flat half-spread floor in bps."),
    lookback: int = typer.Option(20, help="Trailing OHLC window for estimators."),
    n_dates: int = typer.Option(40, help="Dates in the SYNTHETIC panel."),
    n_names: int = typer.Option(4, help="Names in the SYNTHETIC panel."),
    seed: int = typer.Option(7, help="Panel seed."),
    planted_rel_spread: float = typer.Option(
        0.002, help="Planted high-low relative full spread for the SYNTHETIC book."
    ),
    dev: bool = typer.Option(False, "--dev", help="Acknowledge dev-only use; required to run."),
    out_dir: Path = typer.Option(Path("receipts"), help="Receipt output directory."),
    report_path: Path = typer.Option(
        Path("reports/cost_calibration_flat_vs_ohlc.md"),
        help="Markdown report path (flat vs calibrated trial table).",
    ),
) -> None:
    """Flat vs OHLC-calibrated cost trials (dev-only SYNTHETIC diagnostic).

    Matched books under flat half-spread and Corwin–Schultz / Abdi–Ranaldo /
    Roll. Reports decomposed costs only — never Sharpe or live P&L.
    """
    if not dev:
        raise typer.BadParameter(
            "cost-calibration is dev-only evidence tooling; pass --dev to acknowledge."
        )
    from quant_fund.research.cost_calibration import (
        format_cost_calibration_table,
        run_cost_calibration_trials,
        write_cost_calibration_receipt,
        write_cost_calibration_report,
    )

    frame, receipt = run_cost_calibration_trials(
        half_spread_bps=half_spread_bps,
        lookback=lookback,
        n_dates=n_dates,
        n_names=n_names,
        seed=seed,
        planted_rel_spread=planted_rel_spread,
    )
    path = write_cost_calibration_receipt(receipt, out_dir)
    report = write_cost_calibration_report(frame, receipt, report_path)
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo(format_cost_calibration_table(frame))
    typer.echo(f"receipt={path}")
    typer.echo(f"report={report}")


__all__ = [
    "capacity",
    "cost_calibration",
    "execution_sensitivity_cmd",
    "fleet",
    "rankic",
    "research",
    "verify_identities",
    "verify_receipt_cmd",
    "vol_bench",
]
