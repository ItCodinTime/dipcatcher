"""Research-lane commands: validate/forecast/backtest/research/verify."""

from __future__ import annotations

from pathlib import Path

import typer

from ._app import (
    _cfg,
    app,
    format_data_label,
    format_fdr_families,
)


@app.command()
def validate(
    model_id: str,
    config: Path = typer.Option(Path("configs/research.yaml")),
    metrics: Path | None = typer.Option(
        None, help="Optional JSON metrics blob (mean_ic, net_spread, turnover, ...)."
    ),
    claim_live: bool = typer.Option(
        False, help="Set when asserting a live/production claim (fails closed on SYNTHETIC)."
    ),
    # Fail closed: assert the leakage suite passed explicitly (--leakage-ok)
    # rather than defaulting the promotion gate to "passed".
    leakage_ok: bool = typer.Option(False, help="Whether the leakage suite passed."),
) -> None:
    """Fail-closed research / promotion gates (see docs/VALIDATION.md)."""
    import json

    from quant_fund.validation.gates import validate_candidate

    cfg = _cfg(config)
    result = validate_candidate(
        model_id,
        cfg,
        metrics_path=metrics,
        claim_live=claim_live,
        leakage_ok=leakage_ok,
    )
    typer.echo(json.dumps(result, indent=2, default=str))
    if result.get("data_label") == "SYNTHETIC":
        typer.echo("DATA_LABEL=SYNTHETIC")
    raise typer.Exit(code=0 if result.get("ok") else 1)


@app.command()
def forecast(
    config: Path = typer.Option(Path("configs/research.yaml")), date: str | None = None
) -> None:
    from datetime import datetime

    from quant_fund.pipeline.forecast import forecast_asof

    cfg = _cfg(config)
    asof = datetime.fromisoformat(date) if date else None
    state = forecast_asof(cfg, asof)
    if "SYNTHETIC" in state.notes:
        typer.echo("SYNTHETIC")
    for f in state.forecasts[:15]:
        hz = next(iter(f.interval_lo), "5d")
        lo = f.interval_lo.get(hz)
        hi = f.interval_hi.get(hz)
        extra = ""
        if lo is not None and hi is not None:
            extra = (
                f" interval[{hz}]=[{lo:+.4%},{hi:+.4%}] "
                f"interval_alpha={f.interval_alpha} {f.interval_method}"
            )
        typer.echo(
            f"{f.symbol:8} alpha={f.alpha.get('5d', 0):+.4%} rank={f.rank_percentile.get('5d', 0):.2f} "
            f"vol={f.volatility.get('5d', 0):.3f}{extra}"
        )


@app.command("kronos-forecast")
def kronos_forecast(
    config: Path = typer.Option(Path("configs/research.yaml")), date: str | None = None
) -> None:
    """Research-only Kronos candle-path forecasts (requires train.kronos.enabled).

    Loads strictly local, pre-downloaded artifacts — never the network or live
    execution paths. Quantile bands are the predicted candle envelope, not a
    calibrated predictive interval.
    """
    from datetime import datetime

    from quant_fund.pipeline.kronos import forecast_kronos_frame

    cfg = _cfg(config)
    asof = datetime.fromisoformat(date) if date else None
    try:
        state = forecast_kronos_frame(cfg, asof=asof)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if "SYNTHETIC" in state.notes:
        typer.echo("SYNTHETIC")
    typer.echo("kronos.adapter.v1 — research-only candle-path forecast")
    for f in state.forecasts[:15]:
        hz = next(iter(f.expected_returns), "5d")
        q = f.quantiles.get(hz, {})
        typer.echo(
            f"{f.symbol:8} expected={f.expected_returns.get(hz, 0):+.4%} "
            f"q05={q.get(0.05, 0):+.4%} q50={q.get(0.5, 0):+.4%} q95={q.get(0.95, 0):+.4%} "
            f"p_up={f.probability_positive.get(hz, 0):.2f} vol={f.volatility.get(hz, 0):.3f}"
        )


@app.command()
def optimize(
    config: Path = typer.Option(Path("configs/research.yaml")), date: str | None = None
) -> None:
    from datetime import datetime

    from quant_fund.pipeline.forecast import optimize_asof

    cfg = _cfg(config)
    asof = datetime.fromisoformat(date) if date else None
    w = optimize_asof(cfg, asof)
    typer.echo(w.head(20))


@app.command()
def backtest(
    config: Path = typer.Option(Path("configs/backtest.yaml")),
    engine: str = typer.Option(
        "ref", "--engine", help="ref (event loop) or fast (bit-identical vectorized replay)"
    ),
) -> None:
    from quant_fund.backtest.engine import run_backtest
    from quant_fund.backtest.fast_replay import run_backtest_fast
    from quant_fund.pipeline.dataset import ensure_silver
    from quant_fund.pipeline.forecast import build_causal_weight_panel

    if engine not in ("ref", "fast"):
        raise typer.BadParameter("--engine must be 'ref' or 'fast'")
    cfg = _cfg(config)
    bars = ensure_silver(cfg)
    # features for adv/vol
    from quant_fund.features.engine import build_features

    feat = build_features(bars, cfg)
    dates = feat["event_time"].unique().sort().to_list()
    # Causal: optimize_asof(asof=d) per date — no end-of-sample weight broadcast
    weights = build_causal_weight_panel(cfg, dates)
    run = run_backtest_fast if engine == "fast" else run_backtest
    result = run(feat, weights, cfg)
    if result.source_note == "SYNTHETIC":
        typer.echo("SYNTHETIC")
    typer.echo(result.metrics)


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


@app.command("verify-research")
def verify_research(
    path: Path = typer.Argument(Path("data/metadata/research/latest.json")),
) -> None:
    """Verify a notebook, completed Phase-1 run directory, or evidence index.

    Soft-verify includes ``northset_session_means_honesty_errors`` (dispatcher
    over session mean helpers). Research diagnostic only; never live Sharpe.
    """
    import json

    if path.is_dir():
        try:
            directory_manifest = json.loads((path / "manifest.json").read_text())
        except (OSError, UnicodeError, json.JSONDecodeError):
            directory_manifest = None
        if (
            isinstance(directory_manifest, dict)
            and directory_manifest.get("kind") == "forward_shadow_manifest"
        ):
            from quant_fund.paper.forward_shadow import verify

            result = verify(path)
        else:
            from quant_fund.research.phase1_verify import verify_phase1_run

            result = verify_phase1_run(path)
    elif path.is_file() and path.name.endswith(".json"):
        try:
            payload = json.loads(path.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict) and payload.get("kind") == "phase1_evidence_index":
            from quant_fund.research.phase1_verify import verify_phase1_index

            result = verify_phase1_index(path)
        else:
            from quant_fund.research.verify import verify_research_artifact

            result = verify_research_artifact(path)
    else:
        from quant_fund.research.verify import verify_research_artifact

        result = verify_research_artifact(path)
    typer.echo(json.dumps(result, indent=2))
    raise typer.Exit(code=0 if result["valid"] else 1)


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


@app.command(hidden=True)
def lab(config: Path = typer.Option(Path("configs/research.yaml"))) -> None:
    """Legacy compatibility alias for `dipcatcher research`."""
    from quant_fund.research.agent import run_research

    cfg = _cfg(config)
    nb = run_research(cfg)
    if nb.synthetic:
        typer.echo("SYNTHETIC")
    typer.echo(nb.disclaimer)
    typer.echo(nb.artifacts.get("json"))


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
    path = write_fleet_receipt(receipt, out_dir)
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo(frame)
    typer.echo(f"receipt={path}")
