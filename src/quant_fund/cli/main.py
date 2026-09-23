"""quant CLI."""

from __future__ import annotations

from pathlib import Path

import typer

from quant_fund.config import dump_resolved, load_config
from quant_fund.hmm.cli import hmm_app
from quant_fund.pipeline.doctor import doctor as run_doctor
from quant_fund.pipeline.train import train_family
from quant_fund.utils.logging import configure_logging, get_logger


def format_data_label(*, synthetic: bool, data_source: str) -> str:
    """Always-printed DATA_LABEL line for research / paper CLI output."""
    return f"DATA_LABEL={'SYNTHETIC' if synthetic else data_source}"


def format_fdr_families(hypotheses: list) -> str:
    """BH-FDR family split summary — calibration/discovery/bound never pooled."""
    cal_n = sum(1 for h in hypotheses if getattr(h, "family", None) == "calibration")
    disc_n = sum(1 for h in hypotheses if getattr(h, "family", None) == "discovery")
    bound_n = sum(1 for h in hypotheses if getattr(h, "family", None) == "bound")
    cal = sum(
        1
        for h in hypotheses
        if getattr(h, "family", None) == "calibration" and getattr(h, "reject_fdr", False)
    )
    disc = sum(
        1
        for h in hypotheses
        if getattr(h, "family", None) == "discovery" and getattr(h, "reject_fdr", False)
    )
    return (
        "BH-FDR families (split, never pooled): "
        f"calibration n={cal_n} rejects={cal}; "
        f"discovery n={disc_n} rejects={disc}; "
        f"bound n={bound_n} (not FDR-adjusted)"
    )


app = typer.Typer(
    help="Dipcatcher — Artificial Hedge's proprietary research lab. Default mode is research, never live."
)
train_app = typer.Typer(help="Train a forecast family.")
app.add_typer(train_app, name="train")
app.add_typer(hmm_app, name="hmm")


def _cfg(config: Path):
    cfg = load_config(config)
    configure_logging()
    dump_resolved(cfg, Path(cfg.data.root) / "metadata" / "resolved_config.json")
    return cfg


@app.command()
def doctor(config: Path = typer.Option(Path("configs/research.yaml"))) -> None:
    info = run_doctor(str(config))
    for k, v in info.items():
        typer.echo(f"{k}: {v}")
    # Research-only Kyle/OFI dump path (CLI hint; not a live gate).
    typer.echo(
        "kyle_ofi: dipcatcher kyle-ofi --dump-lambda-series PATH "
        "(research_only λ date-series parquet; no Sharpe/pnl)"
    )
    typer.echo(
        "verify-research: runs northset_session_means_honesty_errors "
        "(session soft-verify dispatcher; research_only, no Sharpe)"
    )
    cfg = _cfg(config)
    ns = cfg.northset
    typer.echo(
        f"northset.shape_floors: depth_shape_finite_floor={ns.depth_shape_finite_floor} "
        f"concentration_top_finite_floor={ns.concentration_top_finite_floor} "
        f"queue_priority_finite_floor={ns.queue_priority_finite_floor} "
        f"side_notional_finite_floor={ns.side_notional_finite_floor} "
        f"tob_size_share_finite_floor={ns.tob_size_share_finite_floor} "
        f"(shape_columns_ensured is a receipt stamp when synth ensure runs)"
    )
    required = [
        info.get(f"dir_{part}") == "ok" for part in ("raw", "bronze", "silver", "gold", "metadata")
    ]
    healthy = (
        all(required)
        and info.get("data_manifest") == "ok"
        and info.get("research_receipt") == "ok"
        and info.get("core_imports") == "ok"
        and info.get("live") != "REJECTED"
    )
    if not healthy:
        raise typer.Exit(code=1)


@app.command()
def ingest(config: Path = typer.Option(Path("configs/research.yaml"))) -> None:
    from quant_fund.data.ingest import ingest as do_ingest

    cfg = _cfg(config)
    paths = do_ingest(cfg)
    log = get_logger(cmd="ingest", source=cfg.data.source)
    log.info("ingested", paths={k: str(v) for k, v in paths.items()})
    if cfg.data.source == "synthetic":
        typer.echo("SYNTHETIC ingest complete")
    for k, v in paths.items():
        typer.echo(f"{k}: {v}")


def _collect_param_value(raw: str) -> object:
    """Coerce a --param value to int/float when it cleanly parses, else str."""
    text = raw.strip()
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


@app.command()
def collect(
    config: Path = typer.Option(Path("configs/research.yaml")),
    source: str = typer.Option(..., "--source", help="Registered public source name"),
    param: list[str] = typer.Option(
        [], "--param", help="Adapter fetch kwarg as key=value (repeatable)"
    ),
    filename: str | None = typer.Option(
        None, help="Output filename below raw/sources/ (default <source>.parquet)"
    ),
) -> None:
    """Explicit opt-in public-source collection (may use the network).

    Writes the normalized PIT frame to ``<data.root>/raw/sources/<source>.parquet``
    plus a JSON receipt with sha256, row counts, and provenance. This is
    collection only — the offline ingest pipeline is unchanged.
    """
    from quant_fund.data.collector import collect_source

    cfg = _cfg(config)
    fetch_kwargs: dict[str, object] = {}
    for item in param:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise typer.BadParameter(f"--param must be key=value, got {item!r}")
        fetch_kwargs[key.strip()] = _collect_param_value(value)
    if "path" not in fetch_kwargs and cfg.data.source_path is not None:
        fetch_kwargs["path"] = cfg.data.source_path
    try:
        result = collect_source(source, cfg.data.root, fetch_kwargs=fetch_kwargs, filename=filename)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"source={source} rows={result.frame.height}")
    typer.echo(f"data: {result.data}")
    typer.echo(f"receipt: {result.receipt}")


@app.command("build-features")
def build_features_cmd(config: Path = typer.Option(Path("configs/research.yaml"))) -> None:
    from quant_fund.pipeline.dataset import build_gold

    cfg = _cfg(config)
    feats, _ = build_gold(cfg)
    typer.echo(f"features rows={feats.height} cols={len(feats.columns)}")


@app.command("build-labels")
def build_labels_cmd(config: Path = typer.Option(Path("configs/research.yaml"))) -> None:
    from quant_fund.pipeline.dataset import build_gold

    cfg = _cfg(config)
    _, labs = build_gold(cfg)
    typer.echo(f"labels rows={labs.height}")


@train_app.callback(invoke_without_command=True)
def train_callback(
    ctx: typer.Context,
    config: Path = typer.Option(Path("configs/research.yaml")),
    model: str | None = typer.Option(None),
) -> None:
    if ctx.invoked_subcommand is not None:
        return
    typer.echo(
        "Specify a family: ranking, reinforcement, calibration, distribution, volatility, alpha, regime, tail, covariance, liquidity"
    )


def _train(family: str, config: Path, model: str | None) -> None:
    cfg = _cfg(config)
    result = train_family(cfg, family, model)
    typer.echo(result)


@train_app.command("ranking")
def train_ranking(
    config: Path = typer.Option(Path("configs/research.yaml")),
    model: str = typer.Option(
        "ridge",
        help=(
            "auto, composite, ridge, elasticnet, neural, ensemble, xgboost, "
            "lightgbm, lambdarank, or xendcg"
        ),
    ),
) -> None:
    _train("ranking", config, model)


@train_app.command("distribution")
def train_distribution(
    config: Path = typer.Option(Path("configs/research.yaml")),
    model: str = typer.Option(
        "gaussian",
        help="auto, empirical, gaussian, linear_qr, xgboost, or lightgbm",
    ),
) -> None:
    _train("distribution", config, model)


@train_app.command("calibration")
def train_calibration(
    config: Path = typer.Option(Path("configs/research.yaml")),
    model: str = typer.Option("isotonic", help="auto, isotonic, or platt"),
) -> None:
    _train("calibration", config, model)


@train_app.command("volatility")
def train_volatility(
    config: Path = typer.Option(Path("configs/research.yaml")),
    model: str = typer.Option(
        "ewma",
        help="auto, ewma, rolling, har, garch, or tree",
    ),
) -> None:
    _train("volatility", config, model)


@train_app.command("alpha")
def train_alpha(
    config: Path = typer.Option(Path("configs/research.yaml")), model: str = "ridge"
) -> None:
    _train("alpha", config, model)


@train_app.command("covariance")
def train_covariance(config: Path = typer.Option(Path("configs/research.yaml"))) -> None:
    _train("covariance", config, None)


@train_app.command("regime")
def train_regime(
    config: Path = typer.Option(Path("configs/research.yaml")), model: str = "hmm"
) -> None:
    _train("regime", config, model)


@train_app.command("tail")
def train_tail(
    config: Path = typer.Option(Path("configs/research.yaml")), model: str = "historical"
) -> None:
    _train("tail", config, model)


@train_app.command("reinforcement")
def train_reinforcement(
    config: Path = typer.Option(Path("configs/research.yaml")),
    model: str = typer.Option(
        "linucb", help="auto, linucb, thompson, quantile_thompson, or policy_gradient"
    ),
) -> None:
    """Train a research-only contextual RL policy on the causal gold panel."""
    _train("reinforcement", config, model)


@train_app.command("liquidity")
def train_liquidity(config: Path = typer.Option(Path("configs/research.yaml"))) -> None:
    _train("liquidity", config, None)


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
    to_write = session_book
    out.parent.mkdir(parents=True, exist_ok=True)
    to_write.write_parquet(out)
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
        daily_out.parent.mkdir(parents=True, exist_ok=True)
        daily.write_parquet(daily_out)
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


@app.command("verify-research")
def verify_research(
    path: Path = typer.Argument(Path("data/metadata/research/latest.json")),
) -> None:
    """Verify a persisted research receipt and its immutable artifacts.

    Soft-verify includes ``northset_session_means_honesty_errors`` (dispatcher
    over session mean helpers). Research diagnostic only; never live Sharpe.
    """
    import json

    from quant_fund.research.verify import verify_research_artifact

    result = verify_research_artifact(path)
    typer.echo(json.dumps(result, indent=2))
    raise typer.Exit(code=0 if result["valid"] else 1)


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
def report(
    latest: bool = typer.Option(False, "--latest"),
    config: Path = typer.Option(Path("configs/research.yaml")),
) -> None:
    from quant_fund.reporting.report import latest_report_dir, write_report

    cfg = _cfg(config)
    dest = latest_report_dir(Path(cfg.data.root)) / "latest.md"
    write_report(
        dest,
        "Research report",
        {"config": cfg.dump(), "note": "see MLflow for experiment metrics"},
        synthetic=cfg.data.source == "synthetic",
    )
    typer.echo(dest)


@app.command("tearsheet")
def tearsheet_cmd(
    equity: Path = typer.Option(
        ..., "--equity", help="Equity parquet: event_time, nav (research-only label)"
    ),
    fills: Path | None = typer.Option(None, "--fills", help="Optional fills parquet for IS/TCA"),
    weights: Path | None = typer.Option(
        None, "--weights", help="Optional target-weight panel (event_time, security_id, target_weight)"
    ),
    bars: Path | None = typer.Option(
        None, "--bars", help="Optional per-name bars for per-security attribution"
    ),
    out_md: Path | None = typer.Option(None, "--out-md", help="Markdown output path"),
    out_json: Path | None = typer.Option(None, "--out-json", help="JSON sheet output path"),
    periods_per_year: float = typer.Option(252.0, "--periods-per-year"),
    label: str = typer.Option("BACKTEST_SIM", "--label"),
    synthetic: bool = typer.Option(False, "--synthetic"),
) -> None:
    """Institutional tearsheet: summary stats, drawdowns, period table, costs, attribution.

    Reads backtest/paper artifacts (equity curve, fills, weights) and emits a
    durable report. Never a live-P&L claim — live_pnl_claim=False is stamped.
    """
    import json

    import polars as pl

    from quant_fund.reporting.tearsheet import (
        build_tearsheet,
        tearsheet_markdown,
        write_tearsheet_md,
    )

    eq = pl.read_parquet(equity)
    sheet = build_tearsheet(
        eq,
        fills=pl.read_parquet(fills) if fills is not None else None,
        weights=pl.read_parquet(weights) if weights is not None else None,
        bars=pl.read_parquet(bars) if bars is not None else None,
        periods_per_year=periods_per_year,
        label=label,
        synthetic=synthetic,
    )
    if out_md is not None:
        write_tearsheet_md(out_md, sheet)
        typer.echo(f"markdown={out_md}")
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(sheet, indent=2, default=str))
        typer.echo(f"json={out_json}")
    if out_md is None and out_json is None:
        typer.echo(tearsheet_markdown(sheet))


@app.command()
def api(host: str = "127.0.0.1", port: int = 8000) -> None:
    import os

    import uvicorn

    loopback_hosts = {"127.0.0.1", "::1", "localhost"}
    if host not in loopback_hosts and not os.environ.get("QUANT_API_KEY"):
        raise typer.BadParameter(
            "non-loopback API binding requires QUANT_API_KEY; refusing unauthenticated exposure"
        )
    uvicorn.run("quant_fund.api.app:app", host=host, port=port, reload=False)


@app.command()
def paper(
    config: Path = typer.Option(Path("configs/paper.yaml")),
    max_steps: int | None = typer.Option(None, help="Cap replay steps (paper accelerated)."),
    shadow: bool = typer.Option(True, help="Enable shadow challenger slot (no capital)."),
    wall_clock: bool = typer.Option(False, help="Use wall clock instead of bar replay."),
    halt: bool = typer.Option(False, help="Force kill switch HALT_NEW_ORDERS for this run."),
    resume: bool = typer.Option(False, help="Resume from broker_state.json for --run-id / latest."),
    run_id: str | None = typer.Option(None, help="Paper run_id (new or resume target)."),
    clear_halt: bool = typer.Option(
        False, help="Force kill switch ENABLED (overrides config / pairs with --resume)."
    ),
    from_start: bool = typer.Option(
        False,
        help="Use earliest decision dates (multi-day grind); default prefers latest window.",
    ),
) -> None:
    """Phase 17 paper / shadow loop with simulated broker (no live fills)."""
    import json

    import polars as pl

    from quant_fund.features.engine import build_features
    from quant_fund.paper.ledger import latest_run_id
    from quant_fund.paper.loop import build_scaled_challenger_weights, run_paper_loop
    from quant_fund.pipeline.dataset import ensure_silver
    from quant_fund.pipeline.forecast import build_causal_weight_panel

    cfg = _cfg(config)
    if halt and clear_halt:
        raise typer.BadParameter("pass only one of --halt / --clear-halt")
    if halt:
        cfg.kill_switch.state = "HALT_NEW_ORDERS"
    if clear_halt:
        cfg.kill_switch.state = "ENABLED"
    bars = ensure_silver(cfg)
    feat = build_features(bars, cfg)
    dates = feat["event_time"].unique().sort().to_list()
    steps = max_steps if max_steps is not None else cfg.paper.max_steps
    resume_id = run_id or (
        latest_run_id(cfg.data.root, cfg.paper.ledger_subdir) if resume else None
    )
    if resume and not resume_id:
        raise typer.BadParameter("--resume needs --run-id or a prior latest_run.json")
    if steps is not None and not resume and not from_start:
        # Use the *latest* window so warmup/cov history exists (not day-0 cash).
        n = max(int(steps) + 2, 3)
        dates = dates[-n:]
    elif steps is not None and (resume or from_start):
        # Sequential: keep full history for cov; loop itself caps steps.
        pass
    weights = build_causal_weight_panel(cfg, dates)
    shadow_w = None
    enable_shadow = shadow and cfg.paper.enable_shadow
    if enable_shadow:
        shadow_w = weights.with_columns(
            (pl.col("target_weight") * float(cfg.paper.shadow_scale) * 0.85).alias("target_weight")
        )
    # Extra named scaled challengers from paper.challenger_scales (research-only).
    # Primary shadow above stays for ledger / promotion_dry_run compatibility.
    built = build_scaled_challenger_weights(weights, list(cfg.paper.challenger_scales or []))
    challenger_w = built or None
    result = run_paper_loop(
        feat,
        cfg,
        champion_weights=weights,
        shadow_weights=shadow_w,
        challenger_weights=challenger_w,
        initial_nav=float(cfg.paper.initial_nav),
        max_steps=steps,
        use_wall_clock=wall_clock or cfg.paper.use_wall_clock,
        run_id=run_id,
        resume=resume,
        resume_run_id=resume_id if resume else None,
        prefer_latest=not (resume or from_start),
    )
    typer.echo(f"DATA_LABEL={result.source_note}")
    if result.source_note == "SYNTHETIC":
        typer.echo("SYNTHETIC — paper ledger is simulated research/infrastructure only.")
    typer.echo(f"run_id={result.run_id}")
    typer.echo(f"divergence={result.divergence}")
    m = result.metrics
    typer.echo(
        json.dumps(
            {
                "n_steps": m.get("n_steps"),
                "n_steps_this_run": m.get("n_steps_this_run"),
                "n_fills": m.get("n_fills"),
                "risk_gate_rejects": m.get("risk_gate_rejects"),
                "kill_switch_halts": m.get("kill_switch_halts"),
                "resumed": m.get("resumed"),
                "execution": m.get("execution"),
                "var_es": m.get("var_es"),
                "exposure": m.get("exposure"),
                "stress": m.get("stress"),
                "portfolio_conformal": m.get("portfolio_conformal"),
                "promotion_dry_run": m.get("promotion_dry_run"),
                "research_only": True,
                "live_pnl_claim": False,
                "paths": result.paths,
            },
            indent=2,
            default=str,
        )
    )


@app.command()
def monitor(
    config: Path = typer.Option(Path("configs/paper.yaml")),
    run_id: str | None = typer.Option(None, help="Paper run_id (default: latest_run.json)."),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON instead of markdown."),
    out: Path | None = typer.Option(None, "--out", help="Write output to this path."),
) -> None:
    """Ops snapshot over the latest paper run: limits, staleness, kill state."""
    import json

    import polars as pl

    from quant_fund.monitoring.dashboard import ops_snapshot, render_markdown
    from quant_fund.paper.ledger import latest_run_id, load_broker_state, paper_root

    cfg = _cfg(config)
    rid = run_id or latest_run_id(cfg.data.root, cfg.paper.ledger_subdir)
    if not rid:
        raise typer.BadParameter("no paper run found — pass --run-id or run `dipcatcher paper`")
    state = load_broker_state(cfg.data.root, rid, cfg.paper.ledger_subdir)
    if not state:
        raise typer.BadParameter(f"no broker_state.json for run_id={rid}")
    champion = state.get("champion") or {}

    equity_path = paper_root(cfg.data.root, cfg.paper.ledger_subdir) / rid / "equity.parquet"
    nav = peak = None
    asof = None
    if equity_path.is_file():
        eq = pl.read_parquet(equity_path).sort("asof")
        if eq.height:
            nav = float(eq["nav"][-1])
            peak = float(eq["nav"].to_numpy().max())
            asof = eq["asof"][-1]
    marks = champion.get("last_marks") or {}
    shares = champion.get("shares") or {}
    broker_nav = float(champion.get("cash", 0.0)) + sum(
        float(q) * float(marks.get(s, 0.0)) for s, q in shares.items()
    )
    if nav is None:
        # Fall back to broker cash + marks for a resumed-but-unflushed run.
        nav = broker_nav
    # Reconcile broker state vs the last ledger equity row.
    recon_mismatches = None
    if peak is not None and nav > 0:
        recon_mismatches = 0 if abs(broker_nav - nav) / nav <= 1e-6 else 1

    snap = ops_snapshot(
        nav=nav,
        cash=float(champion.get("cash", 0.0)),
        positions={str(k): float(v) for k, v in shares.items()},
        marks={str(k): float(v) for k, v in marks.items()},
        config=cfg,
        asof=asof,
        mark_age_bars=state.get("mark_ages"),
        n_open_orders=len(champion.get("open_orders") or []),
        kill_switch_state=champion.get("kill_state"),
        peak_nav=peak,
        recon_mismatches=recon_mismatches,
    )
    snap["run_id"] = rid
    text = json.dumps(snap, indent=2, default=str) if json_out else render_markdown(snap)
    if out is not None:
        out.write_text(text)
        typer.echo(f"wrote {out}")
    else:
        typer.echo(text)
    # Nagios-style exit codes so schedulers/alerting can consume status.
    status = snap["overall_status"]
    if status == "breach":
        raise typer.Exit(2)
    if status == "warn":
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
