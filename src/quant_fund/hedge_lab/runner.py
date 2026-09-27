"""Artificial Hedge fund-lab runner.

Causal public-ridge book on the file tape, then Sharpe / Sortino / Calmar /
drawdown via the paper/backtest analytics catalog. Research IC notebooks stay
Sharpe-free. ``blend_weight`` is not moved here.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict

from quant_fund.backtest.engine import run_backtest
from quant_fund.config.models import AppConfig
from quant_fund.data.lake import Lake
from quant_fund.hedge_lab.mirror import negate_target_weights
from quant_fund.hedge_lab.resources import (
    assert_disk_budget,
    claim_workspace,
    ensure_dirs,
    lab_root,
    payload_bytes,
    ram_plan,
)
from quant_fund.hedge_lab.scoreboard import book_economic_scoreboard, moving_block_bootstrap_ci
from quant_fund.models.ranking import drop_oracle_columns
from quant_fund.models.robinhood_plus.compare import train_public_ridge
from quant_fund.pipeline.dataset import (
    build_gold,
    clear_panel_cache,
    ensure_silver,
    panel,
)
from quant_fund.pipeline.forecast import build_causal_weight_panel, clear_forecast_caches
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent
from quant_fund.risk.overlay import BookRiskOverlay


class HedgeLabProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lab_id: str = "artificial_hedge.hedge_lab.v1"
    rebalance_every: int = 5
    lookback_bars: int = 60
    initial_nav: float = 1_000_000.0
    bootstrap: bool = True
    n_boot: int | None = None
    claim_ram: bool = True
    risk_overlay: bool = True
    vol_target: float = 0.025
    dd_limit: float = 0.05
    es_limit: float = 0.006
    var_tail_p: float = 0.01
    overlay_lookback: int = 63
    mirror: bool = False


def _lab_artifact_dirs(cfg: AppConfig, source: str) -> list[Path]:
    """Write under the lake. Publish to repo artifacts only for non-SYNTHETIC runs.

    Synthetic smoke tests use a tmp lake; they must not clobber the file-tape
    ``artifacts/hedge_lab/weights.parquet``.
    """
    local = Path(cfg.data.root) / "artifacts" / "hedge_lab"
    dirs = [local]
    if source != "SYNTHETIC":
        public = lab_root() / "artifacts" / "hedge_lab"
        if public.resolve() != local.resolve():
            dirs.append(public)
    return dirs


def rebalance_dates(times: list[datetime], *, lookback: int, every: int) -> list[datetime]:
    usable = [t for t in times if isinstance(t, datetime)]
    usable = usable[max(int(lookback), 2) :]
    step = max(int(every), 1)
    return usable[::step]


def _equity_returns(equity: pl.DataFrame) -> np.ndarray:
    if equity.is_empty() or "nav" not in equity.columns:
        return np.asarray([], dtype=float)
    nav = equity["nav"].to_numpy().astype(float)
    if nav.size < 2:
        return np.asarray([], dtype=float)
    return np.asarray(nav[1:] / nav[:-1] - 1.0, dtype=float)


def _benchmark_returns(
    feat: pl.DataFrame, equity: pl.DataFrame, benchmark_id: str
) -> np.ndarray | None:
    if not benchmark_id or "security_id" not in feat.columns or "close" not in feat.columns:
        return None
    if equity.is_empty() or "event_time" not in equity.columns:
        return None
    bench = (
        feat.filter(pl.col("security_id") == benchmark_id)
        .select("event_time", "close")
        .unique(subset=["event_time"])
        .sort("event_time")
    )
    if bench.height < 3:
        return None
    joined = (
        equity.select("event_time").join(bench, on="event_time", how="inner").sort("event_time")
    )
    close = joined["close"].to_numpy().astype(float)
    if close.size < 3:
        return None
    return np.asarray(close[1:] / close[:-1] - 1.0, dtype=float)


def _align_book_and_benchmark(book: np.ndarray, bench: np.ndarray | None) -> np.ndarray | None:
    if bench is None:
        return None
    n = min(int(book.size), int(bench.size))
    if n < 2:
        return None
    return bench[-n:]


def _persist_public_feature_gold(config: AppConfig) -> None:
    """Drop SYNTHETIC oracle columns on disk so forecast uses the public ranker."""
    lake = Lake(Path(config.data.root))
    if not lake.exists("gold/features.parquet"):
        return
    feats = lake.read_parquet("gold/features.parquet")
    public = drop_oracle_columns(feats)
    if public.columns != feats.columns:
        lake.write_parquet(public, "gold/features.parquet")
    clear_panel_cache()
    clear_forecast_caches()


def _metric_num(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _backtest_slice(result: Any) -> dict[str, Any]:
    return {
        "total_return": _metric_num(result.metrics.get("total_return")),
        "sharpe": _metric_num(result.metrics.get("sharpe")),
        "max_drawdown": _metric_num(result.metrics.get("max_drawdown")),
        "mean_turnover": _metric_num(result.metrics.get("mean_turnover")),
        "commission": _metric_num(result.metrics.get("commission")),
        "spread": _metric_num(result.metrics.get("spread")),
        "impact": _metric_num(result.metrics.get("impact")),
        "flag_high_sharpe": bool(result.metrics.get("flag_high_sharpe")),
        "risk_gate_rejects": int(result.metrics.get("risk_gate_rejects") or 0),
        "n": _metric_num(result.metrics.get("n")),
        "source_note": result.source_note,
        "book_risk_overlay": result.metrics.get("book_risk_overlay"),
    }


def run_hedge_lab(
    config: AppConfig,
    protocol: HedgeLabProtocol | None = None,
    *,
    bootstrap: bool | None = None,
    n_boot: int | None = None,
    refresh_gold: bool = False,
    claim_ram: bool | None = None,
) -> dict[str, Any]:
    proto = protocol or HedgeLabProtocol()
    disk = assert_disk_budget()
    ram = ram_plan()
    cfg = config.model_copy(deep=True)
    cfg.fusion.skip_intervals = True
    cfg.robinhood_plus.blend_weight = 0.0
    root = Path(cfg.data.root)
    source = "SYNTHETIC" if cfg.data.source == "synthetic" else str(cfg.data.source)
    arts = _lab_artifact_dirs(cfg, source)
    ensure_dirs([root / "metadata", *arts])
    if refresh_gold:
        clear_panel_cache()
        clear_forecast_caches()
        build_gold(cfg, refresh=True)
        clear_panel_cache()
    else:
        panel(cfg)
    _persist_public_feature_gold(cfg)
    gold = drop_oracle_columns(panel(cfg))
    if gold.is_empty():
        raise ValueError(
            "hedge-lab gold panel is empty; fetch a file tape (`dipcatcher hedge-lab --fetch-tape`) "
            "or point the config at a non-empty lake"
        )
    train_public_ridge(cfg, gold)
    times = [t for t in gold["event_time"].unique().sort().to_list() if isinstance(t, datetime)]
    dates = rebalance_dates(times, lookback=proto.lookback_bars, every=proto.rebalance_every)
    if not dates:
        raise ValueError("hedge-lab has no rebalance dates after lookback")
    weights = build_causal_weight_panel(cfg, dates)
    if not weights.is_empty():
        for art in arts:
            weights.write_parquet(art / "weights.parquet")
    # Valuation tape is the full silver lake (union calendar). Membership-filtered
    # gold is for signals only: a name that rotates out of top-N ADV must still
    # be markable so the book can flatten instead of going stale.
    bars = ensure_silver(cfg)
    overlay = None
    if proto.risk_overlay:
        overlay = BookRiskOverlay(
            vol_target=proto.vol_target,
            dd_limit=proto.dd_limit,
            es_limit=proto.es_limit,
            tail_p=proto.var_tail_p,
            lookback=proto.overlay_lookback,
        )
    result = run_backtest(bars, weights, cfg, initial_nav=proto.initial_nav, risk_overlay=overlay)
    rets = _equity_returns(result.equity)
    bench = _align_book_and_benchmark(
        rets, _benchmark_returns(bars, result.equity, str(cfg.data.benchmark_id))
    )
    if bench is not None and bench.size != rets.size:
        n = min(rets.size, bench.size)
        rets = rets[-n:]
        bench = bench[-n:]
    economic = book_economic_scoreboard(rets, data_source=source, benchmark_returns=bench)
    do_boot = bootstrap if bootstrap is not None else proto.bootstrap
    boot_n = n_boot if n_boot is not None else proto.n_boot
    boot: dict[str, Any] = {"status": "skipped"}
    if do_boot and rets.size >= 20:
        boot = moving_block_bootstrap_ci(rets, n_boot=boot_n)
    mirror_blob: dict[str, Any] = {"status": "skipped", "enabled": False}
    if proto.mirror and not weights.is_empty():
        overlay_m = None
        if proto.risk_overlay:
            overlay_m = BookRiskOverlay(
                vol_target=proto.vol_target,
                dd_limit=proto.dd_limit,
                es_limit=proto.es_limit,
                tail_p=proto.var_tail_p,
                lookback=proto.overlay_lookback,
            )
        result_m = run_backtest(
            bars,
            negate_target_weights(weights),
            cfg,
            initial_nav=proto.initial_nav,
            risk_overlay=overlay_m,
        )
        rets_m = _equity_returns(result_m.equity)
        bench_m = _align_book_and_benchmark(
            rets_m, _benchmark_returns(bars, result_m.equity, str(cfg.data.benchmark_id))
        )
        if bench_m is not None and bench_m.size != rets_m.size:
            n_m = min(rets_m.size, bench_m.size)
            rets_m = rets_m[-n_m:]
            bench_m = bench_m[-n_m:]
        economic_m = book_economic_scoreboard(rets_m, data_source=source, benchmark_returns=bench_m)
        boot_m: dict[str, Any] = {"status": "skipped"}
        if do_boot and rets_m.size >= 20:
            boot_m = moving_block_bootstrap_ci(rets_m, n_boot=boot_n)
        if not result_m.equity.is_empty():
            for art in arts:
                result_m.equity.write_parquet(art / "equity_mirror.parquet")
        mirror_blob = {
            "status": "ok",
            "enabled": True,
            "economic": economic_m,
            "backtest": _backtest_slice(result_m),
            "bootstrap": boot_m,
            "n_returns": int(rets_m.size),
            "sharpe_sum": float(economic["sharpe"]) + float(economic_m["sharpe"]),
            "costs_even_under_sign_flip": True,
            "note": (
                "Sign-flipped target weights on the same fills and cost model. "
                "Frictionless CS Sharpe flips sign; spread/commission/impact do not. "
                "A 5% drawdown halt cannot coexist with -150% total return. "
                "Not a live P&L claim. blend_weight stays 0."
            ),
        }
    risk_diag: dict[str, Any] = {"status": "skipped"}
    if rets.size >= 20:
        from quant_fund.risk.pyrisk import BackTesting, ExpectedShortfall, ValueAtRisk
        from quant_fund.risk.pyriskmgmt import ewma_var_es

        var_eng = ValueAtRisk(rets, alpha=proto.var_tail_p)
        es_eng = ExpectedShortfall(rets, alpha=proto.var_tail_p)
        emp_var = float(var_eng.empirical_var())
        tester = BackTesting(rets)
        ewma_var, ewma_es = ewma_var_es(rets, alpha=1.0 - proto.var_tail_p)
        risk_diag = {
            "empirical_var": emp_var,
            "parametric_var": float(var_eng.parametrical_var()),
            "monte_carlo_var": float(var_eng.non_parametrical_var(n_iter=8_000)),
            "evt_var": float(var_eng.extreme_var()),
            "empirical_es": float(es_eng.empirical_cvar()),
            "parametric_es": float(es_eng.parametrical_cvar()),
            "ewma_var": float(ewma_var),
            "ewma_es": float(ewma_es),
            "kupiec": tester.kupiec_test(emp_var, alpha=proto.var_tail_p),
            "christoffersen": tester.christoffersen_test(emp_var, alpha=proto.var_tail_p),
            "sources": ("lprtk/pyRisk", "GianMarcoOddo/pyriskmgmt"),
            "research_only": True,
            "execution_claim": "paper_backtest",
        }
    research_twin = {
        "family": "hedge_lab",
        "lab_id": proto.lab_id,
        "research_only": True,
        "execution_claim": "research_only",
        "data_source": source,
        "synthetic_not_promotable": source == "SYNTHETIC",
        "champion_alias": False,
        "blend_weight": 0.0,
        "n_rebalance_dates": int(len(dates)),
        "n_weight_rows": int(weights.height),
        "n_names": int(gold["security_id"].n_unique()) if "security_id" in gold.columns else 0,
        "n_returns": int(rets.size),
    }
    if not family_blob_forbidden_metrics_absent(research_twin):
        raise AssertionError("hedge-lab research twin leaked nested forbidden keys")
    if not result.equity.is_empty():
        for art in arts:
            result.equity.write_parquet(art / "equity.parquet")
    ram_claim_stats: dict[str, Any] = {"status": "skipped"}
    claimed = None
    do_claim = proto.claim_ram if claim_ram is None else bool(claim_ram)
    if do_claim:
        try:
            claimed, ram_claim_stats = claim_workspace()
            ram_claim_stats = {**ram_claim_stats, "status": "ok"}
        except MemoryError as exc:
            ram_claim_stats = {"status": "skipped_memory_error", "reason": str(exc)}

    receipt: dict[str, Any] = {
        "lab_id": proto.lab_id,
        "firm": "Artificial Hedge",
        "product": "dipcatcher hedge lab",
        "catalog": "hedge_lab_analytics",
        "research_only": True,
        "execution_claim": "paper_backtest",
        "data_source": source,
        "synthetic_not_promotable": source == "SYNTHETIC",
        "champion_alias": False,
        "blend_weight": 0.0,
        "n_rebalance_dates": int(len(dates)),
        "n_weight_rows": int(weights.height),
        "n_names": int(gold["security_id"].n_unique()) if "security_id" in gold.columns else 0,
        "economic": economic,
        "mirror": mirror_blob,
        "research_twin": research_twin,
        "market_risk": risk_diag,
        "backtest": _backtest_slice(result),
        "bootstrap": boot,
        "resources": {
            "disk": disk,
            "payload_bytes": payload_bytes(),
            "ram": ram,
            "ram_claim": ram_claim_stats,
        },
        "frictionless": result.frictionless,
        "note": (
            "Sharpe/Calmar/Sortino are paper-book diagnostics on a session-close "
            "file tape with next-open fills and modeled costs. Not a live P&L claim. "
            "Not a research-family blob. blend_weight stays 0. "
            "Sign-flip mirror is an identity check: costs do not change sign."
        ),
    }
    dest = root / "metadata" / "hedge_lab_receipt.json"
    payload = json.dumps(receipt, indent=2, default=str)
    dest.write_text(payload, encoding="utf-8")
    published = dest
    for art in arts:
        published = art / "latest.json"
        published.write_text(payload, encoding="utf-8")
    receipt["receipt_path"] = str(dest)
    receipt["artifact_path"] = str(published)
    del claimed
    return receipt
