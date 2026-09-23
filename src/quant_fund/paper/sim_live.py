"""Simulated-live PnL driver: SOTA quantile signals through the paper loop.

Pipeline (deterministic, receipt-bound):

1. ``load_deep_bars`` — real collected bars (``*_1d_deep`` / ``*_4h_deep``).
2. ``compute_quantile_panel`` per asset per forecaster spec — causal
   quantiles, cached to an npz keyed by ``(spec, window, taus, closes_sha256)``
   so policy variants never recompute the forecast layer.
3. ``quantile_panels_to_weights`` — the policy map → sparse target panels.
4. ``run_paper_loop`` — champion slot trades the champion spec's panel with
   simulated capital; challenger specs run as named no-capital shadows whose
   L1 divergence from champion lands in the receipt.
5. ``sim_live`` receipt JSON — bars hashes, full strategy spec, equity-curve
   stats, fill/reject accounting, and explicit honesty labels.

Honesty contract: ``live_pnl_claim`` is always ``false``; the equity curve is
simulated against historical bars with modeled costs. The forecast horizon is
next-bar close-to-close while fills execute at next-bar open — the standard
close-signal/next-open convention, disclosed rather than hidden.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from quant_fund.config.models import AppConfig
from quant_fund.paper.loop import PaperLoopResult, run_paper_loop
from quant_fund.paper.quantile_signals import (
    DEFAULT_TAUS,
    QuantilePolicy,
    compute_quantile_panel,
    load_deep_bars,
    quantile_panels_to_weights,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class StrategySlot:
    """One book in the sim: forecaster spec + policy + display name."""

    name: str
    spec: str
    policy: QuantilePolicy


@dataclass
class SimLiveResult:
    run_id: str
    receipt_path: Path
    receipt: dict[str, Any]
    loop: PaperLoopResult | None  # None in bench_only mode


def _equity_stats(equity: pl.DataFrame, bars_per_year: float) -> dict[str, Any]:
    """Honest summary of the simulated champion equity curve."""
    if equity.is_empty() or "nav" not in equity.columns:
        return {"status": "no_equity"}
    nav = equity["nav"].to_numpy().astype(float)
    nav = nav[np.isfinite(nav)]
    if nav.size < 2:
        return {"status": "degenerate"}
    rets = np.diff(nav) / nav[:-1]
    rets = rets[np.isfinite(rets)]
    peak = np.maximum.accumulate(nav)
    dd = nav / peak - 1.0
    ann_ret = float(nav[-1] / nav[0]) ** (bars_per_year / max(nav.size - 1, 1)) - 1.0
    ann_vol = float(np.std(rets, ddof=1) * np.sqrt(bars_per_year)) if rets.size > 1 else float("nan")
    sharpe = float(np.mean(rets) / np.std(rets, ddof=1) * np.sqrt(bars_per_year)) if (
        rets.size > 1 and np.std(rets, ddof=1) > 0
    ) else float("nan")
    return {
        "status": "ok",
        "n_marks": int(nav.size),
        "nav_start": float(nav[0]),
        "nav_end": float(nav[-1]),
        "total_return": float(nav[-1] / nav[0] - 1.0),
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe_simulated": sharpe,
        "max_drawdown": float(np.min(dd)),
        "bars_per_year": float(bars_per_year),
    }


def _bars_per_year(interval: str) -> float:
    return {"1d": 365.25, "4h": 6.0 * 365.25, "1h": 24.0 * 365.25}.get(interval, 365.25)


def _quantile_panel_cached(
    *,
    cache_dir: Path,
    sid: str,
    spec: str,
    window: int,
    taus: np.ndarray,
    closes: np.ndarray,
    min_history: int | None,
) -> tuple[np.ndarray, dict[str, int], str]:
    """Deterministic cache: quantile panel keyed by inputs (sha256)."""
    key = {
        "sid": sid,
        "spec": spec,
        "window": int(window),
        "taus": [float(t) for t in taus],
        "min_history": min_history,
        "closes_sha256": hashlib.sha256(closes.tobytes()).hexdigest(),
    }
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    cache_path = cache_dir / f"qpanel_{sid}_{spec}_{digest}.npz"
    if cache_path.is_file():
        with np.load(cache_path) as z:
            return z["panel"], json.loads(str(z["stats_json"])), digest
    panel, stats = compute_quantile_panel(
        closes, spec, taus, window=window, min_history=min_history
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, panel=panel, stats_json=np.array(json.dumps(stats)))
    return panel, stats, digest


def run_sim_live(
    *,
    bars_root: Path,
    symbols: list[str],
    interval: str,
    config: AppConfig,
    champion: StrategySlot,
    challengers: list[StrategySlot] | None = None,
    window: int = 750,
    taus: np.ndarray | None = None,
    min_history: int | None = None,
    tail_bars: int | None = None,
    cache_dir: Path | None = None,
    out_dir: Path,
    run_id: str | None = None,
    max_steps: int | None = None,
    initial_nav: float | None = None,
    git_sha: str | None = None,
    n_jobs: int = -1,
    resume: bool = False,
    resume_run_id: str | None = None,
    bench: bool = True,
    bench_only: bool = False,
    prefer_latest: bool = True,
    eval_tail_bars: int | None = None,
) -> SimLiveResult:
    """Run the simulated-live book end to end and write the receipt."""
    taus = np.asarray(DEFAULT_TAUS if taus is None else taus, dtype=float)
    slots = [champion, *(challengers or [])]

    def _member_specs(spec: str) -> list[str]:
        # Composite specs (parens avoid the ':' separator in CLI syntax):
        #   vincent(a+b+...) — element-wise mean of member quantile panels
        #     (Vincentization: the arena-proven ensemble primitive).
        #   agree(a,b) — emit a's row only when sign(mean(a)) == sign(mean(b));
        #     disagreement emits a degenerate near-flat row (name goes flat).
        if spec.startswith("vincent(") and spec.endswith(")"):
            return spec[len("vincent("):-1].split("+")
        if spec.startswith("agree(") and spec.endswith(")"):
            return spec[len("agree("):-1].split(",")
        return [spec]

    specs = sorted({m for slot in slots for m in _member_specs(slot.spec)})

    bar_files: dict[str, str] = {}
    for sym in symbols:
        sym_l = sym.lower()
        for suffix in (f"{sym_l}_{interval}_deep.parquet", f"{sym_l}_{interval}.parquet"):
            p = bars_root / suffix
            if p.is_file():
                bar_files[sym.upper()] = _sha256(p)
                break

    bars = load_deep_bars(bars_root, symbols, interval)
    if "close_total_return" not in bars.columns:
        # Spot crypto: no dividends/splits → total-return close equals raw
        # close. run_backtest selects the column; the paper loop prefers it.
        bars = bars.with_columns(pl.col("close").alias("close_total_return"))
    if tail_bars is not None:
        # Trailing window per asset (causal cut: signals still only see <= t).
        bars = bars.filter(
            pl.int_range(pl.len()).over("security_id")
            >= (pl.len().over("security_id") - int(tail_bars))
        )

    cache_dir = cache_dir or (out_dir / "qpanel_cache")
    # Per-asset close series (event_time order) + union panel panels per spec.
    per_sid: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sid in sorted(bars["security_id"].unique().to_list()):
        sub = bars.filter(pl.col("security_id") == sid).sort("event_time")
        closes = sub["close"].to_numpy().astype(float)
        times = sub["event_time"].to_list()
        per_sid[str(sid)] = (closes, np.asarray(times))

    quantile_cache: dict[tuple[str, str], tuple[np.ndarray, dict[str, int]]] = {}
    cache_digests: dict[str, str] = {}
    jobs = [(sid, spec) for sid in per_sid for spec in specs]
    # Per-(asset, spec) panels are independent — parallelize with joblib;
    # results are written to a deterministic cache so ordering is irrelevant.
    try:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_quantile_panel_cached)(
                cache_dir=cache_dir,
                sid=sid,
                spec=spec,
                window=window,
                taus=taus,
                closes=per_sid[sid][0],
                min_history=min_history,
            )
            for sid, spec in jobs
        )
    except Exception:
        results = [
            _quantile_panel_cached(
                cache_dir=cache_dir,
                sid=sid,
                spec=spec,
                window=window,
                taus=taus,
                closes=per_sid[sid][0],
                min_history=min_history,
            )
            for sid, spec in jobs
        ]
    for (sid, spec), (panel, fit_stats, digest) in zip(jobs, results, strict=True):
        quantile_cache[(sid, spec)] = (panel, fit_stats)
        cache_digests[f"{sid}:{spec}"] = digest

    def _panel_for(slot: StrategySlot) -> pl.DataFrame:
        members = _member_specs(slot.spec)
        panels: dict[str, np.ndarray] = {}
        for sid in per_sid:
            member_panels = [quantile_cache[(sid, m)][0] for m in members]
            if slot.spec.startswith("agree(") and len(member_panels) == 2:
                primary, confirmer = member_panels
                flat_row = np.linspace(-1e-6, 1e-6, taus.size)  # mu~0 -> w=0
                ok_p = np.isfinite(primary).all(axis=1)
                ok_c = np.isfinite(confirmer).all(axis=1)
                mu_p = np.full(primary.shape[0], np.nan)
                mu_c = np.full(confirmer.shape[0], np.nan)
                mu_p[ok_p] = primary[ok_p].mean(axis=1)
                mu_c[ok_c] = confirmer[ok_c].mean(axis=1)
                ok = ok_p & ok_c
                keep = ok & (np.sign(mu_p) == np.sign(mu_c))
                panels[sid] = np.where(keep[:, None], primary, flat_row[None, :])
                panels[sid][~ok] = np.nan
            elif len(member_panels) == 1:
                panels[sid] = member_panels[0]
            else:
                # Vincentize: row-wise mean over member quantile rows. Rows
                # with any NaN member stay NaN (fail-closed, no partial blend).
                stacked = np.stack(member_panels, axis=0)
                panels[sid] = np.where(
                    np.isfinite(stacked).all(axis=0), stacked.mean(axis=0), np.nan
                )
        times = {sid: per_sid[sid][1] for sid in per_sid}
        return quantile_panels_to_weights(panels, times, slot.policy, taus)

    champion_w = _panel_for(champion)
    challenger_w = {slot.name: _panel_for(slot) for slot in (challengers or [])}

    if eval_tail_bars is not None:
        # Out-of-sample slice: panels were computed on full history (causal,
        # so tail rows are unaffected); the loop/bench sees only the last N
        # shared dates and starts flat at the cut.
        tail_cut = sorted(bars["event_time"].unique().to_list())[-int(eval_tail_bars)]
        bars = bars.filter(pl.col("event_time") >= tail_cut)
        champion_w = champion_w.filter(pl.col("event_time") >= tail_cut)
        challenger_w = {
            name: w.filter(pl.col("event_time") >= tail_cut)
            for name, w in challenger_w.items()
        }

    result: PaperLoopResult | None = None
    if not bench_only:
        result = run_paper_loop(
            bars,
            config,
            champion_weights=champion_w,
            challenger_weights=challenger_w or None,
            initial_nav=float(initial_nav or config.paper.initial_nav),
            run_id=run_id,
            max_steps=max_steps,
            resume=resume,
            resume_run_id=resume_run_id,
            prefer_latest=prefer_latest,
        )
    effective_run_id = result.run_id if result is not None else (
        run_id or f"bench-{interval}-{champion.name}"
    )

    stats: dict[str, Any] = {}
    if result is not None:
        stats = _equity_stats(result.champion_equity, _bars_per_year(interval))
    book_stats: dict[str, Any] = {}
    if bench or bench_only:
        # Honest per-slot equity comparison on identical fill semantics:
        # every slot's weight panel through the reference engine (the paper
        # loop's shadow slots record intent only — no capital). ~seconds per
        # slot, deterministic, same costs/risk gate as the champion book.
        from quant_fund.backtest.engine import run_backtest

        for slot in slots:
            panel = champion_w if slot.name == champion.name else challenger_w[slot.name]
            try:
                bench_res = run_backtest(
                    bars, panel, config,
                    initial_nav=float(initial_nav or config.paper.initial_nav),
                )
                book_stats[slot.name] = {
                    **_equity_stats(bench_res.equity, _bars_per_year(interval)),
                    "n_fills": int(bench_res.fills.height),
                }
            except Exception as exc:  # noqa: BLE001 — bench failure is reported, not hidden
                book_stats[slot.name] = {"status": "bench_failed", "error": f"{type(exc).__name__}: {exc}"}
    if not stats and champion.name in book_stats:
        stats = book_stats[champion.name]
    forecaster_stats = {
        f"{sid}:{spec}": quantile_cache[(sid, spec)][1] for sid in per_sid for spec in specs
    }
    data_label = (
        result.source_note if result is not None
        else str(getattr(config.data.source, "value", config.data.source))
    )
    receipt: dict[str, Any] = {
        "kind": "sim_live_receipt" if result is not None else "sim_live_bench_receipt",
        "run_id": effective_run_id,
        "live_pnl_claim": False,
        "simulated_only": True,
        "research_only": True,
        "bench_only": bench_only,
        "data_label": data_label,
        "git_sha": git_sha,
        "bars": {
            "root": str(bars_root),
            "interval": interval,
            "symbols": sorted(symbols),
            "sha256": bar_files,
            "tail_bars": tail_bars,
            "eval_tail_bars": eval_tail_bars,
        },
        "strategy": {
            "champion": {"name": champion.name, "spec": champion.spec, "policy": asdict(champion.policy)},
            "challengers": [
                {"name": s.name, "spec": s.spec, "policy": asdict(s.policy)} for s in (challengers or [])
            ],
            "taus": [float(t) for t in taus],
            "window": int(window),
            "min_history": min_history,
        },
        "forecaster_stats": forecaster_stats,
        "quantile_cache_digests": cache_digests,
        "loop_metrics": (
            {
                "n_steps": result.metrics.get("n_steps"),
                "n_steps_this_run": result.metrics.get("n_steps_this_run"),
                "n_fills": result.metrics.get("n_fills"),
                "risk_gate_rejects": result.metrics.get("risk_gate_rejects"),
                "kill_switch_halts": result.metrics.get("kill_switch_halts"),
                "resumed": result.metrics.get("resumed"),
            }
            if result is not None else {"status": "bench_only_no_paper_loop"}
        ),
        "champion_equity_stats": stats,
        "book_stats": book_stats,
        "divergence": result.divergence if result is not None else None,
        "promotion_dry_run": result.promotion if result is not None else None,
        "conventions": [
            "signal at decision bar t uses only closes <= t (causal)",
            "fills at next bar open (config.execution.fill)",
            "forecast horizon = next close-close; fill at next open (disclosed mismatch)",
            "absent name on a present date = flatten (engine sparse-panel semantics)",
            "spot book only: no funding, borrow, or perp legs modeled",
            "close_total_return := close (spot crypto has no corporate actions)",
        ],
        "paths": result.paths if result is not None else {},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = out_dir / f"sim_live_{effective_run_id}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, default=str))
    return SimLiveResult(
        run_id=effective_run_id,
        receipt_path=receipt_path,
        receipt=receipt,
        loop=result,
    )
