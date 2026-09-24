"""Costed research replay of a variable mix of five perpetual signal sleeves.

Each sleeve is simulated on the complete historical clock. The allocator sees
only net sleeve NAV through the completed decision bar; the combined book then
executes its full target snapshot at the next open. The historical tail has
already been examined in this research program, so it is exploratory evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from quant_fund.backtest.adaptive_mix import (  # noqa: E402
    banded_targets,
    dense_targets,
    mix_targets,
    trailing_nav_allocations,
)
from quant_fund.backtest.perp_engine import (  # noqa: E402
    infer_periods_per_year,
    run_perp_backtest,
)
from quant_fund.backtest.sleeves import (  # noqa: E402
    cross_sectional_momentum_weights,
    funding_carry_weights,
    funding_spike_fade_weights,
    slow_trend_weights,
    sweep_reclaim_weights,
)
from quant_fund.config.loader import load_config  # noqa: E402
from quant_fund.metrics.returns import cagr, max_drawdown, sharpe_ratio  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _segments(equity: pl.DataFrame, cut: datetime, ppy: float) -> dict:
    frame = equity.sort("event_time").with_columns(pl.col("nav").pct_change().alias("return"))
    out = {}
    for label, predicate in (
        ("development", pl.col("event_time") < cut),
        ("historical_tail_exploratory", pl.col("event_time") >= cut),
    ):
        r = frame.filter(predicate)["return"].drop_nulls().to_numpy().astype(float)
        out[label] = {
            "n": len(r),
            "net_return": float(np.prod(1.0 + r) - 1.0),
            "cagr": cagr(r, periods_per_year=ppy),
            "sharpe": sharpe_ratio(r, periods_per_year=ppy)["sharpe"],
            "max_drawdown": max_drawdown(r),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=_REPO / "configs/research.yaml")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--min-obs", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument(
        "--band-search",
        action="store_true",
        help="Compare predeclared target-change bands on development data only",
    )
    args = parser.parse_args()
    if not 0 < args.holdout_frac < 0.5:
        raise ValueError("holdout fraction must be between 0 and 0.5")
    sources = args.data_root / "raw" / "sources"
    manifest_path = args.manifest or args.data_root / "collection_manifest.json"
    if args.manifest is None and not manifest_path.exists():
        manifest_path = sources / "collection_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    bar_paths = sorted(sources.glob("*_1d.perp.parquet"))
    funding_paths = sorted(sources.glob("*.funding.parquet"))
    if len(bar_paths) < 5 or not funding_paths:
        raise ValueError("need at least five perp assets with funding")
    bar_ids = {path.name.split("_")[0] for path in bar_paths}
    funding_paths = [p for p in funding_paths if p.name.split(".")[0] in bar_ids]
    if {p.name.split(".")[0] for p in funding_paths} != bar_ids:
        raise ValueError("every perp asset needs funding history")
    input_hashes = {p.name: _sha256(p) for p in [*bar_paths, *funding_paths]}
    for name, digest in input_hashes.items():
        if manifest.get("files", {}).get(name, {}).get("sha256") != digest:
            raise ValueError(f"manifest hash mismatch: {name}")
    bars = pl.concat([pl.read_parquet(p) for p in bar_paths]).sort(["security_id", "event_time"])
    funding = pl.concat([pl.read_parquet(p) for p in funding_paths]).sort(
        ["security_id", "event_time"]
    )
    if set(bars["source"].unique().to_list()) != {"binance_usdtm_perp"}:
        raise ValueError("perp panel source is not Binance")
    if set(funding["source"].unique().to_list()) != {"binance_funding_rate"}:
        raise ValueError("funding panel source is not Binance")
    if bars.filter(pl.col("event_time") > pl.col("available_time")).height:
        raise ValueError("bar availability precedes event time")
    if funding.filter(pl.col("event_time") > pl.col("available_time")).height:
        raise ValueError("funding availability precedes event time")
    config = load_config(args.config)
    if config.costs.frictionless:
        raise ValueError("frictionless configuration is not admissible")
    proposals = {
        "carry": funding_carry_weights(
            bars, funding, lookback_events=24, max_name=0.1, gross_scale=0.5, vol_window=96
        ),
        "fade": funding_spike_fade_weights(
            bars,
            funding,
            lookback_events=30,
            z_threshold=2.0,
            max_name=0.05,
            gross_scale=0.5,
            vol_window=48,
        ),
        "momentum": cross_sectional_momentum_weights(
            bars,
            lookback_bars=168,
            skip_bars=4,
            max_name=0.05,
            gross_scale=0.5,
            vol_window=48,
        ),
        "sweep": sweep_reclaim_weights(
            bars, lookback=24, hold_bars=8, max_name=0.05, gross_scale=0.5
        ),
        "trend": slow_trend_weights(
            bars,
            fast_bars=168,
            slow_bars=720,
            max_name=0.05,
            gross_scale=0.5,
            vol_window=48,
        ),
    }
    targets = {name: dense_targets(bars, p) for name, p in proposals.items()}
    paper = {}
    for name, panel in targets.items():
        paper[name] = run_perp_backtest(bars, funding, panel, config).equity
        print(f"paper sleeve {name} complete", flush=True)
    allocation = trailing_nav_allocations(
        paper, window=args.window, min_obs=args.min_obs, temperature=args.temperature
    )
    equal = allocation.select("event_time").with_columns(
        [pl.lit(1.0 / len(targets)).alias(name) for name in sorted(targets)]
    )
    dates = paper["carry"]["event_time"].to_list()
    cut = dates[int(len(dates) * (1.0 - args.holdout_frac))]
    ppy = infer_periods_per_year(dates)
    if args.band_search:
        base = mix_targets(targets, allocation)
        dev_bars = bars.filter(pl.col("event_time") < cut)
        dev_funding = funding.filter(pl.col("event_time") < cut)
        candidates = []
        for band in (0.0, 0.0025, 0.005, 0.01, 0.02):
            requests = banded_targets(base, band).filter(pl.col("event_time") < cut)
            run = run_perp_backtest(dev_bars, dev_funding, requests, config)
            metrics = _segments(run.equity, cut, ppy)["development"]
            eligible = bool(
                np.isfinite(metrics["sharpe"])
                and metrics["sharpe"] > 0
                and metrics["cagr"] > 0
                and metrics["max_drawdown"] > -0.05
            )
            candidates.append(
                {
                    "band": band,
                    "development": metrics,
                    "eligible": eligible,
                    "n_target_requests": requests.height,
                    "risk_gate_rejects": run.metrics["risk_gate_rejects"],
                    "costs": {k: run.metrics[k] for k in ("commission", "spread", "impact")},
                }
            )
            print(f"development band {band} complete", flush=True)
        admissible = [c for c in candidates if c["eligible"]]
        best = max(admissible, key=lambda c: c["development"]["sharpe"], default=None)
        receipt = {
            "schema": "adaptive_mix_band_search.v1",
            "created_at": datetime.now(tz=UTC).isoformat(),
            "evidence_level": "historical_development_only",
            "research_only": True,
            "live_pnl_claim": False,
            "n_assets": len(bar_ids),
            "holdout_cut": str(cut),
            "selection_rule": "positive net CAGR and Sharpe, max drawdown under 5%; "
            "then highest development Sharpe",
            "candidates": candidates,
            "selected_band": None if best is None else best["band"],
            "input_hashes": input_hashes,
            "manifest_sha256": _sha256(manifest_path),
            "config_sha256": _sha256(args.config),
            "script_sha256": _sha256(Path(__file__)),
            "metrics_sha256": _sha256(_REPO / "src/quant_fund/metrics/returns.py"),
            "engine_sha256": _sha256(_REPO / "src/quant_fund/backtest/perp_engine.py"),
            "allocator_sha256": _sha256(_REPO / "src/quant_fund/backtest/adaptive_mix.py"),
        }
        if best is not None:
            selected = banded_targets(base, best["band"])
            full = run_perp_backtest(bars, funding, selected, config)
            receipt["selected_full_replay_exploratory"] = {
                "segments": _segments(full.equity, cut, ppy),
                "risk_gate_rejects": full.metrics["risk_gate_rejects"],
            }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(receipt, indent=2, default=str))
        print(f"receipt -> {args.out}")
        return 0
    paper_diagnostics = {name: _segments(equity, cut, ppy) for name, equity in paper.items()}
    allocation_means = {}
    for label, predicate in (
        ("development", pl.col("event_time") < cut),
        ("historical_tail_exploratory", pl.col("event_time") >= cut),
    ):
        frame = allocation.filter(predicate)
        allocation_means[label] = {name: float(frame[name].mean()) for name in sorted(targets)}
    results = {}
    for name, alpha in (("adaptive", allocation), ("equal", equal)):
        mixed = mix_targets(targets, alpha)
        result = run_perp_backtest(bars, funding, mixed, config)
        ppy = float(result.metrics["periods_per_year"])
        results[name] = {
            "segments": _segments(result.equity, cut, ppy),
            "full_path_risk_gate_rejects": result.metrics["risk_gate_rejects"],
            "full_path_liquidations": result.metrics["liquidation_count"],
            "full_path_costs": {k: result.metrics[k] for k in ("commission", "spread", "impact")},
        }
        print(f"combined {name} complete", flush=True)
    receipt = {
        "schema": "adaptive_mix_replay.v1",
        "created_at": datetime.now(tz=UTC).isoformat(),
        "evidence_level": "historical_exploratory_tail_reused",
        "research_only": True,
        "live_pnl_claim": False,
        "data_source": "binance_usdtm_perp",
        "n_assets": len(bar_ids),
        "bar_count": bars.height,
        "holdout_cut": str(cut),
        "allocator": {
            "window": args.window,
            "min_obs": args.min_obs,
            "temperature": args.temperature,
            "equal_anchor": 0.2,
            "input": "costed sleeve NAV through completed close",
            "fill": "following bar open",
        },
        "results": results,
        "paper_sleeves": paper_diagnostics,
        "mean_adaptive_allocation": allocation_means,
        "input_hashes": input_hashes,
        "manifest_sha256": _sha256(manifest_path),
        "config_sha256": _sha256(args.config),
        "script_sha256": _sha256(Path(__file__)),
        "metrics_sha256": _sha256(_REPO / "src/quant_fund/metrics/returns.py"),
        "engine_sha256": _sha256(_REPO / "src/quant_fund/backtest/perp_engine.py"),
        "allocator_sha256": _sha256(_REPO / "src/quant_fund/backtest/adaptive_mix.py"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, default=str))
    print(f"receipt -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
