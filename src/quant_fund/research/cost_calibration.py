"""Flat vs OHLC-calibrated cost trials (dev-only SYNTHETIC diagnostic).

Runs matched target-weight books under the flat half-spread floor and under
each opt-in OHLC estimator (Corwin–Schultz, Abdi–Ranaldo, Roll). Reports
decomposed cost totals only — never Sharpe / P&L / NAV claims. Evidence class
is SYNTHETIC correctness / sensitivity, not market evidence.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from quant_fund.backtest.engine import run_backtest
from quant_fund.config import load_config
from quant_fund.config.models import AppConfig, CostConfig, ExecutionConfig, FillConvention
from quant_fund.execution.spread_calibration import (
    CALIBRATED_SPREAD_ESTIMATORS,
    SPREAD_ESTIMATORS,
    corwin_schultz_relative_spread,
    floored_half_spread_bps,
    relative_to_half_spread_bps,
)
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent
from quant_fund.research.fleet_eval import _atomic_write_text
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

COST_CALIBRATION_SCHEMA = "cost_calibration.v1"


def _synthetic_ohlc_panel(
    *,
    n_dates: int = 40,
    n_names: int = 4,
    seed: int = 7,
    planted_rel_spread: float = 0.002,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build a SYNTHETIC OHLC + target-weight panel with a planted high-low range.

    Each bar's high/low is centered on close with relative half-range
    ``planted_rel_spread / 2`` so Corwin–Schultz recovers a positive estimate
    well above a tight flat floor in the trial grid.
    """
    if n_dates < 8 or n_names < 2:
        raise ValueError("need >= 8 dates and >= 2 names")
    if not np.isfinite(planted_rel_spread) or planted_rel_spread <= 0:
        raise ValueError("planted_rel_spread must be finite and positive")
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 2, tzinfo=UTC)
    rows: list[dict[str, Any]] = []
    sids = [f"S{i}" for i in range(n_names)]
    half = planted_rel_spread / 2.0
    for t in range(n_dates):
        et = start + timedelta(days=t)
        for j, sid in enumerate(sids):
            close = 100.0 + j + 0.05 * t + float(rng.normal(0.0, 0.02))
            open_px = close * (1.0 + float(rng.normal(0.0, 0.0005)))
            high = close * (1.0 + half)
            low = close * (1.0 - half)
            rows.append(
                {
                    "event_time": et,
                    "security_id": sid,
                    "open": open_px,
                    "high": high,
                    "low": low,
                    "close": close,
                    "close_total_return": close,
                    "volume": 1_000_000.0,
                    "adv": close * 1_000_000.0,
                    "vol_20": 0.02,
                    "source": "SYNTHETIC",
                }
            )
    bars = pl.DataFrame(rows).sort(["event_time", "security_id"])
    # Equal-weight vs concentrated rebalances so every other day trades.
    w_rows: list[dict[str, Any]] = []
    for t in range(n_dates):
        et = start + timedelta(days=t)
        if (t % 2) == 0:
            targets = {sid: 1.0 / n_names for sid in sids}
        else:
            targets = {sid: 0.05 / (n_names - 1) for sid in sids}
            targets[sids[0]] = 0.95
        for sid, tw in targets.items():
            w_rows.append({"event_time": et, "security_id": sid, "target_weight": tw})
    weights = pl.DataFrame(w_rows).sort(["event_time", "security_id"])
    return bars, weights


def _base_config(*, half_spread_bps: float = 1.0) -> AppConfig:
    cfg = load_config("configs/research.yaml")
    cfg.execution = ExecutionConfig(fill=FillConvention.NEXT_OPEN, allow_close_auction=False)
    cfg.costs = CostConfig(
        commission_bps=1.0,
        half_spread_bps=half_spread_bps,
        impact_y=0.0,
        bps_per_turnover=0.0,
        borrow_bps_per_year=0.0,
        financing_bps_per_year=0.0,
        frictionless=False,
        participation_limit=1.0,
        spread_estimator="flat",
        spread_calibration_lookback=20,
    )
    # Relax name / gross gates so the cost trial is not risk-gate dominated.
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_gross = 2.0
    cfg.risk_gate.max_net = 1.0
    cfg.risk_gate.max_participation = 1.0
    cfg.risk_gate.max_order_notional = 1e18
    return cfg


def trial_cost_summary(
    bars: pl.DataFrame,
    weights: pl.DataFrame,
    config: AppConfig,
    *,
    initial_nav: float = 1_000_000.0,
) -> dict[str, Any]:
    """Run one backtest and return decomposed cost totals (no return headlines)."""
    result = run_backtest(bars, weights, config, initial_nav=initial_nav, fast=False)

    def _metric(name: str) -> float:
        value = result.metrics.get(name, 0.0)
        return float(value) if isinstance(value, (int, float)) else 0.0

    commission = _metric("commission")
    spread = _metric("spread")
    impact = _metric("impact")
    turnover = _metric("turnover")
    return {
        "spread_estimator": config.costs.spread_estimator,
        "half_spread_bps_floor": float(config.costs.half_spread_bps),
        "commission": commission,
        "spread": spread,
        "impact": impact,
        "turnover": turnover,
        "total_cost": commission + spread + impact + turnover,
        "n_fills": int(result.fills.height),
        "data_source": str(result.metrics.get("data_source", "SYNTHETIC")),
        "research_only": True,
    }


def run_cost_calibration_trials(
    *,
    estimators: Sequence[str] | None = None,
    half_spread_bps: float = 1.0,
    lookback: int = 20,
    n_dates: int = 40,
    n_names: int = 4,
    seed: int = 7,
    planted_rel_spread: float = 0.002,
    initial_nav: float = 1_000_000.0,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Matched flat vs calibrated cost trials on a seeded SYNTHETIC book."""
    names = list(estimators) if estimators is not None else list(SPREAD_ESTIMATORS)
    for name in names:
        if name not in SPREAD_ESTIMATORS:
            raise ValueError(f"unknown spread estimator: {name}")
    bars, weights = _synthetic_ohlc_panel(
        n_dates=n_dates,
        n_names=n_names,
        seed=seed,
        planted_rel_spread=planted_rel_spread,
    )
    # Closed-form Corwin–Schultz reference on the planted equal-range book.
    sample_h = bars.filter(pl.col("security_id") == "S0")["high"].to_numpy()
    sample_l = bars.filter(pl.col("security_id") == "S0")["low"].to_numpy()
    cs_rel = corwin_schultz_relative_spread(sample_h, sample_l)
    cs_half_bps = relative_to_half_spread_bps(cs_rel)
    cs_effective = floored_half_spread_bps(half_spread_bps, cs_rel)

    rows: list[dict[str, Any]] = []
    for name in names:
        cfg = _base_config(half_spread_bps=half_spread_bps)
        cfg.costs.spread_estimator = name
        cfg.costs.spread_calibration_lookback = int(lookback)
        summary = trial_cost_summary(bars, weights, cfg, initial_nav=initial_nav)
        rows.append(summary)

    frame = pl.DataFrame(rows)
    receipt: dict[str, Any] = {
        "schema": COST_CALIBRATION_SCHEMA,
        "kind": "cost_calibration_eval",
        "data_label": "SYNTHETIC",
        "live_pnl_claim": False,
        "dev_only": True,
        "research_only": True,
        "claim": "execution_diagnostic_only",
        "generated_at": datetime.now(UTC).isoformat(),
        "git_revision": git_revision(),
        "seed": int(seed),
        "half_spread_bps_floor": float(half_spread_bps),
        "lookback": int(lookback),
        "planted_rel_spread": float(planted_rel_spread),
        "closed_form": {
            "corwin_schultz_relative": float(cs_rel) if np.isfinite(cs_rel) else None,
            "corwin_schultz_half_spread_bps": float(cs_half_bps)
            if np.isfinite(cs_half_bps)
            else None,
            "corwin_schultz_effective_half_spread_bps": float(cs_effective),
            "floor_binds": bool(np.isfinite(cs_half_bps) and cs_half_bps <= half_spread_bps),
        },
        "estimators": names,
        "results": rows,
        "inputs_sha256": hash_bytes(
            canonical_json_bytes(
                {
                    "seed": int(seed),
                    "n_dates": int(n_dates),
                    "n_names": int(n_names),
                    "half_spread_bps": float(half_spread_bps),
                    "lookback": int(lookback),
                    "planted_rel_spread": float(planted_rel_spread),
                    "estimators": names,
                }
            )
        ),
    }
    return frame, receipt


def write_cost_calibration_receipt(
    receipt: Mapping[str, Any],
    receipts_dir: Path | str = Path("receipts"),
) -> Path:
    research_blob = {k: v for k, v in receipt.items() if k != "live_pnl_claim"}
    if (
        receipt.get("schema") != COST_CALIBRATION_SCHEMA
        or receipt.get("kind") != "cost_calibration_eval"
        or receipt.get("data_label") != "SYNTHETIC"
        or receipt.get("live_pnl_claim") is not False
        or receipt.get("dev_only") is not True
        or not family_blob_forbidden_metrics_absent(research_blob)
    ):
        raise ValueError("cost calibration receipt violates the honesty contract")
    payload = dict(receipt)
    digest = hash_bytes(canonical_json_bytes(payload))
    payload["receipt_sha256"] = digest
    path = Path(receipts_dir) / f"cost_calibration_eval_{digest[:16]}.json"
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def format_cost_calibration_table(frame: pl.DataFrame) -> str:
    cols = [
        "spread_estimator",
        "half_spread_bps_floor",
        "commission",
        "spread",
        "total_cost",
        "n_fills",
    ]
    header = " | ".join(f"{c:>22}" for c in cols)
    lines = [header, "-+-".join("-" * 22 for _ in cols)]
    for row in frame.iter_rows(named=True):
        cells = [
            f"{row['spread_estimator']:>22}",
            f"{row['half_spread_bps_floor']:>22.2f}",
            f"{row['commission']:>22.4f}",
            f"{row['spread']:>22.4f}",
            f"{row['total_cost']:>22.4f}",
            f"{row['n_fills']:>22d}",
        ]
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def write_cost_calibration_report(
    frame: pl.DataFrame,
    receipt: Mapping[str, Any],
    path: Path | str = Path("reports/cost_calibration_flat_vs_ohlc.md"),
) -> Path:
    """Markdown report of flat vs calibrated trial costs (no return headlines)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    closed = receipt.get("closed_form", {})
    lines = [
        "# Cost calibration — flat half-spread vs OHLC estimators",
        "",
        "SYNTHETIC execution diagnostic only. Not market evidence. "
        "`live_pnl_claim=false`. Research-headline return ratios and equity "
        "curves are absent by construction.",
        "",
        f"- schema: `{receipt.get('schema')}`",
        f"- seed: `{receipt.get('seed')}`",
        f"- half-spread floor (bps): `{receipt.get('half_spread_bps_floor')}`",
        f"- planted relative full spread: `{receipt.get('planted_rel_spread')}`",
        f"- Corwin–Schultz closed-form relative: `{closed.get('corwin_schultz_relative')}`",
        f"- Corwin–Schultz effective half-spread bps (floored): "
        f"`{closed.get('corwin_schultz_effective_half_spread_bps')}`",
        "",
        "## Trial results",
        "",
        "```",
        format_cost_calibration_table(frame),
        "```",
        "",
        "## Notes",
        "",
        "- Default path is flat `half_spread_bps`; calibrated estimators are opt-in.",
        "- Calibrated cost is `max(floor, estimator_half_spread_bps)`.",
        "- `run_backtest_fast` refuses any non-flat `spread_estimator`.",
        f"- Calibrated estimators: {', '.join(CALIBRATED_SPREAD_ESTIMATORS)}.",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
