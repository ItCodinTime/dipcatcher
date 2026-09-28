"""ULTRAPLAN P5.1 — multi-sleeve dev study on a seeded SYNTHETIC perp book.

One synthetic book with planted structure (persistent-sign funding for the
carry sleeve, drift regimes for the trend sleeve, liquidity-sweep price
paths for the reversal sleeve) is run four ways through
``run_perp_backtest``:

1. each sleeve alone — ``funding_carry``, ``slow_trend``, ``sweep_reclaim``
   (the repo's dipcatch reversal; a true BTC-factor-residual mean-reversion
   sleeve is not in the tree — documented scope, not substituted silently);
2. a trailing-NAV risk-parity mix — per-sleeve equity paths feed
   ``trailing_nav_allocations`` (delay-1 trailing t-scores with an
   equal-weight anchor) and ``mix_targets`` recombines the target frames;
3. the mix under a chained ``VolTargetScaler`` + ``DrawdownGovernor``
   overlay, with a recording wrapper capturing the per-bar scale factor.

Metrics are telemetry only — turnover, decomposed cost, gross/net exposure,
fill counts, overlay scale-factor dispersion, allocation concentration
(HHI). No return, Sharpe, drawdown-as-performance or NAV claims; the book
is SYNTHETIC and ``live_pnl_claim`` stays False.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from quant_fund.backtest.adaptive_mix import dense_targets, mix_targets, trailing_nav_allocations
from quant_fund.backtest.engine import BacktestResult
from quant_fund.backtest.overlay import (
    CompositeScaler,
    DrawdownGovernor,
    VolTargetScaler,
)
from quant_fund.backtest.perp_engine import run_perp_backtest
from quant_fund.backtest.sleeves import (
    funding_carry_weights,
    slow_trend_weights,
    sweep_reclaim_weights,
)
from quant_fund.config.models import AppConfig
from quant_fund.research.fleet_eval import _atomic_write_text
from quant_fund.research.receipt_v2 import build_receipt_v2, seal_receipt

SLEEVE_STUDY_SCHEMA = "sleeve_study_eval.v1"
SLEEVES = ("funding_carry", "slow_trend", "sweep_reclaim")


def _synth_perp_book(seed: int, n_names: int, n_bars: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Seeded OHLCV + 8h funding panel.

    Structure planted for the three sleeves: each name gets a persistent
    funding sign (carry signal), half the names trend for a contiguous
    segment (tsmom signal), and occasional single-bar extreme wicks create
    sweep-reclaim setups.
    """
    rng = np.random.default_rng(seed)
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    rows: list[dict[str, Any]] = []
    fund_rows: list[dict[str, Any]] = []
    for i, sid in enumerate(f"S{j:02d}" for j in range(n_names)):
        base = float(np.exp(rng.normal(np.log(50.0), 0.4)))
        ret = rng.normal(0.0, 0.012, size=n_bars)
        # Planted trend segment in the middle third for even/odd names.
        if i % 2 == 0:
            ret[n_bars // 3 : 2 * n_bars // 3] += 0.004
        else:
            ret[n_bars // 3 : 2 * n_bars // 3] -= 0.003
        close = base * np.exp(np.cumsum(ret))
        open_ = np.concatenate([[base], close[:-1]])
        spread = np.abs(rng.normal(0.0, 0.006, size=n_bars)) * open_
        high = np.maximum(open_, close) + spread
        low = np.maximum(0.01, np.minimum(open_, close) - spread)
        # Isolated extreme wicks: sweep of a local extreme that closes back.
        for wick_t in rng.choice(np.arange(10, n_bars - 2), size=4, replace=False):
            if rng.random() < 0.5:
                low[wick_t] = min(low[wick_t], 0.94 * np.min(low[max(0, wick_t - 20) : wick_t]))
                close[wick_t] = max(
                    low[wick_t] * 1.05, min(open_[wick_t], high[wick_t] - 0.4 * spread[wick_t])
                )
            else:
                high[wick_t] = max(high[wick_t], 1.06 * np.max(high[max(0, wick_t - 20) : wick_t]))
                close[wick_t] = min(
                    high[wick_t] * 0.95, max(open_[wick_t], low[wick_t] + 0.4 * spread[wick_t])
                )
        vol_shares = np.exp(rng.normal(np.log(2e5), 0.5, size=n_bars))
        # Persistent-sign funding: name-level rate with small noise.
        rate = float(rng.choice([-1.0, 1.0])) * float(np.exp(rng.normal(np.log(3e-4), 0.3)))
        for t in range(n_bars):
            rows.append(
                {
                    "event_time": t0 + timedelta(hours=t),
                    "security_id": sid,
                    "open": float(open_[t]),
                    "high": float(high[t]),
                    "low": float(low[t]),
                    "close": float(close[t]),
                    "close_total_return": float(close[t]),
                    "volume": float(vol_shares[t]),
                    "adv": float(vol_shares[t] * close[t]),
                    "vol_20": 0.02,
                    "source": "synthetic",
                }
            )
            if t % 8 == 0:  # standard funding grid
                fund_rows.append(
                    {
                        "event_time": t0 + timedelta(hours=t),
                        "security_id": sid,
                        "value": rate + float(rng.normal(0.0, abs(rate) * 0.1)),
                    }
                )
    return (
        pl.DataFrame(rows).sort(["event_time", "security_id"]),
        pl.DataFrame(fund_rows).sort(["event_time", "security_id"]),
    )


def _sleeve_targets(name: str, bars: pl.DataFrame, funding: pl.DataFrame) -> pl.DataFrame:
    # ``max_name`` sits below the engine's risk-gate name cap (0.03) so the
    # study measures sleeve behaviour, not reject rate.
    cap = 0.025
    if name == "funding_carry":
        return funding_carry_weights(bars, funding, max_name=cap)
    if name == "slow_trend":
        return slow_trend_weights(bars, fast_bars=24, slow_bars=72, max_name=cap)
    if name == "sweep_reclaim":
        return sweep_reclaim_weights(bars, lookback=16, hold_bars=6, max_name=cap)
    raise ValueError(f"unknown sleeve {name!r}")


class _RecordingScaler:
    """Overlay scaler wrapper: delegates to the chained scalers and logs the
    product of their per-bar factors at each ``scale`` call."""

    def __init__(self, scalers: list[Any]) -> None:
        self._inner = CompositeScaler(scalers)
        self._components = list(scalers)
        self.factors: dict[datetime, float] = {}

    def observe(self, dt: datetime, nav: float) -> None:
        self._inner.observe(dt, nav)

    def scale(self, dt: datetime, targets: dict[str, float]) -> dict[str, float]:
        out = self._inner.scale(dt, targets)
        factors = [float(s.factor()) for s in self._components]
        self.factors[dt] = float(np.prod(factors)) if factors else 1.0
        return out


def _run_metrics(result: BacktestResult) -> dict[str, float | int]:
    eq = result.equity
    m = result.metrics
    turns = eq["turnover"].to_numpy() if "turnover" in eq.columns else np.zeros(1)
    gross = eq["gross"].to_numpy() if "gross" in eq.columns else np.zeros(1)

    def _f(key: str) -> float:
        v = m.get(key)
        return float(v) if isinstance(v, (int, float)) and np.isfinite(float(v)) else 0.0

    return {
        "n_fills": int(result.fills.height),
        "turnover": float(np.sum(turns)),
        "gross_exposure_mean": float(np.mean(gross)) if gross.size else 0.0,
        "cost_total": sum(_f(k) for k in ("commission", "spread", "impact", "turnover_bps_cost")),
        "risk_gate_rejects": int(_f("risk_gate_rejects")),
        "cash_rejects": int(_f("cash_rejects")),
    }


def run_sleeve_study(
    bars: pl.DataFrame,
    funding: pl.DataFrame,
    *,
    seed: int = 0,
    alloc_window: int = 20,
    alloc_min_obs: int = 10,
    vol_target: float = 0.15,
    dd_threshold: float = 0.25,
    initial_nav: float = 1_000_000.0,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Run sleeves solo, then parity-mixed, then overlaid; emit receipt.v2."""
    if not 2 <= alloc_min_obs <= alloc_window:
        raise ValueError("need 2 <= alloc_min_obs <= alloc_window")
    cfg = AppConfig()
    rows: list[dict[str, Any]] = []
    targets_map: dict[str, pl.DataFrame] = {}
    equities: dict[str, pl.DataFrame] = {}
    for name in SLEEVES:
        targets = dense_targets(bars, _sleeve_targets(name, bars, funding))
        targets_map[name] = targets
        res = run_perp_backtest(bars, funding, targets, cfg, initial_nav=initial_nav)
        equities[name] = res.equity
        rows.append({"lane": f"solo:{name}", **_run_metrics(res)})
    # Trailing-NAV parity mix: allocations decided on completed closes only.
    allocs = trailing_nav_allocations(equities, window=alloc_window, min_obs=alloc_min_obs)
    mixed = mix_targets(targets_map, allocs)
    mix_res = run_perp_backtest(bars, funding, mixed, cfg, initial_nav=initial_nav)
    alloc_arr = allocs.select(sorted(SLEEVES)).to_numpy().astype(float)
    hhi = float(np.mean(np.sum(alloc_arr**2, axis=1)))
    rows.append(
        {
            "lane": "parity_mix",
            "alloc_hhi": hhi,
            **_run_metrics(mix_res),
        }
    )
    # Overlaid mix: vol-target then drawdown governor, factor path recorded.
    recorder = _RecordingScaler(
        [
            VolTargetScaler(target_ann_vol=vol_target, window=24),
            DrawdownGovernor(dd_soft=dd_threshold / 2.0, dd_hard=dd_threshold),
        ]
    )
    ov_res = run_perp_backtest(bars, funding, mixed, cfg, initial_nav=initial_nav, scaler=recorder)
    factors = np.asarray(list(recorder.factors.values()), dtype=float)
    rows.append(
        {
            "lane": "parity_mix+overlay",
            "overlay_scale_mean": float(np.mean(factors)) if factors.size else 1.0,
            "overlay_scale_min": float(np.min(factors)) if factors.size else 1.0,
            "overlay_cut_frac": (float(np.mean(factors < 1.0 - 1e-12)) if factors.size else 0.0),
            "alloc_hhi": hhi,
            **_run_metrics(ov_res),
        }
    )
    frame = pl.DataFrame(rows)
    inputs_sha = _hash_inputs(bars, funding)
    receipt = build_receipt_v2(
        kind="sleeve_study_eval",
        data_label="SYNTHETIC",
        dataset={
            "schema": SLEEVE_STUDY_SCHEMA,
            "inputs_sha256": inputs_sha,
            "n_bars": bars.height,
            "n_names": bars["security_id"].n_unique(),
            "n_funding_events": funding.height,
        },
        params={
            "seed": int(seed),
            "alloc_window": int(alloc_window),
            "alloc_min_obs": int(alloc_min_obs),
            "vol_target": float(vol_target),
            "dd_threshold": float(dd_threshold),
            "initial_nav": float(initial_nav),
            "sleeves": list(SLEEVES),
            "reversal_sleeve": "sweep_reclaim",
            "scope_note": "Kakushadze residual-MR sleeve not in tree; "
            "sweep-reclaim is the repo's reversal sleeve.",
        },
        code_files=(Path(__file__),),
        verdict="pass",
        payload={"n_rows": len(rows), "results": rows},
    )
    return frame, receipt


def _hash_inputs(bars: pl.DataFrame, funding: pl.DataFrame) -> str:
    from quant_fund.utils.hashing import hash_bytes

    return hash_bytes(
        bars.sort(["event_time", "security_id"]).write_csv().encode()
        + funding.sort(["event_time", "security_id"]).write_csv().encode()
    )


def write_sleeve_study_receipt(
    receipt: Mapping[str, Any],
    receipts_dir: Path | str = Path("receipts"),
) -> Path:
    """Seal a sleeve-study receipt to ``receipts/sleeve_study_<hash>.json``."""
    if receipt.get("kind") != "sleeve_study_eval" or receipt.get("data_label") != "SYNTHETIC":
        raise ValueError("sleeve-study receipt violates the honesty contract")
    sealed = seal_receipt(receipt)
    digest = sealed["receipt_sha256"]
    path = Path(receipts_dir) / f"sleeve_study_{digest[:16]}.json"
    _atomic_write_text(path, json.dumps(sealed, indent=2, sort_keys=True) + "\n")
    return path


def format_sleeve_study_table(frame: pl.DataFrame) -> str:
    cols = ["lane", "n_fills", "turnover", "gross_exposure_mean", "cost_total"]
    extra = ["alloc_hhi", "overlay_scale_mean", "overlay_cut_frac"]
    header = " | ".join(cols + extra)
    lines = [header, "-+-".join("-" * len(c) for c in cols + extra)]
    for row in frame.iter_rows(named=True):
        cells = [f"{row['lane']:>20}"]
        cells += [
            f"{row['n_fills']:>7}",
            f"{row['turnover']:>8.3f}",
            f"{row['gross_exposure_mean']:>18.3f}",
            f"{row['cost_total']:>10.1f}",
        ]
        cells += [
            f"{row.get('alloc_hhi') or 0:>8.3f}",
            f"{row.get('overlay_scale_mean') or 1:>17.3f}",
            f"{row.get('overlay_cut_frac') or 0:>15.3f}",
        ]
        lines.append(" | ".join(cells))
    return "\n".join(lines)


__all__ = [
    "SLEEVE_STUDY_SCHEMA",
    "SLEEVES",
    "run_sleeve_study",
    "write_sleeve_study_receipt",
    "format_sleeve_study_table",
]
