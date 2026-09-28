"""ULTRAPLAN P5.4 — execution-cost surface: maker-fill fee delta + hysteresis band robustness.

Two probes on seeded SYNTHETIC books, emitted as one sealed ``receipt.v2``:

* **maker_fill**: the same target-order stream is driven through
  ``SimulatedBroker`` twice — once as marketable (taker) orders, once as
  limit orders pegged at the decision close (filled only when a later bar's
  range touches the limit, at ``min(open, limit)`` / ``max(open, limit)``).
  Records fee-per-dollar-filled, maker fill rate and residual unfilled
  notional. Passive fills carry ``commission= maker_commission_bps`` and
  zero spread/impact — the disclosed limit-at-touch simplification.
* **hysteresis**: a causal churning target series is passed through
  ``banded_targets`` over a band grid and replayed with ``run_backtest``,
  reporting turnover, per-leg cost decomposition and the banded-vs-raw
  tracking RMSE. The surface is a development diagnostic — the band is not
  retuned on any holdout.

Honesty: all books are labeled SYNTHETIC; payload carries cost, turnover,
fill-rate and tracking metrics only — no return, Sharpe or NAV claims.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from quant_fund.backtest.adaptive_mix import banded_targets
from quant_fund.backtest.engine import run_backtest
from quant_fund.config.models import AppConfig, CostConfig
from quant_fund.execution.maker_taker_probe import maker_taker_fee_delta
from quant_fund.research.fleet_eval import _atomic_write_text
from quant_fund.research.receipt_v2 import build_receipt_v2, seal_receipt

COST_SURFACE_SCHEMA = "cost_surface_eval.v1"
DEFAULT_BANDS = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.04)
_EPS = 1e-12


def _synth_bars(seed: int, n_names: int, n_bars: int) -> pl.DataFrame:
    """Seeded OHLCV panel with ADV/vol — labeled source='synthetic'."""
    rng = np.random.default_rng(seed)
    base = np.exp(rng.normal(np.log(50.0), 0.4, size=n_names))
    drift = rng.normal(0.0, 0.004, size=n_names)
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    rows: list[dict[str, Any]] = []
    for i, sid in enumerate(f"S{j:02d}" for j in range(n_names)):
        ret = rng.normal(drift[i], 0.012, size=n_bars)
        close = base[i] * np.exp(np.cumsum(ret))
        open_ = np.concatenate([[base[i]], close[:-1]])
        spread = np.abs(rng.normal(0.0, 0.006, size=n_bars)) * open_
        high = np.maximum(open_, close) + spread
        low = np.maximum(0.01, np.minimum(open_, close) - spread)
        vol_shares = np.exp(rng.normal(np.log(2e5), 0.5, size=n_bars))
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
    return pl.DataFrame(rows).sort(["event_time", "security_id"])


def _churn_targets(seed: int, bars: pl.DataFrame, amplitude: float) -> pl.DataFrame:
    """Causal AR(1) tilted targets — deliberately noisy so bands have work to do."""
    sids = sorted(bars["security_id"].unique().to_list())
    dates = sorted(bars["event_time"].unique().to_list())
    rng = np.random.default_rng(seed)
    w = np.zeros(len(sids))
    rows: list[dict[str, Any]] = []
    for dt in dates:
        w = 0.9 * w + rng.normal(0.0, amplitude, size=len(sids))
        w -= w.mean()  # dollar-neutral tilt book
        for i, sid in enumerate(sids):
            rows.append({"event_time": dt, "security_id": sid, "target_weight": float(w[i])})
    return pl.DataFrame(rows).sort(["event_time", "security_id"])


def run_cost_surface(
    bars: pl.DataFrame,
    targets: pl.DataFrame,
    *,
    bands: tuple[float, ...] = DEFAULT_BANDS,
    maker_commission_bps: float = 0.5,
    taker_commission_bps: float = 1.0,
    seed: int = 0,
    initial_nav: float = 1_000_000.0,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Run the hysteresis surface + maker/taker probe; return rows + receipt."""
    if bands != tuple(sorted(bands)) or any(not np.isfinite(b) or b < 0 for b in bands):
        raise ValueError("bands must be finite, non-negative and sorted")
    rows: list[dict[str, Any]] = []
    base_costs = CostConfig(commission_bps=taker_commission_bps)
    raw = targets.sort(["event_time", "security_id"])
    for band in bands:
        banded = banded_targets(raw, band)
        result = run_backtest(bars, banded, AppConfig(costs=base_costs), initial_nav=initial_nav)
        metrics = result.metrics
        n_fills = result.fills.height

        def _m(key: str, _metrics: Mapping[str, Any] = metrics) -> float:
            value = _metrics.get(key)
            return float(value) if isinstance(value, (int, float)) else 0.0

        # Tracking error: forward-fill the banded stream over the raw grid so
        # suppressed revisions measure as deviation from the unbanded target.
        keys = ["event_time", "security_id"]
        sent = raw.join(banded.rename({"target_weight": "w_sent"}), on=keys, how="left").sort(keys)
        filled = sent.with_columns(pl.col("w_sent").forward_fill().over("security_id"))
        diff = (filled["target_weight"] - filled["w_sent"].fill_null(0.0)).to_numpy()
        tracking_rmse = float(np.sqrt(np.mean(diff**2))) if diff.size else 0.0
        # Target-space turnover of the *sent* stream is non-increasing in the
        # band (triangle inequality on suppressed increments); realized
        # turnover is reported too — it can jitter when accumulated trades
        # split across bars at the participation cap.
        sent_w = banded.sort(keys).with_columns(
            (pl.col("target_weight") - pl.col("target_weight").shift(1).over("security_id"))
            .abs()
            .fill_null(0.0)
            .alias("dw")
        )
        target_turnover = float(sent_w.select(pl.sum("dw")).item())
        turnover = float(result.equity["turnover"].sum()) if "turnover" in result.equity else 0.0
        rows.append(
            {
                "probe": "hysteresis",
                "band": float(band),
                "n_fills": int(n_fills),
                "suppressed_rows": int(raw.height - banded.height),
                "turnover": turnover,
                "target_turnover": target_turnover,
                "cost_total": sum(
                    _m(k) for k in ("commission", "spread", "impact", "turnover_bps_cost")
                ),
                "cost_commission": _m("commission"),
                "cost_spread": _m("spread"),
                "cost_impact": _m("impact"),
                "tracking_rmse": tracking_rmse,
            }
        )
    maker_costs = CostConfig(
        commission_bps=taker_commission_bps,
        maker_commission_bps=maker_commission_bps,
    )
    probe = maker_taker_fee_delta(bars, targets, maker_costs)
    maker_row: dict[str, Any] = {"probe": "maker_fill", "band": None}
    maker_row.update(probe)
    rows.append(maker_row)
    frame = pl.DataFrame(rows)
    inputs_sha = hash_inputs(bars, targets)
    receipt = build_receipt_v2(
        kind="cost_surface_eval",
        data_label="SYNTHETIC",
        dataset={
            "schema": COST_SURFACE_SCHEMA,
            "inputs_sha256": inputs_sha,
            "n_bars": bars.height,
            "n_names": bars["security_id"].n_unique(),
        },
        params={
            "bands": list(bands),
            "maker_commission_bps": float(maker_commission_bps),
            "taker_commission_bps": float(taker_commission_bps),
            "seed": int(seed),
            "initial_nav": float(initial_nav),
        },
        code_files=(
            Path(__file__),
            Path(__file__).parents[1] / "execution" / "maker_taker_probe.py",
        ),
        verdict="pass",
        payload={"n_rows": len(rows), "results": rows},
    )
    return frame, receipt


def hash_inputs(bars: pl.DataFrame, targets: pl.DataFrame) -> str:
    """Content hash of the evaluated inputs (bars + raw target grid)."""
    from quant_fund.utils.hashing import hash_bytes

    return hash_bytes(
        bars.sort(["event_time", "security_id"]).write_csv().encode()
        + targets.sort(["event_time", "security_id"]).write_csv().encode()
    )


def write_cost_surface_receipt(
    receipt: Mapping[str, Any],
    receipts_dir: Path | str = Path("receipts"),
) -> Path:
    """Seal a cost-surface receipt to ``receipts/cost_surface_<hash>.json``."""
    if receipt.get("kind") != "cost_surface_eval" or receipt.get("data_label") != "SYNTHETIC":
        raise ValueError("cost-surface receipt violates the honesty contract")
    sealed = seal_receipt(receipt)
    digest = sealed["receipt_sha256"]
    path = Path(receipts_dir) / f"cost_surface_{digest[:16]}.json"
    _atomic_write_text(path, json.dumps(sealed, indent=2, sort_keys=True) + "\n")
    return path


def format_cost_surface_table(frame: pl.DataFrame) -> str:
    """Render the surface: band × turnover/cost/tracking, then maker delta."""
    lines = ["probe | band | turnover | cost_total | tracking_rmse | extra"]
    frame = frame.with_columns(pl.col("band").cast(pl.Float64))
    for row in frame.iter_rows(named=True):
        if row["probe"] == "hysteresis":
            lines.append(
                f"hyst  | {row['band']:.4f} | {row['turnover']:.2f} | "
                f"{row['cost_total']:.1f} | {row['tracking_rmse']:.5f} | "
                f"fills={row['n_fills']}"
            )
        else:
            lines.append(
                f"maker | - | - | - | - | fee_bps t={row['taker_fee_bps_per_filled']:.2f} "
                f"m={row['maker_fee_bps_per_filled']:.2f} fill_rate={row['maker_fill_rate']:.3f} "
                f"unfilled={row['unfilled_notional_frac']:.3f}"
            )
    return "\n".join(lines)


__all__ = [
    "COST_SURFACE_SCHEMA",
    "DEFAULT_BANDS",
    "run_cost_surface",
    "write_cost_surface_receipt",
    "format_cost_surface_table",
]
