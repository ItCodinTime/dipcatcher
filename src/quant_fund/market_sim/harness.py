"""Stress a research weight function inside the agent market.

The plain comparison is ``run_backtest`` on the background tape, the tape
the ecology produced with no strategy orders. The strategy's own orders
change the book, so the two paths are not the same experiment. Both reports
are simulation diagnostics. Risk limits on the plain backtest are widened
only so the weight can be executed; that copy is not the research gate.
"""

from __future__ import annotations

import math
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from quant_fund.backtest.engine import run_backtest
from quant_fund.config.loader import load_config
from quant_fund.config.models import AppConfig
from quant_fund.lightspeed.momentum import momentum_scores
from quant_fund.market_sim.config import EVIDENCE, EcologyConfig
from quant_fund.market_sim.scenarios import SCENARIOS, run_scenario
from quant_fund.market_sim.simulator import SimResult, WeightFunction, run_ecology
from quant_fund.metrics.returns import max_drawdown

_REPO = Path(__file__).resolve().parents[3]


def lightspeed_momentum_weight(
    closes: np.ndarray,
    *,
    fast: int = 8,
    slow: int = 21,
    blend: float = 0.5,
    scale: float = 4.0,
) -> float:
    """Map a single-name momentum score to a weight.

    ``momentum_scores`` is the research signal. The scale is a fixed adapter
    so the score has the units of a target weight. It is not a fit to a tape.
    """
    path = np.asarray(closes, dtype=float)
    scores = momentum_scores({"SIM": path}, ("SIM",), fast, slow, blend)
    last = float(scores["SIM"][-1])
    if not math.isfinite(last):
        return 0.0
    return float(last * scale)


def mean_reversion_weight(closes: np.ndarray, *, lookback: int = 20, scale: float = 8.0) -> float:
    """Fade a trailing-mean gap. Fixed lookback and scale, not a fitted rule."""
    path = np.asarray(closes, dtype=float)
    if path.size < lookback:
        return 0.0
    window = path[-lookback:]
    mean = float(window.mean())
    if mean == 0.0 or not math.isfinite(mean):
        return 0.0
    gap = (mean - float(path[-1])) / mean
    if not math.isfinite(gap):
        return 0.0
    return float(gap * scale)


def target_weight_replay(panel: pl.DataFrame, security_id: str = "SIM") -> WeightFunction:
    """Replay a target-weight panel in bar order for one security."""
    required = {"event_time", "security_id", "target_weight"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"target-weight panel missing {sorted(missing)}")
    rows = panel.filter(pl.col("security_id") == security_id).sort("event_time")
    weights = [float(value) for value in rows["target_weight"].to_list()]

    def _fn(closes: np.ndarray) -> float:
        if not weights:
            return 0.0
        index = min(int(np.asarray(closes).size) - 1, len(weights) - 1)
        if index < 0:
            return 0.0
        return weights[index]

    return _fn


def diagnostic_backtest_config(root: Path) -> AppConfig:
    """Research config with limits wide enough that the weight is executable.

    The research risk gate is left untouched on disk. This object is a copy
    used only by the comparison. ``root`` is an empty directory so a host
    GARCH artifact cannot change the gate.
    """
    cfg = load_config(_REPO / "configs" / "research.yaml")
    cfg.data.root = root
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_net = 1.0
    cfg.risk_gate.max_gross = 2.0
    cfg.risk_gate.max_participation = 1.0
    cfg.risk_gate.max_order_notional = 1.0e12
    cfg.risk_gate.stale_price_bars = 10_000
    cfg.risk_gate.max_predicted_vol = 5.0
    cfg.costs.participation_limit = 1.0
    return cfg


def simulation_bars(result: SimResult) -> pl.DataFrame:
    """Background tape as a bar panel. Volume of 0 keeps the previous dollar ADV."""
    start = datetime(2024, 1, 2, tzinfo=UTC)
    closes: list[float] = []
    rows: list[dict[str, object]] = []
    previous_adv: float | None = None
    for i, bar in enumerate(result.bars):
        closes.append(float(bar.close))
        window = closes[max(0, i - 19) : i + 1]
        vol = 0.02
        if len(window) >= 3:
            rets = np.diff(np.log(np.maximum(np.asarray(window, dtype=float), 1e-12)))
            if rets.size >= 2:
                estimate = float(np.std(rets, ddof=1))
                if math.isfinite(estimate) and estimate >= 0.0:
                    vol = estimate
        dollar = float(bar.close) * float(bar.volume)
        if dollar > 0.0:
            adv = dollar
        elif previous_adv is not None and previous_adv > 0.0:
            adv = previous_adv
        else:
            adv = float(bar.close) * 1000.0
        previous_adv = adv
        rows.append(
            {
                "security_id": "SIM",
                "event_time": start + timedelta(minutes=i),
                "open": float(bar.open),
                "close": float(bar.close),
                "close_total_return": float(bar.close),
                "volume": float(bar.volume),
                "adv": float(adv),
                "vol_20": float(vol),
                "source": "synthetic",
            }
        )
    if not rows:
        return pl.DataFrame(
            schema={
                "security_id": pl.Utf8,
                "event_time": pl.Datetime(time_zone="UTC"),
                "open": pl.Float64,
                "close": pl.Float64,
                "close_total_return": pl.Float64,
                "volume": pl.Float64,
                "adv": pl.Float64,
                "vol_20": pl.Float64,
                "source": pl.Utf8,
            }
        )
    return pl.DataFrame(rows)


def _path_change(equity: np.ndarray) -> tuple[float | None, float | None, float | None]:
    path = np.asarray(equity, dtype=float).reshape(-1)
    path = path[np.isfinite(path)]
    if path.size == 0:
        return None, None, None
    start = float(path[0])
    end = float(path[-1])
    change = None
    if start > 0.0 and math.isfinite(end):
        change = end / start - 1.0
    drawdown = None
    if path.size >= 2 and np.all(path > 0.0):
        rets = np.diff(path) / path[:-1]
        value = max_drawdown(rets)
        if math.isfinite(value):
            drawdown = float(value)
    return change, drawdown, end


def _weight_l1(left: list[float], right: list[float]) -> float | None:
    n = min(len(left), len(right))
    if n == 0:
        return None
    gap = [abs(left[i] - right[i]) for i in range(n)]
    return float(sum(gap) / n)


def _clip_weight(weight_fn: WeightFunction, closes: np.ndarray, cap: float) -> float:
    weight = float(weight_fn(closes))
    if not math.isfinite(weight):
        raise ValueError("strategy weight must be finite")
    return float(np.clip(weight, -cap, cap))


def _scenario_report(
    name: str,
    weight_fn: WeightFunction,
    cfg: EcologyConfig,
) -> dict[str, object]:
    seen: list[float] = []
    cap = cfg.strategy_max_weight

    def _recording(closes: np.ndarray) -> float:
        weight = _clip_weight(weight_fn, closes, cap)
        seen.append(weight)
        return weight

    stressed = run_scenario(name, cfg, strategy=_recording)
    background = run_scenario(name, cfg)
    background_weights: list[float] = []
    closes: list[float] = []
    for bar in background.bars:
        closes.append(float(bar.close))
        background_weights.append(_clip_weight(weight_fn, np.asarray(closes, dtype=float), cap))
    change, drawdown, terminal = _path_change(stressed.strategy_equity)
    bars = simulation_bars(background)
    backtest_block: dict[str, object] = {
        "status": "too_few_bars",
        "marked_to_market_change": None,
        "max_drawdown": None,
        "risk_gate_rejects": None,
        "n_fills": 0,
    }
    if bars.height >= 2:
        weights = pl.DataFrame(
            {
                "event_time": bars["event_time"],
                "security_id": ["SIM"] * bars.height,
                "target_weight": background_weights,
            }
        )
        with tempfile.TemporaryDirectory(prefix="market-sim-diag-") as tmp:
            config = diagnostic_backtest_config(Path(tmp))
            plain = run_backtest(bars, weights, config, initial_nav=cfg.strategy_nav)
        metrics = plain.metrics
        bt_change = metrics.get("total_return")
        bt_dd = metrics.get("max_drawdown")
        change_value = bt_change if isinstance(bt_change, (int, float)) else None
        if plain.equity.height >= 1 and "nav" in plain.equity.columns and cfg.strategy_nav > 0.0:
            last = float(plain.equity["nav"][-1])
            if math.isfinite(last):
                change_value = last / float(cfg.strategy_nav) - 1.0
        dd_value = (
            float(bt_dd) if isinstance(bt_dd, (int, float)) and math.isfinite(bt_dd) else None
        )
        if dd_value is None and plain.equity.height >= 2 and "nav" in plain.equity.columns:
            navs = np.asarray(plain.equity["nav"].to_list(), dtype=float)
            if navs.size >= 2 and np.all(np.isfinite(navs)) and np.all(navs > 0.0):
                dd_value = float(max_drawdown(np.diff(navs) / navs[:-1]))
                if not math.isfinite(dd_value):
                    dd_value = None
        rejects = metrics.get("risk_gate_rejects")
        backtest_block = {
            "status": "ok",
            "marked_to_market_change": change_value if isinstance(change_value, float) else None,
            "max_drawdown": dd_value,
            "risk_gate_rejects": int(rejects) if isinstance(rejects, int) else None,
            "n_fills": int(plain.fills.height),
            "initial_marked_value": float(cfg.strategy_nav),
        }
    slip = stressed.strategy_slippage_bps
    return {
        "scenario": name,
        "simulation": {
            "marked_to_market_change": change,
            "max_drawdown": drawdown,
            "slippage_bps": float(slip) if math.isfinite(slip) else None,
            "filled_qty": int(stressed.strategy_filled_qty),
            "requested_qty": int(stressed.strategy_requested_qty),
            "terminal_marked_value": terminal,
            "n_events": int(stressed.n_events),
            "n_trades": int(stressed.n_trades),
            "checksum": int(stressed.checksum),
        },
        "background_checksum": int(background.checksum),
        "weight_l1": _weight_l1(seen, background_weights),
        "mean_abs_strategy_weight": (float(np.mean(np.abs(seen))) if seen else None),
        "backtest": backtest_block,
        "hook": dict(stressed.hook),
        "note": (
            "The strategy trades inside the book. The plain backtest fills "
            "the same weight function on the background tape at the next open "
            "and charges the research cost model. Slippage here is the "
            "volume-weighted adverse tick gap versus the decision mid, in "
            "basis points of that mid."
        ),
    }


def stress_strategy(
    weight_fn: WeightFunction,
    *,
    name: str,
    cfg: EcologyConfig | None = None,
    scenarios: tuple[str, ...] = SCENARIOS,
) -> dict[str, object]:
    """Run one weight function through each named scenario and the plain backtest."""
    if not scenarios:
        raise ValueError("scenarios must be non-empty")
    unknown = [item for item in scenarios if item not in SCENARIOS]
    if unknown:
        raise ValueError(f"unknown scenarios: {unknown}")
    config = cfg or EcologyConfig()
    report: dict[str, object] = dict(EVIDENCE)
    report.update(
        {
            "strategy": name,
            "seed": config.seed,
            "max_events": config.max_events,
            "risk_limit_note": (
                "Plain backtest uses configs/research.yaml with diagnostic "
                "overrides: max_name 1, max_net 1, max_gross 2, "
                "max_participation 1, max_order_notional 1e12, "
                "participation_limit 1, stale_price_bars 10000, "
                "max_predicted_vol 5. Empty data root, so no GARCH artifact."
            ),
            "scenarios": [_scenario_report(scenario, weight_fn, config) for scenario in scenarios],
        }
    )
    return report


def background_ecology(cfg: EcologyConfig | None = None) -> SimResult:
    """Ecology with no strategy. Used when a caller wants the plain tape only."""
    return run_ecology(cfg or EcologyConfig())
