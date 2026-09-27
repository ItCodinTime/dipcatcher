"""Pre-registered strategy sweep recorded on the provenance trial ledger.

The grid, costs, and splits live in ``research/reality/preregistration.json``.
This module does not add a signal. It runs the frozen cells through the
existing backtest engine and appends every cell with
``ProvenanceDB.insert_trial``.

Research diagnostic only. No live-trading claim.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from quant_fund.backtest.engine import run_backtest
from quant_fund.backtest.sleeves import slow_trend_weights, sweep_reclaim_weights
from quant_fund.config.loader import load_config
from quant_fund.config.models import AppConfig
from quant_fund.metrics.inference import bootstrap_sharpe_ci
from quant_fund.metrics.overfitting import deflated_sharpe, moments_from_returns
from quant_fund.metrics.returns import max_drawdown

_NY = ZoneInfo("America/New_York")
_ROOT = Path(__file__).resolve().parents[3]
_PREREG = _ROOT / "research" / "reality" / "preregistration.json"


@dataclass(frozen=True)
class Cell:
    strategy: str
    cluster_id: str
    family: str
    params: dict[str, Any]

    def trial_id(self, study_id: str) -> str:
        from quant_fund.proofcore.contracts import sha256_hex_json

        return sha256_hex_json(
            {"params": self.params, "strategy": self.strategy, "study_id": study_id}
        )


@dataclass
class ScoredCell:
    cell: Cell
    trial_id: str
    by_window: dict[str, dict[str, Any]]
    validation_returns: np.ndarray
    train_returns: np.ndarray
    holdout_returns: np.ndarray
    dates: dict[str, list[str]]
    turnover: dict[str, np.ndarray]
    equity: pl.DataFrame


def load_spec(path: Path | None = None) -> dict[str, Any]:
    spec_path = path or _PREREG
    loaded = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("pre-registration must be a JSON object")
    return loaded


def grid_cells(spec: dict[str, Any]) -> list[Cell]:
    """Expand the frozen grid. Order is the pre-registration's loop order."""
    grid = spec["grid"]
    dip = grid["sweep_reclaim"]
    cells: list[Cell] = []
    for lookback in dip["lookback"]:
        for hold_bars in dip["hold_bars"]:
            for decay in dip["decay"]:
                cells.append(
                    Cell(
                        strategy="sweep_reclaim",
                        cluster_id=str(dip["cluster_id"]),
                        family=str(dip["family"]),
                        params={
                            "decay": decay,
                            "gross_scale": dip["gross_scale"],
                            "hold_bars": hold_bars,
                            "lookback": lookback,
                            "max_name": dip["max_name"],
                        },
                    )
                )
    trend = grid["slow_trend"]
    for fast_bars in trend["fast_bars"]:
        for slow_bars in trend["slow_bars"]:
            if int(fast_bars) >= int(slow_bars):
                continue
            for vol_window in trend["vol_window"]:
                cells.append(
                    Cell(
                        strategy="slow_trend",
                        cluster_id=str(trend["cluster_id"]),
                        family=str(trend["family"]),
                        params={
                            "fast_bars": fast_bars,
                            "gross_scale": trend["gross_scale"],
                            "max_name": trend["max_name"],
                            "slow_bars": slow_bars,
                            "vol_window": vol_window,
                        },
                    )
                )
    base = grid["equal_weight_long"]
    cells.append(
        Cell(
            strategy="equal_weight_long",
            cluster_id=str(base["cluster_id"]),
            family=str(base["family"]),
            params={"name_cap": base["name_cap"], "net_cap": base["net_cap"]},
        )
    )
    expected = spec["expected_trials"]
    counts = {
        "sweep_reclaim": sum(1 for cell in cells if cell.strategy == "sweep_reclaim"),
        "slow_trend": sum(1 for cell in cells if cell.strategy == "slow_trend"),
        "equal_weight_long": sum(1 for cell in cells if cell.strategy == "equal_weight_long"),
    }
    for name, count in counts.items():
        if count != int(expected[name]):
            raise ValueError(f"{name} cell count {count} != pre-registered {expected[name]}")
    if len(cells) != int(expected["total"]):
        raise ValueError(f"grid has {len(cells)} cells, pre-registration says {expected['total']}")
    return cells


def assert_cost_lock(config: AppConfig, spec: dict[str, Any]) -> None:
    """Refuse to run if the loaded cost or risk gate differs from the freeze."""
    costs = spec["costs"]
    gate = spec["risk_gate"]
    pairs: tuple[tuple[float, float, str], ...] = (
        (float(config.costs.commission_bps), float(costs["commission_bps"]), "commission_bps"),
        (float(config.costs.half_spread_bps), float(costs["half_spread_bps"]), "half_spread_bps"),
        (float(config.costs.impact_y), float(costs["impact_y"]), "impact_y"),
        (
            float(config.costs.bps_per_turnover),
            float(costs["bps_per_turnover"]),
            "bps_per_turnover",
        ),
        (
            float(config.costs.borrow_bps_per_year),
            float(costs["borrow_bps_per_year"]),
            "borrow_bps_per_year",
        ),
        (
            float(config.costs.financing_bps_per_year),
            float(costs["financing_bps_per_year"]),
            "financing_bps_per_year",
        ),
        (
            float(config.costs.participation_limit),
            float(costs["participation_limit"]),
            "participation_limit",
        ),
        (
            float(config.risk_gate.max_order_notional),
            float(gate["max_order_notional"]),
            "max_order_notional",
        ),
        (float(config.risk_gate.max_gross), float(gate["max_gross"]), "max_gross"),
        (float(config.risk_gate.max_net), float(gate["max_net"]), "max_net"),
        (float(config.risk_gate.max_name), float(gate["max_name"]), "max_name"),
        (
            float(config.risk_gate.max_participation),
            float(gate["max_participation"]),
            "max_participation",
        ),
        (
            float(config.risk_gate.max_predicted_vol),
            float(gate["max_predicted_vol"]),
            "max_predicted_vol",
        ),
        (
            float(config.risk_gate.stale_price_bars),
            float(gate["stale_price_bars"]),
            "stale_price_bars",
        ),
    )
    for got, want, name in pairs:
        if not math.isclose(got, want, rel_tol=0.0, abs_tol=0.0):
            raise ValueError(f"cost lock: {name} is {got}, pre-registration froze {want}")
    if bool(config.costs.frictionless) is not False:
        raise ValueError("cost lock: frictionless must stay false")
    if str(config.execution.fill.value) != str(costs["fill"]):
        raise ValueError("cost lock: fill convention drifted")


def select_winner(scored: list[ScoredCell]) -> ScoredCell:
    """Highest validation per-period ratio. Ties break on trial id.

    Degenerate validation paths cannot win. Holdout is not an argument.
    """
    eligible = [row for row in scored if not row.by_window["validation"]["degenerate"]]
    if not eligible:
        raise ValueError("every validation path was degenerate; refusing to crown a winner")
    return min(
        eligible,
        key=lambda row: (-float(row.by_window["validation"]["periodic_ratio"]), row.trial_id),
    )


def ny_date(stamp: datetime) -> date:
    if stamp.tzinfo is None:
        raise ValueError("event_time must be timezone-aware")
    return stamp.astimezone(_NY).date()


def _parse_day(value: str) -> date:
    return date.fromisoformat(value)


def _as_date(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return None


def window_bounds(spec: dict[str, Any]) -> dict[str, tuple[date, date]]:
    splits = spec["splits"]
    return {name: (_parse_day(bounds[0]), _parse_day(bounds[1])) for name, bounds in splits.items()}


def _split_arrays(
    equity: pl.DataFrame,
    bounds: dict[str, tuple[date, date]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, list[date]]]:
    if equity.height < 2 or "nav" not in equity.columns:
        empty = np.asarray([], dtype=float)
        return (
            {name: empty.copy() for name in bounds},
            {name: empty.copy() for name in bounds},
            {name: [] for name in bounds},
        )
    nav = [float(value) for value in equity["nav"].to_list()]
    times = equity["event_time"].to_list()
    turn_col = (
        [float(value) for value in equity["turnover"].to_list()]
        if "turnover" in equity.columns
        else [0.0] * len(nav)
    )
    rets: dict[str, list[float]] = {name: [] for name in bounds}
    turns: dict[str, list[float]] = {name: [] for name in bounds}
    dates: dict[str, list[date]] = {name: [] for name in bounds}
    for index in range(1, len(nav)):
        previous = nav[index - 1]
        if previous == 0.0 or not math.isfinite(previous) or not math.isfinite(nav[index]):
            simple = float("nan")
        else:
            simple = nav[index] / previous - 1.0
        session = ny_date(times[index])
        for name, (start, end) in bounds.items():
            if start <= session < end:
                rets[name].append(simple)
                turns[name].append(turn_col[index])
                dates[name].append(session)
                break
    return (
        {name: np.asarray(values, dtype=float) for name, values in rets.items()},
        {name: np.asarray(values, dtype=float) for name, values in turns.items()},
        dates,
    )


def _window_stats(
    returns: np.ndarray,
    turnover: np.ndarray,
    *,
    periods: int,
    n_boot: int,
    seed: int,
    alpha: float,
    bootstrap: bool,
) -> dict[str, Any]:
    ratio, skew, kurt, degenerate = _period_ratio(returns)
    annualized = float("nan")
    if not degenerate:
        annualized = ratio * math.sqrt(float(periods))
    if bootstrap:
        ci_lo, ci_hi, ci_point = bootstrap_sharpe_ci(
            returns,
            n_boot=n_boot,
            periods=periods,
            alpha=alpha,
            seed=seed,
        )
    else:
        ci_lo, ci_hi, ci_point = float("nan"), float("nan"), float("nan")
    compounded = float("nan")
    if returns.size and np.isfinite(returns).all():
        compounded = float(np.prod(1.0 + returns) - 1.0)
    mean_turn = float(np.mean(turnover)) if turnover.size else float("nan")
    return {
        "n_obs": int(returns.size),
        "periodic_ratio": ratio,
        "skew": skew,
        "kurtosis_raw": kurt,
        "degenerate": degenerate,
        "annualized_ratio": annualized,
        "ratio_ci_low": ci_lo,
        "ratio_ci_high": ci_hi,
        "ratio_ci_point": ci_point,
        "compounded_return": compounded,
        "max_drawdown": float(max_drawdown(returns)) if returns.size else float("nan"),
        "mean_turnover": mean_turn,
    }


def _period_ratio(returns: np.ndarray) -> tuple[float, float, float, bool]:
    values = np.asarray(returns, dtype=float).reshape(-1)
    if values.size < 4 or not np.isfinite(values).all():
        return 0.0, 0.0, 3.0, True
    sigma, skew, kurt = moments_from_returns(values)
    if not (np.isfinite(sigma) and np.isfinite(skew) and np.isfinite(kurt)) or sigma == 0.0:
        return 0.0, 0.0, 3.0, True
    return float(np.mean(values) / sigma), float(skew), float(kurt), False


def equal_weight_long(bars: pl.DataFrame, *, name_cap: float, net_cap: float) -> pl.DataFrame:
    """One long-only baseline sized inside the frozen risk-gate caps."""
    if name_cap <= 0 or net_cap <= 0:
        raise ValueError("baseline caps must be positive")
    counts = bars.group_by("event_time").agg(pl.len().alias("_n"))
    frame = bars.join(counts, on="event_time").with_columns(
        pl.when((1.0 / pl.col("_n")) < name_cap)
        .then(1.0 / pl.col("_n"))
        .otherwise(pl.lit(name_cap))
        .alias("_raw")
    )
    frame = frame.with_columns(pl.col("_raw").sum().over("event_time").alias("_gross"))
    frame = frame.with_columns(
        pl.when(pl.col("_gross") > net_cap)
        .then(pl.col("_raw") * net_cap / pl.col("_gross"))
        .otherwise(pl.col("_raw"))
        .alias("target_weight")
    )
    return frame.select("event_time", "security_id", "target_weight").sort(
        ["event_time", "security_id"]
    )


def weights_for(cell: Cell, bars: pl.DataFrame, *, name_cap: float, net_cap: float) -> pl.DataFrame:
    params = cell.params
    if cell.strategy == "sweep_reclaim":
        return sweep_reclaim_weights(
            bars,
            lookback=int(params["lookback"]),
            hold_bars=int(params["hold_bars"]),
            decay=float(params["decay"]),
            max_name=float(params["max_name"]),
            gross_scale=float(params["gross_scale"]),
        )
    if cell.strategy == "slow_trend":
        return slow_trend_weights(
            bars,
            fast_bars=int(params["fast_bars"]),
            slow_bars=int(params["slow_bars"]),
            max_name=float(params["max_name"]),
            gross_scale=float(params["gross_scale"]),
            vol_window=int(params["vol_window"]),
        )
    if cell.strategy == "equal_weight_long":
        return equal_weight_long(bars, name_cap=name_cap, net_cap=net_cap)
    raise ValueError(f"unknown strategy {cell.strategy}")


def prepare_bars(frame: pl.DataFrame) -> pl.DataFrame:
    """Causal dollar-volume and vol columns the engine already knows how to read."""
    ordered = frame.sort(["security_id", "event_time"])
    enriched = ordered.with_columns(
        (pl.col("close") * pl.col("volume")).alias("_dv"),
        pl.col("close").pct_change().over("security_id").alias("_ret"),
    )
    enriched = enriched.with_columns(
        pl.col("_dv").rolling_mean(20, min_samples=1).shift(1).over("security_id").alias("adv"),
        pl.col("_ret").rolling_std(20, min_samples=5).shift(1).over("security_id").alias("vol_20"),
    )
    first_dv = enriched.group_by("security_id").agg(
        pl.col("_dv").drop_nulls().first().alias("_adv0")
    )
    enriched = enriched.join(first_dv, on="security_id", how="left")
    return enriched.with_columns(
        pl.col("adv").fill_null(pl.col("_adv0")).fill_null(0.0).alias("adv"),
        pl.col("vol_20").fill_null(0.02).alias("vol_20"),
        pl.col("close").alias("close_total_return"),
    ).select(
        "security_id",
        "event_time",
        "open",
        "high",
        "low",
        "close",
        "close_total_return",
        "volume",
        "adv",
        "vol_20",
        "source",
    )


def fetch_yahoo_panel(spec: dict[str, Any], cache: Path) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Download US_LIQUID daily bars. Abort on HTTP failure. Do not invent prices."""
    from quant_fund.data.adapters.yahoo_eod import fetch_yahoo_chart, parse_yahoo_chart

    data = spec["data"]
    start = datetime.fromisoformat(str(data["fetch_start_utc"]))
    end = datetime.fromisoformat(str(data["fetch_end_utc"]))
    symbols = [str(symbol) for symbol in data["symbols"]]
    frames: list[pl.DataFrame] = []
    errors: dict[str, str] = {}
    for index, symbol in enumerate(symbols):
        if index:
            time.sleep(0.35)
        try:
            payload = fetch_yahoo_chart(symbol, start=start, end=end, retries=3, timeout=30.0)
            frame = parse_yahoo_chart(payload, security_id=symbol, yahoo_symbol=symbol)
        except Exception as exc:
            errors[symbol] = f"{type(exc).__name__}: {exc}"
            continue
        if frame.is_empty():
            errors[symbol] = "empty frame"
            continue
        frames.append(frame)
    if errors:
        raise RuntimeError(
            "Yahoo fetch failed; stopping without a synthetic substitute: " + json.dumps(errors)
        )
    raw = pl.concat(frames, how="diagonal_relaxed")
    raw = raw.with_columns(
        pl.col("event_time").dt.convert_time_zone("America/New_York").dt.date().alias("_ny")
    )
    early = date.fromisoformat("2014-06-30")
    late = date.fromisoformat("2024-06-28")
    included: list[str] = []
    excluded: list[dict[str, str]] = []
    for symbol in symbols:
        part = raw.filter(pl.col("security_id") == symbol)
        first = _as_date(part["_ny"].min())
        last = _as_date(part["_ny"].max())
        if first is None or last is None or first > early or last < late:
            excluded.append({"symbol": symbol, "first": str(first), "last": str(last)})
            continue
        included.append(symbol)
    if len(included) < 8:
        raise RuntimeError(f"fewer than 8 names survived the inclusion rule: {excluded}")
    panel = raw.filter(pl.col("security_id").is_in(included)).drop("_ny")
    duplicate = panel.select(pl.struct(["security_id", "event_time"]).is_duplicated().any()).item()
    if duplicate:
        raise RuntimeError("duplicate security_id/event_time in the Yahoo panel")
    prepared = prepare_bars(panel)
    cache.mkdir(parents=True, exist_ok=True)
    parquet_path = cache / "bars.parquet"
    prepared.write_parquet(parquet_path)
    from quant_fund.proofcore.contracts import sha256_hex_bytes

    digest = sha256_hex_bytes(parquet_path.read_bytes())
    meta = {
        "symbols_included": included,
        "symbols_excluded": excluded,
        "n_bars": int(prepared.height),
        "dataset_sha256": digest,
        "parquet_path": str(parquet_path),
    }
    return prepared, meta


def _returns_sha(returns: np.ndarray) -> str:
    from quant_fund.proofcore.contracts import sha256_hex_json

    rounded = [round(float(value), 12) for value in np.asarray(returns, dtype=float).reshape(-1)]
    return sha256_hex_json(rounded)


def score_cell(
    cell: Cell,
    bars: pl.DataFrame,
    config: AppConfig,
    spec: dict[str, Any],
) -> ScoredCell:
    name_cap = float(spec["risk_gate"]["max_name"])
    net_cap = float(spec["risk_gate"]["max_net"])
    weights = weights_for(cell, bars, name_cap=name_cap, net_cap=net_cap)
    result = run_backtest(
        bars,
        weights,
        config,
        initial_nav=float(spec["costs"]["initial_nav"]),
    )
    bounds = window_bounds(spec)
    returns, turns, dates = _split_arrays(result.equity, bounds)
    diag = spec["diagnostics"]["bootstrap_ci"]
    validation = _window_stats(
        returns["validation"],
        turns["validation"],
        periods=int(diag["periods"]),
        n_boot=int(diag["n_boot"]),
        seed=int(diag["seed"]),
        alpha=float(diag["alpha"]),
        bootstrap=False,
    )
    study_id = str(spec["study_id"])
    return ScoredCell(
        cell=cell,
        trial_id=cell.trial_id(study_id),
        by_window={"validation": validation},
        validation_returns=returns["validation"],
        train_returns=returns["train"],
        holdout_returns=returns["holdout"],
        dates={name: [day.isoformat() for day in dates[name]] for name in dates},
        turnover=turns,
        equity=result.equity,
    )


def _attach_window_reports(scored: list[ScoredCell], spec: dict[str, Any]) -> None:
    """Full window reports, including holdout, after the ledger rows exist.

    Validation ratios used for selection and the ledger are left unchanged.
    """
    diag = spec["diagnostics"]["bootstrap_ci"]
    for row in scored:
        series = {
            "train": row.train_returns,
            "validation": row.validation_returns,
            "holdout": row.holdout_returns,
        }
        frozen_validation = row.by_window["validation"]["periodic_ratio"]
        for name, values in series.items():
            stats = _window_stats(
                values,
                row.turnover[name],
                periods=int(diag["periods"]),
                n_boot=int(diag["n_boot"]),
                seed=int(diag["seed"]),
                alpha=float(diag["alpha"]),
                bootstrap=True,
            )
            stats["dates"] = row.dates[name]
            row.by_window[name] = stats
        if row.by_window["validation"]["periodic_ratio"] != frozen_validation:
            raise RuntimeError("validation ratio changed after the ledger write")


def pbo_on_pre_holdout(scored: list[ScoredCell], spec: dict[str, Any]) -> dict[str, Any]:
    """CSCV on train-then-validation returns. Holdout stays out."""
    from quant_fund.reality.cscv import cscv_pbo

    blocks = int(spec["diagnostics"]["pbo"]["s_blocks"])
    columns: list[np.ndarray] = []
    for row in scored:
        series = np.concatenate([row.train_returns, row.validation_returns])
        columns.append(series)
    length = int(columns[0].size)
    if any(int(column.size) != length for column in columns):
        raise RuntimeError("pre-holdout return lengths differ across trials")
    usable = length - (length % blocks)
    if usable < 2 * blocks:
        return {"pbo": None, "n_periods": usable, "reason": "shorter than two blocks"}
    matrix = np.column_stack([column[-usable:] for column in columns])
    result = cscv_pbo(matrix, s_blocks=blocks)
    pbo = result["pbo"]
    return {
        "pbo": None if isinstance(pbo, float) and not math.isfinite(pbo) else pbo,
        "n_periods": usable,
        "n_splits": result.get("n_splits"),
        "n_dropped": result.get("n_dropped"),
        "s_blocks": blocks,
    }


def raw_count_deflated(scored: list[ScoredCell], winner: ScoredCell) -> float:
    """Same deflated-ratio function, using the recorded trial count rather than clusters."""
    ratios = np.asarray(
        [float(row.by_window["validation"]["periodic_ratio"]) for row in scored],
        dtype=float,
    )
    variance = float(np.var(ratios, ddof=1)) if ratios.size > 1 else 0.0
    stats = winner.by_window["validation"]
    return deflated_sharpe(
        float(stats["periodic_ratio"]),
        int(stats["n_obs"]),
        float(stats["skew"]),
        float(stats["kurtosis_raw"]),
        len(scored),
        variance,
    )


def _ledger_row(row: ScoredCell, *, bundle_hash: str, created_utc: str, periods: float) -> Any:
    from quant_fund.proofcore.contracts import TrialLedgerRow

    stats = row.by_window["validation"]
    return TrialLedgerRow(
        trial_id=row.trial_id,
        bundle_hash=bundle_hash,
        family=row.cell.family,  # type: ignore[arg-type]
        strategy=row.cell.strategy,
        cluster_id=row.cell.cluster_id,
        created_utc=created_utc,
        n_obs=int(stats["n_obs"]),
        periods_per_year=float(periods),
        sharpe_periodic=float(stats["periodic_ratio"]),
        skew=float(stats["skew"]),
        kurtosis_raw=float(stats["kurtosis_raw"]),
        returns_sha256=_returns_sha(row.validation_returns),
    )


def persist_trials(db_path: Path, bundle: Any, rows: list[Any]) -> int:
    """Insert the bundle, then every trial row. Losers are not filtered."""
    from quant_fund.proofcore.provenance import ProvenanceDB

    with ProvenanceDB(db_path) as db:
        db.insert_bundle(bundle, None)
        for row in rows:
            db.insert_trial(row)
        stored = db.trials()
    if len(stored) < len(rows):
        raise RuntimeError(f"provenance DB stored {len(stored)} trials, sweep produced {len(rows)}")
    return len(rows)


def _sanitize(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return _sanitize(value.item())
    return value


def _public_window(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        key: stats[key]
        for key in (
            "n_obs",
            "periodic_ratio",
            "annualized_ratio",
            "ratio_ci_low",
            "ratio_ci_high",
            "ratio_ci_point",
            "compounded_return",
            "max_drawdown",
            "mean_turnover",
            "degenerate",
        )
    }


def write_results_markdown(path: Path, receipt: dict[str, Any], audit_entry_hash: str) -> None:
    """Short results note. Numbers are copied from the receipt, not recomputed."""
    gate = receipt["reality_gate"]
    selected = next(
        row for row in receipt["trials"] if row["trial_id"] == receipt["selected_trial_id"]
    )
    lines = [
        "# Reality-filter trial",
        "",
        "Research diagnostic on the pre-registered Yahoo daily sweep.",
        "Annualized ratio means Sharpe ratio: per-period mean over sample",
        "standard deviation, times the square root of the periods-per-year count.",
        "It is not a promotion and not a live-trading claim.",
        "Proper-score research elsewhere in this repo is unchanged.",
        "Reality-filter thresholds were not edited.",
        "",
        f"Verdict: `{gate['verdict']}`.",
        f"Trials recorded: {gate['n_trials']}.",
        f"Gate best ledger row: `{gate['best_trial_id']}`.",
        f"Effective trials (cluster count): {gate['n_effective_trials']}.",
        f"Gate deflated probability: {gate['dsr']}.",
        f"Gate PSR of the best ledger row: {gate['psr']}.",
        f"Gate PBO: {gate['pbo']} (unset when the JSONL ledger has no return series).",
        f"CSCV PBO on train-then-validation returns: {receipt['pbo']['pbo']}.",
        f"Deflated probability using the raw trial count: {receipt['deflated_probability_raw_count']}.",
        f"Selected trial: `{receipt['selected_trial_id']}` ({selected['strategy']}).",
        f"Dataset sha256: `{receipt['provenance']['dataset_sha256']}`.",
        f"Receipt sha256: `{receipt['receipt_sha256']}`.",
        f"Audit entry hash: `{audit_entry_hash}`.",
        "",
        "## Selected trial",
        "",
        "| window | n | compounded net return | per-period ratio | annualized ratio | CI low | CI high | max drawdown | mean turnover |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("train", "validation", "holdout"):
        window = selected["windows"][name]
        lines.append(
            "| {name} | {n} | {compounded} | {periodic} | {annual} | {lo} | {hi} | {dd} | {turn} |".format(
                name=name,
                n=window["n_obs"],
                compounded=_fmt(window["compounded_return"]),
                periodic=_fmt(window["periodic_ratio"]),
                annual=_fmt(window["annualized_ratio"]),
                lo=_fmt(window["ratio_ci_low"]),
                hi=_fmt(window["ratio_ci_high"]),
                dd=_fmt(window["max_drawdown"]),
                turn=_fmt(window["mean_turnover"]),
            )
        )
    lines.extend(
        [
            "",
            "Holdout was summarized once, after every trial had been inserted.",
            "Selection used the validation per-period ratio only.",
            "",
            "## Every trial",
            "",
            "| strategy | params | validation annualized | holdout annualized | validation compounded | holdout compounded |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in receipt["trials"]:
        lines.append(
            "| {strategy} | `{params}` | {val} | {hold} | {val_c} | {hold_c} |".format(
                strategy=row["strategy"],
                params=json.dumps(row["params"], sort_keys=True),
                val=_fmt(row["windows"]["validation"]["annualized_ratio"]),
                hold=_fmt(row["windows"]["holdout"]["annualized_ratio"]),
                val_c=_fmt(row["windows"]["validation"]["compounded_return"]),
                hold_c=_fmt(row["windows"]["holdout"]["compounded_return"]),
            )
        )
    lines.extend(
        [
            "",
            "## Data",
            "",
            f"Included symbols: {', '.join(receipt['symbols_included'])}.",
            f"Excluded symbols: {json.dumps(receipt['symbols_excluded'])}.",
            "Prices stayed out of git. The return panel hash is in the receipt.",
            "US_LIQUID is a present-day list. Quote OHLC is the adapter series;",
            "the total-return mark equals that close.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "null" if value is None else str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return "null"
    return f"{float(value):.6g}"


def run_sweep(
    *,
    spec_path: Path | None = None,
    db_path: Path | None = None,
    ledger_path: Path | None = None,
    csv_path: Path | None = None,
    receipt_path: Path | None = None,
    returns_path: Path | None = None,
    audit_dir: Path | None = None,
    results_path: Path | None = None,
    cache_dir: Path | None = None,
    bundle_dir: Path | None = None,
) -> dict[str, Any]:
    """Fetch, score every cell, insert every trial, export the ledger."""
    spec_file = spec_path or _PREREG
    spec = load_spec(spec_file)
    cells = grid_cells(spec)
    config = load_config(_ROOT / "configs" / "backtest.yaml")
    assert_cost_lock(config, spec)
    cache = cache_dir or (_ROOT / "data" / "reality_sweep")
    bars, meta = fetch_yahoo_panel(spec, cache)
    print(
        f"dataset_sha256={meta['dataset_sha256']} names={','.join(meta['symbols_included'])} "
        f"bars={meta['n_bars']}",
        flush=True,
    )
    scored = [score_cell(cell, bars, config, spec) for cell in cells]
    if len(scored) != len(cells):
        raise RuntimeError("a cell was dropped before recording")
    winner = select_winner(scored)
    lengths = {int(row.validation_returns.size) for row in scored}
    if len(lengths) != 1:
        raise RuntimeError(f"validation lengths differ: {sorted(lengths)}")

    from quant_fund.proof.bundle import build_bundle

    signal = pl.DataFrame(
        {
            "trial_id": [row.trial_id for row in scored],
            "strategy": [row.cell.strategy for row in scored],
            "cluster_id": [row.cell.cluster_id for row in scored],
            "params_json": [json.dumps(row.cell.params, sort_keys=True) for row in scored],
        }
    )
    trade = winner.equity.select(pl.col("event_time").alias("fill_time"), pl.col("nav"))
    last_stamp = bars["event_time"].max()
    if not isinstance(last_stamp, datetime):
        raise RuntimeError("panel event_time max is not a datetime")
    content_hash = str(meta["dataset_sha256"])
    from quant_fund.proofcore.contracts import (
        DataAccessRecord,
        DataManifestSummary,
        merkle_root_hex,
        sha256_hex_bytes,
    )

    manifest = DataManifestSummary(
        reads=(
            DataAccessRecord(
                dataset="reality/yahoo_us_liquid_daily",
                asof_utc=last_stamp.astimezone(UTC).isoformat(),
                params={
                    "revision_id": "YAHOO_VENDOR_ADJ",
                    "source": "yahoo",
                    "study_id": str(spec["study_id"]),
                },
                rows=int(meta["n_bars"]),
                content_sha256=content_hash,
            ),
        ),
        merkle_root=merkle_root_hex([content_hash]),
        n_reads=1,
    )
    bundle = build_bundle(
        run_kind="research",
        data_manifest=manifest,
        config_dump=spec,
        seed=int(spec["diagnostics"]["bootstrap_ci"]["seed"]),
        signal_log=signal,
        trade_log=trade,
        engine_metrics={
            "n_names": float(len(meta["symbols_included"])),
            "n_trials": float(len(scored)),
        },
        bundle_dir=bundle_dir or (_ROOT / "proofs" / "reality_sweep"),
    )
    periods = float(spec["ledger"]["periods_per_year"])
    ledger_rows = [
        _ledger_row(
            row, bundle_hash=bundle.bundle_id, created_utc=bundle.created_utc, periods=periods
        )
        for row in scored
    ]
    db = db_path or (_ROOT / "data" / "metadata" / "proofcore.duckdb")
    persist_trials(db, bundle, ledger_rows)
    _attach_window_reports(scored, spec)

    pbo = pbo_on_pre_holdout(scored, spec)
    raw_dsr = raw_count_deflated(scored, winner)
    from quant_fund.reality.report import build_reality_report

    gate = build_reality_report(ledger_rows, q=float(spec["reality_filter"]["fdr_q"]))
    from quant_fund.proofcore.cli import write_trial_csv
    from quant_fund.proofcore.provenance import ProvenanceDB

    ledger = ledger_path or (_ROOT / "research" / "reality" / "trials.jsonl")
    csv_out = csv_path or (_ROOT / "research" / "reality" / "trials.csv")
    with ProvenanceDB(db) as prov:
        stored = prov.trials()
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("w", encoding="utf-8") as handle:
        for row in stored:
            handle.write(json.dumps(row.model_dump(mode="json"), sort_keys=True) + "\n")
    write_trial_csv(stored, csv_out)

    return_rows: list[dict[str, Any]] = []
    for scored_row in scored:
        series_by_window = {
            "train": scored_row.train_returns,
            "validation": scored_row.validation_returns,
            "holdout": scored_row.holdout_returns,
        }
        for window, series in series_by_window.items():
            session_dates = scored_row.by_window[window]["dates"]
            for day, value in zip(session_dates, series, strict=True):
                return_rows.append(
                    {
                        "trial_id": scored_row.trial_id,
                        "strategy": scored_row.cell.strategy,
                        "window": window,
                        "session_date": day,
                        "simple_return": float(value),
                    }
                )
    returns_frame = pl.DataFrame(return_rows)
    returns_file = returns_path or (_ROOT / "research" / "reality" / "returns.parquet")
    returns_frame.write_parquet(returns_file)
    returns_sha = sha256_hex_bytes(returns_file.read_bytes())

    from quant_fund.utils.reproducibility import git_revision, git_worktree_sha256

    trials_public = [
        {
            "trial_id": row.trial_id,
            "strategy": row.cell.strategy,
            "cluster_id": row.cell.cluster_id,
            "params": row.cell.params,
            "returns_sha256": _returns_sha(row.validation_returns),
            "windows": {name: _public_window(row.by_window[name]) for name in row.by_window},
        }
        for row in sorted(scored, key=lambda item: item.trial_id)
    ]
    receipt_body: dict[str, Any] = {
        "schema": "dipcatcher.reality_sweep_receipt.v1",
        "data_source": "yahoo",
        "live_pnl_claim": False,
        "study_id": spec["study_id"],
        "preregistration_sha256": sha256_hex_bytes(spec_file.read_bytes()),
        "symbols_included": meta["symbols_included"],
        "symbols_excluded": meta["symbols_excluded"],
        "n_bars": meta["n_bars"],
        "n_trials": len(scored),
        "selected_trial_id": winner.trial_id,
        "bundle_id": bundle.bundle_id,
        "pbo": pbo,
        "deflated_probability_raw_count": raw_dsr,
        "reality_gate": gate.model_dump(mode="json"),
        "returns_parquet_sha256": returns_sha,
        "trials": trials_public,
        "provenance": {
            "run_id": spec["study_id"],
            "git_revision": git_revision(),
            "git_worktree_sha256": git_worktree_sha256(),
            "config_sha256": sha256_hex_bytes(spec_file.read_bytes()),
            "dataset_sha256": content_hash,
            "dataset_content_sha256": content_hash,
            "execution_claim": "historical_backtest_simulation",
            "point_in_time": False,
        },
    }
    safe = _sanitize(receipt_body)
    if not isinstance(safe, dict):
        raise RuntimeError("receipt sanitizer dropped the object")
    encoded = json.dumps(safe, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    safe["receipt_sha256"] = sha256_hex_bytes(encoded)
    receipt_file = receipt_path or (_ROOT / "research" / "reality" / "receipt.json")
    receipt_file.write_text(
        json.dumps(safe, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    from quant_fund.audit.ledger import AuditLedger
    from quant_fund.audit.record import record_research_receipt
    from quant_fund.audit.signing import Ed25519Signer

    audit_root = audit_dir or (_ROOT / "research" / "reality" / "audit")
    if (audit_root / "entries.jsonl").exists():
        raise RuntimeError("audit ledger already exists; refusing to re-sign with a new key")
    signer = Ed25519Signer.generate()
    audit = AuditLedger(audit_root, signer=signer, sign_every=1)
    entry = record_research_receipt(audit, receipt_file)
    (audit_root / "trust_pub.hex").write_text(signer.public_key_hex + "\n", encoding="utf-8")
    results = results_path or (_ROOT / "docs" / "REALITY_TRIAL_2026.md")
    write_results_markdown(results, safe, entry.entry_hash)
    print(
        f"verdict={gate.verdict} n_trials={gate.n_trials} "
        f"bundle={bundle.bundle_id} audit={entry.entry_hash}",
        flush=True,
    )
    return safe


def main() -> None:
    run_sweep()


if __name__ == "__main__":
    main()
