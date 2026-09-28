"""Regime-split strategy performance reporting.

Reporting only — does not change any strategy, signal, or backtest path.

Splits strategy period returns by three independent regime labels:

1. **Volatility terciles** — lagged rolling realized vol (PIT-safe) assigned
   via :func:`quant_fund.metrics.conformal.assign_terciles`.
2. **Benchmark drawdown state** — ``in_drawdown`` vs ``out_of_drawdown`` from
   the benchmark underwater path (:func:`drawdown_series`).
3. **H.15 interest-rate regime** — rate-level terciles from the bundled
   public-domain H.15 extract, computed **only** on dates that overlap the
   strategy calendar. Non-overlapping dates are labeled ``no_overlap`` and
   excluded from the rate-regime summaries.

All outputs are research diagnostics: ``live_pnl_claim`` is always false,
synthetic inputs are labeled, and empty inputs fail closed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_fund.metrics.conformal import assign_terciles
from quant_fund.metrics.returns import drawdown_series
from quant_fund.native import rolling_std
from quant_fund.stress.bundle import YIELD_FACTORS, load_bundle

Array = NDArray[np.float64]

_EXCLUDED_LABELS = frozenset({"unknown", "no_overlap", ""})
_DEFAULT_VOL_WINDOW = 20
_DEFAULT_H15_SERIES = "DGS10"


def equity_returns(equity: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Derive simple returns and date strings from an equity ``(event_time, nav)`` frame.

    Return length is ``n_nav - 1``. Dates correspond to the *end* of each return
    bar (``event_time[1:]``), matching the bar on which the return is realized.
    """
    if "nav" not in equity.columns:
        raise ValueError("equity frame requires a 'nav' column")
    if equity.height < 2:
        return np.asarray([], dtype=float), []
    frame = equity
    if "event_time" in equity.columns:
        frame = equity.sort("event_time")
    nav = frame["nav"].to_numpy().astype(float)
    finite = np.isfinite(nav)
    if not bool(np.all(finite)):
        # Fail closed: partial NAV paths must not silently drop bars.
        raise ValueError("equity nav contains non-finite values")
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = nav[1:] / nav[:-1] - 1.0
    if "event_time" in frame.columns:
        dates = [_date_key(t) for t in frame["event_time"].to_list()[1:]]
    else:
        dates = [str(i) for i in range(1, int(nav.size))]
    return np.asarray(rets, dtype=float), dates


def _date_key(value: object) -> str:
    """Normalize timestamps / date strings to ``YYYY-MM-DD`` when possible."""
    if hasattr(value, "strftime"):
        try:
            return str(value.strftime("%Y-%m-%d"))  # type: ignore[union-attr]
        except (TypeError, ValueError):
            pass
    text = str(value)
    # ISO datetime → date prefix.
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10]
    return text


def lagged_realized_vol(returns: Array, window: int = _DEFAULT_VOL_WINDOW) -> Array:
    """Population rolling std, shifted by one bar for PIT-safe vol labels.

    Index ``i`` uses the rolling window ending at ``i - 1``. The first
    ``window`` entries are NaN (insufficient history after the lag).
    """
    r = np.asarray(returns, dtype=float).reshape(-1)
    if window < 2:
        raise ValueError("vol window must be >= 2")
    if r.size == 0:
        return np.asarray([], dtype=float)
    raw = np.asarray(rolling_std(r, window), dtype=float)
    out = np.full(r.size, np.nan, dtype=float)
    if r.size > 1:
        out[1:] = raw[:-1]
    return out


def volatility_tercile_labels(
    returns: Array,
    *,
    window: int = _DEFAULT_VOL_WINDOW,
    cuts: Array | None = None,
) -> tuple[np.ndarray, Array]:
    """Assign ``low_vol`` / ``mid_vol`` / ``high_vol`` from lagged realized vol.

    Bars with non-finite lagged vol keep the mid label from
    :func:`assign_terciles` but are also marked so callers can exclude them
    via the finite-vol mask in :func:`summarize_by_regime` when desired.
    Non-finite vol bars are labeled ``unknown``.
    """
    vol = lagged_realized_vol(returns, window=window)
    labels = np.full(vol.size, "unknown", dtype=object)
    finite = np.isfinite(vol)
    if not bool(np.any(finite)):
        cuts_arr = np.asarray([0.0, 0.0] if cuts is None else cuts, dtype=float)
        return labels, cuts_arr
    assigned, cuts_arr = assign_terciles(vol[finite], cuts=cuts, prefix="vol")
    labels[finite] = assigned
    return labels, np.asarray(cuts_arr, dtype=float)


def benchmark_drawdown_labels(benchmark_returns: Array) -> np.ndarray:
    """Label each bar ``in_drawdown`` (dd < 0) or ``out_of_drawdown`` (dd >= 0)."""
    r = np.asarray(benchmark_returns, dtype=float).reshape(-1)
    if r.size == 0:
        return np.asarray([], dtype=object)
    dd = drawdown_series(r)
    labels = np.full(dd.size, "unknown", dtype=object)
    finite = np.isfinite(dd)
    labels[finite & (dd < 0.0)] = "in_drawdown"
    labels[finite & (dd >= 0.0)] = "out_of_drawdown"
    return labels


def bundled_h15_rate_map(series_id: str = _DEFAULT_H15_SERIES) -> dict[str, float]:
    """Flatten all crisis windows for one H.15 series into ``date -> level``.

    Duplicate dates across windows keep the first observed print (windows are
    crisis extracts; overlapping dates should agree within a vintage).
    """
    if series_id not in YIELD_FACTORS:
        raise KeyError(
            f"series {series_id!r} is not an H.15 yield series in the public-domain bundle"
        )
    bundle = load_bundle()
    series = bundle.get("series", {})
    if not isinstance(series, dict) or series_id not in series:
        raise KeyError(f"series {series_id!r} is not in the public-domain bundle")
    block = series[series_id]
    windows = block.get("windows", {}) if isinstance(block, dict) else {}
    if not isinstance(windows, dict):
        raise ValueError(f"malformed windows for {series_id}")
    out: dict[str, float] = {}
    for _window_id, window in windows.items():
        if not isinstance(window, dict):
            continue
        for row in window.get("observations", []) or []:
            if not isinstance(row, dict):
                continue
            date = str(row["date"])
            value = float(row["value"])
            if not np.isfinite(value):
                continue
            # First print wins; crisis windows rarely overlap on the same date.
            out.setdefault(date, value)
    return out


def h15_rate_regime_labels(
    dates: Sequence[str],
    *,
    series_id: str = _DEFAULT_H15_SERIES,
    rate_map: Mapping[str, float] | None = None,
    cuts: Array | None = None,
) -> tuple[np.ndarray, Array, int]:
    """Assign rate-level terciles only where ``dates`` overlap the H.15 map.

    Non-overlapping dates are labeled ``no_overlap`` and must be excluded from
    rate-regime performance summaries. Returns ``(labels, cuts, n_overlap)``.
    """
    rates = dict(rate_map) if rate_map is not None else bundled_h15_rate_map(series_id)
    n = len(dates)
    labels = np.full(n, "no_overlap", dtype=object)
    if n == 0:
        return labels, np.asarray([0.0, 0.0] if cuts is None else cuts, dtype=float), 0

    levels = np.full(n, np.nan, dtype=float)
    for i, date in enumerate(dates):
        key = _date_key(date)
        if key in rates:
            levels[i] = float(rates[key])

    overlap = np.isfinite(levels)
    n_overlap = int(np.sum(overlap))
    if n_overlap == 0:
        return labels, np.asarray([0.0, 0.0] if cuts is None else cuts, dtype=float), 0

    assigned, cuts_arr = assign_terciles(levels[overlap], cuts=cuts, prefix="rate")
    labels[overlap] = assigned
    return labels, np.asarray(cuts_arr, dtype=float), n_overlap


def summarize_by_regime(
    strategy_returns: Array,
    labels: Array,
    *,
    exclude_labels: frozenset[str] = _EXCLUDED_LABELS,
) -> dict[str, dict[str, float | int]]:
    """Per-regime counts and mean / compound return (research diagnostic)."""
    r = np.asarray(strategy_returns, dtype=float).reshape(-1)
    lab = np.asarray(labels, dtype=object).reshape(-1)
    if r.size != lab.size:
        raise ValueError("strategy_returns and labels must have identical length")
    out: dict[str, dict[str, float | int]] = {}
    if r.size == 0:
        return out
    for name in sorted({str(x) for x in lab}):
        if name in exclude_labels:
            continue
        mask = (lab == name) & np.isfinite(r)
        subset = r[mask]
        n = int(subset.size)
        if n == 0:
            out[name] = {
                "n": 0,
                "mean_ret": float("nan"),
                "total_return": float("nan"),
                "hit_rate": float("nan"),
            }
            continue
        wealth = float(np.prod(1.0 + subset) - 1.0)
        out[name] = {
            "n": n,
            "mean_ret": float(np.mean(subset)),
            "total_return": wealth,
            "hit_rate": float(np.mean(subset > 0.0)),
        }
    return out


def build_regime_performance_report(
    strategy_returns: Array,
    *,
    dates: Sequence[str] | None = None,
    benchmark_returns: Array | None = None,
    vol_window: int = _DEFAULT_VOL_WINDOW,
    h15_series_id: str = _DEFAULT_H15_SERIES,
    h15_rate_map: Mapping[str, float] | None = None,
    periods_per_year: float = 252.0,
    label: str = "BACKTEST_SIM",
    synthetic: bool = False,
) -> dict[str, Any]:
    """Assemble the regime-split performance report dict.

    Volatility terciles use lagged vol of ``benchmark_returns`` when provided,
    otherwise lagged vol of ``strategy_returns``. Drawdown state always requires
    ``benchmark_returns``. H.15 rate regimes require ``dates`` and only score
    overlapping prints.
    """
    strat = np.asarray(strategy_returns, dtype=float).reshape(-1)
    report: dict[str, Any] = {
        "label": label,
        "synthetic": bool(synthetic),
        "periods_per_year": float(periods_per_year),
        "live_pnl_claim": False,
        "research_only": True,
        "n_bars": int(strat.size),
        "vol_window": int(vol_window),
        "h15_series_id": str(h15_series_id),
    }
    if strat.size == 0:
        report["status"] = "empty_or_short"
        return report

    # --- Volatility terciles -------------------------------------------------
    vol_source = (
        np.asarray(benchmark_returns, dtype=float).reshape(-1)
        if benchmark_returns is not None
        else strat
    )
    if vol_source.size != strat.size:
        raise ValueError("benchmark_returns must match strategy_returns length")
    vol_labels, vol_cuts = volatility_tercile_labels(vol_source, window=vol_window)
    report["volatility_terciles"] = {
        "cuts": [float(vol_cuts[0]), float(vol_cuts[1])],
        "source": "benchmark" if benchmark_returns is not None else "strategy",
        "regimes": summarize_by_regime(strat, vol_labels),
        "n_labeled": int(np.sum(vol_labels != "unknown")),
    }

    # --- Benchmark drawdown state -------------------------------------------
    if benchmark_returns is None:
        report["benchmark_drawdown"] = {"status": "benchmark_required"}
    else:
        dd_labels = benchmark_drawdown_labels(vol_source)
        report["benchmark_drawdown"] = {
            "regimes": summarize_by_regime(strat, dd_labels),
            "n_in_drawdown": int(np.sum(dd_labels == "in_drawdown")),
            "n_out_of_drawdown": int(np.sum(dd_labels == "out_of_drawdown")),
        }

    # --- H.15 interest-rate regime (overlap only) ---------------------------
    if dates is None:
        report["h15_rate_regime"] = {"status": "dates_required"}
    else:
        if len(dates) != strat.size:
            raise ValueError("dates must match strategy_returns length")
        rate_labels, rate_cuts, n_overlap = h15_rate_regime_labels(
            dates,
            series_id=h15_series_id,
            rate_map=h15_rate_map,
        )
        report["h15_rate_regime"] = {
            "series_id": str(h15_series_id),
            "n_overlap": n_overlap,
            "n_no_overlap": int(strat.size - n_overlap),
            "cuts": [float(rate_cuts[0]), float(rate_cuts[1])],
            "regimes": summarize_by_regime(strat, rate_labels),
        }
    return report


def build_regime_performance_from_equity(
    equity: pl.DataFrame,
    *,
    benchmark: pl.DataFrame | None = None,
    vol_window: int = _DEFAULT_VOL_WINDOW,
    h15_series_id: str = _DEFAULT_H15_SERIES,
    h15_rate_map: Mapping[str, float] | None = None,
    periods_per_year: float = 252.0,
    label: str = "BACKTEST_SIM",
    synthetic: bool = False,
) -> dict[str, Any]:
    """Convenience wrapper: equity NAV frames → regime performance report."""
    strat_rets, dates = equity_returns(equity)
    bench_rets: np.ndarray | None = None
    if benchmark is not None:
        bench_rets, _bench_dates = equity_returns(benchmark)
        if bench_rets.size != strat_rets.size:
            raise ValueError("benchmark equity must yield the same number of returns as strategy")
    return build_regime_performance_report(
        strat_rets,
        dates=dates,
        benchmark_returns=bench_rets,
        vol_window=vol_window,
        h15_series_id=h15_series_id,
        h15_rate_map=h15_rate_map,
        periods_per_year=periods_per_year,
        label=label,
        synthetic=synthetic,
    )


def regime_performance_markdown(report: dict[str, Any]) -> str:
    """Render a compact markdown view of the regime report."""
    lines = [f"# Regime performance — {report.get('label', 'book')}", ""]
    if report.get("synthetic"):
        lines += ["> SYNTHETIC DATA. Not evidence of live profitability.", ""]
    lines.append("> Research diagnostic only; not a live P&L claim.")
    lines.append("")

    def _fmt(v: object) -> str:
        if isinstance(v, float):
            if np.isnan(v):
                return "n/a"
            return f"{v:.6g}"
        return str(v)

    lines.append(f"- n_bars: {report.get('n_bars', 0)}")
    lines.append("")

    for key, title in (
        ("volatility_terciles", "Volatility terciles"),
        ("benchmark_drawdown", "Benchmark drawdown state"),
        ("h15_rate_regime", "H.15 interest-rate regime"),
    ):
        block = report.get(key)
        if not isinstance(block, dict):
            continue
        lines.append(f"## {title}")
        if "status" in block:
            lines.append(f"- status: {block['status']}")
            lines.append("")
            continue
        for meta_key in (
            "source",
            "cuts",
            "n_labeled",
            "n_in_drawdown",
            "n_out_of_drawdown",
            "series_id",
            "n_overlap",
            "n_no_overlap",
        ):
            if meta_key in block:
                lines.append(f"- {meta_key}: {_fmt(block[meta_key])}")
        regimes = block.get("regimes") or {}
        if regimes:
            lines.append("")
            lines.append("| regime | n | mean_ret | total_return | hit_rate |")
            lines.append("|---|---:|---:|---:|---:|")
            for name, stats in regimes.items():
                lines.append(
                    f"| {name} | {stats['n']} | {_fmt(stats['mean_ret'])} "
                    f"| {_fmt(stats['total_return'])} | {_fmt(stats['hit_rate'])} |"
                )
        lines.append("")
    return "\n".join(lines)


def write_regime_performance_md(path: Path, report: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(regime_performance_markdown(report))
    return path


__all__ = [
    "benchmark_drawdown_labels",
    "build_regime_performance_from_equity",
    "build_regime_performance_report",
    "bundled_h15_rate_map",
    "equity_returns",
    "h15_rate_regime_labels",
    "lagged_realized_vol",
    "regime_performance_markdown",
    "summarize_by_regime",
    "volatility_tercile_labels",
    "write_regime_performance_md",
]
