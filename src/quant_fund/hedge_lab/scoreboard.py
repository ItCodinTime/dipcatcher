"""Economic book scoreboard for the Artificial Hedge fund lab.

This catalog is the paper/backtest analytics path: Sharpe, Sortino, Calmar,
max drawdown, CAGR, PSR, MinTRL. It is not a research-family blob. SYNTHETIC
equity is labeled and cannot promote.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.hedge_lab.resources import ram_plan
from quant_fund.metrics.overfitting import (
    min_track_record_length,
    moments_from_returns,
    probabilistic_sharpe,
)
from quant_fund.metrics.returns import (
    annualized_vol,
    cagr,
    calmar_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    wealth_index,
)

Array = NDArray[np.float64]
CATALOG = "hedge_lab_analytics"


def book_economic_scoreboard(
    returns: Array,
    *,
    periods_per_year: float = 252.0,
    data_source: str = "file",
    benchmark_returns: Array | None = None,
) -> dict[str, Any]:
    """Sharpe / Sortino / Calmar / DD on a simulated book path."""
    r = np.asarray(returns, dtype=float).reshape(-1)
    sr = sharpe_ratio(r, periods_per_year=periods_per_year)
    sharpe = float(sr["sharpe"])
    n = int(r.size)
    _sig, skew, kurt = moments_from_returns(r)
    psr = probabilistic_sharpe(sharpe, 0.0, n, float(skew), float(kurt))
    min_trl = min_track_record_length(sharpe, float(skew), float(kurt))
    blob: dict[str, Any] = {
        "catalog": CATALOG,
        "data_source": data_source,
        "research_only": True,
        "execution_claim": "paper_backtest",
        "sharpe": sharpe,
        "flag_high_sharpe": bool(sr["flag_high_sharpe"]),
        "sortino": float(sortino_ratio(r, periods_per_year=periods_per_year)),
        "calmar": float(calmar_ratio(r, periods_per_year=periods_per_year)),
        "max_drawdown": float(max_drawdown(r)) if n else 0.0,
        "cagr": float(cagr(r, periods_per_year)) if n else float("nan"),
        "total_return": (
            float(wealth_index(r)[-1] - 1.0) if n and np.all(np.isfinite(r)) else float("nan")
        ),
        "ann_vol": float(annualized_vol(r, periods_per_year)) if n else float("nan"),
        "n_returns": n,
        "psr_vs_zero": float(psr),
        "min_trl_days": float(min_trl),
        "synthetic_not_promotable": str(data_source).upper() == "SYNTHETIC",
    }
    if benchmark_returns is not None:
        b = np.asarray(benchmark_returns, dtype=float).reshape(-1)
        if b.size == r.size and b.size >= 2:
            active = r - b
            ir = sharpe_ratio(active, periods_per_year=periods_per_year)
            blob["information_ratio"] = float(ir["sharpe"])
            blob["active_ann_vol"] = float(
                annualized_vol(active, periods_per_year=periods_per_year)
            )
        else:
            blob["information_ratio"] = float("nan")
            blob["active_ann_vol"] = float("nan")
    return blob


def moving_block_bootstrap_ci(
    returns: Array,
    *,
    n_boot: int | None = None,
    block: int | None = None,
    seed: int = 7,
    periods_per_year: float = 252.0,
) -> dict[str, Any]:
    """Vectorized moving-block bootstrap of Sharpe and Calmar.

    Workspace size tracks ``ram_plan()`` so a 128 GiB box uses tens of GiB
    without pinning the OS. Paths + wealth are the peak pair; ``n_boot`` is
    capped so that pair stays inside the planned workspace.
    """
    r = np.asarray(returns, dtype=float).reshape(-1)
    n = int(r.size)
    if n < 20:
        return {"status": "insufficient_returns", "n": n, "n_boot": 0}
    plan = ram_plan()
    workspace = int(plan["workspace_bytes"])
    block_len = int(block) if block is not None else max(5, min(21, n // 10))
    n_blocks = int(np.ceil(n / block_len))
    bytes_peak_per_boot = int(n * 8 * 2)
    auto_boot = int(workspace * 0.35 // max(bytes_peak_per_boot, 16))
    auto_boot = int(min(max(auto_boot, 20_000), 20_000_000))
    wanted = int(n_boot) if n_boot is not None else auto_boot
    wanted = int(min(max(wanted, 1), auto_boot if n_boot is None else max(wanted, 1)))
    if n_boot is not None:
        wanted = min(int(n_boot), max(auto_boot, 20_000))
    rng = np.random.default_rng(int(seed))
    actual = int(wanted)
    last_error: BaseException | None = None
    sharpes = np.asarray([], dtype=float)
    calmars = np.asarray([], dtype=float)
    while actual >= 5_000 or (n_boot is not None and actual >= 64):
        try:
            starts = rng.integers(0, n - block_len + 1, size=(actual, n_blocks), dtype=np.int64)
            offsets = np.arange(block_len, dtype=np.int64)
            idx = (starts[..., None] + offsets).reshape(actual, n_blocks * block_len)[:, :n]
            del starts
            paths = r[idx]
            del idx
            mean = paths.mean(axis=1)
            std = paths.std(axis=1, ddof=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                sharpes = np.where(std > 0, mean / std * np.sqrt(periods_per_year), np.nan)
            wealth = np.cumprod(1.0 + paths, axis=1)
            del paths
            peak = np.maximum.accumulate(wealth, axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                dd = np.where(peak > 0, wealth / peak - 1.0, np.nan)
            mdd = np.min(dd, axis=1)
            years = n / periods_per_year
            with np.errstate(divide="ignore", invalid="ignore"):
                growth = np.where(
                    wealth[:, -1] > 0, wealth[:, -1] ** (1.0 / years) - 1.0, np.nan
                )
                calmars = np.where(np.abs(mdd) > 0, growth / np.abs(mdd), np.nan)
            del wealth, peak, dd
            break
        except MemoryError as exc:
            last_error = exc
            actual = max(actual // 2, 1)
    if sharpes.size == 0:
        raise MemoryError("block-bootstrap workspace could not be allocated") from last_error
    return {
        "status": "ok",
        "n": n,
        "n_boot": int(actual),
        "block": block_len,
        "workspace_gib_requested": float(plan["workspace_gib"]),
        "ram_total_gib": float(plan["total_gib"]),
        "sharpe_p05": float(np.nanpercentile(sharpes, 5)),
        "sharpe_p50": float(np.nanpercentile(sharpes, 50)),
        "sharpe_p95": float(np.nanpercentile(sharpes, 95)),
        "calmar_p05": float(np.nanpercentile(calmars, 5)),
        "calmar_p50": float(np.nanpercentile(calmars, 50)),
        "calmar_p95": float(np.nanpercentile(calmars, 95)),
        "catalog": CATALOG,
    }
