"""Stat-arb evaluation lane: planted-pair screen + signal-alignment scoring.

Runs the cointegration screen on a labeled SYNTHETIC panel with one planted
cointegrated pair, then scores the point-in-time z-score signal against
forward spread changes. Reporting stays in proper-score framing —
correlation/alignment statistics and detection truth — never a trading or
live-execution claim. ``write_pairs_receipt`` seals the JSON evidence under
``receipts/`` the same way ``cross_sectional.write_rankic_receipt`` does.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy import stats as sstats

from quant_fund.research.catalog import family_blob_forbidden_metrics_absent
from quant_fund.research.fleet_eval import _atomic_write_text
from quant_fund.research.pairs.cointegration import screen_pairs
from quant_fund.research.pairs.fixtures import planted_pair_panel
from quant_fund.research.pairs.pit import pit_pair_signals
from quant_fund.research.pairs.spread import ou_half_life
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

Array = NDArray[np.float64]

PAIRS_SCHEMA = "stat_arb_pairs_eval.v1"

_NAN = float("nan")


def _is_planted(row: Mapping[str, Any], pair: tuple[int, int]) -> bool:
    return {int(row["a"]), int(row["b"])} == {pair[0], pair[1]}


def _signal_alignment(
    z: Array,
    spread: Array,
    horizon: int,
    entry: float,
) -> dict[str, float]:
    """Alignment of the reversion signal with forward spread changes.

    The signal direction is ``-z`` (short a rich spread). The target is the
    ``horizon``-step forward change of the estimated spread — named with a
    ``fwd_`` prefix per the repo leakage contract (it is an evaluation
    target, not a contemporaneous feature). ``band_hit_rate`` is the share
    of in-band observations (``|z| >= entry``) whose forward spread change
    has the reversion sign.
    """
    h = int(horizon)
    if h < 1:
        raise ValueError("horizon must be >= 1")
    fwd_spread_change = np.full(spread.size, np.nan)
    if spread.size > h:
        fwd_spread_change[:-h] = spread[h:] - spread[:-h]
    mask = np.isfinite(z) & np.isfinite(fwd_spread_change)
    n = int(mask.sum())
    out = {
        "n_obs": float(n),
        "pearson_ic": _NAN,
        "spearman_ic": _NAN,
        "band_hit_rate": _NAN,
        "n_band_obs": 0.0,
    }
    if n < 10:
        return out
    signal_dir = -z[mask]
    fwd = fwd_spread_change[mask]
    if float(np.std(signal_dir)) <= 0 or float(np.std(fwd)) <= 0:
        return out
    out["pearson_ic"] = float(sstats.pearsonr(signal_dir, fwd).statistic)
    out["spearman_ic"] = float(sstats.spearmanr(signal_dir, fwd).statistic)
    in_band = np.abs(z[mask]) >= float(entry)
    if int(in_band.sum()) >= 10:
        out["n_band_obs"] = float(in_band.sum())
        out["band_hit_rate"] = float(np.mean(np.sign(signal_dir[in_band]) == np.sign(fwd[in_band])))
    return out


def run_pairs_eval(
    *,
    seed: int = 0,
    n_assets: int = 8,
    n_dates: int = 600,
    window: int = 120,
    z_window: int = 60,
    min_corr: float = 0.5,
    alpha: float = 0.05,
    entry: float = 2.0,
    exit: float = 0.5,
    hedge_method: str = "ols",
    kalman_q: float = 1e-4,
    eval_horizon: int = 1,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Screen the planted panel and score the PIT signal; frame + receipt.

    The frame is the cointegration screen table (one row per tested pair,
    sorted by BH-adjusted p-value). The receipt carries the planted-pair
    detection truth (rank + pass flags), the signal-alignment statistics
    (Pearson/Spearman IC of ``-z`` against the forward spread change, and
    the in-band hit rate), and labeled descriptive stats.
    """
    panel = planted_pair_panel(n_assets=n_assets, n_dates=n_dates, seed=seed)
    prices = np.asarray(panel["prices"], dtype=float)
    pair = cast(tuple[int, int], panel["pair"])
    frame = screen_pairs(prices, min_corr=min_corr, alpha=alpha)
    rows = frame.to_dicts()
    planted_rows = [r for r in rows if _is_planted(r, pair)]
    planted_rank = rows.index(planted_rows[0]) + 1 if planted_rows else -1
    planted = planted_rows[0] if planted_rows else {}
    i, j = pair
    sig = pit_pair_signals(
        prices[:, i],
        prices[:, j],
        window=window,
        z_window=z_window,
        entry=entry,
        exit=exit,
        method=hedge_method,
        kalman_q=kalman_q,
    )
    spread = sig["spread"]
    spread_tail = spread[np.isfinite(spread)]
    est_half_life = ou_half_life(spread_tail) if spread_tail.size >= 60 else _NAN
    alignment = _signal_alignment(sig["z"], spread, eval_horizon, entry)
    z = sig["z"]
    z_valid = z[np.isfinite(z)]
    descriptive = {
        "n_assets": float(n_assets),
        "n_dates": float(n_dates),
        "n_pairs_tested": float(len(rows)),
        "n_pairs_pass_bh": float(np.sum([bool(r["passes_bh"]) for r in rows])),
        "n_pairs_pass_eg_cv": float(np.sum([bool(r["passes_eg_cv"]) for r in rows])),
        "mean_abs_z": float(np.mean(np.abs(z_valid))) if z_valid.size else _NAN,
        "share_in_position": float(np.mean(np.abs(sig["position"]) > 0)),
        "n_entries": float(np.sum(np.diff((np.abs(sig["position"]) > 0).astype(int)) > 0)),
        "est_half_life": est_half_life,
        "true_half_life": float(cast(float, panel["true_half_life"])),
    }
    inputs_sha256 = hash_bytes(prices.tobytes())
    receipt: dict[str, Any] = {
        "schema": PAIRS_SCHEMA,
        "kind": "stat_arb_pairs_eval",
        "data_label": "SYNTHETIC",
        "live_pnl_claim": False,
        "generated_at": datetime.now(UTC).isoformat(),
        "git_revision": git_revision(),
        "seed": int(seed),
        "params": {
            "n_assets": int(n_assets),
            "n_dates": int(n_dates),
            "window": int(window),
            "z_window": int(z_window),
            "min_corr": float(min_corr),
            "alpha": float(alpha),
            "entry": float(entry),
            "exit": float(exit),
            "hedge_method": str(hedge_method),
            "kalman_q": float(kalman_q),
            "eval_horizon": int(eval_horizon),
        },
        "inputs_sha256": inputs_sha256,
        "planted": {
            "pair": [int(i), int(j)],
            "detected": bool(planted_rank == 1 and planted.get("passes_bh") is True),
            "rank_by_p_bh": int(planted_rank),
            "row": planted,
        },
        "alignment": alignment,
        "descriptive": descriptive,
        "screen": rows,
        "notes": [
            "adf_pvalue applies the ordinary DF surface to estimated residuals "
            "and is liberal; passes_eg_cv (tau < EG 5% CV) is the strict gate.",
            "BH/Bonferroni control error within the post-correlation-filter family only.",
            "All results are SYNTHETIC correctness evidence, not market data.",
        ],
    }
    return frame, receipt


def write_pairs_receipt(
    receipt: Mapping[str, Any],
    receipts_dir: Path | str = Path("receipts"),
) -> Path:
    """Seal a pairs-eval receipt as ``receipts/stat_arb_pairs_eval_<hash>.json``."""
    research_blob = {key: value for key, value in receipt.items() if key != "live_pnl_claim"}
    if (
        receipt.get("schema") != PAIRS_SCHEMA
        or receipt.get("data_label") != "SYNTHETIC"
        or receipt.get("live_pnl_claim") is not False
        or not isinstance(receipt.get("screen"), list)
        or not receipt["screen"]
        or not family_blob_forbidden_metrics_absent(research_blob)
    ):
        raise ValueError("pairs receipt violates its synthetic research contract")
    canonical = json.loads(canonical_json_bytes(dict(receipt)))
    digest = hash_bytes(canonical_json_bytes(canonical))
    payload = {**canonical, "receipt_sha256": digest}
    path = Path(receipts_dir) / f"stat_arb_pairs_eval_{digest[:16]}.json"
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def format_pairs_table(frame: pl.DataFrame, max_rows: int = 15) -> str:
    """Compact text table of the screen for the CLI echo."""
    header = "a | b | corr | dir | hedge | tau | p_adf | p_bh | hl | pass_eg | pass_bh"
    lines = [header]
    for row in frame.head(max_rows).iter_rows(named=True):
        hl = row["half_life"]
        hl_s = "inf" if hl is None or not np.isfinite(hl) else f"{hl:.1f}"
        lines.append(
            f"{row['a']} | {row['b']} | {row['corr_diff']:+.3f} | {row['direction']} | "
            f"{row['hedge_ratio']:+.3f} | {row['adf_tau']:+.2f} | "
            f"{row['adf_pvalue']:.3g} | {row['p_bh']:.3g} | {hl_s} | "
            f"{row['passes_eg_cv']} | {row['passes_bh']}"
        )
    return "\n".join(lines)
