"""Systematic estimator-identity prover (SYNTHETIC, research_only).

The repo encodes microstructure identities manually: session candles must
reconstruct the daily OHLC envelope, session volumes must conserve the
daily total, top-of-book metrics must satisfy their defining algebra, and
paired estimator implementations (polars vs NumPy Kyle λ and
Cont–Kukanov–Stoikov OFI) must agree exactly. This module enumerates
those identities — introspecting ``research.catalog`` guard names that
share an identity family — then proves each on ``n_trials`` seeded
SYNTHETIC bundles and records the max-abs residual per identity into an
immutable JSON receipt (``verify-identities`` CLI / CI gate candidate).

Residuals and pass/fail verdicts only — never a Sharpe/P&L/NAV claim.
Every draw is SYNTHETIC correctness evidence, not market evidence.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from importlib import import_module
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_fund.microstructure.book_metrics import (
    concentration_top_finite_rate,
    depth_shape_finite_rate,
    metric_rows_from_frame,
    queue_priority_finite_rate,
    side_notional_finite_rate,
    tob_size_share_finite_rate,
)
from quant_fund.microstructure.synthetic_lob import (
    aggregate_session_book_to_daily,
    synthesize_l2_from_bars,
    synthesize_session_l2,
    synthesize_snapshots_from_bars,
)
from quant_fund.northset.candles import candle_geometry
from quant_fund.northset.estimators import (
    kyle_lambda,
    ohlc_variance_frame,
    order_flow_imbalance,
    queue_imbalance,
    session_realized_variance,
    vpin_proxy,
)
from quant_fund.northset.identities import (
    book_uncrossed_rate,
    gap_finite_rate,
    ohlc_identity_rate,
    session_candles_from_daily,
    session_chain_rate,
    session_reconstructs_daily_rate,
    session_volume_conservation_rate,
)
from quant_fund.northset.kyle_ofi import (
    cont_ofi_by_security,
    fuse_bars_l2_kyle_frame,
    kyle_lambda_by_date,
    kyle_lambda_dispersion,
    kyle_lambda_ofi_depth_corr,
    kyle_lambda_ols,
)
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent
from quant_fund.schemas.order_book import OrderBookSnapshot
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

Array = NDArray[np.float64]

IDENTITY_SWEEP_SCHEMA_VERSION = 1

_CATALOG_MODULES = (
    "quant_fund.research.catalog.session",
    "quant_fund.research.catalog.candle",
    "quant_fund.research.catalog.kyle",
    "quant_fund.research.catalog.hypotheses",
    "quant_fund.research.catalog.dispatch",
    "quant_fund.research.catalog.rates",
    "quant_fund.research.catalog.ic_packs",
    "quant_fund.research.catalog.receipt",
    "quant_fund.research.catalog.sweep",
    "quant_fund.research.catalog.never_equate",
    "quant_fund.research.catalog.never_equate_gap",
)
_IMPL_MODULES = (
    "quant_fund.northset.identities",
    "quant_fund.northset.estimators",
    "quant_fund.northset.kyle_ofi",
    "quant_fund.northset.candles",
    "quant_fund.microstructure.book_metrics",
    "quant_fund.microstructure.synthetic_lob",
)

_GUARD_RE = re.compile(r"^([a-z0-9]+(?:_[a-z0-9]+)*)_(?:honesty|consistency)_errors$")
# Greedy left stem: nested pairs like ``dm_gk_vs_park_p_vs_gap_finite`` bind
# as (dm_gk_vs_park_p, gap_finite) — the catalog's outermost pair.
_NEVER_EQUATE_RE = re.compile(
    r"^([a-z0-9]+(?:_[a-z0-9]+)*)_vs_([a-z0-9]+(?:_[a-z0-9]+)*)"
    r"_never_equate_(?:honesty|consistency)_errors$"
)


@dataclass(frozen=True)
class SyntheticBundle:
    """One seeded SYNTHETIC draw shared by every identity in a trial.

    ``bars`` are daily OHLCV; ``session`` / ``session_book`` are the
    reconstructed intraday views; ``book`` / ``session_daily`` are the L2
    metric panel and its per-parent aggregate; ``fused`` is the Kyle/OFI
    research frame; ``snapshots`` are the raw books behind ``book``.
    """

    bars: pl.DataFrame
    session: pl.DataFrame
    book: pl.DataFrame
    session_book: pl.DataFrame
    session_daily: pl.DataFrame
    fused: pl.DataFrame
    snapshots: tuple[OrderBookSnapshot, ...]
    n_session_candles: int


@dataclass(frozen=True)
class IdentitySpec:
    """One proven identity: residual = |left - right| or a rate shortfall.

    ``evaluate`` returns the per-trial residual (0.0 == identity holds
    exactly); it must raise on degenerate input (fail-closed) rather than
    fabricate a pass. ``tolerance`` is the max residual that still passes.
    """

    name: str
    family: str
    statement: str
    tolerance: float
    evaluate: Callable[[SyntheticBundle], float]


# ---------------------------------------------------------------------------
# Synthetic bundle generator
# ---------------------------------------------------------------------------


def make_synthetic_bundle(
    seed: int,
    *,
    n_days: int = 26,
    n_session_candles: int = 6,
    depth: int = 4,
    securities: tuple[str, ...] = ("SYNTH_A", "SYNTH_B", "SYNTH_C", "SYNTH_D"),
) -> SyntheticBundle:
    """Deterministic SYNTHETIC OHLCV + session + L2 bundle for one seed.

    Honest generator: GBM closes, high/low padded around the open/close
    range (strict ``high > low``), lognormal volume. All downstream views
    (session candles, L2 panels, fused Kyle frame) derive from the same
    bars so every catalog identity should hold exactly.
    """
    if n_days < 22:
        raise ValueError("n_days must be >= 22 (Kyle λ needs >= 20 finite obs per name)")
    if n_session_candles < 2:
        raise ValueError("n_session_candles must be >= 2")
    if depth < 2:
        raise ValueError("depth must be >= 2 (depth-shape identities need multi-level books)")
    if not securities:
        raise ValueError("securities must be non-empty")
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, Any]] = []
    start = datetime(2024, 1, 2, 16, 0, tzinfo=UTC)
    for s_idx, sid in enumerate(securities):
        px = 40.0 + 30.0 * s_idx + float(rng.uniform(0.0, 5.0))
        for i in range(n_days):
            open_ = px
            close = max(open_ * (1.0 + float(rng.normal(0.0, 0.012))), 0.5)
            pad = open_ * 1e-4
            high = max(open_, close) + open_ * abs(float(rng.normal(0.0, 0.004))) + pad
            low = min(open_, close) - open_ * abs(float(rng.normal(0.0, 0.004))) - pad
            ts = start + timedelta(days=i)
            rows.append(
                {
                    "security_id": sid,
                    "symbol": sid,
                    "event_time": ts,
                    "available_time": ts,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": float(rng.lognormal(12.0, 0.5)),
                }
            )
            px = close
    bars = pl.DataFrame(rows).sort(["security_id", "event_time"])
    session = session_candles_from_daily(bars, n_candles=n_session_candles, seed=seed)
    snapshots = tuple(synthesize_snapshots_from_bars(bars, depth=depth, seed=seed))
    book = synthesize_l2_from_bars(bars, depth=depth, seed=seed)
    session_book = synthesize_session_l2(session, depth=depth, seed=seed)
    session_daily = aggregate_session_book_to_daily(session_book)
    fused = fuse_bars_l2_kyle_frame(bars, book)
    return SyntheticBundle(
        bars=bars,
        session=session,
        book=book,
        session_book=session_book,
        session_daily=session_daily,
        fused=fused,
        snapshots=snapshots,
        n_session_candles=int(n_session_candles),
    )


# ---------------------------------------------------------------------------
# Residual helpers — every evaluator returns |lhs - rhs| or a rate shortfall
# ---------------------------------------------------------------------------


def _rate_residual(rate: float, name: str) -> float:
    """|1 - rate|; NaN rate is fail-closed, not a silent pass."""
    if not math.isfinite(rate):
        raise ValueError(f"{name} undefined on this bundle")
    return abs(1.0 - float(rate))


def _col(frame: pl.DataFrame, name: str) -> Array:
    if name not in frame.columns:
        raise ValueError(f"frame missing required column {name!r}")
    return np.asarray(frame[name].to_numpy(), dtype=float)


def _max_abs_diff(a: Array, b: Array) -> float:
    if a.size != b.size:
        raise ValueError("identity operands must align")
    left_finite = np.isfinite(a)
    right_finite = np.isfinite(b)
    if not np.array_equal(left_finite, right_finite):
        raise ValueError("identity operands have mismatched finite rows")
    mask = left_finite
    if int(mask.sum()) == 0:
        raise ValueError("no aligned finite rows to prove identity on")
    return float(np.max(np.abs(a[mask] - b[mask])))


def _bound_violation(values: Array, *, lo: float, hi: float) -> float:
    """Magnitude by which ``values`` leaves [lo, hi]; 0 when inside."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("no finite values to bound")
    below = np.maximum(lo - finite, 0.0)
    above = np.maximum(finite - hi, 0.0)
    return float(np.max(np.maximum(below, above)))


def _book_rows(bundle: SyntheticBundle) -> list[dict[str, float]]:
    return metric_rows_from_frame(bundle.book)


# ---------------------------------------------------------------------------
# Rate / structure identities (northset.identities + synthetic_lob)
# ---------------------------------------------------------------------------


def _r_ohlc_identity(b: SyntheticBundle) -> float:
    return _rate_residual(ohlc_identity_rate(b.bars), "ohlc_identity_rate(bars)")


def _r_session_ohlc_identity(b: SyntheticBundle) -> float:
    return _rate_residual(ohlc_identity_rate(b.session), "ohlc_identity_rate(session)")


def _r_session_reconstructs(b: SyntheticBundle) -> float:
    return _rate_residual(
        session_reconstructs_daily_rate(b.bars, b.session), "session_reconstructs_daily_rate"
    )


def _r_session_volume(b: SyntheticBundle) -> float:
    return _rate_residual(
        session_volume_conservation_rate(b.bars, b.session), "session_volume_conservation_rate"
    )


def _r_session_chain(b: SyntheticBundle) -> float:
    return _rate_residual(session_chain_rate(b.session), "session_chain_rate")


def _r_gap_finite(b: SyntheticBundle) -> float:
    return _rate_residual(gap_finite_rate(b.bars), "gap_finite_rate")


def _r_book_uncrossed(b: SyntheticBundle) -> float:
    return _rate_residual(book_uncrossed_rate(b.book), "book_uncrossed_rate(book)")


def _r_session_book_uncrossed(b: SyntheticBundle) -> float:
    return _rate_residual(book_uncrossed_rate(b.session_book), "book_uncrossed_rate(session_book)")


def _r_session_book_count(b: SyntheticBundle) -> float:
    """Every (security, parent) must carry exactly ``n_session_candles`` snaps."""
    if b.session_book.height == 0:
        raise ValueError("session_book is empty")
    counts = b.session_book.group_by(["security_id", "parent_event_time"]).agg(
        pl.len().alias("n_snaps")
    )
    dev = (counts["n_snaps"] - b.n_session_candles).abs()
    worst = dev.max()
    if worst is None:
        raise ValueError("session_book produced no parent groups")
    return float(cast(int, worst))


def _r_session_book_vpin_bound(b: SyntheticBundle) -> float:
    return _bound_violation(_col(b.session_daily, "session_book_vpin"), lo=0.0, hi=1.0)


# ---------------------------------------------------------------------------
# Finite-rate coverage identities (book_metrics eligibility ratios)
# ---------------------------------------------------------------------------


def _r_depth_shape_finite(b: SyntheticBundle) -> float:
    return _rate_residual(depth_shape_finite_rate(_book_rows(b)), "depth_shape_finite_rate")


def _r_concentration_top_finite(b: SyntheticBundle) -> float:
    return _rate_residual(
        concentration_top_finite_rate(_book_rows(b)), "concentration_top_finite_rate"
    )


def _r_queue_priority_finite(b: SyntheticBundle) -> float:
    return _rate_residual(queue_priority_finite_rate(_book_rows(b)), "queue_priority_finite_rate")


def _r_side_notional_finite(b: SyntheticBundle) -> float:
    return _rate_residual(side_notional_finite_rate(_book_rows(b)), "side_notional_finite_rate")


def _r_tob_size_share_finite(b: SyntheticBundle) -> float:
    return _rate_residual(tob_size_share_finite_rate(_book_rows(b)), "tob_size_share_finite_rate")


# ---------------------------------------------------------------------------
# Top-of-book algebraic identities (book_metrics defining formulas)
# ---------------------------------------------------------------------------


def _r_mid(b: SyntheticBundle) -> float:
    return _max_abs_diff(
        _col(b.book, "mid"), 0.5 * (_col(b.book, "best_bid") + _col(b.book, "best_ask"))
    )


def _r_spread(b: SyntheticBundle) -> float:
    spread = _col(b.book, "best_ask") - _col(b.book, "best_bid")
    return max(
        _max_abs_diff(_col(b.book, "spread"), spread),
        _max_abs_diff(_col(b.book, "quoted_spread"), spread),
        _max_abs_diff(_col(b.book, "effective_spread"), spread),
    )


def _r_half_spread(b: SyntheticBundle) -> float:
    half = 0.5 * (_col(b.book, "best_ask") - _col(b.book, "best_bid"))
    mid = _col(b.book, "mid")
    return max(
        _max_abs_diff(_col(b.book, "half_spread"), half),
        _max_abs_diff(mid + _col(b.book, "half_spread"), _col(b.book, "best_ask")),
        _max_abs_diff(mid - _col(b.book, "half_spread"), _col(b.book, "best_bid")),
    )


def _r_spread_bps(b: SyntheticBundle) -> float:
    spread_bps = 1e4 * _col(b.book, "spread") / _col(b.book, "mid")
    return max(
        _max_abs_diff(_col(b.book, "spread_bps"), spread_bps),
        _max_abs_diff(_col(b.book, "quoted_spread_bps"), spread_bps),
        _max_abs_diff(_col(b.book, "half_spread_bps"), 0.5 * spread_bps),
        _max_abs_diff(
            _col(b.book, "spread_over_mid"), _col(b.book, "spread") / _col(b.book, "mid")
        ),
    )


def _r_microprice(b: SyntheticBundle) -> float:
    top_bid = _col(b.book, "top_bid_size")
    top_ask = _col(b.book, "top_ask_size")
    w = top_bid / (top_bid + top_ask)
    balanced = w * _col(b.book, "best_ask") + (1.0 - w) * _col(b.book, "best_bid")
    return max(
        _max_abs_diff(_col(b.book, "microprice"), balanced),
        _max_abs_diff(_col(b.book, "microprice_weight_balance"), w),
    )


def _r_microprice_minus_mid(b: SyntheticBundle) -> float:
    delta = _col(b.book, "microprice") - _col(b.book, "mid")
    return max(
        _max_abs_diff(_col(b.book, "microprice_minus_mid"), delta),
        _max_abs_diff(_col(b.book, "microprice_minus_mid_bps"), 1e4 * delta / _col(b.book, "mid")),
    )


def _r_imbalance_top(b: SyntheticBundle) -> float:
    top_bid = _col(b.book, "top_bid_size")
    top_ask = _col(b.book, "top_ask_size")
    imb = (top_bid - top_ask) / (top_bid + top_ask)
    qi = queue_imbalance(b.book)["queue_imbalance"].to_numpy().astype(float)
    return max(
        _max_abs_diff(_col(b.book, "imbalance_top"), imb),
        _max_abs_diff(_col(b.book, "touch_size_imbalance"), imb),
        _max_abs_diff(qi, imb),
    )


def _r_imbalance_depth(b: SyntheticBundle) -> float:
    bid_d = _col(b.book, "bid_depth")
    ask_d = _col(b.book, "ask_depth")
    imb = (bid_d - ask_d) / (bid_d + ask_d)
    return max(
        _max_abs_diff(_col(b.book, "imbalance_depth"), imb),
        _max_abs_diff(_col(b.book, "depth_imbalance_abs"), np.abs(imb)),
    )


def _r_depth_equals_level_sum(b: SyntheticBundle) -> float:
    """Panel ``bid_depth``/``ask_depth`` must equal the summed snapshot levels."""
    if not b.snapshots:
        raise ValueError("bundle has no snapshots")
    panel = {
        (str(r["security_id"]), r["event_time"]): (float(r["bid_depth"]), float(r["ask_depth"]))
        for r in b.book.select("security_id", "event_time", "bid_depth", "ask_depth").iter_rows(
            named=True
        )
    }
    worst = 0.0
    for snap in b.snapshots:
        key = (snap.security_id, snap.event_time)
        if key not in panel:
            raise ValueError(f"book panel missing snapshot row {key!r}")
        bid_sum = sum(level.size for level in snap.bids)
        ask_sum = sum(level.size for level in snap.asks)
        panel_bid, panel_ask = panel[key]
        worst = max(worst, abs(panel_bid - bid_sum), abs(panel_ask - ask_sum))
    return float(worst)


def _r_notional_algebra(b: SyntheticBundle) -> float:
    best_bid = _col(b.book, "best_bid")
    best_ask = _col(b.book, "best_ask")
    top_bid = _col(b.book, "top_bid_size")
    top_ask = _col(b.book, "top_ask_size")
    bid_d = _col(b.book, "bid_depth")
    ask_d = _col(b.book, "ask_depth")
    tob = best_bid * top_bid + best_ask * top_ask
    bid_notional = _col(b.book, "side_notional_proxy_bid")
    ask_notional = _col(b.book, "side_notional_proxy_ask")
    return max(
        _max_abs_diff(_col(b.book, "top_of_book_notional_proxy"), tob),
        _max_abs_diff(bid_notional, best_bid * bid_d),
        _max_abs_diff(ask_notional, best_ask * ask_d),
        _max_abs_diff(
            _col(b.book, "notional_imbalance"),
            (bid_notional - ask_notional) / (bid_notional + ask_notional),
        ),
        _max_abs_diff(_col(b.book, "tob_notional_share"), tob / (bid_notional + ask_notional)),
        _max_abs_diff(_col(b.book, "tob_size_share"), (top_bid + top_ask) / (bid_d + ask_d)),
    )


def _r_concentration_top(b: SyntheticBundle) -> float:
    return max(
        _max_abs_diff(
            _col(b.book, "bid_size_concentration_top"),
            _col(b.book, "top_bid_size") / _col(b.book, "bid_depth"),
        ),
        _max_abs_diff(
            _col(b.book, "ask_size_concentration_top"),
            _col(b.book, "top_ask_size") / _col(b.book, "ask_depth"),
        ),
    )


def _r_queue_priority(b: SyntheticBundle) -> float:
    top_bid = _col(b.book, "top_bid_size")
    top_ask = _col(b.book, "top_ask_size")
    return max(
        _max_abs_diff(
            _col(b.book, "queue_priority_proxy"), top_bid / (top_bid + _col(b.book, "bid_depth"))
        ),
        _max_abs_diff(
            _col(b.book, "ask_queue_priority_proxy"),
            top_ask / (top_ask + _col(b.book, "ask_depth")),
        ),
    )


# ---------------------------------------------------------------------------
# Paired estimator identities (two implementations must agree exactly)
# ---------------------------------------------------------------------------


def _r_ofi_dual(b: SyntheticBundle) -> float:
    """polars ``order_flow_imbalance`` == NumPy ``cont_ofi_by_security``."""
    left = order_flow_imbalance(b.book).select("security_id", "event_time", "ofi")
    right = cont_ofi_by_security(b.book).select(
        "security_id", "event_time", pl.col("ofi").alias("ofi_np")
    )
    joined = left.join(right, on=["security_id", "event_time"], how="inner").filter(
        pl.col("ofi").is_not_null()
    )
    if joined.height == 0:
        raise ValueError("no paired OFI rows to compare")
    return _max_abs_diff(
        np.asarray(joined["ofi"].to_numpy(), dtype=float),
        np.asarray(joined["ofi_np"].to_numpy(), dtype=float),
    )


def _r_kyle_lambda_dual(b: SyntheticBundle) -> float:
    """``estimators.kyle_lambda`` == ``kyle_ofi.kyle_lambda_ols`` per name/flow."""
    worst = 0.0
    compared = 0
    for _sid, grp in b.fused.group_by("security_id", maintain_order=True):
        delta_mid = np.asarray(grp["delta_mid"].to_numpy(), dtype=float)
        for flow_col in ("signed_depth", "ofi"):
            flow = np.asarray(grp[flow_col].to_numpy(), dtype=float)
            lam_loop, _r2 = kyle_lambda(delta_mid, flow)
            lam_ols = kyle_lambda_ols(delta_mid, flow)
            if math.isfinite(lam_loop) and math.isfinite(lam_ols):
                worst = max(worst, abs(float(lam_loop) - float(lam_ols)))
                compared += 1
            elif math.isfinite(lam_loop) != math.isfinite(lam_ols):
                # One implementation defined, the other NaN — a real mismatch.
                return float("inf")
    if compared == 0:
        raise ValueError("no finite Kyle λ pairs to compare")
    return float(worst)


def _r_session_ofi_abs_dominates(b: SyntheticBundle) -> float:
    """Per parent day: |Σ session OFI| <= Σ |session OFI| (triangle inequality)."""
    ofi_sum = _col(b.session_daily, "session_ofi_sum")
    abs_sum = _col(b.session_daily, "session_ofi_abs_sum")
    mask = np.isfinite(ofi_sum) & np.isfinite(abs_sum)
    if int(mask.sum()) == 0:
        raise ValueError("no finite session OFI aggregates")
    excess = np.abs(ofi_sum[mask]) - abs_sum[mask]
    return float(max(0.0, np.max(excess)))


def _r_session_log_ret_telescoping(b: SyntheticBundle) -> float:
    """Σ session log returns telescopes to log(last_close / first_open)."""
    rv = session_realized_variance(b.session)
    if rv.height == 0:
        raise ValueError("session_realized_variance returned no rows")
    ordered = b.session.sort(["security_id", "parent_event_time", "session_index"]).with_columns(
        pl.when(pl.col("session_index") == 0)
        .then((pl.col("close") / pl.col("open")).log())
        .otherwise(
            (
                pl.col("close")
                / pl.col("close").shift(1).over(["security_id", "parent_event_time"])
            ).log()
        )
        .alias("sess_log_ret")
    )
    sums = ordered.group_by(["security_id", "parent_event_time"]).agg(
        pl.col("sess_log_ret").sum().alias("ret_sum"),
        pl.col("open").first().alias("first_open"),
        pl.col("close").last().alias("last_close"),
    )
    lhs = np.asarray(sums["ret_sum"].to_numpy(), dtype=float)
    rhs = np.asarray((sums["last_close"] / sums["first_open"]).log().to_numpy(), dtype=float)
    return _max_abs_diff(lhs, rhs)


def _r_session_rv_nonneg(b: SyntheticBundle) -> float:
    rv = session_realized_variance(b.session)
    if rv.height == 0:
        raise ValueError("session_realized_variance returned no rows")
    return _bound_violation(_col(rv, "session_rv"), lo=0.0, hi=float("inf"))


def _r_variance_telescoping(b: SyntheticBundle) -> float:
    """var_cc == (overnight + open_close)^2; Parkinson / Rogers–Satchell >= 0."""
    frame = ohlc_variance_frame(b.bars)
    var_cc = _col(frame, "var_cc")
    decomp = (_col(frame, "overnight_log") + _col(frame, "open_close_log")) ** 2
    worst = _max_abs_diff(var_cc, decomp)
    for col in ("var_park", "var_rs"):
        worst = max(worst, _bound_violation(_col(frame, col), lo=0.0, hi=float("inf")))
    return worst


def _r_kyle_lambda_dispersion(b: SyntheticBundle) -> float:
    """Dispersion deciles are monotone; the rolling HAC band contains its mean."""
    receipt = kyle_lambda_by_date(b.fused, min_names=3)
    series = np.asarray(receipt.get("kyle_lambda_series") or [], dtype=float)
    if series.size < 3:
        raise ValueError("need >= 3 per-date lambdas for the dispersion identity")
    disp = kyle_lambda_dispersion(series)
    deciles = np.asarray(
        [float(disp[f"kyle_lambda_p{q}"]) for q in range(10, 100, 10)], dtype=float
    )
    if not np.all(np.isfinite(deciles)):
        raise ValueError("dispersion deciles non-finite on a finite lambda series")
    mono = float(np.max(np.maximum(-np.diff(deciles), 0.0)))
    mean = float(disp["kyle_lambda_rolling_mean"])
    lo = float(disp["kyle_lambda_rolling_hac_lo"])
    hi = float(disp["kyle_lambda_rolling_hac_hi"])
    band = 0.0
    if math.isfinite(mean) and math.isfinite(lo) and math.isfinite(hi):
        band = max(0.0, lo - mean, mean - hi)
    return max(mono, band)


def _r_kyle_lambda_ofi_depth_corr(b: SyntheticBundle) -> float:
    """OFI-vs-depth per-date lambda correlations stay in [-1, 1], p in [0, 1]."""
    out = kyle_lambda_ofi_depth_corr(b.fused, min_names=3)
    if int(out.get("n_dates_aligned") or 0) < 3:
        raise ValueError("corr needs >= 3 aligned dates to be meaningful")
    worst = 0.0
    for key in ("kyle_lambda_ofi_depth_spearman", "kyle_lambda_ofi_depth_pearson"):
        value = float(out[key])
        if math.isfinite(value):
            worst = max(worst, max(0.0, abs(value) - 1.0))
    p_value = float(out["kyle_lambda_ofi_depth_prod_hac_p"])
    if math.isfinite(p_value):
        worst = max(worst, max(0.0, -p_value, p_value - 1.0))
    return worst


def _r_vpin_bound(b: SyntheticBundle) -> float:
    framed = vpin_proxy(b.book)
    return _bound_violation(_col(framed, "vpin"), lo=0.0, hi=1.0)


def _r_clv_bound(b: SyntheticBundle) -> float:
    framed = candle_geometry(b.bars)
    return _bound_violation(_col(framed, "close_location_value"), lo=-1.0, hi=1.0)


def _r_candle_wick_partition(b: SyntheticBundle) -> float:
    """body_frac + upper_wick_frac + lower_wick_frac == 1 per candle."""
    framed = candle_geometry(b.bars)
    total = (
        _col(framed, "candle_body_frac")
        + _col(framed, "candle_upper_wick_frac")
        + _col(framed, "candle_lower_wick_frac")
    )
    return _max_abs_diff(total, np.ones_like(total))


# ---------------------------------------------------------------------------
# Curated registry
# ---------------------------------------------------------------------------

_RATE_TOL = 1e-12
_ALG_TOL = 1e-6
_PAIR_TOL = 1e-9

IDENTITY_REGISTRY: tuple[IdentitySpec, ...] = (
    IdentitySpec(
        "ohlc_identity",
        "ohlc_identity",
        "Every daily bar satisfies the OHLC inequalities (high>=open/close/low, low<=all).",
        _RATE_TOL,
        _r_ohlc_identity,
    ),
    IdentitySpec(
        "session_ohlc_identity",
        "session_ohlc",
        "Every reconstructed session candle satisfies the OHLC inequalities.",
        _RATE_TOL,
        _r_session_ohlc_identity,
    ),
    IdentitySpec(
        "session_reconstructs_daily",
        "session_reconstructs_daily",
        "Session candle envelope (first open, last close, max high, min low) == daily OHLC.",
        _RATE_TOL,
        _r_session_reconstructs,
    ),
    IdentitySpec(
        "session_volume_conservation",
        "session_volume_conservation",
        "Sum of session volumes == daily volume per (security, parent day).",
        _RATE_TOL,
        _r_session_volume,
    ),
    IdentitySpec(
        "session_chain",
        "session_chain",
        "Consecutive session candles chain: close_i == open_{i+1}.",
        _RATE_TOL,
        _r_session_chain,
    ),
    IdentitySpec(
        "gap_finite",
        "gap_finite",
        "Overnight gap open_t/close_{t-1}-1 is finite wherever a prior close exists.",
        _RATE_TOL,
        _r_gap_finite,
    ),
    IdentitySpec(
        "book_uncrossed",
        "book_uncrossed",
        "Every L2 snapshot is uncrossed: best_bid < best_ask and spread > 0.",
        _RATE_TOL,
        _r_book_uncrossed,
    ),
    IdentitySpec(
        "session_book_uncrossed",
        "session_book",
        "Every session L2 snapshot is uncrossed: best_bid < best_ask and spread > 0.",
        _RATE_TOL,
        _r_session_book_uncrossed,
    ),
    IdentitySpec(
        "session_book_count",
        "session_book",
        "Each (security, parent day) carries exactly n_session_candles book snaps.",
        _RATE_TOL,
        _r_session_book_count,
    ),
    IdentitySpec(
        "session_book_vpin_bound",
        "session_book_vpin",
        "session_book_vpin = |ofi_sum| / ofi_abs_sum lies in [0, 1] when defined.",
        _RATE_TOL,
        _r_session_book_vpin_bound,
    ),
    IdentitySpec(
        "depth_shape_finite",
        "depth_shape_finite",
        "Depth-shape metrics are finite on every deep-side (n_levels>=2) cell.",
        _RATE_TOL,
        _r_depth_shape_finite,
    ),
    IdentitySpec(
        "concentration_top_finite",
        "concentration_top",
        "Side concentration_top cells are finite wherever side depth > 0.",
        _RATE_TOL,
        _r_concentration_top_finite,
    ),
    IdentitySpec(
        "queue_priority_finite",
        "queue_priority",
        "Queue-priority proxy cells are finite wherever side depth > 0.",
        _RATE_TOL,
        _r_queue_priority_finite,
    ),
    IdentitySpec(
        "side_notional_finite",
        "side_notional",
        "Side notional proxies are finite wherever price>0 and depth>0.",
        _RATE_TOL,
        _r_side_notional_finite,
    ),
    IdentitySpec(
        "tob_size_share_finite",
        "tob_size_share",
        "tob_size_share is finite wherever bid_depth + ask_depth > 0.",
        _RATE_TOL,
        _r_tob_size_share_finite,
    ),
    IdentitySpec(
        "mid_from_touch",
        "mid",
        "mid == 0.5 * (best_bid + best_ask).",
        _ALG_TOL,
        _r_mid,
    ),
    IdentitySpec(
        "spread_from_touch",
        "spread",
        "spread == quoted_spread == effective_spread == best_ask - best_bid.",
        _ALG_TOL,
        _r_spread,
    ),
    IdentitySpec(
        "half_spread_recovers_touch",
        "half_spread",
        "half_spread == spread/2 and mid +/- half_spread recover best ask/bid.",
        _ALG_TOL,
        _r_half_spread,
    ),
    IdentitySpec(
        "spread_bps_scale",
        "spread_bps",
        "spread_bps == quoted_spread_bps == 1e4*spread/mid; half_spread_bps and spread_over_mid agree.",
        _ALG_TOL,
        _r_spread_bps,
    ),
    IdentitySpec(
        "microprice_balance",
        "microprice",
        "microprice == w*best_ask + (1-w)*best_bid with w = top_bid/(top_bid+top_ask).",
        _ALG_TOL,
        _r_microprice,
    ),
    IdentitySpec(
        "microprice_minus_mid",
        "microprice",
        "microprice_minus_mid == microprice - mid; _bps == 1e4*delta/mid.",
        _ALG_TOL,
        _r_microprice_minus_mid,
    ),
    IdentitySpec(
        "imbalance_top",
        "imbalance_top",
        "imbalance_top == touch_size_imbalance == queue_imbalance == (tb-ta)/(tb+ta).",
        _ALG_TOL,
        _r_imbalance_top,
    ),
    IdentitySpec(
        "imbalance_depth",
        "imbalance_depth",
        "imbalance_depth == (bid_depth-ask_depth)/(bid_depth+ask_depth); abs alias agrees.",
        _ALG_TOL,
        _r_imbalance_depth,
    ),
    IdentitySpec(
        "depth_equals_level_sum",
        "depth",
        "bid_depth/ask_depth equal the summed sizes of the snapshot's bid/ask levels.",
        _ALG_TOL,
        _r_depth_equals_level_sum,
    ),
    IdentitySpec(
        "notional_algebra",
        "notional",
        "TOB notional, side notionals, notional_imbalance, tob_notional_share, tob_size_share agree with definitions.",
        _ALG_TOL,
        _r_notional_algebra,
    ),
    IdentitySpec(
        "concentration_top_algebra",
        "concentration_top",
        "bid/ask_size_concentration_top == top_size / side_depth.",
        _ALG_TOL,
        _r_concentration_top,
    ),
    IdentitySpec(
        "queue_priority_algebra",
        "queue_priority",
        "queue_priority_proxy == top/(top+side_depth) per side.",
        _ALG_TOL,
        _r_queue_priority,
    ),
    IdentitySpec(
        "ofi_dual_implementation",
        "ofi",
        "polars order_flow_imbalance == NumPy cont_ofi_by_security (Cont-Kukanov-Stoikov).",
        _PAIR_TOL,
        _r_ofi_dual,
    ),
    IdentitySpec(
        "kyle_lambda_dual",
        "kyle_lambda",
        "estimators.kyle_lambda == kyle_ofi.kyle_lambda_ols per name on signed_depth and ofi.",
        _PAIR_TOL,
        _r_kyle_lambda_dual,
    ),
    IdentitySpec(
        "kyle_lambda_dispersion_shape",
        "kyle_lambda_dispersion",
        "Per-date lambda deciles are nondecreasing; rolling HAC band contains the rolling mean.",
        _RATE_TOL,
        _r_kyle_lambda_dispersion,
    ),
    IdentitySpec(
        "kyle_lambda_ofi_depth_corr_bounds",
        "kyle_lambda_ofi_depth_corr",
        "kyle_lambda_ofi_depth_corr spearman/pearson in [-1,1] and HAC p in [0,1].",
        _RATE_TOL,
        _r_kyle_lambda_ofi_depth_corr,
    ),
    IdentitySpec(
        "session_ofi_abs_dominates",
        "session_ofi",
        "Per parent day |session_ofi_sum| <= session_ofi_abs_sum (triangle inequality).",
        _RATE_TOL,
        _r_session_ofi_abs_dominates,
    ),
    IdentitySpec(
        "session_log_ret_telescoping",
        "session",
        "Sum of session log returns telescopes to log(last_close / first_open) per day.",
        _ALG_TOL,
        _r_session_log_ret_telescoping,
    ),
    IdentitySpec(
        "session_rv_nonneg",
        "session",
        "Session realized variance is non-negative on every parent day.",
        _RATE_TOL,
        _r_session_rv_nonneg,
    ),
    IdentitySpec(
        "ohlc_variance_decomposition",
        "ohlc_variance",
        "var_cc == (overnight_log + open_close_log)^2; Parkinson and Rogers-Satchell >= 0.",
        _ALG_TOL,
        _r_variance_telescoping,
    ),
    IdentitySpec(
        "vpin_bound",
        "vpin",
        "vpin_proxy rolling toxicity lies in [0, 1] wherever finite.",
        _RATE_TOL,
        _r_vpin_bound,
    ),
    IdentitySpec(
        "clv_bound",
        "clv",
        "close_location_value == (2*close - high - low)/(high - low) lies in [-1, 1].",
        _RATE_TOL,
        _r_clv_bound,
    ),
    IdentitySpec(
        "candle_wick_partition",
        "candle",
        "body_frac + upper_wick_frac + lower_wick_frac == 1 per candle.",
        _ALG_TOL,
        _r_candle_wick_partition,
    ),
)


def registered_families() -> frozenset[str]:
    """Identity families covered by the curated registry."""
    return frozenset(spec.family for spec in IDENTITY_REGISTRY)


# ---------------------------------------------------------------------------
# Programmatic enumeration of catalog identity pairs
# ---------------------------------------------------------------------------


def _catalog_guard_names() -> list[str]:
    """Catalog helper functions named ``*_honesty_errors``/``*_consistency_errors``."""
    names: set[str] = set()
    for modname in _CATALOG_MODULES:
        mod = import_module(modname)
        for name, obj in inspect.getmembers(mod, inspect.isfunction):
            if getattr(obj, "__module__", None) != modname:
                continue
            if _GUARD_RE.match(name):
                names.add(name)
    return sorted(names)


def _impl_function_index() -> dict[str, str]:
    """Public estimator/rate function name -> qualified ``module.name``."""
    index: dict[str, str] = {}
    for modname in _IMPL_MODULES:
        mod = import_module(modname)
        for name, obj in inspect.getmembers(mod, inspect.isfunction):
            if name.startswith("_"):
                continue
            if getattr(obj, "__module__", modname) != modname:
                continue
            index.setdefault(name, f"{modname}.{name}")
    return index


def enumerate_catalog_identity_pairs() -> list[dict[str, str]]:
    """Introspect catalog guards and pair them across an identity family.

    Two pair kinds:

    - ``never_equate``: ``X_vs_Y_never_equate_*`` helpers encode pairs of
      rates the catalog forbids aliasing — the pair of identity families is
      (X, Y).
    - ``guard_impl``: a guard stem that exactly names a public estimator
      (e.g. ``ohlc_identity_rate_honesty_errors`` ->
      ``northset.identities.ohlc_identity_rate``) — the catalog guard and
      the implementation it watches.
    """
    impl = _impl_function_index()
    pairs: list[dict[str, str]] = []
    for name in _catalog_guard_names():
        pair = _NEVER_EQUATE_RE.match(name)
        if pair:
            pairs.append(
                {
                    "kind": "never_equate",
                    "left": pair.group(1),
                    "right": pair.group(2),
                    "guard": name,
                }
            )
            continue
        stem = _GUARD_RE.match(name)
        if stem is None:
            continue
        stem_name = stem.group(1)
        if stem_name not in impl:
            continue
        family = stem_name[: -len("_rate")] if stem_name.endswith("_rate") else stem_name
        pairs.append(
            {
                "kind": "guard_impl",
                "family": family,
                "stem": stem_name,
                "guard": name,
                "implementation": impl[stem_name],
            }
        )
    return pairs


def catalog_identity_families() -> frozenset[str]:
    """All identity-family stems referenced by catalog guard helpers."""
    families: set[str] = set()
    for pair in enumerate_catalog_identity_pairs():
        if pair["kind"] == "never_equate":
            families.add(pair["left"])
            families.add(pair["right"])
        else:
            families.add(pair["family"])
    return frozenset(families)


# ---------------------------------------------------------------------------
# Sweep + receipt
# ---------------------------------------------------------------------------


def run_identity_sweep(
    *,
    n_trials: int = 8,
    seed: int = 7,
    registry: tuple[IdentitySpec, ...] | list[IdentitySpec] | None = None,
    bundle_factory: Callable[[int], SyntheticBundle] | None = None,
) -> dict[str, Any]:
    """Prove every registered identity on ``n_trials`` seeded bundles.

    Fail-closed: an evaluator that raises or yields a non-finite residual
    records a failed verdict (with the error captured) rather than a pass.
    Returns the receipt dict; use ``write_identity_receipt`` to persist.
    """
    specs = tuple(registry) if registry is not None else IDENTITY_REGISTRY
    if not specs:
        raise ValueError("identity registry must be non-empty")
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("identity registry names must be unique")
    if any(not math.isfinite(spec.tolerance) or spec.tolerance < 0.0 for spec in specs):
        raise ValueError("identity tolerances must be finite and >= 0")
    if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials < 1:
        raise ValueError("n_trials must be a positive integer")
    factory = bundle_factory or make_synthetic_bundle

    residuals: dict[str, list[float]] = {name: [] for name in names}
    errors: dict[str, list[str]] = {name: [] for name in names}
    for trial in range(int(n_trials)):
        bundle = factory(int(seed) * 1009 + trial)
        for spec in specs:
            try:
                value = float(spec.evaluate(bundle))
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError("identity residual must be finite and non-negative")
            except Exception as exc:  # fail-closed: record, never fabricate a pass
                residuals[spec.name].append(float("nan"))
                errors[spec.name].append(f"{type(exc).__name__}: {exc}")
                continue
            residuals[spec.name].append(value)

    results: list[dict[str, Any]] = []
    for spec in specs:
        res = residuals[spec.name]
        errs = errors[spec.name]
        finite = [r for r in res if math.isfinite(r)]
        residual_max = max(finite) if finite else float("nan")
        passed = not errs and len(finite) == len(res) and residual_max <= spec.tolerance
        record: dict[str, Any] = {
            "name": spec.name,
            "family": spec.family,
            "statement": spec.statement,
            "tolerance": float(spec.tolerance),
            "residual_max": residual_max,
            "residuals": res,
            "trials": len(res),
            "verdict": "pass" if passed else "fail",
        }
        if errs:
            record["errors"] = errs
        results.append(record)

    n_passed = sum(1 for r in results if r["verdict"] == "pass")
    receipt: dict[str, Any] = {
        "kind": "identity_sweep",
        "schema_version": IDENTITY_SWEEP_SCHEMA_VERSION,
        "firm": "Artificial Hedge",
        "product": "Dipcatcher",
        "claim": "research_only",
        "synthetic": True,
        "data_source": "SYNTHETIC",
        "generated_at": datetime.now(UTC).isoformat(),
        "seed": int(seed),
        "n_trials": int(n_trials),
        "git_revision": git_revision(),
        "n_identities": len(results),
        "n_passed": n_passed,
        "n_failed": len(results) - n_passed,
        "all_passed": n_passed == len(results),
        "registered_families": sorted(
            registered_families() if registry is None else {s.family for s in specs}
        ),
        "enumerated_pairs": enumerate_catalog_identity_pairs(),
        "identities": results,
    }
    if not family_blob_forbidden_metrics_absent(receipt):
        raise AssertionError("identity sweep receipt leaked forbidden research keys")
    receipt["receipt_sha256"] = hash_bytes(canonical_json_bytes(receipt))
    return receipt


def format_identity_table(receipt: dict[str, Any]) -> str:
    """Fixed-width identity/residual/verdict table for CLI output."""
    identities = receipt.get("identities") or []
    header = f"{'identity':32} {'family':26} {'residual_max':>13} {'tol':>9} verdict"
    lines = [header, "-" * len(header)]
    for row in identities:
        residual = row.get("residual_max")
        residual_s = f"{float(residual):.3e}" if isinstance(residual, (int, float)) else "nan"
        lines.append(
            f"{row['name']:32} {row['family']:26} {residual_s:>13} "
            f"{float(row['tolerance']):>9.1e} {row['verdict']}"
        )
    lines.append(
        f"{receipt['n_passed']}/{receipt['n_identities']} identities hold "
        f"({receipt['n_trials']} seeded trials, seed={receipt['seed']})"
    )
    return "\n".join(lines)


def _identity_contract_errors(receipt: Mapping[str, Any]) -> list[str]:
    """The synthetic-contract predicates every identity receipt must satisfy."""
    errors: list[str] = []
    if receipt.get("kind") != "identity_sweep":
        errors.append("kind")
    if receipt.get("schema_version") != IDENTITY_SWEEP_SCHEMA_VERSION:
        errors.append("schema_version")
    if receipt.get("synthetic") is not True:
        errors.append("synthetic")
    if receipt.get("data_source") != "SYNTHETIC":
        errors.append("data_source")
    if receipt.get("claim") != "research_only":
        errors.append("claim")
    if not family_blob_forbidden_metrics_absent(dict(receipt)):
        errors.append("forbidden_metrics")
    return errors


def _identity_seal_errors(receipt: Mapping[str, Any]) -> list[str]:
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    expected_digest = hash_bytes(canonical_json_bytes(unsigned))
    return [] if receipt.get("receipt_sha256") == expected_digest else ["receipt_sha256"]


def identity_dataset_identity(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """What was evaluated: the seeded SYNTHETIC trial set and registered families."""
    return {
        "seed": receipt["seed"],
        "n_trials": receipt["n_trials"],
        "registered_families": list(receipt["registered_families"]),
        "enumerated_pairs": list(receipt["enumerated_pairs"]),
    }


def identity_params(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {"seed": receipt["seed"], "n_trials": receipt["n_trials"]}


def identity_verdict(receipt: Mapping[str, Any]) -> str:
    return "pass" if receipt.get("all_passed") is True else "fail"


def identity_receipt_v2(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Wrap an ``identity_sweep`` payload in the unified ``receipt.v2`` envelope.

    The v1 payload is embedded verbatim under ``payload``; the envelope binds
    the seeded trial-set identity, run params, this module's source hash, and
    the loaded numeric stack. A malformed payload is never wrapped.
    """
    from quant_fund.research.receipt_v2 import build_receipt_v2

    if _identity_contract_errors(receipt) or _identity_seal_errors(receipt):
        raise ValueError("identity receipt violates its synthetic research contract")
    return build_receipt_v2(
        kind=str(receipt["kind"]),
        data_label=str(receipt["data_source"]),
        dataset=identity_dataset_identity(receipt),
        params=identity_params(receipt),
        code_files=(Path(__file__),),
        verdict=identity_verdict(receipt),
        payload=dict(receipt),
        generated_at=str(receipt["generated_at"]),
        revision=str(receipt["git_revision"]),
    )


def identity_v2_consistency_errors(envelope: Mapping[str, Any]) -> list[str]:
    """Re-derive an identity receipt.v2 envelope's bound digests from its payload."""
    errors: list[str] = []
    payload = envelope.get("payload")
    if not isinstance(payload, Mapping):
        return ["payload_not_object"]
    contract_errors = _identity_contract_errors(payload)
    errors.extend(f"payload_{name}" for name in contract_errors)
    errors.extend(f"payload_{name}" for name in _identity_seal_errors(payload))
    if contract_errors:
        return errors
    try:
        dataset = identity_dataset_identity(payload)
        params = identity_params(payload)
    except (KeyError, TypeError) as exc:
        return [*errors, f"payload_missing_field:{exc}"]
    if hash_bytes(canonical_json_bytes(dataset)) != envelope.get("dataset_hash"):
        errors.append("dataset_hash_mismatch")
    if hash_bytes(canonical_json_bytes(params)) != envelope.get("params_hash"):
        errors.append("params_hash_mismatch")
    if identity_verdict(payload) != envelope.get("verdict"):
        errors.append("verdict_mismatch")
    return errors


def write_identity_receipt(
    path: Path, receipt: dict[str, Any], *, receipt_version: int = 1
) -> Path:
    """Atomically publish an immutable, hash-verified SYNTHETIC receipt.

    ``receipt_version=2`` wraps the payload in the unified ``receipt.v2``
    envelope before writing; the envelope seal is recomputed over the wrap.
    """
    path = Path(path)
    if receipt_version == 1:
        body: dict[str, Any] | Mapping[str, Any] = receipt
    elif receipt_version == 2:
        body = identity_receipt_v2(receipt)
        from quant_fund.research.receipt_v2 import seal_receipt

        body = seal_receipt(body)
    else:
        raise ValueError(f"receipt_version must be 1 or 2, got {receipt_version!r}")
    if receipt_version == 1:
        if _identity_contract_errors(body):
            raise ValueError("identity receipt violates its synthetic research contract")
        if _identity_seal_errors(body):
            raise ValueError("identity receipt hash mismatch")
    canonical = json.loads(canonical_json_bytes(dict(body)))
    content = json.dumps(canonical, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise FileExistsError(f"receipt path is a symlink: {path}")
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"receipt already exists with different content: {path}")
        return path
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            if path.is_symlink() or path.read_text(encoding="utf-8") != content:
                raise FileExistsError(
                    f"receipt already exists with different content: {path}"
                ) from None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return path
