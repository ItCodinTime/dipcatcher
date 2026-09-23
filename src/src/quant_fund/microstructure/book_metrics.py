"""Order-book derived metrics (microprice, imbalance, depth).

Depth shape metrics (`bid_log_size_slope` / `ask_log_size_slope`,
`bid_log_price_slope` / `ask_log_price_slope`, and mean log tick spacings)
are defined only when a side has ≥2 positive finite levels; otherwise NaN
(fail-closed). Crossed / locked / empty books are rejected by
``OrderBookSnapshot`` before metrics run; this module also fail-closes if a
caller passes a thin or crossed book.

``SIDE_STRUCTURE_FIELDS`` (``bid/ask_size_concentration_top``) are
finite whenever side depth > 0 — unlike ``DEPTH_SHAPE_FIELDS``,
which require ``n_*_levels ≥ 2``. Panel optional lists should treat
them separately (this module does not edit ``book_panel``).
``microprice_weight_balance`` is the top-bid share used in the
microprice convex combination. ``effective_spread`` and ``quoted_spread`` alias ``spread``
(best_ask − best_bid; ``quoted_spread`` is the northset-safe name). ``depth_imbalance_abs`` is |imbalance_depth|.
``spread_over_mid`` is spread/mid (equals ``spread_bps``/1e4 when mid>0).
``half_spread`` is spread/2; mid±half_spread recovers best ask/bid.
``quoted_spread_bps`` aliases ``spread_bps``; ``half_spread_bps`` is
1e4·half_spread/mid. ``touch_size_imbalance`` aliases ``imbalance_top``.
``queue_priority_proxy`` is top/(top+side_depth) (thin-safe NaN).
See ``QUEUE_STRUCTURE_FIELDS`` for the export tuple (not DEPTH_SHAPE).
``top_of_book_notional_proxy`` is best_bid·top_bid_size + best_ask·top_ask_size.
``side_notional_proxy_bid/ask`` is best·side_depth (thin-safe NaN).
``notional_imbalance`` is (bid−ask)/(bid+ask) notionals ∈[-1,1].
``tob_notional_share`` is TOB/(bid+ask) notional where TOB is
best_bid·top_bid_size + best_ask·top_ask_size (∈(0,1] when finite).
``tob_size_share`` = (top_bid+top_ask)/(bid_depth+ask_depth) is the
size-domain ∈(0,1] touch share (distinct field, same bound).

``METRICS_REQUIRED_FINITE_KEYS`` lists scalars that must be present and
finite on every valid snapshot metrics dict (TOB structure / always-on
fields). Depth-shape keys are excluded — they are NaN when n_levels < 2.
Call ``assert_metrics_required_finite`` / ``assert_metrics_key_partition`` to enforce REQUIRED finite and OPTIONAL not-±inf.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from quant_fund.schemas.order_book import BookLevel, OrderBookSnapshot

# Shape fields that require n_levels ≥ 2 (else NaN). Panel/validators may key off these.
DEPTH_SHAPE_FIELDS: tuple[str, ...] = (
    "bid_log_size_slope",
    "ask_log_size_slope",
    "bid_log_price_slope",
    "ask_log_price_slope",
    "bid_mean_log_tick_spacing",
    "ask_mean_log_tick_spacing",
)

# Top-of-book structure (finite whenever side_depth > 0). Panel optional lists
# may key off these separately from DEPTH_SHAPE_FIELDS (which need n≥2).
SIDE_STRUCTURE_FIELDS: tuple[str, ...] = (
    "bid_size_concentration_top",
    "ask_size_concentration_top",
)

# Queue-priority proxies (finite when side_depth > 0). Panel optionals may
# key these separately from SIDE_STRUCTURE_FIELDS / DEPTH_SHAPE_FIELDS.
QUEUE_STRUCTURE_FIELDS: tuple[str, ...] = (
    "queue_priority_proxy",
    "ask_queue_priority_proxy",
)

SIDE_NOTIONAL_FIELDS: tuple[str, ...] = (
    "side_notional_proxy_bid",
    "side_notional_proxy_ask",
)

# Touch size share (honest ∈(0,1]). Adjacent to SIDE_STRUCTURE conceptually.
TOB_SHARE_FIELDS: tuple[str, ...] = ("tob_size_share",)

# Scalar keys that must be present and finite on every valid OrderBookSnapshot
# metrics dict (top-of-book / always-defined structure). Depth-shape keys are
# excluded — they are NaN when n_levels < 2.

# One-line meanings for METRICS_REQUIRED_FINITE_KEYS (integrity catalog).
METRICS_REQUIRED_FINITE_KEY_DOCS: dict[str, str] = {
    "best_bid": "Top bid price",
    "best_ask": "Top ask price",
    "mid": "0.5 * (best_bid + best_ask)",
    "spread": "best_ask - best_bid",
    "quoted_spread": "Alias of spread (distinct from quoted_spread_bps)",
    "effective_spread": "Alias of spread",
    "half_spread": "spread / 2",
    "spread_bps": "1e4 * spread / mid",
    "quoted_spread_bps": "Alias of spread_bps",
    "half_spread_bps": "1e4 * half_spread / mid",
    "spread_over_mid": "spread / mid",
    "microprice": "Size-weighted mid from top sizes",
    "microprice_minus_mid": "microprice - mid",
    "microprice_minus_mid_bps": "1e4 * (microprice - mid) / mid",
    "imbalance_top": "(top_bid - top_ask) / (top_bid + top_ask)",
    "touch_size_imbalance": "Alias of imbalance_top",
    "imbalance_depth": "(bid_depth - ask_depth) / (bid_depth + ask_depth)",
    "depth_imbalance_abs": "|imbalance_depth|",
    "bid_depth": "Sum of bid sizes",
    "ask_depth": "Sum of ask sizes",
    "top_bid_size": "Best bid size",
    "top_ask_size": "Best ask size",
    "microprice_weight_balance": "top_bid_size / (top_bid_size + top_ask_size)",
    "n_bid_levels": "len(bids)",
    "n_ask_levels": "len(asks)",
    "top_of_book_notional_proxy": "best_bid * top_bid_size + best_ask * top_ask_size",
    "notional_imbalance": "(bid_notional - ask_notional) / (bid_notional + ask_notional)",
    "tob_notional_share": "TOB notional / (bid+ask) notional (∈ (0,1] with touch-priced TOB)",
}

METRICS_REQUIRED_FINITE_KEYS: frozenset[str] = frozenset(
    {
        "best_bid",
        "best_ask",
        "mid",
        "spread",
        "quoted_spread",
        "effective_spread",
        "half_spread",
        "spread_bps",
        "quoted_spread_bps",
        "half_spread_bps",
        "spread_over_mid",
        "microprice",
        "microprice_minus_mid",
        "microprice_minus_mid_bps",
        "imbalance_top",
        "touch_size_imbalance",
        "imbalance_depth",
        "depth_imbalance_abs",
        "bid_depth",
        "ask_depth",
        "top_bid_size",
        "top_ask_size",
        "microprice_weight_balance",
        "n_bid_levels",
        "n_ask_levels",
        "top_of_book_notional_proxy",
        "notional_imbalance",
        "tob_notional_share",
    }
)

# Optional / depth-dependent keys that may be NaN on valid thin books (n_levels < 2).
# Disjoint from METRICS_REQUIRED_FINITE_KEYS. SIDE/QUEUE/NOTIONAL structure fields
# that are always finite on valid snaps live in REQUIRED (or structure tuples).
# Optional catalogs: DEPTH_SHAPE may be NaN on thin books; SIDE/QUEUE/NOTIONAL/
# TOB_SHARE are panel-optional structure fields (finite on valid depth>0 snaps,
# but not required by assert_metrics_required_finite). Disjoint from REQUIRED.
METRICS_OPTIONAL_NAN_OK_KEYS: frozenset[str] = frozenset(DEPTH_SHAPE_FIELDS).union(
    SIDE_STRUCTURE_FIELDS,
    QUEUE_STRUCTURE_FIELDS,
    SIDE_NOTIONAL_FIELDS,
    TOB_SHARE_FIELDS,
)

METRICS_OPTIONAL_STRUCTURE_CATALOGS: tuple[tuple[str, ...], ...] = (
    DEPTH_SHAPE_FIELDS,
    SIDE_STRUCTURE_FIELDS,
    QUEUE_STRUCTURE_FIELDS,
    SIDE_NOTIONAL_FIELDS,
    TOB_SHARE_FIELDS,
)


def microprice(snapshot: OrderBookSnapshot) -> float:
    """Size-weighted mid using top-of-book sizes."""
    bid = snapshot.bids[0]
    ask = snapshot.asks[0]
    denom = bid.size + ask.size
    if denom <= 0.0 or not math.isfinite(denom):
        return snapshot.mid
    return (ask.price * bid.size + bid.price * ask.size) / denom


def book_metrics_from_snapshot(snapshot: OrderBookSnapshot) -> dict[str, float]:
    """Research diagnostics from one L2 snapshot.

    Multi-level honesty: with depth ≥ 2, log-size slopes, log-price slopes,
    and mean log tick spacings are finite. Top-of-book-only sides
    (``n_*_levels < 2``) yield NaN for those shape fields — same contract
    vendor top-of-book remaps stamp for size slopes.
    """
    if not snapshot.bids or not snapshot.asks:
        raise ValueError("empty book side: bids and asks must be non-empty")
    if snapshot.best_bid >= snapshot.best_ask:
        raise ValueError("crossed or locked book: best bid must be < best ask")

    bid_sz = sum(level.size for level in snapshot.bids)
    ask_sz = sum(level.size for level in snapshot.asks)
    top_bid_sz = snapshot.bids[0].size
    top_ask_sz = snapshot.asks[0].size
    imb_top = (top_bid_sz - top_ask_sz) / (top_bid_sz + top_ask_sz)
    imb_depth = (bid_sz - ask_sz) / (bid_sz + ask_sz) if (bid_sz + ask_sz) > 0 else 0.0
    mp = microprice(snapshot)
    mid = snapshot.mid
    bid_notional = _side_notional_proxy(snapshot.best_bid, bid_sz)
    ask_notional = _side_notional_proxy(snapshot.best_ask, ask_sz)
    tob_notional = (
        float(snapshot.best_bid * top_bid_sz + snapshot.best_ask * top_ask_sz)
        if (
            math.isfinite(snapshot.best_bid)
            and math.isfinite(snapshot.best_ask)
            and snapshot.best_bid > 0.0
            and snapshot.best_ask > 0.0
        )
        else float("nan")
    )
    out = {
        "best_bid": float(snapshot.best_bid),
        "best_ask": float(snapshot.best_ask),
        "mid": float(mid),
        "spread": float(snapshot.spread),
        "quoted_spread": float(snapshot.spread),  # alias of spread (northset-safe name)
        "half_spread": float(0.5 * snapshot.spread),
        "effective_spread": float(snapshot.spread),  # alias: best_ask - best_bid
        "spread_bps": float(snapshot.spread_bps),
        "quoted_spread_bps": float(snapshot.spread_bps),  # alias
        "half_spread_bps": (
            float(1e4 * (0.5 * snapshot.spread) / snapshot.mid)
            if snapshot.mid > 0.0 and math.isfinite(snapshot.mid)
            else float("nan")
        ),
        "spread_over_mid": (
            float(snapshot.spread / snapshot.mid)
            if snapshot.mid > 0.0 and math.isfinite(snapshot.mid)
            else float("nan")
        ),
        "microprice": float(mp),
        "microprice_minus_mid": float(mp - mid),
        "microprice_minus_mid_bps": float(0.0 if mid <= 0 else 1e4 * (mp - mid) / mid),
        "imbalance_top": float(imb_top),
        "touch_size_imbalance": float(imb_top),  # alias of imbalance_top
        "imbalance_depth": float(imb_depth),
        "depth_imbalance_abs": float(abs(imb_depth)) if math.isfinite(imb_depth) else float("nan"),
        "bid_depth": float(bid_sz),
        "ask_depth": float(ask_sz),
        "top_bid_size": float(top_bid_sz),
        "top_ask_size": float(top_ask_sz),
        "bid_size_concentration_top": _size_concentration_top(top_bid_sz, bid_sz),
        "ask_size_concentration_top": _size_concentration_top(top_ask_sz, ask_sz),
        "queue_priority_proxy": _queue_priority_proxy(top_bid_sz, bid_sz),
        "ask_queue_priority_proxy": _queue_priority_proxy(top_ask_sz, ask_sz),
        "microprice_weight_balance": _microprice_weight_balance(top_bid_sz, top_ask_sz),
        "top_of_book_notional_proxy": tob_notional,
        "side_notional_proxy_bid": bid_notional,
        "side_notional_proxy_ask": ask_notional,
        "notional_imbalance": _notional_imbalance(bid_notional, ask_notional),
        "tob_notional_share": _tob_notional_share(tob_notional, bid_notional, ask_notional),
        "tob_size_share": _tob_size_share(top_bid_sz, top_ask_sz, bid_sz, ask_sz),
        "n_bid_levels": float(len(snapshot.bids)),
        "n_ask_levels": float(len(snapshot.asks)),
        "bid_log_size_slope": _log_size_slope(snapshot.bids),
        "ask_log_size_slope": _log_size_slope(snapshot.asks),
        # Multi-level price geometry (log-price slope + mean log tick spacing)
        "bid_log_price_slope": _log_price_slope(snapshot.bids),
        "ask_log_price_slope": _log_price_slope(snapshot.asks),
        "bid_mean_log_tick_spacing": _mean_log_tick_spacing(snapshot.bids),
        "ask_mean_log_tick_spacing": _mean_log_tick_spacing(snapshot.asks),
    }
    return assert_metrics_required_finite(out)


def _ols_log_y_on_index(values: np.ndarray) -> float:
    """OLS slope of log(values) on 0..n-1. Fail-closed → NaN if n<2."""
    if values.size < 2:
        return float("nan")
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        return float("nan")
    y = np.log(values)
    x = np.arange(values.size, dtype=float)
    x_var = float(np.var(x))
    if x_var <= 1e-18:
        return float("nan")
    return float(np.cov(x, y, ddof=0)[0, 1] / x_var)


def _log_size_slope(levels: list[BookLevel]) -> float:
    """OLS slope of log(size) on level index (0 = inside)."""
    return _ols_log_y_on_index(np.asarray([level.size for level in levels], dtype=float))


def _log_price_slope(levels: list[BookLevel]) -> float:
    """OLS slope of log(price) on level index (0 = inside).

    Captures multi-level log-price geometry. Fail-closed → NaN when n<2.
    """
    return _ols_log_y_on_index(np.asarray([level.price for level in levels], dtype=float))


def _mean_log_tick_spacing(levels: list[BookLevel]) -> float:
    """Mean log adjacent price gap (tick-spacing shape).

    For levels sorted best→worse, spacing_i = |p_i - p_{i-1}|.
    Finite only when n≥2 and all gaps are positive finite; else NaN.
    """
    if len(levels) < 2:
        return float("nan")
    prices = np.asarray([level.price for level in levels], dtype=float)
    if not np.isfinite(prices).all():
        return float("nan")
    gaps = np.abs(np.diff(prices))
    if not np.isfinite(gaps).all() or np.any(gaps <= 0.0):
        return float("nan")
    return float(np.mean(np.log(gaps)))


def metric_rows_from_frame(frame: Any) -> list[dict[str, float]]:
    """Convert a panel frame to the numeric-row input the *_finite_rate helpers expect.

    Non-numeric cells (timestamps, security ids, None) are dropped rather than
    coerced, so missing optional metric columns read as ineligible/NaN while
    present numeric cells keep their values. ``None`` never raises.
    """
    rows: list[dict[str, float]] = []
    for row in frame.to_dicts():
        values: dict[str, float] = {}
        for key, value in row.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            values[key] = float(value)
        rows.append(values)
    return rows


def depth_shape_finite_rate(
    rows: list[dict[str, float]],
    *,
    fields: tuple[str, ...] = DEPTH_SHAPE_FIELDS,
) -> float:
    """Fraction of (row, field) cells that are finite among deep-side eligibles.

    A cell is *eligible* when the matching side claims multi-level depth:
    - bid_* fields require ``n_bid_levels >= 2``
    - ask_* fields require ``n_ask_levels >= 2``

    Returns the finite / eligible ratio in ``[0, 1]``. No eligible cells →
    ``NaN`` (fail-closed diagnostic, not a fake 1.0). Empty ``rows`` → NaN.
    """
    if not rows:
        return float("nan")
    eligible = 0
    finite = 0
    for row in rows:
        n_bid = float(row.get("n_bid_levels", float("nan")))
        n_ask = float(row.get("n_ask_levels", float("nan")))
        for field in fields:
            if field.startswith("bid_"):
                deep_ok = math.isfinite(n_bid) and n_bid >= 2.0
            elif field.startswith("ask_"):
                deep_ok = math.isfinite(n_ask) and n_ask >= 2.0
            else:
                # Unknown field prefix: require both sides deep
                deep_ok = (
                    math.isfinite(n_bid) and n_bid >= 2.0 and math.isfinite(n_ask) and n_ask >= 2.0
                )
            if not deep_ok:
                continue
            eligible += 1
            value = float(row.get(field, float("nan")))
            if math.isfinite(value):
                finite += 1
    if eligible == 0:
        return float("nan")
    return float(finite) / float(eligible)


def _size_concentration_top(top_size: float, side_depth: float) -> float:
    """top_size / side_depth; NaN if side_depth ≤ 0 or non-finite inputs (thin-safe)."""
    if not math.isfinite(top_size) or not math.isfinite(side_depth) or side_depth <= 0.0:
        return float("nan")
    return float(top_size) / float(side_depth)


def concentration_top_finite_rate(
    rows: list[dict[str, float]],
    *,
    fields: tuple[str, ...] = SIDE_STRUCTURE_FIELDS,
) -> float:
    """Fraction of finite concentration cells among sides with depth > 0.

    Eligible cells:
    - ``bid_size_concentration_top`` when ``bid_depth > 0`` (finite)
    - ``ask_size_concentration_top`` when ``ask_depth > 0`` (finite)

    Empty ``rows`` or zero eligible cells → ``NaN`` (fail-closed diagnostic).
    """
    if not rows:
        return float("nan")
    eligible = 0
    finite = 0
    for row in rows:
        bid_depth = float(row.get("bid_depth", float("nan")))
        ask_depth = float(row.get("ask_depth", float("nan")))
        for field in fields:
            if field.startswith("bid_"):
                ok = math.isfinite(bid_depth) and bid_depth > 0.0
            elif field.startswith("ask_"):
                ok = math.isfinite(ask_depth) and ask_depth > 0.0
            else:
                ok = (
                    math.isfinite(bid_depth)
                    and bid_depth > 0.0
                    and math.isfinite(ask_depth)
                    and ask_depth > 0.0
                )
            if not ok:
                continue
            eligible += 1
            value = float(row.get(field, float("nan")))
            if math.isfinite(value):
                finite += 1
    if eligible == 0:
        return float("nan")
    return float(finite) / float(eligible)


def _microprice_weight_balance(top_bid_size: float, top_ask_size: float) -> float:
    """top_bid_size / (top_bid_size + top_ask_size); NaN if denom ≤ 0 / non-finite.

    With this weight w, microprice == best_ask * w + best_bid * (1 - w).
    """
    if not math.isfinite(top_bid_size) or not math.isfinite(top_ask_size):
        return float("nan")
    denom = top_bid_size + top_ask_size
    if denom <= 0.0:
        return float("nan")
    return float(top_bid_size) / float(denom)


def _queue_priority_proxy(top_size: float, side_depth: float) -> float:
    """top_size / (top_size + side_depth); NaN if side_depth ≤ 0 / non-finite (thin-safe).

    Distinct from size_concentration_top (= top/side_depth): denominator is
    top + full side depth. Fail-closed when side_depth ≤ 0.
    """
    if not math.isfinite(top_size) or not math.isfinite(side_depth) or side_depth <= 0.0:
        return float("nan")
    denom = top_size + side_depth
    return float(top_size) / float(denom)


def queue_priority_finite_rate(
    rows: list[dict[str, float]],
    *,
    fields: tuple[str, ...] = QUEUE_STRUCTURE_FIELDS,
) -> float:
    """Fraction of finite queue-priority cells among sides with depth > 0.

    Eligible cells:
    - ``queue_priority_proxy`` when ``bid_depth > 0`` (finite)
    - ``ask_queue_priority_proxy`` when ``ask_depth > 0`` (finite)

    Empty ``rows`` or zero eligible cells → ``NaN`` (fail-closed diagnostic).
    """
    if not rows:
        return float("nan")
    eligible = 0
    finite = 0
    for row in rows:
        bid_depth = float(row.get("bid_depth", float("nan")))
        ask_depth = float(row.get("ask_depth", float("nan")))
        for field in fields:
            if field.startswith("ask_"):
                ok = math.isfinite(ask_depth) and ask_depth > 0.0
            elif field.startswith("queue_") or field.startswith("bid_"):
                # queue_priority_proxy is bid-side; bid_* also bid
                ok = math.isfinite(bid_depth) and bid_depth > 0.0
            else:
                ok = (
                    math.isfinite(bid_depth)
                    and bid_depth > 0.0
                    and math.isfinite(ask_depth)
                    and ask_depth > 0.0
                )
            if not ok:
                continue
            eligible += 1
            value = float(row.get(field, float("nan")))
            if math.isfinite(value):
                finite += 1
    if eligible == 0:
        return float("nan")
    return float(finite) / float(eligible)


def _side_notional_proxy(best_price: float, side_depth: float) -> float:
    """best_price * side_depth; NaN if price/depth non-finite or ≤ 0 (thin-safe)."""
    if (
        not math.isfinite(best_price)
        or not math.isfinite(side_depth)
        or best_price <= 0.0
        or side_depth <= 0.0
    ):
        return float("nan")
    return float(best_price) * float(side_depth)


def side_notional_finite_rate(
    rows: list[dict[str, float]],
    *,
    fields: tuple[str, ...] = SIDE_NOTIONAL_FIELDS,
) -> float:
    """Fraction of finite side-notional cells among eligible sides.

    Eligible:
    - ``side_notional_proxy_bid`` when ``best_bid > 0`` and ``bid_depth > 0``
    - ``side_notional_proxy_ask`` when ``best_ask > 0`` and ``ask_depth > 0``

    Empty ``rows`` or zero eligible → ``NaN``.
    """
    if not rows:
        return float("nan")
    eligible = 0
    finite = 0
    for row in rows:
        best_bid = float(row.get("best_bid", float("nan")))
        best_ask = float(row.get("best_ask", float("nan")))
        bid_depth = float(row.get("bid_depth", float("nan")))
        ask_depth = float(row.get("ask_depth", float("nan")))
        for field in fields:
            if field.endswith("bid"):
                ok = (
                    math.isfinite(best_bid)
                    and best_bid > 0.0
                    and math.isfinite(bid_depth)
                    and bid_depth > 0.0
                )
            elif field.endswith("ask"):
                ok = (
                    math.isfinite(best_ask)
                    and best_ask > 0.0
                    and math.isfinite(ask_depth)
                    and ask_depth > 0.0
                )
            else:
                continue
            if not ok:
                continue
            eligible += 1
            value = float(row.get(field, float("nan")))
            if math.isfinite(value):
                finite += 1
    if eligible == 0:
        return float("nan")
    return float(finite) / float(eligible)


def _notional_imbalance(bid_notional: float, ask_notional: float) -> float:
    """(bid - ask) / (bid + ask) notionals; NaN if either non-finite or sum ≤ 0."""
    if not math.isfinite(bid_notional) or not math.isfinite(ask_notional):
        return float("nan")
    denom = bid_notional + ask_notional
    if denom <= 0.0:
        return float("nan")
    return float(bid_notional - ask_notional) / float(denom)


def _tob_notional_share(tob_notional: float, bid_notional: float, ask_notional: float) -> float:
    """TOB notional / (bid+ask side notional); NaN if non-finite or denom ≤ 0."""
    if (
        not math.isfinite(tob_notional)
        or not math.isfinite(bid_notional)
        or not math.isfinite(ask_notional)
    ):
        return float("nan")
    denom = bid_notional + ask_notional
    if denom <= 0.0:
        return float("nan")
    return float(tob_notional) / float(denom)


def _tob_size_share(top_bid: float, top_ask: float, bid_depth: float, ask_depth: float) -> float:
    """(top_bid+top_ask)/(bid_depth+ask_depth); NaN if denom ≤ 0 / non-finite.

    Always ∈ (0, 1] when finite on a valid book (tops ≤ side depths).
    """
    vals = (top_bid, top_ask, bid_depth, ask_depth)
    if not all(math.isfinite(v) for v in vals):
        return float("nan")
    if top_bid <= 0.0 or top_ask <= 0.0 or bid_depth <= 0.0 or ask_depth <= 0.0:
        return float("nan")
    denom = bid_depth + ask_depth
    return float(top_bid + top_ask) / float(denom)


def tob_size_share_finite_rate(
    rows: list[dict[str, float]],
    *,
    field: str = "tob_size_share",
) -> float:
    """Fraction of rows with finite tob_size_share among eligible books.

    Eligible when ``bid_depth + ask_depth > 0`` (both depths finite).
    Empty ``rows`` or zero eligible → ``NaN``.
    """
    if not rows:
        return float("nan")
    eligible = 0
    finite = 0
    for row in rows:
        bid_depth = float(row.get("bid_depth", float("nan")))
        ask_depth = float(row.get("ask_depth", float("nan")))
        if not (math.isfinite(bid_depth) and math.isfinite(ask_depth)):
            continue
        if bid_depth + ask_depth <= 0.0:
            continue
        eligible += 1
        value = float(row.get(field, float("nan")))
        if math.isfinite(value):
            finite += 1
    if eligible == 0:
        return float("nan")
    return float(finite) / float(eligible)


def assert_metrics_required_finite(metrics: dict[str, float]) -> dict[str, float]:
    """Public integrity helper: required keys present and finite.

    Used by ``book_metrics_from_snapshot``; safe for panel/validators to call.
    """
    missing = sorted(METRICS_REQUIRED_FINITE_KEYS - metrics.keys())
    if missing:
        if len(missing) == 1:
            raise ValueError(f"book metrics missing required key: {missing[0]}")
        raise ValueError(f"book metrics missing required keys: {missing}")
    bad = sorted(k for k in METRICS_REQUIRED_FINITE_KEYS if not math.isfinite(float(metrics[k])))
    if bad:
        raise ValueError(f"book metrics required keys not finite: {bad}")
    return metrics


def assert_metrics_optional_not_inf(metrics: dict[str, float]) -> dict[str, float]:
    """Fail-closed: OPTIONAL_NAN_OK keys may be NaN but must not be ±inf when present.

    Missing optional keys are allowed (panel may omit). Present values must be
    finite or NaN — never ±inf.
    """
    bad = sorted(
        k for k in METRICS_OPTIONAL_NAN_OK_KEYS if k in metrics and math.isinf(float(metrics[k]))
    )
    if bad:
        raise ValueError(f"book metrics optional keys are ±inf (NaN ok): {bad}")
    return metrics


def assert_metrics_key_partition(metrics: dict[str, float]) -> dict[str, float]:
    """REQUIRED finite + OPTIONAL present values not ±inf. Fail-closed."""
    assert_metrics_required_finite(metrics)
    assert_metrics_optional_not_inf(metrics)
    return metrics
