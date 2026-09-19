"""Research bench: candle + order-book fused features vs next-bar return.

SYNTHETIC / research-diagnostic only. Forbidden Sharpe/pnl keys stay out.
``live_pnl_claim`` is always false.

Institutional scoring: **date-level** Spearman/Pearson IC via ``date_ic_series``
(HAC t/p on the IC series) — never pooled stacked Spearman across the panel.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import polars as pl

from quant_fund.metrics.cross_section import date_ic_series
from quant_fund.microstructure.book_metrics import DEPTH_SHAPE_FIELDS, depth_shape_finite_rate
from quant_fund.microstructure.candle_book_features import attach_candle_book_features


def _structure_finite_rate_from_companions(*rates: float) -> float:
    """Nanmean of finite companion rates; all-missing → NaN."""
    vals: list[float] = []
    for x in rates:
        try:
            v = float(x)
        except (TypeError, ValueError):
            continue
        if v == v and abs(v) != float("inf"):
            vals.append(v)
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _finite_rate_column(
    frame: pl.DataFrame,
    column: str,
    *,
    require_positive: str | None = None,
) -> float:
    """Fraction of finite ``column`` values (optionally among require_positive > 0)."""
    if column not in frame.columns or frame.height == 0:
        return float("nan")
    values = frame[column].to_numpy().astype(float)
    if require_positive is not None and require_positive in frame.columns:
        depth = frame[require_positive].to_numpy().astype(float)
        mask = np.isfinite(depth) & (depth > 0.0)
        if int(mask.sum()) == 0:
            return float("nan")
        values = values[mask]
    finite = np.isfinite(values)
    if values.size == 0:
        return float("nan")
    return float(finite.sum()) / float(values.size)


_FEATURE_COLS = (
    "candle_body_ret",
    "candle_body_frac",
    "candle_range_frac",
    "candle_direction",
    "wick_skew",
    "imbalance_top",
    "imbalance_depth",
    "depth_imbalance_abs",
    "spread_bps",
    "half_spread_bps",
    "spread_over_mid",
    "microprice_minus_mid",
    "microprice_minus_mid_bps",
    "candle_dir_x_imbalance",
    "spread_x_range",
    "imbalance_x_body_frac",
    "ofi",
    "queue_imbalance",
    "bid_log_size_slope",
    "ask_log_size_slope",
    "signed_vol_x_imbalance",
    "bid_log_price_slope",
    "ask_log_price_slope",
    "bid_mean_log_tick_spacing",
    "ask_mean_log_tick_spacing",
    "bid_size_concentration_top",
    "ask_size_concentration_top",
    "queue_priority_proxy",
    "ask_queue_priority_proxy",
    "tob_size_share",
    "notional_imbalance",
    "microprice_weight_balance",
)
FEATURE_COLS = _FEATURE_COLS


def bench_candle_order_book(
    bars: pl.DataFrame,
    *,
    book: pl.DataFrame | None = None,
    depth: int = 5,
    seed: int = 7,
    label: str = "SYNTHETIC",
    min_names: int = 4,
) -> dict[str, Any]:
    """Date-level IC of fused candle/LOB features vs next close return (HAC).

    Pass ``book`` for an external/vendor panel; otherwise SYNTHETIC L2 is built.
    """
    if bars.height == 0:
        raise ValueError("bars must be non-empty")
    fused = attach_candle_book_features(bars, book=book, depth=depth, seed=seed)
    fused = fused.sort(["security_id", "event_time"]).with_columns(
        (pl.col("candle_close").shift(-1).over("security_id") / pl.col("candle_close") - 1.0).alias(
            "fwd_ret_1"
        )
    )
    sample = fused.drop_nulls(["fwd_ret_1"])
    book_source = (
        str(fused["book_source"][0])
        if "book_source" in fused.columns and fused.height
        else "synthetic_lob"
    )
    book_dgp = (
        str(fused["book_dgp"][0])
        if "book_dgp" in fused.columns and fused.height
        else "synthetic_lob"
    )
    y = sample["fwd_ret_1"].to_numpy().astype(float)
    dates = sample["event_time"].to_numpy()
    ics: dict[str, float] = {}
    for col in _FEATURE_COLS:
        if col not in sample.columns:
            ics[f"ic_{col}"] = float("nan")
            ics[f"ic_{col}_t"] = float("nan")
            ics[f"ic_{col}_p"] = float("nan")
            ics[f"ic_{col}_n_dates"] = 0.0
            continue
        x = sample[col].to_numpy().astype(float)
        result = date_ic_series(x, y, dates, min_names=min_names)
        ics[f"ic_{col}"] = float(result.mean_spearman)
        ics[f"ic_{col}_pearson"] = float(result.mean_pearson)
        ics[f"ic_{col}_t"] = float(result.t_spearman)
        ics[f"ic_{col}_p"] = float(result.p_spearman)
        ics[f"ic_{col}_n_dates"] = float(result.n_dates)

    finite_ics = [
        v
        for k, v in ics.items()
        if k.startswith("ic_")
        and not k.endswith(("_t", "_p", "_n_dates", "_pearson"))
        and np.isfinite(v)
    ]
    best_name = None
    best_abs = -1.0
    for name, value in ics.items():
        if not name.startswith("ic_") or name.endswith(("_t", "_p", "_n_dates", "_pearson")):
            continue
        if np.isfinite(value) and abs(value) > best_abs:
            best_abs = abs(value)
            best_name = name

    shape_cols = ["n_bid_levels", "n_ask_levels", *DEPTH_SHAPE_FIELDS]
    shape_present = [c for c in shape_cols if c in fused.columns]
    if {"n_bid_levels", "n_ask_levels", *DEPTH_SHAPE_FIELDS} <= set(shape_present):
        shape_rows = fused.select(shape_present).to_dicts()
        depth_shape_rate = depth_shape_finite_rate(shape_rows)
    else:
        # Absent shape columns read as unmeasured (NaN), never a fake 0.0 rate:
        # external panels may legally omit the optional slope/tick columns.
        depth_shape_rate = float("nan")

    fr_micro = _finite_rate_column(fused, "microprice_minus_mid")
    fr_bid_conc = _finite_rate_column(
        fused, "bid_size_concentration_top", require_positive="bid_depth"
    )
    fr_ask_conc = _finite_rate_column(
        fused, "ask_size_concentration_top", require_positive="ask_depth"
    )
    structure_finite = _structure_finite_rate_from_companions(fr_micro, fr_bid_conc, fr_ask_conc)

    out: dict[str, Any] = {
        "family": "candle_order_book",
        "label": str(label),
        "data_source": "SYNTHETIC" if label.upper().startswith("SYN") else str(label),
        "dgp": book_dgp,
        "book_dgp": book_dgp,
        "book_source": book_source,
        "ic_method": "date_level_spearman_hac",
        "n_bars": int(bars.height),
        "n_fused": int(fused.height),
        "n_scored": int(sample.height),
        "join_coverage": float(cast(Any, fused["join_coverage"][0]))
        if "join_coverage" in fused.columns and fused.height
        else float("nan"),
        "mean_book_age_seconds": float(cast(Any, fused["book_age_seconds"].mean()))
        if "book_age_seconds" in fused.columns and fused.height
        else float("nan"),
        "max_book_age_seconds": float(cast(Any, fused["book_age_seconds"].max()))
        if "book_age_seconds" in fused.columns and fused.height
        else float("nan"),
        "depth_shape_finite_rate": float(depth_shape_rate),
        # Structure finite_rate companions for structure_finite_rate_honesty_errors.
        "finite_rate_microprice_minus_mid": float(fr_micro),
        "finite_rate_bid_size_concentration_top": float(fr_bid_conc),
        "finite_rate_ask_size_concentration_top": float(fr_ask_conc),
        # Aggregate of finite_rate_* companions (honest CLI/northset-parity key).
        "structure_finite_rate": float(structure_finite),
        "depth": int(depth),
        "min_names": int(min_names),
        "mean_abs_ic": float(np.mean(np.abs(finite_ics))) if finite_ics else float("nan"),
        "best_feature_ic_key": best_name or "",
        "best_feature_ic": float(ics[best_name]) if best_name else float("nan"),
        "mean_microprice_weight_balance": float(
            np.nanmean(fused["microprice_weight_balance"].to_numpy().astype(float))
        )
        if "microprice_weight_balance" in fused.columns
        else float("nan"),
        "mean_notional_imbalance": float(
            np.nanmean(fused["notional_imbalance"].to_numpy().astype(float))
        )
        if "notional_imbalance" in fused.columns
        else float("nan"),
        "mean_depth_imbalance": float(np.nanmean(fused["imbalance_depth"].to_numpy().astype(float)))
        if "imbalance_depth" in fused.columns
        else float("nan"),
        "mean_depth_imbalance_abs": float(
            np.nanmean(fused["depth_imbalance_abs"].to_numpy().astype(float))
        )
        if "depth_imbalance_abs" in fused.columns
        else float("nan"),
        "mean_tob_size_share": float(np.nanmean(fused["tob_size_share"].to_numpy().astype(float)))
        if "tob_size_share" in fused.columns
        else float("nan"),
        "mean_imbalance_top": float(np.nanmean(fused["imbalance_top"].to_numpy().astype(float)))
        if "imbalance_top" in fused.columns
        else float("nan"),
        "mean_queue_imbalance": float(np.nanmean(fused["queue_imbalance"].to_numpy().astype(float)))
        if "queue_imbalance" in fused.columns
        else float("nan"),
        "mean_bid_size_concentration_top": float(
            np.nanmean(fused["bid_size_concentration_top"].to_numpy().astype(float))
        )
        if "bid_size_concentration_top" in fused.columns
        else float("nan"),
        "mean_ask_size_concentration_top": float(
            np.nanmean(fused["ask_size_concentration_top"].to_numpy().astype(float))
        )
        if "ask_size_concentration_top" in fused.columns
        else float("nan"),
        "mean_ofi": float(np.nanmean(fused["ofi"].to_numpy().astype(float)))
        if "ofi" in fused.columns
        else float("nan"),
        "mean_queue_priority_proxy": float(
            np.nanmean(fused["queue_priority_proxy"].to_numpy().astype(float))
        )
        if "queue_priority_proxy" in fused.columns
        else float("nan"),
        "mean_ask_queue_priority_proxy": float(
            np.nanmean(fused["ask_queue_priority_proxy"].to_numpy().astype(float))
        )
        if "ask_queue_priority_proxy" in fused.columns
        else float("nan"),
        "mean_bid_log_size_slope": float(
            np.nanmean(fused["bid_log_size_slope"].to_numpy().astype(float))
        )
        if "bid_log_size_slope" in fused.columns
        else float("nan"),
        "mean_ask_log_size_slope": float(
            np.nanmean(fused["ask_log_size_slope"].to_numpy().astype(float))
        )
        if "ask_log_size_slope" in fused.columns
        else float("nan"),
        "mean_microprice_minus_mid": float(
            np.nanmean(fused["microprice_minus_mid"].to_numpy().astype(float))
        )
        if "microprice_minus_mid" in fused.columns
        else float("nan"),
        # Spread-alias receipt means (quoted == effective alias; half = quoted/2;
        # bps = 2·half_bps). Missing column → honest NaN, never invented.
        "mean_quoted_spread": float(np.nanmean(fused["spread"].to_numpy().astype(float)))
        if "spread" in fused.columns
        else float("nan"),
        "mean_effective_spread": float(
            np.nanmean(fused["effective_spread"].to_numpy().astype(float))
        )
        if "effective_spread" in fused.columns
        else float("nan"),
        "mean_half_spread": float(np.nanmean(fused["half_spread"].to_numpy().astype(float)))
        if "half_spread" in fused.columns
        else float("nan"),
        "mean_half_spread_bps": float(np.nanmean(fused["half_spread_bps"].to_numpy().astype(float)))
        if "half_spread_bps" in fused.columns
        else float("nan"),
        "mean_spread_bps": float(np.nanmean(fused["spread_bps"].to_numpy().astype(float)))
        if "spread_bps" in fused.columns
        else float("nan"),
        "mean_spread_over_mid": float(np.nanmean(fused["spread_over_mid"].to_numpy().astype(float)))
        if "spread_over_mid" in fused.columns
        else float("nan"),
        # Candle close–mid diagnostic (2·|C−mid|/mid) — never the book spread.
        "mean_close_mid_abs_rel": float(
            np.nanmean(fused["close_mid_abs_rel"].to_numpy().astype(float))
        )
        if "close_mid_abs_rel" in fused.columns
        else float("nan"),
        "mean_candle_body_ret": float(np.nanmean(fused["candle_body_ret"].to_numpy().astype(float)))
        if "candle_body_ret" in fused.columns
        else float("nan"),
        "mean_candle_body_frac": float(
            np.nanmean(fused["candle_body_frac"].to_numpy().astype(float))
        )
        if "candle_body_frac" in fused.columns
        else float("nan"),
        "mean_candle_range_frac": float(
            np.nanmean(fused["candle_range_frac"].to_numpy().astype(float))
        )
        if "candle_range_frac" in fused.columns
        else float("nan"),
        "mean_candle_direction": float(
            np.nanmean(fused["candle_direction"].to_numpy().astype(float))
        )
        if "candle_direction" in fused.columns
        else float("nan"),
        "mean_wick_skew": float(np.nanmean(fused["wick_skew"].to_numpy().astype(float)))
        if "wick_skew" in fused.columns
        else float("nan"),
        "mean_microprice_minus_mid_bps": float(
            np.nanmean(fused["microprice_minus_mid_bps"].to_numpy().astype(float))
        )
        if "microprice_minus_mid_bps" in fused.columns
        else float("nan"),
        "mean_candle_dir_x_imbalance": float(
            np.nanmean(fused["candle_dir_x_imbalance"].to_numpy().astype(float))
        )
        if "candle_dir_x_imbalance" in fused.columns
        else float("nan"),
        "mean_spread_x_range": float(np.nanmean(fused["spread_x_range"].to_numpy().astype(float)))
        if "spread_x_range" in fused.columns
        else float("nan"),
        "mean_imbalance_x_body_frac": float(
            np.nanmean(fused["imbalance_x_body_frac"].to_numpy().astype(float))
        )
        if "imbalance_x_body_frac" in fused.columns
        else float("nan"),
        "mean_signed_vol_x_imbalance": float(
            np.nanmean(fused["signed_vol_x_imbalance"].to_numpy().astype(float))
        )
        if "signed_vol_x_imbalance" in fused.columns
        else float("nan"),
        "mean_bid_log_price_slope": float(
            np.nanmean(fused["bid_log_price_slope"].to_numpy().astype(float))
        )
        if "bid_log_price_slope" in fused.columns
        else float("nan"),
        "mean_ask_log_price_slope": float(
            np.nanmean(fused["ask_log_price_slope"].to_numpy().astype(float))
        )
        if "ask_log_price_slope" in fused.columns
        else float("nan"),
        "mean_bid_mean_log_tick_spacing": float(
            np.nanmean(fused["bid_mean_log_tick_spacing"].to_numpy().astype(float))
        )
        if "bid_mean_log_tick_spacing" in fused.columns
        else float("nan"),
        "mean_ask_mean_log_tick_spacing": float(
            np.nanmean(fused["ask_mean_log_tick_spacing"].to_numpy().astype(float))
        )
        if "ask_mean_log_tick_spacing" in fused.columns
        else float("nan"),
        "research_only": True,
        "claim": "research_diagnostic_only",
        **ics,
    }
    _forbidden = ("sharpe", "pnl", "live_pnl_claim")
    leaked = [k for k in out if any(tok in k.lower() for tok in _forbidden)]
    if leaked:
        raise AssertionError(f"candle_order_book bench leaked forbidden research keys: {leaked}")
    return out
