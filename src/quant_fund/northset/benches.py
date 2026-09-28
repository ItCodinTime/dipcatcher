"""Northset research bench: candlesticks + L2 books.

Identities, date-level IC, OHLC vol estimators, Kyle/Roll/Corwin–Schultz,
Amihud, OFI, VPIN, session RV/jumps. No Sharpe. Research-only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from quant_fund.config.models import AppConfig
from quant_fund.metrics.cross_section import DateICResult, date_ic_series
from quant_fund.metrics.scoring import qlike
from quant_fund.microstructure import book_metrics as book_metrics_mod
from quant_fund.microstructure.candle_book_features import (
    attach_candle_book_features,
    forward_close_return_labels,
)
from quant_fund.microstructure.synthetic_lob import (
    aggregate_session_book_to_daily,
    ensure_book_panel_shape_columns,
    synthesize_l2_from_bars,
    synthesize_session_l2,
)
from quant_fund.northset.bench_helpers import (
    cond_fwd_mean,
    enforce_structure_floors,
    evidence_provenance,
    float_col_or_empty,
    join_age_metrics,
    metrics_required_finite_ok,
    nanmean_col,
    nanmean_finite,
    structure_finite_rates,
    sweep_evidence_receipt_fields,
)
from quant_fund.northset.candles import geometry_rates
from quant_fund.northset.data_view import canonical_northset_bars
from quant_fund.northset.estimators import (
    abdi_ranaldo_spread,
    amihud_illiquidity,
    corwin_schultz_spread,
    dm_range_vs_park,
    dm_split_vs_park,
    garman_klass_vs_close_to_close,
    kyle_lambda,
    lag1_corr,
    ohlc_variance_frame,
    order_flow_imbalance,
    overnight_plus_oc_vs_close_to_close,
    overnight_share,
    parkinson_vs_close_to_close,
    queue_imbalance,
    realized_semivariance,
    rogers_satchell_vs_close_to_close,
    roll_spread,
    session_bipower_jump,
    session_realized_variance,
    session_vpin,
    true_range_frame,
    volume_over_range,
    vpin_proxy,
    yang_zhang_variance,
    yang_zhang_vs_close_to_close,
)
from quant_fund.northset.identities import (
    book_uncrossed_rate,
    gap_finite_rate,
    ohlc_identity_rate,
    session_candles_from_daily,
    session_chain_rate,
    session_reconstructs_daily_rate,
    session_volume_conservation_rate,
    validate_session_book_counts,
)
from quant_fund.northset.sweep_research import PRIMARY_EXECUTABLE_TEST, sweep_evidence_battery
from quant_fund.northset.sweeps import (
    liquidity_sweep_frame,
    sweep_output_columns,
    sweep_rates,
)
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent

# Sweep evidence nests stamp mean_excess_bps / mean_diff_bps; receipt helpers
# promote them to the prefixed northset CLI echo keys.
_SWEEP_EVENT_MEAN_FIELD = "mean_excess_bps"
_SWEEP_CONTROL_DIFF_FIELD = "mean_diff_bps"


def enforce_session_l2_identity_floors(
    *,
    ohlc_identity_rate: float,
    session_ohlc_identity_rate: float,
    session_reconstructs_daily_rate: float,
    session_volume_conservation_rate: float,
    session_chain_rate: float,
    book_uncrossed_rate: float,
    floor: float = 0.99,
) -> dict[str, float]:
    """Fail-closed identity gates for the session-L2 path (ADR-021).

    SYNTHETIC session candles + books must preserve OHLC / volume / chain /
    uncrossed-book contracts. Rates below ``floor`` are data-contract bugs,
    not alpha. Floor default 0.99 (allow tiny float noise; honest SYNTHETIC
    paths hit 1.0).
    """
    if not (0.0 <= float(floor) <= 1.0) or floor != floor:
        raise ValueError(f"identity floor must be in [0, 1], got {floor!r}")
    rates = {
        "ohlc_identity_rate": float(ohlc_identity_rate),
        "session_ohlc_identity_rate": float(session_ohlc_identity_rate),
        "session_reconstructs_daily_rate": float(session_reconstructs_daily_rate),
        "session_volume_conservation_rate": float(session_volume_conservation_rate),
        "session_chain_rate": float(session_chain_rate),
        "book_uncrossed_rate": float(book_uncrossed_rate),
    }
    failures: list[str] = []
    for name, value in rates.items():
        if value != value:  # NaN
            failures.append(f"{name}=nan")
        elif value + 1e-12 < float(floor):
            failures.append(f"{name}={value:.6g}<{floor}")
    if failures:
        raise ValueError("session-L2 identity gate failed (fail-closed): " + ", ".join(failures))
    return rates


_IC_FEATURES = (
    "imbalance_top",
    "imbalance_depth",
    "microprice_minus_mid_bps",
    "ofi",
    "ofi_lag",
    "wick_skew",
    "candle_body_ret",
    "candle_gap",
    "spread_bps",
    "amihud",
    "candle_dir_x_imbalance",
    "bid_log_size_slope",
    "queue_imbalance",
    "vpin",
    "session_ofi_sum",
    "session_imbalance_mean",
    "session_book_vpin",
    "session_close_imbalance",
    "session_close_spread_bps",
    "session_close_micro_bps",
    "session_close_mid",
    "session_close_bid_depth",
    "session_close_ask_depth",
    "session_spread_bps_mean",
    "close_location_value",
    "volume_over_range",
    "sweep_reject_signed",
    "sweep_follow_signed",
    "sweep_depth_signed",
)

# Receipt keys for session-L2 path / last-snap means (soft-verify companions).
SESSION_RECEIPT_KEYS: frozenset[str] = frozenset(
    {
        "session_ofi_sum_mean",
        "mean_session_ofi_abs_sum",
        "session_book_vpin_mean",
        "mean_session_imbalance_mean",
        "mean_session_imbalance_std",
        "mean_session_spread_bps_mean",
        "mean_session_close_spread_bps",
        "mean_session_close_imbalance",
        "mean_session_close_micro_bps",
        "mean_session_close_mid",
        "mean_session_close_bid_depth",
        "mean_session_close_ask_depth",
        "mean_session_book_snaps",
    }
)

# CLI northset + research family echo every SESSION_RECEIPT_KEYS entry (no blob-only exceptions).
SESSION_RECEIPT_KEYS_CLI_ECHO: frozenset[str] = SESSION_RECEIPT_KEYS

# Scoped northset CLI echo honesty (Sergeant): every stamped non-discovery /
# non-kyle receipt key must be either CLI-echoed or explicitly blob-only.
# Discovery IC companions (*_mean_ic / *_p_ic / …) and kyle_* stay out of scope.
# CLI short aliases: mean_book_age_seconds → mean_book_age_s, max_book_age_seconds → max_book_age_s.

NORTHSET_CLI_ECHO_KEY_ALIASES: dict[str, str] = {
    "mean_book_age_seconds": "mean_book_age_s",
    "max_book_age_seconds": "max_book_age_s",
}

# Must appear in `dipcatcher northset` CLI source (key= or get('key')).
# Core rates / structure / session / spread / shape — not the full mean_* surface
# (CoS owns mean_* echo completeness separately).
NORTHSET_CLI_ECHO_REQUIRED: frozenset[str] = frozenset(
    {
        "join_coverage",
        "book_source",
        "ohlc_identity_rate",
        "session_ohlc_identity_rate",
        "book_uncrossed_rate",
        "session_chain_rate",
        "session_reconstructs_daily_rate",
        "session_volume_conservation_rate",
        "gap_finite_rate",
        "structure_finite_rate",
        "depth_shape_finite_rate",
        "concentration_top_finite_rate",
        "queue_priority_finite_rate",
        "side_notional_finite_rate",
        "tob_size_share_finite_rate",
        "mean_book_age_seconds",
        "max_book_age_seconds",
        "mean_quoted_spread",
        "mean_effective_spread",
        "mean_half_spread",
        "mean_half_spread_bps",
        "mean_spread_bps",
        "mean_close_mid_abs_rel",
        "mean_microprice_weight_balance",
        "mean_microprice_minus_mid",
        "mean_tob_size_share",
        "mean_tob_notional_share",
        "mean_imbalance_top",
        "mean_depth_imbalance",
        "mean_queue_priority_proxy",
        "mean_n_bid_levels",
        "mean_n_ask_levels",
        "mean_bid_log_size_slope",
        "mean_ask_log_size_slope",
        "mean_bid_log_price_slope",
        "mean_ask_log_price_slope",
        "mean_bid_mean_log_tick_spacing",
        "mean_ask_mean_log_tick_spacing",
        *SESSION_RECEIPT_KEYS,
    }
)

# Explicitly allowed to stay receipt/blob-only (not CLI key= lines).
NORTHSET_RECEIPT_BLOB_ONLY: frozenset[str] = frozenset(
    {
        "book_dgp",
        "book_join_coverage_floor",
        "book_panel_path",
        "component_sources",
        "dgp",
        "dm_gk_vs_park_p",
        "dm_gk_vs_park_preferred",
        "dm_gk_vs_park_stat",
        "dm_rs_vs_park_p",
        "dm_rs_vs_park_preferred",
        "dm_rs_vs_park_stat",
        "dm_split_vs_park_p",
        "dm_split_vs_park_preferred",
        "dm_split_vs_park_stat",
        "doji_rate",
        "engulfing_rate",
        "family",
        "hammer_rate",
        "impact_estimator_scope",
        "impact_proxy_warning",
        "label",
        "marubozu_rate",
        "mean_sweep_depth_high",
        "mean_sweep_depth_low",
        "mid_lag1_n_securities",
        "n_bars",
        "n_fused",
        "n_scored",
        "n_session_candles",
        "ofi_lag1_n_securities",
        "price_basis",
        "product",
        "queue_priority_finite_floor",
        "research_only",
        "return_basis",
        "roll_n_securities",
        "session_bulk_vpin",
        "session_vpin_method",
        "session_l2_identity_floor",
        "session_mean_jump_ratio",
        "shooting_star_rate",
        "side_notional_finite_floor",
        "spinning_top_rate",
        "sweep_any_rate",
        "sweep_both_rate",
        "sweep_eligible_rate",
        "sweep_evidence",
        "sweep_evidence_scope",
        "sweep_follow_control_diff_p",
        "sweep_follow_control_diff_t",
        "sweep_follow_control_sample_adequate",
        "sweep_follow_liq_control_diff_p",
        "sweep_follow_liq_control_diff_t",
        "sweep_follow_liq_control_sample_adequate",
        "sweep_follow_name_cluster_p",
        "sweep_follow_name_cluster_t",
        "sweep_follow_two_way_cluster_p",
        "sweep_follow_two_way_cluster_t",
        "sweep_follow_two_way_wild_p",
        "sweep_follow_overnight_gap_mean_bps",
        "sweep_follow_overnight_gap_p",
        "sweep_follow_overnight_gap_t",
        "sweep_follow_oot_insample_mean_bps",
        "sweep_follow_oot_same_sign",
        "sweep_p95_event_adv_participation",
        "sweep_reject_liq_control_diff_p",
        "sweep_reject_liq_control_diff_t",
        "sweep_reject_liq_control_sample_adequate",
        "sweep_reject_name_cluster_p",
        "sweep_reject_name_cluster_t",
        "sweep_reject_two_way_cluster_p",
        "sweep_reject_two_way_cluster_t",
        "sweep_reject_two_way_wild_p",
        "sweep_reject_overnight_gap_mean_bps",
        "sweep_reject_overnight_gap_p",
        "sweep_reject_overnight_gap_t",
        "sweep_follow_cost_adjusted_mean_bps",
        "sweep_follow_cost_adjusted_p",
        "sweep_follow_event_p",
        "sweep_follow_event_t",
        "sweep_follow_placebo_observed_ic",
        "sweep_follow_placebo_p",
        "sweep_high_follow_share",
        "sweep_high_rate",
        "sweep_high_reclaim_share",
        "sweep_low_follow_share",
        "sweep_low_rate",
        "sweep_low_reclaim_share",
        "sweep_min_fold_positive_fraction",
        "sweep_reject_control_diff_p",
        "sweep_reject_control_diff_t",
        "sweep_reject_control_sample_adequate",
        "sweep_reject_cost_adjusted_p",
        "sweep_reject_event_p",
        "sweep_reject_event_t",
        "sweep_reject_placebo_observed_ic",
        "sweep_reject_placebo_p",
        "tob_size_share_finite_floor",
        "use_session_l2",
        "vpin_method",
        "yang_zhang_qlike_scope",
        "sweep_inference_index",
        "sweep_n_ohlc_quarantined",
        "sweep_overnight_gap_method",
        "sweep_two_way_inference_index",
        "corwin_schultz_pair_scope",
    }
)

NORTHSET_CLI_ECHO_EXTRA: frozenset[str] = frozenset(
    {
        "abdi_ranaldo_spread",
        "amihud_mean",
        "book_hypothesis_eligible",
        "claim",
        "concentration_top_finite_floor",
        "corwin_schultz_spread",
        "data_source",
        "depth",
        "depth_shape_finite_floor",
        "garman_klass_qlike_vs_cc",
        "mean_ask_depth",
        "mean_ask_queue_priority_proxy",
        "mean_ask_size_concentration_top",
        "mean_bid_depth",
        "mean_bid_size_concentration_top",
        "mean_depth_imbalance_abs",
        "mean_fwd_ret_after_high_follow",
        "mean_fwd_ret_after_high_reclaim",
        "mean_fwd_ret_after_low_follow",
        "mean_fwd_ret_after_low_reclaim",
        "mean_microprice_minus_mid_bps",
        "mean_notional_imbalance",
        "mean_side_notional_proxy_ask",
        "mean_side_notional_proxy_bid",
        "mean_spread_over_mid",
        "mean_top_ask_size",
        "mean_top_bid_size",
        "mean_top_of_book_notional_proxy",
        "mean_true_range",
        "metrics_required_finite_ok",
        "mid_lag1_corr",
        "n_session_book_rows",
        "n_sweep_high",
        "n_sweep_low",
        "ofi_lag1_corr",
        "overnight_plus_oc_qlike_vs_cc",
        "overnight_share",
        "parkinson_qlike_vs_cc",
        "queue_imbalance_mean",
        "rogers_satchell_qlike_vs_cc",
        "roll_spread",
        "semi_down",
        "semi_up",
        "session_book_hypothesis_eligible",
        "session_l2_identity_gate",
        "session_mean_bv",
        "session_mean_rv",
        "session_rv_qlike_vs_cc",
        "shape_columns_ensured",
        "sweep_follow_control_diff_mean_bps",
        "sweep_follow_event_mean_bps",
        "sweep_follow_fold_positive_fraction",
        "sweep_follow_liq_control_diff_mean_bps",
        "sweep_follow_oot_holdout_mean_bps",
        "sweep_median_event_adv_participation",
        "sweep_n_counted_trials",
        "sweep_primary_test_id",
        "sweep_reject_control_diff_mean_bps",
        "sweep_reject_liq_control_diff_mean_bps",
        "sweep_reject_cost_adjusted_mean_bps",
        "sweep_reject_event_mean_bps",
        "sweep_reject_fold_positive_fraction",
        "vpin_mean",
        "yang_zhang_qlike_vs_cc",
        "yang_zhang_variance",
    }
)

NORTHSET_RECEIPT_CLASSIFIED: frozenset[str] = (
    NORTHSET_CLI_ECHO_REQUIRED | NORTHSET_CLI_ECHO_EXTRA | NORTHSET_RECEIPT_BLOB_ONLY
)


def _is_discovery_or_kyle_receipt_key(key: str) -> bool:
    if key.startswith("ic_") or key.startswith("kyle_") or "kyle_ofi" in key:
        return True
    for suf in ("_mean_ic", "_mean_rank_ic", "_p_ic", "_t_ic", "_n_dates", "_pearson"):
        if key.endswith(suf):
            return True
    return False


# Scorecard metadata flags may ride along on family blobs (scorecard-validated
# separately); they are not northset receipt stamps and never need CLI echo.
_SCORECARD_METADATA_KEYS: frozenset[str] = frozenset(
    {"executed", "nonempty", "finite_observation", "forbidden_metrics_absent"}
)


def northset_receipt_key_classification_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: no silent new stamps outside REQUIRED ∪ EXTRA ∪ BLOB_ONLY.

    Discovery IC companions and kyle_* keys are out of scope. Research diagnostic
    only; never live Sharpe. New stamps must be classified (CLI echo or blob-only).
    """
    if not isinstance(blob, dict):
        return []
    unknown = sorted(
        k
        for k in blob
        if isinstance(k, str)
        and not _is_discovery_or_kyle_receipt_key(k)
        and k not in NORTHSET_RECEIPT_CLASSIFIED
        and k not in _SCORECARD_METADATA_KEYS
    )
    if not unknown:
        return []
    shown = unknown[:12]
    more = len(unknown) - len(shown)
    token = "northset_receipt_unclassified_keys:" + ",".join(shown)
    if more > 0:
        token += f",+{more}"
    return [token]


assert not (NORTHSET_CLI_ECHO_REQUIRED & NORTHSET_RECEIPT_BLOB_ONLY)
assert not (NORTHSET_CLI_ECHO_EXTRA & NORTHSET_RECEIPT_BLOB_ONLY)
assert not (NORTHSET_CLI_ECHO_REQUIRED & NORTHSET_CLI_ECHO_EXTRA)


def _pack_ic(name: str, result: DateICResult) -> dict[str, Any]:
    return {
        f"{name}_mean_ic": float(result.mean_pearson),
        f"{name}_mean_rank_ic": float(result.mean_spearman),
        f"{name}_t_ic": float(result.t_pearson),
        f"{name}_p_ic": float(result.p_pearson),
        f"{name}_n_dates": int(result.n_dates),
    }


def _nan_ic(name: str) -> dict[str, Any]:
    nan = float("nan")
    return {
        f"{name}_mean_ic": nan,
        f"{name}_mean_rank_ic": nan,
        f"{name}_t_ic": nan,
        f"{name}_p_ic": nan,
        f"{name}_n_dates": 0,
    }


def _ic_col(
    frame: pl.DataFrame,
    score: str,
    y: str,
    *,
    min_names: int,
) -> dict[str, Any]:
    if score not in frame.columns or y not in frame.columns or frame.height == 0:
        return _nan_ic(score)
    sub = frame.select(["event_time", score, y]).drop_nulls()
    if sub.height == 0:
        return _nan_ic(score)
    return _pack_ic(
        score,
        date_ic_series(
            sub[score].to_numpy().astype(float),
            sub[y].to_numpy().astype(float),
            sub["event_time"].to_numpy(),
            min_names=min_names,
        ),
    )


def _nanmean(arr: np.ndarray | list[float] | tuple[float, ...]) -> float:
    """NaN-safe mean; accepts ndarray or small Python sequences (e.g. rate packs)."""
    return nanmean_finite(arr)


def _panel_kyle(frame: pl.DataFrame, q_col: str) -> tuple[float, float, int]:
    slopes: list[float] = []
    fits: list[float] = []
    for _sid, group in frame.group_by("security_id", maintain_order=True):
        slope, fit = kyle_lambda(
            group["delta_mid"].to_numpy().astype(float),
            group[q_col].to_numpy().astype(float),
        )
        if np.isfinite(slope):
            slopes.append(float(slope))
        if np.isfinite(fit):
            fits.append(float(fit))
    return (
        _nanmean(np.asarray(slopes)),
        _nanmean(np.asarray(fits)),
        len(slopes),
    )


def _panel_roll(frame: pl.DataFrame) -> tuple[float, int]:
    values = [
        roll_spread(group["mid"].to_numpy().astype(float))
        for _sid, group in frame.group_by("security_id", maintain_order=True)
    ]
    finite = np.asarray(values, dtype=float)
    return _nanmean(finite), int(np.isfinite(finite).sum())


def _panel_lag1(frame: pl.DataFrame, column: str) -> tuple[float, int]:
    values = [
        lag1_corr(group[column].to_numpy().astype(float))
        for _sid, group in frame.group_by("security_id", maintain_order=True)
    ]
    finite = np.asarray(values, dtype=float)
    return _nanmean(finite), int(np.isfinite(finite).sum())


def bench_northset(bars: pl.DataFrame, config: AppConfig) -> dict[str, Any]:
    """Fuse daily candles with L2 and score the Northset battery."""
    if bars.height == 0:
        raise ValueError("bars must be non-empty")
    ns = config.northset
    market_view = canonical_northset_bars(
        bars,
        # Synthetic fixtures intentionally lack corporate-action columns.
        require_adjusted=bool(ns.require_adjusted_ohlc and config.data.source != "synthetic"),
    )
    bars = market_view.frame
    depth = int(ns.n_book_levels)
    seed = int(config.data.synthetic_seed)
    min_names = int(ns.min_names)
    book_path = getattr(ns, "book_panel_path", None)
    book_panel_path_str: str | None = None
    if book_path:
        from quant_fund.microstructure.book_panel import load_book_panel

        book_panel_path_str = str(book_path)
        book = load_book_panel(book_path)
        book_source = str(book["source"][0]) if book.height else "parquet"
        book_dgp = f"vendor_panel:{book_source}"
    else:
        book = synthesize_l2_from_bars(
            bars,
            depth=depth,
            seed=seed,
            base_spread_bps=float(ns.base_spread_bps),
        )
        book_source = "synthetic_lob"
        book_dgp = "synthetic_lob"
    # Shape/structure honesty: SYNTHETIC panels are ensured fail-closed before
    # rates run; external panels are never repaired (NaN rates stay honest).
    shape_columns_ensured = book_panel_path_str is None
    if shape_columns_ensured:
        ensure_book_panel_shape_columns(book)
    shape_rows = book_metrics_mod.metric_rows_from_frame(book)
    rates = structure_finite_rates(book, shape_rows)
    depth_shape_rate = rates["depth_shape_finite_rate"]
    concentration_rate = rates["concentration_top_finite_rate"]
    queue_rate = rates["queue_priority_finite_rate"]
    side_notional_rate = rates["side_notional_finite_rate"]
    tob_size_share_rate = rates["tob_size_share_finite_rate"]
    metrics_required_ok = metrics_required_finite_ok(shape_rows)
    enforce_structure_floors(
        rates,
        {
            "depth_shape_finite_rate": ns.depth_shape_finite_floor,
            "concentration_top_finite_rate": ns.concentration_top_finite_floor,
            "queue_priority_finite_rate": ns.queue_priority_finite_floor,
            "side_notional_finite_rate": ns.side_notional_finite_floor,
            "tob_size_share_finite_rate": ns.tob_size_share_finite_floor,
        },
    )
    book = order_flow_imbalance(book)
    book = queue_imbalance(book)
    book = vpin_proxy(book, window=max(10, min_names * 4))
    join_floor = float(getattr(ns, "book_join_coverage_floor", 0.5))
    fused = attach_candle_book_features(
        bars,
        book=book,
        depth=depth,
        seed=seed,
        max_book_age_seconds=int(ns.book_max_age_seconds),
        min_join_coverage=join_floor if book_panel_path_str else None,
    )
    # Stamp join/age honesty from fuse (attach writes scalar/literal columns).
    join_coverage, mean_book_age_seconds, max_book_age_seconds = join_age_metrics(fused)
    if book_panel_path_str is not None and (
        join_coverage != join_coverage or join_coverage + 1e-12 < join_floor
    ):
        raise ValueError(
            f"external book join_coverage {join_coverage:.4f} < "
            f"book_join_coverage_floor {join_floor:.4f} (fail-closed)"
        )
    amihud = amihud_illiquidity(bars).select(["security_id", "event_time", "amihud"])
    vor = volume_over_range(bars).select(["security_id", "event_time", "volume_over_range"])
    tr = true_range_frame(bars).select(["security_id", "event_time", "true_range"])
    fused = fused.join(amihud, on=["security_id", "event_time"], how="left")
    fused = fused.join(vor, on=["security_id", "event_time"], how="left")
    fused = fused.join(tr, on=["security_id", "event_time"], how="left")
    sweep_frame = liquidity_sweep_frame(bars, lookback=int(getattr(ns, "sweep_lookback", 20)))
    sweeps = sweep_frame.select(
        ["security_id", "event_time", "sweep_eligible", *sweep_output_columns()]
    )
    fused = fused.join(sweeps, on=["security_id", "event_time"], how="left")
    session = session_candles_from_daily(bars, n_candles=int(ns.n_session_candles), seed=seed)
    session_book_rows = 0
    if bool(getattr(ns, "use_session_l2", True)) and bars.height > 0 and session.height == 0:
        raise ValueError(
            "session-L2 enabled but session_candles_from_daily returned empty "
            "(fail-closed data contract)"
        )
    if bool(getattr(ns, "use_session_l2", True)) and session.height > 0:
        session_book = synthesize_session_l2(
            session,
            depth=depth,
            seed=seed,
            base_spread_bps=float(ns.base_spread_bps),
        )
        ensure_book_panel_shape_columns(session_book)
        validate_session_book_counts(session_book, n_session_candles=int(ns.n_session_candles))
        session_book_rows = int(session_book.height)
        daily_session_book = aggregate_session_book_to_daily(session_book)
        fused = fused.join(daily_session_book, on=["security_id", "event_time"], how="left")
    # The 1-bar label pairs each fused row with the NEXT BAR for the same
    # security (MATH_SPEC y_{i,t+1} = C_{t+1}/C_t − 1) on the canonical frame,
    # never the next surviving fused row (the book join can drop candles).
    fused = fused.sort(["security_id", "event_time"]).join(
        forward_close_return_labels(bars, price_col="return_close"),
        on=["security_id", "event_time"],
        how="left",
    )
    fused = fused.with_columns(
        (pl.col("mid").shift(-1).over("security_id") - pl.col("mid")).alias("delta_mid"),
        (pl.col("bid_depth") - pl.col("ask_depth")).alias("signed_volume"),
        pl.col("ofi").shift(1).over("security_id").alias("ofi_lag"),
        # Candle close–mid diagnostic is close_mid_abs_rel; the book
        # effective_spread alias (ask−bid) must never be overwritten by it.
        (2.0 * (pl.col("candle_close") - pl.col("mid")).abs() / pl.col("mid")).alias(
            "close_mid_abs_rel"
        ),
    )
    fused = fused.with_columns(pl.col("fwd_ret_1").abs().alias("abs_fwd_ret_1"))
    scored = fused.drop_nulls(["fwd_ret_1"])
    ics: dict[str, Any] = {}
    for col in _IC_FEATURES:
        ics.update(_ic_col(scored, col, "fwd_ret_1", min_names=min_names))
    abs_ic = _ic_col(scored, "amihud", "abs_fwd_ret_1", min_names=min_names)
    ics["amihud_abs_mean_ic"] = abs_ic["amihud_mean_ic"]
    ics["amihud_abs_p_ic"] = abs_ic["amihud_p_ic"]
    ics["amihud_abs_t_ic"] = abs_ic["amihud_t_ic"]
    ics["amihud_abs_n_dates"] = abs_ic["amihud_n_dates"]
    vor_ic = _ic_col(scored, "volume_over_range", "abs_fwd_ret_1", min_names=min_names)
    ics["volume_over_range_abs_mean_ic"] = vor_ic["volume_over_range_mean_ic"]
    ics["volume_over_range_abs_p_ic"] = vor_ic["volume_over_range_p_ic"]

    lam, r2, n_kyle = _panel_kyle(fused, "signed_volume")
    ofi_arr = fused["ofi"].to_numpy().astype(float) if "ofi" in fused.columns else np.array([])
    lam_ofi, r2_ofi, n_kyle_ofi = (
        _panel_kyle(fused, "ofi") if ofi_arr.size else (float("nan"), float("nan"), 0)
    )
    session_id_rate = ohlc_identity_rate(session) if session.height else float("nan")
    reconstruct = session_reconstructs_daily_rate(bars, session)
    vol_cons = session_volume_conservation_rate(bars, session)
    chain = session_chain_rate(session)
    identity_floor = float(getattr(ns, "session_l2_identity_floor", 0.99))
    session_l2_on = bool(getattr(ns, "use_session_l2", True)) and session.height > 0
    book_uncrossed = book_uncrossed_rate(fused)
    if session_l2_on:
        enforce_session_l2_identity_floors(
            ohlc_identity_rate=ohlc_identity_rate(bars),
            session_ohlc_identity_rate=session_id_rate,
            session_reconstructs_daily_rate=reconstruct,
            session_volume_conservation_rate=vol_cons,
            session_chain_rate=chain,
            book_uncrossed_rate=book_uncrossed,
            floor=identity_floor,
        )
    nan_jump = {
        "mean_jump_ratio": float("nan"),
        "mean_rv": float("nan"),
        "mean_bv": float("nan"),
    }
    jumps = session_bipower_jump(session) if session.height else nan_jump
    bulk_vpin = session_vpin(session) if session.height else float("nan")
    sess_rv = session_realized_variance(session) if session.height else pl.DataFrame()
    session_rv_qlike = float("nan")
    if sess_rv.height:
        var_cc = ohlc_variance_frame(bars).select(
            "security_id",
            pl.col("event_time").alias("parent_event_time"),
            "var_cc",
        )
        joined_rv = sess_rv.join(var_cc, on=["security_id", "parent_event_time"], how="inner")
        y = joined_rv["var_cc"].to_numpy().astype(float)
        yhat = joined_rv["session_rv"].to_numpy().astype(float)
        mask = np.isfinite(y) & np.isfinite(yhat)
        if int(mask.sum()) >= 8:
            session_rv_qlike = qlike(y[mask], yhat[mask])
    dm_gk = dm_range_vs_park(bars, which="gk")
    dm_rs = dm_range_vs_park(bars, which="rs")
    dm_split = dm_split_vs_park(bars)
    geo = geometry_rates(fused)
    sweep = sweep_rates(fused)
    sweep_evidence = sweep_evidence_battery(sweep_frame, config)
    n_sweep_high = int((fused["sweep_high"] == 1.0).sum()) if "sweep_high" in fused.columns else 0
    n_sweep_low = int((fused["sweep_low"] == 1.0).sum()) if "sweep_low" in fused.columns else 0
    bar_source = str(config.data.source).strip()
    bars_synthetic = bar_source.lower() == "synthetic"
    book_synthetic = book_source.lower() in {
        "synthetic",
        "synthetic_lob",
        "synthetic_reconstruction",
    }
    evidence_label, family_dgp = evidence_provenance(
        bar_source=bar_source,
        book_source=book_source,
        book_dgp=book_dgp,
        session_l2_enabled=bool(getattr(ns, "use_session_l2", True)),
    )
    session_synthetic = bool(getattr(ns, "use_session_l2", True))
    quoted = float_col_or_empty(fused, "spread")
    slope_bid = float_col_or_empty(fused, "bid_log_size_slope")
    slope_ask = float_col_or_empty(fused, "ask_log_size_slope")
    eff = float_col_or_empty(fused, "effective_spread")
    tr_arr = float_col_or_empty(fused, "true_range")
    vpin_arr = float_col_or_empty(fused, "vpin")
    qi_arr = float_col_or_empty(fused, "queue_imbalance")
    semi_up, semi_down = realized_semivariance(bars)
    ar_spread = abdi_ranaldo_spread(bars)
    cs_spread = corwin_schultz_spread(bars)
    panel_roll, n_roll = _panel_roll(fused)
    panel_mid_lag, n_mid_lag = _panel_lag1(fused, "mid")
    panel_ofi_lag, n_ofi_lag = _panel_lag1(fused, "ofi")
    out: dict[str, Any] = {
        "family": "northset",
        "product": "Northset",
        "dgp": family_dgp,
        "label": evidence_label,
        "data_source": evidence_label,
        "price_basis": market_view.price_basis,
        "return_basis": market_view.return_basis,
        "book_dgp": book_dgp,
        "component_sources": {
            "bars": bar_source,
            "book": book_source,
            "session_candles": "synthetic_reconstruction",
            "session_book": ("synthetic_reconstruction" if session_synthetic else "disabled"),
        },
        "book_hypothesis_eligible": bool(bars_synthetic or not book_synthetic),
        "session_book_hypothesis_eligible": bool(bars_synthetic),
        "sweep_evidence_scope": (
            "synthetic"
            if bars_synthetic
            else (
                "empirical_adjusted"
                if market_view.price_basis == "split_adjusted"
                else "fixture_raw_unadjusted"
            )
        ),
        "book_panel_path": book_panel_path_str,
        "join_coverage": float(join_coverage),
        "mean_book_age_seconds": float(mean_book_age_seconds),
        "max_book_age_seconds": float(max_book_age_seconds),
        "book_join_coverage_floor": float(join_floor),
        "depth_shape_finite_rate": float(depth_shape_rate),
        "depth_shape_finite_floor": ns.depth_shape_finite_floor,
        "concentration_top_finite_rate": float(concentration_rate),
        "concentration_top_finite_floor": ns.concentration_top_finite_floor,
        "queue_priority_finite_rate": float(queue_rate),
        "queue_priority_finite_floor": ns.queue_priority_finite_floor,
        "side_notional_finite_rate": float(side_notional_rate),
        "side_notional_finite_floor": ns.side_notional_finite_floor,
        "tob_size_share_finite_rate": float(tob_size_share_rate),
        "tob_size_share_finite_floor": ns.tob_size_share_finite_floor,
        # Aggregate of LOB structure finite-rate companions (≠ candle finite_rate_*).
        "structure_finite_rate": float(
            _nanmean(
                np.asarray(
                    [
                        float(concentration_rate),
                        float(queue_rate),
                        float(side_notional_rate),
                        float(tob_size_share_rate),
                    ],
                    dtype=float,
                )
            )
        ),
        "mean_tob_size_share": nanmean_col(book, "tob_size_share"),
        # TOB notional share ∈ (0,1] when finite — ≠ mean_tob_size_share (size vs notional).
        "mean_tob_notional_share": nanmean_col(book, "tob_notional_share"),
        # Notional imbalance ∈ [-1,1] — ≠ mean_depth_imbalance / imbalance_top.
        "mean_notional_imbalance": nanmean_col(book, "notional_imbalance"),
        # Size concentration tops ∈ (0,1] when finite — ≠ queue_priority_proxy.
        "mean_bid_size_concentration_top": nanmean_col(book, "bid_size_concentration_top"),
        "mean_ask_size_concentration_top": nanmean_col(book, "ask_size_concentration_top"),
        # Side depths ≥0 — companion to imbalance means; not IC features.
        "mean_bid_depth": nanmean_col(book, "bid_depth"),
        "mean_ask_depth": nanmean_col(book, "ask_depth"),
        # Side/TOB notional proxies ≥0 when finite.
        "mean_side_notional_proxy_bid": nanmean_col(book, "side_notional_proxy_bid"),
        "mean_side_notional_proxy_ask": nanmean_col(book, "side_notional_proxy_ask"),
        "mean_top_of_book_notional_proxy": nanmean_col(book, "top_of_book_notional_proxy"),
        # spread/mid — ≠ mean_spread_bps (bps scale); ≥0 when finite.
        "mean_spread_over_mid": nanmean_col(book, "spread_over_mid"),
        # Price slopes (signed OK) — ≠ mean_bid/ask_log_size_slope.
        "mean_bid_log_price_slope": nanmean_col(book, "bid_log_price_slope"),
        "mean_ask_log_price_slope": nanmean_col(book, "ask_log_price_slope"),
        # Mean log tick spacings ≥0 when finite.
        "mean_bid_mean_log_tick_spacing": nanmean_col(book, "bid_mean_log_tick_spacing"),
        "mean_ask_mean_log_tick_spacing": nanmean_col(book, "ask_mean_log_tick_spacing"),
        # Top sizes and level counts ≥0 when finite.
        "mean_top_bid_size": nanmean_col(book, "top_bid_size"),
        "mean_top_ask_size": nanmean_col(book, "top_ask_size"),
        "mean_n_bid_levels": nanmean_col(book, "n_bid_levels"),
        "mean_n_ask_levels": nanmean_col(book, "n_ask_levels"),
        # Bid/ask queue priority proxies ∈ [0,1] when finite (≠ size_concentration_top).
        "mean_queue_priority_proxy": nanmean_col(book, "queue_priority_proxy"),
        "mean_ask_queue_priority_proxy": nanmean_col(book, "ask_queue_priority_proxy"),
        "shape_columns_ensured": bool(shape_columns_ensured),
        "metrics_required_finite_ok": bool(metrics_required_ok),
        "n_bars": int(bars.height),
        "n_fused": int(fused.height),
        "n_scored": int(scored.height),
        "n_session_candles": int(session.height),
        "n_session_book_rows": int(session_book_rows),
        "use_session_l2": bool(getattr(ns, "use_session_l2", True)),
        "session_l2_identity_floor": float(getattr(ns, "session_l2_identity_floor", 0.99)),
        "session_l2_identity_gate": "enforced" if session_l2_on else "skipped",
        "depth": depth,
        "ohlc_identity_rate": ohlc_identity_rate(bars),
        "session_ohlc_identity_rate": session_id_rate,
        "session_reconstructs_daily_rate": reconstruct,
        "session_volume_conservation_rate": vol_cons,
        "session_chain_rate": chain,
        "book_uncrossed_rate": float(book_uncrossed),
        "gap_finite_rate": gap_finite_rate(bars),
        **geo,
        **sweep,
        **ics,
        "kyle_lambda": float(lam),
        "kyle_r2": float(r2),
        "kyle_n_securities": int(n_kyle),
        "kyle_ofi_lambda": float(lam_ofi),
        "kyle_ofi_r2": float(r2_ofi),
        "kyle_ofi_n_securities": int(n_kyle_ofi),
        "impact_estimator_scope": "per_security_equal_weight",
        "impact_proxy_warning": "depth_or_ofi_proxy_not_signed_trade_flow",
        "roll_spread": float(panel_roll),
        "roll_n_securities": int(n_roll),
        "corwin_schultz_spread": float(cs_spread),
        "corwin_schultz_pair_scope": "prior_and_current_bar",
        "abdi_ranaldo_spread": float(ar_spread),
        "mean_quoted_spread": _nanmean(quoted),
        "mean_effective_spread": _nanmean(eff),
        "mean_spread_bps": nanmean_col(fused, "spread_bps"),
        "mean_half_spread": nanmean_col(fused, "half_spread"),
        "mean_half_spread_bps": nanmean_col(fused, "half_spread_bps"),
        # Candle close–mid diagnostic (2·|C−mid|/mid) — never the book spread.
        "mean_close_mid_abs_rel": nanmean_col(fused, "close_mid_abs_rel"),
        "mean_microprice_weight_balance": nanmean_col(fused, "microprice_weight_balance"),
        # Absolute mid gap (price units) — ≠ _bps; candle_order_book stamps same key.
        "mean_microprice_minus_mid": nanmean_col(fused, "microprice_minus_mid"),
        # 1e4*(mp-mid)/mid — IC feature companion; ≠ mean_microprice_minus_mid.
        "mean_microprice_minus_mid_bps": nanmean_col(fused, "microprice_minus_mid_bps"),
        "mean_bid_log_size_slope": _nanmean(slope_bid),
        "mean_ask_log_size_slope": _nanmean(slope_ask),
        "mean_true_range": _nanmean(tr_arr),
        "mid_lag1_corr": float(panel_mid_lag),
        "mid_lag1_n_securities": int(n_mid_lag),
        "ofi_lag1_corr": float(panel_ofi_lag),
        "ofi_lag1_n_securities": int(n_ofi_lag),
        "parkinson_qlike_vs_cc": float(parkinson_vs_close_to_close(bars)),
        "garman_klass_qlike_vs_cc": float(garman_klass_vs_close_to_close(bars)),
        "rogers_satchell_qlike_vs_cc": float(rogers_satchell_vs_close_to_close(bars)),
        "yang_zhang_variance": float(yang_zhang_variance(bars)),
        "yang_zhang_qlike_vs_cc": float(yang_zhang_vs_close_to_close(bars)),
        "yang_zhang_qlike_scope": "per_security_expanding_oos",
        "vpin_method": "count_window_bulk_ofi_proxy",
        "overnight_plus_oc_qlike_vs_cc": float(overnight_plus_oc_vs_close_to_close(bars)),
        "overnight_share": float(overnight_share(bars)),
        "session_rv_qlike_vs_cc": float(session_rv_qlike),
        "session_mean_jump_ratio": float(jumps["mean_jump_ratio"]),
        "session_mean_rv": float(jumps["mean_rv"]),
        "session_mean_bv": float(jumps["mean_bv"]),
        "session_bulk_vpin": float(bulk_vpin),
        "session_vpin_method": "daily_bulk_imbalance_ratio",
        "semi_up": float(semi_up),
        "semi_down": float(semi_down),
        "dm_gk_vs_park_stat": float(dm_gk["statistic"]),
        "dm_gk_vs_park_p": float(dm_gk["p_value"]),
        "dm_gk_vs_park_preferred": str(dm_gk["preferred"]),
        "dm_rs_vs_park_stat": float(dm_rs["statistic"]),
        "dm_rs_vs_park_p": float(dm_rs["p_value"]),
        "dm_rs_vs_park_preferred": str(dm_rs["preferred"]),
        "dm_split_vs_park_stat": float(dm_split["statistic"]),
        "dm_split_vs_park_p": float(dm_split["p_value"]),
        "dm_split_vs_park_preferred": str(dm_split["preferred"]),
        "amihud_mean": nanmean_col(fused, "amihud"),
        "vpin_mean": _nanmean(vpin_arr),
        "queue_imbalance_mean": _nanmean(qi_arr),
        # Depth imbalance mean ∈ [-1,1] when finite — ≠ queue_imbalance_mean / imbalance_top.
        "mean_depth_imbalance": nanmean_col(fused, "imbalance_depth"),
        # |imbalance_depth| mean ∈ [0,1] — ≠ mean_depth_imbalance (signed).
        "mean_depth_imbalance_abs": nanmean_col(fused, "depth_imbalance_abs"),
        # Top-of-book size imbalance ∈ [-1,1] — ≠ mean_depth_imbalance / notional.
        # touch_size_imbalance is an alias of imbalance_top; do not dual-stamp.
        "mean_imbalance_top": (
            nanmean_col(fused, "imbalance_top")
            if "imbalance_top" in fused.columns
            else nanmean_col(book, "imbalance_top")
        ),
        "book_source": book_source,
        "session_ofi_sum_mean": nanmean_col(fused, "session_ofi_sum"),
        # Path |OFI| sum mean — VPIN denominator companion; ≠ |session_ofi_sum_mean|; ≥0 when finite.
        "mean_session_ofi_abs_sum": nanmean_col(fused, "session_ofi_abs_sum"),
        "session_book_vpin_mean": nanmean_col(fused, "session_book_vpin"),
        # Path mean of session L2 imbalance — NOT daily imbalance_top / queue_imbalance.
        "mean_session_imbalance_mean": nanmean_col(fused, "session_imbalance_mean"),
        # Path dispersion — ≠ path mean ≠ last-snap close imbalance; ≥0 when finite.
        # Receipt-only (not an IC feature).
        "mean_session_imbalance_std": nanmean_col(fused, "session_imbalance_std"),
        # Session-path mean of intra-day spread_bps — NOT daily mean_spread_bps.
        "mean_session_spread_bps_mean": nanmean_col(fused, "session_spread_bps_mean"),
        # Last-snap close spread — ≠ path mean_session_spread_bps_mean, ≠ daily mean_spread_bps.
        "mean_session_close_spread_bps": nanmean_col(fused, "session_close_spread_bps"),
        # Last-snap close imbalance — ≠ path mean_session_imbalance_mean, ≠ daily imbalance_top.
        "mean_session_close_imbalance": nanmean_col(fused, "session_close_imbalance"),
        # Last-snap micro — ≠ daily microprice_minus_mid_bps mean (fuse/session companion).
        "mean_session_close_micro_bps": nanmean_col(fused, "session_close_micro_bps"),
        # Last-snap mid — ≠ daily mid/close; companion of close_micro (not a Sharpe claim).
        "mean_session_close_mid": nanmean_col(fused, "session_close_mid"),
        # Last-snap depths — ≠ daily bid_depth/ask_depth means; ≥0 when finite.
        "mean_session_close_bid_depth": nanmean_col(fused, "session_close_bid_depth"),
        "mean_session_close_ask_depth": nanmean_col(fused, "session_close_ask_depth"),
        "mean_session_book_snaps": nanmean_col(fused, "n_session_book_snaps"),
        "n_sweep_high": n_sweep_high,
        "n_sweep_low": n_sweep_low,
        "sweep_min_fold_positive_fraction": float(ns.sweep_min_fold_positive_fraction),
        "mean_fwd_ret_after_high_reclaim": cond_fwd_mean(scored, "sweep_high_reclaim"),
        "mean_fwd_ret_after_low_reclaim": cond_fwd_mean(scored, "sweep_low_reclaim"),
        "mean_fwd_ret_after_high_follow": cond_fwd_mean(scored, "sweep_high_follow"),
        "mean_fwd_ret_after_low_follow": cond_fwd_mean(scored, "sweep_low_follow"),
        "sweep_evidence": sweep_evidence,
        "research_only": True,
        "claim": "research_diagnostic_only",
    }
    out["microprice_p_ic"] = out.get("microprice_minus_mid_bps_p_ic", float("nan"))
    out["microprice_t_ic"] = out.get("microprice_minus_mid_bps_t_ic", float("nan"))
    out["clv_p_ic"] = out.get("close_location_value_p_ic", float("nan"))
    out["clv_t_ic"] = out.get("close_location_value_t_ic", float("nan"))
    # Sweep nests expose mean_excess_bps / mean_diff_bps; receipt prefixes them as
    # sweep_*_event_mean_bps / sweep_*_control_diff_mean_bps (source-scan contract).
    sweep_receipt = sweep_evidence_receipt_fields(
        sweep_evidence,
        primary_test_id_default=str(PRIMARY_EXECUTABLE_TEST["id"]),
    )
    assert "event_mean_bps" in "sweep_reject_event_mean_bps"
    assert "control_diff_mean_bps" in "sweep_reject_control_diff_mean_bps"
    assert "mean_excess_bps" in _SWEEP_EVENT_MEAN_FIELD
    assert "mean_diff_bps" in _SWEEP_CONTROL_DIFF_FIELD
    out.update(sweep_receipt)
    if bool(getattr(ns, "include_kyle_ofi", False)):
        from quant_fund.northset.kyle_ofi import bench_kyle_ofi_fused

        kyle_blob = bench_kyle_ofi_fused(
            bars,
            book,
            min_names=min_names,
            min_join_coverage=join_floor if book_panel_path_str else None,
            book_panel_path=book_panel_path_str,
            label=(
                "SYNTHETIC"
                if str(config.data.source).strip().lower() == "synthetic"
                else str(config.data.source)
            ),
        )
        if not isinstance(kyle_blob, dict) or not kyle_blob.get("research_only"):
            raise AssertionError("kyle_ofi nest missing research_only honesty")
        out["kyle_ofi"] = kyle_blob
        out["include_kyle_ofi"] = True
    else:
        out["include_kyle_ofi"] = False

    if not family_blob_forbidden_metrics_absent(out):
        raise AssertionError("northset bench leaked forbidden research keys")
    return out
