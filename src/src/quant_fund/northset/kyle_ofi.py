"""Kyle λ and OFI→Δmid research diagnostics (SYNTHETIC / research_only).

Fuses candle/OHLCV bars with L2 top-of-book metrics, then scores with
**date-level IC** via ``quant_fund.metrics.cross_section.date_ic_series``
(Pearson/Spearman IC per date, then Newey–West HAC t/p on the IC series).

Never pooled stacked Spearman. Never a live-tape or live-P&L claim.

Formulas (see docs/MATH_SPEC.md):

* Cont–Kukanov–Stoikov (2014) OFI on consecutive top-of-book snapshots.
* Kyle (1985) λ from OLS ``Δm_t = λ q_t + ε_t`` with ``q`` = signed depth
  (``bid_depth - ask_depth``) or OFI.

Label every receipt SYNTHETIC / research_only. Forbidden Sharpe/pnl keys stay out.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_fund.metrics.cross_section import date_ic_series
from quant_fund.metrics.inference import mean_tstat

Array = NDArray[np.float64]

FlowSource = Literal["signed_depth", "ofi"]


def cont_ofi_series(
    best_bid: Array,
    top_bid_size: Array,
    best_ask: Array,
    top_ask_size: Array,
) -> Array:
    """Cont–Kukanov–Stoikov OFI for one security path. First observation is 0."""
    bp = np.asarray(best_bid, dtype=float)
    bs = np.asarray(top_bid_size, dtype=float)
    ap = np.asarray(best_ask, dtype=float)
    az = np.asarray(top_ask_size, dtype=float)
    n = int(bp.size)
    if not all(int(x.size) == n for x in (bs, ap, az)):
        raise ValueError("OFI inputs must share length")
    ofi = np.zeros(n, dtype=float)
    for i in range(1, n):
        if not all(
            np.isfinite(v)
            for v in (bp[i], bp[i - 1], bs[i], bs[i - 1], ap[i], ap[i - 1], az[i], az[i - 1])
        ):
            ofi[i] = float("nan")
            continue
        if bp[i] >= bp[i - 1]:
            ofi[i] += bs[i]
        if bp[i] <= bp[i - 1]:
            ofi[i] -= bs[i - 1]
        if ap[i] <= ap[i - 1]:
            ofi[i] -= az[i]
        if ap[i] >= ap[i - 1]:
            ofi[i] += az[i - 1]
    return ofi


def cont_ofi_by_security(book: pl.DataFrame) -> pl.DataFrame:
    """Add Cont OFI column per ``security_id`` chronology on a book panel.

    Required columns: ``security_id``, ``event_time``, ``best_bid``, ``best_ask``,
    ``top_bid_size``, ``top_ask_size``. Returns sorted frame with ``ofi``.
    """
    required = (
        "security_id",
        "event_time",
        "best_bid",
        "best_ask",
        "top_bid_size",
        "top_ask_size",
    )
    missing = [c for c in required if c not in book.columns]
    if missing:
        raise ValueError(f"book missing required columns: {missing}")
    if book.height == 0:
        return book.with_columns(pl.Series("ofi", [], dtype=pl.Float64))

    ordered = book.sort(["security_id", "event_time"])
    parts: list[pl.DataFrame] = []
    for _sid, grp in ordered.group_by("security_id", maintain_order=True):
        ofi = cont_ofi_series(
            grp["best_bid"].to_numpy(),
            grp["top_bid_size"].to_numpy(),
            grp["best_ask"].to_numpy(),
            grp["top_ask_size"].to_numpy(),
        )
        parts.append(grp.with_columns(pl.Series("ofi", ofi)))
    return pl.concat(parts).sort(["security_id", "event_time"])


def kyle_lambda_ols(delta_mid: Array, flow: Array) -> float:
    """OLS Kyle λ: ``Δmid = λ · flow + ε``. Returns NaN if under-identified."""
    y = np.asarray(delta_mid, dtype=float)
    x = np.asarray(flow, dtype=float)
    if y.shape != x.shape:
        raise ValueError("delta_mid and flow must align")
    mask = np.isfinite(y) & np.isfinite(x)
    if int(mask.sum()) < 3:
        return float("nan")
    y = y[mask]
    x = x[mask]
    x_var = float(np.dot(x - x.mean(), x - x.mean()))
    if x_var <= 1e-18:
        return float("nan")
    # demeaned OLS slope (intercept absorbed)
    cov = float(np.dot(x - x.mean(), y - y.mean()))
    return cov / x_var


def _stamp_book_honesty(book: pl.DataFrame, *, synthesized: bool) -> tuple[str, str]:
    """Require source honesty for external panels; return (book_source, book_dgp)."""
    if synthesized:
        return "synthetic_lob", "synthetic_lob"
    if "source" not in book.columns:
        raise ValueError(
            "external book panel missing required 'source' column "
            "(stamp vendor/synthetic before fuse — fail-closed honesty)"
        )
    sources = (
        book.select(pl.col("source").cast(pl.Utf8)).unique().to_series().drop_nulls().to_list()
    )
    if not sources:
        raise ValueError("external book panel has empty/null source values")
    if len(sources) > 1:
        raise ValueError(f"external book panel mixes sources {sources!r} — refuse silent DGP mix")
    book_source = str(sources[0])
    if book_source.lower() in {"synthetic", "synthetic_lob"}:
        return book_source, "synthetic_lob"
    return book_source, f"vendor_panel:{book_source}"


def fuse_bars_l2_kyle_frame(
    bars: pl.DataFrame,
    book: pl.DataFrame | None = None,
    *,
    mid_col: str = "mid",
    min_join_coverage: float | None = None,
) -> pl.DataFrame:
    """Join bars to L2 metrics and add ``signed_depth``, ``ofi``, ``delta_mid``.

    If ``book`` is None, bars must already carry top-of-book columns (SYNTHETIC path).
    External panels must stamp a single ``source`` column. Always writes
    ``book_source`` / ``book_dgp`` / ``join_coverage``. Fail-closed on empty join
    and coverage below ``min_join_coverage`` (1.0 synthetic, 0.5 external).
    """
    if bars.height == 0:
        raise ValueError("bars must be non-empty")
    n_bar_rows = int(bars.height)
    synthesized = book is None
    frame = bars
    book_source, book_dgp = "synthetic_lob", "synthetic_lob"
    if book is not None:
        book_source, book_dgp = _stamp_book_honesty(book, synthesized=False)
        join_keys = ["security_id", "event_time"]
        for k in join_keys:
            if k not in bars.columns or k not in book.columns:
                raise ValueError(f"bars/book missing join key {k}")
        overlap = [c for c in book.columns if c in bars.columns and c not in join_keys]
        book_use = book.drop(overlap) if overlap else book
        frame = bars.join(book_use, on=join_keys, how="inner")
        if frame.height == 0:
            raise ValueError("bars/book join produced empty frame (timestamp mismatch)")
    else:
        if "source" in bars.columns:
            book_source, book_dgp = _stamp_book_honesty(bars, synthesized=False)
        else:
            book_source, book_dgp = "synthetic_lob", "synthetic_lob"

    join_coverage = float(frame.height) / float(n_bar_rows) if n_bar_rows else 0.0
    if min_join_coverage is None:
        min_join_coverage = 1.0 if synthesized else 0.5
    if join_coverage + 1e-12 < float(min_join_coverage):
        raise ValueError(
            f"bars/book join coverage {join_coverage:.4f} < min_join_coverage "
            f"{float(min_join_coverage):.4f} (n_bars={n_bar_rows}, n_fused={frame.height})"
        )

    need = (
        "security_id",
        "event_time",
        mid_col,
        "best_bid",
        "best_ask",
        "top_bid_size",
        "top_ask_size",
    )
    missing = [c for c in need if c not in frame.columns]
    if missing:
        raise ValueError(f"fused frame missing columns: {missing}")

    if "bid_depth" not in frame.columns:
        frame = frame.with_columns(pl.col("top_bid_size").alias("bid_depth"))
    if "ask_depth" not in frame.columns:
        frame = frame.with_columns(pl.col("top_ask_size").alias("ask_depth"))

    fwd_exprs = [
        (pl.col("bid_depth") - pl.col("ask_depth")).alias("signed_depth"),
        (pl.col(mid_col).shift(-1).over("security_id") - pl.col(mid_col)).alias("delta_mid"),
        (pl.col(mid_col).shift(-1).over("security_id") - pl.col(mid_col)).alias("fwd_delta_mid"),
        (pl.col(mid_col).shift(-1).over("security_id") / pl.col(mid_col) - 1.0).alias("fwd_ret_1"),
        (pl.col(mid_col).shift(-2).over("security_id") / pl.col(mid_col) - 1.0).alias("fwd_ret_2"),
        (pl.col(mid_col).shift(-3).over("security_id") / pl.col(mid_col) - 1.0).alias("fwd_ret_3"),
        pl.lit(book_source).alias("book_source"),
        pl.lit(book_dgp).alias("book_dgp"),
        pl.lit(float(join_coverage)).alias("join_coverage"),
    ]
    # Prefer close-to-close forward return when OHLCV close is present
    if "close" in frame.columns:
        fwd_exprs.append(
            (pl.col("close").shift(-1).over("security_id") / pl.col("close") - 1.0).alias(
                "fwd_close_ret_1"
            )
        )
        fwd_exprs.append(
            (pl.col("close").shift(-2).over("security_id") / pl.col("close") - 1.0).alias(
                "fwd_close_ret_2"
            )
        )
        fwd_exprs.append(
            (pl.col("close").shift(-3).over("security_id") / pl.col("close") - 1.0).alias(
                "fwd_close_ret_3"
            )
        )
    frame = frame.sort(["security_id", "event_time"]).with_columns(fwd_exprs)
    if "fwd_close_ret_1" in frame.columns:
        frame = frame.with_columns(pl.col("fwd_close_ret_1").alias("fwd_ret_1"))
    if "fwd_close_ret_2" in frame.columns:
        frame = frame.with_columns(pl.col("fwd_close_ret_2").alias("fwd_ret_2"))
    if "fwd_close_ret_3" in frame.columns:
        frame = frame.with_columns(pl.col("fwd_close_ret_3").alias("fwd_ret_3"))
    if "ofi" not in frame.columns:
        ofi_frame = cont_ofi_by_security(
            frame.select(
                "security_id",
                "event_time",
                "best_bid",
                "best_ask",
                "top_bid_size",
                "top_ask_size",
            )
        )
        frame = frame.join(
            ofi_frame.select("security_id", "event_time", "ofi"),
            on=["security_id", "event_time"],
            how="left",
        )
    return frame.sort(["security_id", "event_time"])


def ensure_forward_targets(fused: pl.DataFrame, *, mid_col: str = "mid") -> pl.DataFrame:
    """Add missing ``fwd_ret_*`` / ``fwd_delta_mid`` columns without editing fuse.

    research_only helper for residual/predictive paths when an older fuse
    frame lacks horizon stamps. Prefer close returns when ``close`` exists.
    """
    frame = fused
    if mid_col not in frame.columns and "close" in frame.columns:
        mid_col = "close"
    if mid_col not in frame.columns:
        return frame
    extras: list[pl.Expr] = []
    if "fwd_delta_mid" not in frame.columns:
        extras.append(
            (pl.col(mid_col).shift(-1).over("security_id") - pl.col(mid_col)).alias("fwd_delta_mid")
        )
    for h, name in ((1, "fwd_ret_1"), (2, "fwd_ret_2"), (3, "fwd_ret_3")):
        if name not in frame.columns:
            px = "close" if "close" in frame.columns else mid_col
            extras.append((pl.col(px).shift(-h).over("security_id") / pl.col(px) - 1.0).alias(name))
    if extras:
        frame = frame.sort(["security_id", "event_time"]).with_columns(extras)
    return frame


def residual_flow_date_ic(
    fused: pl.DataFrame,
    *,
    flow: FlowSource = "ofi",
    control: FlowSource = "signed_depth",
    target: str = "fwd_delta_mid",
    min_names: int = 3,
    hac_lags: int | None = None,
) -> dict[str, Any]:
    """Date-level IC of residual flow vs target after within-date linear control.

    residual = flow − proj(flow onto control) in each date cross-section.
    Scores with ``date_ic_series`` + HAC. research_diagnostic_only.
    """
    fused = ensure_forward_targets(fused)
    flow_col = "signed_depth" if flow == "signed_depth" else "ofi"
    ctrl_col = "signed_depth" if control == "signed_depth" else "ofi"
    if flow_col == ctrl_col:
        raise ValueError("flow and control must differ")
    if target not in fused.columns:
        raise ValueError(f"fused missing target {target}")
    for col in (flow_col, ctrl_col):
        if col not in fused.columns:
            raise ValueError(f"fused missing {col}")

    parts: list[pl.DataFrame] = []
    ordered = fused.sort(["event_time", "security_id"])
    for _date, grp in ordered.group_by("event_time", maintain_order=True):
        if grp.height < min_names:
            continue
        f = grp[flow_col].to_numpy().astype(float)
        c = grp[ctrl_col].to_numpy().astype(float)
        y = grp[target].to_numpy().astype(float)
        mask = np.isfinite(f) & np.isfinite(c) & np.isfinite(y)
        if int(mask.sum()) < min_names:
            continue
        f2, c2 = f[mask], c[mask]
        c_var = float(np.dot(c2 - c2.mean(), c2 - c2.mean()))
        if c_var <= 1e-18:
            resid = f2 - f2.mean()
        else:
            beta = float(np.dot(c2 - c2.mean(), f2 - f2.mean()) / c_var)
            resid = f2 - (f2.mean() + beta * (c2 - c2.mean()))
        keep = [bool(x) for x in mask.tolist()]
        sub = grp.filter(pl.Series(keep)).with_columns(pl.Series("residual_flow", resid))
        parts.append(sub)
    if not parts:
        raise ValueError("no dates with enough names for residual flow IC")
    sample = pl.concat(parts)
    ic = date_ic_series(
        sample["residual_flow"].to_numpy().astype(float),
        sample[target].to_numpy().astype(float),
        sample["event_time"].to_numpy(),
        min_names=min_names,
        hac_lags=hac_lags,
    )
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
    out: dict[str, Any] = {
        "family": "kyle_ofi",
        "diagnostic": f"residual_{flow}_ex_{control}_to_{target}",
        "label": "SYNTHETIC",
        "data_source": "SYNTHETIC" if book_dgp == "synthetic_lob" else book_source,
        "dgp": book_dgp,
        "book_source": book_source,
        "book_dgp": book_dgp,
        "research_only": True,
        "claim": "research_diagnostic_only",
        "ic_method": "date_level_spearman_hac",
        "flow": flow,
        "control": control,
        "target": target,
        "n_rows": int(sample.height),
        "mean_spearman": float(ic.mean_spearman),
        "mean_pearson": float(ic.mean_pearson),
        "t_spearman": float(ic.t_spearman),
        "p_spearman": float(ic.p_spearman),
        "n_dates": float(ic.n_dates),
        "min_names": int(min_names),
    }
    _assert_no_forbidden(out)
    return out


def kyle_lambda_dispersion(
    lam_series: Array,
    *,
    window: int = 10,
    hac_lags: int | None = None,
) -> dict[str, Any]:
    """Cross-date Kyle λ dispersion deciles + trailing-window HAC band.

    research_only diagnostic — not a live risk limit. Deciles are on the full
    per-date λ series; rolling band uses the last ``window`` finite λs.
    """
    lam = np.asarray(lam_series, dtype=float)
    lam = lam[np.isfinite(lam)]
    out: dict[str, Any] = {
        "research_only": True,
        "claim": "research_diagnostic_only",
        "n_dates": int(lam.size),
        "window": int(window),
    }
    if lam.size == 0:
        for q in (10, 20, 30, 40, 50, 60, 70, 80, 90):
            out[f"kyle_lambda_p{q}"] = float("nan")
        out.update(
            {
                "kyle_lambda_rolling_mean": float("nan"),
                "kyle_lambda_rolling_hac_t": float("nan"),
                "kyle_lambda_rolling_hac_p": float("nan"),
                "kyle_lambda_rolling_hac_lo": float("nan"),
                "kyle_lambda_rolling_hac_hi": float("nan"),
            }
        )
        return out

    for q in (10, 20, 30, 40, 50, 60, 70, 80, 90):
        out[f"kyle_lambda_p{q}"] = float(np.nanpercentile(lam, q))

    w = max(int(window), 3)
    trail = lam[-w:] if lam.size >= w else lam
    if trail.size >= 3:
        r_mean, r_t, r_p = mean_tstat(trail, hac_lags)
    else:
        r_mean = float(np.mean(trail)) if trail.size else float("nan")
        r_t, r_p = float("nan"), float("nan")
    # Approx HAC band from mean ± 1.96 * se with se = |mean/t|
    if np.isfinite(r_mean) and np.isfinite(r_t) and abs(float(r_t)) > 1e-12:
        se = abs(float(r_mean) / float(r_t))
        lo = float(r_mean) - 1.96 * se
        hi = float(r_mean) + 1.96 * se
    else:
        lo = hi = float("nan")
    out["kyle_lambda_rolling_mean"] = float(r_mean)
    out["kyle_lambda_rolling_hac_t"] = float(r_t)
    out["kyle_lambda_rolling_hac_p"] = float(r_p)
    out["kyle_lambda_rolling_hac_lo"] = lo
    out["kyle_lambda_rolling_hac_hi"] = hi
    _assert_no_forbidden(out)
    return out


def kyle_lambda_date_series_frame(
    fused: pl.DataFrame,
    *,
    flow: FlowSource = "signed_depth",
    target: str = "delta_mid",
    min_names: int = 3,
) -> pl.DataFrame:
    """Export per-date Kyle λ as a research_only panel (no Sharpe/pnl).

    Columns: event_time, kyle_lambda, flow, target, book_source, book_dgp, research_only.
    """
    receipt = kyle_lambda_by_date(fused, flow=flow, target=target, min_names=min_names)
    dates = receipt.get("kyle_lambda_dates") or []
    series = receipt.get("kyle_lambda_series") or []
    if len(dates) != len(series):
        raise ValueError("kyle_lambda dates/series length mismatch")
    book_source = str(receipt.get("book_source", "synthetic_lob"))
    book_dgp = str(receipt.get("book_dgp", "synthetic_lob"))
    if not series:
        return pl.DataFrame(
            schema={
                "event_time": pl.Utf8,
                "kyle_lambda": pl.Float64,
                "flow": pl.Utf8,
                "target": pl.Utf8,
                "book_source": pl.Utf8,
                "book_dgp": pl.Utf8,
                "research_only": pl.Boolean,
                "claim": pl.Utf8,
            }
        )
    return pl.DataFrame(
        {
            "event_time": dates,
            "kyle_lambda": [float(x) for x in series],
            "flow": [str(flow)] * len(series),
            "target": [str(target)] * len(series),
            "book_source": [book_source] * len(series),
            "book_dgp": [book_dgp] * len(series),
            "research_only": [True] * len(series),
            "claim": ["research_diagnostic_only"] * len(series),
        }
    )


def kyle_lambda_ofi_depth_corr(
    fused: pl.DataFrame,
    *,
    target: str = "delta_mid",
    min_names: int = 3,
    hac_lags: int | None = None,
) -> dict[str, Any]:
    """Spearman/Pearson correlation of per-date Kyle λ (OFI vs signed_depth).

    Aligns on shared event_time. research_only — not a live claim.
    """
    depth = kyle_lambda_by_date(
        fused, flow="signed_depth", target=target, min_names=min_names, hac_lags=hac_lags
    )
    ofi = kyle_lambda_by_date(
        fused, flow="ofi", target=target, min_names=min_names, hac_lags=hac_lags
    )
    depth_dates = depth.get("kyle_lambda_dates") or []
    depth_series = depth.get("kyle_lambda_series") or []
    ofi_dates = ofi.get("kyle_lambda_dates") or []
    ofi_series = ofi.get("kyle_lambda_series") or []
    if len(depth_dates) != len(depth_series) or len(ofi_dates) != len(ofi_series):
        raise ValueError("kyle_lambda dates/series length mismatch")
    d_map = {
        str(d): float(v)
        for d, v in zip(
            depth_dates,
            depth_series,
            strict=True,
        )
    }
    o_map = {
        str(d): float(v)
        for d, v in zip(
            ofi_dates,
            ofi_series,
            strict=True,
        )
    }
    shared = sorted(set(d_map) & set(o_map))
    if len(shared) < 3:
        spearman = pearson = t_s = p_s = float("nan")
        n = float(len(shared))
    else:
        x = np.asarray([d_map[k] for k in shared], dtype=float)
        y = np.asarray([o_map[k] for k in shared], dtype=float)
        # rank Spearman
        rx = x.argsort().argsort().astype(float)
        ry = y.argsort().argsort().astype(float)
        if np.std(rx) < 1e-18 or np.std(ry) < 1e-18:
            spearman = float("nan")
        else:
            spearman = float(np.corrcoef(rx, ry)[0, 1])
        if np.std(x) < 1e-18 or np.std(y) < 1e-18:
            pearson = float("nan")
        else:
            pearson = float(np.corrcoef(x, y)[0, 1])
        # HAC on Fisher-ish series proxy: demeaned product as weak dependence probe
        prod = (x - x.mean()) * (y - y.mean())
        _, t_s, p_s = mean_tstat(prod, hac_lags)
        n = float(len(shared))

    book_source = str(depth.get("book_source", "synthetic_lob"))
    book_dgp = str(depth.get("book_dgp", "synthetic_lob"))
    out: dict[str, Any] = {
        "family": "kyle_ofi",
        "diagnostic": "kyle_lambda_ofi_vs_signed_depth_corr",
        "label": "SYNTHETIC",
        "data_source": "SYNTHETIC" if book_dgp == "synthetic_lob" else book_source,
        "dgp": book_dgp,
        "book_source": book_source,
        "book_dgp": book_dgp,
        "research_only": True,
        "claim": "research_diagnostic_only",
        "ic_method": "date_level_spearman_hac",
        "target": target,
        "n_dates_aligned": n,
        "kyle_lambda_ofi_depth_spearman": spearman,
        "kyle_lambda_ofi_depth_pearson": pearson,
        "kyle_lambda_ofi_depth_prod_hac_t": float(t_s) if n >= 3 else float("nan"),
        "kyle_lambda_ofi_depth_prod_hac_p": float(p_s) if n >= 3 else float("nan"),
        "min_names": int(min_names),
    }
    _assert_no_forbidden(out)
    return out


def kyle_lambda_by_date(
    fused: pl.DataFrame,
    *,
    flow: FlowSource = "signed_depth",
    target: str = "delta_mid",
    min_names: int = 3,
    hac_lags: int | None = None,
) -> dict[str, Any]:
    """Per-date Kyle λ, then HAC on the λ series + date-level IC of flow→target.

    ``target`` defaults to ``delta_mid``; use ``fwd_ret_1`` for return-space λ.
    Returns a research_only receipt. SYNTHETIC labeled.
    """
    flow_col = "signed_depth" if flow == "signed_depth" else "ofi"
    if target not in {"delta_mid", "fwd_delta_mid", "fwd_ret_1", "fwd_ret_2", "fwd_ret_3"}:
        raise ValueError(f"unsupported kyle target: {target}")
    if flow_col not in fused.columns or target not in fused.columns:
        raise ValueError(f"fused must contain {flow_col} and {target}")

    sample = fused.drop_nulls([target, flow_col])
    if sample.height == 0:
        raise ValueError(f"no finite {target}/flow rows")

    # Per-date OLS λ across names (cross-section of names that day)
    lambdas: list[float] = []
    kept_dates: list[object] = []
    for date, grp in sample.group_by("event_time", maintain_order=True):
        if grp.height < min_names:
            continue
        lam = kyle_lambda_ols(
            grp[target].to_numpy().astype(float),
            grp[flow_col].to_numpy().astype(float),
        )
        if not np.isfinite(lam):
            continue
        lambdas.append(float(lam))
        kept_dates.append(date[0] if isinstance(date, tuple) else date)

    lam_arr = np.asarray(lambdas, dtype=float)
    if lam_arr.size >= 3:
        mean_lam, t_lam, p_lam = mean_tstat(lam_arr, hac_lags)
    else:
        mean_lam = float(np.mean(lam_arr)) if lam_arr.size else float("nan")
        t_lam, p_lam = float("nan"), float("nan")

    # Date-level IC of contemporaneous flow vs Δmid (discovery score)
    y = sample[target].to_numpy().astype(float)
    x = sample[flow_col].to_numpy().astype(float)
    dates = sample["event_time"].to_numpy()
    ic = date_ic_series(x, y, dates, min_names=min_names, hac_lags=hac_lags)

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
    out: dict[str, Any] = {
        "family": "kyle_ofi",
        "diagnostic": f"kyle_lambda_{flow}_{target}",
        "label": "SYNTHETIC",
        "data_source": "SYNTHETIC" if book_dgp == "synthetic_lob" else book_source,
        "dgp": book_dgp,
        "book_source": book_source,
        "book_dgp": book_dgp,
        "research_only": True,
        "claim": "research_diagnostic_only",
        "ic_method": "date_level_spearman_hac",
        "flow": flow,
        "target": target,
        "n_rows": int(sample.height),
        "n_dates_lambda": int(lam_arr.size),
        "kyle_lambda_mean": float(mean_lam),
        "kyle_lambda_t": float(t_lam),
        "kyle_lambda_p": float(p_lam),
        "kyle_lambda_std": float(np.nanstd(lam_arr, ddof=1)) if lam_arr.size >= 2 else float("nan"),
        "kyle_lambda_iqr": (
            float(np.nanpercentile(lam_arr, 75) - np.nanpercentile(lam_arr, 25))
            if lam_arr.size >= 2
            else float("nan")
        ),
        "kyle_lambda_series": lam_arr.tolist(),
        "kyle_lambda_dates": [str(d) for d in kept_dates],
        "flow_delta_mid_mean_spearman": float(ic.mean_spearman),
        "flow_delta_mid_mean_pearson": float(ic.mean_pearson),
        "flow_delta_mid_t": float(ic.t_spearman),
        "flow_delta_mid_p": float(ic.p_spearman),
        "flow_delta_mid_n_dates": float(ic.n_dates),
        "min_names": int(min_names),
    }
    disp = kyle_lambda_dispersion(lam_arr, window=10, hac_lags=hac_lags)
    for k, v in disp.items():
        if k in {"research_only", "claim", "n_dates", "window"}:
            continue
        out[k] = v
    out["kyle_lambda_dispersion_window"] = int(disp["window"])
    _assert_no_forbidden(out)
    return out


def ofi_delta_mid_date_ic(
    fused: pl.DataFrame,
    *,
    min_names: int = 3,
    hac_lags: int | None = None,
    lag: int = 0,
) -> dict[str, Any]:
    """Date-level IC of OFI (optionally lagged) vs Δmid via ``date_ic_series`` + HAC.

    ``lag=0``: contemporaneous OFI→Δmid. ``lag=1``: prior OFI vs current Δmid
    (predictive discovery score on SYNTHETIC books).
    """
    if "ofi" not in fused.columns or "delta_mid" not in fused.columns:
        raise ValueError("fused must contain ofi and delta_mid")
    if lag < 0:
        raise ValueError("lag must be >= 0")

    frame = fused.sort(["security_id", "event_time"])
    if lag > 0:
        frame = frame.with_columns(pl.col("ofi").shift(lag).over("security_id").alias("ofi_score"))
    else:
        frame = frame.with_columns(pl.col("ofi").alias("ofi_score"))

    sample = frame.drop_nulls(["delta_mid", "ofi_score"])
    y = sample["delta_mid"].to_numpy().astype(float)
    x = sample["ofi_score"].to_numpy().astype(float)
    dates = sample["event_time"].to_numpy()
    ic = date_ic_series(x, y, dates, min_names=min_names, hac_lags=hac_lags)

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
    out: dict[str, Any] = {
        "family": "kyle_ofi",
        "diagnostic": "ofi_to_delta_mid",
        "label": "SYNTHETIC",
        "data_source": "SYNTHETIC" if book_dgp == "synthetic_lob" else book_source,
        "dgp": book_dgp,
        "book_source": book_source,
        "book_dgp": book_dgp,
        "research_only": True,
        "claim": "research_diagnostic_only",
        "ic_method": "date_level_spearman_hac",
        "ofi_lag": int(lag),
        "n_rows": int(sample.height),
        "mean_spearman": float(ic.mean_spearman),
        "mean_pearson": float(ic.mean_pearson),
        "t_spearman": float(ic.t_spearman),
        "p_spearman": float(ic.p_spearman),
        "t_pearson": float(ic.t_pearson),
        "p_pearson": float(ic.p_pearson),
        "n_dates": float(ic.n_dates),
        "min_names": int(min_names),
    }
    _assert_no_forbidden(out)
    return out


def predictive_flow_fwd_date_ic(
    fused: pl.DataFrame,
    *,
    flow: FlowSource = "ofi",
    target: str = "fwd_delta_mid",
    min_names: int = 3,
    hac_lags: int | None = None,
) -> dict[str, Any]:
    """Date-level IC: lag-0 flow vs explicit forward target (``fwd_delta_mid`` / ``fwd_ret_1``).

    Feature is contemporaneous (not lagged). Target is next-bar via ``shift(-1)``
    columns stamped in ``fuse_bars_l2_kyle_frame``. Scores with ``date_ic_series`` + HAC.
    """
    flow_col = "signed_depth" if flow == "signed_depth" else "ofi"
    if target not in {
        "fwd_delta_mid",
        "fwd_ret_1",
        "fwd_ret_2",
        "fwd_ret_3",
        "fwd_close_ret_1",
        "fwd_close_ret_2",
        "fwd_close_ret_3",
    }:
        raise ValueError("target must be a stamped forward column")
    if flow_col not in fused.columns:
        raise ValueError(f"fused must contain {flow_col}")
    if target not in fused.columns:
        raise ValueError(f"fused must contain {target} (run fuse_bars_l2_kyle_frame first)")

    sample = fused.drop_nulls([target, flow_col])
    y = sample[target].to_numpy().astype(float)
    x = sample[flow_col].to_numpy().astype(float)
    dates = sample["event_time"].to_numpy()
    ic = date_ic_series(x, y, dates, min_names=min_names, hac_lags=hac_lags)

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
    out: dict[str, Any] = {
        "family": "kyle_ofi",
        "diagnostic": f"{flow}_to_{target}",
        "label": "SYNTHETIC",
        "data_source": "SYNTHETIC" if book_dgp == "synthetic_lob" else book_source,
        "dgp": book_dgp,
        "book_source": book_source,
        "book_dgp": book_dgp,
        "research_only": True,
        "claim": "research_diagnostic_only",
        "ic_method": "date_level_spearman_hac",
        "flow": flow,
        "target": target,
        "feature_lag": 0,
        "n_rows": int(sample.height),
        "mean_spearman": float(ic.mean_spearman),
        "mean_pearson": float(ic.mean_pearson),
        "t_spearman": float(ic.t_spearman),
        "p_spearman": float(ic.p_spearman),
        "t_pearson": float(ic.t_pearson),
        "p_pearson": float(ic.p_pearson),
        "n_dates": float(ic.n_dates),
        "min_names": int(min_names),
    }
    _assert_no_forbidden(out)
    return out


def flow_delta_mid_date_ic(
    fused: pl.DataFrame,
    *,
    flow: FlowSource = "signed_depth",
    min_names: int = 3,
    hac_lags: int | None = None,
    lag: int = 0,
) -> dict[str, Any]:
    """Date-level IC of flow (signed_depth or ofi), optionally lagged, vs Δmid."""
    flow_col = "signed_depth" if flow == "signed_depth" else "ofi"
    if flow_col not in fused.columns or "delta_mid" not in fused.columns:
        raise ValueError(f"fused must contain {flow_col} and delta_mid")
    if lag < 0:
        raise ValueError("lag must be >= 0")

    frame = fused.sort(["security_id", "event_time"])
    score_col = f"{flow_col}_score"
    if lag > 0:
        frame = frame.with_columns(pl.col(flow_col).shift(lag).over("security_id").alias(score_col))
    else:
        frame = frame.with_columns(pl.col(flow_col).alias(score_col))

    sample = frame.drop_nulls(["delta_mid", score_col])
    y = sample["delta_mid"].to_numpy().astype(float)
    x = sample[score_col].to_numpy().astype(float)
    dates = sample["event_time"].to_numpy()
    ic = date_ic_series(x, y, dates, min_names=min_names, hac_lags=hac_lags)

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
    out: dict[str, Any] = {
        "family": "kyle_ofi",
        "diagnostic": f"{flow}_to_delta_mid",
        "label": "SYNTHETIC",
        "data_source": "SYNTHETIC" if book_dgp == "synthetic_lob" else book_source,
        "dgp": book_dgp,
        "book_source": book_source,
        "book_dgp": book_dgp,
        "research_only": True,
        "claim": "research_diagnostic_only",
        "ic_method": "date_level_spearman_hac",
        "flow": flow,
        "flow_lag": int(lag),
        "n_rows": int(sample.height),
        "mean_spearman": float(ic.mean_spearman),
        "mean_pearson": float(ic.mean_pearson),
        "t_spearman": float(ic.t_spearman),
        "p_spearman": float(ic.p_spearman),
        "t_pearson": float(ic.t_pearson),
        "p_pearson": float(ic.p_pearson),
        "n_dates": float(ic.n_dates),
        "min_names": int(min_names),
    }
    _assert_no_forbidden(out)
    return out


def bench_kyle_ofi_fused(
    bars: pl.DataFrame,
    book: pl.DataFrame | None = None,
    *,
    min_names: int = 3,
    hac_lags: int | None = None,
    label: str = "SYNTHETIC",
    min_join_coverage: float | None = None,
    book_panel_path: str | None = None,
) -> dict[str, Any]:
    """Full research bench: fuse bars+L2, Kyle λ (depth + OFI), OFI→Δmid date IC.

    External/vendor books require ``source`` honesty; stamps ``book_source`` /
    ``book_dgp``. SYNTHETIC / research_only. Scores with date_ic_series + HAC.
    """
    fused = fuse_bars_l2_kyle_frame(bars, book, min_join_coverage=min_join_coverage)
    fused = ensure_forward_targets(fused)
    book_source = str(fused["book_source"][0]) if fused.height else "synthetic_lob"
    book_dgp = str(fused["book_dgp"][0]) if fused.height else "synthetic_lob"
    join_coverage = float(fused["join_coverage"][0]) if fused.height else 0.0

    depth = kyle_lambda_by_date(fused, flow="signed_depth", min_names=min_names, hac_lags=hac_lags)
    ofi_kyle = kyle_lambda_by_date(fused, flow="ofi", min_names=min_names, hac_lags=hac_lags)
    ofi_ic0 = ofi_delta_mid_date_ic(fused, min_names=min_names, hac_lags=hac_lags, lag=0)
    ofi_ic1 = ofi_delta_mid_date_ic(fused, min_names=min_names, hac_lags=hac_lags, lag=1)
    depth_ic1 = flow_delta_mid_date_ic(
        fused, flow="signed_depth", min_names=min_names, hac_lags=hac_lags, lag=1
    )
    ofi_fwd_mid = predictive_flow_fwd_date_ic(
        fused, flow="ofi", target="fwd_delta_mid", min_names=min_names, hac_lags=hac_lags
    )
    ofi_fwd_ret = predictive_flow_fwd_date_ic(
        fused, flow="ofi", target="fwd_ret_1", min_names=min_names, hac_lags=hac_lags
    )
    depth_fwd_mid = predictive_flow_fwd_date_ic(
        fused, flow="signed_depth", target="fwd_delta_mid", min_names=min_names, hac_lags=hac_lags
    )
    depth_fwd_ret = predictive_flow_fwd_date_ic(
        fused, flow="signed_depth", target="fwd_ret_1", min_names=min_names, hac_lags=hac_lags
    )
    depth_kyle_ret = kyle_lambda_by_date(
        fused, flow="signed_depth", target="fwd_ret_1", min_names=min_names, hac_lags=hac_lags
    )
    ofi_kyle_ret = kyle_lambda_by_date(
        fused, flow="ofi", target="fwd_ret_1", min_names=min_names, hac_lags=hac_lags
    )
    ofi_fwd_ret2 = predictive_flow_fwd_date_ic(
        fused, flow="ofi", target="fwd_ret_2", min_names=min_names, hac_lags=hac_lags
    )
    depth_fwd_ret2 = predictive_flow_fwd_date_ic(
        fused, flow="signed_depth", target="fwd_ret_2", min_names=min_names, hac_lags=hac_lags
    )
    depth_kyle_fwd_mid = kyle_lambda_by_date(
        fused, flow="signed_depth", target="fwd_delta_mid", min_names=min_names, hac_lags=hac_lags
    )
    ofi_kyle_fwd_mid = kyle_lambda_by_date(
        fused, flow="ofi", target="fwd_delta_mid", min_names=min_names, hac_lags=hac_lags
    )
    ofi_fwd_ret3 = predictive_flow_fwd_date_ic(
        fused, flow="ofi", target="fwd_ret_3", min_names=min_names, hac_lags=hac_lags
    )
    depth_fwd_ret3 = predictive_flow_fwd_date_ic(
        fused, flow="signed_depth", target="fwd_ret_3", min_names=min_names, hac_lags=hac_lags
    )
    ofi_depth_corr = kyle_lambda_ofi_depth_corr(
        fused, target="delta_mid", min_names=min_names, hac_lags=hac_lags
    )

    residual_ofi = residual_flow_date_ic(
        fused,
        flow="ofi",
        control="signed_depth",
        target="fwd_delta_mid",
        min_names=min_names,
        hac_lags=hac_lags,
    )
    residual_depth = residual_flow_date_ic(
        fused,
        flow="signed_depth",
        control="ofi",
        target="fwd_delta_mid",
        min_names=min_names,
        hac_lags=hac_lags,
    )
    residual_depth_fwd_ret1 = residual_flow_date_ic(
        fused,
        flow="signed_depth",
        control="ofi",
        target="fwd_ret_1",
        min_names=min_names,
        hac_lags=hac_lags,
    )
    residual_ofi_fwd_ret1 = residual_flow_date_ic(
        fused,
        flow="ofi",
        control="signed_depth",
        target="fwd_ret_1",
        min_names=min_names,
        hac_lags=hac_lags,
    )
    residual_depth_fwd_ret2 = residual_flow_date_ic(
        fused,
        flow="signed_depth",
        control="ofi",
        target="fwd_ret_2",
        min_names=min_names,
        hac_lags=hac_lags,
    )
    residual_ofi_fwd_ret2 = residual_flow_date_ic(
        fused,
        flow="ofi",
        control="signed_depth",
        target="fwd_ret_2",
        min_names=min_names,
        hac_lags=hac_lags,
    )
    residual_depth_fwd_ret3 = residual_flow_date_ic(
        fused,
        flow="signed_depth",
        control="ofi",
        target="fwd_ret_3",
        min_names=min_names,
        hac_lags=hac_lags,
    )
    residual_ofi_fwd_ret3 = residual_flow_date_ic(
        fused,
        flow="ofi",
        control="signed_depth",
        target="fwd_ret_3",
        min_names=min_names,
        hac_lags=hac_lags,
    )

    data_source = "SYNTHETIC" if book_dgp == "synthetic_lob" else book_source
    out: dict[str, Any] = {
        "family": "kyle_ofi",
        "label": str(label),
        "data_source": data_source
        if not str(label).upper().startswith("SYN")
        else ("SYNTHETIC" if book_dgp == "synthetic_lob" else book_source),
        "dgp": book_dgp,
        "book_source": book_source,
        "book_dgp": book_dgp,
        "join_coverage": join_coverage,
        "book_panel_path": book_panel_path,
        "research_only": True,
        "claim": "research_diagnostic_only",
        "ic_method": "date_level_spearman_hac",
        "n_fused": int(fused.height),
        "n_scored": int(fused.drop_nulls(["delta_mid"]).height),
        "min_names": int(min_names),
        "kyle_lambda_depth_mean": depth["kyle_lambda_mean"],
        "kyle_lambda_depth_t": depth["kyle_lambda_t"],
        "kyle_lambda_depth_p": depth["kyle_lambda_p"],
        "kyle_lambda_depth_n_dates": depth["n_dates_lambda"],
        "depth_flow_delta_mid_mean_spearman": depth["flow_delta_mid_mean_spearman"],
        "depth_flow_delta_mid_t": depth["flow_delta_mid_t"],
        "depth_flow_delta_mid_p": depth["flow_delta_mid_p"],
        "depth_flow_delta_mid_n_dates": depth["flow_delta_mid_n_dates"],
        "kyle_lambda_ofi_mean": ofi_kyle["kyle_lambda_mean"],
        "kyle_lambda_ofi_t": ofi_kyle["kyle_lambda_t"],
        "kyle_lambda_ofi_p": ofi_kyle["kyle_lambda_p"],
        "kyle_lambda_ofi_n_dates": ofi_kyle["n_dates_lambda"],
        "ofi_flow_delta_mid_mean_spearman": ofi_kyle["flow_delta_mid_mean_spearman"],
        "ofi_flow_delta_mid_t": ofi_kyle["flow_delta_mid_t"],
        "ofi_flow_delta_mid_p": ofi_kyle["flow_delta_mid_p"],
        "ofi_flow_delta_mid_n_dates": ofi_kyle["flow_delta_mid_n_dates"],
        "ofi_delta_mid_lag0_mean_spearman": ofi_ic0["mean_spearman"],
        "ofi_delta_mid_lag0_t": ofi_ic0["t_spearman"],
        "ofi_delta_mid_lag0_p": ofi_ic0["p_spearman"],
        "ofi_delta_mid_lag0_n_dates": ofi_ic0["n_dates"],
        "ofi_delta_mid_lag1_mean_spearman": ofi_ic1["mean_spearman"],
        "ofi_delta_mid_lag1_t": ofi_ic1["t_spearman"],
        "ofi_delta_mid_lag1_p": ofi_ic1["p_spearman"],
        "ofi_delta_mid_lag1_n_dates": ofi_ic1["n_dates"],
        "signed_depth_delta_mid_lag1_mean_spearman": depth_ic1["mean_spearman"],
        "signed_depth_delta_mid_lag1_t": depth_ic1["t_spearman"],
        "signed_depth_delta_mid_lag1_p": depth_ic1["p_spearman"],
        "signed_depth_delta_mid_lag1_n_dates": depth_ic1["n_dates"],
        "kyle_lambda_depth_std": depth["kyle_lambda_std"],
        "kyle_lambda_depth_iqr": depth["kyle_lambda_iqr"],
        "kyle_lambda_ofi_std": ofi_kyle["kyle_lambda_std"],
        "kyle_lambda_ofi_iqr": ofi_kyle["kyle_lambda_iqr"],
        "kyle_lambda_depth_p10": depth.get("kyle_lambda_p10"),
        "kyle_lambda_depth_p50": depth.get("kyle_lambda_p50"),
        "kyle_lambda_depth_p90": depth.get("kyle_lambda_p90"),
        "kyle_lambda_ofi_p10": ofi_kyle.get("kyle_lambda_p10"),
        "kyle_lambda_ofi_p50": ofi_kyle.get("kyle_lambda_p50"),
        "kyle_lambda_ofi_p90": ofi_kyle.get("kyle_lambda_p90"),
        "kyle_lambda_depth_rolling_mean": depth.get("kyle_lambda_rolling_mean"),
        "kyle_lambda_depth_rolling_hac_t": depth.get("kyle_lambda_rolling_hac_t"),
        "kyle_lambda_depth_rolling_hac_lo": depth.get("kyle_lambda_rolling_hac_lo"),
        "kyle_lambda_depth_rolling_hac_hi": depth.get("kyle_lambda_rolling_hac_hi"),
        "kyle_lambda_ofi_rolling_mean": ofi_kyle.get("kyle_lambda_rolling_mean"),
        "kyle_lambda_ofi_rolling_hac_t": ofi_kyle.get("kyle_lambda_rolling_hac_t"),
        "kyle_lambda_ofi_rolling_hac_lo": ofi_kyle.get("kyle_lambda_rolling_hac_lo"),
        "kyle_lambda_ofi_rolling_hac_hi": ofi_kyle.get("kyle_lambda_rolling_hac_hi"),
        "kyle_lambda_dispersion_window": depth.get("kyle_lambda_dispersion_window", 10),
        "kyle_lambda_ofi_depth_spearman": ofi_depth_corr["kyle_lambda_ofi_depth_spearman"],
        "kyle_lambda_ofi_depth_pearson": ofi_depth_corr["kyle_lambda_ofi_depth_pearson"],
        "kyle_lambda_ofi_depth_n_dates": ofi_depth_corr["n_dates_aligned"],
        "kyle_lambda_ofi_depth_prod_hac_t": ofi_depth_corr["kyle_lambda_ofi_depth_prod_hac_t"],
        "kyle_lambda_ofi_depth_prod_hac_p": ofi_depth_corr["kyle_lambda_ofi_depth_prod_hac_p"],
        "residual_ofi_ex_depth_fwd_delta_mid_mean_spearman": residual_ofi["mean_spearman"],
        "residual_ofi_ex_depth_fwd_delta_mid_t": residual_ofi["t_spearman"],
        "residual_ofi_ex_depth_fwd_delta_mid_n_dates": residual_ofi["n_dates"],
        "residual_depth_ex_ofi_fwd_delta_mid_mean_spearman": residual_depth["mean_spearman"],
        "residual_depth_ex_ofi_fwd_delta_mid_t": residual_depth["t_spearman"],
        "residual_depth_ex_ofi_fwd_delta_mid_n_dates": residual_depth["n_dates"],
        "residual_depth_ex_ofi_fwd_ret_1_mean_spearman": residual_depth_fwd_ret1["mean_spearman"],
        "residual_depth_ex_ofi_fwd_ret_1_t": residual_depth_fwd_ret1["t_spearman"],
        "residual_depth_ex_ofi_fwd_ret_1_p": residual_depth_fwd_ret1["p_spearman"],
        "residual_depth_ex_ofi_fwd_ret_1_n_dates": residual_depth_fwd_ret1["n_dates"],
        "residual_ofi_ex_depth_fwd_ret_1_mean_spearman": residual_ofi_fwd_ret1["mean_spearman"],
        "residual_ofi_ex_depth_fwd_ret_1_t": residual_ofi_fwd_ret1["t_spearman"],
        "residual_ofi_ex_depth_fwd_ret_1_n_dates": residual_ofi_fwd_ret1["n_dates"],
        "residual_depth_ex_ofi_fwd_ret_2_mean_spearman": residual_depth_fwd_ret2["mean_spearman"],
        "residual_depth_ex_ofi_fwd_ret_2_t": residual_depth_fwd_ret2["t_spearman"],
        "residual_depth_ex_ofi_fwd_ret_2_n_dates": residual_depth_fwd_ret2["n_dates"],
        "residual_ofi_ex_depth_fwd_ret_2_mean_spearman": residual_ofi_fwd_ret2["mean_spearman"],
        "residual_ofi_ex_depth_fwd_ret_2_t": residual_ofi_fwd_ret2["t_spearman"],
        "residual_ofi_ex_depth_fwd_ret_2_n_dates": residual_ofi_fwd_ret2["n_dates"],
        "residual_depth_ex_ofi_fwd_ret_3_mean_spearman": residual_depth_fwd_ret3["mean_spearman"],
        "residual_depth_ex_ofi_fwd_ret_3_t": residual_depth_fwd_ret3["t_spearman"],
        "residual_depth_ex_ofi_fwd_ret_3_n_dates": residual_depth_fwd_ret3["n_dates"],
        "residual_ofi_ex_depth_fwd_ret_3_mean_spearman": residual_ofi_fwd_ret3["mean_spearman"],
        "residual_ofi_ex_depth_fwd_ret_3_t": residual_ofi_fwd_ret3["t_spearman"],
        "residual_ofi_ex_depth_fwd_ret_3_n_dates": residual_ofi_fwd_ret3["n_dates"],
        "kyle_lambda_date_series_n_depth": int(len(depth.get("kyle_lambda_series") or [])),
        "kyle_lambda_date_series_n_ofi": int(len(ofi_kyle.get("kyle_lambda_series") or [])),
        "ofi_fwd_delta_mid_mean_spearman": ofi_fwd_mid["mean_spearman"],
        "ofi_fwd_delta_mid_t": ofi_fwd_mid["t_spearman"],
        "ofi_fwd_delta_mid_p": ofi_fwd_mid["p_spearman"],
        "ofi_fwd_delta_mid_n_dates": ofi_fwd_mid["n_dates"],
        "ofi_fwd_ret_1_mean_spearman": ofi_fwd_ret["mean_spearman"],
        "ofi_fwd_ret_1_t": ofi_fwd_ret["t_spearman"],
        "ofi_fwd_ret_1_p": ofi_fwd_ret["p_spearman"],
        "ofi_fwd_ret_1_n_dates": ofi_fwd_ret["n_dates"],
        "signed_depth_fwd_delta_mid_mean_spearman": depth_fwd_mid["mean_spearman"],
        "signed_depth_fwd_delta_mid_t": depth_fwd_mid["t_spearman"],
        "signed_depth_fwd_delta_mid_p": depth_fwd_mid["p_spearman"],
        "signed_depth_fwd_delta_mid_n_dates": depth_fwd_mid["n_dates"],
        "signed_depth_fwd_ret_1_mean_spearman": depth_fwd_ret["mean_spearman"],
        "signed_depth_fwd_ret_1_t": depth_fwd_ret["t_spearman"],
        "signed_depth_fwd_ret_1_p": depth_fwd_ret["p_spearman"],
        "signed_depth_fwd_ret_1_n_dates": depth_fwd_ret["n_dates"],
        "kyle_lambda_depth_fwd_ret_1_mean": depth_kyle_ret["kyle_lambda_mean"],
        "kyle_lambda_depth_fwd_ret_1_t": depth_kyle_ret["kyle_lambda_t"],
        "kyle_lambda_depth_fwd_ret_1_std": depth_kyle_ret["kyle_lambda_std"],
        "kyle_lambda_ofi_fwd_ret_1_mean": ofi_kyle_ret["kyle_lambda_mean"],
        "kyle_lambda_ofi_fwd_ret_1_t": ofi_kyle_ret["kyle_lambda_t"],
        "kyle_lambda_ofi_fwd_ret_1_std": ofi_kyle_ret["kyle_lambda_std"],
        "ofi_fwd_ret_2_mean_spearman": ofi_fwd_ret2["mean_spearman"],
        "ofi_fwd_ret_2_t": ofi_fwd_ret2["t_spearman"],
        "ofi_fwd_ret_2_n_dates": ofi_fwd_ret2["n_dates"],
        "signed_depth_fwd_ret_2_mean_spearman": depth_fwd_ret2["mean_spearman"],
        "signed_depth_fwd_ret_2_t": depth_fwd_ret2["t_spearman"],
        "signed_depth_fwd_ret_2_n_dates": depth_fwd_ret2["n_dates"],
        "kyle_lambda_depth_fwd_delta_mid_mean": depth_kyle_fwd_mid["kyle_lambda_mean"],
        "kyle_lambda_depth_fwd_delta_mid_t": depth_kyle_fwd_mid["kyle_lambda_t"],
        "kyle_lambda_depth_fwd_delta_mid_std": depth_kyle_fwd_mid["kyle_lambda_std"],
        "kyle_lambda_ofi_fwd_delta_mid_mean": ofi_kyle_fwd_mid["kyle_lambda_mean"],
        "kyle_lambda_ofi_fwd_delta_mid_t": ofi_kyle_fwd_mid["kyle_lambda_t"],
        "kyle_lambda_ofi_fwd_delta_mid_std": ofi_kyle_fwd_mid["kyle_lambda_std"],
        "ofi_fwd_ret_3_mean_spearman": ofi_fwd_ret3["mean_spearman"],
        "ofi_fwd_ret_3_t": ofi_fwd_ret3["t_spearman"],
        "ofi_fwd_ret_3_n_dates": ofi_fwd_ret3["n_dates"],
        "signed_depth_fwd_ret_3_mean_spearman": depth_fwd_ret3["mean_spearman"],
        "signed_depth_fwd_ret_3_t": depth_fwd_ret3["t_spearman"],
        "signed_depth_fwd_ret_3_n_dates": depth_fwd_ret3["n_dates"],
    }
    _assert_no_forbidden(out)
    return out


def _assert_no_forbidden(blob: dict[str, Any]) -> None:
    forbidden = ("sharpe", "sortino", "calmar", "pnl", "nav", "live_pnl_claim")
    leaked = [k for k in blob if any(tok in k.lower() for tok in forbidden)]
    if leaked:
        raise AssertionError(f"kyle_ofi receipt leaked forbidden research keys: {leaked}")
