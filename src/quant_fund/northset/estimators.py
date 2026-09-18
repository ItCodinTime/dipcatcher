"""Northset microstructure estimators.

Classic closed-form diagnostics (Kyle, Roll, Parkinson, Garman–Klass,
Rogers–Satchell, Yang–Zhang, Corwin–Schultz, Amihud, OFI). Honest NaN when
undefined. SYNTHETIC LOB is labeled plumbing, not live edge.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_fund.metrics.inference import diebold_mariano
from quant_fund.metrics.scoring import qlike

Array = NDArray[np.float64]
_LN2 = math.log(2.0)
_GK_OC = 2.0 * _LN2 - 1.0
_CS_DEN = 3.0 - 2.0 * math.sqrt(2.0)
_FLOOR = 1e-12


def kyle_lambda(delta_mid: Array, signed_volume: Array) -> tuple[float, float]:
    """OLS Kyle λ: Δmid = λ · signed_volume + ε. Returns (λ, R²)."""
    y = np.asarray(delta_mid, dtype=float).reshape(-1)
    q = np.asarray(signed_volume, dtype=float).reshape(-1)
    if y.size != q.size:
        raise ValueError("delta_mid and signed_volume must align")
    mask = np.isfinite(y) & np.isfinite(q)
    if int(mask.sum()) < 20:
        return float("nan"), float("nan")
    y = y[mask]
    q = q[mask]
    q_var = float(np.var(q))
    if q_var <= 1e-18:
        return float("nan"), float("nan")
    lam = float(np.cov(y, q, ddof=0)[0, 1] / q_var)
    intercept = float(np.mean(y) - lam * np.mean(q))
    fitted = intercept + lam * q
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = float("nan") if ss_tot <= 1e-18 else float(1.0 - ss_res / ss_tot)
    return lam, r2


def roll_spread(mid: Array) -> float:
    """Roll (1984) implied spread 2√(-γ₁) from mid changes. Positive γ₁ → NaN."""
    px = np.asarray(mid, dtype=float).reshape(-1)
    px = px[np.isfinite(px)]
    if px.size < 12:
        return float("nan")
    dp = np.diff(px)
    a = dp[1:]
    b = dp[:-1]
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 10:
        return float("nan")
    gamma = float(np.cov(a[mask], b[mask], ddof=0)[0, 1])
    if not math.isfinite(gamma) or gamma >= 0.0:
        return float("nan")
    return float(2.0 * math.sqrt(-gamma))


def _qlike_elements(realized: Array, forecast: Array, floor: float = _FLOOR) -> Array:
    y = np.clip(np.asarray(realized, dtype=float).reshape(-1), floor, None)
    yhat = np.clip(np.asarray(forecast, dtype=float).reshape(-1), floor, None)
    if y.size != yhat.size:
        raise ValueError("realized and forecast must align")
    ratio = y / yhat
    return ratio - np.log(ratio) - 1.0


def ohlc_variance_frame(bars: pl.DataFrame) -> pl.DataFrame:
    """Per-bar close-to-close, Parkinson, Garman–Klass, and Rogers–Satchell variances."""
    required = ("security_id", "event_time", "open", "high", "low", "close")
    missing = [c for c in required if c not in bars.columns]
    if missing:
        raise ValueError(f"bars missing columns: {missing}")
    frame = bars.sort(["security_id", "event_time"]).with_columns(
        pl.col("close").shift(1).over("security_id").alias("prev_close")
    )
    hi = frame["high"].to_numpy().astype(float)
    lo = frame["low"].to_numpy().astype(float)
    opn = frame["open"].to_numpy().astype(float)
    close = frame["close"].to_numpy().astype(float)
    prev = frame["prev_close"].to_numpy().astype(float)
    ok = (
        np.isfinite(hi)
        & np.isfinite(lo)
        & np.isfinite(opn)
        & np.isfinite(close)
        & (hi > 0.0)
        & (lo > 0.0)
        & (opn > 0.0)
        & (close > 0.0)
        & (hi + 1e-12 >= lo)
    )
    hl = np.full(hi.shape, np.nan)
    oc = np.full(hi.shape, np.nan)
    cc = np.full(hi.shape, np.nan)
    overnight = np.full(hi.shape, np.nan)
    rs = np.full(hi.shape, np.nan)
    hl[ok] = np.log(hi[ok] / lo[ok])
    oc[ok] = np.log(close[ok] / opn[ok])
    prev_ok = ok & np.isfinite(prev) & (prev > 0.0)
    cc[prev_ok] = np.log(close[prev_ok] / prev[prev_ok]) ** 2
    overnight[prev_ok] = np.log(opn[prev_ok] / prev[prev_ok])
    rs[ok] = np.log(hi[ok] / close[ok]) * np.log(hi[ok] / opn[ok]) + np.log(
        lo[ok] / close[ok]
    ) * np.log(lo[ok] / opn[ok])
    park = (hl**2) / (4.0 * _LN2)
    gk = 0.5 * (hl**2) - _GK_OC * (oc**2)
    return frame.with_columns(
        pl.Series("var_cc", cc),
        pl.Series("var_park", park),
        pl.Series("var_gk", gk),
        pl.Series("var_rs", rs),
        pl.Series("overnight_log", overnight),
        pl.Series("open_close_log", oc),
    )


def parkinson_vs_close_to_close(bars: pl.DataFrame) -> float:
    """QLIKE of close-to-close log-return squares vs Parkinson range variance."""
    frame = ohlc_variance_frame(bars)
    y = frame["var_cc"].to_numpy().astype(float)
    yhat = frame["var_park"].to_numpy().astype(float)
    mask = np.isfinite(y) & np.isfinite(yhat) & (yhat > 0.0)
    if int(mask.sum()) < 8:
        return float("nan")
    return qlike(y[mask], yhat[mask])


def garman_klass_vs_close_to_close(bars: pl.DataFrame) -> float:
    """QLIKE of close-to-close RV vs Garman–Klass (1980). Negative GK clipped by QLIKE floor."""
    frame = ohlc_variance_frame(bars)
    y = frame["var_cc"].to_numpy().astype(float)
    yhat = frame["var_gk"].to_numpy().astype(float)
    mask = np.isfinite(y) & np.isfinite(yhat)
    if int(mask.sum()) < 8:
        return float("nan")
    return qlike(y[mask], yhat[mask])


def rogers_satchell_vs_close_to_close(bars: pl.DataFrame) -> float:
    """QLIKE of close-to-close RV vs Rogers–Satchell (1991)."""
    frame = ohlc_variance_frame(bars)
    y = frame["var_cc"].to_numpy().astype(float)
    yhat = frame["var_rs"].to_numpy().astype(float)
    mask = np.isfinite(y) & np.isfinite(yhat)
    if int(mask.sum()) < 8:
        return float("nan")
    return qlike(y[mask], yhat[mask])


def yang_zhang_variance(bars: pl.DataFrame) -> float:
    """Yang–Zhang (2000) OHLC variance (sample statistic, not per-bar)."""
    frame = ohlc_variance_frame(bars)
    o = frame["overnight_log"].to_numpy().astype(float)
    c = frame["open_close_log"].to_numpy().astype(float)
    rs = frame["var_rs"].to_numpy().astype(float)
    mask = np.isfinite(o) & np.isfinite(c) & np.isfinite(rs)
    n = int(mask.sum())
    if n < 8:
        return float("nan")
    k = 0.34 / (1.34 + (n + 1.0) / (n - 1.0))
    return float(np.var(o[mask]) + k * np.var(c[mask]) + (1.0 - k) * float(np.mean(rs[mask])))


def yang_zhang_vs_close_to_close(bars: pl.DataFrame) -> float:
    """QLIKE of close-to-close RV vs per-security expanding Yang–Zhang (PIT).

    Forecast at t uses only bars strictly before t on that name. This is not the
    in-sample pooled constant previously used as a diagnostic. Running moments
    are O(n) per name (population variance, matching ``np.var`` ddof=0).
    """
    frame = ohlc_variance_frame(bars).sort(["security_id", "event_time"])
    losses: list[float] = []
    for _sid, grp in frame.group_by("security_id", maintain_order=True):
        o = grp["overnight_log"].to_numpy().astype(float)
        c = grp["open_close_log"].to_numpy().astype(float)
        rs = grp["var_rs"].to_numpy().astype(float)
        y = grp["var_cc"].to_numpy().astype(float)
        count = 0
        sum_o = sum_o2 = sum_c = sum_c2 = sum_rs = 0.0
        for t in range(int(o.size)):
            if count >= 8 and np.isfinite(y[t]) and y[t] > 0.0:
                mean_o = sum_o / count
                mean_c = sum_c / count
                vo = max(0.0, sum_o2 / count - mean_o * mean_o)
                vc = max(0.0, sum_c2 / count - mean_c * mean_c)
                mrs = sum_rs / count
                k = 0.34 / (1.34 + (count + 1.0) / (count - 1.0))
                yhat = vo + k * vc + (1.0 - k) * mrs
                if math.isfinite(yhat) and yhat > 0.0:
                    losses.append(float(qlike(np.asarray([y[t]]), np.asarray([yhat]))))
            if np.isfinite(o[t]) and np.isfinite(c[t]) and np.isfinite(rs[t]):
                count += 1
                sum_o += float(o[t])
                sum_o2 += float(o[t]) * float(o[t])
                sum_c += float(c[t])
                sum_c2 += float(c[t]) * float(c[t])
                sum_rs += float(rs[t])
    if len(losses) < 8:
        return float("nan")
    return float(np.mean(losses))


def dm_range_vs_park(bars: pl.DataFrame, which: str = "gk") -> dict[str, float | str]:
    """Diebold–Mariano on date-level QLIKE: Garman–Klass or Rogers–Satchell vs Parkinson."""
    frame = ohlc_variance_frame(bars)
    col = "var_gk" if which == "gk" else "var_rs"
    sub = frame.select(["security_id", "event_time", "var_cc", "var_park", col]).drop_nulls()
    if sub.height < 16:
        return {"statistic": float("nan"), "p_value": float("nan"), "preferred": "inconclusive"}
    loss_park = _qlike_elements(sub["var_cc"].to_numpy(), sub["var_park"].to_numpy())
    loss_alt = _qlike_elements(sub["var_cc"].to_numpy(), sub[col].to_numpy())
    dated = pl.DataFrame(
        {
            "event_time": sub["event_time"],
            "loss_park": loss_park,
            "loss_alt": loss_alt,
        }
    )
    daily = dated.group_by("event_time", maintain_order=True).agg(
        pl.col("loss_park").mean(),
        pl.col("loss_alt").mean(),
    )
    a = daily["loss_alt"].to_numpy().astype(float)
    b = daily["loss_park"].to_numpy().astype(float)
    finite = np.isfinite(a) & np.isfinite(b)
    if int(finite.sum()) < 8:
        return {"statistic": float("nan"), "p_value": float("nan"), "preferred": "inconclusive"}
    dm = diebold_mariano(a[finite], b[finite], name_a=which, name_b="park")
    return {
        "statistic": float(dm.statistic),
        "p_value": float(dm.p_value),
        "preferred": str(dm.preferred),
    }


def corwin_schultz_spread(bars: pl.DataFrame) -> float:
    """Corwin–Schultz (2012) high-low spread (mean of PIT two-day pairs).

    Each pair is ``(t-1, t)`` so the estimate is known at close t. Using
    ``(t, t+1)`` would peek at tomorrow's range.
    """
    required = ("security_id", "event_time", "high", "low")
    missing = [c for c in required if c not in bars.columns]
    if missing:
        raise ValueError(f"bars missing columns: {missing}")
    frame = bars.sort(["security_id", "event_time"]).with_columns(
        pl.col("high").shift(1).over("security_id").alias("high_prev"),
        pl.col("low").shift(1).over("security_id").alias("low_prev"),
    )
    h = frame["high"].to_numpy().astype(float)
    lo = frame["low"].to_numpy().astype(float)
    hn = frame["high_prev"].to_numpy().astype(float)
    ln = frame["low_prev"].to_numpy().astype(float)
    ok = (
        np.isfinite(h)
        & np.isfinite(lo)
        & np.isfinite(hn)
        & np.isfinite(ln)
        & (h > 0.0)
        & (lo > 0.0)
        & (hn > 0.0)
        & (ln > 0.0)
        & (h >= lo)
        & (hn >= ln)
    )
    if int(ok.sum()) < 8:
        return float("nan")
    beta = (np.log(h[ok] / lo[ok]) ** 2) + (np.log(hn[ok] / ln[ok]) ** 2)
    h2 = np.maximum(h[ok], hn[ok])
    l2 = np.minimum(lo[ok], ln[ok])
    gamma = np.log(h2 / l2) ** 2
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / _CS_DEN - np.sqrt(
        np.clip(gamma, 0.0, None) / _CS_DEN
    )
    exp_a = np.exp(alpha)
    spread = 2.0 * (exp_a - 1.0) / (1.0 + exp_a)
    finite = np.isfinite(spread) & (spread > 0.0) & (spread < 1.0)
    if int(finite.sum()) < 4:
        return float("nan")
    return float(np.mean(spread[finite]))


def amihud_illiquidity(bars: pl.DataFrame) -> pl.DataFrame:
    """Amihud |r| / dollar volume. PIT at close; uses lagged close for the return."""
    required = ("security_id", "event_time", "close", "volume")
    missing = [c for c in required if c not in bars.columns]
    if missing:
        raise ValueError(f"bars missing columns: {missing}")
    return (
        bars.sort(["security_id", "event_time"])
        .with_columns(pl.col("close").shift(1).over("security_id").alias("prev_close"))
        .with_columns(
            (
                (pl.col("close") / pl.col("prev_close") - 1.0).abs()
                / (pl.col("close") * pl.col("volume")).clip(lower_bound=_FLOOR)
            ).alias("amihud")
        )
    )


def order_flow_imbalance(book: pl.DataFrame) -> pl.DataFrame:
    """Cont–Kukanov–Stoikov top-of-book OFI between consecutive snapshots (per name)."""
    required = ("security_id", "event_time", "best_bid", "best_ask", "top_bid_size", "top_ask_size")
    missing = [c for c in required if c not in book.columns]
    if missing:
        raise ValueError(f"book missing columns: {missing}")
    frame = book.sort(["security_id", "event_time"]).with_columns(
        pl.col("best_bid").shift(1).over("security_id").alias("prev_bid"),
        pl.col("best_ask").shift(1).over("security_id").alias("prev_ask"),
        pl.col("top_bid_size").shift(1).over("security_id").alias("prev_top_bid"),
        pl.col("top_ask_size").shift(1).over("security_id").alias("prev_top_ask"),
    )
    bid_up = pl.col("best_bid") >= pl.col("prev_bid")
    bid_dn = pl.col("best_bid") <= pl.col("prev_bid")
    ask_dn = pl.col("best_ask") <= pl.col("prev_ask")
    ask_up = pl.col("best_ask") >= pl.col("prev_ask")
    ofi = (
        pl.when(bid_up).then(pl.col("top_bid_size")).otherwise(0.0)
        - pl.when(bid_dn).then(pl.col("prev_top_bid")).otherwise(0.0)
        - pl.when(ask_dn).then(pl.col("top_ask_size")).otherwise(0.0)
        + pl.when(ask_up).then(pl.col("prev_top_ask")).otherwise(0.0)
    )
    return frame.with_columns(
        pl.when(pl.col("prev_bid").is_null() | pl.col("prev_ask").is_null())
        .then(None)
        .otherwise(ofi)
        .alias("ofi")
    )


def session_realized_variance(session: pl.DataFrame) -> pl.DataFrame:
    """Sum of session log-return squares per daily parent bar."""
    required = ("security_id", "parent_event_time", "session_index", "open", "close")
    missing = [c for c in required if c not in session.columns]
    if missing:
        raise ValueError(f"session missing columns: {missing}")
    frame = session.sort(["security_id", "parent_event_time", "session_index"]).with_columns(
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
    return frame.group_by(["security_id", "parent_event_time"]).agg(
        (pl.col("sess_log_ret") ** 2).sum().alias("session_rv")
    )


def overnight_plus_oc_vs_close_to_close(bars: pl.DataFrame) -> float:
    """QLIKE of close-to-close RV vs overnight² + open-to-close²."""
    frame = ohlc_variance_frame(bars)
    y = frame["var_cc"].to_numpy().astype(float)
    overnight = frame["overnight_log"].to_numpy().astype(float)
    oc = frame["open_close_log"].to_numpy().astype(float)
    yhat = overnight**2 + oc**2
    mask = np.isfinite(y) & np.isfinite(yhat)
    if int(mask.sum()) < 8:
        return float("nan")
    return qlike(y[mask], yhat[mask])


def dm_split_vs_park(bars: pl.DataFrame) -> dict[str, float | str]:
    """Diebold–Mariano: overnight+OC split vs Parkinson on date-level QLIKE."""
    frame = ohlc_variance_frame(bars)
    overnight = frame["overnight_log"].to_numpy().astype(float)
    oc = frame["open_close_log"].to_numpy().astype(float)
    split = overnight**2 + oc**2
    dated = pl.DataFrame(
        {
            "event_time": frame["event_time"],
            "var_cc": frame["var_cc"],
            "var_park": frame["var_park"],
            "var_split": split,
        }
    ).drop_nulls()
    if dated.height < 16:
        return {"statistic": float("nan"), "p_value": float("nan"), "preferred": "inconclusive"}
    loss_park = _qlike_elements(dated["var_cc"].to_numpy(), dated["var_park"].to_numpy())
    loss_split = _qlike_elements(dated["var_cc"].to_numpy(), dated["var_split"].to_numpy())
    daily = (
        pl.DataFrame(
            {
                "event_time": dated["event_time"],
                "loss_park": loss_park,
                "loss_split": loss_split,
            }
        )
        .group_by("event_time", maintain_order=True)
        .agg(
            pl.col("loss_park").mean(),
            pl.col("loss_split").mean(),
        )
    )
    a = daily["loss_split"].to_numpy().astype(float)
    b = daily["loss_park"].to_numpy().astype(float)
    finite = np.isfinite(a) & np.isfinite(b)
    if int(finite.sum()) < 8:
        return {"statistic": float("nan"), "p_value": float("nan"), "preferred": "inconclusive"}
    dm = diebold_mariano(a[finite], b[finite], name_a="split", name_b="park")
    return {
        "statistic": float(dm.statistic),
        "p_value": float(dm.p_value),
        "preferred": str(dm.preferred),
    }


def overnight_share(bars: pl.DataFrame) -> float:
    """Mean overnight² / mean close-to-close RV. Honest NaN if undefined."""
    frame = ohlc_variance_frame(bars)
    on = frame["overnight_log"].to_numpy().astype(float) ** 2
    cc = frame["var_cc"].to_numpy().astype(float)
    mask = np.isfinite(on) & np.isfinite(cc)
    if int(mask.sum()) < 8:
        return float("nan")
    denom = float(np.mean(cc[mask]))
    if denom <= _FLOOR:
        return float("nan")
    return float(np.mean(on[mask]) / denom)


def realized_semivariance(bars: pl.DataFrame) -> tuple[float, float]:
    """Close-to-close upside / downside realized semivariance."""
    frame = bars.sort(["security_id", "event_time"]).with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("security_id")).log().alias("r")
    )
    r = frame["r"].to_numpy().astype(float)
    r = r[np.isfinite(r)]
    if r.size < 8:
        return float("nan"), float("nan")
    up = float(np.mean(np.square(np.clip(r, 0.0, None))))
    down = float(np.mean(np.square(np.clip(-r, 0.0, None))))
    return up, down


def abdi_ranaldo_spread(bars: pl.DataFrame) -> float:
    """Abdi–Ranaldo (2017) close-high-low relative spread (mean of two-day pairs)."""
    required = ("security_id", "event_time", "high", "low", "close")
    missing = [c for c in required if c not in bars.columns]
    if missing:
        raise ValueError(f"bars missing columns: {missing}")
    frame = bars.sort(["security_id", "event_time"]).with_columns(
        ((pl.col("high") + pl.col("low")) / 2.0).alias("hl_mid")
    )
    frame = frame.with_columns(
        pl.col("close").shift(1).over("security_id").alias("prev_close"),
        pl.col("hl_mid").shift(1).over("security_id").alias("prev_hl_mid"),
    )
    c = frame["close"].to_numpy().astype(float)
    m = frame["hl_mid"].to_numpy().astype(float)
    pc = frame["prev_close"].to_numpy().astype(float)
    pm = frame["prev_hl_mid"].to_numpy().astype(float)
    ok = np.isfinite(c) & np.isfinite(m) & np.isfinite(pc) & np.isfinite(pm) & (c > 0.0)
    if int(ok.sum()) < 8:
        return float("nan")
    eta = (c[ok] - m[ok]) * (pc[ok] - pm[ok])
    spread = 2.0 * np.sqrt(np.clip(eta, 0.0, None))
    rel = spread / c[ok]
    finite = np.isfinite(rel) & (rel >= 0.0) & (rel < 1.0)
    if int(finite.sum()) < 4:
        return float("nan")
    return float(np.mean(rel[finite]))


def volume_over_range(bars: pl.DataFrame) -> pl.DataFrame:
    """Volume / (high − low). High values are a liquidity proxy, not a return claim."""
    required = ("security_id", "event_time", "high", "low", "volume")
    missing = [c for c in required if c not in bars.columns]
    if missing:
        raise ValueError(f"bars missing columns: {missing}")
    return bars.with_columns(
        (pl.col("volume") / (pl.col("high") - pl.col("low")).clip(lower_bound=_FLOOR)).alias(
            "volume_over_range"
        )
    )


def true_range_frame(bars: pl.DataFrame) -> pl.DataFrame:
    """Wilder true range vs previous close."""
    required = ("security_id", "event_time", "high", "low", "close")
    missing = [c for c in required if c not in bars.columns]
    if missing:
        raise ValueError(f"bars missing columns: {missing}")
    frame = bars.sort(["security_id", "event_time"]).with_columns(
        pl.col("close").shift(1).over("security_id").alias("prev_close")
    )
    return frame.with_columns(
        pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("prev_close")).abs(),
            (pl.col("low") - pl.col("prev_close")).abs(),
        ).alias("true_range")
    )


def lag1_corr(values: Array) -> float:
    """Lag-1 Pearson correlation. Short / non-finite → NaN."""
    x = np.asarray(values, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 12:
        return float("nan")
    a = x[1:]
    b = x[:-1]
    if float(np.std(a)) <= 1e-18 or float(np.std(b)) <= 1e-18:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def session_bipower_jump(session: pl.DataFrame) -> dict[str, float]:
    """Barndorff-Nielsen–Shephard jump share from session log returns."""
    required = ("security_id", "parent_event_time", "session_index", "open", "close")
    missing = [c for c in required if c not in session.columns]
    if missing:
        raise ValueError(f"session missing columns: {missing}")
    if session.height == 0:
        return {"mean_jump_ratio": float("nan"), "mean_rv": float("nan"), "mean_bv": float("nan")}
    frame = session.sort(["security_id", "parent_event_time", "session_index"]).with_columns(
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
    frame = frame.with_columns(
        pl.col("sess_log_ret").shift(1).over(["security_id", "parent_event_time"]).alias("prev_r")
    )
    grouped = frame.group_by(["security_id", "parent_event_time"]).agg(
        (pl.col("sess_log_ret") ** 2).sum().alias("rv"),
        ((math.pi / 2.0) * (pl.col("sess_log_ret").abs() * pl.col("prev_r").abs()).sum()).alias(
            "bv"
        ),
    )
    rv = grouped["rv"].to_numpy().astype(float)
    bv = grouped["bv"].to_numpy().astype(float)
    mask = np.isfinite(rv) & np.isfinite(bv) & (rv > _FLOOR)
    if int(mask.sum()) < 4:
        return {"mean_jump_ratio": float("nan"), "mean_rv": float("nan"), "mean_bv": float("nan")}
    jump = np.clip(1.0 - bv[mask] / rv[mask], 0.0, 1.0)
    return {
        "mean_jump_ratio": float(np.mean(jump)),
        "mean_rv": float(np.mean(rv[mask])),
        "mean_bv": float(np.mean(bv[mask])),
    }


def session_vpin(session: pl.DataFrame) -> float:
    """Bulk-volume VPIN proxy: |signed session volume| / total volume per parent day."""
    required = ("security_id", "parent_event_time", "open", "close", "volume")
    missing = [c for c in required if c not in session.columns]
    if missing or session.height == 0:
        return float("nan")
    signed = session.with_columns(
        (
            pl.col("volume")
            * pl.when(pl.col("close") > pl.col("open"))
            .then(1.0)
            .when(pl.col("close") < pl.col("open"))
            .then(-1.0)
            .otherwise(0.0)
        ).alias("signed_volume")
    )
    daily = signed.group_by(["security_id", "parent_event_time"]).agg(
        pl.col("signed_volume").sum().alias("net"),
        pl.col("volume").sum().alias("tot"),
    )
    tot = daily["tot"].to_numpy().astype(float)
    net = daily["net"].to_numpy().astype(float)
    mask = np.isfinite(tot) & np.isfinite(net) & (tot > _FLOOR)
    if int(mask.sum()) < 4:
        return float("nan")
    return float(np.mean(np.abs(net[mask]) / tot[mask]))


def queue_imbalance(book: pl.DataFrame) -> pl.DataFrame:
    """Top-of-book queue imbalance (bid−ask)/(bid+ask). Alias of imbalance_top when present.

    Recomputes from sizes so vendor panels without ``imbalance_top`` still work.
    """
    required = ("security_id", "event_time", "top_bid_size", "top_ask_size")
    missing = [c for c in required if c not in book.columns]
    if missing:
        raise ValueError(f"book missing columns: {missing}")
    denom = pl.col("top_bid_size") + pl.col("top_ask_size")
    qi = (
        pl.when(denom > 0.0)
        .then((pl.col("top_bid_size") - pl.col("top_ask_size")) / denom)
        # Degenerate/locked or null tops are undefined, not perfectly balanced.
        .otherwise(None)
        .alias("queue_imbalance")
    )
    return book.with_columns(qi)


def _volume_clock_vpin(
    buy: Array,
    sell: Array,
    *,
    bucket_volume: float,
    window: int,
) -> Array:
    """Assign volume-clock VPIN; incomplete buckets carry the last known value."""
    n = int(buy.size)
    out = np.full(n, np.nan, dtype=float)
    bucket_tox: list[float] = []
    acc_buy = 0.0
    acc_sell = 0.0
    last = float("nan")
    for i in range(n):
        b = float(buy[i]) if np.isfinite(buy[i]) else 0.0
        s = float(sell[i]) if np.isfinite(sell[i]) else 0.0
        acc_buy += max(b, 0.0)
        acc_sell += max(s, 0.0)
        vol = acc_buy + acc_sell
        if vol >= bucket_volume:
            tox = abs(acc_buy - acc_sell) / vol
            bucket_tox.append(float(tox))
            last = float(np.mean(bucket_tox[-int(window) :]))
            acc_buy = 0.0
            acc_sell = 0.0
        out[i] = last
    return out


def vpin_proxy(
    book: pl.DataFrame,
    *,
    bucket_volume: float | None = None,
    window: int = 50,
) -> pl.DataFrame:
    """Bulk-volume VPIN proxy on consecutive snapshot signed top-size changes.

    Research diagnostic only. Uses Cont-style signed order-flow magnitude as a
    trade-volume stand-in when true trade prints are absent. Default is a
    count-window rolling mean of |buy−sell| / (buy+sell). When ``bucket_volume``
    is set, fills equal-volume buckets (Easley et al. volume clock) and rolls
    the last ``window`` bucket toxicities.
    """
    required = ("security_id", "event_time", "top_bid_size", "top_ask_size", "best_bid", "best_ask")
    missing = [c for c in required if c not in book.columns]
    if missing:
        raise ValueError(f"book missing columns: {missing}")
    if window < 2:
        raise ValueError("window must be >= 2")
    if bucket_volume is not None and float(bucket_volume) <= 0.0:
        raise ValueError("bucket_volume must be > 0")
    frame = book.sort(["security_id", "event_time"]).with_columns(
        pl.col("best_bid").shift(1).over("security_id").alias("_pb"),
        pl.col("best_ask").shift(1).over("security_id").alias("_pa"),
        pl.col("top_bid_size").shift(1).over("security_id").alias("_ptb"),
        pl.col("top_ask_size").shift(1).over("security_id").alias("_pta"),
    )
    # Approximate buy/sell volume from top-of-book updates (OFI absolute legs).
    buy = pl.when(pl.col("best_bid") >= pl.col("_pb")).then(pl.col("top_bid_size")).otherwise(
        0.0
    ) + pl.when(pl.col("best_ask") <= pl.col("_pa")).then(pl.col("_pta")).otherwise(0.0)
    sell = pl.when(pl.col("best_bid") <= pl.col("_pb")).then(pl.col("_ptb")).otherwise(
        0.0
    ) + pl.when(pl.col("best_ask") >= pl.col("_pa")).then(pl.col("top_ask_size")).otherwise(0.0)
    frame = frame.with_columns(buy.alias("_buy"), sell.alias("_sell"))
    vol = pl.col("_buy") + pl.col("_sell")
    toxicity = (
        pl.when(vol > 0.0)
        .then((pl.col("_buy") - pl.col("_sell")).abs() / vol)
        .otherwise(None)
        .alias("_tox")
    )
    frame = frame.with_columns(toxicity)
    if bucket_volume is None:
        out = frame.with_columns(
            pl.col("_tox")
            .rolling_mean(window_size=int(window), min_samples=max(2, window // 5))
            .over("security_id")
            .alias("vpin")
        )
    else:
        buy_a = frame["_buy"].to_numpy().astype(float)
        sell_a = frame["_sell"].to_numpy().astype(float)
        sids = frame["security_id"].to_list()
        vpin_out = np.full(frame.height, np.nan, dtype=float)
        i = 0
        n = int(frame.height)
        while i < n:
            j = i + 1
            while j < n and sids[j] == sids[i]:
                j += 1
            vpin_out[i:j] = _volume_clock_vpin(
                buy_a[i:j],
                sell_a[i:j],
                bucket_volume=float(bucket_volume),
                window=int(window),
            )
            i = j
        out = frame.with_columns(pl.Series("vpin", vpin_out))
    drop = [c for c in out.columns if c.startswith("_")]
    return out.drop(drop)
