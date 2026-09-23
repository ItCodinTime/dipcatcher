"""Causal crypto factor model: factor returns, rolling OLS betas, and
book-level factor P&L decomposition.

Factors are built cross-sectionally from the bar panel (all quantities are
known at the bar's close — no look-ahead):

- ``mkt``: equal-weight mean 1-bar return (the crypto market factor).
- ``mom``: long top-tercile / short bottom-tercile on the trailing
  ``lookback``-bar cumulative return.
- ``liq``: long top-tercile / short bottom-tercile on dollar volume
  (big-minus-small liquidity factor).
- ``carry`` (only when a funding frame is supplied): long bottom-tercile /
  short top-tercile on the last funding rate — the book that *receives*
  funding.

Betas are trailing-window OLS estimated on returns strictly before each
timestamp (solved via rolling normal equations — no look-ahead).
Decomposition uses the backtest convention ``book_ret_t = Σ_i w_{i,t-1}
r_{i,t}`` and attributes ``Σ_i w_{i,t-1} β_{i,k,t-1} · f_{k,t}`` to factor
``k``; the remainder is alpha + idiosyncratic residual.

Research diagnostics only — never a live P&L claim.
"""

from __future__ import annotations

import numpy as np
import polars as pl

FACTOR_NAMES = ("mkt", "mom", "liq", "carry")


def _tercile_spread(signal: np.ndarray, forward: np.ndarray) -> float:
    """Top-tercile minus bottom-tercile mean of ``forward`` ranked by ``signal``.

    NaN when fewer than 6 names or the terciles are degenerate.
    """
    ok = np.isfinite(signal) & np.isfinite(forward)
    if ok.sum() < 6:
        return float("nan")
    s = signal[ok]
    f = forward[ok]
    n = s.size
    order = np.argsort(s, kind="stable")
    third = max(n // 3, 1)
    lo = f[order[:third]]
    hi = f[order[n - third :]]
    if lo.size == 0 or hi.size == 0 or not (np.isfinite(lo).all() and np.isfinite(hi).all()):
        return float("nan")
    return float(np.mean(hi) - np.mean(lo))


def crypto_factor_returns(
    bars: pl.DataFrame,
    *,
    lookback: int = 20,
    funding: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Cross-sectional factor returns per ``event_time``.

    ``bars``: ``security_id, event_time, close, volume``. ``funding``
    (optional): ``security_id, event_time, value`` — joined asof backward
    so the last observed rate at each bar is used.
    """
    required = {"security_id", "event_time", "close", "volume"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"bars missing columns: {sorted(missing)}")
    if lookback < 2:
        raise ValueError("lookback must be >= 2")

    panel = bars.select("security_id", "event_time", "close", "volume").sort(
        ["security_id", "event_time"]
    )
    panel = panel.with_columns(
        (pl.col("close") / pl.col("close").shift(1) - 1.0).over("security_id").alias("ret1"),
        (pl.col("close") / pl.col("close").shift(lookback) - 1.0)
        .over("security_id")
        .alias("mom_sig"),
        (pl.col("close") * pl.col("volume")).alias("adv"),
    )
    if funding is not None and funding.height > 0:
        fmiss = {"security_id", "event_time", "value"} - set(funding.columns)
        if fmiss:
            raise ValueError(f"funding missing columns: {sorted(fmiss)}")
        froll = (
            funding.select("security_id", "event_time", "value")
            .sort(["security_id", "event_time"])
            .rename({"value": "funding_rate"})
        )
        panel = panel.join_asof(
            froll, on="event_time", by="security_id", strategy="backward"
        )
    else:
        panel = panel.with_columns(pl.lit(None, dtype=pl.Float64).alias("funding_rate"))

    rows: list[dict] = []
    for (et,) , grp in panel.group_by("event_time", maintain_order=True):
        ret1 = grp["ret1"].to_numpy().astype(float)
        mom_sig = grp["mom_sig"].to_numpy().astype(float)
        adv = grp["adv"].to_numpy().astype(float)
        rate = grp["funding_rate"].to_numpy().astype(float)
        rows.append(
            {
                "event_time": et,
                "mkt": float(np.nanmean(ret1)) if np.isfinite(ret1).any() else float("nan"),
                # mom factor = long recent winners / short losers
                "mom": _tercile_spread(mom_sig, ret1),
                # liq factor = long liquid / short illiquid
                "liq": _tercile_spread(adv, ret1),
                # carry = long names that receive funding / short payers
                "carry": -_tercile_spread(rate, ret1)
                if np.isfinite(rate).any()
                else float("nan"),
                "n_names": int(np.isfinite(ret1).sum()),
            }
        )
    return pl.DataFrame(rows).sort("event_time")


def _rolling_beta(
    y: np.ndarray, x: np.ndarray, window: int, ridge: float
) -> np.ndarray:
    """Trailing-window OLS betas (no intercept handling — caller centers).

    Returns ``(T, K)`` array; row t fits y[t-window:t] on x[t-window:t]
    (strictly past rows — no look-ahead). NaN where insufficient data.
    """
    t_n, k = x.shape
    out = np.full((t_n, k), np.nan)
    if t_n <= window or k == 0:
        return out
    ok = np.isfinite(y) & np.isfinite(x).all(axis=1)
    xc = np.where(ok[:, None], x, 0.0)
    yc = np.where(ok, y, 0.0)
    # cumulative sums of normal-equation terms
    cs_xx = np.concatenate(
        [np.zeros((1, k, k)), np.cumsum(xc[:, :, None] * xc[:, None, :], axis=0)]
    )
    cs_xy = np.concatenate([np.zeros((1, k)), np.cumsum(xc * yc[:, None], axis=0)])
    cs_n = np.concatenate([[0.0], np.cumsum(ok.astype(float))])
    for t in range(window, t_n):
        sxx = cs_xx[t] - cs_xx[t - window]
        sxy = cs_xy[t] - cs_xy[t - window]
        n_ok = cs_n[t] - cs_n[t - window]
        if n_ok < k + 2:
            continue
        a = sxx + ridge * np.eye(k)
        try:
            out[t] = np.linalg.solve(a, sxy)
        except np.linalg.LinAlgError:
            continue
    return out


def estimate_factor_betas(
    bars: pl.DataFrame,
    factors: pl.DataFrame,
    *,
    window: int = 60,
    ridge: float = 1e-6,
) -> pl.DataFrame:
    """Rolling causal OLS betas of each security's returns on factor returns.

    Output: long frame ``event_time, security_id, beta_<factor>`` with betas
    valid from ``window`` observations onward; earlier rows are NaN.
    """
    if window < 4:
        raise ValueError("window must be >= 4")
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and non-negative")
    frets = (
        bars.select("security_id", "event_time", "close")
        .sort(["security_id", "event_time"])
        .with_columns(
            (pl.col("close") / pl.col("close").shift(1) - 1.0)
            .over("security_id")
            .alias("ret1")
        )
        .select("security_id", "event_time", "ret1")
    )
    fcols = [c for c in FACTOR_NAMES if c in factors.columns]
    if not fcols:
        raise ValueError("factors frame carries none of the factor columns")
    fmat = factors.select("event_time", *fcols).sort("event_time")
    joined = frets.join(fmat, on="event_time", how="left")

    out_rows: list[dict] = []
    for (sid,), grp in joined.group_by("security_id", maintain_order=True):
        y = grp["ret1"].to_numpy().astype(float)
        x = np.column_stack([grp[c].to_numpy().astype(float) for c in fcols])
        # drop rows whose factor set is incomplete from the fit stream
        beta = _rolling_beta(y, x, window, ridge)
        times = grp["event_time"].to_list()
        for t in range(len(times)):
            row: dict = {"security_id": sid, "event_time": times[t]}
            for j, c in enumerate(fcols):
                row[f"beta_{c}"] = float(beta[t, j]) if np.isfinite(beta[t, j]) else None
            out_rows.append(row)
    return pl.DataFrame(out_rows)


def decompose_book_factors(
    weights: pl.DataFrame,
    bars: pl.DataFrame,
    factors: pl.DataFrame,
    betas: pl.DataFrame,
) -> pl.DataFrame:
    """Per-``event_time`` factor decomposition of book returns.

    ``book_ret_t = Σ_i w_{i,t-1} r_{i,t}``; factor k contributes
    ``(Σ_i w_{i,t-1} β_{i,k,t-1}) · f_{k,t}``; ``alpha_resid`` is the rest.
    All right-hand-side quantities use information available at t-1 or the
    realized factor/return at t — causal.
    """
    for name, frame, cols in (
        ("weights", weights, {"event_time", "security_id", "target_weight"}),
        ("bars", bars, {"event_time", "security_id", "close"}),
        ("factors", factors, {"event_time"}),
    ):
        missing = cols - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")

    rets = (
        bars.select("security_id", "event_time", "close")
        .sort(["security_id", "event_time"])
        .with_columns(
            (pl.col("close") / pl.col("close").shift(1) - 1.0)
            .over("security_id")
            .alias("ret1")
        )
        .select("security_id", "event_time", "ret1")
    )
    # Weights/betas persist until re-declared: asof declarations onto the
    # bar grid, then shift one bar — the value in effect during bar t.
    grid = rets.select("security_id", "event_time").sort(["security_id", "event_time"])
    declared_w = weights.select("security_id", "event_time", "target_weight").sort(
        ["security_id", "event_time"]
    )
    w_prev = (
        grid.join_asof(declared_w, on="event_time", by="security_id", strategy="backward")
        .with_columns(pl.col("target_weight").shift(1).over("security_id").alias("w_prev"))
        .select("security_id", "event_time", "w_prev")
    )
    bcols = [c for c in betas.columns if c.startswith("beta_")]
    declared_b = betas.select("security_id", "event_time", *bcols).sort(
        ["security_id", "event_time"]
    )
    b_prev = (
        grid.join_asof(declared_b, on="event_time", by="security_id", strategy="backward")
        .with_columns([pl.col(c).shift(1).over("security_id").alias(c) for c in bcols])
    )
    frame = (
        rets.join(w_prev, on=["security_id", "event_time"], how="inner")
        .join(b_prev, on=["security_id", "event_time"], how="left")
        .filter(pl.col("w_prev") != 0.0)
    )
    fcols = [c for c in FACTOR_NAMES if c in factors.columns and f"beta_{c}" in bcols]
    fmat = factors.select("event_time", *fcols)

    agg_exprs = [
        (pl.col("w_prev") * pl.col("ret1")).sum().alias("book_ret"),
    ]
    for c in fcols:
        # exposure to factor k summed across names; multiply by f_k after join
        agg_exprs.append(
            (pl.col("w_prev") * pl.col(f"beta_{c}")).sum().alias(f"expo_{c}")
        )
    per_t = frame.group_by("event_time", maintain_order=True).agg(agg_exprs)
    per_t = per_t.join(fmat, on="event_time", how="left")
    contrib_cols = []
    for c in fcols:
        per_t = per_t.with_columns((pl.col(f"expo_{c}") * pl.col(c)).alias(f"contrib_{c}"))
        contrib_cols.append(f"contrib_{c}")
    total_contrib = pl.sum_horizontal([pl.col(c) for c in contrib_cols]) if contrib_cols else pl.lit(0.0)
    return per_t.with_columns((pl.col("book_ret") - total_contrib).alias("alpha_resid"))


def factor_summary(frame: pl.DataFrame) -> dict[str, object]:
    """Cumulative factor attribution summary of ``decompose_book_factors``."""
    if frame.height == 0:
        return {"status": "empty", "live_pnl_claim": False, "research_only": True}
    total = float(frame["book_ret"].sum())
    contrib_cols = [c for c in frame.columns if c.startswith("contrib_")]
    by_factor = {
        c[len("contrib_"):]: float(frame[c].sum()) for c in contrib_cols
    }
    resid = float(frame["alpha_resid"].sum())
    denom = abs(total) if abs(total) > 0 else None
    return {
        "book_ret_total": total,
        "by_factor": by_factor,
        "alpha_resid": resid,
        "factor_share_of_abs_book": (
            {k: (v / denom) for k, v in by_factor.items()} if denom else None
        ),
        "convention": "w_prev x beta_prev x factor_t; causal",
        "live_pnl_claim": False,
        "research_only": True,
    }
