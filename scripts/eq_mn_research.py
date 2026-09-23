"""Fast equity market-neutral research harness (screening only).

Mirrors run_backtest semantics closely enough for signal screening:
- signal computed from data through close of day t
- weights executed at split-adjusted open of day t+1
- marks at close_total_return
- costs: commission + half-spread + sqrt impact (participation capped)
- borrow on shorts; dollar-neutral LS construction

Final candidates must still be confirmed through the real `run_backtest`.
Not a live P&L claim.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def load_panel(path: str = "data/file_us_wide/silver/bars.parquet") -> dict:
    df = pl.read_parquet(
        path,
        columns=[
            "symbol",
            "event_time",
            "open_split_adjusted",
            "close_split_adjusted",
            "close_total_return",
            "volume_split_adjusted",
            "sector",
        ],
    )
    syms = sorted(df["symbol"].unique().to_list())
    dates = sorted(df["event_time"].unique().to_list())
    sidx = {s: i for i, s in enumerate(syms)}
    didx = {d: i for i, d in enumerate(dates)}
    T, N = len(dates), len(syms)
    open_ = np.full((T, N), np.nan)
    close = np.full((T, N), np.nan)
    tr = np.full((T, N), np.nan)
    volu = np.full((T, N), np.nan)
    for row in df.iter_rows(named=True):
        i, j = didx[row["event_time"]], sidx[row["symbol"]]
        open_[i, j] = row["open_split_adjusted"]
        close[i, j] = row["close_split_adjusted"]
        tr[i, j] = row["close_total_return"]
        volu[i, j] = row["volume_split_adjusted"]
    sector_list = [df.filter(pl.col("symbol") == s)["sector"][0] or "?" for s in syms]
    sectors = np.array(sector_list)
    adv = np.nanmedian(tr * 0 + close * volu, axis=0)  # placeholder; recomputed below
    adv = np.nanmedian(close * volu, axis=0)
    return {
        "dates": dates,
        "syms": syms,
        "open": open_,
        "close": close,
        "tr": tr,
        "volu": volu,
        "sectors": sectors,
        "adv": adv,
    }


def book_metrics(nav: np.ndarray, dates, periods: float = 252.0) -> dict:
    r = np.diff(nav) / nav[:-1]
    r = r[np.isfinite(r)]
    if r.size < 2:
        return {
            "sharpe": float("nan"),
            "mdd": float("nan"),
            "cagr": float("nan"),
            "total": float("nan"),
            "vol": float("nan"),
        }
    mu, sd = r.mean(), r.std(ddof=1)
    sharpe = mu / sd * np.sqrt(periods) if sd > 0 else float("nan")
    w = np.cumprod(1 + r)
    mdd = float((w / np.maximum.accumulate(w) - 1).min())
    yrs = r.size / periods
    cagr = w[-1] ** (1 / yrs) - 1 if yrs > 0 and w[-1] > 0 else float("nan")
    return {
        "sharpe": float(sharpe),
        "mdd": mdd,
        "cagr": float(cagr),
        "total": float(w[-1] - 1),
        "vol": float(sd * np.sqrt(periods)),
        "n": int(r.size),
        "t0": str(dates[0])[:10],
        "t1": str(dates[-1])[:10],
    }


def fast_bt(
    open_: np.ndarray,
    tr: np.ndarray,
    volu: np.ndarray,
    weights: np.ndarray,
    *,
    nav0: float = 1e6,
    commission_bps: float = 1.0,
    half_spread_bps: float = 5.0,
    impact_y: float = 0.1,
    borrow_bps_year: float = 50.0,
    part_cap: float = 0.10,
    sigma_hint: float = 0.02,
) -> dict:
    """Share-level sim: weights[:,d] decided at close d -> filled at open d+1.

    weights row t applies to day t+1. Mirrors engine cost stack + borrow.
    """
    T, N = open_.shape
    nav = np.empty(T)
    cash = nav0
    sh = np.zeros(N)
    trn = np.zeros(T)
    cost_tot = 0.0
    for t in range(T):
        px_open = open_[t]
        px_mark = tr[t]
        adv_d = np.nan_to_num(px_open * volu[t], nan=1.0)
        tgt = weights[t - 1] if t > 0 else np.zeros(N)
        nav_prev = cash + np.sum(sh * np.nan_to_num(px_mark, nan=0.0))
        if nav_prev <= 0:
            nav[t:] = np.nan
            break
        # desired positions at today's open
        valid = np.isfinite(px_open) & (px_open > 0)
        desired = np.where(valid, tgt * nav_prev / np.where(px_open > 0, px_open, 1.0), 0.0)
        delta = np.where(valid, desired - sh, 0.0)
        # participation cap
        max_qty = part_cap * np.where(adv_d > 0, adv_d, 0.0) / np.where(px_open > 0, px_open, 1.0)
        delta = np.clip(delta, -max_qty, max_qty)
        notion = np.abs(delta) * np.nan_to_num(px_open)
        part = np.where(adv_d > 0, notion / np.maximum(adv_d, 1e-12), 0.0)
        cost = (
            notion * commission_bps / 1e4
            + notion * half_spread_bps / 1e4
            + notion * impact_y * sigma_hint * np.sqrt(np.clip(part, 0, None))
        )
        cost = np.where(np.isfinite(cost), cost, 0.0)
        # sequential cash feasibility for buys (engine semantics approx: skip if unaffordable)
        need = notion + cost
        cash_avail = cash
        order = np.argsort(-notion)  # deterministic, largest first
        for j in order:
            if delta[j] > 0 and need[j] > cash_avail:
                delta[j] = 0.0
                cost[j] = 0.0
                notion[j] = 0.0
            elif delta[j] != 0:
                cash_avail -= need[j]
        cash -= float(np.sum(delta * np.nan_to_num(px_open)) + np.sum(cost))
        sh = sh + delta
        trn[t] = float(np.sum(notion) / max(nav_prev, 1e-12))
        cost_tot += float(np.sum(cost))
        # mark + borrow
        short_not = np.sum(np.abs(np.minimum(sh, 0.0)) * np.nan_to_num(px_mark))
        borrow = short_not * borrow_bps_year / 1e4 / 252.0
        cash -= borrow
        nav[t] = cash + np.sum(sh * np.nan_to_num(px_mark))
    return {"nav": nav, "turnover": trn, "cost_total": cost_tot}


# ---------- signal library (all causal, computed on close of day t) ----------


def zscore_xs(x: np.ndarray, axis: int = 1, clip: float = 3.0) -> np.ndarray:
    mu = np.nanmean(x, axis=axis, keepdims=True)
    sd = np.nanstd(x, axis=axis, keepdims=True)
    z = (x - mu) / np.where(sd > 1e-12, sd, np.nan)
    return np.clip(z, -clip, clip)


def demean_groups(z: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = np.full_like(z, np.nan)
    for g in np.unique(groups):
        m = groups == g
        sub = z[:, m]
        out[:, m] = sub - np.nanmean(sub, axis=1, keepdims=True)
    return out


def rolling_sum(x: np.ndarray, k: int) -> np.ndarray:
    c = np.nancumsum(np.nan_to_num(x), axis=0)
    out = np.full_like(x, np.nan)
    out[k:] = c[k:] - c[:-k]
    out[:k] = np.nan
    cnt = np.cumsum(np.isfinite(x).astype(int), axis=0)
    cntk = np.full_like(x, np.nan)
    cntk[k:] = cnt[k:] - cnt[:-k]
    out[cntk < k] = np.nan
    return out


def ewma(x: np.ndarray, hl: float) -> np.ndarray:
    a = 1 - 0.5 ** (1 / hl)
    out = np.full_like(x, np.nan)
    prev = np.zeros(x.shape[1])
    init = False
    for i in range(x.shape[0]):
        row = x[i]
        if not init:
            prev = np.where(np.isfinite(row), row, 0.0)
            init = True
        else:
            prev = a * np.nan_to_num(row) + (1 - a) * prev
        out[i] = prev
    out[~np.isfinite(x)] = np.nan
    return out


def signals(
    tr: np.ndarray, open_: np.ndarray, close: np.ndarray, volu: np.ndarray, sectors: np.ndarray
) -> dict[str, np.ndarray]:
    """Each signal: (T,N), higher = more attractive (long). Causal at close t."""
    r1 = np.full_like(tr, np.nan)
    r1[1:] = tr[1:] / tr[:-1] - 1
    lr1 = np.log1p(np.clip(r1, -0.99, 10))
    ret_k = {k: rolling_sum(lr1, k) for k in (1, 2, 3, 5, 10, 21)}

    # vol estimates
    vol20 = np.full_like(tr, np.nan)
    cum2 = np.nancumsum(np.nan_to_num(lr1**2), axis=0)
    n20 = 20
    v2 = np.full_like(tr, np.nan)
    v2[n20:] = (cum2[n20:] - cum2[:-n20]) / n20
    vol20 = np.sqrt(np.clip(v2, 0, None) * 252)

    sig = {}
    for k, r in ret_k.items():
        sig[f"rev{k}"] = -zscore_xs(r)
        sig[f"rev{k}_sec"] = -demean_groups(zscore_xs(r), sectors)
    # beta-neutral residual reversal: subtract sector mean return cumulatively
    for k, r in ret_k.items():
        sec_mean = demean_groups(np.nan_to_num(r), sectors)
        sig[f"resid_rev{k}"] = -zscore_xs(np.where(np.isfinite(r), r - sec_mean, np.nan))
    # momentum 12-1 (skip last 21d): mom[t] = sum lr1[t-251:t-21]
    lr_cum = np.nancumsum(np.nan_to_num(lr1), axis=0)
    mom = np.full_like(tr, np.nan)
    a = lr_cum
    mom[252:] = a[252 - 21 : -21 or None][0 : tr.shape[0] - 252] - a[: tr.shape[0] - 252]
    sig["mom12_1"] = zscore_xs(mom)
    sig["mom12_1_sec"] = demean_groups(zscore_xs(mom), sectors)
    # z vs ma20
    ma20 = np.full_like(tr, np.nan)
    c = np.nancumsum(np.nan_to_num(close), axis=0)
    ma20[20:] = (c[20:] - c[:-20]) / 20
    z_ma = (close - ma20) / np.where(ma20 > 0, ma20, np.nan)
    sig["zma20"] = -zscore_xs(z_ma)
    sig["zma20_sec"] = -demean_groups(zscore_xs(z_ma), sectors)
    # overnight reversal: -(open/close_prev - 1)
    on = np.full_like(tr, np.nan)
    on[1:] = open_[1:] / close[:-1] - 1
    sig["onrev"] = -zscore_xs(on)
    # vol-scaled reversal
    sig["rev5_vol"] = sig["rev5"] / np.where(vol20 > 0.05, vol20, np.nan)
    sig["rev5_vol"] = zscore_xs(np.clip(sig["rev5_vol"], -1e3, 1e3))
    sig["vol20"] = vol20
    return sig


def to_weights(
    z: np.ndarray,
    *,
    gross: float = 2.0,
    cap: float = 0.03,
    sector_neutral: np.ndarray | None = None,
    sectors: np.ndarray | None = None,
) -> np.ndarray:
    """z (T,N) -> dollar-neutral L1 book. cap = per-name |w| cap."""
    w = z - np.nanmean(z, axis=1, keepdims=True)
    if sector_neutral is not None and sectors is not None:
        w = demean_groups(np.nan_to_num(w), sectors)
        w = np.where(np.isfinite(z), w, 0.0)
    w = np.nan_to_num(w)
    l1 = np.sum(np.abs(w), axis=1, keepdims=True)
    w = w / np.where(l1 > 0, l1, 1.0) * gross
    w = np.clip(w, -cap, cap)
    # renormalize to gross after cap
    l1b = np.sum(np.abs(w), axis=1, keepdims=True)
    scale = np.minimum(1.0, np.where(l1b > 0, gross / np.where(l1b > 0, l1b, 1.0), 0.0))
    w = w * scale
    return w


def vol_target_weights(
    w: np.ndarray,
    tr: np.ndarray,
    *,
    target_vol: float = 0.10,
    lb: int = 40,
    max_lev: float = 3.0,
    open_: np.ndarray | None = None,
) -> np.ndarray:
    """Causal book-level vol target: scale w by trailing realized book vol.

    Book ret approximation uses close-to-close returns lagged one bar — same
    information set as the weight decision (close of day t).
    """
    r1 = np.full_like(tr, np.nan)
    r1[1:] = tr[1:] / tr[:-1] - 1
    br = np.full(w.shape[0], np.nan)
    br[1:] = np.nansum(w[:-1] * np.nan_to_num(r1[1:]), axis=1)
    out = np.copy(w)
    for t in range(w.shape[0]):
        if t > lb:
            win = br[t - lb : t]
            win = win[np.isfinite(win)]
            if win.size >= 20:
                vol = float(np.std(win, ddof=1) * np.sqrt(252))
                if np.isfinite(vol) and vol > 1e-4:
                    out[t] = w[t] * min(max_lev, target_vol / vol)
    return out


def smooth_weights(w: np.ndarray, hl: float) -> np.ndarray:
    a = 1 - 0.5 ** (1 / hl)
    out = np.zeros_like(w)
    prev = np.zeros(w.shape[1])
    for t in range(w.shape[0]):
        prev = a * w[t] + (1 - a) * prev
        out[t] = prev
    return out


def eval_cfg(
    sig,
    panel,
    *,
    gross=2.0,
    cap=0.03,
    sec=None,
    t0=0,
    t1=None,
    vt=None,
    vt_lb=40,
    smooth=None,
    **bt_kw,
):
    w = to_weights(
        sig,
        gross=gross,
        cap=cap,
        sector_neutral=np.ones(1) if sec else None,
        sectors=panel["sectors"],
    )
    if smooth:
        w = smooth_weights(w, smooth)
    if vt is not None:
        w = vol_target_weights(w, panel["tr"], target_vol=vt, lb=vt_lb, open_=panel["open"])
    if t1 is None:
        t1 = w.shape[0]
    res = fast_bt(panel["open"], panel["tr"], panel["volu"], w, **bt_kw)
    nav = res["nav"][t0:t1]
    dates = panel["dates"][t0:t1]
    m = book_metrics(nav, dates)
    m["mean_turn"] = float(np.nanmean(res["turnover"][t0:t1]))
    m["cost"] = res["cost_total"]
    return m, w, res


if __name__ == "__main__":
    import json

    panel = load_panel()
    print("loaded", len(panel["dates"]), "days x", len(panel["syms"]))
    sigs = signals(panel["tr"], panel["open"], panel["close"], panel["volu"], panel["sectors"])
    print("signals:", sorted(sigs))
    # dev / holdout split at 2025-01-01
    import datetime as dt

    cut = next(
        i for i, d in enumerate(panel["dates"]) if d >= dt.datetime(2025, 1, 1, tzinfo=d.tzinfo)
    )
    print("split idx", cut, panel["dates"][cut])
    out = {}
    for name, s in sigs.items():
        if name == "vol20":
            continue
        for sec in (False, True):
            key = f"{name}{'_SN' if sec else ''}"
            m, w, r = eval_cfg(s, panel, gross=2.0, cap=0.03, sec=sec, t0=252, t1=cut)
            out[key] = {k: m[k] for k in ("sharpe", "mdd", "cagr", "total", "mean_turn")}
            print(key, json.dumps(out[key], default=str))
    json_path = "artifacts/eqmn_screen.json"
    import pathlib

    pathlib.Path("artifacts").mkdir(exist_ok=True)
    pathlib.Path(json_path).write_text(json.dumps(out, indent=2, default=str))
    print("wrote", json_path)
