"""Compute the ``dip_har`` challenger column over an existing shard's origin grid.

``dip_har`` is a HAR (Heterogeneous AutoRegressive, Corsi 2009) realized-
volatility challenger built on OHLC range information that the close-close
challengers ignore. Per origin, over the trailing ``garch_window``:

* Per-bar realized variance proxy: Parkinson range variance
  ``pk_t = (ln high_t - ln low_t)^2 / (4 ln 2)`` for the bar ending each
  close-close return. ``pk_t`` is ~5x more efficient than ``r_t^2``; any bar
  whose ``pk_t`` is non-finite or negative falls back to ``r_t^2``.
* HAR regression in vol form: ``sv = sqrt(RV)``,
  ``sv_{t+1} ~ c + b1*sv_t + b2*mean(sv_{t-4..t}) + b3*mean(sv_{t-21..t})``
  (1-bar / ~5-bar / ~22-bar components), OLS on the trailing window with
  >= 60 regression rows, coefficients clipped >= 0 to keep sigma sensible.
* ``sigma_hat = max(prediction, 1e-6)``; ``mu_hat = 0.5 * mean(rets_long)``
  (trailing mean shrunk 50% toward zero).
* Quantiles: filtered-historical-simulation style — empirical quantiles of
  standardized residuals ``z_t = r_t / sqrt(RV_t)`` rescaled by ``sigma_hat``:
  ``q_tau = mu_hat + sigma_hat * quantile(z, tau)``. Requires >= 30 finite z.
* CRPS via ``crps_from_quantiles(y, LGBM_TAUS, q)`` — same convention as
  ``_challenger_col.py``.

Replicates the origin loop of ``sota_eval_kronos.py`` (first_origin formula,
garch-window slicing, close-to-close target); binds to the shard via
``bars_sha256`` + config fields so the splice tool can verify row alignment.
Fully deterministic and causal.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sota_eval_kronos import (  # noqa: E402
    LGBM_TAUS,
    TAUS,
    _sha256,
    crps_from_quantiles,
)

from quant_fund.metrics.scoring import pinball_loss  # noqa: E402
from quant_fund.research.sota_evidence import validate_bars  # noqa: E402

MODEL = "dip_har"
PARKINSON_DENOM = 4.0 * np.log(2.0)
HAR_MIN_OBS = 60  # minimum OLS regression rows
HAR_LAG_WEEK = 5  # ~weekly component
HAR_LAG_MONTH = 22  # ~monthly component
Z_MIN = 30  # minimum finite standardized residuals for the FHS quantiles


def _parkinson_rv(highs: np.ndarray, lows: np.ndarray, rets: np.ndarray) -> tuple[np.ndarray, int]:
    """Per-bar RV aligned to ``rets`` (rv[k] is the bar ending return k).

    Parkinson range variance where valid; ``r_k^2`` fallback otherwise.
    Returns (rv, n_parkinson).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        pk = np.log(highs / lows) ** 2 / PARKINSON_DENOM
    ok = np.isfinite(pk) & (pk >= 0.0)
    rv = np.where(ok, pk, rets**2)
    return rv, int(ok.sum())


def _har_forecast(sv: np.ndarray) -> tuple[float, np.ndarray] | None:
    """HAR(1,5,22) OLS on sqrt-RV; (sigma_next, coefs[4]) or None."""
    n = sv.size
    n_reg = n - HAR_LAG_MONTH
    if n_reg < HAR_MIN_OBS:
        return None
    cs = np.concatenate([[0.0], np.cumsum(sv)])  # cs[k] = sum sv[:k]
    t_idx = np.arange(HAR_LAG_MONTH - 1, n - 1)  # predict sv[t+1] from data <= t
    x1 = sv[t_idx]
    x5 = (cs[t_idx + 1] - cs[t_idx + 1 - HAR_LAG_WEEK]) / HAR_LAG_WEEK
    x22 = (cs[t_idx + 1] - cs[t_idx + 1 - HAR_LAG_MONTH]) / HAR_LAG_MONTH
    yv = sv[t_idx + 1]
    x = np.column_stack([np.ones(t_idx.size), x1, x5, x22])
    beta, *_ = np.linalg.lstsq(x, yv, rcond=None)
    if not np.isfinite(beta).all():
        return None
    beta = np.maximum(beta, 0.0)
    f = np.array(
        [1.0, sv[-1], sv[-HAR_LAG_WEEK:].mean(), sv[-HAR_LAG_MONTH:].mean()]
    )
    return max(float(f @ beta), 1e-6), beta


def har_quantiles(
    rets_long: np.ndarray, highs: np.ndarray, lows: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    """(q@LGBM_TAUS, q@TAUS, stats) for one origin, or None -> honest NaN."""
    rv, n_pk = _parkinson_rv(highs, lows, rets_long)
    if not np.isfinite(rv).all() or rv.size < HAR_MIN_OBS + HAR_LAG_MONTH:
        return None
    sv = np.sqrt(rv)
    fc = _har_forecast(sv)
    if fc is None:
        return None
    sigma_hat, beta = fc
    mu_hat = 0.5 * float(np.mean(rets_long))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = rets_long / sv
    z = z[np.isfinite(z)]
    if z.size < Z_MIN:
        return None
    q_lgbm = mu_hat + sigma_hat * np.quantile(z, LGBM_TAUS)
    q_taus = mu_hat + sigma_hat * np.quantile(z, np.asarray(TAUS))
    stats = {
        "beta": beta,
        "n_parkinson": n_pk,
        "n_r2_fallback": int(rv.size - n_pk),
        "n_z": int(z.size),
        "sigma_hat": sigma_hat,
    }
    return q_lgbm, q_taus, stats


def _find_bars(bars_root: Path, bars_sha256: dict[str, str]) -> Path:
    """Resolve the shard's ``{filename: sha256}`` binding to a local file."""
    if len(bars_sha256) != 1:
        raise ValueError(f"expected single-asset shard, got {sorted(bars_sha256)}")
    name, want = next(iter(bars_sha256.items()))
    cand = bars_root / name
    if not cand.is_file():
        found = [p for p in sorted(bars_root.glob("*.parquet")) if _sha256(p) == want]
        if len(found) != 1:
            raise ValueError(f"{name}: {len(found)} hash matches in {bars_root}")
        cand = found[0]
    if _sha256(cand) != want:
        raise ValueError(f"{cand.name}: sha256 mismatch vs shard record")
    return cand


def compute_column(bars_path: Path, cfg: dict) -> dict[str, np.ndarray]:
    """(crps[R], pin[R,T], target_time_ns[R]) over the shard's origin grid."""
    frame = pl.read_parquet(bars_path)
    event_times, interval = validate_bars(frame)
    closes = frame["close"].to_numpy().astype(float)
    highs = frame["high"].to_numpy().astype(float)
    lows = frame["low"].to_numpy().astype(float)
    n = closes.size
    origins = int(cfg["origins_per_asset"])
    lookback = int(cfg["lookback"])
    window = int(cfg["window"])
    garch_window = int(cfg["garch_window"])
    first_origin = max(lookback, window + 1, n - origins - 1)
    if first_origin >= n - 1:
        raise ValueError(f"{bars_path.name}: not enough bars ({n})")
    crps = np.full(n - 1 - first_origin, np.nan)
    pin = np.full((n - 1 - first_origin, len(TAUS)), np.nan)
    tt = np.empty(n - 1 - first_origin, dtype=np.int64)
    betas = np.full((n - 1 - first_origin, 4), np.nan)
    n_pk = np.zeros(n - 1 - first_origin, dtype=np.int64)
    n_r2 = np.zeros(n - 1 - first_origin, dtype=np.int64)
    n_z = np.zeros(n - 1 - first_origin, dtype=np.int64)
    sig_hats = np.full(n - 1 - first_origin, np.nan)
    for row, i in enumerate(range(first_origin, n - 1)):
        closes_hist = closes[: i + 1]
        long_start = max(0, closes_hist.size - (garch_window + 1))
        rets_long = np.diff(closes_hist[long_start:]) / closes_hist[long_start:-1]
        # bars long_start+1 .. i produce rets_long; their high/low give pk_t.
        hi_w = highs[long_start + 1 : i + 1]
        lo_w = lows[long_start + 1 : i + 1]
        y = float(closes[i + 1] / closes[i] - 1.0)
        tt[row] = int(event_times[i + 1])
        try:
            res = har_quantiles(rets_long, hi_w, lo_w)
            if res is None:
                continue
            q_lgbm, q_taus, stats = res
            crps[row] = crps_from_quantiles(y, LGBM_TAUS, q_lgbm)
            for k, tau in enumerate(TAUS):
                if np.isfinite(q_taus[k]):
                    pin[row, k] = float(
                        pinball_loss(np.array([y]), np.array([q_taus[k]]), tau)[0]
                    )
            betas[row] = stats["beta"]
            n_pk[row] = stats["n_parkinson"]
            n_r2[row] = stats["n_r2_fallback"]
            n_z[row] = stats["n_z"]
            sig_hats[row] = stats["sigma_hat"]
        except Exception:  # noqa: BLE001 - honest NaN, row stays disclosed
            continue
    return {
        "crps_col": crps,
        "pin_cols": pin,
        "target_time_ns": tt,
        "bar_interval_ns": np.asarray(interval, dtype=np.int64),
        "har_betas": betas,
        "n_parkinson": n_pk,
        "n_r2_fallback": n_r2,
        "n_z": n_z,
        "sigma_hat": sig_hats,
    }


def _beta_stats(betas: np.ndarray) -> dict:
    """Per-coefficient finite-mean/min/max over fitted origins."""
    names = ("c", "b_daily", "b_weekly", "b_monthly")
    out = {}
    for k, name in enumerate(names):
        col = betas[:, k]
        col = col[np.isfinite(col)]
        out[name] = {
            "mean": float(col.mean()) if col.size else None,
            "min": float(col.min()) if col.size else None,
            "max": float(col.max()) if col.size else None,
        }
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=Path, required=True, help="existing losses shard to match")
    p.add_argument("--bars-root", type=Path, default=Path("data/raw/sources"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    with np.load(args.shard, allow_pickle=False) as z:
        meta = json.loads(str(z["meta_json"]))
        n_rows = int(z["crps_matrix"].shape[0])
        shard_names = [str(x) for x in z["model_names"]]
    cfg = meta["config"]
    want_cols = int(cfg.get("origins_per_asset", 0)) >= 1
    if not want_cols or MODEL in shard_names:
        raise ValueError(f"{args.shard.name}: already has {MODEL} or bad config")

    bars = _find_bars(args.bars_root, meta["bars_sha256"])
    t0 = time.time()
    out = compute_column(bars, cfg)
    if out["crps_col"].shape[0] != n_rows:
        raise ValueError(
            f"{args.shard.name}: grid mismatch {out['crps_col'].shape[0]} != {n_rows}"
        )
    fitted = np.isfinite(out["har_betas"]).all(axis=1)
    meta_out = {
        "tool": Path(__file__).name,
        "model": MODEL,
        "shard": args.shard.name,
        "shard_sha256": _sha256(args.shard),
        "bars": bars.name,
        "bars_sha256": meta["bars_sha256"],
        "bars_file_sha256": _sha256(bars),
        "asset_names": meta.get("asset_names"),
        "config": {k: cfg.get(k) for k in
                   ("origins_per_asset", "lookback", "window", "garch_window",
                    "taus", "seed")},
        "n_rows": n_rows,
        "n_finite_crps": int(np.isfinite(out["crps_col"]).sum()),
        "har": {
            "rv_proxy": "parkinson_(ln(h/l))^2/(4ln2); fallback r^2",
            "regression": "sqrt(RV)_{t+1} ~ c + b1*sv_t + b2*mean(sv_{t-4..t}) + b3*mean(sv_{t-21..t})",
            "min_ols_obs": HAR_MIN_OBS,
            "coef_clip": ">=0",
            "mu_shrinkage": 0.5,
            "z_min": Z_MIN,
            "n_fitted": int(fitted.sum()),
            "coef_stats": _beta_stats(out["har_betas"]),
            "rv_source_counts": {
                "parkinson_bars": int(out["n_parkinson"].sum()),
                "r2_fallback_bars": int(out["n_r2_fallback"].sum()),
            },
            "sigma_hat_mean": float(np.nanmean(out["sigma_hat"]))
            if fitted.any()
            else None,
        },
        "elapsed_s": round(time.time() - t0, 3),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        crps_col=out["crps_col"],
        pin_cols=out["pin_cols"],
        target_time_ns=out["target_time_ns"],
        bar_interval_ns=out["bar_interval_ns"],
        meta_json=np.array(json.dumps(meta_out)),
    )
    print(
        f"{args.shard.name}: {MODEL} rows={n_rows} "
        f"finite={meta_out['n_finite_crps']} mean_crps={np.nanmean(out['crps_col']):.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
