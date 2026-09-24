"""Compute the ``dip_volm`` challenger column over an existing shard's origin grid.

``dip_volm`` is a volume-conditioned volatility challenger (GARCH-X / MDH
style): the bars carry a ``volume`` column that no existing challenger uses,
and the volume-volatility correlation is one of the strongest stylized facts in
crypto — high-volume bars precede high-vol bars. The next-bar dispersion is
therefore conditioned on trailing volume surprise.

Design (causal, deterministic). Per origin i the trailing ``garch_window``
returns ``r_t`` and aligned volumes ``v_t`` are used:

* Volume surprise ``vs_t = v_t / median(v_{t-19..t})`` — a 20-bar trailing
  median baseline ending at and including bar t (bars with non-finite or
  non-positive volume, or fewer than 5 valid baseline bars, emit NaN ``vs_t``
  and are skipped everywhere ``vs`` is consumed). ``vs`` depends only on
  volumes up to bar t, so it is computed once on the full series.
* Recent surprise ``S``: EWMA (lam=0.7, weights lam^age normalized) over the
  last 10 window positions' ``vs`` values; positions with invalid ``vs`` are
  dropped from both numerator and weight mass.
* Base sigma: RiskMetrics EWMA ``ewma_next_sigma(r, lam=0.94)`` — the forecast
  sigma for the next bar. The same recursion supplies per-bar forecasts
  ``sig_t`` (sqrt of the variance state *before* absorbing ``r_t``) used to
  standardize residuals.
* Elasticity ``gamma``: per origin the OLS slope of ``log|r_t|`` on
  ``log(vs_t)`` over the window is clipped to [0,1] (``gamma_reg``) and blended
  50/50 with the fixed prior 0.5: ``gamma_hat = 0.5*gamma_reg + 0.25``. When the
  regression is infeasible (<30 usable pairs or degenerate variance) the fixed
  prior is used alone (``gamma_hat = 0.5``). Half-weight on the data-driven
  estimate keeps the conditioning honest while bounding overfit noise; the
  blend is fixed and disclosed rather than tuned.
* Conditioned sigma ``sigma_cond = sigma_base * clip(S**gamma_hat, 0.6, 1.8)``.
* Standardized residuals ``z_t = r_t / (sig_t * vs_t**gamma_hat)`` over window
  bars with valid ``vs`` and positive ``sig_t``; requires >=30 finite z.
* Quantiles ``q_tau = mu_hat + sigma_cond * quantile(z, tau)`` at
  ``LGBM_TAUS``; ``mu_hat = sum(r)/(n+50)`` — the trailing mean shrunk toward
  zero by a 50-observation pseudo-prior. CRPS via ``crps_from_quantiles``;
  pinball at ``TAUS`` via ``qf_at``.

Honest-NaN policy: a row is NaN when fewer than 50% of the window bars have a
valid volume surprise, when the recent-surprise tail has no valid ``vs``, or
when fewer than 30 standardized residuals are finite.

Binds to the shard via ``bars_sha256`` + config fields exactly like
``_challenger_col.py`` so ``splice_challenger_column.py`` can verify row
alignment by construction.
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
    ewma_next_sigma,
    qf_at,
)

from quant_fund.metrics.scoring import pinball_loss  # noqa: E402
from quant_fund.research.sota_evidence import validate_bars  # noqa: E402

MODEL = "dip_volm"

# Fixed, disclosed hyperparameters (not tuned per origin).
VOL_BASE_WIN = 20  # trailing-median baseline window, inclusive of bar t
VOL_S_WIN = 10  # trailing window positions entering the recent surprise S
VOL_S_LAM = 0.7  # EWMA decay for S
EWMA_LAM = 0.94  # RiskMetrics lambda for base sigma
GAMMA_FIXED = 0.5  # prior elasticity of sigma wrt volume surprise
GAMMA_BLEND = 0.5  # weight on the per-origin regression estimate
MULT_LO, MULT_HI = 0.6, 1.8  # clip bounds on S**gamma_hat
MIN_Z = 30  # minimum finite standardized residuals
MIN_REG = 30  # minimum pairs for the gamma regression
MIN_VOL_COVER = 0.5  # minimum fraction of window bars with valid vs
MU_PSEUDO = 50.0  # shrinkage pseudo-count pulling the trailing mean to 0


def _volume_surprise(vols: np.ndarray) -> np.ndarray:
    """vs_j = v_j / median(v_{j-19..j}); NaN where undefined or v_j invalid."""
    n = vols.size
    vs = np.full(n, np.nan)
    ok = np.isfinite(vols) & (vols > 0.0)
    for j in range(n):
        if not ok[j]:
            continue
        lo = max(0, j - VOL_BASE_WIN + 1)
        base = vols[lo : j + 1][ok[lo : j + 1]]
        if base.size < 5:
            continue
        med = float(np.median(base))
        if med > 0.0:
            vs[j] = float(vols[j]) / med
    return vs


def _ewma_sigma_path(rets: np.ndarray, lam: float = EWMA_LAM) -> np.ndarray:
    """Per-bar EWMA forecast sigmas: ``sig[t]`` forecasts bar t from data < t.

    Mirrors ``ewma_next_sigma``: the variance state after absorbing ``r_t`` is
    the forecast for bar t+1, so ``sig[t]`` is the sqrt of the state before
    ``r_t`` is absorbed. ``sig[0]`` is NaN (no prior bar).
    """
    n = rets.size
    sig = np.full(n, np.nan)
    var = float(rets[0] ** 2)
    for t in range(1, n):
        sig[t] = np.sqrt(var)
        var = lam * var + (1.0 - lam) * float(rets[t]) * float(rets[t])
    return sig


def _volm_quantiles(
    rets_long: np.ndarray, vs_long: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    """Quantile vector at LGBM_TAUS plus diagnostics (NaN vector on failure)."""
    nan = np.full(LGBM_TAUS.size, np.nan)
    diag = {
        "gamma_reg": np.nan,
        "gamma_hat": GAMMA_FIXED,
        "vol_cover": 0.0,
        "surprise": np.nan,
        "mult": np.nan,
        "n_z": 0.0,
    }
    n = rets_long.size
    if n < MIN_Z + 1:
        return nan, diag
    vs_ok = np.isfinite(vs_long) & (vs_long > 0.0)
    cover = float(vs_ok.mean())
    diag["vol_cover"] = cover
    if cover < MIN_VOL_COVER:
        return nan, diag

    # Recent volume surprise S: lam-weighted mean of valid vs in the tail.
    tail_vs = vs_long[-VOL_S_WIN:]
    tail_w = VOL_S_LAM ** np.arange(tail_vs.size - 1, -1, -1.0)
    tail_ok = np.isfinite(tail_vs) & (tail_vs > 0.0)
    if not tail_ok.any():
        return nan, diag
    s_val = float(np.sum(tail_w[tail_ok] * tail_vs[tail_ok]) / np.sum(tail_w[tail_ok]))
    diag["surprise"] = s_val

    # Elasticity regression: slope of log|r| on log(vs), clipped to [0,1].
    r_ok = np.isfinite(rets_long) & (rets_long != 0.0)
    reg_ok = vs_ok & r_ok
    gamma_hat = GAMMA_FIXED
    if int(reg_ok.sum()) >= MIN_REG:
        lx = np.log(vs_long[reg_ok])
        ly = np.log(np.abs(rets_long[reg_ok]))
        if float(np.var(lx)) > 0.0:
            slope = float(np.cov(lx, ly, ddof=0)[0, 1] / np.var(lx))
            if np.isfinite(slope):
                gamma_reg = float(np.clip(slope, 0.0, 1.0))
                diag["gamma_reg"] = gamma_reg
                gamma_hat = GAMMA_BLEND * gamma_reg + (1.0 - GAMMA_BLEND) * GAMMA_FIXED
    diag["gamma_hat"] = gamma_hat

    # Standardized residuals under the volume-conditioned EWMA sigma.
    sig = _ewma_sigma_path(rets_long)
    z_ok = vs_ok & np.isfinite(rets_long) & np.isfinite(sig) & (sig > 0.0)
    z = rets_long[z_ok] / (sig[z_ok] * np.power(vs_long[z_ok], gamma_hat))
    z = z[np.isfinite(z)]
    diag["n_z"] = float(z.size)
    if z.size < MIN_Z:
        return nan, diag

    sigma_base = ewma_next_sigma(rets_long, lam=EWMA_LAM)
    mult = float(np.clip(s_val**gamma_hat, MULT_LO, MULT_HI))
    diag["mult"] = mult
    sigma_cond = sigma_base * mult
    if not np.isfinite(sigma_cond) or sigma_cond <= 0.0:
        return nan, diag
    mu_hat = float(np.sum(rets_long) / (n + MU_PSEUDO))
    q = mu_hat + sigma_cond * np.quantile(z, LGBM_TAUS)
    return np.maximum.accumulate(q), diag


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


def compute_column(bars_path: Path, cfg: dict) -> tuple[dict[str, np.ndarray], dict]:
    """(crps[R], pin[R,T], target_time_ns[R]) over the shard's origin grid."""
    frame = pl.read_parquet(bars_path)
    event_times, interval = validate_bars(frame)
    closes = frame["close"].to_numpy().astype(float)
    vols = frame["volume"].to_numpy().astype(float)
    vs_all = _volume_surprise(vols)
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
    gamma_regs: list[float] = []
    gamma_hats: list[float] = []
    covers: list[float] = []
    n_below_cover = 0
    for row, i in enumerate(range(first_origin, n - 1)):
        closes_hist = closes[: i + 1]
        long_start = max(0, closes_hist.size - (garch_window + 1))
        rets_long = np.diff(closes_hist[long_start:]) / closes_hist[long_start:-1]
        # vs aligned to rets_long: return k spans bars long_start+k -> +1.
        vs_long = vs_all[long_start + 1 : i + 1]
        y = float(closes[i + 1] / closes[i] - 1.0)
        tt[row] = int(event_times[i + 1])
        try:
            q, diag = _volm_quantiles(rets_long, vs_long)
            covers.append(diag["vol_cover"])
            if diag["vol_cover"] < MIN_VOL_COVER:
                n_below_cover += 1
            if np.isfinite(diag["gamma_reg"]):
                gamma_regs.append(diag["gamma_reg"])
            gamma_hats.append(diag["gamma_hat"])
            if not np.isfinite(q).all():
                continue  # honest NaN row
            crps[row] = crps_from_quantiles(y, LGBM_TAUS, q)
            for k, tau in enumerate(TAUS):
                qk = qf_at(LGBM_TAUS, q, tau)
                if np.isfinite(qk):
                    pin[row, k] = float(pinball_loss(np.array([y]), np.array([qk]), tau)[0])
        except Exception:  # noqa: BLE001 - honest NaN, row stays disclosed
            pass
    stats = {
        "gamma_reg_mean": float(np.mean(gamma_regs)) if gamma_regs else None,
        "gamma_reg_median": float(np.median(gamma_regs)) if gamma_regs else None,
        "gamma_reg_min": float(np.min(gamma_regs)) if gamma_regs else None,
        "gamma_reg_max": float(np.max(gamma_regs)) if gamma_regs else None,
        "gamma_hat_mean": float(np.mean(gamma_hats)) if gamma_hats else None,
        "n_gamma_reg_feasible": len(gamma_regs),
        "vol_cover_mean": float(np.mean(covers)) if covers else None,
        "vol_cover_min": float(np.min(covers)) if covers else None,
        "n_below_vol_cover": n_below_cover,
    }
    return {
        "crps_col": crps,
        "pin_cols": pin,
        "target_time_ns": tt,
        "bar_interval_ns": np.asarray(interval, dtype=np.int64),
    }, stats


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
    out, stats = compute_column(bars, cfg)
    if out["crps_col"].shape[0] != n_rows:
        raise ValueError(f"{args.shard.name}: grid mismatch {out['crps_col'].shape[0]} != {n_rows}")
    meta_out = {
        "tool": Path(__file__).name,
        "model": MODEL,
        "shard": args.shard.name,
        "shard_sha256": _sha256(args.shard),
        "bars": bars.name,
        "bars_sha256": meta["bars_sha256"],
        "bars_file_sha256": _sha256(bars),
        "asset_names": meta.get("asset_names"),
        "config": {
            k: cfg.get(k)
            for k in ("origins_per_asset", "lookback", "window", "garch_window", "taus", "seed")
        },
        "volm_params": {
            "vol_base_win": VOL_BASE_WIN,
            "vol_s_win": VOL_S_WIN,
            "vol_s_lam": VOL_S_LAM,
            "ewma_lam": EWMA_LAM,
            "gamma_fixed": GAMMA_FIXED,
            "gamma_blend": GAMMA_BLEND,
            "mult_clip": [MULT_LO, MULT_HI],
            "min_z": MIN_Z,
            "min_vol_cover": MIN_VOL_COVER,
            "mu_pseudo": MU_PSEUDO,
        },
        "volm_stats": stats,
        "n_rows": n_rows,
        "n_finite_crps": int(np.isfinite(out["crps_col"]).sum()),
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
