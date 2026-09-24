"""Compute the ``dip_mid`` challenger column over a shard's origin grid.

Replicates the origin loop of ``sota_eval_kronos.py`` (first_origin formula,
garch-window slicing, close-to-close target) but only evaluates one
deterministic challenger — no target models, no other challengers. Output binds
to the shard via ``bars_sha256`` + config fields so the splice tool can verify
row alignment by construction (the grid is deterministic given the same bars
bytes and the same config).

``dip_mid`` is a GARCH-MIDAS-style mixed-frequency challenger
(Engle-Ghysels-Soja): the 1-step conditional vol is a fast RiskMetrics EWMA
tilted by a coarse-scale realized-vol regime ratio, and the shape is the
filtered-historical-simulation empirical quantile function of EWMA-standardized
residuals.

Per origin, over the trailing ``garch_window`` returns ``r_t``:

* fast component — ``sigma_short_t`` is the RiskMetrics EWMA forecast
  (``ewma_next_sigma``, lam=0.94) of the trailing ``MID_FAST_BARS``=10 returns
  ending at t-1, so the weight mass sits on ~10 bars. ``sigma_short`` is the
  same forecast for the next bar. Standardized residuals ``z_t = r_t /
  sigma_short_t`` (FHS-style).
* long component — the last ``K * bars_per_coarse`` returns are grouped into
  ``K`` consecutive blocks of ``bars_per_coarse`` bars (leading remainder
  dropped, so the final block ends at the origin). ``bars_per_coarse`` is
  ``round(1 day / bar_interval)`` for sub-daily bars (6 for 4h) and 5 for
  daily bars (a trading week); consecutive-index bucketing is used instead of
  calendar grouping (bars are gap-free per ``validate_bars``, so the two are
  equivalent up to holiday alignment). The coarse level ``c_j`` is the per-bar
  RMS of block j; ``sigma_long_term`` = equal-weight mean of the last
  ``MID_COARSE_MEAN``=30 blocks and ``sigma_coarse_longrun`` = mean over all
  ``K`` blocks (its own long-run mean).
* ``sigma_cond = sigma_short * clip(sigma_long_term / sigma_coarse_longrun,
  MID_TILT_LO, MID_TILT_HI)`` — a regime tilt of the fast vol, not a level
  replacement.
* ``mu = 0.5 * mean(rets_long)`` (50% shrink toward 0);
  ``q(levels) = mu + sigma_cond * quantile(z, levels)`` emitted at
  ``LGBM_TAUS``; CRPS via ``crps_from_quantiles``; TAUS quantiles read off the
  same quantile function via ``qf_at``.

Honest NaN row when fewer than ``MIN_Z``=30 finite z, the fast sigma is
degenerate, or fewer than ``MID_COARSE_MEAN`` coarse blocks exist. Tilt-clip
hit rates are recorded in the artifact meta.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

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

MODEL = "dip_mid"

MID_LAM = 0.94  # RiskMetrics decay for the fast component
MID_FAST_BARS = 10  # trailing bars feeding the fast EWMA
MID_COARSE_MEAN = 30  # coarse blocks in the current long-run level (~30 days/wks)
MID_TILT_LO = 0.7
MID_TILT_HI = 1.5
MIN_Z = 30  # minimum finite standardized residuals for an honest row
DAY_NS = 86_400_000_000_000


def _bars_per_coarse(interval_ns: int) -> int:
    """Coarse-aggregation block size for the MIDAS filter.

    Sub-daily bars aggregate to one day; daily bars aggregate to a 5-bar
    trading week; supra-daily bars fall back to a 4-bar block.
    """
    if interval_ns < DAY_NS:
        return max(2, int(round(DAY_NS / interval_ns)))
    if interval_ns == DAY_NS:
        return 5
    return 4


def _fast_sigma_path(rets: np.ndarray) -> np.ndarray:
    """EWMA(lam) next-bar sigma forecast at every bar index.

    ``out[t]`` forecasts bar ``t`` from ``rets[max(0, t-MID_FAST_BARS):t]``;
    ``out[0]`` is NaN (no trailing return). ``out[W]`` would be the next-bar
    forecast — returned separately via ``_ewma_trailing``.
    """
    w = rets.size
    sq = rets * rets
    var = np.full(w, np.nan)
    # Bars t >= MID_FAST_BARS: fixed-length trailing window, vectorized.
    # ewma_next_sigma on a length-m slice gives var = lam^(m-1) s0^2 +
    # (1-lam) * sum_j lam^(m-1-j) s_j^2; with m=MID_FAST_BARS:
    m = MID_FAST_BARS
    w_full = np.empty(m)
    w_full[0] = MID_LAM ** (m - 1)
    w_full[1:] = (1.0 - MID_LAM) * MID_LAM ** np.arange(m - 2, -1, -1.0)
    if w >= m:
        win = sliding_window_view(sq, m)  # win[u] -> forecast for bar u+m
        var[m:] = win[: w - m] @ w_full
        # small prefix bars t=1..m-1 use their full available history
        for t in range(1, min(m, w)):
            var[t] = _ewma_var(rets[:t])
    else:
        for t in range(1, w):
            var[t] = _ewma_var(rets[:t])
    return np.sqrt(var)


def _ewma_var(rets: np.ndarray) -> float:
    """Squared ``ewma_next_sigma`` without recomputing the sqrt."""
    s = ewma_next_sigma(rets, lam=MID_LAM)
    return s * s


def _mid_forecast(rets_long: np.ndarray, bars_per_coarse: int) -> tuple[np.ndarray, dict]:
    """(quantiles@LGBM_TAUS, stats) for one origin, or (nan_vector, stats)."""
    nan9 = np.full(LGBM_TAUS.size, np.nan)
    stats: dict[str, float] = {
        "tilt": np.nan,
        "tilt_clipped_lo": 0.0,
        "tilt_clipped_hi": 0.0,
        "n_z": 0.0,
        "n_coarse": 0.0,
    }
    w = rets_long.size
    bpc = int(bars_per_coarse)
    k = w // bpc
    if w < MID_FAST_BARS + 1 or k < MID_COARSE_MEAN:
        return nan9, stats
    sig_path = _fast_sigma_path(rets_long)
    sigma_short = float(ewma_next_sigma(rets_long[-MID_FAST_BARS:], lam=MID_LAM))
    z = rets_long[1:] / sig_path[1:]
    z = z[np.isfinite(z)]
    stats["n_z"] = float(z.size)
    if z.size < MIN_Z or not np.isfinite(sigma_short) or sigma_short <= 0.0:
        return nan9, stats
    # Coarse level: per-bar RMS of each consecutive bpc-bar block, aligned so
    # the last block ends at the window end (leading remainder dropped).
    blocks = rets_long[w - k * bpc :].reshape(k, bpc)
    coarse = np.sqrt(np.mean(blocks * blocks, axis=1))
    coarse = coarse[np.isfinite(coarse)]
    stats["n_coarse"] = float(coarse.size)
    if coarse.size < MID_COARSE_MEAN:
        return nan9, stats
    sigma_long_term = float(np.mean(coarse[-MID_COARSE_MEAN:]))
    sigma_longrun = float(np.mean(coarse))
    if not np.isfinite(sigma_longrun) or sigma_longrun <= 0.0:
        return nan9, stats
    tilt_raw = sigma_long_term / sigma_longrun
    tilt = float(np.clip(tilt_raw, MID_TILT_LO, MID_TILT_HI))
    stats["tilt"] = tilt
    stats["tilt_clipped_lo"] = float(tilt_raw <= MID_TILT_LO)
    stats["tilt_clipped_hi"] = float(tilt_raw >= MID_TILT_HI)
    sigma_cond = sigma_short * tilt
    if not np.isfinite(sigma_cond) or sigma_cond <= 0.0:
        return nan9, stats
    mu = 0.5 * float(np.mean(rets_long))
    return mu + sigma_cond * np.quantile(z, LGBM_TAUS), stats


def compute_column(bars_path: Path, cfg: dict) -> dict[str, np.ndarray]:
    """(crps[R], pin[R,T], target_time_ns[R]) over the shard's origin grid."""
    frame = pl.read_parquet(bars_path)
    event_times, interval = validate_bars(frame)
    closes = frame["close"].to_numpy().astype(float)
    n = closes.size
    origins = int(cfg["origins_per_asset"])
    lookback = int(cfg["lookback"])
    window = int(cfg["window"])
    garch_window = int(cfg["garch_window"])
    first_origin = max(lookback, window + 1, n - origins - 1)
    if first_origin >= n - 1:
        raise ValueError(f"{bars_path.name}: not enough bars ({n})")
    n_rows = n - 1 - first_origin
    crps = np.full(n_rows, np.nan)
    pin = np.full((n_rows, len(TAUS)), np.nan)
    tt = np.empty(n_rows, dtype=np.int64)
    bpc = _bars_per_coarse(int(interval))
    tilts = np.full(n_rows, np.nan)
    clip_lo = np.zeros(n_rows, dtype=np.int64)
    clip_hi = np.zeros(n_rows, dtype=np.int64)
    for row, i in enumerate(range(first_origin, n - 1)):
        closes_hist = closes[: i + 1]
        long_start = max(0, closes_hist.size - (garch_window + 1))
        rets_long = np.diff(closes_hist[long_start:]) / closes_hist[long_start:-1]
        y = float(closes[i + 1] / closes[i] - 1.0)
        tt[row] = int(event_times[i + 1])
        try:
            q9, stats = _mid_forecast(rets_long, bpc)
            tilts[row] = stats["tilt"]
            clip_lo[row] = int(stats["tilt_clipped_lo"])
            clip_hi[row] = int(stats["tilt_clipped_hi"])
            if not np.isfinite(q9).all():
                continue  # honest NaN row
            crps[row] = crps_from_quantiles(y, LGBM_TAUS, q9)
            for k, tau in enumerate(TAUS):
                q_k = qf_at(LGBM_TAUS, q9, tau)
                if np.isfinite(q_k):
                    pin[row, k] = float(pinball_loss(np.array([y]), np.array([q_k]), tau)[0])
        except Exception:  # noqa: BLE001 - honest NaN, row stays disclosed
            continue
    return {
        "crps_col": crps,
        "pin_cols": pin,
        "target_time_ns": tt,
        "bar_interval_ns": np.asarray(interval, dtype=np.int64),
        "tilt": tilts,
        "n_tilt_clip_lo": np.asarray(clip_lo.sum(), dtype=np.int64),
        "n_tilt_clip_hi": np.asarray(clip_hi.sum(), dtype=np.int64),
    }


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
        raise ValueError(f"{args.shard.name}: grid mismatch {out['crps_col'].shape[0]} != {n_rows}")
    finite_tilt = np.isfinite(out["tilt"])
    meta_out = {
        "tool": Path(__file__).name,
        "model": MODEL,
        "mid_spec": (
            "sigma_cond = EWMA(lam=0.94,last 10 bars) * clip(mean(coarse RMS, "
            "last 30 blocks)/mean(coarse RMS, all blocks), 0.7, 1.5); FHS "
            "quantiles of z_t = r_t/sigma_short_t; mu = 0.5*mean(rets_long)"
        ),
        "coarse_scheme": {
            "bars_per_coarse": int(_bars_per_coarse(int(out["bar_interval_ns"]))),
            "bar_interval_ns": int(out["bar_interval_ns"]),
            "grouping": "consecutive_bars",
            "coarse_mean_blocks": MID_COARSE_MEAN,
        },
        "n_tilt_clip_lo": int(out["n_tilt_clip_lo"]),
        "n_tilt_clip_hi": int(out["n_tilt_clip_hi"]),
        "frac_tilt_clip_lo": round(float(out["n_tilt_clip_lo"]) / n_rows, 4),
        "frac_tilt_clip_hi": round(float(out["n_tilt_clip_hi"]) / n_rows, 4),
        "mean_tilt": round(float(np.nanmean(out["tilt"])), 6) if finite_tilt.any() else None,
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
        f"finite={meta_out['n_finite_crps']} mean_crps={np.nanmean(out['crps_col']):.6f} "
        f"clip_lo={meta_out['n_tilt_clip_lo']} clip_hi={meta_out['n_tilt_clip_hi']} "
        f"mean_tilt={meta_out['mean_tilt']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
