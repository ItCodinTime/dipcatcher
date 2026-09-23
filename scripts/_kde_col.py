"""Compute the ``dip_kde`` challenger column over an existing shard's origin grid.

Replicates the origin loop of ``sota_eval_kronos.py`` (first_origin formula,
window/garch-window slicing, close-to-close target) but only evaluates one
deterministic challenger — no target models, no other challengers. Output binds
to the shard via ``bars_sha256`` + config fields so the splice tool can verify
row alignment by construction (the grid is deterministic given the same bars
bytes and the same config).

``dip_kde`` is a Gaussian kernel-density challenger on the trailing
``garch_window`` returns. Bandwidth is the Silverman rule-of-thumb scaled by an
EWMA vol ratio: ``h = 0.9 * min(s, IQR/1.34) * n^{-1/5} *
clip(recent_vol/long_vol, 0.7, 1.4)`` where ``recent_vol`` is the EWMA
next-bar sigma over the last ``KDE_VOL_RECENT`` returns and ``long_vol`` is the
same recursion over the full window — the bandwidth widens when recent vol
exceeds long-run vol (adaptive smoothing), capped both directions.

Quantiles are exact numerical inversion of the KDE CDF
``F(x) = mean_i Phi((x - r_i)/h)`` on a deterministic grid spanning
``[min(r) - 4h, max(r) + 4h]``; linear interpolation of the monotone CDF gives
quantiles at ``LGBM_TAUS`` (which contains ``TAUS``) and at ``KDE_DENSE``
equiprobable midpoints.

CRPS is ``crps_empirical(y, q_dense)`` — the 512 equiprobable KDE quantiles
treated as an ensemble sample, matching the ``dip_skt``/``dip_conf_t``/
``dip_regime`` convention in this arena (chosen over ``crps_from_quantiles`` so
the whole KDE shape enters the score, not a trapezoid on a coarse grid).
Fully deterministic: fixed grid, no RNG anywhere. Edge cases: ``n < 30`` or
``h <= 0`` → honest NaN row.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sota_eval_kronos import (  # noqa: E402
    LGBM_TAUS,
    TAUS,
    _sha256,
    ewma_next_sigma,
)

from quant_fund.metrics.scoring import (  # noqa: E402
    crps_empirical,
    pinball_loss,
)
from quant_fund.research.sota_evidence import validate_bars  # noqa: E402

MODEL = "dip_kde"

KDE_MIN_N = 30  # honest NaN below this many trailing returns
KDE_GRID = 4097  # deterministic CDF grid points over [min-4h, max+4h]
KDE_DENSE = 512  # equiprobable quantile levels feeding crps_empirical
KDE_VOL_RECENT = 60  # trailing returns for the recent EWMA sigma
KDE_VOL_CLIP = (0.7, 1.4)  # clip bounds for recent/long vol ratio
KDE_TAIL_H = 4.0  # grid padding in units of h beyond the data range


def _kde_bandwidth(rets: np.ndarray) -> tuple[float, float]:
    """(h, vol_ratio) — Silverman bandwidth times clipped EWMA vol ratio."""
    n = rets.size
    s = float(np.std(rets, ddof=1))
    q25, q75 = np.quantile(rets, [0.25, 0.75])
    iqr = float(q75 - q25)
    base = 0.9 * min(s, iqr / 1.34) * n ** (-1.0 / 5.0)
    long_vol = ewma_next_sigma(rets)
    recent = rets[-KDE_VOL_RECENT:] if n > KDE_VOL_RECENT else rets
    recent_vol = ewma_next_sigma(recent)
    ratio = (
        float(np.clip(recent_vol / long_vol, *KDE_VOL_CLIP))
        if long_vol > 0.0
        else 1.0
    )
    return base * ratio, ratio


def _kde_quantiles(rets: np.ndarray, h: float, levels: np.ndarray) -> np.ndarray:
    """Invert the Gaussian-KDE CDF at ``levels`` on a deterministic x-grid."""
    lo = float(rets.min()) - KDE_TAIL_H * h
    hi = float(rets.max()) + KDE_TAIL_H * h
    xg = np.linspace(lo, hi, KDE_GRID)
    cdf = st.norm.cdf((xg[:, None] - rets[None, :]) / h).mean(axis=1)
    # Mean of strictly increasing CDFs is increasing; accumulate guards
    # against floating-point flat spots in the extreme tails so interp is
    # well-defined on a non-decreasing support.
    cdf = np.maximum.accumulate(cdf)
    return np.interp(levels, cdf, xg)


def score_origin(rets_long: np.ndarray, y: float) -> tuple[float, np.ndarray]:
    """(crps, quantiles@TAUS) for dip_kde at one origin; NaN on degenerate input."""
    n = rets_long.size
    if n < KDE_MIN_N:
        return float("nan"), np.full(len(TAUS), np.nan)
    h, _ratio = _kde_bandwidth(rets_long)
    if not np.isfinite(h) or h <= 0.0:
        return float("nan"), np.full(len(TAUS), np.nan)
    dense_levels = (np.arange(KDE_DENSE) + 0.5) / KDE_DENSE
    levels = np.unique(np.concatenate([LGBM_TAUS, dense_levels]))
    q_all = _kde_quantiles(rets_long, h, levels)
    q_dense = np.interp(dense_levels, levels, q_all)
    q_taus = np.interp(np.asarray(TAUS), levels, q_all)
    return crps_empirical(y, q_dense), q_taus


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
    bw = np.full(n - 1 - first_origin, np.nan)
    vr = np.full(n - 1 - first_origin, np.nan)
    for row, i in enumerate(range(first_origin, n - 1)):
        closes_hist = closes[: i + 1]
        long_start = max(0, closes_hist.size - (garch_window + 1))
        rets_long = np.diff(closes_hist[long_start:]) / closes_hist[long_start:-1]
        y = float(closes[i + 1] / closes[i] - 1.0)
        tt[row] = int(event_times[i + 1])
        try:
            if rets_long.size >= KDE_MIN_N:
                h, ratio = _kde_bandwidth(rets_long)
                if np.isfinite(h) and h > 0.0:
                    bw[row], vr[row] = h, ratio
            c, q = score_origin(rets_long, y)
            crps[row] = c
            for k, tau in enumerate(TAUS):
                if np.isfinite(q[k]):
                    pin[row, k] = float(
                        pinball_loss(np.array([y]), np.array([q[k]]), tau)[0]
                    )
        except Exception:  # noqa: BLE001 - honest NaN, row stays disclosed
            pass
    return {
        "crps_col": crps,
        "pin_cols": pin,
        "target_time_ns": tt,
        "bar_interval_ns": np.asarray(interval, dtype=np.int64),
        "bandwidth": bw,
        "vol_ratio": vr,
    }


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
    bw = out["bandwidth"]
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
        "kde": {
            "kernel": "gaussian",
            "bandwidth": "0.9*min(s,IQR/1.34)*n^-1/5*clip(recent_vol/long_vol,"
                         f"{KDE_VOL_CLIP[0]},{KDE_VOL_CLIP[1]})",
            "vol_windows": {"recent": KDE_VOL_RECENT, "long": "garch_window"},
            "grid_points": KDE_GRID,
            "grid_span": f"data_range +/- {KDE_TAIL_H}h",
            "dense_levels": KDE_DENSE,
            "crps": "crps_empirical(y, 512 equiprobable KDE quantiles)",
            "min_n": KDE_MIN_N,
            "h_min": float(np.nanmin(bw)) if np.isfinite(bw).any() else None,
            "h_mean": float(np.nanmean(bw)) if np.isfinite(bw).any() else None,
            "h_max": float(np.nanmax(bw)) if np.isfinite(bw).any() else None,
            "vol_ratio_mean": float(np.nanmean(out["vol_ratio"]))
            if np.isfinite(out["vol_ratio"]).any()
            else None,
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
        f"finite={meta_out['n_finite_crps']} mean_crps={np.nanmean(out['crps_col']):.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
