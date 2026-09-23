"""Compute the ``dip_lgbm_qv`` challenger column over a shard's origin grid.

Feature-augmented LightGBM quantile regression: identical to ``dip_lgbm_q``
(same ``LGBM_TAUS`` output grid, same ``lgbm_quantiles`` trainer — same
hyperparameters, seed, and per-origin refit cadence) but the causal feature
table is extended with bar-microstructure columns the return-only model
ignores. All added columns at row ``j`` (the row that forecasts
``rets_all[j] = close[j+1]/close[j] - 1``) use only bars ``<= j``:

- ``park_vol``  : sqrt(mean Parkinson variance over the last 20 bars),
                  per-bar variance ``(ln high/low)^2 / (4 ln 2)``, bars j-19..j
- ``rng_ratio`` : ``(high - low) / close`` of the last bar (j)
- ``vol_surp``  : ``volume[j] / median(volume[j-19..j])``
- ``amihud``    : mean ``|r_k| / max(volume[k], 1e-12)`` over bars j-19..j,
                  ``r_k = close[k]/close[k-1] - 1`` (the same trailing
                  20-return window the base features use)
- ``gap``       : ``open[j] / close[j-1] - 1`` of the last bar

The first 8 columns are exactly ``_lgbm_features`` output, so ``dip_lgbm_qv``
is a strict superset of ``dip_lgbm_q``'s information set. Rows with any
non-finite feature are dropped from the fit by ``lgbm_quantiles``' own mask;
a LightGBM import/fit failure yields an honest NaN row (counted in meta).

Output binds to the shard via ``bars_sha256`` + protocol fields so
``splice_challenger_column.py`` can verify row alignment by construction.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sota_eval_kronos import (  # noqa: E402
    LGBM_TAUS,
    TAUS,
    _lgbm_features,
    _sha256,
    crps_from_quantiles,
    lgbm_quantiles,
    qf_at,
)

from quant_fund.metrics.scoring import pinball_loss  # noqa: E402
from quant_fund.research.sota_evidence import validate_bars  # noqa: E402

MODEL = "dip_lgbm_qv"

# Column order of the extended feature table: the 8 base columns are fixed by
# _lgbm_features; the 5 appended microstructure columns are declared here.
BASE_FEATURES = ("r1", "r2", "mu5", "mu20", "sd5", "sd20", "ew_vol", "dow")
EXTRA_FEATURES = ("park_vol", "rng_ratio", "vol_surp", "amihud", "gap")
FEATURES = BASE_FEATURES + EXTRA_FEATURES
AMIHUD_EPS = 1e-12
LOG4 = 4.0 * np.log(2.0)


def _lgbmqv_features(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    rets_all: np.ndarray,
    dow: np.ndarray,
) -> np.ndarray:
    """(n-1, 13) causal feature table: ``_lgbm_features`` columns plus the 5
    microstructure columns. Row ``j`` forecasts ``rets_all[j]``; every column
    at row ``j`` is computable from bars ``<= j`` (plus the known-ahead
    weekday ``dow[j]``). Warmup rows (``j < 20``) stay NaN, matching the base
    feature convention so the training-row mask is unchanged."""
    X = _lgbm_features(rets_all, dow)  # (n-1, 8)
    n = closes.size
    m = rets_all.size  # n - 1 rows; row j forecasts the j -> j+1 return
    F = np.full((m, len(EXTRA_FEATURES)), np.nan)

    # Per-bar causal series indexed by bar k (0..n-1).
    with np.errstate(divide="ignore", invalid="ignore"):
        park_k = np.log(highs / lows) ** 2 / LOG4
        rng_k = (highs - lows) / closes
        # r_k = close[k]/close[k-1] - 1 == rets_all[k-1] for k >= 1
        absr_k = np.full(n, np.nan)
        absr_k[1:] = np.abs(rets_all)
        amihud_k = absr_k / np.maximum(volumes, AMIHUD_EPS)
        gap_k = np.full(n, np.nan)
        gap_k[1:] = opens[1:] / closes[:-1] - 1.0

    for j in range(20, m):
        k0 = j - 19  # last-20-bars window is bars j-19..j inclusive
        pv = float(np.mean(park_k[k0 : j + 1]))
        med = float(np.median(volumes[k0 : j + 1]))
        F[j] = (
            np.sqrt(pv) if pv >= 0.0 else np.nan,
            rng_k[j],
            volumes[j] / med if med > 0.0 else np.nan,
            float(np.mean(amihud_k[k0 : j + 1])),
            gap_k[j],
        )
    return np.concatenate([X, F], axis=1)


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
    """(crps[R], pin[R,T], target_time_ns[R], n_fit_failures) on the shard grid."""
    frame = pl.read_parquet(bars_path).sort("event_time")
    event_times, interval = validate_bars(frame)
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{bars_path}: missing columns {sorted(missing)}")
    closes = frame["close"].cast(pl.Float64).to_numpy().astype(float)
    opens = frame["open"].cast(pl.Float64).to_numpy().astype(float)
    highs = frame["high"].cast(pl.Float64).to_numpy().astype(float)
    lows = frame["low"].cast(pl.Float64).to_numpy().astype(float)
    volumes = frame["volume"].cast(pl.Float64).to_numpy().astype(float)

    n = closes.size
    origins = int(cfg["origins_per_asset"])
    lookback = int(cfg["lookback"])
    window = int(cfg["window"])
    garch_window = int(cfg["garch_window"])
    first_origin = max(lookback, window + 1, n - origins - 1)
    if first_origin >= n - 1:
        raise ValueError(f"{bars_path.name}: not enough bars ({n})")

    # Same indexing contract as the evaluator: rets_all[j] is the return from
    # bar j to bar j+1; dow[j] is the (known-ahead) weekday of target bar j+1.
    rets_all = np.diff(closes) / closes[:-1]
    dow = pd.to_datetime(event_times[1:], unit="ns", utc=True).dayofweek.to_numpy(
        dtype=float
    )
    XV = _lgbmqv_features(opens, highs, lows, closes, volumes, rets_all, dow)

    n_rows = n - 1 - first_origin
    crps = np.full(n_rows, np.nan)
    pin = np.full((n_rows, len(TAUS)), np.nan)
    tt = np.empty(n_rows, dtype=np.int64)
    n_fail = 0
    for row, i in enumerate(range(first_origin, n - 1)):
        y = float(closes[i + 1] / closes[i] - 1.0)
        tt[row] = int(event_times[i + 1])
        try:
            q_l = lgbm_quantiles(XV, rets_all, i, garch_window)
            crps[row] = crps_from_quantiles(y, LGBM_TAUS, q_l)
            for k, tau in enumerate(TAUS):
                qk = qf_at(LGBM_TAUS, q_l, tau)
                pin[row, k] = (
                    float(pinball_loss(np.array([y]), np.array([qk]), tau)[0])
                    if np.isfinite(qk)
                    else np.nan
                )
        except Exception:  # noqa: BLE001 - honest NaN, row stays disclosed
            n_fail += 1
    return {
        "crps_col": crps,
        "pin_cols": pin,
        "target_time_ns": tt,
        "bar_interval_ns": np.asarray(interval, dtype=np.int64),
        "n_fit_failures": np.asarray(n_fail, dtype=np.int64),
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
    if int(cfg.get("origins_per_asset", 0)) < 1 or MODEL in shard_names:
        raise ValueError(f"{args.shard.name}: already has {MODEL} or bad config")

    bars = _find_bars(args.bars_root, meta["bars_sha256"])
    t0 = time.time()
    out = compute_column(bars, cfg)
    if out["crps_col"].shape[0] != n_rows:
        raise ValueError(
            f"{args.shard.name}: grid mismatch {out['crps_col'].shape[0]} != {n_rows}"
        )
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
        "features": list(FEATURES),
        "n_rows": n_rows,
        "n_finite_crps": int(np.isfinite(out["crps_col"]).sum()),
        "n_fit_failures": int(out["n_fit_failures"]),
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
        f"finite={meta_out['n_finite_crps']} fails={meta_out['n_fit_failures']} "
        f"mean_crps={np.nanmean(out['crps_col']):.6f} "
        f"elapsed={meta_out['elapsed_s']}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
