"""Compute the dip_xbeta challenger column over an existing shard's origin grid.

``dip_xbeta`` is the first cross-asset challenger: it conditions the target
asset's one-step-ahead return distribution on a covariate asset's trailing
returns (BTCUSDT for every target; ETHUSDT when the target itself is BTCUSDT).

Causal, deterministic model (no RNG). At origin ``i`` for target asset A:

1. Trailing simple returns ``r_A`` over ``garch_window`` bars (same slice as
   the univariate challengers). The covariate file is inner-joined on
   ``event_time``; the covariate return aligned to a target bar is the
   covariate close-to-close return ENDING at that bar's event_time. If fewer
   than 90% of the window's target timestamps have a covariate match, the row
   is left NaN (honest failure, never fabricated).
2. Rolling OLS beta of ``r_A`` on ``r_M`` over the trailing 250 aligned
   observations (minimum 60): ``beta = cov(r_A, r_M) / var(r_M)``; a
   degenerate zero-variance covariate window yields ``beta = 0``.
3. Covariate state: ``sigma_M = ewma_next_sigma(r_M, lam=0.94)`` over the full
   aligned window; ``sigma_idio = ewma_next_sigma(eps, 0.94)`` on residuals
   ``eps = r_A - alpha - beta * r_M`` with OLS intercept
   ``alpha = mean(r_A) - beta * mean(r_M)`` from the beta window.
4. ``sigma = sqrt(max(0.15, W_MKT) * (beta * sigma_M)^2 + W_IDIO * sigma_idio^2)``
   with ``W_MKT = 0.6``, ``W_IDIO = 0.4``. The mean is the OLS-implied
   conditional mean ``alpha + beta * E[r_M]`` shrunk 50% toward zero.
5. Quantiles on ``LGBM_TAUS``: primary path is a Student-t with df fitted on
   the residuals by ``_fit_student_t`` (clipped to [3, 30]) and unit-variance
   scaling ``sqrt((nu-2)/nu)`` — the same convention as ``dip_garch_t``. When
   the t-fit is non-finite the fallback is empirical quantiles of the
   standardized residuals ``eps / sigma_idio`` rescaled by ``sigma``. The
   t-path is finite essentially everywhere; the empirical fallback exists so a
   row is NaN only when both paths fail (documented choice: whichever path is
   finite, t first).
6. CRPS via ``crps_from_quantiles`` on the ``LGBM_TAUS`` grid (identical to
   the ``dip_qar`` scoring path); pinball at ``TAUS`` via ``qf_at``.

Replicates the origin loop of ``sota_eval_kronos.py`` (first_origin formula,
window/garch-window slicing, close-to-close target). Output binds to the
shard via ``bars_sha256`` + config fields; the covariate file is disclosed via
``aux_bars`` / ``aux_bars_sha256``.
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
    _fit_student_t,
    _sha256,
    crps_from_quantiles,
    ewma_next_sigma,
    qf_at,
)

from quant_fund.metrics.scoring import pinball_loss  # noqa: E402
from quant_fund.research.sota_evidence import validate_bars  # noqa: E402

MODEL = "dip_xbeta"
BETA_WINDOW = 250  # trailing aligned observations for the OLS beta
BETA_MIN = 60  # minimum aligned observations required for a beta
EWMA_LAM = 0.94
W_MKT = 0.6
W_IDIO = 0.4
MEAN_SHRINK = 0.5  # shrink the conditional mean toward zero
MIN_OVERLAP = 0.90  # min fraction of window timestamps matched to covariate
DF_MIN, DF_MAX = 3.0, 30.0


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


def _covariate_name(bars_name: str, target_asset: str) -> str:
    """Covariate deep-file name: ETHUSDT for BTCUSDT targets, else BTCUSDT."""
    base = target_asset.split("-", 1)[0].upper()
    cov_asset = "ethusdt" if base == "BTCUSDT" else "btcusdt"
    suffix = bars_name.split("_", 1)[1]  # e.g. "1d_deep.parquet", "4h_deep.parquet"
    return f"{cov_asset}_{suffix}"


def _load_covariate_returns(
    bars_root: Path, cov_name: str, interval: int
) -> tuple[dict[int, float], Path | None, bool]:
    """{end_bar_event_time_ns: covariate simple return} for the covariate file.

    Returns (mapping, path, ok). ``ok`` is False when the file is missing,
    fails validation, or has a different bar interval — the caller then leaves
    every row NaN (honest failure).
    """
    path = bars_root / cov_name
    if not path.is_file():
        return {}, path, False
    try:
        frame = pl.read_parquet(path)
        times, cov_interval = validate_bars(frame)
    except Exception:  # noqa: BLE001 - dishonest to guess; report all-NaN
        return {}, path, False
    if cov_interval != interval:
        return {}, path, False
    closes = frame["close"].to_numpy().astype(float)
    rets = np.diff(closes) / closes[:-1]
    return {int(t): float(r) for t, r in zip(times[1:], rets)}, path, True


def _xbeta_quantiles(
    ra: np.ndarray, rm: np.ndarray
) -> tuple[float, np.ndarray, str]:
    """(mu, quantiles@LGBM_TAUS, path) for one origin's aligned windows."""
    L = ra.size
    W = min(BETA_WINDOW, L)
    if W < BETA_MIN:
        raise ValueError(f"aligned window {W} < {BETA_MIN}")
    ra_w, rm_w = ra[-W:], rm[-W:]
    rm_c = rm_w - float(np.mean(rm_w))
    denom = float(rm_c @ rm_c)
    beta = float((ra_w - float(np.mean(ra_w))) @ rm_c / denom) if denom > 0 else 0.0
    alpha = float(np.mean(ra_w)) - beta * float(np.mean(rm_w))
    mu = MEAN_SHRINK * (alpha + beta * float(np.mean(rm_w)))
    eps = ra - (alpha + beta * rm)
    sigma_m = ewma_next_sigma(rm, EWMA_LAM)
    sigma_idio = ewma_next_sigma(eps, EWMA_LAM)
    sigma = float(
        np.sqrt(max(0.15, W_MKT) * (beta * sigma_m) ** 2 + W_IDIO * sigma_idio**2)
    )
    if not np.isfinite(sigma) or sigma <= 0.0 or not np.isfinite(mu):
        raise ValueError("degenerate sigma/mu")
    # Primary: Student-t with df fitted on residuals (clipped [3, 30]).
    nu, _loc, _scale = _fit_student_t(eps)
    if np.isfinite(nu):
        nu = float(np.clip(nu, DF_MIN, DF_MAX))
        q = mu + sigma * np.sqrt((nu - 2.0) / nu) * st.t.ppf(LGBM_TAUS, nu)
        if np.isfinite(q).all():
            return mu, np.maximum.accumulate(q), "t"
    # Fallback: empirical quantiles of standardized residuals rescaled.
    z = eps / sigma_idio if sigma_idio > 0.0 else np.zeros_like(eps)
    q = mu + sigma * np.quantile(z[np.isfinite(z)], LGBM_TAUS)
    if np.isfinite(q).all():
        return mu, np.maximum.accumulate(q), "emp"
    raise ValueError("no finite quantile path")


def compute_column(
    bars_path: Path, aux_path: Path | None, cov_ret_by_time: dict[int, float], cfg: dict
) -> tuple[dict[str, np.ndarray], dict[str, float | int | str]]:
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
    full_rets = np.diff(closes) / closes[:-1]  # full_rets[k] ends at bar k+1
    # Covariate return aligned to each target bar's event_time (NaN if absent).
    cov_at = np.full(n, np.nan)
    for j in range(1, n):
        r = cov_ret_by_time.get(int(event_times[j]))
        if r is not None:
            cov_at[j] = r
    stats = {"n_overlap_fail": 0, "n_short_window": 0, "n_path_t": 0,
             "n_path_emp": 0, "min_overlap": np.nan, "mean_overlap": np.nan}
    overlaps: list[float] = []
    for row, i in enumerate(range(first_origin, n - 1)):
        long_start = max(0, i - garch_window)  # returns slice [long_start, i)
        ra_all = full_rets[long_start:i]
        rm_all = cov_at[long_start + 1 : i + 1]
        y = float(closes[i + 1] / closes[i] - 1.0)
        tt[row] = int(event_times[i + 1])
        try:
            mask = np.isfinite(rm_all) & np.isfinite(ra_all)
            overlap = float(mask.mean()) if mask.size else 0.0
            overlaps.append(overlap)
            if overlap < MIN_OVERLAP:
                stats["n_overlap_fail"] += 1
                continue  # honest NaN: covariate misaligned this origin
            ra, rm = ra_all[mask], rm_all[mask]
            if min(BETA_WINDOW, ra.size) < BETA_MIN:
                stats["n_short_window"] += 1
                continue
            _mu, q_grid, path = _xbeta_quantiles(ra, rm)
            stats[f"n_path_{path}"] += 1
            crps[row] = crps_from_quantiles(y, LGBM_TAUS, q_grid)
            for k, tau in enumerate(TAUS):
                qk = qf_at(LGBM_TAUS, q_grid, tau)
                if np.isfinite(qk):
                    pin[row, k] = float(
                        pinball_loss(np.array([y]), np.array([qk]), tau)[0]
                    )
        except Exception:  # noqa: BLE001 - honest NaN, row stays disclosed
            continue
    if overlaps:
        stats["min_overlap"] = float(np.min(overlaps))
        stats["mean_overlap"] = float(np.mean(overlaps))
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
    target_asset = str(meta.get("asset_names", [""])[0])
    cov_name = _covariate_name(bars.name, target_asset)
    # Interval is read from the target file inside compute_column; load the
    # covariate against it by first validating the target frame here.
    tgt_frame = pl.read_parquet(bars)
    _tgt_times, tgt_interval = validate_bars(tgt_frame)
    cov_ret_by_time, aux_path, aux_ok = _load_covariate_returns(
        args.bars_root, cov_name, tgt_interval
    )

    t0 = time.time()
    out, stats = compute_column(bars, aux_path, cov_ret_by_time, cfg)
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
        "aux_bars": aux_path.name if aux_path is not None else cov_name,
        "aux_bars_sha256": _sha256(aux_path) if aux_path is not None and aux_path.is_file() else None,
        "aux_ok": aux_ok,
        "asset_names": meta.get("asset_names"),
        "config": {k: cfg.get(k) for k in
                   ("origins_per_asset", "lookback", "window", "garch_window",
                    "taus", "seed")},
        "xbeta": {
            "beta_window": BETA_WINDOW,
            "beta_min": BETA_MIN,
            "ewma_lam": EWMA_LAM,
            "w_mkt": W_MKT,
            "w_idio": W_IDIO,
            "mean_shrink": MEAN_SHRINK,
            "min_overlap": MIN_OVERLAP,
            "df_clip": [DF_MIN, DF_MAX],
            "quantile_path": "student-t df on residuals, empirical-std fallback",
            **stats,
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
        f"aux={meta_out['aux_bars']} ok={aux_ok}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
