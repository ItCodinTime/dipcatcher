"""Compute the ``dip_evt`` challenger column over a shard's origin grid.

Replicates the origin loop of ``sota_eval_kronos.py`` (first_origin formula,
garch-window slicing, close-to-close target) but only evaluates one
deterministic challenger — no target models, no other challengers. Output binds
to the shard via ``bars_sha256`` + config fields so the splice tool can verify
row alignment by construction (the grid is deterministic given the same bars
bytes and the same config).

``dip_evt`` is the EVT tail-augmented empirical challenger. The arena scores the
5% and 95% pinball levels (and the CRPS integral's tail mass) exactly where the
raw empirical CDF is noisiest, so the column replaces both empirical tails with
peaks-over-threshold Generalized Pareto fits (Pickands/Balkema-de Haan) — the
standard actuarial tail-quantile model — while keeping the empirical body.

Per origin, on the trailing ``garch_window`` returns ``r`` (causal,
deterministic):

1. Thresholds ``u_lo = q0.10(r)`` and ``u_hi = q0.90(r)``; exceedances are
   ``u_lo - r`` over ``r < u_lo`` (lower) and ``r - u_hi`` over ``r > u_hi``
   (upper), each with nominal tail mass 0.10.
2. GPD fit per tail by the closed-form method of moments on the exceedances
   (mean m, sample variance v): ``xi = 0.5*(1 - m^2/v)``,
   ``beta = 0.5*m*(m^2/v + 1)``, with ``xi`` clipped to [-0.5, 0.5]. MoM is
   chosen over ``scipy.stats.genpareto.fit`` MLE deliberately: it is
   deterministic, O(n), and has no optimizer failure modes — the documented
   estimator for this challenger.
3. Quantile function: empirical quantile for tau in [0.10, 0.90]; GPD tail
   outside, anchored so the function is continuous at the thresholds
   (``q(0.10) = u_lo``, ``q(0.90) = u_hi`` exactly — tail mass is fixed at the
   nominal 0.10 rather than the realized exceedance fraction, which is what
   "blend at boundary / match quantile at u" means here).
4. Volatility-aware rescale: all quantiles multiplied by
   ``s = clip(ewma_next_sigma(r) / std(r, ddof=1), 0.7, 1.4)`` (RiskMetrics
   lambda=0.94 recursion from ``sota_eval_kronos``), matching the other
   challengers' vol conditioning. ``s > 0`` preserves quantile monotonicity.

Scoring: CRPS is the 512-point midpoint-quantile copy scored by
``crps_empirical`` — the same convention ``dip_skt``/``dip_conf_t``/
``dip_regime`` use — so the fitted GPD tail shape enters the full integral
rather than being compressed into two quantile levels. Pinball at TAUS uses the
rescaled hybrid quantile function directly. Quantiles are also emitted at
``LGBM_TAUS`` for auditability.

Fallback: if either tail has fewer than ``EVT_MIN_EXCEEDANCES`` (8) exceedances
or produces an unstable fit (degenerate moments, non-finite/non-positive beta,
non-finite quantiles), the origin falls back to the plain empirical quantile
function — unscaled, same 512-copy scoring — and is disclosed in the fallback
counters. Unexpected errors leave an honest NaN row.
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
    ewma_next_sigma,
)

from quant_fund.metrics.scoring import crps_empirical, pinball_loss  # noqa: E402
from quant_fund.research.sota_evidence import validate_bars  # noqa: E402

MODEL = "dip_evt"
EVT_TAIL_MASS = 0.10  # threshold quantile level on each side
EVT_MIN_EXCEEDANCES = 8
EVT_XI_CLIP = 0.5
EVT_GRID_N = 512  # midpoint quantile grid for the CRPS copy
EVT_SCALE_LO, EVT_SCALE_HI = 0.7, 1.4


def _gpd_mom(exceedances: np.ndarray) -> tuple[float, float]:
    """Method-of-moments GPD (xi, beta) on positive exceedances.

    For a GPD tail, E[X] = beta/(1-xi) and Var(X) = beta^2/((1-xi)^2 (1-2xi)),
    so matching sample mean/variance gives ``xi = 0.5*(1 - m^2/v)`` and
    ``beta = 0.5*m*(m^2/v + 1)``. Raises ValueError on too few exceedances or
    degenerate/unstable moments; xi is clipped to [-0.5, 0.5] (the MoM variance
    ratio is only meaningful for xi < 0.5 anyway).
    """
    x = np.asarray(exceedances, dtype=float)
    x = x[np.isfinite(x) & (x > 0.0)]
    if x.size < EVT_MIN_EXCEEDANCES:
        raise ValueError(f"insufficient exceedances ({x.size})")
    m = float(np.mean(x))
    v = float(np.var(x, ddof=1))
    if not np.isfinite(m) or not np.isfinite(v) or m <= 0.0 or v <= 0.0:
        raise ValueError("unstable gpd moments (degenerate exceedances)")
    ratio = m * m / v
    xi = 0.5 * (1.0 - ratio)
    beta = 0.5 * m * (ratio + 1.0)
    if not np.isfinite(xi) or not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("unstable gpd moments (non-finite fit)")
    return float(np.clip(xi, -EVT_XI_CLIP, EVT_XI_CLIP)), float(beta)


def _evt_fit(rets: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """Two-sided POT fit on raw returns: (u_lo, xi_lo, beta_lo, u_hi, xi_hi, beta_hi)."""
    u_lo = float(np.quantile(rets, EVT_TAIL_MASS))
    u_hi = float(np.quantile(rets, 1.0 - EVT_TAIL_MASS))
    if not (np.isfinite(u_lo) and np.isfinite(u_hi)) or u_lo >= u_hi:
        raise ValueError("unstable thresholds (degenerate window)")
    xi_lo, beta_lo = _gpd_mom(u_lo - rets[rets < u_lo])
    xi_hi, beta_hi = _gpd_mom(rets[rets > u_hi] - u_hi)
    return u_lo, xi_lo, beta_lo, u_hi, xi_hi, beta_hi


def _evt_quantiles(
    rets: np.ndarray,
    taus: np.ndarray,
    fit: tuple[float, float, float, float, float, float],
) -> np.ndarray:
    """Hybrid quantile function: GPD tails, empirical body, continuous at u.

    Lower tail (tau < 0.10): with tail mass 0.10 anchored at u_lo,
    ``q = u_lo - (beta_lo/xi_lo) * ((tau/0.10)^(-xi_lo) - 1)`` (xi->0 limit
    ``u_lo + beta_lo*ln(tau/0.10)``). Upper tail (tau > 0.90):
    ``q = u_hi + (beta_hi/xi_hi) * (((1-tau)/0.10)^(-xi_hi) - 1)`` (xi->0 limit
    ``u_hi - beta_hi*ln((1-tau)/0.10)``). Body is ``np.quantile``; the seams are
    exact, so the function is non-decreasing by construction —
    ``maximum.accumulate`` is a cheap guard only.
    """
    u_lo, xi_lo, beta_lo, u_hi, xi_hi, beta_hi = fit
    taus = np.asarray(taus, dtype=float)
    out = np.empty(taus.size, dtype=float)
    lo = taus < EVT_TAIL_MASS
    hi = taus > 1.0 - EVT_TAIL_MASS
    body = ~(lo | hi)
    if body.any():
        out[body] = np.quantile(rets, taus[body])
    if lo.any():
        p = taus[lo] / EVT_TAIL_MASS  # in (0, 1)
        if abs(xi_lo) < 1e-12:
            out[lo] = u_lo + beta_lo * np.log(p)
        else:
            out[lo] = u_lo - (beta_lo / xi_lo) * (p ** (-xi_lo) - 1.0)
    if hi.any():
        p = (1.0 - taus[hi]) / EVT_TAIL_MASS  # in (0, 1)
        if abs(xi_hi) < 1e-12:
            out[hi] = u_hi - beta_hi * np.log(p)
        else:
            out[hi] = u_hi + (beta_hi / xi_hi) * (p ** (-xi_hi) - 1.0)
    return np.maximum.accumulate(out)


def _vol_ratio(rets: np.ndarray) -> float:
    """clip(ewma_next_sigma / long-run std, 0.7, 1.4); 1.0 when degenerate."""
    sd = float(np.std(rets, ddof=1))
    ew = ewma_next_sigma(rets)
    if not (np.isfinite(sd) and np.isfinite(ew)) or sd <= 0.0 or ew <= 0.0:
        return 1.0
    return float(np.clip(ew / sd, EVT_SCALE_LO, EVT_SCALE_HI))


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
    q_taus = np.full((n_rows, len(TAUS)), np.nan)
    q_lgbm = np.full((n_rows, LGBM_TAUS.size), np.nan)
    xi_lo_col = np.full(n_rows, np.nan)
    xi_hi_col = np.full(n_rows, np.nan)
    scale_col = np.full(n_rows, np.nan)
    fallback = np.zeros(n_rows, dtype=np.int64)
    grid = (np.arange(EVT_GRID_N) + 0.5) / EVT_GRID_N
    taus_arr = np.asarray(TAUS)
    reasons = {"insufficient_exceedances": 0, "unstable_fit": 0, "nonfinite_quantiles": 0}
    for row, i in enumerate(range(first_origin, n - 1)):
        closes_hist = closes[: i + 1]
        long_start = max(0, closes_hist.size - (garch_window + 1))
        rets_long = np.diff(closes_hist[long_start:]) / closes_hist[long_start:-1]
        y = float(closes[i + 1] / closes[i] - 1.0)
        tt[row] = int(event_times[i + 1])
        try:
            try:
                fit = _evt_fit(rets_long)
            except ValueError as exc:
                fit = None
                key = "insufficient_exceedances" if "insufficient" in str(exc) else "unstable_fit"
                reasons[key] += 1
            if fit is not None:
                qg = _evt_quantiles(rets_long, grid, fit)
                qt = _evt_quantiles(rets_long, taus_arr, fit)
                ql = _evt_quantiles(rets_long, LGBM_TAUS, fit)
                if not (np.isfinite(qg).all() and np.isfinite(qt).all() and np.isfinite(ql).all()):
                    reasons["nonfinite_quantiles"] += 1
                    fit = None
            if fit is None:
                # Plain empirical quantile function, unscaled, same scoring.
                qg = np.quantile(rets_long, grid)
                qt = np.quantile(rets_long, taus_arr)
                ql = np.quantile(rets_long, LGBM_TAUS)
                s = 1.0
                fallback[row] = 1
            else:
                u_lo, xi_lo, beta_lo, u_hi, xi_hi, beta_hi = fit
                s = _vol_ratio(rets_long)
                xi_lo_col[row] = xi_lo
                xi_hi_col[row] = xi_hi
            qg = qg * s
            qt = qt * s
            ql = ql * s
            c = crps_empirical(y, qg)
            if not np.isfinite(c):
                raise ValueError("nonfinite crps")
            crps[row] = c
            for k, tau in enumerate(TAUS):
                pin[row, k] = float(pinball_loss(np.array([y]), np.array([qt[k]]), tau)[0])
            q_taus[row] = qt
            q_lgbm[row] = ql
            scale_col[row] = s
        except Exception:  # noqa: BLE001 - honest NaN, row stays disclosed
            continue
    return {
        "crps_col": crps,
        "pin_cols": pin,
        "target_time_ns": tt,
        "bar_interval_ns": np.asarray(interval, dtype=np.int64),
        "q_taus": q_taus,
        "q_lgbm": q_lgbm,
        "xi_lo_col": xi_lo_col,
        "xi_hi_col": xi_hi_col,
        "scale_col": scale_col,
        "fallback_col": fallback,
        "reason_insufficient": np.asarray(reasons["insufficient_exceedances"], dtype=np.int64),
        "reason_unstable": np.asarray(reasons["unstable_fit"], dtype=np.int64),
        "reason_nonfinite": np.asarray(reasons["nonfinite_quantiles"], dtype=np.int64),
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
    xi_lo_fin = out["xi_lo_col"][np.isfinite(out["xi_lo_col"])]
    xi_hi_fin = out["xi_hi_col"][np.isfinite(out["xi_hi_col"])]
    meta_out = {
        "tool": Path(__file__).name,
        "model": MODEL,
        "evt_spec": (
            "two-sided POT GPD tails on garch_window returns; thresholds q10/q90, "
            "MoM xi=0.5*(1-m^2/v) beta=0.5*m*(m^2/v+1), xi clipped [-0.5,0.5]; "
            "empirical body on [0.10,0.90], GPD tails anchored at u (tail mass "
            "0.10, continuous at the seam); quantiles scaled by "
            "clip(ewma_next_sigma/std, 0.7, 1.4); CRPS = 512-midpoint quantile "
            "copy via crps_empirical; fallback = plain empirical quantiles "
            "unscaled under the same scoring"
        ),
        "evt_tail_mass": EVT_TAIL_MASS,
        "evt_min_exceedances": EVT_MIN_EXCEEDANCES,
        "evt_grid_n": EVT_GRID_N,
        "n_gpd_fits": int(n_rows - int(out["fallback_col"].sum())),
        "n_fallback_empirical": int(out["fallback_col"].sum()),
        "n_fallback_insufficient_exceedances": int(out["reason_insufficient"]),
        "n_fallback_unstable_fit": int(out["reason_unstable"]),
        "n_fallback_nonfinite_quantiles": int(out["reason_nonfinite"]),
        "mean_xi_lower": float(np.mean(xi_lo_fin)) if xi_lo_fin.size else None,
        "mean_xi_upper": float(np.mean(xi_hi_fin)) if xi_hi_fin.size else None,
        "mean_vol_ratio": float(np.nanmean(out["scale_col"])),
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
        q_taus=out["q_taus"],
        q_lgbm=out["q_lgbm"],
        xi_lo_col=out["xi_lo_col"],
        xi_hi_col=out["xi_hi_col"],
        scale_col=out["scale_col"],
        fallback_col=out["fallback_col"],
        meta_json=np.array(json.dumps(meta_out)),
    )
    print(
        f"{args.shard.name}: {MODEL} rows={n_rows} "
        f"finite={meta_out['n_finite_crps']} mean_crps={np.nanmean(out['crps_col']):.6f} "
        f"gpd={meta_out['n_gpd_fits']} fb={meta_out['n_fallback_empirical']} "
        f"xi_lo={meta_out['mean_xi_lower']} xi_hi={meta_out['mean_xi_upper']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
