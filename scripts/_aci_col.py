"""Adaptive Conformal Inference (ACI) challenger column over a shard's grid.

Wraps ``dip_garch_t`` (GARCH(1,1)-t 1-step quantiles on --garch-window returns)
with the online level-adjustment scheme of Gibbs & Candes (2021, NeurIPS,
"Adaptive Conformal Inference Under Distribution Shift"). Per grid level
``g`` in ``LGBM_TAUS`` an adjusted quantile level ``alpha_g`` is maintained:

    alpha_g <- clip(alpha_g + gamma * (g - 1{y <= Q_base(alpha_g)}), lo, hi)

applied AFTER the origin is scored, so the recursion is causal and never
trains on the current origin. The emitted quantile at level ``g`` is the base
quantile function resampled at ``alpha_g`` via ``qf_at`` (the shared linear /
exponential-tail completion over the LGBM_TAUS grid), so coverage errors feed
back into quantile location. The same adjustment drives both the pinball
targets (TAUS, a subset of the grid) and the full-grid vector whose quantile
integral is the CRPS — every level's alpha adapts independently, matching the
"per-target miscoverage" form of the algorithm.

Warmup: the first ``ACI_WARMUP`` origins emit the raw base quantiles (alphas
still update, so adaptation has begun when emission switches to adjusted
quantiles). If the base GARCH-t fit fails at an origin, that row stays honest
NaN and the alphas are left untouched.

Output binds to the shard via ``bars_sha256`` + protocol config fields, the
same contract as ``_challenger_col.py``, so ``splice_challenger_column.py``
can verify row alignment by construction.
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
    _arch_fit,
    _sha256,
    crps_from_quantiles,
    qf_at,
)

from quant_fund.metrics.scoring import pinball_loss  # noqa: E402
from quant_fund.research.sota_evidence import validate_bars  # noqa: E402

MODEL = "dip_aci"
BASE_MODEL = "dip_garch_t"
ACI_GAMMA = 0.02  # Gibbs & Candes step size
ACI_WARMUP = 30  # origins emitting raw base quantiles (alphas still update)
ACI_LO, ACI_HI = 0.01, 0.99  # level clip bounds


def _garch_t_quantiles(rets_long: np.ndarray) -> np.ndarray:
    """dip_garch_t quantile vector at LGBM_TAUS (NaN on failure).

    Identical code path to the ``dip_garch_t`` branch of
    ``_challenger_col._stack_base_quantiles``.
    """
    g = LGBM_TAUS
    nan = np.full(g.size, np.nan)
    try:
        fit = _arch_fit(rets_long * 100.0, vol="GARCH", dist="t", o=0)
        nu_g = float(fit.params["nu"])
        mu_g = float(fit.params.get("mu", 0.0)) / 100.0
        sig_g = float(np.sqrt(fit.forecast(horizon=1).variance.iloc[-1, 0])) / 100.0
        scale_g = sig_g * np.sqrt((nu_g - 2.0) / nu_g) if nu_g > 2.0 else np.nan
        return st.t.ppf(g, nu_g, loc=mu_g, scale=scale_g) if np.isfinite(scale_g) else nan
    except Exception:  # noqa: BLE001 - honest NaN base
        return nan


def _aci_column(
    closes: np.ndarray,
    event_times: np.ndarray,
    interval: int,
    cfg: dict,
    *,
    gamma: float,
    warmup: int,
) -> dict[str, np.ndarray]:
    """Stateful dip_aci pass: alpha levels at origin i adapt to origins < i."""
    n = closes.size
    origins = int(cfg["origins_per_asset"])
    lookback = int(cfg["lookback"])
    window = int(cfg["window"])
    garch_window = int(cfg["garch_window"])
    first_origin = max(lookback, window + 1, n - origins - 1)
    if first_origin >= n - 1:
        raise ValueError(f"not enough bars ({n})")
    n_rows = n - 1 - first_origin
    crps = np.full(n_rows, np.nan)
    pin = np.full((n_rows, len(TAUS)), np.nan)
    tt = np.empty(n_rows, dtype=np.int64)
    # Diagnostics only: raw base scored under the SAME quantile-grid convention
    # (crps_from_quantiles / pinball on q_base), so the ACI delta is measured
    # against the identical scoring path — not the shard's closed-form
    # crps_student_t column.
    base_crps = np.full(n_rows, np.nan)
    base_pin = np.full((n_rows, len(TAUS)), np.nan)
    alpha_path = np.full((n_rows, LGBM_TAUS.size), np.nan)
    tau_idx = [int(np.where(t == LGBM_TAUS)[0][0]) for t in TAUS]
    alpha = LGBM_TAUS.copy()  # adjusted level per grid quantile
    abs_dev = np.zeros(LGBM_TAUS.size)  # sum |alpha - level| over updates
    n_updates = 0
    for row, i in enumerate(range(first_origin, n - 1)):
        closes_hist = closes[: i + 1]
        long_start = max(0, closes_hist.size - (garch_window + 1))
        rets_long = np.diff(closes_hist[long_start:]) / closes_hist[long_start:-1]
        y = float(closes[i + 1] / closes[i] - 1.0)
        tt[row] = int(event_times[i + 1])
        q_base = _garch_t_quantiles(rets_long)
        if not np.isfinite(q_base).all():
            continue  # honest NaN row; alphas untouched
        alpha_path[row] = alpha
        base_crps[row] = crps_from_quantiles(y, LGBM_TAUS, q_base)
        for k, ti in enumerate(tau_idx):
            base_pin[row, k] = float(
                pinball_loss(np.array([y]), np.array([q_base[ti]]), TAUS[k])[0]
            )
        # Base quantile function resampled at the current adjusted levels.
        q_adj = np.array([qf_at(LGBM_TAUS, q_base, a) for a in alpha])
        q_emit = q_base if row < warmup else q_adj
        crps[row] = crps_from_quantiles(y, LGBM_TAUS, q_emit)
        for k, ti in enumerate(tau_idx):
            pin[row, k] = float(pinball_loss(np.array([y]), np.array([q_emit[ti]]), TAUS[k])[0])
        # ACI update AFTER scoring (never trains on the current origin): the
        # coverage indicator is evaluated at the adjusted-level quantile, per
        # the Gibbs & Candes recursion.
        covered = (y <= q_adj).astype(float)
        alpha = np.clip(alpha + gamma * (LGBM_TAUS - covered), ACI_LO, ACI_HI)
        abs_dev += np.abs(alpha - LGBM_TAUS)
        n_updates += 1
    diag = {
        "alpha_final": {f"{g:.2f}": float(a) for g, a in zip(LGBM_TAUS, alpha, strict=True)},
        "alpha_mean_abs_dev": {
            f"{g:.2f}": float(abs_dev[j] / n_updates) if n_updates else None
            for j, g in enumerate(LGBM_TAUS)
        },
        "alpha_mean_abs_dev_taus": {
            f"{TAUS[k]:.2f}": (float(abs_dev[tau_idx[k]] / n_updates) if n_updates else None)
            for k in range(len(TAUS))
        },
        "n_alpha_updates": n_updates,
    }
    return {
        "crps_col": crps,
        "pin_cols": pin,
        "target_time_ns": tt,
        "bar_interval_ns": np.asarray(interval, dtype=np.int64),
        "base_crps_col": base_crps,
        "base_pin_cols": base_pin,
        "alpha_path": alpha_path,
        "aci_diag": diag,
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


def compute_column(
    bars_path: Path, cfg: dict, *, gamma: float, warmup: int
) -> dict[str, np.ndarray]:
    """(crps[R], pin[R,T], target_time_ns[R]) over the shard's origin grid."""
    frame = pl.read_parquet(bars_path)
    event_times, interval = validate_bars(frame)
    closes = frame["close"].to_numpy().astype(float)
    return _aci_column(closes, event_times, interval, cfg, gamma=gamma, warmup=warmup)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=Path, required=True, help="existing losses shard to match")
    p.add_argument("--bars-root", type=Path, default=Path("data/raw/sources"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gamma", type=float, default=ACI_GAMMA)
    p.add_argument("--warmup", type=int, default=ACI_WARMUP)
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
    out = compute_column(bars, cfg, gamma=args.gamma, warmup=args.warmup)
    if out["crps_col"].shape[0] != n_rows:
        raise ValueError(f"{args.shard.name}: grid mismatch {out['crps_col'].shape[0]} != {n_rows}")
    diag = out.pop("aci_diag")
    meta_out = {
        "tool": Path(__file__).name,
        "model": MODEL,
        "base_model": BASE_MODEL,
        "aci": {
            "gamma": args.gamma,
            "warmup": args.warmup,
            "clip": [ACI_LO, ACI_HI],
            "reference": "Gibbs & Candes (2021) Adaptive Conformal Inference",
            **diag,
        },
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
        "mean_crps": float(np.nanmean(out["crps_col"])),
        "mean_crps_base_grid": float(np.nanmean(out["base_crps_col"])),
        "elapsed_s": round(time.time() - t0, 3),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        crps_col=out["crps_col"],
        pin_cols=out["pin_cols"],
        target_time_ns=out["target_time_ns"],
        bar_interval_ns=out["bar_interval_ns"],
        base_crps_col=out["base_crps_col"],
        base_pin_cols=out["base_pin_cols"],
        alpha_path=out["alpha_path"],
        meta_json=np.array(json.dumps(meta_out)),
    )
    print(
        f"{args.shard.name}: {MODEL} rows={n_rows} "
        f"finite={meta_out['n_finite_crps']} mean_crps={np.nanmean(out['crps_col']):.6f} "
        f"base_grid={np.nanmean(out['base_crps_col']):.6f} "
        f"updates={diag['n_alpha_updates']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
