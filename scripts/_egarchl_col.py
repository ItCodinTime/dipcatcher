"""Compute the ``dip_egarch_l`` challenger column over a shard's origin grid.

Replicates the origin loop of ``sota_eval_kronos.py`` (first_origin formula,
garch-window slicing, close-to-close target) but only evaluates one
deterministic challenger — no target models, no other challengers. Output binds
to the shard via ``bars_sha256`` + config fields so the splice tool can verify
row alignment by construction (the grid is deterministic given the same bars
bytes and the same config).

``dip_egarch_l`` is the TRUE asymmetric-EGARCH challenger: an EGARCH fit via
the shared ``_arch_fit`` helper (arch.univariate.arch_model, mean="Constant",
vol="EGARCH", p=1, o=1, q=1, dist="t", rescale=False) on the same ``rets_long``
the ``dip_garch_t`` challenger consumes, scored by the identical closed-form
Student-t CRPS path (``crps_student_t`` on fitted nu/mu/1-step sigma forecast).
Unlike ``dip_egarch`` (which landed on o=0 everywhere and therefore never
engaged the leverage/gamma terms), this variant puts o=1 first so the gamma
coefficient is actually fitted — the real leverage channel.

Spec chain per origin (deterministic): vol="EGARCH", dist="t", o=1 first. The
o=1 result is rejected when the fit raises (``n_o1_exception``) or when its
1-step sigma forecast is degenerate (``n_o1_degenerate``): non-finite,
non-positive, or > ``SIG_CAP`` (1.0 = a >100% per-bar stdev forecast, which is
definitionally broken for the 1d/4h majors grid; observed sane forecasts are
<=~0.2, observed explosive ones 32…1e151). A rejected o=1 falls back to o=0 for
that origin (same degeneracy test); if o=0 also fails the row is an honest NaN
(``n_convergence_failures``). Convergence *warnings* are treated as normal —
``_arch_fit`` already suppresses them; only hard failures reach the counter.

Gamma accounting: for every origin scored by o=1 the fitted ``gamma[1]``
coefficient is recorded. In ``arch``'s EGARCH parameterization a *negative*
gamma means volatility rises more after negative returns than after positive
ones of the same size — i.e. the classical leverage effect. Per-column meta
reports n_gamma_neg / n_gamma_nonneg and gamma mean/median/min/max over the
accepted o=1 fits so the leverage claim can be checked from the artifacts.
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

from sota_eval_kronos import TAUS, _arch_fit, _sha256  # noqa: E402

from quant_fund.metrics.scoring import (  # noqa: E402
    crps_student_t,
    pinball_loss,
)
from quant_fund.research.sota_evidence import validate_bars  # noqa: E402

MODEL = "dip_egarch_l"
SIG_CAP = 1.0  # 1-step sigma forecasts above 100%/bar are degenerate


class _DegenerateForecast(Exception):
    """Raised when a fit produces a non-finite/non-positive/explosive sigma."""


def _try_spec(rets_pct: np.ndarray, o: int):
    """Fit EGARCH(1,1)-t at asymmetry order ``o``; return (fit, sig_g).

    Raises ``_DegenerateForecast`` when the 1-step sigma forecast is not
    finite, not positive, or exceeds ``SIG_CAP``; any other exception from the
    fit itself propagates.
    """
    fit = _arch_fit(rets_pct, vol="EGARCH", dist="t", o=o)
    var_next = float(fit.forecast(horizon=1).variance.iloc[-1, 0])
    sig = np.sqrt(var_next) / 100.0 if var_next > 0.0 else np.nan
    if not np.isfinite(sig) or sig <= 0.0 or sig > SIG_CAP:
        raise _DegenerateForecast(f"o={o}: sigma forecast {sig!r}")
    return fit, sig


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
    spec_o1 = 0
    spec_o0 = 0
    o1_exception = 0
    o1_degenerate = 0
    failures = 0
    gammas: list[float] = []
    nan3 = np.full(len(TAUS), np.nan)
    for row, i in enumerate(range(first_origin, n - 1)):
        closes_hist = closes[: i + 1]
        long_start = max(0, closes_hist.size - (garch_window + 1))
        rets_long = np.diff(closes_hist[long_start:]) / closes_hist[long_start:-1]
        y = float(closes[i + 1] / closes[i] - 1.0)
        tt[row] = int(event_times[i + 1])
        try:
            try:
                fit, sig_g = _try_spec(rets_long * 100.0, o=1)
                spec_o1 += 1
                gammas.append(float(fit.params.get("gamma[1]", np.nan)))
            except _DegenerateForecast:
                o1_degenerate += 1
                fit, sig_g = _try_spec(rets_long * 100.0, o=0)
                spec_o0 += 1
            except Exception:  # noqa: BLE001 - deterministic fallback spec
                o1_exception += 1
                fit, sig_g = _try_spec(rets_long * 100.0, o=0)
                spec_o0 += 1
            nu_g = float(fit.params["nu"])
            mu_g = float(fit.params.get("mu", 0.0)) / 100.0
            scale_g = sig_g * np.sqrt((nu_g - 2.0) / nu_g) if nu_g > 2.0 else np.nan
            c = float(crps_student_t(np.array([y]), np.array([mu_g]), np.array([scale_g]), nu_g)[0])
            q = st.t.ppf(TAUS, nu_g, loc=mu_g, scale=scale_g) if np.isfinite(scale_g) else nan3
            crps[row] = c
            for k, tau in enumerate(TAUS):
                if np.isfinite(q[k]):
                    pin[row, k] = float(pinball_loss(np.array([y]), np.array([q[k]]), tau)[0])
        except Exception:  # noqa: BLE001 - honest NaN, row stays disclosed
            failures += 1
    g = np.asarray(gammas, dtype=float)
    g_fin = g[np.isfinite(g)]
    return {
        "crps_col": crps,
        "pin_cols": pin,
        "target_time_ns": tt,
        "bar_interval_ns": np.asarray(interval, dtype=np.int64),
        "n_spec_o1": np.asarray(spec_o1, dtype=np.int64),
        "n_spec_o0": np.asarray(spec_o0, dtype=np.int64),
        "n_o1_exception": np.asarray(o1_exception, dtype=np.int64),
        "n_o1_degenerate": np.asarray(o1_degenerate, dtype=np.int64),
        "n_convergence_failures": np.asarray(failures, dtype=np.int64),
        "n_gamma_neg": np.asarray(int((g_fin < 0.0).sum()), dtype=np.int64),
        "n_gamma_nonneg": np.asarray(int((g_fin >= 0.0).sum()), dtype=np.int64),
        "gamma_mean": np.asarray(float(g_fin.mean()) if g_fin.size else np.nan),
        "gamma_median": np.asarray(float(np.median(g_fin)) if g_fin.size else np.nan),
        "gamma_min": np.asarray(float(g_fin.min()) if g_fin.size else np.nan),
        "gamma_max": np.asarray(float(g_fin.max()) if g_fin.size else np.nan),
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
    meta_out = {
        "tool": Path(__file__).name,
        "model": MODEL,
        "arch_spec": "vol=EGARCH p=1 o=1 q=1 dist=t mean=Constant rescale=False; "
        "o=1 primary, o=0 fallback on exception or degenerate "
        "1-step sigma (non-finite/<=0/>1.0)",
        "n_spec_o1": int(out["n_spec_o1"]),
        "n_spec_o0": int(out["n_spec_o0"]),
        "n_o1_exception": int(out["n_o1_exception"]),
        "n_o1_degenerate": int(out["n_o1_degenerate"]),
        "n_convergence_failures": int(out["n_convergence_failures"]),
        "n_gamma_neg": int(out["n_gamma_neg"]),
        "n_gamma_nonneg": int(out["n_gamma_nonneg"]),
        "gamma_mean": float(out["gamma_mean"]),
        "gamma_median": float(out["gamma_median"]),
        "gamma_min": float(out["gamma_min"]),
        "gamma_max": float(out["gamma_max"]),
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
        f"o1={meta_out['n_spec_o1']} o0={meta_out['n_spec_o0']} "
        f"(exc={meta_out['n_o1_exception']} deg={meta_out['n_o1_degenerate']}) "
        f"fail={meta_out['n_convergence_failures']} "
        f"gamma_neg={meta_out['n_gamma_neg']}/{meta_out['n_spec_o1']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
