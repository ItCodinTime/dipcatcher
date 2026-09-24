"""Compute a single challenger column over an existing shard's exact origin grid.

Replicates the origin loop of ``sota_eval_kronos.py`` (first_origin formula,
window/garch-window slicing, close-to-close target) but only evaluates one
deterministic challenger — no target models, no other challengers. Output binds
to the shard via ``bars_sha256`` + config fields so the splice tool can verify
row alignment by construction (the grid is deterministic given the same bars
bytes and the same config).

Supported ``--model`` values: ``dip_gmm_k``, ``dip_skt``, ``dip_qar``,
``dip_conf_t``, ``dip_regime``, ``dip_stack``.

``dip_stack`` is a causal quantile-stacking challenger: per-origin the base
challengers emit their quantile vectors at ``LGBM_TAUS``; per quantile level a
simplex weight vector is fit by exponentiated gradient descent on pinball loss
over a trailing buffer of past (forecast, realized) pairs. Vincentization
(per-level convex combination) preserves monotonicity. Warmup origins predict
with uniform weights. Fully deterministic.
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
    _conf_t_quantiles,
    _fit_gmm,
    _fit_skt,
    _fit_student_t,
    _qar_quantiles,
    _regime_quantiles,
    _sha256,
    _skt_quantiles,
    crps_from_quantiles,
    qf_at,
)

from quant_fund.metrics.scoring import (  # noqa: E402
    crps_empirical,
    crps_gaussian_mixture,
    gaussian_mixture_quantiles,
    pinball_loss,
)
from quant_fund.research.sota_evidence import validate_bars  # noqa: E402

MODELS = ("dip_gmm_k", "dip_skt", "dip_qar", "dip_conf_t", "dip_regime", "dip_stack")

# Stacking bases: the cheap challengers plus the two leaders. skt/qar are
# excluded — their per-origin MLE/QuantReg cost buys mid-pack accuracy.
STACK_BASES = (
    "dip_gauss",
    "dip_student_t",
    "dip_ewma_emp",
    "dip_empirical",
    "dip_garch_t",
    "dip_fhs",
    "dip_gmm_k",
    "dip_conf_t",
    "dip_regime",
)
STACK_WARMUP = 50  # origins predicted with uniform weights
STACK_BUFFER = 150  # trailing origins used to fit the stack weights
STACK_ITERS = 100
STACK_ETA = 2.0


def score_origin(model: str, rets_long: np.ndarray, y: float) -> tuple[float, np.ndarray]:
    """(crps, quantiles@TAUS) for one challenger at one origin."""
    if model == "dip_gmm_k":
        w, mu, sig = _fit_gmm(rets_long)
        return (
            float(crps_gaussian_mixture(np.array([y]), w, mu, sig)[0]),
            gaussian_mixture_quantiles(w, mu, sig, np.asarray(TAUS)),
        )
    if model == "dip_skt":
        xi, om, al, nu = _fit_skt(rets_long)
        grid = (np.arange(512) + 0.5) / 512.0
        samp = _skt_quantiles(xi, om, al, nu, grid)
        return crps_empirical(y, samp), _skt_quantiles(xi, om, al, nu, np.asarray(TAUS))
    if model == "dip_qar":
        q_qar = _qar_quantiles(rets_long)
        return (
            crps_from_quantiles(y, LGBM_TAUS, q_qar),
            np.array([qf_at(LGBM_TAUS, q_qar, t) for t in TAUS]),
        )
    if model in ("dip_conf_t", "dip_regime"):
        fn = _conf_t_quantiles if model == "dip_conf_t" else _regime_quantiles
        grid = (np.arange(512) + 0.5) / 512.0
        return crps_empirical(y, fn(rets_long, grid)), fn(rets_long, np.asarray(TAUS))
    raise ValueError(f"unknown challenger {model}")


def _stack_base_quantiles(model: str, rets_long: np.ndarray) -> np.ndarray:
    """Quantile vector at LGBM_TAUS for one base challenger (NaN on failure)."""
    g = LGBM_TAUS
    nan = np.full(g.size, np.nan)
    try:
        if model == "dip_gauss":
            mu, sd = float(np.mean(rets_long)), float(np.std(rets_long, ddof=1))
            return st.norm.ppf(g, loc=mu, scale=sd)
        if model == "dip_student_t":
            nu, loc, sc = _fit_student_t(rets_long)
            return st.t.ppf(g, nu, loc=loc, scale=sc) if np.isfinite(nu) else nan
        if model == "dip_empirical":
            return np.quantile(rets_long, g)
        if model == "dip_ewma_emp":
            x_raw = np.asarray(rets_long, dtype=float)
            w = 0.97 ** np.arange(x_raw.size - 1, -1, -1.0)
            w /= w.sum()
            order = np.argsort(x_raw)
            x, w = x_raw[order], w[order]
            cumw = np.cumsum(w)
            n = x.size
            return np.array([x[min(np.searchsorted(cumw, t), n - 1)] for t in g])
        if model == "dip_garch_t":
            fit = _arch_fit(rets_long * 100.0, vol="GARCH", dist="t", o=0)
            nu_g = float(fit.params["nu"])
            mu_g = float(fit.params.get("mu", 0.0)) / 100.0
            sig_g = float(np.sqrt(fit.forecast(horizon=1).variance.iloc[-1, 0])) / 100.0
            scale_g = sig_g * np.sqrt((nu_g - 2.0) / nu_g) if nu_g > 2.0 else np.nan
            return st.t.ppf(g, nu_g, loc=mu_g, scale=scale_g) if np.isfinite(scale_g) else nan
        if model == "dip_fhs":
            fit = _arch_fit(rets_long * 100.0, vol="GARCH", dist="normal", o=1)
            sig_next = float(np.sqrt(fit.forecast(horizon=1).variance.iloc[-1, 0])) / 100.0
            mu_f = float(fit.params.get("mu", 0.0)) / 100.0
            cond_vol = np.asarray(fit.conditional_volatility, dtype=float) / 100.0
            resid = np.asarray(fit.resid, dtype=float) / 100.0
            ok = cond_vol > 0.0
            std_resid = resid[ok] / cond_vol[ok]
            std_resid = std_resid[np.isfinite(std_resid)]
            if std_resid.size < 20 or not np.isfinite(sig_next):
                return nan
            return mu_f + sig_next * np.quantile(std_resid, g)
        if model == "dip_gmm_k":
            w, mu, sig = _fit_gmm(rets_long)
            return gaussian_mixture_quantiles(w, mu, sig, g)
        if model == "dip_conf_t":
            return _conf_t_quantiles(rets_long, g)
        if model == "dip_regime":
            return _regime_quantiles(rets_long, g)
    except Exception:  # noqa: BLE001 - honest NaN base
        return nan
    raise ValueError(f"unknown stack base {model}")


def _stack_fit(buf_q: np.ndarray, buf_y: np.ndarray) -> np.ndarray:
    """Per-tau simplex weights (T,B)->(B,T) via exponentiated gradient on pinball.

    ``buf_q`` is (n_past, n_bases, n_taus); ``buf_y`` is (n_past,). Deterministic:
    uniform init, fixed iteration count, fixed step size.
    """
    n_bases = buf_q.shape[1]
    n_taus = buf_q.shape[2]
    w = np.full((n_bases, n_taus), 1.0 / n_bases)
    t_expand = LGBM_TAUS[None, :]  # (1, T)
    for _ in range(STACK_ITERS):
        # q_hat[t, tau] = sum_m w[m, tau] * q[t, m, tau]
        q_hat = np.einsum("tml,ml->tl", buf_q, w)
        # pinball subgradient wrt q_hat: tau - 1{y < q}
        grad_q = t_expand - (buf_y[:, None] < q_hat).astype(float)  # (t, T)
        # dL/dw[m,tau] = mean_t grad_q[t,tau] * q[t,m,tau]
        grad_w = np.einsum("tl,tml->ml", grad_q, buf_q) / buf_q.shape[0]
        w = w * np.exp(-STACK_ETA * grad_w)
        w /= w.sum(axis=0, keepdims=True)
    return w


def _stack_column(
    closes: np.ndarray,
    event_times: np.ndarray,
    interval: int,
    cfg: dict,
) -> dict[str, np.ndarray]:
    """Stateful dip_stack pass: stack weights at origin i fit on origins < i."""
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
    tau_idx = [int(np.where(t == LGBM_TAUS)[0][0]) for t in TAUS]
    buf_q: list[np.ndarray] = []  # past (B,T) base-quantile matrices
    buf_y: list[float] = []
    for row, i in enumerate(range(first_origin, n - 1)):
        closes_hist = closes[: i + 1]
        long_start = max(0, closes_hist.size - (garch_window + 1))
        rets_long = np.diff(closes_hist[long_start:]) / closes_hist[long_start:-1]
        y = float(closes[i + 1] / closes[i] - 1.0)
        tt[row] = int(event_times[i + 1])
        try:
            base_q = np.stack([_stack_base_quantiles(m, rets_long) for m in STACK_BASES])  # (B, T)
            finite = np.isfinite(base_q).all(axis=1)
            if finite.sum() < 2:
                continue  # honest NaN: too few bases this origin
            if len(buf_y) >= STACK_WARMUP:
                w = _stack_fit(
                    np.stack(buf_q[-STACK_BUFFER:]),
                    np.asarray(buf_y[-STACK_BUFFER:]),
                )
            else:
                w = np.full((len(STACK_BASES), LGBM_TAUS.size), 1.0 / len(STACK_BASES))
            w_f = w[finite]
            w_f = w_f / w_f.sum(axis=0, keepdims=True)
            q_stack = np.einsum("bt,bt->t", w_f, base_q[finite])
            crps[row] = crps_from_quantiles(y, LGBM_TAUS, q_stack)
            for k, ti in enumerate(tau_idx):
                pin[row, k] = float(
                    pinball_loss(np.array([y]), np.array([q_stack[ti]]), TAUS[k])[0]
                )
            # Bank the origin only after scoring (never trains on itself).
            if np.isfinite(base_q).all():
                buf_q.append(base_q)
                buf_y.append(y)
        except Exception:  # noqa: BLE001 - honest NaN row
            continue
    return {
        "crps_col": crps,
        "pin_cols": pin,
        "target_time_ns": tt,
        "bar_interval_ns": np.asarray(interval, dtype=np.int64),
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


def compute_column(bars_path: Path, cfg: dict, model: str) -> dict[str, np.ndarray]:
    """(crps[R], pin[R,T], target_time_ns[R]) over the shard's origin grid."""
    frame = pl.read_parquet(bars_path)
    event_times, interval = validate_bars(frame)
    closes = frame["close"].to_numpy().astype(float)
    if model == "dip_stack":
        return _stack_column(closes, event_times, interval, cfg)
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
    for row, i in enumerate(range(first_origin, n - 1)):
        closes_hist = closes[: i + 1]
        long_start = max(0, closes_hist.size - (garch_window + 1))
        rets_long = np.diff(closes_hist[long_start:]) / closes_hist[long_start:-1]
        y = float(closes[i + 1] / closes[i] - 1.0)
        tt[row] = int(event_times[i + 1])
        try:
            c, q = score_origin(model, rets_long, y)
            crps[row] = c
            for k, tau in enumerate(TAUS):
                if np.isfinite(q[k]):
                    pin[row, k] = float(pinball_loss(np.array([y]), np.array([q[k]]), tau)[0])
        except Exception:  # noqa: BLE001 - honest NaN, row stays disclosed
            pass
    return {
        "crps_col": crps,
        "pin_cols": pin,
        "target_time_ns": tt,
        "bar_interval_ns": np.asarray(interval, dtype=np.int64),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=Path, required=True, help="existing losses shard to match")
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--bars-root", type=Path, default=Path("data/raw/sources"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    with np.load(args.shard, allow_pickle=False) as z:
        meta = json.loads(str(z["meta_json"]))
        n_rows = int(z["crps_matrix"].shape[0])
        shard_names = [str(x) for x in z["model_names"]]
    cfg = meta["config"]
    want_cols = int(cfg.get("origins_per_asset", 0)) >= 1
    if not want_cols or args.model in shard_names:
        raise ValueError(f"{args.shard.name}: already has {args.model} or bad config")

    bars = _find_bars(args.bars_root, meta["bars_sha256"])
    t0 = time.time()
    out = compute_column(bars, cfg, args.model)
    if out["crps_col"].shape[0] != n_rows:
        raise ValueError(f"{args.shard.name}: grid mismatch {out['crps_col'].shape[0]} != {n_rows}")
    meta_out = {
        "tool": Path(__file__).name,
        "model": args.model,
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
        f"{args.shard.name}: {args.model} rows={n_rows} "
        f"finite={meta_out['n_finite_crps']} mean_crps={np.nanmean(out['crps_col']):.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
