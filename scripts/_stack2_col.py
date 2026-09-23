"""Compute the ``dip_stack2`` challenger column over a shard's origin grid.

Second-generation causal quantile stack — same engine as ``dip_stack`` in
``_challenger_col.py`` (per-origin base quantiles at ``LGBM_TAUS``,
exponentiated-gradient simplex weights on pinball over a trailing buffer,
Vincentized per-level convex combination, warmup 50 / buffer 150 / iters 100 /
eta 2.0, honest NaN when <2 finite bases, bank-after-scoring discipline) but
with an upgraded base set: the nine ``dip_stack`` bases plus the new top-tier
challengers ``dip_egarch`` and ``dip_evt`` and the conditioning challengers
``dip_seas`` (calendar-slot shrunk empirical) and ``dip_volm``
(volume-conditioned sigma).

The new bases are vendored — the minimal fitter math is copied here so this
file is self-contained and does not import the other lane files (which may be
mid-write on parallel lanes):

- ``dip_egarch``: ``_arch_fit(rets*100, vol="EGARCH", dist="t", o=0)`` with the
  ``o=1`` deterministic fallback, mapped to Student-t quantiles exactly like
  the ``dip_garch_t`` base (scale ``sigma*sqrt((nu-2)/nu)``).
- ``dip_evt``: two-sided POT GPD tails (MoM fit, xi clipped to [-0.5, 0.5])
  over an empirical body at thresholds q10/q90, quantiles rescaled by
  ``clip(ewma_next_sigma/std, 0.7, 1.4)``; falls back to plain unscaled
  empirical quantiles when a tail fit is infeasible (mirrors
  ``_evt_col.py``).
- ``dip_seas``: slot-conditional empirical quantiles shrunk toward the pooled
  fit (pseudo-count 40, min slot obs 15); needs bar calendar slots, which are
  computed once from ``event_time`` (daily -> UTC day-of-week, else
  ``hour//4``) exactly like ``_seas_col.py``.
- ``dip_volm``: volume-surprise-conditioned EWMA sigma with shrunk
  standardized-residual quantiles; needs the bar ``volume`` column exactly
  like ``_volm_col.py``.

Output binds to the shard via ``bars_sha256`` + config fields so the splice
tool can verify row alignment by construction (the grid is deterministic
given the same bars bytes and the same config). Fully deterministic.
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
    _fit_student_t,
    _regime_quantiles,
    _sha256,
    crps_from_quantiles,
    ewma_next_sigma,
)

from quant_fund.metrics.scoring import (  # noqa: E402
    gaussian_mixture_quantiles,
    pinball_loss,
)
from quant_fund.research.sota_evidence import validate_bars  # noqa: E402

MODEL = "dip_stack2"

# Stack2 bases: the nine dip_stack bases plus the four new-generation
# challengers vendored below. seas/volm add conditioning diversity; egarch/evt
# are the new top-tier tails.
STACK2_BASES = (
    "dip_gauss",
    "dip_student_t",
    "dip_ewma_emp",
    "dip_empirical",
    "dip_garch_t",
    "dip_fhs",
    "dip_gmm_k",
    "dip_conf_t",
    "dip_regime",
    "dip_egarch",
    "dip_evt",
    "dip_seas",
    "dip_volm",
)
STACK_WARMUP = 50  # origins predicted with uniform weights
STACK_BUFFER = 150  # trailing origins used to fit the stack weights
STACK_ITERS = 100
STACK_ETA = 2.0

# ---------------------------------------------------------------------------
# Vendored from _egarch_col.py
# ---------------------------------------------------------------------------


def _egarch_fit(rets_pct: np.ndarray):
    """EGARCH(1,1)-t via the shared helper; o=0 primary, o=1 fallback."""
    try:
        return _arch_fit(rets_pct, vol="EGARCH", dist="t", o=0)
    except Exception:  # noqa: BLE001 - deterministic fallback spec
        return _arch_fit(rets_pct, vol="EGARCH", dist="t", o=1)


# ---------------------------------------------------------------------------
# Vendored from _evt_col.py (constants + POT/GPD fitters + vol rescale)
# ---------------------------------------------------------------------------

EVT_TAIL_MASS = 0.10  # threshold quantile level on each side
EVT_MIN_EXCEEDANCES = 8
EVT_XI_CLIP = 0.5
EVT_SCALE_LO, EVT_SCALE_HI = 0.7, 1.4


def _gpd_mom(exceedances: np.ndarray) -> tuple[float, float]:
    """Method-of-moments GPD (xi, beta) on positive exceedances."""
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
    """Hybrid quantile function: GPD tails, empirical body, continuous at u."""
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


def _evt_base_quantiles(rets: np.ndarray, g: np.ndarray) -> np.ndarray:
    """dip_evt quantiles at ``g``: GPD-tail hybrid, else plain empirical."""
    try:
        fit = _evt_fit(rets)
    except ValueError:
        fit = None
    if fit is not None:
        q = _evt_quantiles(rets, g, fit)
        if not np.isfinite(q).all():
            fit = None
    if fit is None:
        return np.quantile(rets, g)  # fallback: unscaled empirical
    return q * _vol_ratio(rets)


# ---------------------------------------------------------------------------
# Vendored from _seas_col.py (calendar slots + shrunk slot quantiles)
# ---------------------------------------------------------------------------

SEAS_SHRINK_K = 40.0  # pseudo-count shrinkage of slot quantiles toward pooled
SEAS_MIN_SLOT_OBS = 15  # slot pools thinner than this fall back to pooled fit
NS_PER_HOUR = 3_600_000_000_000
NS_PER_DAY = 86_400_000_000_000


def _slot_ids(event_times: np.ndarray, interval_ns: int) -> tuple[np.ndarray, str]:
    """Calendar slot per bar index from UTC event timestamps (int64 ns)."""
    if interval_ns == NS_PER_DAY:
        days = np.floor_divide(event_times, NS_PER_DAY)
        return ((days + 3) % 7).astype(np.int64), "dow"
    hours = np.floor_divide(event_times, NS_PER_HOUR) % 24
    return (hours // 4).astype(np.int64), "hour4"


def _seas_quantiles(
    rets_long: np.ndarray, train_slots: np.ndarray, slot: int
) -> np.ndarray:
    """Shrunk slot-conditional empirical quantiles at ``LGBM_TAUS``."""
    q_global = np.quantile(rets_long, LGBM_TAUS)
    if not np.isfinite(q_global).all():
        return np.full(LGBM_TAUS.size, np.nan)
    pool = rets_long[train_slots == slot]
    n_s = int(pool.size)
    if n_s < SEAS_MIN_SLOT_OBS:
        return q_global
    q_slot = np.quantile(pool, LGBM_TAUS)
    if not np.isfinite(q_slot).all():
        return q_global
    return (n_s * q_slot + SEAS_SHRINK_K * q_global) / (n_s + SEAS_SHRINK_K)


# ---------------------------------------------------------------------------
# Vendored from _volm_col.py (volume surprise + conditioned sigma quantiles)
# ---------------------------------------------------------------------------

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
    """Per-bar EWMA forecast sigmas: ``sig[t]`` forecasts bar t from data < t."""
    n = rets.size
    sig = np.full(n, np.nan)
    var = float(rets[0] ** 2)
    for t in range(1, n):
        sig[t] = np.sqrt(var)
        var = lam * var + (1.0 - lam) * float(rets[t]) * float(rets[t])
    return sig


def _volm_quantiles(rets_long: np.ndarray, vs_long: np.ndarray) -> np.ndarray:
    """Volume-conditioned quantile vector at LGBM_TAUS (NaN on failure)."""
    nan = np.full(LGBM_TAUS.size, np.nan)
    n = rets_long.size
    if n < MIN_Z + 1:
        return nan
    vs_ok = np.isfinite(vs_long) & (vs_long > 0.0)
    cover = float(vs_ok.mean())
    if cover < MIN_VOL_COVER:
        return nan

    # Recent volume surprise S: lam-weighted mean of valid vs in the tail.
    tail_vs = vs_long[-VOL_S_WIN:]
    tail_w = VOL_S_LAM ** np.arange(tail_vs.size - 1, -1, -1.0)
    tail_ok = np.isfinite(tail_vs) & (tail_vs > 0.0)
    if not tail_ok.any():
        return nan
    s_val = float(np.sum(tail_w[tail_ok] * tail_vs[tail_ok]) / np.sum(tail_w[tail_ok]))

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
                gamma_hat = GAMMA_BLEND * gamma_reg + (1.0 - GAMMA_BLEND) * GAMMA_FIXED

    # Standardized residuals under the volume-conditioned EWMA sigma.
    sig = _ewma_sigma_path(rets_long)
    z_ok = vs_ok & np.isfinite(rets_long) & np.isfinite(sig) & (sig > 0.0)
    z = rets_long[z_ok] / (sig[z_ok] * np.power(vs_long[z_ok], gamma_hat))
    z = z[np.isfinite(z)]
    if z.size < MIN_Z:
        return nan

    sigma_base = ewma_next_sigma(rets_long, lam=EWMA_LAM)
    mult = float(np.clip(s_val ** gamma_hat, MULT_LO, MULT_HI))
    sigma_cond = sigma_base * mult
    if not np.isfinite(sigma_cond) or sigma_cond <= 0.0:
        return nan
    mu_hat = float(np.sum(rets_long) / (n + MU_PSEUDO))
    q = mu_hat + sigma_cond * np.quantile(z, LGBM_TAUS)
    return np.maximum.accumulate(q)


# ---------------------------------------------------------------------------
# Stack engine (identical to _challenger_col.py's dip_stack path)
# ---------------------------------------------------------------------------


def _stack2_base_quantiles(
    model: str, rets_long: np.ndarray, ctx: dict
) -> np.ndarray:
    """Quantile vector at LGBM_TAUS for one base challenger (NaN on failure).

    ``ctx`` carries the per-origin extras the new bases need:
    ``train_slots``/``slot`` for dip_seas and ``vs_long`` for dip_volm.
    """
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
            return (
                st.t.ppf(g, nu_g, loc=mu_g, scale=scale_g) if np.isfinite(scale_g) else nan
            )
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
        if model == "dip_egarch":
            fit = _egarch_fit(rets_long * 100.0)
            nu_g = float(fit.params["nu"])
            mu_g = float(fit.params.get("mu", 0.0)) / 100.0
            var_next = float(fit.forecast(horizon=1).variance.iloc[-1, 0])
            sig_g = np.sqrt(var_next) / 100.0 if var_next > 0.0 else np.nan
            if not np.isfinite(sig_g) or sig_g <= 0.0:
                return nan
            scale_g = sig_g * np.sqrt((nu_g - 2.0) / nu_g) if nu_g > 2.0 else np.nan
            return (
                st.t.ppf(g, nu_g, loc=mu_g, scale=scale_g) if np.isfinite(scale_g) else nan
            )
        if model == "dip_evt":
            return _evt_base_quantiles(rets_long, g)
        if model == "dip_seas":
            return _seas_quantiles(rets_long, ctx["train_slots"], ctx["slot"])
        if model == "dip_volm":
            return _volm_quantiles(rets_long, ctx["vs_long"])
    except Exception:  # noqa: BLE001 - honest NaN base
        return nan
    raise ValueError(f"unknown stack base {model}")


def _stack2_fit(
    buf_q: np.ndarray, buf_y: np.ndarray
) -> np.ndarray:
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


def _stack2_column(
    closes: np.ndarray,
    event_times: np.ndarray,
    interval: int,
    cfg: dict,
    vols: np.ndarray,
) -> dict[str, np.ndarray]:
    """Stateful dip_stack2 pass: stack weights at origin i fit on origins < i."""
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
    # Per-shard precompute for the conditioning bases (causal by construction:
    # slots derive from event_time, vs from volume at/before each bar).
    slots, slot_scheme = _slot_ids(event_times, int(interval))
    vs_all = _volume_surprise(vols)
    n_bases = len(STACK2_BASES)
    w_eff_sum = np.zeros((n_bases, LGBM_TAUS.size))  # effective weights used
    w_fin_sum = np.zeros((n_bases, LGBM_TAUS.size))  # weights when base finite
    finite_cnt = np.zeros(n_bases, dtype=np.int64)
    n_scored = 0
    n_uniform = 0
    for row, i in enumerate(range(first_origin, n - 1)):
        closes_hist = closes[: i + 1]
        long_start = max(0, closes_hist.size - (garch_window + 1))
        rets_long = np.diff(closes_hist[long_start:]) / closes_hist[long_start:-1]
        y = float(closes[i + 1] / closes[i] - 1.0)
        tt[row] = int(event_times[i + 1])
        ctx = {
            # rets_long[k] lands on bar index long_start+1+k; same alignment
            # as _seas_col.py / _volm_col.py.
            "train_slots": slots[long_start + 1 : i + 1],
            "slot": int(slots[i + 1]),
            "vs_long": vs_all[long_start + 1 : i + 1],
        }
        try:
            base_q = np.stack(
                [_stack2_base_quantiles(m, rets_long, ctx) for m in STACK2_BASES]
            )  # (B, T)
            finite = np.isfinite(base_q).all(axis=1)
            if finite.sum() < 2:
                continue  # honest NaN: too few bases this origin
            if len(buf_y) >= STACK_WARMUP:
                w = _stack2_fit(
                    np.stack(buf_q[-STACK_BUFFER:]),
                    np.asarray(buf_y[-STACK_BUFFER:]),
                )
            else:
                n_uniform += 1
                w = np.full((n_bases, LGBM_TAUS.size), 1.0 / n_bases)
            w_f = w[finite]
            w_f = w_f / w_f.sum(axis=0, keepdims=True)
            q_stack = np.einsum("bt,bt->t", w_f, base_q[finite])
            crps[row] = crps_from_quantiles(y, LGBM_TAUS, q_stack)
            for k, ti in enumerate(tau_idx):
                pin[row, k] = float(
                    pinball_loss(np.array([y]), np.array([q_stack[ti]]), TAUS[k])[0]
                )
            # Weight diagnostics (effective weight = 0 for excluded bases).
            w_eff = np.zeros((n_bases, LGBM_TAUS.size))
            w_eff[finite] = w_f
            w_eff_sum += w_eff
            w_fin_sum[finite] += w_f
            finite_cnt += finite.astype(np.int64)
            n_scored += 1
            # Bank the origin only after scoring (never trains on itself).
            if np.isfinite(base_q).all():
                buf_q.append(base_q)
                buf_y.append(y)
        except Exception:  # noqa: BLE001 - honest NaN row
            continue
    diag = {
        "bases": list(STACK2_BASES),
        "slot_scheme": slot_scheme,
        "n_scored": int(n_scored),
        "n_warmup_uniform": int(n_uniform),
        "n_banked_all_finite": int(len(buf_q)),
        "base_finite_origins": {
            m: int(c) for m, c in zip(STACK2_BASES, finite_cnt, strict=True)
        },
        # Mean effective stack weight per base per tau (0 on excluded origins);
        # each column sums to 1 over bases at every scored origin.
        "mean_weight_by_tau": {
            m: [
                round(float(w_eff_sum[b, t] / max(n_scored, 1)), 6)
                for t in range(LGBM_TAUS.size)
            ]
            for b, m in enumerate(STACK2_BASES)
        },
        # Headline diagnostic: mean effective weight per base over origins+taus.
        "mean_weight": {
            m: round(float(w_eff_sum[b].mean() / max(n_scored, 1)), 6)
            for b, m in enumerate(STACK2_BASES)
        },
        # Mean weight conditional on the base being finite that origin.
        "mean_weight_when_finite": {
            m: round(
                float(w_fin_sum[b].mean() / max(int(finite_cnt[b]), 1)), 6
            )
            for b, m in enumerate(STACK2_BASES)
        },
    }
    return {
        "crps_col": crps,
        "pin_cols": pin,
        "target_time_ns": tt,
        "bar_interval_ns": np.asarray(interval, dtype=np.int64),
        "stack_diag": diag,
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


def compute_column(bars_path: Path, cfg: dict) -> tuple[dict[str, np.ndarray], dict]:
    """(crps[R], pin[R,T], target_time_ns[R]) over the shard's origin grid."""
    frame = pl.read_parquet(bars_path)
    event_times, interval = validate_bars(frame)
    closes = frame["close"].to_numpy().astype(float)
    vols = frame["volume"].to_numpy().astype(float)
    out = _stack2_column(closes, event_times, interval, cfg, vols)
    diag = out.pop("stack_diag")
    return out, diag


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
    out, diag = compute_column(bars, cfg)
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
        "stack_params": {
            "warmup": STACK_WARMUP,
            "buffer": STACK_BUFFER,
            "iters": STACK_ITERS,
            "eta": STACK_ETA,
        },
        "stack_diag": diag,
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
        f"scored={diag['n_scored']} banked={diag['n_banked_all_finite']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
