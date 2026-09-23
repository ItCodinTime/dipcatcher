"""Causal quantile forecasters and distribution-to-weight policy mapping.

This is the *production* twin of the research lane columns: the same math that
scored in the verified arena (``dip_fhs``, ``dip_evt``, ``dip_garch_t``,
``dip_egarch_l``, ``dip_empirical_long``, ``dip_ewma_emp``) re-expressed as
per-origin callables over a close series. Every forecaster at bar index ``i``
consumes only ``closes[: i + 1]`` — the trailing ``window`` returns — and emits
a predictive quantile vector for the *next* close-to-close return.

The policy mapper converts a per-asset quantile row into target weights for the
paper loop's ``WeightFn``/panel contract. Signal conventions:

* ``mu``    — distributional mean estimate: mean of the tau-quantile grid.
* ``disp``  — dispersion estimate: ``(q_hi - q_lo) / z_width`` of the central
              band (default 5–95% ⇒ z_width = 2·1.6449 for a normal law — used
              only as a consistent dispersion unit, not a normality claim).
* ``edge``  — ``mu / disp``, a per-bar predicted Sharpe-like score.

Modes:
* ``long_flat``   — the dipcatcher book: long when ``mu`` clears the modeled
                    round-trip cost gate, else flat. No shorts.
* ``symmetric``   — long/short by sign of ``mu`` beyond the same gate.

Sizing is ``kappa * edge`` clipped to ``name_cap``, gross-normalized to
``gross_target`` when the book overflows. ``deadband`` suppresses re-emission
of a target within ``deadband`` of the previously emitted target (turnover
control at the signal layer; execution costs are still charged by the broker).

Fail-closed contract: insufficient history, a failed fit, or a non-monotone /
non-finite quantile row produces NO row for that origin (the paper loop treats
a missing name as "no new target" — the prior target carries, per the sparse
rebalance semantics) and increments the lane's failure counter.

research-only; no live-PnL claim — simulated fills only.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import polars as pl
from scipy import stats as st

# ---------------------------------------------------------------------------
# Forecasters: (rets_window, taus) -> quantile vector | raise -> honest skip
# ---------------------------------------------------------------------------

DEFAULT_TAUS: tuple[float, ...] = tuple(
    float(x) for x in np.linspace(0.05, 0.95, 19)
)


def _ewma_next_sigma(rets: np.ndarray, lam: float = 0.94) -> float:
    """RiskMetrics EWMA next-bar sigma (mirrors scripts/sota_eval_kronos)."""
    r = np.asarray(rets, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    w = lam ** np.arange(r.size - 1, -1, -1.0)
    var = float((1.0 - lam) * np.sum(w * r * r) + (lam ** r.size) * r[0] ** 2)
    return float(np.sqrt(var)) if var > 0 else float("nan")


def _arch_fit(
    rets_pct: np.ndarray,
    vol: Literal["GARCH", "ARCH", "EGARCH", "FIGARCH", "APARCH", "HARCH"],
    dist: Literal[
        "normal", "gaussian", "t", "studentst", "skewstudent", "skewt",
        "ged", "generalized error",
    ],
    o: int,
):
    """Shared arch fit helper (mirrors scripts/sota_eval_kronos._arch_fit)."""
    from arch import arch_model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = arch_model(
            rets_pct, mean="Constant", vol=vol, p=1, o=o, q=1, dist=dist, rescale=False
        )
        return model.fit(disp="off", show_warning=False, options={"maxiter": 300})


def _require_monotone(q: np.ndarray, n_tau: int) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q.shape != (n_tau,) or not np.isfinite(q).all():
        raise ValueError("non-finite or mis-shaped quantile row")
    if np.any(np.diff(q) < -1e-12):
        raise ValueError("non-monotone quantile row")
    return q


def empirical_quantiles(rets: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """``dip_empirical_long`` — plain empirical quantiles of the window."""
    if rets.size < 60:
        raise ValueError("insufficient history")
    return _require_monotone(np.quantile(rets, taus), taus.size)


def ewma_emp_quantiles(rets: np.ndarray, taus: np.ndarray, lam: float = 0.97) -> np.ndarray:
    """``dip_ewma_emp`` — EWMA-weighted empirical quantile function."""
    x = np.asarray(rets, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 60:
        raise ValueError("insufficient history")
    w = lam ** np.arange(x.size - 1, -1, -1.0)
    w /= w.sum()
    order = np.argsort(x)
    xs, ws = x[order], w[order]
    cumw = np.cumsum(ws)
    q = np.array([xs[min(int(np.searchsorted(cumw, t)), x.size - 1)] for t in taus])
    return _require_monotone(q, taus.size)


def garch_t_quantiles(rets: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """``dip_garch_t`` — GARCH(1,1)-t scaled Student-t quantiles."""
    if rets.size < 200:
        raise ValueError("insufficient history")
    fit = _arch_fit(rets * 100.0, vol="GARCH", dist="t", o=0)
    nu = float(fit.params["nu"])
    mu = float(fit.params.get("mu", 0.0)) / 100.0
    sig = float(np.sqrt(fit.forecast(horizon=1).variance.iloc[-1, 0])) / 100.0
    scale = sig * np.sqrt((nu - 2.0) / nu) if nu > 2.0 else np.nan
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("degenerate garch_t scale")
    return _require_monotone(st.t.ppf(taus, nu, loc=mu, scale=scale), taus.size)


def fhs_quantiles(rets: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """``dip_fhs`` — GJR-GARCH(1,1,1)-normal filtered historical simulation."""
    if rets.size < 200:
        raise ValueError("insufficient history")
    fit = _arch_fit(rets * 100.0, vol="GARCH", dist="normal", o=1)
    sig_next = float(np.sqrt(fit.forecast(horizon=1).variance.iloc[-1, 0])) / 100.0
    mu = float(fit.params.get("mu", 0.0)) / 100.0
    cond_vol = np.asarray(fit.conditional_volatility, dtype=float) / 100.0
    resid = np.asarray(fit.resid, dtype=float) / 100.0
    ok = cond_vol > 0.0
    z = resid[ok] / cond_vol[ok]
    z = z[np.isfinite(z)]
    if z.size < 20 or not np.isfinite(sig_next) or sig_next <= 0.0:
        raise ValueError("fhs residual set too small")
    return _require_monotone(np.quantile(mu + sig_next * z, taus), taus.size)


_EGARCH_SIG_CAP = 1.0  # >100%/bar 1-step sigma is definitionally broken


def _egarch_scaled_t(fit) -> tuple[float, float, float]:
    nu = float(fit.params["nu"])
    mu = float(fit.params.get("mu", 0.0)) / 100.0
    sig = float(np.sqrt(fit.forecast(horizon=1).variance.iloc[-1, 0])) / 100.0
    if not np.isfinite(sig) or sig <= 0.0 or sig > _EGARCH_SIG_CAP:
        raise ValueError(f"degenerate egarch sigma {sig!r}")
    scale = sig * np.sqrt((nu - 2.0) / nu) if nu > 2.0 else np.nan
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("degenerate egarch scale")
    return mu, scale, nu


def egarch_l_quantiles(rets: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """``dip_egarch_l`` — EGARCH(1,1)-t o=1 with o=0 fallback on degeneracy."""
    if rets.size < 200:
        raise ValueError("insufficient history")
    try:
        fit = _arch_fit(rets * 100.0, vol="EGARCH", dist="t", o=1)
        mu, scale, nu = _egarch_scaled_t(fit)
    except Exception:
        fit = _arch_fit(rets * 100.0, vol="EGARCH", dist="t", o=0)
        mu, scale, nu = _egarch_scaled_t(fit)
    return _require_monotone(st.t.ppf(taus, nu, loc=mu, scale=scale), taus.size)


_EVT_TAIL_MASS = 0.10
_EVT_MIN_EXCEEDANCES = 8
_EVT_XI_CLIP = 0.5
_EVT_SCALE_LO, _EVT_SCALE_HI = 0.7, 1.4


def _gpd_mom(exceedances: np.ndarray) -> tuple[float, float]:
    x = np.asarray(exceedances, dtype=float)
    x = x[np.isfinite(x) & (x > 0.0)]
    if x.size < _EVT_MIN_EXCEEDANCES:
        raise ValueError("insufficient exceedances")
    m, v = float(np.mean(x)), float(np.var(x, ddof=1))
    if not np.isfinite(m) or not np.isfinite(v) or m <= 0.0 or v <= 0.0:
        raise ValueError("unstable gpd moments")
    ratio = m * m / v
    xi, beta = 0.5 * (1.0 - ratio), 0.5 * m * (ratio + 1.0)
    if not np.isfinite(xi) or not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("unstable gpd fit")
    return float(np.clip(xi, -_EVT_XI_CLIP, _EVT_XI_CLIP)), float(beta)


def evt_quantiles(rets: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """``dip_evt`` — POT-GPD tails + empirical body + EWMA rescale."""
    x = np.asarray(rets, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 100:
        raise ValueError("insufficient history")
    try:
        u_lo = float(np.quantile(x, _EVT_TAIL_MASS))
        u_hi = float(np.quantile(x, 1.0 - _EVT_TAIL_MASS))
        if not (np.isfinite(u_lo) and np.isfinite(u_hi)) or u_lo >= u_hi:
            raise ValueError("unstable thresholds")
        xi_lo, beta_lo = _gpd_mom(u_lo - x[x < u_lo])
        xi_hi, beta_hi = _gpd_mom(x[x > u_hi] - u_hi)
        out = np.empty(taus.size)
        lo = taus < _EVT_TAIL_MASS
        hi = taus > 1.0 - _EVT_TAIL_MASS
        body = ~(lo | hi)
        if body.any():
            out[body] = np.quantile(x, taus[body])
        if lo.any():
            p = taus[lo] / _EVT_TAIL_MASS
            out[lo] = (
                u_lo + beta_lo * np.log(p)
                if abs(xi_lo) < 1e-12
                else u_lo - (beta_lo / xi_lo) * (p ** (-xi_lo) - 1.0)
            )
        if hi.any():
            p = (1.0 - taus[hi]) / _EVT_TAIL_MASS
            out[hi] = (
                u_hi - beta_hi * np.log(p)
                if abs(xi_hi) < 1e-12
                else u_hi + (beta_hi / xi_hi) * (p ** (-xi_hi) - 1.0)
            )
        q = np.maximum.accumulate(out)
    except Exception:
        # Disclosed fallback in the lane: plain empirical quantile function.
        q = np.quantile(x, taus)
    sd, ew = float(np.std(x, ddof=1)), _ewma_next_sigma(x)
    scale = (
        float(np.clip(ew / sd, _EVT_SCALE_LO, _EVT_SCALE_HI))
        if np.isfinite(sd) and np.isfinite(ew) and sd > 0.0 and ew > 0.0
        else 1.0
    )
    return _require_monotone(q * scale, taus.size)


Forecaster = Callable[[np.ndarray, np.ndarray], np.ndarray]

FORECASTERS: dict[str, tuple[Forecaster, int]] = {
    # name -> (fn, min_returns_history)
    "empirical": (empirical_quantiles, 60),
    "ewma_emp": (ewma_emp_quantiles, 60),
    # lam variants: EWMA memory is the real robustness axis for ewma_emp —
    # lam^k makes trailing-window length beyond ~3/(1-lam) a no-op.
    "ewma_emp94": (partial(ewma_emp_quantiles, lam=0.94), 60),
    "ewma_emp99": (partial(ewma_emp_quantiles, lam=0.99), 60),
    "evt": (evt_quantiles, 100),
    "garch_t": (garch_t_quantiles, 200),
    "fhs": (fhs_quantiles, 200),
    "egarch_l": (egarch_l_quantiles, 200),
}


# ---------------------------------------------------------------------------
# Policy: quantile row -> target weights
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantilePolicy:
    """Distribution-to-position mapping (parameters are part of the receipt)."""

    mode: str = "long_flat"  # long_flat | symmetric
    kappa: float = 0.30  # size per unit of predicted per-bar Sharpe
    gross_target: float = 1.0  # gross cap on the emitted book
    name_cap: float = 0.25  # per-name |weight| cap
    cost_gate: float = 0.0020  # gate threshold (units depend on gate_on)
    deadband: float = 0.01  # re-emit only when |Δtarget| >= deadband
    band_lo: float = 0.05  # dispersion band lower tau
    band_hi: float = 0.95  # dispersion band upper tau
    gate_on: str = "mu"  # "mu": |mu| return gate | "edge": |mu/disp| z-gate
    sizing: str = "edge"  # "edge": w=kappa*edge | "risk": w=kappa*edge/disp (vol-parity)
    book_vol_target: float | None = None  # scale book so Σ|w·disp| <= this
    tail_gate: float | None = None  # longs need q_lo > -tail_gate; shorts q_hi < +tail_gate (return units)
    persist_bars: int = 1  # consecutive gate-passing dates before (re-)entry
    mkt_disp_cut: float | None = None  # flat book when median cross-asset disp exceeds this

    def __post_init__(self) -> None:
        if self.mode not in {"long_flat", "symmetric"}:
            raise ValueError("mode must be long_flat or symmetric")
        if self.gate_on not in {"mu", "edge"}:
            raise ValueError("gate_on must be 'mu' or 'edge'")
        if self.sizing not in {"edge", "risk"}:
            raise ValueError("sizing must be 'edge' or 'risk'")
        for name in ("book_vol_target", "tail_gate", "mkt_disp_cut"):
            v = getattr(self, name)
            if v is not None and (not np.isfinite(float(v)) or float(v) < 0):
                raise ValueError(f"{name} must be finite and non-negative")
        if int(self.persist_bars) < 1:
            raise ValueError("persist_bars must be >= 1")
        for name in ("kappa", "gross_target", "name_cap", "cost_gate", "deadband"):
            if not np.isfinite(float(getattr(self, name))) or float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not (0.0 < self.band_lo < self.band_hi < 1.0):
            raise ValueError("band taus must satisfy 0 < lo < hi < 1")


def quantile_moments(q: np.ndarray, taus: np.ndarray, policy: QuantilePolicy) -> tuple[float, float]:
    """(mu, disp) from a quantile row: grid mean + central-band dispersion."""
    q = np.asarray(q, dtype=float)
    taus = np.asarray(taus, dtype=float)
    mu = float(np.mean(q))
    i_lo = int(np.argmin(np.abs(taus - policy.band_lo)))
    i_hi = int(np.argmin(np.abs(taus - policy.band_hi)))
    z_width = float(st.norm.ppf(policy.band_hi) - st.norm.ppf(policy.band_lo))
    disp = float((q[i_hi] - q[i_lo]) / z_width) if z_width > 0 else float("nan")
    return mu, disp


def weights_from_quantiles(
    q_rows: dict[str, np.ndarray],
    taus: np.ndarray,
    policy: QuantilePolicy,
    prev_targets: dict[str, float] | None = None,
    streaks: dict[str, int] | None = None,
) -> dict[str, float]:
    """Map per-name quantile rows to target weights under ``policy``.

    Names whose quantile row is absent/non-finite simply do not appear —
    under the loop's sparse-rebalance semantics their prior target carries.
    ``prev_targets`` supplies the deadband reference (the last *emitted*
    book, not the broker's drifted positions). The returned dict is the
    complete target map for the date: the engine flattens names absent
    from it, so names whose signal went quiet are omitted *because* their
    desired target is zero, while deadbanded names re-emit their prior
    target to hold.
    """
    prev_targets = prev_targets or {}
    raw: dict[str, float] = {}
    disp_map: dict[str, float] = {}
    for sid, q in q_rows.items():
        try:
            q = _require_monotone(np.asarray(q, dtype=float), np.asarray(taus).size)
        except ValueError:
            continue
        mu, disp = quantile_moments(q, taus, policy)
        if not np.isfinite(mu) or not np.isfinite(disp) or disp <= 0.0:
            continue
        disp_map[sid] = disp
        edge = mu / disp
        gate_metric = abs(mu) if policy.gate_on == "mu" else abs(edge)
        passed = gate_metric > policy.cost_gate
        if passed and policy.tail_gate is not None:
            # Tail conviction: the forecast's own bound must be benign in the
            # trade direction — longs need q_lo > -tail_gate, shorts need
            # q_hi < +tail_gate (return units). Uses the full shape.
            i_lo = int(np.argmin(np.abs(taus - policy.band_lo)))
            i_hi = int(np.argmin(np.abs(taus - policy.band_hi)))
            passed = not (
                (edge > 0 and q[i_lo] <= -policy.tail_gate)
                or (edge < 0 and q[i_hi] >= policy.tail_gate)
            )
        if streaks is not None:
            streaks[sid] = streaks.get(sid, 0) + 1 if passed else 0
        prior = float(prev_targets.get(sid, 0.0))
        confirmed = policy.persist_bars <= 1 or abs(prior) > 1e-12 or (
            streaks is not None and streaks.get(sid, 0) >= policy.persist_bars
        )
        if not passed or not confirmed:
            w = 0.0
        else:
            raw_w = policy.kappa * edge if policy.sizing == "edge" else policy.kappa * edge / disp
            w = float(np.clip(raw_w, -policy.name_cap, policy.name_cap))
            if policy.mode == "long_flat":
                w = max(w, 0.0)
        raw[sid] = w
    if policy.book_vol_target is not None:
        # Book-level vol target (both directions): Σ|w_i·disp_i| is a
        # conservative (perfect-corr) book-vol proxy; scale weights toward
        # constant risk — up in quiet regimes, down in turbulent ones. The
        # gross_target cap below remains the hard bound on scaling up.
        book_vol = sum(abs(w) * disp_map.get(sid, 0.0) for sid, w in raw.items())
        if book_vol > 0.0 and policy.book_vol_target > 0.0:
            scale = policy.book_vol_target / book_vol
            raw = {sid: w * scale for sid, w in raw.items()}
    gross = sum(abs(w) for w in raw.values())
    if gross > policy.gross_target > 0.0:
        scale = policy.gross_target / gross
        raw = {sid: w * scale for sid, w in raw.items()}
    out: dict[str, float] = {}
    for sid, w in raw.items():
        prior = float(prev_targets.get(sid, 0.0))
        if abs(w - prior) < policy.deadband:
            # Deadband: hold the prior target. The name must still be emitted —
            # the engine flattens names absent from a present date's target map.
            w = prior
        if abs(w) < 1e-12 and abs(prior) < 1e-12:
            continue  # flat -> flat: keep the panel sparse
        out[sid] = w
    return out


# ---------------------------------------------------------------------------
# Panel computation: causal per-origin quantiles over a close series
# ---------------------------------------------------------------------------


def compute_quantile_panel(
    closes: np.ndarray,
    spec: str,
    taus: np.ndarray,
    window: int,
    min_history: int | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Causal quantile panel ``[n_bars, n_taus]`` for one asset.

    Row ``i`` forecasts bar ``i + 1``'s close-close return using
    ``diff(closes[max(0, i-window+1):i+1])`` — at most ``window`` trailing
    returns, matching the lane evaluators' ``rets_long`` slice. Rows before
    the warmup are NaN. Returns ``(panel, stats)`` where stats counts
    emitted/failed/warmup rows.
    """
    spec = spec.strip().lower()
    if spec not in FORECASTERS:
        raise ValueError(f"unknown forecaster spec {spec!r}; have {sorted(FORECASTERS)}")
    fn, spec_min = FORECASTERS[spec]
    closes = np.asarray(closes, dtype=float)
    n = closes.size
    taus = np.asarray(taus, dtype=float)
    panel = np.full((n, taus.size), np.nan)
    stats = {"emitted": 0, "failed": 0, "warmup": 0}
    need = max(int(min_history or spec_min), 2)
    for i in range(n - 1):  # last bar has no next bar to forecast
        start = max(0, i + 1 - (window + 1))
        rets = np.diff(closes[start : i + 1]) / closes[start:i]
        rets = rets[np.isfinite(rets)]
        if rets.size < need:
            stats["warmup"] += 1
            continue
        try:
            panel[i] = fn(rets, taus)
            stats["emitted"] += 1
        except Exception:  # noqa: BLE001 — honest skip, row stays NaN
            stats["failed"] += 1
    return panel, stats


def quantile_panels_to_weights(
    panels: dict[str, np.ndarray],
    event_times: dict[str, np.ndarray],
    policy: QuantilePolicy,
    taus: np.ndarray,
) -> pl.DataFrame:
    """Apply ``policy`` sequentially per date → target-weight panel.

    ``panels[sid]`` is ``[n_bars, n_taus]`` aligned to ``event_times[sid]``.
    The decision at index ``i`` (emitted at ``event_times[sid][i]``) uses row
    ``i`` — computed from closes ``<= i`` — so no future data enters. Weight
    rows are keyed by the *decision* bar's event_time (the loop executes at
    the next bar under next-open convention). The deadband state is the last
    emitted target per name.
    """
    taus = np.asarray(taus, dtype=float)
    prev: dict[str, float] = {}
    streaks: dict[str, int] = {}
    out_t: list[Any] = []
    out_s: list[str] = []
    out_w: list[float] = []
    # Union timeline over assets that may not share a calendar: index each
    # asset's own event_times so a decision row is emitted only on bars the
    # asset actually prints (missing asset on a date -> no new target -> the
    # loop's sparse-rebalance carry semantics hold honestly).
    row_index: dict[str, dict[Any, int]] = {
        sid: {t: i for i, t in enumerate(event_times[sid])} for sid in panels
    }
    timeline = sorted({t for sid in panels for t in event_times[sid]})
    for t_i in timeline:
        q_rows = {}
        for sid, panel in panels.items():
            i = row_index[sid].get(t_i)
            if i is None:
                continue
            q = panel[i]
            if np.isfinite(q).all():
                q_rows[sid] = q
        if not q_rows:
            continue
        if policy.mkt_disp_cut is not None:
            # Market-wide vol breaker: median forecast dispersion across names
            # above the cut -> flatten the whole book for this decision date.
            # Causal (same-bar quantiles only); emits explicit zeros so held
            # names de-risk rather than carry under sparse semantics.
            disps = []
            for q in q_rows.values():
                _, d = quantile_moments(np.asarray(q, dtype=float), taus, policy)
                if np.isfinite(d):
                    disps.append(d)
            if disps and float(np.median(disps)) > policy.mkt_disp_cut:
                targets = {sid: 0.0 for sid in q_rows if abs(prev.get(sid, 0.0)) > 1e-12}
                streaks.clear()
                prev = {}
                for sid, w in targets.items():
                    out_t.append(t_i)
                    out_s.append(sid)
                    out_w.append(w)
                continue
        targets = weights_from_quantiles(q_rows, taus, policy, prev, streaks)
        # The emitted dict IS the carried book: names dropped from it are
        # flattened by the engine, so the deadband reference resets wholesale.
        prev = dict(targets)
        for sid, w in targets.items():
            out_t.append(t_i)
            out_s.append(sid)
            out_w.append(w)
    return pl.DataFrame(
        {"event_time": out_t, "security_id": out_s, "target_weight": out_w}
    ).sort(["event_time", "security_id"])


def load_deep_bars(
    bars_root: Path, symbols: list[str], interval: str, *, shared_calendar: bool = True
) -> pl.DataFrame:
    """Load ``<sym>_<interval>_deep.parquet`` (or plain) into a long panel.

    Derives ``adv`` (close×volume in USD terms — Binance volume is base units)
    and ``vol_20`` (20-bar rolling close-close sigma) which the paper loop and
    risk gate consume. Point-in-time: signals use ``event_time`` causality;
    ``available_time`` is asserted to not precede ``event_time`` when present.

    ``shared_calendar`` clips every asset to ``[max(first_t), min(last_t)]``
    — the span where all names print — so a held name never loses marks at
    the book's ragged edges (the loop correctly fail-closes on stale marks;
    ragged file boundaries are a data artifact, not a trading signal).
    Mid-series gaps are NOT repaired and still fail closed downstream.
    """
    frames = []
    for sym in symbols:
        sym_l = sym.lower()
        for suffix in (f"{sym_l}_{interval}_deep.parquet", f"{sym_l}_{interval}.parquet"):
            path = bars_root / suffix
            if path.is_file():
                df = pl.read_parquet(path)
                df = df.with_columns(pl.lit(sym.upper()).alias("security_id"))
                frames.append(df)
                break
        else:
            raise FileNotFoundError(f"{sym}: no {interval} bars under {bars_root}")
    panel = pl.concat(frames, how="diagonal_relaxed").sort(["security_id", "event_time"])
    if shared_calendar:
        bounds = panel.group_by("security_id").agg(
            pl.col("event_time").min().alias("t0"),
            pl.col("event_time").max().alias("t1"),
        )
        t0 = cast(Any, bounds["t0"].max())
        t1 = cast(Any, bounds["t1"].min())
        if t0 is None or t1 is None or t0 >= t1:
            raise ValueError("no shared calendar span across symbols")
        panel = panel.filter(
            (pl.col("event_time") >= t0) & (pl.col("event_time") <= t1)
        )
    if "available_time" in panel.columns:
        bad = panel.filter(pl.col("available_time") < pl.col("event_time"))
        if bad.height:
            raise ValueError(f"{bad.height} bars with available_time < event_time (PIT violation)")
    if "adv" not in panel.columns:
        panel = panel.with_columns((pl.col("close") * pl.col("volume")).alias("adv"))
    if "vol_20" not in panel.columns:
        panel = panel.with_columns(
            pl.col("close")
            .pct_change()
            .rolling_std(20, min_samples=5)
            .over("security_id")
            .alias("vol_20")
        )
    return panel
