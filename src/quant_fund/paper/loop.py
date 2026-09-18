"""Paper / shadow event loop.

Champion slot: simulated capital + fills.
Shadow challenger slot(s): weights tracked, orders recorded, no capital unless promoted.
Supports a primary shadow plus optional named multi-challengers with rolling L1 metrics.

Uses ReplayClock over bar times by default (accelerated paper). WallClock is
available for live-clock paper mode without broker connectivity.

Supports multi-day persistence/resume via broker_state.json on the ledger.
Promotion dry-run compares champion vs shadow; never moves live capital.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import uuid4

import numpy as np
import polars as pl

from quant_fund.config.models import AppConfig, FillConvention
from quant_fund.execution.simulated_broker import OrderRecord, SimulatedBroker
from quant_fund.metrics.analytics import (
    analytics_export_digest,
    book_diagnostics,
    equity_curve_analytics,
    export_analytics_dict,
    rolling_challenger_metrics,
    validate_analytics_export,
)
from quant_fund.monitoring.kill_switch import ENABLED, HALT_NEW_ORDERS
from quant_fund.paper.clock import ReplayClock, WallClock
from quant_fund.paper.ledger import (
    PaperLedger,
    load_broker_state,
    promotion_dry_run,
    validate_promotion_dry_run_receipt,
)
from quant_fund.portfolio.portfolio_conformal import SplitPortfolioCQR, gaussian_band
from quant_fund.utils.hashing import hash_bytes

WeightFn = Callable[[datetime, AppConfig], dict[str, float]]


@dataclass
class PaperLoopResult:
    run_id: str
    champion_equity: pl.DataFrame
    shadow_equity: pl.DataFrame
    orders: pl.DataFrame
    metrics: dict[str, Any]
    paths: dict[str, Any]
    source_note: str
    divergence: dict[str, float]
    promotion: dict[str, Any] | None = None


class StaleValuationError(RuntimeError):
    """Raised when bounded mark carry-forward expires for a held position."""


def _require_valuation_marks(broker: SimulatedBroker, prices: dict[str, float]) -> None:
    missing = sorted(
        security_id
        for security_id, shares in broker.shares.items()
        if shares != 0.0 and security_id not in prices
    )
    if missing:
        raise StaleValuationError(
            "bounded valuation mark expired for held security(s): " + ", ".join(missing)
        )


def _merge_divergence_summary(
    prior: dict[str, Any] | None, current: list[float], *, total_steps: int
) -> tuple[float, float]:
    """Preserve full-run divergence statistics across a resumed paper run."""
    finite_current = [float(value) for value in current if np.isfinite(value)]
    if not finite_current:
        if prior is not None:
            return float(prior.get("mean_l1", float("nan"))), float(
                prior.get("max_l1", float("nan"))
            )
        return float("nan"), float("nan")
    prior_n = 0
    prior_mean = float("nan")
    prior_max = float("nan")
    if prior is not None:
        try:
            # The prior receipt's mean covers every step before this run; the
            # current samples are the ones taken since resume. Weight the prior
            # mean by its own sample count (cumulative steps minus current),
            # not by prior["n_steps"] minus current (which double-discounts).
            prior_n = max(int(total_steps) - len(finite_current), 0)
            prior_mean = float(prior.get("mean_l1", float("nan")))
            prior_max = float(prior.get("max_l1", float("nan")))
        except (TypeError, ValueError):
            prior_n = 0
    current_sum = float(np.sum(finite_current))
    if prior_n > 0 and np.isfinite(prior_mean):
        mean = (prior_mean * prior_n + current_sum) / (prior_n + len(finite_current))
    else:
        mean = current_sum / len(finite_current)
    maxima = finite_current + ([prior_max] if prior_n > 0 and np.isfinite(prior_max) else [])
    return float(mean), float(max(maxima))


def _valid_price(value: object) -> float | None:
    if value is None:
        return None
    try:
        price = float(str(value))
    except (TypeError, ValueError):
        return None
    return price if np.isfinite(price) and price > 0 else None


def _bar_maps(
    day: pl.DataFrame, *, use_open: bool
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    marks: dict[str, float] = {}
    close_marks: dict[str, float] = {}
    advs: dict[str, float] = {}
    vols: dict[str, float] = {}
    for row in day.iter_rows(named=True):
        sid = str(row["security_id"])
        raw = _valid_price(row.get("open") if use_open else row.get("close"))
        if raw is not None:
            marks[sid] = raw
        tr = _valid_price(row.get("close_total_return"))
        cl = _valid_price(row.get("close"))
        if tr is not None:
            close_marks[sid] = tr
        elif cl is not None:
            close_marks[sid] = cl
        advs[sid] = _valid_price(row.get("adv")) or 1.0
        vols[sid] = _valid_price(row.get("vol_20")) or 0.02
    return marks, close_marks, advs, vols


def _bounded_valuation_marks(
    broker: SimulatedBroker,
    fresh_marks: dict[str, float],
    mark_ages: dict[str, int],
    *,
    max_stale_bars: int,
) -> dict[str, float]:
    """Merge fresh marks with bounded carry-forward marks for held positions.

    Execution remains fresh-mark-only; this map is exclusively for valuation and
    persistence.  A resumed broker has no trustworthy per-name age metadata, so
    its restored marks are treated as observed at the first resumed bar and then
    age deterministically from there.
    """
    if max_stale_bars < 0:
        raise ValueError("max_stale_bars must be non-negative")
    observed = set(fresh_marks)
    for sid in broker.last_marks:
        # Restored marks have no persisted age; the first resumed bar is the
        # reference bar, so an absent mark starts at age zero.
        mark_ages.setdefault(sid, -1)
    for sid in list(mark_ages):
        mark_ages[sid] = 0 if sid in observed else mark_ages[sid] + 1
    for sid in observed:
        mark_ages[sid] = 0

    carried = dict(fresh_marks)
    for sid, price in broker.last_marks.items():
        if sid not in observed and mark_ages.get(sid, 0) <= max_stale_bars:
            carried[sid] = price
    return carried


def build_scaled_challenger_weights(
    champion_weights: pl.DataFrame,
    scales: list[float] | tuple[float, ...] | None,
) -> dict[str, pl.DataFrame]:
    """Build named scaled weight panels for ``challenger_weights``.

    Primary shadow is *not* included — callers keep ``shadow_weights`` separate
    for ledger / promotion_dry_run compatibility. Empty scales → {}.
    """
    out: dict[str, pl.DataFrame] = {}
    if not scales:
        return out
    for i, scale in enumerate(scales):
        name = f"scale_{float(scale):g}"
        if name in out:
            name = f"{name}_{i}"
        out[name] = champion_weights.with_columns(
            (pl.col("target_weight") * float(scale)).alias("target_weight")
        )
    return out


def _validate_weight_panel(weights: pl.DataFrame) -> None:
    required = {"event_time", "security_id", "target_weight"}
    missing = required.difference(weights.columns)
    if missing:
        raise ValueError(f"target weights missing required columns: {sorted(missing)}")
    duplicates = (
        weights.group_by(["event_time", "security_id"])
        .agg(pl.len().alias("_n"))
        .filter(pl.col("_n") > 1)
    )
    if duplicates.height:
        raise ValueError("duplicate target weights for event_time/security_id")


def _default_weights_from_panel(
    weights: pl.DataFrame,
) -> WeightFn:
    _validate_weight_panel(weights)

    def _fn(dt: datetime, _cfg: AppConfig) -> dict[str, float]:
        rows = weights.filter(pl.col("event_time") == dt)
        return {
            str(r["security_id"]): float(r["target_weight"]) for r in rows.iter_rows(named=True)
        }

    return _fn


def _weight_l1_divergence(champ: dict[str, float], shadow: dict[str, float]) -> float:
    ids = set(champ) | set(shadow)
    if not ids:
        return 0.0
    return float(sum(abs(champ.get(i, 0.0) - shadow.get(i, 0.0)) for i in ids))


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _paper_resume_fingerprint(bars: pl.DataFrame, config: AppConfig, cutoff: datetime) -> str:
    """Bind config and the already-processed bar prefix for safe resume."""
    sort_keys = ["event_time"]
    if "security_id" in bars.columns:
        sort_keys.append("security_id")
    prefix = bars.filter(pl.col("event_time") <= cutoff).sort(sort_keys)
    payload = {
        "config": config.dump(),
        "columns": sorted(prefix.columns),
        "rows": prefix.height,
        "row_hashes": [int(value) for value in prefix.hash_rows().to_list()],
    }
    return hash_bytes(json.dumps(payload, sort_keys=True, default=str).encode())


def _paper_portfolio_conformal(returns: np.ndarray) -> dict[str, Any]:
    """Book-level conformal on paper NAV returns (research diagnostic)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    out: dict[str, Any] = {
        "n": int(r.size),
        "role": "paper_book_research_diagnostic",
        "live_pnl_claim": False,
    }
    if r.size < 30:
        out["status"] = "insufficient_sample"
        return out
    # Naive Gaussian band then SplitPortfolioCQR on the scalar series.
    mu = float(np.mean(r[: max(r.size // 2, 8)]))
    sig = float(np.std(r[: max(r.size // 2, 8)], ddof=1) or 1e-4)
    lo_all = np.empty(r.size)
    hi_all = np.empty(r.size)
    for i in range(r.size):
        lo_all[i], hi_all[i] = gaussian_band(mu, sig, 0.10)
    n_cal = max(r.size // 3, 10)
    cqr = SplitPortfolioCQR(0.10).calibrate(r[:n_cal], lo_all[:n_cal], hi_all[:n_cal])
    lo_t, hi_t = cqr.predict_sets(lo_all[n_cal:], hi_all[n_cal:])
    y = r[n_cal:]
    cover = float(np.mean((y >= lo_t) & (y <= hi_t))) if y.size else float("nan")
    out.update(
        {
            "status": "ok",
            "alpha": 0.10,
            "qhat": float(cqr.qhat) if cqr.qhat is not None else float("nan"),
            "coverage_test": cover,
            "n_cal": int(n_cal),
            "n_test": int(y.size),
            "mean_width": float(np.mean(hi_t - lo_t)) if y.size else float("nan"),
        }
    )
    return out


def run_paper_loop(
    bars: pl.DataFrame,
    config: AppConfig,
    *,
    champion_weights: pl.DataFrame | None = None,
    shadow_weights: pl.DataFrame | None = None,
    champion_fn: WeightFn | None = None,
    shadow_fn: WeightFn | None = None,
    challenger_weights: dict[str, pl.DataFrame] | None = None,
    challenger_fns: dict[str, WeightFn] | None = None,
    initial_nav: float = 1_000_000.0,
    run_id: str | None = None,
    max_steps: int | None = None,
    use_wall_clock: bool = False,
    resume: bool = False,
    resume_run_id: str | None = None,
    prefer_latest: bool = True,
    halt_after_steps: int | None = None,
    rolling_window: int = 10,
) -> PaperLoopResult:
    """Run paper (champion capital) + optional shadow challenger(s) (no capital).

    Fill convention follows ``config.execution.fill`` (default next_open):
    targets at decision date ``t`` execute at open of ``t+1``.

    Set ``prefer_latest=False`` (or resume) for sequential multi-day windows.

    When ``resume`` is True, restore champion/shadow cash+shares from
    ``broker_state.json`` for ``resume_run_id`` (or ``run_id``) and skip
    decision dates already processed (``last_decision``).

    ``challenger_weights`` / ``challenger_fns`` enable multi-challenger shadow
    metrics (no capital). ``halt_after_steps`` trips the kill switch mid-run
    after that many *new* steps (E2E infrastructure test).
    """
    resume_id = resume_run_id or getattr(config.paper, "resume_run_id", None) or run_id
    prior_state: dict[str, Any] | None = None
    if resume:
        if not resume_id:
            raise ValueError("resume=True requires resume_run_id or run_id")
        prior_state = load_broker_state(config.data.root, resume_id, config.paper.ledger_subdir)
        if prior_state is None:
            raise FileNotFoundError(f"no broker_state.json for run_id={resume_id}; cannot resume")
        if prior_state.get("run_id") != resume_id:
            raise ValueError("cannot resume from broker state with mismatched run_id")
        raw_step = prior_state.get("step", 0)
        if isinstance(raw_step, bool) or not isinstance(raw_step, int) or raw_step < 0:
            raise ValueError("cannot resume from broker state with invalid step")
        for cursor_name in ("last_decision", "last_exec"):
            raw_cursor = prior_state.get(cursor_name)
            if raw_cursor is not None and (
                not isinstance(raw_cursor, str) or _parse_iso(raw_cursor) is None
            ):
                raise ValueError(f"cannot resume from broker state with invalid {cursor_name}")
        raw_fingerprint = prior_state.get("resume_fingerprint")
        if raw_fingerprint is not None and (
            not isinstance(raw_fingerprint, str)
            or len(raw_fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in raw_fingerprint)
        ):
            raise ValueError("cannot resume from broker state with invalid resume_fingerprint")
        raw_mark_ages = prior_state.get("mark_ages", {})
        if raw_mark_ages is None:
            raw_mark_ages = {}
        if not isinstance(raw_mark_ages, dict) or any(
            not isinstance(security_id, str)
            or isinstance(age, bool)
            or not isinstance(age, int)
            or age < 0
            for security_id, age in raw_mark_ages.items()
        ):
            raise ValueError("cannot resume from broker state with invalid mark_ages")
        run_id = resume_id
    run_id = run_id or f"paper-{uuid4().hex[:10]}"

    px = bars
    if "adv" not in px.columns:
        px = px.with_columns((pl.col("close") * pl.col("volume")).alias("adv"))
    if "vol_20" not in px.columns:
        px = px.with_columns(pl.lit(0.02).alias("vol_20"))

    dates = sorted(px["event_time"].unique().to_list())
    if prior_state and prior_state.get("resume_fingerprint"):
        prior_cutoff = _parse_iso(prior_state.get("last_exec"))
        if prior_cutoff is not None:
            current_fingerprint = _paper_resume_fingerprint(px, config, prior_cutoff)
            if current_fingerprint != prior_state["resume_fingerprint"]:
                raise ValueError(
                    "resume input/config fingerprint mismatch; refusing to replay altered evidence"
                )
    use_next_open = (
        config.execution.fill is FillConvention.NEXT_OPEN
        and not config.execution.allow_close_auction
    )
    # Decision dates: all but last when next-open
    decision_dates = dates[:-1] if use_next_open and len(dates) > 1 else dates
    skip_after = _parse_iso(prior_state.get("last_decision")) if prior_state else None
    if skip_after is not None:
        decision_dates = [d for d in decision_dates if d > skip_after]
    if max_steps is not None:
        # Prefer the latest window for smoke benches (warmup/cov history).
        # Multi-day resume / named sequential runs use from-start / next-N.
        if resume or not prefer_latest:
            decision_dates = decision_dates[: int(max_steps)]
        else:
            decision_dates = decision_dates[-int(max_steps) :]

    if champion_fn is None:
        if champion_weights is None:
            raise ValueError("champion_weights or champion_fn required")
        champion_fn = _default_weights_from_panel(champion_weights)
    if shadow_fn is None and shadow_weights is not None:
        shadow_fn = _default_weights_from_panel(shadow_weights)

    # Multi-challenger map (name -> WeightFn). Primary "shadow" stays separate
    # for broker/ledger compatibility; extra challengers are metrics-only.
    extra_challengers: dict[str, WeightFn] = {}
    if challenger_fns:
        extra_challengers.update(challenger_fns)
    if challenger_weights:
        for name, panel_w in challenger_weights.items():
            if name not in extra_challengers:
                extra_challengers[str(name)] = _default_weights_from_panel(panel_w)
    # If only extras provided and no primary shadow, promote first as primary
    if shadow_fn is None and extra_challengers:
        first_name = next(iter(extra_challengers))
        shadow_fn = extra_challengers.pop(first_name)

    if prior_state and prior_state.get("champion"):
        try:
            champ = SimulatedBroker.from_state(config, prior_state["champion"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("cannot resume from invalid champion broker state") from exc
    else:
        champ = SimulatedBroker(
            config=config, initial_cash=initial_nav, slot="champion", allow_capital=True
        )
    shadow: SimulatedBroker | None = None
    if shadow_fn is not None:
        if prior_state and prior_state.get("shadow"):
            try:
                shadow = SimulatedBroker.from_state(config, prior_state["shadow"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("cannot resume from invalid shadow broker state") from exc
        else:
            shadow = SimulatedBroker(
                config=config, initial_cash=0.0, slot="shadow", allow_capital=False
            )

    ledger = PaperLedger(
        config.data.root,
        run_id,
        config.paper.ledger_subdir,
        load_existing=bool(resume),
    )
    prior_promo: dict[str, Any] | None = None
    if resume:
        prior_promo_path = ledger.root / "promotion_dry_run.json"
        if prior_promo_path.is_file():
            try:
                raw_prior_promo: Any = json.loads(prior_promo_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    "cannot resume from invalid promotion receipt: unreadable"
                ) from exc
            prior_errors = validate_promotion_dry_run_receipt(raw_prior_promo)
            if prior_errors:
                raise ValueError(
                    "cannot resume from invalid promotion receipt: " + ",".join(prior_errors)
                )
            prior_promo = raw_prior_promo
    sources = (
        {str(value).lower() for value in px["source"].drop_nulls().to_list()}
        if "source" in px.columns
        else set()
    )
    synthetic = (
        "synthetic" in sources if "source" in px.columns else config.data.source == "synthetic"
    )
    ledger.set_meta(
        data_source="SYNTHETIC" if synthetic else str(config.data.source),
        mode="paper",
        fill=config.execution.fill.value,
        initial_nav=initial_nav,
        has_shadow=shadow is not None,
        label="PAPER_SIMULATED" + ("_SYNTHETIC" if synthetic else ""),
        resumed=bool(resume),
        schema_version=PaperLedger.SCHEMA_VERSION,
    )

    if use_wall_clock:
        clock: WallClock | ReplayClock = WallClock()
    else:
        clock = ReplayClock(event_times=list(decision_dates))

    champ_eq: list[dict[str, Any]] = []
    shadow_eq: list[dict[str, Any]] = []
    all_orders: list[dict[str, Any]] = []
    div_series: list[float] = []
    challenger_divs: dict[str, list[float]] = {n: [] for n in extra_challengers}
    # A resumed non-enabled state is evidence of a prior kill trip (or an
    # explicitly halted run). Preserve the promotion veto across process
    # boundaries instead of resetting this aggregate metric on every resume.
    kill_tripped_mid_run = champ.kill.state != ENABLED or (
        shadow is not None and shadow.kill.state != ENABLED
    )
    step = int(prior_state.get("step", 0)) if prior_state else 0
    steps_this_run = 0
    last_decision: datetime | None = skip_after
    last_exec: datetime | None = _parse_iso(prior_state.get("last_exec")) if prior_state else None

    # Seed equity from prior ledger if resuming (for conformal / diagnostics continuity)
    if resume:
        prior_eq = ledger.load_equity()
        if prior_eq.height and "nav" in prior_eq.columns:
            champ_eq.extend(
                {
                    "event_time": row.get("event_time", row.get("asof")),
                    "nav": row.get("nav"),
                    "gross": row.get("gross", 0.0),
                    "net": row.get("net", 0.0),
                    "cash": row.get("cash", 0.0),
                    "slot": row.get("slot", "champion"),
                    "cash_nav_residual": row.get("cash_nav_residual", 0.0) or 0.0,
                }
                for row in prior_eq.to_dicts()
            )
        prior_shadow_eq = ledger.load_shadow_equity()
        if prior_shadow_eq.height:
            shadow_eq.extend(prior_shadow_eq.to_dicts())

    raw_resume_ages = prior_state.get("mark_ages", {}) if prior_state else {}
    if raw_resume_ages is None:
        raw_resume_ages = {}
    mark_ages: dict[str, int] = {
        str(security_id): int(age) for security_id, age in raw_resume_ages.items()
    }
    while not clock.exhausted():
        dt = clock.tick()
        if dt is None:
            break
        # Map decision date index into dates list
        try:
            i = dates.index(dt)
        except ValueError:
            # nearest
            nearest = next((j for j, d in enumerate(dates) if d >= dt), None)
            if nearest is None:
                break
            i = nearest
            dt = dates[i]
        exec_dt = dates[i + 1] if use_next_open and i + 1 < len(dates) else dt
        day = px.filter(pl.col("event_time") == exec_dt)
        if day.is_empty():
            continue
        marks, close_marks, advs, vols = _bar_maps(day, use_open=use_next_open)
        pretrade_marks = _bounded_valuation_marks(
            champ,
            marks,
            mark_ages,
            max_stale_bars=config.risk_gate.stale_price_bars,
        )
        # Never let an execution-day close leak into a next-open decision. Fresh
        # opens plus bounded prior marks are the only pre-trade valuation inputs.
        champ.mark(pretrade_marks)
        if shadow is not None:
            shadow.mark(pretrade_marks)

        c_tgt = champion_fn(dt, config)
        s_tgt = shadow_fn(dt, config) if shadow_fn is not None else {}
        div_series.append(_weight_l1_divergence(c_tgt, s_tgt))
        for cname, cfn in extra_challengers.items():
            challenger_divs[cname].append(_weight_l1_divergence(c_tgt, cfn(dt, config)))

        # Mid-run kill switch (infrastructure E2E): trip after N new steps.
        if (
            halt_after_steps is not None
            and not kill_tripped_mid_run
            and steps_this_run >= int(halt_after_steps)
        ):
            champ.kill.state = HALT_NEW_ORDERS
            if shadow is not None:
                shadow.kill.state = HALT_NEW_ORDERS
            kill_tripped_mid_run = True

        _require_valuation_marks(champ, pretrade_marks)
        nav = champ.nav(pretrade_marks)
        if nav <= 0:
            break

        # Champion orders + fills
        c_orders = champ.target_to_orders(c_tgt, marks, signal_time=dt, order_time=exec_dt, nav=nav)
        step_recs: list[OrderRecord] = []
        for order in c_orders:
            sid = order.security_id
            if sid not in marks:
                continue
            rec = champ.submit(
                order,
                price=marks[sid],
                nav=nav,
                adv_dollars=advs.get(sid, 1.0),
                sigma=vols.get(sid, 0.02),
            )
            step_recs.append(rec)
            all_orders.append(
                {
                    "asof": exec_dt,
                    "slot": "champion",
                    "order_id": rec.order.order_id,
                    "security_id": sid,
                    "side": rec.order.side.value,
                    "quantity": rec.order.quantity,
                    "status": rec.order.status.value,
                    "reject_reason": rec.reject_reason,
                }
            )

        # Shadow: intent only (no capital)
        if shadow is not None and s_tgt:
            s_orders = shadow.target_to_orders(
                s_tgt, marks, signal_time=dt, order_time=exec_dt, nav=1.0, min_notional=1e-9
            )
            for order in s_orders:
                sid = order.security_id
                if sid not in marks:
                    continue
                rec = shadow.submit(
                    order,
                    price=marks[sid],
                    nav=1.0,
                    adv_dollars=advs.get(sid, 1.0),
                    sigma=vols.get(sid, 0.02),
                )
                step_recs.append(rec)
                shadow.shares[sid] = float(s_tgt.get(sid, 0.0))
                all_orders.append(
                    {
                        "asof": exec_dt,
                        "slot": "shadow",
                        "order_id": rec.order.order_id,
                        "security_id": sid,
                        "side": rec.order.side.value,
                        "quantity": rec.order.quantity,
                        "status": rec.order.status.value,
                        "reject_reason": rec.reject_reason,
                    }
                )

        ledger.record_orders(step_recs, exec_dt)

        # Mark to close + borrow. Close marks are applied only after fills so
        # next-open execution never consumes same-bar close information.
        valuation_marks = dict(pretrade_marks)
        valuation_marks.update(close_marks)
        for sid in close_marks:
            mark_ages[sid] = 0
        champ.mark(valuation_marks)
        if shadow is not None:
            shadow.mark(valuation_marks)
        _require_valuation_marks(champ, valuation_marks)
        nav_close = champ.nav(valuation_marks)
        short_notional = sum(
            abs(min(champ.shares.get(s, 0.0), 0.0)) * valuation_marks.get(s, 0.0)
            for s in champ.shares
        )
        borrow = short_notional * (config.costs.borrow_bps_per_year / 1e4) / 252.0
        if not config.costs.frictionless:
            champ.cash -= borrow
            nav_close -= borrow
            if borrow:
                ledger._cash_events.append(
                    {
                        "asof": exec_dt,
                        "slot": "champion",
                        "order_id": "",
                        "security_id": "",
                        "cash_delta": -float(borrow),
                        "notional": 0.0,
                        "fees": float(borrow),
                        "side": "borrow",
                    }
                )
        g, n = champ.exposures(valuation_marks)
        identity = champ.cash_nav_identity(valuation_marks)
        row = {
            "event_time": exec_dt,
            "nav": nav_close,
            "gross": g,
            "net": n,
            "cash": champ.cash,
            "slot": "champion",
            "cash_nav_residual": identity["residual"],
        }
        champ_eq.append(row)
        ledger.record_snapshot(champ.snapshot(asof=exec_dt, prices=valuation_marks))

        if shadow is not None:
            sg = float(sum(abs(v) for v in shadow.shares.values()))
            sn = float(sum(shadow.shares.values()))
            shadow_row = {
                "event_time": exec_dt,
                "nav": float("nan"),  # no capital
                "gross": sg,
                "net": sn,
                "cash": 0.0,
                "slot": "shadow",
            }
            shadow_eq.append(shadow_row)
            ledger.record_shadow_equity(shadow_row)

        step += 1
        steps_this_run += 1
        last_decision = dt
        last_exec = exec_dt
        # Persist broker state every step for crash-safe multi-day resume
        ledger.save_broker_state(
            champ,
            shadow,
            last_decision=last_decision,
            last_exec=last_exec,
            step=step,
            resume_fingerprint=_paper_resume_fingerprint(px, config, exec_dt),
            mark_ages=mark_ages,
        )
        if max_steps is not None and steps_this_run >= max_steps:
            break

    eq = (
        pl.DataFrame(champ_eq, infer_schema_length=None)
        if champ_eq
        else pl.DataFrame({"event_time": [], "nav": []})
    )
    sh_eq = pl.DataFrame(shadow_eq, infer_schema_length=None) if shadow_eq else pl.DataFrame()
    # Book diagnostics (labeled)
    rets = (
        eq["nav"].pct_change().drop_nulls().to_numpy()
        if eq.height >= 2 and "nav" in eq.columns
        else np.array([])
    )
    gross_series = eq["gross"].to_numpy() if "gross" in eq.columns and eq.height else None
    net_series = eq["net"].to_numpy() if "net" in eq.columns and eq.height else None
    # Last champion weight snapshot for stress
    last_w = None
    if champ.shares and champ.last_marks:
        nav_m = max(champ.nav(champ.last_marks), 1e-12)
        last_w = np.array(
            [
                champ.shares.get(s, 0.0) * champ.last_marks.get(s, 0.0) / nav_m
                for s in sorted(set(champ.shares) | set(champ.last_marks))
            ],
            dtype=float,
        )
    diag = book_diagnostics(
        rets,
        gross_exposure=gross_series,
        net_exposure=net_series,
        turnover=None,
        weights=last_w,
        label="PAPER_SIMULATED",
        data_source="SYNTHETIC" if synthetic else str(config.data.source),
    )
    port_conf = _paper_portfolio_conformal(rets)
    mean_l1, max_l1 = _merge_divergence_summary(prior_promo, div_series, total_steps=step)
    champ_nav = float(eq["nav"][-1]) if eq.height and "nav" in eq.columns else None
    shadow_gross = float(sh_eq["gross"][-1]) if sh_eq.height and "gross" in sh_eq.columns else None
    # Primary + extra challenger divergence for rolling / multi-challenger report
    div_map: dict[str, list[float] | np.ndarray] = {}
    if shadow is not None or div_series:
        div_map["shadow"] = list(div_series)
    div_map.update({k: list(v) for k, v in challenger_divs.items()})
    roll_win = max(int(rolling_window), 1)
    challenger_report = rolling_challenger_metrics(div_map, window=roll_win) if div_map else None
    rolling_mean_l1 = None
    if div_series:
        rolling_mean_l1 = float(np.mean(div_series[-min(roll_win, len(div_series)) :]))
    risk_acct = champ.reject_accounting()
    promo = promotion_dry_run(
        run_id=run_id,
        mean_l1=mean_l1,
        max_l1=max_l1,
        n_steps=step,
        champion_nav=champ_nav,
        shadow_gross=shadow_gross,
        max_mean_l1=float(getattr(config.paper, "promote_max_mean_l1", 0.25)),
        min_steps=int(getattr(config.paper, "promote_min_steps", 5)),
        data_source="SYNTHETIC" if synthetic else str(config.data.source),
        rolling_mean_l1=rolling_mean_l1,
        rolling_window=roll_win,
        challenger_metrics=challenger_report,
        primary_challenger="shadow" if shadow is not None else None,
        risk_accounting=risk_acct,
        kill_tripped_mid_run=bool(kill_tripped_mid_run),
        allow_missing_divergence=True,
    )
    promo_path = ledger.write_promotion_dry_run(promo)

    nav_arr = (
        eq["nav"].to_numpy().astype(float)
        if eq.height and "nav" in eq.columns
        else np.array([], dtype=float)
    )
    eq_analytics = equity_curve_analytics(nav_arr)

    metrics: dict[str, Any] = {
        **diag,
        "equity_analytics": eq_analytics,
        "risk_gate_rejects": champ.risk_gate_reject_count,
        "kill_switch_halts": champ.halt_count,
        "reject_total": champ.reject_count,
        "reject_accounting": risk_acct,
        "n_steps": step,
        "n_steps_this_run": steps_this_run,
        "n_fills": len(champ.fills),
        "n_order_records_this_run": len(all_orders),
        "mean_weight_l1_divergence": mean_l1,
        "max_weight_l1_divergence": max_l1,
        "rolling_mean_l1_divergence": rolling_mean_l1,
        "rolling_window": roll_win,
        "challenger_metrics": challenger_report,
        "has_shadow": shadow is not None,
        "n_extra_challengers": len(extra_challengers),
        "kill_tripped_mid_run": bool(kill_tripped_mid_run),
        "research_only": True,
        "live_pnl_claim": False,
        "resumed": bool(resume),
        "portfolio_conformal": port_conf,
        "promotion_dry_run": promo,
    }
    metrics["analytics_export"] = export_analytics_dict(diag, extra={"run_id": run_id})
    metrics["analytics_export"]["analytics_export_sha256"] = analytics_export_digest(
        metrics["analytics_export"]
    )
    metrics["stress_report"] = diag.get("stress_report", metrics.get("stress_report", {}))
    metrics["drawdown_duration"] = diag.get(
        "drawdown_duration", metrics.get("drawdown_duration", {})
    )
    ledger.set_meta(metrics=metrics)
    # Final broker state
    ledger.save_broker_state(
        champ,
        shadow,
        last_decision=last_decision,
        last_exec=last_exec,
        step=step,
        resume_fingerprint=(
            _paper_resume_fingerprint(px, config, last_exec) if last_exec is not None else None
        ),
        mark_ages=mark_ages,
    )
    paths = ledger.flush()
    # Schema-aligned analytics JSON (same keys as backtest export where possible)
    import json as _json_ae

    ae_path = ledger.root / "analytics_export.json"
    ae_path.write_text(_json_ae.dumps(metrics["analytics_export"], indent=2, default=str))
    paths["analytics_export"] = ae_path
    ae_report = validate_analytics_export(metrics["analytics_export"])
    metrics["analytics_export_validation"] = ae_report
    metrics["analytics_export_ok"] = bool(ae_report.get("ok"))
    # Persist validation onto meta after export (fail-closed flags in metrics).
    ledger.set_meta(metrics=metrics)
    ledger.flush()
    paths["broker_state"] = ledger.root / "broker_state.json"
    paths["promotion_dry_run"] = promo_path
    orders_df = (
        pl.DataFrame(ledger._orders, infer_schema_length=None) if ledger._orders else pl.DataFrame()
    )

    divergence = {
        "mean_l1": mean_l1,
        "max_l1": max_l1,
    }
    note = "SYNTHETIC" if synthetic else str(config.data.source)
    return PaperLoopResult(
        run_id=run_id,
        champion_equity=eq,
        shadow_equity=sh_eq,
        orders=orders_df,
        metrics=metrics,
        paths={k: str(v) for k, v in paths.items()},
        source_note=note,
        divergence=divergence,
        promotion=promo,
    )
