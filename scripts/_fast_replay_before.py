"""Vectorized fast path for ``run_backtest``.

Same economics as the reference event loop — decision at close ``t`` fills at
open ``t+1`` (or same-bar close when ``use_next_open`` is false), per-order
risk gate, kill switch, stale-valuation fail-closed, sequential cash and
contributor accounting in sorted-``security_id`` order — but the per-date
panel scan runs on pre-indexed numpy matrices instead of per-row polars
iteration.

Scope (fail-closed): this path exists for the matched-workload class used by
the incumbent benchmarks — target-weight panels, market fills, the full
``CostConfig`` surface, and the full ``RiskGateConfig`` surface. It refuses
configs it does not replicate (``allow_close_auction=True``).

Float-order contract: NAV and exposure sums follow the reference engine's
summation orders — shares-dict insertion order for book NAV, sorted
contributors for projected exposures — so outputs are expected to be
bit-identical to ``run_backtest``; the conformance suite asserts that.
"""

from __future__ import annotations

import inspect
import math
from typing import Any

import numpy as np
import polars as pl

from quant_fund.backtest.engine import (
    BacktestResult,
    StaleValuationError,
    _validate_target_weight_panel,
    run_backtest,
)
from quant_fund.config.models import AppConfig, FillConvention
from quant_fund.monitoring.kill_switch import KillSwitch
from quant_fund.schemas.errors import KillSwitchActive

try:
    from quant_fund.pipeline.forecast import (  # type: ignore[attr-defined]
        market_risk_overlay_asof as _mro_asof,
    )
except Exception:  # pragma: no cover - older lineage lacks the module
    _mro_asof = None  # type: ignore[assignment]


def _order_costs(
    d: float, px: float, adv_d: float, vol_d: float, cfg: Any
) -> tuple[float, float, float, float]:
    """Identical expression order/types as ``total_cost`` (commission, spread,
    impact, total). ``d``/``px`` keep their operand types — a capped delta is
    np.float64 and that propagates exactly as it does in the reference."""
    # total_cost validates inputs before the frictionless early
    # return — a non-finite delta must fail closed, not trade.
    if not math.isfinite(d) or not math.isfinite(px) or px <= 0:
        raise ValueError("quantity must be finite and price must be finite and positive")
    if cfg.frictionless:
        return 0.0, 0.0, 0.0, 0.0
    nt = abs(d) * px
    comm = abs(nt) * cfg.commission_bps / 1e4
    spr = abs(nt) * cfg.half_spread_bps / 1e4
    part = nt / max(adv_d, 1e-12)
    imp = float(nt * cfg.impact_y * vol_d * math.sqrt(part))
    bpt = abs(nt) * cfg.bps_per_turnover / 1e4
    return comm, spr, imp, float(comm + spr + imp + bpt)


_BAR_COLS = (
    "security_id",
    "event_time",
    "open",
    "close",
    "close_total_return",
    "volume",
    "adv",
    "vol_20",
)


def _bars_to_matrices(
    bars: pl.DataFrame,
) -> tuple[
    list[Any],
    np.ndarray,
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    bool,
]:
    """Build per-(date, asset) matrices mirroring ``day_rows_map`` semantics.

    Returns (dates, sids, open_px, close_px, ctr, adv, vol, synthetic).
    A missing row and an invalid price are indistinguishable downstream of
    ``_valid_price``, so absent cells are simply NaN.
    """
    px = bars.select(
        "security_id",
        "event_time",
        "open",
        "close",
        "close_total_return",
        "volume",
        pl.col("adv")
        if "adv" in bars.columns
        else (pl.col("close") * pl.col("volume")).alias("adv"),
        pl.col("vol_20") if "vol_20" in bars.columns else pl.lit(0.02).alias("vol_20"),
        "source",
    )
    sids = sorted(str(s) for s in px["security_id"].unique().to_list())
    sid_idx = {s: i for i, s in enumerate(sids)}
    et_raw = px["event_time"].cast(pl.Int64).to_numpy()
    dates_ns = np.unique(et_raw)
    t_idx = np.searchsorted(dates_ns, et_raw)
    a_idx = np.array([sid_idx[str(s)] for s in px["security_id"].to_list()], dtype=np.int64)

    n_dates = int(dates_ns.size)
    n_assets = len(sids)

    def _to_f64(name: str) -> np.ndarray:
        # strict=False mirrors _valid_price: unparseable/non-finite inputs
        # become null -> NaN downstream instead of raising at ingest.
        return px[name].cast(pl.Float64, strict=False).to_numpy()

    def _mark_col(name: str) -> np.ndarray:
        # Marks use "last *valid* write wins": an invalid duplicate row does
        # not clobber an earlier valid one (dict assignment is skipped when
        # _valid_price returns None).
        arr = np.full((n_dates, n_assets), np.nan)
        out = _to_f64(name)
        ok = np.isfinite(out) & (out > 0)
        arr[t_idx[ok], a_idx[ok]] = out[ok]
        return arr

    def _raw_col(name: str) -> np.ndarray:
        # adv/vol overwrite unconditionally per row — the last duplicate row
        # wins even when invalid (the default is applied downstream).
        arr = np.full((n_dates, n_assets), np.nan)
        arr[t_idx, a_idx] = _to_f64(name)
        return arr

    open_px = _mark_col("open")
    close_px = _mark_col("close")
    ctr = _mark_col("close_total_return")
    adv = _raw_col("adv")
    vol = _raw_col("vol_20")
    synthetic = (
        "synthetic" in set(px["source"].drop_nulls().to_list()) if "source" in px.columns else False
    )
    dates = sorted(set(px["event_time"].to_list()))
    assert len(dates) == n_dates
    return dates, dates_ns, sids, open_px, close_px, ctr, adv, vol, synthetic


def run_backtest_fast(
    bars: pl.DataFrame,
    weights: pl.DataFrame,
    config: AppConfig,
    *,
    initial_nav: float = 1_000_000.0,
    risk_overlay: Any = None,
) -> BacktestResult:
    """Fast replay of ``run_backtest`` on the supported config class."""
    if config.execution.allow_close_auction:
        raise ValueError("fast replay does not support allow_close_auction")
    if risk_overlay is not None:
        raise ValueError("fast replay does not support risk_overlay")
    _validate_target_weight_panel(weights)

    # Lineage detection: the newer engine signature carries ``risk_overlay``
    # and treats weight panels as a sparse rebalance grid — the last target
    # holds until the next decision date, and names without a fresh mark are
    # flattened. The older lineage zero-fills absent dates. Detect once and
    # replicate whichever semantics the deployed ``run_backtest`` implements.
    ref_carries = "risk_overlay" in inspect.signature(run_backtest).parameters

    dates, dates_ns, sids, open_px, close_px, ctr, adv, vol, synthetic = _bars_to_matrices(bars)
    n_dates = len(dates)
    n_assets = len(sids)
    sid_idx = {s: i for i, s in enumerate(sids)}

    # Target-weight matrix on bar dates only — the reference loop only ever
    # looks up weights on dates that have bars. Pivot is polars-native; the
    # panel was dedup-validated upstream so `first` is lossless, and absent
    # cells arrive as NaN -> 0.0 (the reference's `.get(sid, 0.0)` semantics).
    w_mat = np.zeros((n_dates, n_assets))
    piv = weights.pivot(
        on="security_id",
        index="event_time",
        values="target_weight",
        aggregate_function="first",
    )
    if piv.height:
        piv_ns = piv["event_time"].cast(pl.Int64).to_numpy()
        piv_ti = np.searchsorted(dates_ns, piv_ns)
        in_range = (piv_ti < n_dates) & (dates_ns[np.minimum(piv_ti, n_dates - 1)] == piv_ns)
        for col in piv.columns:
            if col == "event_time":
                continue
            ai = sid_idx.get(col)
            if ai is None:
                # Weight on a symbol with no bars — validated upstream but can
                # never trade; identical to the reference loop's skip.
                continue
            v = piv[col].cast(pl.Float64, strict=False).to_numpy()
            w_mat[piv_ti[in_range], ai] = np.where(np.isfinite(v), v, 0.0)[in_range]

    market_vols: list | None = None
    if ref_carries:
        # ``if dt in weights_by_date: last_target_w = ...`` — presence is at
        # the whole-row level: a date with ANY row replaces the carried target
        # (absent sids target 0); a date with none inherits the prior row.
        w_present = np.zeros(n_dates, dtype=bool)
        if piv.height:
            w_present[piv_ti[in_range]] = True
        for i in range(n_dates):
            if not w_present[i]:
                w_mat[i] = w_mat[i - 1] if i else 0.0
        # Market overlay: artifact existence is config/fs-level — probe the
        # last date once; (None, None) means it can never engage.
        if _mro_asof is not None:
            probe_v, _ = _mro_asof(config, bars, dates[-1])
            if probe_v is not None:
                market_vols = [_mro_asof(config, bars, d)[0] for d in dates]

    use_next_open = (
        config.execution.fill is FillConvention.NEXT_OPEN
        and not config.execution.allow_close_auction
    )

    # Per-(date, asset) rows converted to Python lists once — indexing a list
    # is faster than a per-day ``.tolist()`` on a numpy row. Values are
    # identical (``tolist`` preserves float64 bit patterns exactly).
    src_mat = open_px if use_next_open else close_px
    exec_valid_m = np.isfinite(src_mat) & (src_mat > 0)
    src_l = src_mat.tolist()
    adv_l = adv.tolist()
    vol_l = vol.tolist()
    dec_px_l = close_px.tolist()
    dec_ok_l = (np.isfinite(close_px) & (close_px > 0)).tolist()
    # Whole-matrix mark validity — the per-day mark update only carries
    # last_mark/mark_age state; today's valid marks are row lookups.
    ctr_ok_m = np.isfinite(ctr) & (ctr > 0)
    close_ok_m = np.isfinite(close_px) & (close_px > 0)
    new_mark_m = np.where(ctr_ok_m, ctr, np.where(close_ok_m, close_px, 0.0))

    costs_cfg = config.costs
    gate = config.risk_gate
    kill = KillSwitch(config.kill_switch)

    cash = float(initial_nav)
    # Python list (not ndarray): element types must propagate exactly like the
    # reference's shares dict — float until a participation cap stores
    # np.float64.
    shares: list[float] = [0.0] * n_assets
    pos_seq: list[int] = []  # shares-dict insertion order for NAV sums
    in_book = np.zeros(n_assets, dtype=bool)
    last_mark = np.zeros(n_assets)  # carried close marks (0 = never marked)
    ever_marked = np.zeros(n_assets, dtype=bool)
    mark_age = np.zeros(n_assets, dtype=np.int64)

    navs: list[dict] = []
    fill_rows: list[dict] = []
    cost_sum = {"commission": 0.0, "spread": 0.0, "impact": 0.0}
    reject_count = 0
    cash_reject_count = 0
    halt_count = 0
    order_seq = 0

    # KillSwitch state is fixed at construction — evaluate once; each blocked
    # order is still counted identically in the loop below.
    try:
        kill.assert_new_orders_allowed()
        kill_blocks = False
    except KillSwitchActive:
        kill_blocks = True

    loop_range = range(n_dates - 1) if use_next_open else range(n_dates)
    for i in loop_range:
        exec_t = i + 1 if use_next_open else i
        exec_src = src_mat[exec_t]
        exec_valid = exec_valid_m[exec_t]

        # --- mark update (close_total_return preferred, close fallback) ----
        marked_today = ctr_ok_m[exec_t] | close_ok_m[exec_t]
        mark_age = np.where(marked_today, 0, mark_age + 1)
        last_mark = np.where(marked_today, new_mark_m[exec_t], last_mark)
        ever_marked |= marked_today

        # --- stale-valuation fail-closed on held positions -----------------
        # Detail order follows book.shares insertion order (pos_seq).
        stale = [
            a
            for a in pos_seq
            if abs(shares[a]) > 1e-12
            and (not ever_marked[a] or mark_age[a] > gate.stale_price_bars)
        ]
        if stale:
            details = ", ".join(
                f"{sids[a]}={int(mark_age[a]) if ever_marked[a] else 'unknown'}" for a in stale
            )
            raise StaleValuationError(
                "held position valuation is stale beyond the configured limit: " + details
            )

        # --- decision marks (signal bar close at decision date) ------------
        dec_l_i = dec_px_l[i]
        dec_ok_i = dec_ok_l[i]

        # --- nav at execution marks (exec price overrides carried close) ---
        # book.nav uses builtin sum() — CPython 3.12+ applies Neumaier
        # compensated summation, which differs from a naive += loop at the
        # last ulp. Keep builtin sum over the same insertion-ordered terms
        # for bit-identical floats. NB the compensated path only applies to
        # exact Python floats — np.float64 terms must be coerced per-product.
        nav_price = np.where(exec_valid, exec_src, np.where(ever_marked, last_mark, 0.0))
        # Prices coerce to Python float; shares keep their dict type so a
        # capped fill's np.float64 propagates into products exactly as the
        # reference's does (np.float64 term → sum() degrades to naive).
        npv = nav_price.tolist()
        pos_val = sum(shares[a] * npv[a] for a in pos_seq)
        nav = cash + pos_val
        if nav <= 0:
            break

        esrc = src_l[exec_t]
        adv_row = adv_l[exec_t]
        vol_row = vol_l[exec_t]
        dec_l = dec_l_i
        dec_ok_d = dec_ok_i
        # ``ids = exec | shares | target``; unpriced sids are skipped inside.
        # Per-day contributor products shares*price; position a's term is
        # patched per order. A zero product is dropped — bitwise-identical to
        # the reference's (proj!=0 and price!=0) contributor filter since a
        # zero term never changes a compensated sum.
        terms_base = [shares[b] * npv[b] for b in range(n_assets)]
        nav_safe = max(nav, 1e-12)
        traded_turn = 0.0
        # Date-level market overlay vol (newer lineage): replaces name-level
        # vol for the max_predicted_vol gate; absent -> name vol governs.
        mv = market_vols[i] if market_vols is not None else None
        mv_bad = mv is not None and (not math.isfinite(mv) or mv < 0.0)
        # Carrying lineage flattens names without a fresh mark — a missing
        # mark is an exit, not a ghost hold.
        tw_arr = np.where(marked_today, w_mat[i], 0.0) if ref_carries else w_mat[i]
        tw_l = tw_arr.tolist()
        # Candidate mask: assets with an executable mark whose dust check
        # passes. Vectorized IEEE-754 float64 math — the scalar loop below
        # recomputes the identical delta in Python for the order path;
        # flatnonzero preserves ascending sid order (reference iterates
        # sorted ids). ``shares`` keeps mixed types; sh_arr is values-only.
        px_ok = np.where(exec_valid, exec_src, 1.0)
        desired_arr = tw_arr * nav / px_ok
        sh_arr = np.asarray(shares)
        cand = exec_valid & (np.abs(desired_arr - sh_arr) * px_ok >= 1.0)
        for a in np.flatnonzero(cand).tolist():
            price = esrc[a]
            tw = tw_l[a]
            desired = tw * nav / price
            current = shares[a]
            delta = desired - current
            adv_a = adv_row[a]
            adv_eff = adv_a if math.isfinite(adv_a) and adv_a > 0 else 1.0
            vol_a = vol_row[a]
            vol_eff = vol_a if math.isfinite(vol_a) and vol_a > 0 else 0.02

            comm, spr, imp, total_trade_cost = _order_costs(
                delta, price, adv_eff, vol_eff, costs_cfg
            )
            max_qty = costs_cfg.participation_limit * (adv_eff / price)
            if abs(delta) > max_qty > 0:
                # np.sign() yields np.float64; the reference does NOT coerce it
                # back — the np.float64 delta contaminates shares/cash and
                # degrades later sum() calls to the naive path. Replicate.
                delta = np.sign(delta) * max_qty
                comm, spr, imp, total_trade_cost = _order_costs(
                    delta, price, adv_eff, vol_eff, costs_cfg
                )

            if kill_blocks:
                halt_count += 1
                continue

            # --- risk gate (same checks, same order outcomes) --------------
            current_w = current * npv[a] / nav_safe
            # _projected_exposures sums |v| and v over sorted contributors
            # with builtin sum(); price operands are float, share operands
            # keep dict types — identical propagation to the reference.
            terms = terms_base[:]
            terms[a] = (current + delta) * npv[a]
            terms = [t for t in terms if t != 0.0]
            gross_after = sum(map(abs, terms)) / nav_safe
            net_after = sum(terms) / nav_safe
            participation = abs(delta) * price / max(adv_eff, 1e-12)
            order_seq += 1
            # check_order's signed_qty = +/-|delta| by side — i.e. delta itself.
            # gate_vol: a present market overlay vol replaces name vol for the
            # max_predicted_vol comparison (resolve_gate_predicted_vol).
            gate_vol = mv if mv is not None else vol_eff
            rejected = (
                mv_bad
                or any(
                    not math.isfinite(v)
                    for v in (nav, price, current_w, gross_after, net_after, participation, vol_eff)
                )
                or nav <= 0.0
                or price <= 0.0
                or gross_after < 0.0
                or participation < 0.0
                or vol_eff < 0.0
                or not math.isfinite(delta)
                or delta == 0.0
                or abs(delta) * price > gate.max_order_notional
                or abs(current_w + (delta * price) / nav_safe) > gate.max_name
                or gross_after > gate.max_gross
                or abs(net_after) > gate.max_net
                or participation > gate.max_participation
                or gate_vol > gate.max_predicted_vol
            )
            if rejected:
                reject_count += 1
                continue

            notional = delta * price
            if delta > 0 and cash < notional + total_trade_cost:
                cash_reject_count += 1
                continue
            cash -= notional + total_trade_cost
            if not in_book[a]:
                in_book[a] = True
                pos_seq.append(a)
            shares[a] = current + delta
            terms_base[a] = shares[a] * npv[a]
            traded_turn += abs(notional) / nav_safe
            cost_sum["commission"] += float(comm)
            cost_sum["spread"] += float(spr)
            cost_sum["impact"] += float(imp)
            fill_rows.append(
                {
                    "fill_time": dates[exec_t],
                    "signal_time": dates[i],
                    "security_id": sids[a],
                    "quantity": delta,
                    "price": price,
                    "fee": comm,
                    "spread_cost": spr,
                    "impact_cost": imp,
                    "decision_price": dec_l[a] if dec_ok_d[a] else None,
                }
            )

        # --- mark to close ---------------------------------------------------
        # Builtin sum() over shares-order terms; marks coerce to float, shares
        # keep dict types — same propagation as the reference.
        lm = last_mark.tolist()
        pos_close = sum(shares[a] * lm[a] for a in pos_seq)
        nav_close = cash + pos_close
        short_notional = sum(abs(min(shares[a], 0.0)) * lm[a] for a in pos_seq)
        gross_v = sum(abs(shares[a] * lm[a]) for a in pos_seq)
        # net exposure is the identical summation as pos_close — reuse.
        net_v = pos_close
        borrow = short_notional * (costs_cfg.borrow_bps_per_year / 1e4) / 252.0
        if not costs_cfg.frictionless:
            cash -= borrow
            nav_close -= borrow
        navs.append(
            {
                "event_time": dates[exec_t],
                "nav": nav_close,
                "gross": gross_v / max(nav_close, 1e-12),
                "net": net_v / max(nav_close, 1e-12),
                "turnover": traded_turn,
            }
        )

    # ---- identical metrics tail as the reference engine --------------------
    from quant_fund.backtest.engine import _build_result

    return _build_result(
        navs=navs,
        fill_rows=fill_rows,
        cost_sum=cost_sum,
        reject_count=reject_count,
        cash_reject_count=cash_reject_count,
        halt_count=halt_count,
        synthetic=synthetic,
        config=config,
        initial_nav=initial_nav,
    )


__all__ = ["run_backtest_fast"]
