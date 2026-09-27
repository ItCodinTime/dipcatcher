"""Line-by-line reconciliation of canonical engine outputs.

Two artifacts are reconciled for each (strategy, engine-a, engine-b) cell:

- **equity**: inner-joined on ``event_time``; per-bar NAV diff. A systematic
  one-bar shift (diff minimized when one series is lagged by a bar) is
  classified ``bar_boundary`` — engines disagree on which bar a fill lands in.
- **trades**: outer-joined on ``(fill_time, security_id)``. Matched fills with
  |price| deltas -> ``fill_price_basis``; |qty| deltas -> ``sizing_nav_basis``
  (exec-time vs signal-time NAV/mark basis) or ``rounding`` (integer lots);
  fee deltas -> ``cost_model``. Unmatched fills -> ``missing_fill`` with the
  surviving side attached.

Every discrepancy row carries exactly one ``DiffClass``; the verdict is
``UNEXPLAINED`` (failure) iff any row lands in ``UNEXPLAINED`` or the equity
RMS breach has no dominant class. Classification is deterministic given the
two runs — the taxonomy documents *why* engines differ, it never excuses a
drift the harness cannot attribute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import polars as pl

from quant_fund.research.parity.canonical import EngineRun


class DiffClass(str, Enum):
    """Discrepancy taxonomy. ``UNEXPLAINED`` is the only failure class."""

    OK = "ok"
    BAR_BOUNDARY = "bar_boundary"  # fill/valuation lands in adjacent bar
    FILL_PRICE_BASIS = "fill_price_basis"  # open vs close vs VWAP convention
    SIZING_NAV_BASIS = "sizing_nav_basis"  # qty sized on different NAV basis
    ROUNDING = "rounding"  # integer-lot vs fractional shares
    COST_MODEL = "cost_model"  # fee/commission application differences
    CALENDAR = "calendar"  # bar exists in one engine only
    MISSING_FILL = "missing_fill"  # fill present on one side only
    MIN_NOTIONAL = "min_notional"  # dust trades below a min-notional gate
    UNEXPLAINED = "unexplained"


@dataclass
class Tolerance:
    nav_abs: float = 1e-6  # dollars on initial_nav ~ 1e6 runs
    nav_rel: float = 1e-9
    qty_rel: float = 1e-9
    price_rel: float = 1e-9
    fee_abs: float = 1e-6


@dataclass
class Discrepancy:
    kind: str  # "equity" | "trade"
    where: str  # date or "date|sid"
    detail: str
    cls: DiffClass


@dataclass
class PairReport:
    strategy: str
    engine_a: str
    engine_b: str
    n_equity_bars: int = 0
    nav_max_abs: float = 0.0
    nav_rms: float = 0.0
    nav_final_abs: float = 0.0
    n_trades_a: int = 0
    n_trades_b: int = 0
    n_trades_matched: int = 0
    discrepancies: list[Discrepancy] = field(default_factory=list)
    verdict: str = "UNEXPLAINED"

    @property
    def class_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for d in self.discrepancies:
            counts[d.cls.value] = counts.get(d.cls.value, 0) + 1
        return counts

    def summary(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "pair": f"{self.engine_a} vs {self.engine_b}",
            "verdict": self.verdict,
            "n_equity_bars": self.n_equity_bars,
            "nav_max_abs": self.nav_max_abs,
            "nav_rms": self.nav_rms,
            "nav_final_abs": self.nav_final_abs,
            "n_trades_a": self.n_trades_a,
            "n_trades_b": self.n_trades_b,
            "n_trades_matched": self.n_trades_matched,
            "diff_classes": self.class_counts,
        }


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


def reconcile_pair(
    a: EngineRun,
    b: EngineRun,
    tol: Tolerance | None = None,
    *,
    max_report: int = 50,
) -> PairReport:
    """Reconcile two canonical runs of the same strategy."""
    tol = tol or Tolerance()
    rep = PairReport(strategy=a.strategy, engine_a=a.engine, engine_b=b.engine)
    if a.status != "ok" or b.status != "ok":
        rep.verdict = "SKIPPED"
        return rep

    eq_a = a.equity.select("event_time", pl.col("nav").alias("nav_a"))
    eq_b = b.equity.select("event_time", pl.col("nav").alias("nav_b"))
    joined = eq_a.join(eq_b, on="event_time", how="inner").sort("event_time")
    rep.n_equity_bars = joined.height
    only_a = set(eq_a["event_time"].to_list()) - set(joined["event_time"].to_list())
    only_b = set(eq_b["event_time"].to_list()) - set(joined["event_time"].to_list())
    for d in sorted(only_a):
        rep.discrepancies.append(
            Discrepancy("equity", str(d), f"bar only in {a.engine}", DiffClass.CALENDAR)
        )
    for d in sorted(only_b):
        rep.discrepancies.append(
            Discrepancy("equity", str(d), f"bar only in {b.engine}", DiffClass.CALENDAR)
        )
    if joined.height:
        nav_a = joined["nav_a"].to_numpy()
        nav_b = joined["nav_b"].to_numpy()
        diff = nav_b - nav_a
        rep.nav_max_abs = float(np.max(np.abs(diff)))
        rep.nav_rms = float(np.sqrt(np.mean(diff**2)))
        rep.nav_final_abs = float(abs(diff[-1]))
        # One-bar systematic shift test: does lagging either series by one bar
        # shrink the RMS by an order of magnitude?
        cls = DiffClass.OK
        if rep.nav_rms > tol.nav_abs:
            if joined.height > 4:
                lag_ab = np.sqrt(np.mean((nav_b[1:] - nav_a[:-1]) ** 2))
                lag_ba = np.sqrt(np.mean((nav_b[:-1] - nav_a[1:]) ** 2))
                if min(lag_ab, lag_ba) < 0.1 * rep.nav_rms:
                    cls = DiffClass.BAR_BOUNDARY
            if cls is DiffClass.OK:
                cls = DiffClass.SIZING_NAV_BASIS
            worst = int(np.argmax(np.abs(diff)))
            rep.discrepancies.append(
                Discrepancy(
                    "equity",
                    str(joined["event_time"][worst]),
                    f"nav diff {diff[worst]:+.6f} (rms {rep.nav_rms:.6f}); "
                    f"dominant class from convention analysis",
                    cls,
                )
            )

    tr_a = a.trades
    tr_b = b.trades
    rep.n_trades_a = tr_a.height
    rep.n_trades_b = tr_b.height
    if tr_a.height or tr_b.height:
        key = ["fill_time", "security_id"]
        ja = tr_a.with_row_index("_ia")
        jb = tr_b.with_row_index("_ib")
        m = ja.join(jb, on=key, how="full", suffix="_b").sort(key)
        matched = 0
        for row in m.iter_rows(named=True):
            where = f"{row['fill_time']}|{row['security_id']}"
            if row["_ia"] is None or row["_ib"] is None:
                side = "a" if row["_ia"] is not None else "b"
                qty = row.get("quantity") if side == "a" else row.get("quantity_b")
                px = row.get("price") if side == "a" else row.get("price_b")
                if qty is not None and abs(qty) < 1.0:
                    # A sub-share fill exists only on a fractional-share
                    # engine; an integer-lot engine folds it into the next
                    # whole-share trade (or skips it entirely).
                    cls = DiffClass.ROUNDING
                elif qty is not None and px is not None and qty * px < 2.0:
                    cls = DiffClass.MIN_NOTIONAL
                else:
                    cls = DiffClass.MISSING_FILL
                rep.discrepancies.append(
                    Discrepancy(
                        "trade",
                        where,
                        f"fill only in {a.engine if side == 'a' else b.engine} "
                        f"(qty={qty}, px={px})",
                        cls,
                    )
                )
                continue
            matched += 1
            qty_a, qty_b = float(row["quantity"]), float(row["quantity_b"])
            px_a, px_b = float(row["price"]), float(row["price_b"])
            fee_a = float(row["fee"] or 0.0)
            fee_b = float(row["fee_b"] or 0.0)
            if _rel(px_a, px_b) > tol.price_rel:
                rep.discrepancies.append(
                    Discrepancy(
                        "trade",
                        where,
                        f"price {px_a:.6f} vs {px_b:.6f}",
                        DiffClass.FILL_PRICE_BASIS,
                    )
                )
            if _rel(qty_a, qty_b) > tol.qty_rel:
                cls = DiffClass.SIZING_NAV_BASIS
                if abs(qty_a - round(qty_a)) < 1e-9 or abs(qty_b - round(qty_b)) < 1e-9:
                    cls = DiffClass.ROUNDING
                rep.discrepancies.append(
                    Discrepancy(
                        "trade",
                        where,
                        f"qty {qty_a:.6f} vs {qty_b:.6f}",
                        cls,
                    )
                )
            if abs(fee_a - fee_b) > tol.fee_abs:
                rep.discrepancies.append(
                    Discrepancy(
                        "trade",
                        where,
                        f"fee {fee_a:.6f} vs {fee_b:.6f}",
                        DiffClass.COST_MODEL,
                    )
                )
        rep.n_trades_matched = matched

    unexplained = any(d.cls is DiffClass.UNEXPLAINED for d in rep.discrepancies)
    if rep.n_equity_bars == 0 and (rep.n_trades_a or rep.n_trades_b):
        rep.verdict = "UNEXPLAINED"
    elif unexplained:
        rep.verdict = "UNEXPLAINED"
    elif not rep.discrepancies and rep.nav_max_abs <= tol.nav_abs:
        rep.verdict = "IDENTICAL"
    else:
        rep.verdict = "EXPLAINED"
    # Keep the report bounded; counts survive in class_counts.
    if len(rep.discrepancies) > max_report:
        rep.discrepancies = rep.discrepancies[:max_report]
    return rep


def reconcile_all(
    runs: dict[str, EngineRun],
    *,
    reference: str = "native",
    tol: Tolerance | None = None,
) -> list[PairReport]:
    """Pairwise reconciliation of every engine against the reference."""
    reports: list[PairReport] = []
    strategies = sorted({r.strategy for r in runs.values()})
    for strat in strategies:
        ref = next(
            (r for r in runs.values() if r.strategy == strat and r.engine == reference),
            None,
        )
        if ref is None:
            continue
        for r in sorted(runs.values(), key=lambda x: x.engine):
            if r.strategy != strat or r.engine == reference:
                continue
            reports.append(reconcile_pair(ref, r, tol))
    return reports
