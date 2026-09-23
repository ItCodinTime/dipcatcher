"""Reconciliation utilities: expected-vs-actual parity checks.

Compares a broker state dict (``SimulatedBroker.to_dict``), an equity path,
or a fills frame against the counterpart produced by another path —
e.g. a resumed paper run vs its persisted state, or a paper ledger vs the
research backtest over the same signals. Reports deltas and a mismatch
list; never raises on content mismatches (the report *is* the verdict),
only on malformed inputs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def reconcile_broker_states(
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    tol: float = 1e-9,
) -> dict[str, Any]:
    """Diff two ``SimulatedBroker.to_dict()`` states.

    Checks cash, every held share (union of both books), last marks,
    counters (reject/halt/risk_gate/fill/order), kill state, and fill
    convention. ``tol`` is an absolute dollar/unit tolerance.
    """
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        raise ValueError("broker states must be dicts")
    if tol < 0 or not np.isfinite(tol):
        raise ValueError("tol must be finite and non-negative")
    mismatches: list[str] = []
    deltas: dict[str, float] = {}

    def _cmp_scalar(key: str) -> None:
        a, b = _num(expected.get(key)), _num(actual.get(key))
        if a is None or b is None:
            if expected.get(key) != actual.get(key):
                mismatches.append(f"{key}: {expected.get(key)!r} != {actual.get(key)!r}")
            return
        deltas[f"{key}_delta"] = b - a
        if abs(b - a) > tol:
            mismatches.append(f"{key}: expected {a} got {b}")

    _cmp_scalar("cash")
    for counter in (
        "reject_count",
        "halt_count",
        "risk_gate_reject_count",
        "n_fills",
        "n_orders",
    ):
        _cmp_scalar(counter)

    exp_shares = {str(k): _num(v) for k, v in dict(expected.get("shares") or {}).items()}
    act_shares = {str(k): _num(v) for k, v in dict(actual.get("shares") or {}).items()}
    share_deltas: dict[str, float] = {}
    for sid in sorted(set(exp_shares) | set(act_shares)):
        a = exp_shares.get(sid) or 0.0
        b = act_shares.get(sid) or 0.0
        if a is None or b is None:
            mismatches.append(f"shares[{sid}]: non-numeric")
            continue
        if abs(b - a) > tol:
            share_deltas[sid] = b - a
            mismatches.append(f"shares[{sid}]: expected {a} got {b}")
    if share_deltas:
        deltas["share_deltas"] = share_deltas  # type: ignore[assignment]

    exp_marks = {str(k): _num(v) for k, v in dict(expected.get("last_marks") or {}).items()}
    act_marks = {str(k): _num(v) for k, v in dict(actual.get("last_marks") or {}).items()}
    missing_marks = sorted(set(exp_marks) ^ set(act_marks))
    for sid in missing_marks:
        mismatches.append(f"last_marks[{sid}]: present in one book only")

    for key in ("kill_state", "fill_convention", "slot"):
        if expected.get(key) != actual.get(key):
            mismatches.append(f"{key}: {expected.get(key)!r} != {actual.get(key)!r}")

    return {
        "match": not mismatches,
        "n_mismatches": len(mismatches),
        "mismatches": mismatches[:50],
        "deltas": deltas,
        "tol": float(tol),
        "live_pnl_claim": False,
        "research_only": True,
    }


def reconcile_equity(
    expected: pl.DataFrame,
    actual: pl.DataFrame,
    *,
    time_col: str = "event_time",
    tol: float = 1e-9,
) -> dict[str, Any]:
    """Compare two NAV paths keyed on a shared timestamp column.

    Both frames need ``time_col`` and ``nav``. Matched rows report
    nav-delta stats; unmatched timestamps are counted on each side.
    """
    for name, frame in (("expected", expected), ("actual", actual)):
        missing = {time_col, "nav"} - set(frame.columns)
        if missing:
            raise ValueError(f"{name} equity missing columns: {sorted(missing)}")
    joined = expected.select(time_col, pl.col("nav").alias("nav_exp")).join(
        actual.select(time_col, pl.col("nav").alias("nav_act")),
        on=time_col,
        how="full",
    )
    both = joined.drop_nulls()
    only_exp = joined.filter(pl.col("nav_act").is_null()).height
    only_act = joined.filter(pl.col("nav_exp").is_null()).height
    if both.height == 0:
        return {
            "match": False,
            "n_matched": 0,
            "unmatched_expected": int(only_exp),
            "unmatched_actual": int(only_act),
            "status": "no_shared_timestamps",
            "live_pnl_claim": False,
            "research_only": True,
        }
    delta = (both["nav_act"] - both["nav_exp"]).to_numpy().astype(float)
    max_abs = float(np.max(np.abs(delta)))
    return {
        "match": bool(max_abs <= tol and only_exp == 0 and only_act == 0),
        "n_matched": int(both.height),
        "unmatched_expected": int(only_exp),
        "unmatched_actual": int(only_act),
        "max_abs_nav_delta": max_abs,
        "mean_nav_delta": float(np.mean(delta)),
        "last_nav_delta": float(delta[-1]),
        "tol": float(tol),
        "live_pnl_claim": False,
        "research_only": True,
    }


def reconcile_fills(
    expected: pl.DataFrame,
    actual: pl.DataFrame,
    *,
    key: tuple[str, ...] = ("security_id", "fill_time"),
    qty_col: str = "quantity",
    price_col: str = "price",
    tol: float = 1e-9,
) -> dict[str, Any]:
    """Match two fills frames on ``key`` and compare quantity/price.

    Rows must be unique on the key in each frame (duplicates are counted,
    not silently dropped — a duplicated fill key is itself a finding).
    """
    for name, frame in (("expected", expected), ("actual", actual)):
        missing = set(key) | {qty_col, price_col}
        missing -= set(frame.columns)
        if missing:
            raise ValueError(f"{name} fills missing columns: {sorted(missing)}")
    exp_dup = expected.height - expected.select(key).unique().height
    act_dup = actual.height - actual.select(key).unique().height
    joined = expected.select(
        *key, pl.col(qty_col).alias("q_exp"), pl.col(price_col).alias("p_exp")
    ).join(
        actual.select(*key, pl.col(qty_col).alias("q_act"), pl.col(price_col).alias("p_act")),
        on=list(key),
        how="full",
    )
    both = joined.drop_nulls()
    only_exp = int(joined.filter(pl.col("q_act").is_null()).height)
    only_act = int(joined.filter(pl.col("q_exp").is_null()).height)
    mismatches: list[str] = []
    if exp_dup:
        mismatches.append(f"expected frame has {exp_dup} duplicated key rows")
    if act_dup:
        mismatches.append(f"actual frame has {act_dup} duplicated key rows")
    qd = np.abs((both["q_act"] - both["q_exp"]).to_numpy().astype(float)) if both.height else np.array([])
    pd_ = np.abs((both["p_act"] - both["p_exp"]).to_numpy().astype(float)) if both.height else np.array([])
    n_qty_bad = int((qd > tol).sum()) if qd.size else 0
    n_px_bad = int((pd_ > tol).sum()) if pd_.size else 0
    if n_qty_bad:
        mismatches.append(f"{n_qty_bad} matched fills differ on quantity")
    if n_px_bad:
        mismatches.append(f"{n_px_bad} matched fills differ on price")
    if only_exp or only_act:
        mismatches.append(f"unmatched fills: expected-only {only_exp}, actual-only {only_act}")
    return {
        "match": not mismatches,
        "n_matched": int(both.height),
        "unmatched_expected": only_exp,
        "unmatched_actual": only_act,
        "n_qty_mismatch": n_qty_bad,
        "n_price_mismatch": n_px_bad,
        "max_abs_qty_delta": float(qd.max()) if qd.size else 0.0,
        "max_abs_price_delta": float(pd_.max()) if pd_.size else 0.0,
        "mismatches": mismatches,
        "tol": float(tol),
        "live_pnl_claim": False,
        "research_only": True,
    }
