"""Bar-by-bar parity between a backtest reference ledger and a shadow ledger.

Every divergent row gets exactly one cause, using a fixed precedence so a
later symptom is not mistaken for the root:

1. ``data`` — vendor, revision, release time, or tape prices differ
2. ``timing`` — decision clock differs from the other origin
3. ``code_path`` — call-trace digest differs
4. ``state_drift`` — broker or strategy state differs after a restart
5. ``rounding`` — lot size or the rounded weight differs
6. ``costs`` — commission, spread, or impact parameters differ
7. ``fills`` — fill price, quantity, or explicit cost dollars differ

A state-digest mismatch with no restart, and no earlier cause, is still
``state_drift`` (detail ``state_digest``). A target-weight mismatch with
every input matched is ``code_path`` (detail ``output_mismatch``): the
strategy returned a different weight from the same observed inputs.

The checker does not raise on mismatches. The ledger is the verdict.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

CAUSE_PRECEDENCE: tuple[str, ...] = (
    "data",
    "timing",
    "code_path",
    "state_drift",
    "rounding",
    "costs",
    "fills",
)

_DATA_FIELDS: tuple[str, ...] = (
    "source",
    "revision_id",
    "available_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "exec_open",
    "exec_close",
    "exec_source",
    "exec_revision",
)
_NUMERIC_DATA = frozenset({"open", "high", "low", "close", "volume", "exec_open", "exec_close"})
_COST_FIELDS = ("commission_bps", "half_spread_bps", "impact_y", "bps_per_turnover")
_FILL_FIELDS = (
    "fill_signed_qty",
    "fill_price",
    "fill_fee",
    "fill_spread",
    "fill_impact",
    "explicit_cost",
)


def _close(left: object, right: object, tol: float) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    if isinstance(left, datetime) or isinstance(right, datetime):
        return left == right
    if isinstance(left, str) or isinstance(right, str):
        return left == right
    try:
        lf = float(left)  # type: ignore[arg-type]
        rf = float(right)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return left == right
    return abs(lf - rf) <= tol


def _diff_fields(
    left: Mapping[str, Any], right: Mapping[str, Any], fields: Sequence[str], tol: float
) -> list[str]:
    return [name for name in fields if not _close(left.get(name), right.get(name), tol)]


def _rows_of(run: Any) -> list[dict[str, Any]]:
    rows = getattr(run, "rows", None)
    if rows is not None:
        return [dict(row) for row in rows]
    ledger = getattr(run, "ledger", None)
    if ledger is None:
        raise TypeError("parity run has no ledger")
    if hasattr(ledger, "to_dicts"):
        return [dict(row) for row in ledger.to_dicts()]
    return [dict(row) for row in ledger]


def _key(row: Mapping[str, Any]) -> tuple[Any, str]:
    return (row.get("event_time"), str(row.get("security_id")))


def _missing_cause(
    present: Mapping[str, Any], other: Sequence[Mapping[str, Any]]
) -> tuple[str, str]:
    sid = str(present.get("security_id"))
    decision_time = present.get("decision_time")
    event_time = present.get("event_time")
    for row in other:
        if str(row.get("security_id")) != sid:
            continue
        if row.get("decision_time") == decision_time and row.get("event_time") != event_time:
            return "timing", "shifted_bar"
    return "data", "missing_bar"


def attribute_pair(
    backtest: Mapping[str, Any] | None,
    shadow: Mapping[str, Any] | None,
    *,
    tol: float = 1e-6,
    other_backtest: Sequence[Mapping[str, Any]] = (),
    other_shadow: Sequence[Mapping[str, Any]] = (),
) -> tuple[str | None, str]:
    """Return ``(cause, detail)``. ``cause`` is ``None`` when the rows match."""
    if tol < 0.0 or tol != tol:
        raise ValueError("tol must be finite and non-negative")
    if backtest is None and shadow is None:
        raise ValueError("at least one side of a parity row must be present")
    if backtest is None:
        assert shadow is not None
        return _missing_cause(shadow, other_backtest)
    if shadow is None:
        return _missing_cause(backtest, other_shadow)

    data_diff = _diff_fields(backtest, shadow, _DATA_FIELDS, tol)
    timing_diff = not _close(backtest.get("decision_time"), shadow.get("decision_time"), tol)
    code_diff = backtest.get("code_path_digest") != shadow.get("code_path_digest")
    state_diff = backtest.get("state_digest") != shadow.get("state_digest")
    gen_bt = int(backtest.get("restart_generation") or 0)
    gen_sh = int(shadow.get("restart_generation") or 0)
    restarted = bool(backtest.get("restarted") or shadow.get("restarted") or gen_bt or gen_sh)
    lot_diff = not _close(backtest.get("lot_size"), shadow.get("lot_size"), tol)
    raw_diff = not _close(backtest.get("raw_weight"), shadow.get("raw_weight"), tol)
    rounded_diff = not _close(backtest.get("rounded_weight"), shadow.get("rounded_weight"), tol)
    cost_diff = _diff_fields(backtest, shadow, _COST_FIELDS, tol)
    fill_diff = _diff_fields(backtest, shadow, _FILL_FIELDS, tol)

    same = not any(
        (
            data_diff,
            timing_diff,
            code_diff,
            state_diff,
            lot_diff,
            raw_diff,
            rounded_diff,
            cost_diff,
            fill_diff,
        )
    )
    if same:
        return None, ""
    if data_diff:
        return "data", ",".join(data_diff)
    if timing_diff:
        return "timing", "decision_time"
    if code_diff:
        return "code_path", "call_trace"
    if state_diff and restarted:
        return "state_drift", "after_restart"
    if lot_diff or (not raw_diff and rounded_diff):
        return "rounding", "lot_size" if lot_diff else "rounded_weight"
    if cost_diff:
        return "costs", ",".join(cost_diff)
    if fill_diff:
        return "fills", ",".join(fill_diff)
    if state_diff:
        return "state_drift", "state_digest"
    if raw_diff or rounded_diff:
        return "code_path", "output_mismatch"
    return "fills", "unclassified"


def _weight_delta(backtest: Mapping[str, Any] | None, shadow: Mapping[str, Any] | None) -> float:
    def _w(row: Mapping[str, Any] | None) -> float:
        if row is None:
            return 0.0
        value = row.get("rounded_weight")
        if value is None:
            return 0.0
        return float(value)

    return abs(_w(backtest) - _w(shadow))


def _side_snapshot(row: Mapping[str, Any] | None, prefix: str) -> dict[str, Any]:
    if row is None:
        return {
            f"{prefix}_decision_time": None,
            f"{prefix}_revision_id": None,
            f"{prefix}_rounded_weight": None,
            f"{prefix}_fill_price": None,
            f"{prefix}_fill_signed_qty": None,
            f"{prefix}_state_digest": None,
            f"{prefix}_code_path_digest": None,
        }
    return {
        f"{prefix}_decision_time": row.get("decision_time"),
        f"{prefix}_revision_id": row.get("revision_id"),
        f"{prefix}_rounded_weight": row.get("rounded_weight"),
        f"{prefix}_fill_price": row.get("fill_price"),
        f"{prefix}_fill_signed_qty": row.get("fill_signed_qty"),
        f"{prefix}_state_digest": row.get("state_digest"),
        f"{prefix}_code_path_digest": row.get("code_path_digest"),
    }


def check_parity(backtest: Any, shadow: Any, *, tol: float = 1e-6) -> dict[str, Any]:
    """Align ``backtest`` and ``shadow`` ledgers and attribute divergences.

    Both arguments are :class:`quant_fund.parity.replay.ParityRun` values
    or any object with a ``rows`` list or a polars ``ledger``.
    """
    bt_rows = _rows_of(backtest)
    sh_rows = _rows_of(shadow)
    bt_index = {_key(row): row for row in bt_rows}
    sh_index = {_key(row): row for row in sh_rows}
    if len(bt_index) != len(bt_rows) or len(sh_index) != len(sh_rows):
        raise ValueError("parity ledger has duplicate (event_time, security_id) keys")
    keys = sorted(set(bt_index) | set(sh_index), key=lambda item: (str(item[0]), item[1]))
    ledger: list[dict[str, Any]] = []
    by_cause = {cause: 0 for cause in CAUSE_PRECEDENCE}
    max_abs = 0.0
    n_matched = 0
    n_missing_backtest = 0
    n_missing_shadow = 0
    for key in keys:
        bt = bt_index.get(key)
        sh = sh_index.get(key)
        if bt is None:
            n_missing_backtest += 1
        if sh is None:
            n_missing_shadow += 1
        cause, detail = attribute_pair(
            bt, sh, tol=tol, other_backtest=bt_rows, other_shadow=sh_rows
        )
        max_abs = max(max_abs, _weight_delta(bt, sh))
        if cause is None:
            n_matched += 1
            continue
        by_cause[cause] = by_cause.get(cause, 0) + 1
        event_time, security_id = key
        entry = {
            "event_time": event_time,
            "security_id": security_id,
            "cause": cause,
            "detail": detail,
        }
        entry.update(_side_snapshot(bt, "backtest"))
        entry.update(_side_snapshot(sh, "shadow"))
        ledger.append(entry)
    n_bars = len(keys)
    n_divergent = len(ledger)
    first = None
    if ledger:
        first = {
            "event_time": ledger[0]["event_time"],
            "security_id": ledger[0]["security_id"],
            "cause": ledger[0]["cause"],
            "detail": ledger[0]["detail"],
        }
    synthetic = bool(getattr(backtest, "synthetic", False) or getattr(shadow, "synthetic", False))
    return {
        "match": n_divergent == 0,
        "n_bars": n_bars,
        "n_matched": n_matched,
        "n_divergent": n_divergent,
        "n_missing_backtest": n_missing_backtest,
        "n_missing_shadow": n_missing_shadow,
        "divergence_rate": (n_divergent / n_bars) if n_bars else 0.0,
        "by_cause": by_cause,
        "first_divergence": first,
        "max_abs_weight_delta": max_abs,
        "tol": float(tol),
        "cause_precedence": list(CAUSE_PRECEDENCE),
        "ledger": ledger,
        "synthetic": synthetic,
        "data_source": "SYNTHETIC" if synthetic else "recorded_session",
        "live_pnl_claim": False,
        "research_only": True,
        "would_promote_live": False,
        "evidence": "simulated_broker_replay",
    }
