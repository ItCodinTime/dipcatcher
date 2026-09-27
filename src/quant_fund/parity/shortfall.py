"""Paper-versus-backtest implementation-shortfall attribution.

Decomposes the simulated terminal-value gap

    backtest_terminal_delta - paper_terminal_delta

into six additive components. A positive component means the paper book
underperformed the backtest book by that many dollars of marked value.

* ``delay`` — shared quantity times (paper exec price - backtest exec price)
* ``spread`` — paper spread dollars minus backtest spread dollars
* ``impact`` — paper impact dollars minus backtest impact dollars
* ``fees`` — other explicit costs (commission, and turnover bps if the
  broker charged them) paper minus backtest
* ``missed_fills`` — backtest-only quantity marked from its exec price to
  the backtest terminal print
* ``opportunity`` — the paper-only quantity's marked value, signed so it
  explains the gap, plus any shared-quantity difference in the terminal
  print (a vendor/revision mark gap)

Fills are paired in listed order within ``(event_time, security_id)``.
The six components sum to the fill-implied gap for any pairing. This is
a simulated research diagnostic. It is not a live P&L claim.

Per-fill arrival shortfall versus the decision price reuses
:func:`quant_fund.execution.implementation_shortfall.fill_shortfall` and
is reported separately. It does not enter the six-way identity.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from quant_fund.execution.implementation_shortfall import fill_shortfall

_COMPONENTS = ("delay", "spread", "impact", "fees", "missed_fills", "opportunity")


def _as_rows(fills: Any) -> list[dict[str, Any]]:
    if hasattr(fills, "to_dicts"):
        return [dict(row) for row in fills.to_dicts()]
    return [dict(row) for row in fills]


def _f(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number):
        raise ValueError("fill fields must be finite")
    return number


def _mark_pair(marks: Mapping[str, Any], sid: str) -> tuple[float, float]:
    if sid not in marks:
        raise ValueError(f"missing terminal mark for {sid}")
    value = marks[sid]
    if isinstance(value, tuple | list):
        if len(value) != 2:
            raise ValueError(f"terminal mark for {sid} must be a price or a (backtest, paper) pair")
        bt, paper = float(value[0]), float(value[1])
    else:
        bt = paper = float(value)
    if not math.isfinite(bt) or not math.isfinite(paper) or bt <= 0.0 or paper <= 0.0:
        raise ValueError(f"terminal mark for {sid} must be finite and positive")
    return bt, paper


def _explicit_parts(row: Mapping[str, Any]) -> tuple[float, float, float]:
    """Return ``(fees, spread, impact)`` with fees = explicit - spread - impact."""
    spread = _f(row.get("spread", row.get("fill_spread")))
    impact = _f(row.get("impact", row.get("fill_impact")))
    fee = _f(row.get("fee", row.get("fill_fee")))
    if "explicit_cost" in row and row.get("explicit_cost") is not None:
        explicit = _f(row.get("explicit_cost"))
    else:
        explicit = fee + spread + impact
    if spread < 0.0 or impact < 0.0 or explicit < -1e-8:
        raise ValueError("explicit costs must be non-negative")
    fees = explicit - spread - impact
    return fees, spread, impact


def _leg_value(qty: float, price: float, explicit: float, mark: float) -> float:
    if qty == 0.0:
        return -explicit
    if price <= 0.0:
        raise ValueError("fill price must be positive")
    return qty * (mark - price) - explicit


def _zero() -> dict[str, float]:
    return {name: 0.0 for name in _COMPONENTS}


def _split_pair(
    bt: Mapping[str, Any] | None,
    sh: Mapping[str, Any] | None,
    mark_bt: float,
    mark_sh: float,
) -> dict[str, float]:
    out = _zero()
    q_bt = _f(bt.get("signed_qty")) if bt is not None else 0.0
    q_sh = _f(sh.get("signed_qty")) if sh is not None else 0.0
    px_bt = _f(bt.get("price"), default=mark_bt) if bt is not None else mark_bt
    px_sh = _f(sh.get("price"), default=mark_sh) if sh is not None else mark_sh
    fees_bt, spread_bt, impact_bt = _explicit_parts(bt) if bt is not None else (0.0, 0.0, 0.0)
    fees_sh, spread_sh, impact_sh = _explicit_parts(sh) if sh is not None else (0.0, 0.0, 0.0)
    out["fees"] = fees_sh - fees_bt
    out["spread"] = spread_sh - spread_bt
    out["impact"] = impact_sh - impact_bt
    same_direction = q_bt * q_sh > 0.0
    if same_direction:
        sign = 1.0 if q_bt > 0.0 else -1.0
        q_c = sign * min(abs(q_bt), abs(q_sh))
        q_bt_only = q_bt - q_c
        q_sh_only = q_sh - q_c
    else:
        q_c = 0.0
        q_bt_only = q_bt
        q_sh_only = q_sh
    out["delay"] = q_c * (px_sh - px_bt) if q_c != 0.0 else 0.0
    out["missed_fills"] = q_bt_only * (mark_bt - px_bt) if q_bt_only != 0.0 else 0.0
    unmatched = -(q_sh_only * (mark_sh - px_sh)) if q_sh_only != 0.0 else 0.0
    shared_mark_gap = q_c * (mark_bt - mark_sh) if q_c != 0.0 else 0.0
    out["opportunity"] = unmatched + shared_mark_gap
    out["shared_mark_gap"] = shared_mark_gap
    out["unmatched_shadow"] = unmatched
    return out


def _group(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[Any, str], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, str], list[dict[str, Any]]] = {}
    for row in rows:
        if "security_id" not in row:
            raise ValueError("fill row missing security_id")
        key = (row.get("event_time"), str(row["security_id"]))
        grouped.setdefault(key, []).append(dict(row))
    for key in grouped:
        grouped[key].sort(
            key=lambda item: (
                float(item.get("price") or 0.0),
                float(item.get("signed_qty") or 0.0),
            )
        )
    return grouped


def _arrival(paper_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Supplemental decision-price shortfall. Not part of the cross-book identity."""
    total = {"drift": 0.0, "fee": 0.0, "spread_cost": 0.0, "impact_cost": 0.0, "total_is": 0.0}
    n = 0
    for row in paper_rows:
        qty = _f(row.get("signed_qty"))
        decision = row.get("decision_price")
        if qty == 0.0 or decision is None:
            continue
        side = 1.0 if qty > 0.0 else -1.0
        part = fill_shortfall(
            side_sign=side,
            quantity=abs(qty),
            decision_price=float(decision),
            exec_price=float(row["price"]),
            fee=_f(row.get("fee", row.get("fill_fee"))),
            spread_cost=_f(row.get("spread", row.get("fill_spread"))),
            impact_cost=_f(row.get("impact", row.get("fill_impact"))),
        )
        total["drift"] += part["drift"]
        total["fee"] += part["fee"]
        total["spread_cost"] += part["spread_cost"]
        total["impact_cost"] += part["impact_cost"]
        total["total_is"] += part["total_is"]
        n += 1
    total["n_fills"] = float(n)
    return {
        "in_terminal_gap": False,
        "n_fills": n,
        "drift": total["drift"],
        "fee": total["fee"],
        "spread_cost": total["spread_cost"],
        "impact_cost": total["impact_cost"],
        "total_is": total["total_is"],
        "live_pnl_claim": False,
        "research_only": True,
    }


def attribute_shortfall(
    backtest_fills: Any,
    paper_fills: Any,
    end_marks: Mapping[str, Any],
    *,
    backtest_terminal_delta: float | None = None,
    paper_terminal_delta: float | None = None,
) -> dict[str, Any]:
    """Decompose the simulated book gap into delay, spread, impact, fees, missed fills, and opportunity."""
    bt_rows = _as_rows(backtest_fills)
    sh_rows = _as_rows(paper_fills)
    bt_groups = _group(bt_rows)
    sh_groups = _group(sh_rows)
    keys = sorted(set(bt_groups) | set(sh_groups), key=lambda item: (str(item[0]), item[1]))
    components = _zero()
    shared_mark_gap = 0.0
    unmatched_shadow = 0.0
    bt_value = 0.0
    sh_value = 0.0
    by_security: dict[str, dict[str, float]] = {}
    for key in keys:
        _event, sid = key
        mark_bt, mark_sh = _mark_pair(end_marks, sid)
        bt_list = bt_groups.get(key, [])
        sh_list = sh_groups.get(key, [])
        n = max(len(bt_list), len(sh_list))
        bucket = by_security.setdefault(sid, _zero())
        for index in range(n):
            bt = bt_list[index] if index < len(bt_list) else None
            sh = sh_list[index] if index < len(sh_list) else None
            part = _split_pair(bt, sh, mark_bt, mark_sh)
            for name in _COMPONENTS:
                components[name] += part[name]
                bucket[name] += part[name]
            shared_mark_gap += part["shared_mark_gap"]
            unmatched_shadow += part["unmatched_shadow"]
            if bt is not None:
                fees, spread, impact = _explicit_parts(bt)
                bt_value += _leg_value(
                    _f(bt.get("signed_qty")),
                    _f(bt.get("price"), default=mark_bt),
                    fees + spread + impact,
                    mark_bt,
                )
            if sh is not None:
                fees, spread, impact = _explicit_parts(sh)
                sh_value += _leg_value(
                    _f(sh.get("signed_qty")),
                    _f(sh.get("price"), default=mark_sh),
                    fees + spread + impact,
                    mark_sh,
                )
    fill_gap = bt_value - sh_value
    component_sum = math.fsum(components[name] for name in _COMPONENTS)
    algebraic_residual = fill_gap - component_sum
    if backtest_terminal_delta is None or paper_terminal_delta is None:
        terminal_gap = fill_gap
    else:
        terminal_gap = float(backtest_terminal_delta) - float(paper_terminal_delta)
    return {
        "terminal_gap": terminal_gap,
        "fill_gap": fill_gap,
        "components": {name: components[name] for name in _COMPONENTS},
        "component_sum": component_sum,
        "residual": terminal_gap - component_sum,
        "algebraic_residual": algebraic_residual,
        "opportunity_detail": {
            "unmatched_shadow_quantity": unmatched_shadow,
            "shared_mark_gap": shared_mark_gap,
        },
        "by_security": [{"security_id": sid, **by_security[sid]} for sid in sorted(by_security)],
        "paper_arrival_shortfall": _arrival(sh_rows),
        "convention": (
            "positive component means the paper book underperformed the "
            "backtest book in simulated marked value"
        ),
        "identity": "delay + spread + impact + fees + missed_fills + opportunity = fill_gap",
        "live_pnl_claim": False,
        "research_only": True,
        "would_promote_live": False,
    }
