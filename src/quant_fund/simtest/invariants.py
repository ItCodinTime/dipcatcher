"""Session invariants checked by the swarm.

A passing session either completes a step or records ``fail_closed`` /
``recovered``. Cash and champion shares reconstruct from fills plus borrow.
Filled order ids are unique. A second resume does not append orders. The
promotion receipt verifies, and the run stays synthetic and research-only.
"""

from __future__ import annotations

from dataclasses import dataclass

from quant_fund.simtest.session import SessionResult

_OUTCOMES = frozenset({"ok", "fail_closed", "recovered"})


@dataclass(frozen=True)
class InvariantReport:
    ok: bool
    failures: tuple[str, ...]


def check_invariants(result: SessionResult) -> InvariantReport:
    failures: list[str] = []
    if result.research_only is not True:
        failures.append("research_only")
    if result.live_pnl_claim is not False:
        failures.append("live_pnl_claim")
    if result.data_source != "SYNTHETIC":
        failures.append("data_source")
    for outcome in result.outcomes:
        if outcome not in _OUTCOMES:
            failures.append(f"outcome:{outcome}")
    if not result.idempotent_resume:
        failures.append("resume_not_idempotent")
    if result.receipt_errors:
        failures.append("receipt:" + ",".join(result.receipt_errors))
    if result.cash is not None and not result.ledger_ok:
        failures.append("ledger_schema")
    for error in result.conservation_errors:
        failures.append(error)
    if len(result.fill_ids) != len(set(result.fill_ids)):
        failures.append("duplicate_fill_ids")
    extra_hashes = 1 if result.backtest != "not_run" else 0
    if len(result.state_hashes) != len(result.outcomes) + extra_hashes:
        failures.append("state_hash_count")
    if not result.state_hashes:
        failures.append("short_trace")
    return InvariantReport(ok=not failures, failures=tuple(failures))
