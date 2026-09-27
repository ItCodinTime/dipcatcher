"""Deterministic simulation testing for the research-to-paper path.

Clock, randomness, cache IO, and a simulated research API are injected.
The paper loop and simulated broker keep their default behaviour when this
package is not used. Sessions are synthetic and research-only.
"""

from quant_fund.simtest.eventlog import ReplayDivergence
from quant_fund.simtest.faults import Fault, FaultSchedule, schedule_from_seed, shrink_schedule
from quant_fund.simtest.invariants import InvariantReport, check_invariants
from quant_fund.simtest.runtime import DeterministicRuntime
from quant_fund.simtest.session import SessionResult, run_session
from quant_fund.simtest.swarm import SwarmReport, run_swarm

__all__ = [
    "DeterministicRuntime",
    "Fault",
    "FaultSchedule",
    "InvariantReport",
    "ReplayDivergence",
    "SessionResult",
    "SwarmReport",
    "check_invariants",
    "run_session",
    "run_swarm",
    "schedule_from_seed",
    "shrink_schedule",
]
