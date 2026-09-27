"""Research-only event-driven execution simulator.

A strict generalization of the daily next-open backtest. Zero signal-to-order
latency, zero order-to-exchange latency, zero costs, and the next-open fill
reproduce ``run_backtest`` / ``run_backtest_fast``. Nothing here routes live
orders.
"""

from quant_fund.backtest.event_sim.clock import EventClock, EventKind, ScheduledEvent
from quant_fund.backtest.event_sim.sensitivity import (
    execution_sensitivity,
    format_sensitivity_table,
)
from quant_fund.backtest.event_sim.simulator import EventSimResult, EventSimSpec, run_event_backtest

__all__ = [
    "EventClock",
    "EventKind",
    "EventSimResult",
    "EventSimSpec",
    "ScheduledEvent",
    "execution_sensitivity",
    "format_sensitivity_table",
    "run_event_backtest",
]
