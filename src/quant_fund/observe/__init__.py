"""Optional tracing, metrics, and JSON logs. Off unless DIPCATCHER_OBSERVE is set."""

from quant_fund.observe.flags import observe_enabled
from quant_fund.observe.logging import log_event
from quant_fund.observe.metrics import render_prometheus
from quant_fund.observe.tracing import span

__all__ = ["log_event", "observe_enabled", "render_prometheus", "span"]
