"""Shadow replay of a recorded or streamed tape.

Calls the same :func:`quant_fund.parity.replay.replay_session` as the
backtest reference, with ``origin='shadow'``. Wall-clock pacing sleeps
between decision times; accelerated pacing does not. Orders go to
``SimulatedBroker`` only. Nothing in this module submits a real order.
"""

from __future__ import annotations

from collections.abc import Iterable

from quant_fund.config.models import AppConfig
from quant_fund.parity.replay import ParityRun, ReplayOptions, replay_session
from quant_fund.parity.session import Bar, MarketSession
from quant_fund.parity.strategy import DecideFn, Strategy


def run_shadow_session(
    session: MarketSession | Iterable[Bar],
    strategy: Strategy | DecideFn,
    config: AppConfig,
    *,
    options: ReplayOptions | None = None,
) -> ParityRun:
    """Replay ``strategy`` on the shadow clock through the simulated broker."""
    return replay_session(
        session,
        strategy,
        config,
        origin="shadow",
        options=options,
    )
