"""Backtest-timed reference replay.

Calls the same :func:`quant_fund.parity.replay.replay_session` as the
shadow, with ``origin='backtest'``. Orders go to ``SimulatedBroker`` only.
"""

from __future__ import annotations

from collections.abc import Iterable

from quant_fund.config.models import AppConfig
from quant_fund.parity.replay import ParityRun, ReplayOptions, replay_session
from quant_fund.parity.session import Bar, MarketSession
from quant_fund.parity.strategy import DecideFn, Strategy


def run_backtest_session(
    session: MarketSession | Iterable[Bar],
    strategy: Strategy | DecideFn,
    config: AppConfig,
    *,
    options: ReplayOptions | None = None,
) -> ParityRun:
    """Replay ``strategy`` on the backtest clock through the simulated broker."""
    return replay_session(
        session,
        strategy,
        config,
        origin="backtest",
        options=options,
    )
