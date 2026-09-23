"""Frozen Lightspeed / stockbook parameters. Research-only Dipcatcher port.

Yahoo expected-metrics from the private repo are *not* Dipcatcher claims.
``live_disabled`` is forced True. Champion ridge and ``blend_weight`` stay put.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TqqqParams:
    fast_ema: int = 20
    slow_ema: int = 180
    vol_window: int = 20
    inverse_confirm: int = 0
    signal_delay_sessions: int = 1
    rebalance_every: int = 5
    vol_budget: float = 10.0
    min_tqqq_weight: float = 0.0
    max_tqqq_weight: float = 0.98
    max_risk_weight: float = 1.0
    weight_quantum: float = 0.01
    no_trade_band: float = 0.02
    crash_vol: float = 0.23
    crash_gap_max: float = 0.05
    weak_trend_gap: float = 0.05
    weak_trend_multiplier: float = 1.0
    sqqq_target: float = 0.0
    vol_ref: str = "signal"
    flatten_when_fast_below_slow: bool = True
    crash_lookback: int = 0
    crash_return: float = 0.0


@dataclass(frozen=True)
class TqqqInstruments:
    signal: str = "QQQ"
    risk_on: str = "TQQQ"
    defensive: str = "SGOV"
    inverse: str | None = None


@dataclass(frozen=True)
class TqqqSpec:
    family: str
    name: str
    status: str
    instruments: TqqqInstruments
    params: TqqqParams
    cost_bps: float = 10.0
    live_disabled: bool = True

    @property
    def warmup(self) -> int:
        p = self.params
        return max(p.slow_ema, p.vol_window, p.inverse_confirm, 2) + 1


@dataclass(frozen=True)
class MomentumParams:
    mom_fast: int = 63
    mom_slow: int = 126
    mom_blend: float = 1.0
    trend_sma: int = 200
    min_score: float = 0.0
    top_k: int = 1
    vol_window: int = 20
    vol_budget: float = 0.6
    max_position_weight: float = 0.95
    max_gross_weight: float = 0.98
    rebalance_every: int = 5
    signal_delay_sessions: int = 1
    weight_quantum: float = 0.01
    no_trade_band: float = 0.05
    crash_lookback: int = 10
    crash_return: float = -0.2

    @property
    def warmup(self) -> int:
        return (
            max(
                self.trend_sma,
                self.mom_slow,
                self.mom_fast,
                self.vol_window,
                self.crash_lookback,
            )
            + 2
        )


@dataclass(frozen=True)
class MomentumUniverse:
    risk: tuple[str, ...]
    defensive: str = "SGOV"
    benchmarks: tuple[str, ...] = ("SPY", "QQQ")

    @property
    def traded(self) -> tuple[str, ...]:
        return tuple(self.risk) + (self.defensive,)


@dataclass(frozen=True)
class MomentumSpec:
    family: str
    name: str
    status: str
    universe: MomentumUniverse
    params: MomentumParams
    max_single_name_weight: float = 0.95
    cost_bps: float = 10.0
    live_disabled: bool = True


# Frozen Lightspeed selection / holdout cut. Params were picked on IS
# through selection_end; holdout is not for retuning.
SELECTION_END = "2024-12-31"
HOLDOUT_START = "2025-01-02"

STOCK_MOMENTUM_UNIVERSE: tuple[str, ...] = (
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "AVGO",
    "TSLA",
    "AMD",
    "NFLX",
    "XLK",
    "XLY",
    "XLE",
    "XLF",
    "XLV",
    "XLU",
    "SMH",
    "GLD",
    "TLT",
)

NAUTICA_EXTRA: tuple[str, ...] = ("PLTR", "MU", "SMCI", "ARM")


def tqqq_long_full_v1() -> TqqqSpec:
    """Frozen tqqq-long-full-v1. Alpaca ALL-LIVE stays off in this port."""
    return TqqqSpec(
        family="tqqq-long-full-v1",
        name="Full-LONG TQQQ with crash exit v1",
        status="frozen",
        instruments=TqqqInstruments(),
        params=TqqqParams(),
        live_disabled=True,
    )


def stock_momentum_v1() -> MomentumSpec:
    """Frozen stock-momentum-v1 (19-name concentrated book)."""
    return MomentumSpec(
        family="stock-momentum-v1",
        name="Concentrated stock/sector momentum with crash exit v1",
        status="frozen",
        universe=MomentumUniverse(risk=STOCK_MOMENTUM_UNIVERSE),
        params=MomentumParams(),
        live_disabled=True,
    )


def nautica_momentum_v1() -> MomentumSpec:
    """Declared nautica-momentum-v1: frozen params + tug-of-war names."""
    return MomentumSpec(
        family="nautica-momentum-v1",
        name="Nautica concentrated momentum + tug-of-war names v1",
        status="declared",
        universe=MomentumUniverse(risk=STOCK_MOMENTUM_UNIVERSE + NAUTICA_EXTRA),
        params=MomentumParams(),
        live_disabled=True,
    )


def frozen_families() -> dict[str, object]:
    tqqq = tqqq_long_full_v1()
    stock = stock_momentum_v1()
    nautica = nautica_momentum_v1()
    return {
        "source": "https://github.com/cosmic-hydra/lightspeed",
        "research_only": True,
        "live_pnl_claim": False,
        "live_disabled": True,
        "broker": None,
        "champion": "ridge",
        "blend_weight": 0,
        "families": {
            tqqq.family: {
                "status": tqqq.status,
                "instruments": {
                    "signal": tqqq.instruments.signal,
                    "risk_on": tqqq.instruments.risk_on,
                    "defensive": tqqq.instruments.defensive,
                },
                "params": tqqq.params.__dict__,
                "cost_bps": tqqq.cost_bps,
            },
            stock.family: {
                "status": stock.status,
                "risk": list(stock.universe.risk),
                "defensive": stock.universe.defensive,
                "params": {**stock.params.__dict__},
                "cost_bps": stock.cost_bps,
            },
            nautica.family: {
                "status": nautica.status,
                "risk": list(nautica.universe.risk),
                "defensive": nautica.universe.defensive,
                "params": {**nautica.params.__dict__},
                "cost_bps": nautica.cost_bps,
            },
        },
        "note": (
            "Yahoo expected-metrics from Lightspeed are not Dipcatcher P&L. "
            "AFML metalabel can only reduce risk. nautica is a CS challenger; "
            "champion remains public ridge."
        ),
    }
