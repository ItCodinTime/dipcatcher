"""Labeled synthetic equity panel. Results are not evidence of live edge."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from quant_fund.data.calendars import session_days
from quant_fund.utils.seeds import set_global_seed

SOURCE = "synthetic"
REVISION = "SYNTHETIC"
TZ = UTC
SECTORS = ["Tech", "Health", "Finance", "Energy"]


def _close_ts(d) -> datetime:
    return datetime(d.year, d.month, d.day, 16, 0, tzinfo=TZ)


class SyntheticMarketProvider:
    """Factor model with a regime vol shift, one split, and one delisting.

    `planted_signal` at close t is a labeled SYNTHETIC oracle for r_{t+1}.
    It is an AR(1) residual known at t; next-day residual mean is
    `oracle_beta * planted_signal_t`. Recovery is a correctness test, not
    evidence of live edge.
    """

    def __init__(
        self,
        n_assets: int = 12,
        n_days: int = 320,
        seed: int = 7,
        start: datetime | None = None,
        oracle_beta: float = 0.015,
        oracle_phi: float = 0.70,
    ) -> None:
        if int(n_assets) < 2:
            raise ValueError("n_assets must be >= 2 (market + at least one name)")
        if int(n_days) < 16:
            raise ValueError("n_days must be >= 16 (delist window needs 15 trailing sessions)")
        seed_i = int(seed)
        if seed_i < 0 or seed_i > 2**32 - 1:
            raise ValueError("seed must be between 0 and 2**32 - 1")
        set_global_seed(seed_i)
        self.n_assets = int(n_assets)
        self.n_days = int(n_days)
        self.seed = seed_i
        self.oracle_beta = float(oracle_beta)
        self.oracle_phi = float(oracle_phi)
        start_d = (start or datetime(2018, 1, 2, tzinfo=TZ)).date()
        self.days = session_days(start_d, start_d + timedelta(days=int(n_days * 2)))[:n_days]
        self._bars, self._actions, self._master = self._simulate()

    def _simulate(self) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        rng = np.random.default_rng(self.seed)
        n, t = self.n_assets, len(self.days)
        # Tight betas so market-excess is mostly idiosyncratic + the labeled oracle.
        betas = 0.95 + 0.10 * rng.random(n)
        sector_idx = np.arange(n) % len(SECTORS)
        factor = rng.normal(0.0003, 0.008, size=t)
        factor[t // 2 :] *= 2.5
        eps = rng.normal(0.0, 0.012, size=(t, n))
        vol_scale = np.ones(t)
        vol_scale[t // 2 :] = 2.0
        eps *= vol_scale[:, None]
        shock = rng.normal(0.0, 1.0, size=(t, n))
        signal = np.zeros((t, n))
        signal[0] = shock[0]
        phi = self.oracle_phi
        innov = float(np.sqrt(max(1.0 - phi * phi, 1e-8)))
        for i in range(1, t):
            signal[i] = phi * signal[i - 1] + innov * shock[i]
        cs_mean = signal.mean(axis=1, keepdims=True)
        cs_std = signal.std(axis=1, keepdims=True) + 1e-12
        signal = (signal - cs_mean) / cs_std
        alpha = np.zeros((t, n))
        # Scale the plant with the same vol regime so SNR (and IC) is stable OOS.
        alpha[1:] = self.oracle_beta * signal[:-1] * vol_scale[1:, None]
        rets = betas * factor[:, None] + eps + alpha
        rets[:, 0] = factor  # asset 0 is the market
        prices = 50.0 * np.exp(np.cumsum(rets, axis=0))
        # split 2-for-1 on asset 1 at 60% of sample: raw price halves
        split_i = int(0.6 * t)
        prices[split_i:, 1] *= 0.5
        volumes = rng.integers(200_000, 800_000, size=(t, n)).astype(float)
        ingested = datetime.now(tz=TZ)

        bar_rows: list[dict] = []
        for j in range(n):
            sid = "SEC_MKT" if j == 0 else f"SEC_{j:04d}"
            sym = "MKT" if j == 0 else f"S{j:04d}"
            for i, d in enumerate(self.days):
                if j == n - 1 and i >= t - 15:
                    continue  # delist last name
                px = float(prices[i, j])
                high = px * (1.0 + abs(float(rng.normal(0, 0.004))))
                low = px * (1.0 - abs(float(rng.normal(0, 0.004))))
                opn = float(prices[i - 1, j]) if i else px
                ts = _close_ts(d)
                bar_rows.append(
                    {
                        "security_id": sid,
                        "symbol": sym,
                        "event_time": ts,
                        "available_time": ts,
                        "ingested_time": ingested,
                        "source": SOURCE,
                        "revision_id": REVISION,
                        "open": opn,
                        "high": max(high, px, opn),
                        "low": min(low, px, opn),
                        "close": px,
                        "volume": float(volumes[i, j]),
                        "currency": "USD",
                        "session": "rth",
                        "planted_signal": float(signal[i, j]),
                    }
                )
        bars = pl.DataFrame(bar_rows)

        split_ts = _close_ts(self.days[split_i])
        delist_ts = _close_ts(self.days[t - 15])
        delist_sid = f"SEC_{n - 1:04d}"
        actions = pl.DataFrame(
            [
                {
                    "security_id": "SEC_0001",
                    "event_time": split_ts,
                    "available_time": split_ts,
                    "ingested_time": ingested,
                    "source": SOURCE,
                    "revision_id": REVISION,
                    "action_type": "split",
                    "factor": 2.0,
                    "amount": None,
                    "new_ticker": None,
                },
                {
                    "security_id": delist_sid,
                    "event_time": delist_ts,
                    "available_time": delist_ts,
                    "ingested_time": ingested,
                    "source": SOURCE,
                    "revision_id": REVISION,
                    "action_type": "delist",
                    "factor": None,
                    "amount": None,
                    "new_ticker": None,
                },
            ]
        )
        master_rows = []
        for j in range(n):
            sid = "SEC_MKT" if j == 0 else f"SEC_{j:04d}"
            ticker = "MKT" if j == 0 else f"S{j:04d}"
            valid_to = _close_ts(self.days[-16]) if j == n - 1 else None
            valid_from = _close_ts(self.days[0])
            master_rows.append(
                {
                    "security_id": sid,
                    "ticker": ticker,
                    "name": ticker,
                    "exchange": "XNYS",
                    "currency": "USD",
                    "sector": "Market" if j == 0 else SECTORS[int(sector_idx[j])],
                    "industry": "Index" if j == 0 else SECTORS[int(sector_idx[j])],
                    "security_type": "common_stock",
                    "valid_from": valid_from,
                    "valid_to": valid_to,
                    "available_time": valid_from,
                    "ingested_time": ingested,
                    "source": SOURCE,
                    "revision_id": REVISION,
                }
            )
        master = pl.DataFrame(master_rows)
        return bars, actions, master

    def get_bars(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        security_ids: list[str] | None = None,
    ) -> pl.DataFrame:
        df = self._bars
        if start is not None:
            df = df.filter(pl.col("event_time") >= start)
        if end is not None:
            df = df.filter(pl.col("event_time") <= end)
        if security_ids is not None:
            df = df.filter(pl.col("security_id").is_in(security_ids))
        return df

    def get_corporate_actions(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        df = self._actions
        if start is not None:
            df = df.filter(pl.col("event_time") >= start)
        if end is not None:
            df = df.filter(pl.col("event_time") <= end)
        return df

    def get_security_master(self) -> pl.DataFrame:
        return self._master
