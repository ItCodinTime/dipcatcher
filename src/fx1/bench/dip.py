"""Dip Quality Score — the flagship fx-1/dipcatcher bench.

Research question, made calibratable: *for each significant drawdown event,
what is the probability of recovery within 1 / 3 / 6 / 12 months?* The bench
detects events causally (no forward-looking data at detection time), scores
probabilistic forecasts with proper rules (Brier, log-loss, ECE), and
publishes every event — failures included. Naive dip-buying mostly loses in
the literature; this bench measures *when* dips are overreactions instead of
asserting they are.

Event universe is frozen before scoring; honesty gates mirror the lab:
forbidden headline tokens in bench outputs fail closed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Mirrors the lab's forbidden research-headline tokens.
_FORBIDDEN_TOKENS = frozenset({"sharpe", "sortino", "calmar", "pnl", "nav"})


@dataclass(frozen=True)
class DipEvent:
    """A causally-detected drawdown event on one asset."""

    asset: str
    peak_date: str
    trough_date: str
    depth: float  # drawdown magnitude, positive fraction (e.g. 0.12)
    # Recovery flags per horizon (in bars); None = horizon not yet observable.
    recovered: dict[str, bool | None] = field(default_factory=dict)


@dataclass(frozen=True)
class DipForecast:
    """Calibrated recovery probabilities for one event."""

    asset: str
    trough_date: str
    probabilities: dict[str, float]  # horizon -> P(recovery)


def detect_dip_events(
    closes: list[float],
    dates: list[str],
    asset: str,
    *,
    threshold: float = 0.10,
    horizons_bars: dict[str, int] | None = None,
) -> list[DipEvent]:
    """Causal dip detection over a close series.

    An event fires when the close falls *threshold* below the running peak.
    Recovery within a horizon means the close regains the peak within that
    many bars after the trough-crossing date. Horizons extending past the
    data are recorded as None (not yet observable) — never imputed.
    """
    if len(closes) != len(dates):
        raise ValueError("closes and dates must align")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be in (0, 1)")
    horizons_bars = horizons_bars or {"1m": 21, "3m": 63, "6m": 126, "12m": 252}
    events: list[DipEvent] = []
    peak = -math.inf
    peak_date = ""
    in_dip = False
    for i, (price, date) in enumerate(zip(closes, dates, strict=True)):
        if price >= peak:
            peak, peak_date = price, date
            in_dip = False
            continue
        depth = (peak - price) / peak
        if depth >= threshold and not in_dip:
            recovered: dict[str, bool | None] = {}
            for horizon, bars in horizons_bars.items():
                end = i + bars
                if end > len(closes):
                    recovered[horizon] = None
                else:
                    recovered[horizon] = any(closes[j] >= peak for j in range(i, end))
            events.append(
                DipEvent(
                    asset=asset,
                    peak_date=peak_date,
                    trough_date=date,
                    depth=depth,
                    recovered=recovered,
                )
            )
            in_dip = True
        elif in_dip and depth < threshold:
            # Shallow recovery below the peak re-arms detection so a later,
            # distinct dip on the same peak is recorded as its own event.
            in_dip = False
    return events


def unconditional_baseline(events: list[DipEvent]) -> dict[str, float]:
    """Empirical recovery frequency per horizon — the mandatory baseline."""
    totals: dict[str, list[bool]] = {}
    for event in events:
        for horizon, flag in event.recovered.items():
            if flag is not None:
                totals.setdefault(horizon, []).append(flag)
    return {
        h: (sum(flags) / len(flags) if flags else float("nan"))
        for h, flags in sorted(totals.items())
    }


def _brier(p: float, y: bool) -> float:
    return (p - float(y)) ** 2


def _log_loss(p: float, y: bool) -> float:
    eps = 1e-7
    p = min(max(p, eps), 1 - eps)
    return -(float(y) * math.log(p) + (1 - float(y)) * math.log(1 - p))


def evaluate_forecasts(
    events: list[DipEvent], forecasts: list[DipForecast], *, n_bins: int = 10
) -> dict[str, float]:
    """Proper scores + calibration for a forecast set against frozen events.

    Returns Brier, log-loss, and ECE per horizon plus a headline
    ``brier_overall``; events with unobservable horizons are excluded per
    horizon, never imputed. Baseline comparison is the caller's duty — the
    bench only measures, honestly.
    """
    event_index = {(e.asset, e.trough_date): e for e in events}
    per_horizon: dict[str, list[tuple[float, bool]]] = {}
    for fc in forecasts:
        event = event_index.get((fc.asset, fc.trough_date))
        if event is None:
            continue
        for horizon, p in fc.probabilities.items():
            flag = event.recovered.get(horizon)
            if flag is None:
                continue
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"probability out of range: {p}")
            per_horizon.setdefault(horizon, []).append((p, flag))
    metrics: dict[str, float] = {}
    all_pairs: list[tuple[float, bool]] = []
    for horizon, pairs in sorted(per_horizon.items()):
        if not pairs:
            continue
        all_pairs.extend(pairs)
        metrics[f"brier_{horizon}"] = sum(_brier(p, y) for p, y in pairs) / len(pairs)
        metrics[f"log_loss_{horizon}"] = sum(_log_loss(p, y) for p, y in pairs) / len(pairs)
        metrics[f"ece_{horizon}"] = _ece(pairs, n_bins=n_bins)
        metrics[f"n_{horizon}"] = float(len(pairs))
    if all_pairs:
        metrics["brier_overall"] = sum(_brier(p, y) for p, y in all_pairs) / len(all_pairs)
    return metrics


def _ece(pairs: list[tuple[float, bool]], *, n_bins: int) -> float:
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for p, y in pairs:
        bins[min(int(p * n_bins), n_bins - 1)].append((p, y))
    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        avg_p = sum(p for p, _ in bucket) / len(bucket)
        freq = sum(float(y) for _, y in bucket) / len(bucket)
        ece += len(bucket) / len(pairs) * abs(avg_p - freq)
    return ece


def assert_bench_output_honest(metrics: dict[str, float]) -> None:
    """Fail-closed: bench output keys must not contain forbidden tokens."""
    for key in metrics:
        tokens = set(key.lower().replace("-", "_").split("_"))
        if tokens & _FORBIDDEN_TOKENS:
            raise ValueError(f"bench metric key {key!r} contains a forbidden headline token")
