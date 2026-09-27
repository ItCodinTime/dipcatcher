"""In-process Prometheus metrics. No client library and no work when disabled.

Gauges that describe simulation P&L or exposure are labeled as simulation in
their help text. They are operational telemetry, not research headlines, and
they are not a live P&L claim.
"""

from __future__ import annotations

import math
import threading
from typing import Any

from quant_fund.observe.flags import STAGES, observe_enabled

BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
_ERROR_STAGES = frozenset((*STAGES, "otel_export"))

METRIC_HELP = {
    "dipcatcher_stage_latency_seconds": "Latency of an instrumented stage, in seconds.",
    "dipcatcher_errors_total": "Errors observed in an instrumented stage.",
    "dipcatcher_data_freshness_seconds": "Age of the newest input used by a stage, in seconds.",
    "dipcatcher_simulation_pnl": (
        "Simulated mark-to-market result. Not a live P&L claim and not a research score."
    ),
    "dipcatcher_simulation_gross_exposure": (
        "Simulated gross exposure. Not a live trading position and not a research score."
    ),
    "dipcatcher_simulation_net_exposure": (
        "Simulated net exposure. Not a live trading position and not a research score."
    ),
}


class _Histogram:
    def __init__(self) -> None:
        self.bucket_counts = [0 for _ in range(len(BUCKETS) + 1)]
        self.total = 0.0
        self.n = 0

    def observe(self, value: float) -> None:
        if not math.isfinite(value) or value < 0:
            return
        self.total += value
        self.n += 1
        for index, bound in enumerate(BUCKETS):
            if value <= bound:
                self.bucket_counts[index] += 1
                return
        self.bucket_counts[-1] += 1


class _Registry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.histograms: dict[str, _Histogram] = {}
        self.errors: dict[str, int] = {}
        self.gauges: dict[str, float] = {}

    def observe(self, stage: str, seconds: float) -> None:
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage}")
        with self._lock:
            histogram = self.histograms.get(stage)
            if histogram is None:
                histogram = _Histogram()
                self.histograms[stage] = histogram
            histogram.observe(seconds)

    def error(self, stage: str) -> None:
        if stage not in _ERROR_STAGES:
            raise ValueError(f"unknown stage {stage}")
        with self._lock:
            self.errors[stage] = self.errors.get(stage, 0) + 1

    def gauge(self, name: str, value: float) -> None:
        if name not in METRIC_HELP or not math.isfinite(value):
            raise ValueError(f"cannot record gauge {name}")
        with self._lock:
            self.gauges[name] = float(value)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            histograms = {
                stage: {
                    "bucket_counts": list(histogram.bucket_counts),
                    "total": histogram.total,
                    "n": histogram.n,
                }
                for stage, histogram in self.histograms.items()
            }
            return {
                "histograms": histograms,
                "errors": dict(self.errors),
                "gauges": dict(self.gauges),
            }

    def reset(self) -> None:
        with self._lock:
            self.histograms.clear()
            self.errors.clear()
            self.gauges.clear()


REGISTRY = _Registry()


def reset_metrics() -> None:
    REGISTRY.reset()


def record_latency(stage: str, seconds: float) -> None:
    if not observe_enabled():
        return
    REGISTRY.observe(stage, seconds)


def record_error(stage: str) -> None:
    if not observe_enabled():
        return
    REGISTRY.error(stage)


def set_data_freshness(seconds: float) -> None:
    if not observe_enabled():
        return
    REGISTRY.gauge("dipcatcher_data_freshness_seconds", seconds)


def set_simulation_state(*, pnl: float, gross_exposure: float, net_exposure: float) -> None:
    """Record simulation gauges. The values are not research results."""
    if not observe_enabled():
        return
    REGISTRY.gauge("dipcatcher_simulation_pnl", pnl)
    REGISTRY.gauge("dipcatcher_simulation_gross_exposure", gross_exposure)
    REGISTRY.gauge("dipcatcher_simulation_net_exposure", net_exposure)


def histogram_quantile_upper(stage: str, quantile: float) -> float:
    """Bucket upper bound covering ``quantile``. ``nan`` when the stage has no samples.

    This is the histogram bucket bound, not an interpolated quantile.
    """
    snap = REGISTRY.snapshot()["histograms"].get(stage)
    if not snap or snap["n"] == 0:
        return math.nan
    target = quantile * snap["n"]
    cumulative = 0
    for index, bound in enumerate(BUCKETS):
        cumulative += snap["bucket_counts"][index]
        if cumulative >= target:
            return bound
    return math.inf


def render_prometheus() -> str:
    """Prometheus text exposition 0.0.4 for the in-process registry."""
    snap = REGISTRY.snapshot()
    lines: list[str] = []
    if snap["histograms"]:
        _help(lines, "dipcatcher_stage_latency_seconds", "histogram")
    for stage in STAGES:
        histogram = snap["histograms"].get(stage)
        if histogram is None:
            continue
        cumulative = 0
        for index, bound in enumerate(BUCKETS):
            cumulative += histogram["bucket_counts"][index]
            lines.append(
                f'dipcatcher_stage_latency_seconds_bucket{{stage="{stage}",le="{_num(bound)}"}} {cumulative}'
            )
        cumulative += histogram["bucket_counts"][-1]
        lines.append(
            f'dipcatcher_stage_latency_seconds_bucket{{stage="{stage}",le="+Inf"}} {cumulative}'
        )
        lines.append(
            f'dipcatcher_stage_latency_seconds_sum{{stage="{stage}"}} {_num(histogram["total"])}'
        )
        lines.append(f'dipcatcher_stage_latency_seconds_count{{stage="{stage}"}} {histogram["n"]}')
    if snap["errors"]:
        _help(lines, "dipcatcher_errors_total", "counter")
    for stage in (*STAGES, "otel_export"):
        if stage in snap["errors"]:
            lines.append(f'dipcatcher_errors_total{{stage="{stage}"}} {snap["errors"][stage]}')
    for name in (
        "dipcatcher_data_freshness_seconds",
        "dipcatcher_simulation_pnl",
        "dipcatcher_simulation_gross_exposure",
        "dipcatcher_simulation_net_exposure",
    ):
        if name in snap["gauges"]:
            _help(lines, name, "gauge")
            lines.append(f"{name} {_num(snap['gauges'][name])}")
    text = "\n".join(lines)
    if text:
        text += "\n"
    return text


def _help(lines: list[str], name: str, metric_type: str) -> None:
    lines.append(f"# HELP {name} {METRIC_HELP[name]}")
    lines.append(f"# TYPE {name} {metric_type}")


def _num(value: float) -> str:
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    if value == 0:
        return "0"
    return format(value, ".12g")
