"""Opt-in wrappers around stable pipeline entry points.

``install_passive_hooks`` does nothing unless ``DIPCATCHER_OBSERVE`` is set,
and it does not import the pipeline modules in that case. Production modules
are not edited. Wrappers call the original function and re-raise its
exceptions unchanged.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from quant_fund.observe.flags import metrics_host, metrics_port, observe_enabled
from quant_fund.observe.tracing import span

_UNDO: list[tuple[Any, str, Any]] = []
_INSTALLED = False


def wrap_callable(fn: Callable[..., Any], stage: str) -> Callable[..., Any]:
    """Return ``fn`` unchanged when observability is off."""
    if not observe_enabled():
        return fn

    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with span(stage):
            return fn(*args, **kwargs)

    wrapped.__audit_wrapped__ = True  # type: ignore[attr-defined]
    return wrapped


def install_passive_hooks() -> int:
    """Wrap ingest, features, model, decision, and simulated execution.

    Returns the number of callables wrapped. A second call is a no-op.
    """
    global _INSTALLED
    if not observe_enabled():
        return 0
    if _INSTALLED:
        return 0
    targets = _load_targets()
    for owner, name, original, stage in targets:
        current = getattr(owner, name)
        if getattr(current, "__audit_wrapped__", False):
            continue
        setattr(owner, name, wrap_callable(original, stage))
        _UNDO.append((owner, name, original))
    _INSTALLED = True
    port = metrics_port()
    if port is not None:
        from quant_fund.observe.serve import start_metrics_server

        start_metrics_server(port, host=metrics_host())
    return len(_UNDO)


def uninstall_passive_hooks() -> None:
    """Restore every callable this process wrapped."""
    global _INSTALLED
    while _UNDO:
        owner, name, original = _UNDO.pop()
        setattr(owner, name, original)
    _INSTALLED = False


def _load_targets() -> list[tuple[Any, str, Any, str]]:
    import importlib

    # ``quant_fund.data.ingest`` is shadowed by the function exported from
    # ``quant_fund.data``, so import the module object explicitly.
    ingest = importlib.import_module("quant_fund.data.ingest")
    data_pkg = importlib.import_module("quant_fund.data")
    features = importlib.import_module("quant_fund.features.engine")
    features_pkg = importlib.import_module("quant_fund.features")
    forecast = importlib.import_module("quant_fund.pipeline.forecast")
    dataset = importlib.import_module("quant_fund.pipeline.dataset")
    risk_gate = importlib.import_module("quant_fund.portfolio.risk_gate")
    broker = importlib.import_module("quant_fund.execution.simulated_broker")
    # These modules bind the originals at import. Rebind those names too, or a
    # span on the defining module would not cover an already-imported caller.
    return [
        (ingest, "ingest", ingest.ingest, "ingest"),
        (data_pkg, "ingest", data_pkg.ingest, "ingest"),
        (dataset, "ingest", dataset.ingest, "ingest"),
        (features, "build_features", features.build_features, "features"),
        (features_pkg, "build_features", features_pkg.build_features, "features"),
        (dataset, "build_features", dataset.build_features, "features"),
        (forecast, "forecast_asof", forecast.forecast_asof, "model"),
        (risk_gate, "check_order", risk_gate.check_order, "decision"),
        (broker, "check_order", broker.check_order, "decision"),
        (broker.SimulatedBroker, "submit", broker.SimulatedBroker.submit, "simulated_execution"),
    ]
