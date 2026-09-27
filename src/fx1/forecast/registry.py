"""Name-keyed forecaster registry.

Built-ins are reference models for tests and smoke runs (``dummy-zero``,
``dummy-momentum``). The name ``fx-1`` resolves only through an explicit
``module:attr`` entrypoint so the model cannot be silently stubbed.
"""

from __future__ import annotations

import importlib
from typing import Any

from fx1.forecast.protocol import ForecastModel, Fx1Model


class ModelNotRegistered(LookupError):
    """No forecaster is bound to this name."""


def load_symbol(spec: str) -> Any:
    """Import ``module:attr``. The attribute may be a class or a factory."""
    if ":" not in spec:
        raise ValueError("entrypoint must be 'module:attr'")
    module_name, attr = spec.split(":", 1)
    if not module_name or not attr:
        raise ValueError("entrypoint must be 'module:attr'")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ModelNotRegistered(f"{spec} does not resolve") from exc


def _builtin(name: str) -> ForecastModel:
    # Imported lazily so registry import does not require dummy construction.
    from fx1.forecast.dummy import MomentumForecastModel, ZeroForecastModel

    factories: dict[str, type[ForecastModel]] = {
        "dummy-zero": ZeroForecastModel,
        "dummy-momentum": MomentumForecastModel,
    }
    cls = factories.get(name)
    if cls is None:
        raise ModelNotRegistered(
            f"unknown forecaster {name!r}. Built-ins are {sorted(factories)}; "
            "fx-1 must be supplied via model.entrypoint."
        )
    return cls()


def create_model(name: str, *, entrypoint: str | None = None) -> ForecastModel:
    """Return a forecaster instance.

    ``entrypoint`` wins over ``name``. ``fx-1`` / ``fx1`` without an
    entrypoint raises :class:`ModelNotRegistered`.
    """
    key = name.strip().lower()
    if entrypoint:
        obj = load_symbol(entrypoint)
        if isinstance(obj, type):
            obj = obj()
        if not isinstance(obj, ForecastModel):
            raise TypeError(
                f"entrypoint {entrypoint!r} must be a ForecastModel instance or subclass"
            )
        if key in {"fx-1", "fx1"} and not isinstance(obj, Fx1Model):
            raise TypeError("the fx-1 entrypoint must subclass fx1.forecast.Fx1Model")
        return obj
    if key in {"fx-1", "fx1"}:
        raise ModelNotRegistered(
            "fx-1 is an external forecasting model and is not implemented in this "
            "repository. Set model.entrypoint to a module:Class that subclasses Fx1Model."
        )
    return _builtin(key)
