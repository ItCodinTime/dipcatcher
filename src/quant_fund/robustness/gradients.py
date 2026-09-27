"""Gradient interface for attacks.

A differentiable backtester — including one written in JAX — implements
:class:`GradientBackend` and registers it. This module never imports JAX.
Finite differences are opt-in and are labeled numerical, not exact.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class GradientBackend(Protocol):
    """Value and gradient of a scalar margin with respect to a path.

    ``margin`` is signed: positive means the long decision. The gradient is
    aligned with ``path``. A JAX fill simulator can implement this by
    differentiating through its own code and converting the gradient to a
    NumPy array at this boundary.
    """

    name: str

    def value_and_grad(self, path: FloatArray) -> tuple[float, FloatArray]:
        """Return ``(margin, d(margin) / d(path))``."""


_BACKENDS: dict[str, GradientBackend] = {}


def register_gradient_backend(backend: GradientBackend) -> None:
    """Register ``backend`` under ``backend.name``, replacing a previous one."""
    if not getattr(backend, "name", ""):
        raise ValueError("gradient backend needs a non-empty name")
    _BACKENDS[str(backend.name)] = backend


def unregister_gradient_backend(name: str) -> None:
    """Drop a registered backend. Missing names are ignored."""
    _BACKENDS.pop(name, None)


def get_gradient_backend(name: str) -> GradientBackend:
    """Return a registered backend or raise ``KeyError``."""
    if name not in _BACKENDS:
        known = ", ".join(sorted(_BACKENDS)) or "(none)"
        raise KeyError(f"unknown gradient backend {name!r}; registered: {known}")
    return _BACKENDS[name]


def registered_backend_names() -> tuple[str, ...]:
    """Names currently registered. Order is sorted for stability."""
    return tuple(sorted(_BACKENDS))


class FiniteDifferenceBackend:
    """Central differences. Empirical numerical gradient, not a proof."""

    name = "finite_difference"

    def __init__(self, margin_fn: Callable[[FloatArray], float], step: float = 1e-6) -> None:
        if not callable(margin_fn):
            raise TypeError("margin_fn must be callable")
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("step must be finite and positive")
        self._margin_fn = margin_fn
        self.step = float(step)

    def value_and_grad(self, path: FloatArray) -> tuple[float, FloatArray]:
        base = np.asarray(path, dtype=float).reshape(-1)
        if base.size == 0 or not np.all(np.isfinite(base)):
            raise ValueError("path must be a non-empty finite vector")
        margin = float(self._margin_fn(base))
        gradient = np.empty_like(base)
        for index in range(base.size):
            up = base.copy()
            down = base.copy()
            up[index] += self.step
            down[index] -= self.step
            gradient[index] = (float(self._margin_fn(up)) - float(self._margin_fn(down))) / (
                2.0 * self.step
            )
        return margin, gradient


class CallableBackend:
    """Adapter around a ``(path) -> (margin, gradient)`` callable.

    Use this to wrap a JAX backtester without importing JAX here::

        register_gradient_backend(CallableBackend("jax_backtester", jax_value_and_grad))
    """

    def __init__(
        self,
        name: str,
        fn: Callable[[FloatArray], tuple[float, FloatArray]],
    ) -> None:
        if not name or not isinstance(name, str):
            raise ValueError("backend name must be a non-empty string")
        if not callable(fn):
            raise TypeError("fn must be callable")
        self.name = name
        self._fn = fn

    def value_and_grad(self, path: FloatArray) -> tuple[float, FloatArray]:
        margin, gradient = self._fn(np.asarray(path, dtype=float))
        grad = np.asarray(gradient, dtype=float).reshape(-1)
        path_arr = np.asarray(path, dtype=float).reshape(-1)
        if grad.shape != path_arr.shape or not np.all(np.isfinite(grad)):
            raise ValueError("backend gradient must be finite and aligned with the path")
        margin_f = float(margin)
        if not np.isfinite(margin_f):
            raise ValueError("backend margin must be finite")
        return margin_f, grad
