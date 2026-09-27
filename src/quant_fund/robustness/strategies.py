"""Strategy interface and toy decision rules with known radii.

A strategy reads a finite array and returns a discrete decision plus a
position path. Optional ``margin_and_grad`` is the hook a differentiable
backtester implements. This package does not import JAX.

Toys are correctness checks. Their analytic radii live in ``analytic``.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from quant_fund.robustness.analytic import (
    linear_l2_radius,
    linear_margin,
    linear_positive_probability,
    linear_spike_radius,
)

FloatArray = NDArray[np.float64]


class Strategy(Protocol):
    """Decision rule evaluated on one finite path or feature window."""

    name: str

    def decision(self, sample: FloatArray) -> int:
        """Discrete decision in ``{-1, 0, 1}``."""

    def positions(self, sample: FloatArray) -> FloatArray:
        """Position held on each coordinate of ``sample``. Aligned, finite."""


class LinearMargin:
    """``sign(w · x + b)``.

    The population randomized-smoothing radius under isotropic Gaussian noise
    equals the Euclidean distance to the hyperplane. See ``analytic``.
    """

    def __init__(self, weights: FloatArray, bias: float = 0.0, name: str = "linear_margin") -> None:
        self.weights = np.asarray(weights, dtype=float).reshape(-1).copy()
        self.bias = float(bias)
        self.name = name
        if self.weights.size == 0 or not np.all(np.isfinite(self.weights)):
            raise ValueError("weights must be a non-empty finite vector")

    def decision(self, sample: FloatArray) -> int:
        margin = linear_margin(self.weights, self.bias, sample)
        if margin > 0.0:
            return 1
        if margin < 0.0:
            return -1
        return 0

    def positions(self, sample: FloatArray) -> FloatArray:
        """Hold the decision on every bar. One entry from a flat book."""
        path = np.asarray(sample, dtype=float).reshape(-1)
        if path.shape != self.weights.shape:
            raise ValueError("sample must align with weights")
        return np.full(path.shape, float(self.decision(path)))

    def margin_and_grad(self, sample: FloatArray) -> tuple[float, FloatArray]:
        """Exact margin and gradient. A JAX backtester can replace this."""
        margin = linear_margin(self.weights, self.bias, sample)
        return margin, self.weights.copy()

    def analytic_l2_radius(self, sample: FloatArray) -> float:
        return linear_l2_radius(self.weights, self.bias, sample)

    def analytic_spike_radius(self, sample: FloatArray, scale: FloatArray | None = None) -> float:
        return linear_spike_radius(self.weights, self.bias, sample, scale)

    def population_positive_probability(self, sample: FloatArray, sigma: float) -> float:
        return linear_positive_probability(self.weights, self.bias, sample, sigma)

    def decide_many(self, samples: FloatArray) -> NDArray[np.int64]:
        """Vectorized decisions. ``samples`` has shape ``(n, dimension)``."""
        batch = np.asarray(samples, dtype=float)
        if batch.ndim != 2 or batch.shape[1] != self.weights.shape[0]:
            raise ValueError("samples must have shape (n, dimension)")
        margins = batch @ self.weights + self.bias
        out = np.sign(margins).astype(np.int64)
        out[margins == 0.0] = 0
        return np.asarray(out, dtype=np.int64)


class Threshold:
    """One-dimensional threshold. A linear margin with weight 1 and bias ``-level``."""

    def __init__(self, level: float = 0.0, name: str = "threshold") -> None:
        self._inner = LinearMargin(np.asarray([1.0]), bias=-float(level), name=name)
        self.name = name
        self.level = float(level)

    def decision(self, sample: FloatArray) -> int:
        return self._inner.decision(sample)

    def positions(self, sample: FloatArray) -> FloatArray:
        return self._inner.positions(sample)

    def margin_and_grad(self, sample: FloatArray) -> tuple[float, FloatArray]:
        return self._inner.margin_and_grad(sample)

    def analytic_l2_radius(self, sample: FloatArray) -> float:
        return self._inner.analytic_l2_radius(sample)

    def analytic_spike_radius(self, sample: FloatArray, scale: FloatArray | None = None) -> float:
        return self._inner.analytic_spike_radius(sample, scale)

    def population_positive_probability(self, sample: FloatArray, sigma: float) -> float:
        return self._inner.population_positive_probability(sample, sigma)

    def decide_many(self, samples: FloatArray) -> NDArray[np.int64]:
        return self._inner.decide_many(samples)


class MeanSign:
    """``sign(sum(x))``, the linear margin with equal weights."""

    def __init__(self, dimension: int, name: str = "mean_sign") -> None:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise ValueError("dimension must be a positive integer")
        self._inner = LinearMargin(np.ones(dimension), bias=0.0, name=name)
        self.name = name
        self.dimension = dimension

    def decision(self, sample: FloatArray) -> int:
        return self._inner.decision(sample)

    def positions(self, sample: FloatArray) -> FloatArray:
        return self._inner.positions(sample)

    def margin_and_grad(self, sample: FloatArray) -> tuple[float, FloatArray]:
        return self._inner.margin_and_grad(sample)

    def analytic_l2_radius(self, sample: FloatArray) -> float:
        return self._inner.analytic_l2_radius(sample)

    def analytic_spike_radius(self, sample: FloatArray, scale: FloatArray | None = None) -> float:
        return self._inner.analytic_spike_radius(sample, scale)

    def population_positive_probability(self, sample: FloatArray, sigma: float) -> float:
        return self._inner.population_positive_probability(sample, sigma)

    def decide_many(self, samples: FloatArray) -> NDArray[np.int64]:
        return self._inner.decide_many(samples)


class FixedSchedule:
    """Positions fixed in advance. The path only affects the simulated sum.

    Used to check timing jitter. The decision is the sign of the aligned
    gross sum, so a roll that zeros that sum flips the decision. There is
    no useful gradient: the schedule does not depend on the path.
    """

    def __init__(self, schedule: FloatArray, name: str = "fixed_schedule") -> None:
        self.schedule = np.asarray(schedule, dtype=float).reshape(-1).copy()
        self.name = name
        if self.schedule.size == 0 or not np.all(np.isfinite(self.schedule)):
            raise ValueError("schedule must be a non-empty finite vector")

    def positions(self, sample: FloatArray) -> FloatArray:
        path = np.asarray(sample, dtype=float).reshape(-1)
        if path.shape != self.schedule.shape:
            raise ValueError("sample must align with the schedule")
        return self.schedule.copy()

    def decision(self, sample: FloatArray) -> int:
        gross = float(np.sum(self.positions(sample) * np.asarray(sample, dtype=float).reshape(-1)))
        if gross > 0.0:
            return 1
        if gross < 0.0:
            return -1
        return 0


class OpaqueSign:
    """``sign`` of the first coordinate, without exposing a gradient.

    The rule is the same threshold a linear margin would use. Gradient
    attacks stay unavailable until a backend is registered, which is the
    seam a differentiable backtester plugs into.
    """

    def __init__(self, dimension: int, name: str = "opaque_sign") -> None:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise ValueError("dimension must be a positive integer")
        self.dimension = dimension
        self.name = name

    def decision(self, sample: FloatArray) -> int:
        path = np.asarray(sample, dtype=float).reshape(-1)
        if path.shape != (self.dimension,):
            raise ValueError("sample dimension does not match the strategy")
        value = float(path[0])
        if value > 0.0:
            return 1
        if value < 0.0:
            return -1
        return 0

    def positions(self, sample: FloatArray) -> FloatArray:
        path = np.asarray(sample, dtype=float).reshape(-1)
        return np.full(path.shape, float(self.decision(path)))
