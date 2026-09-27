"""Streaming moments and quantiles with a defined merge.

Welford / Chan moments merge by the parallel formula. The t-digest merge
concatenates centroids and compresses. Neither operation is associative in
floating point, so the engine always folds chunks in chunk-id order. That
order does not depend on which worker finished first.

P² (Jain and Chlamtac, 1985) is a single-stream quantile sketch. It has no
merge. The engine applies it only by streaming retained losses in path order.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
WelfordState = tuple[int, float, float]


class Welford:
    """One-pass mean and second moment (Welford; Chan et al. parallel merge)."""

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, value: float) -> None:
        x = float(value)
        if not math.isfinite(x):
            raise ValueError("Welford.add requires a finite value")
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    def add_all(self, values: FloatArray) -> None:
        arr = np.asarray(values, dtype=np.float64).ravel()
        if arr.size == 0:
            return
        if not np.isfinite(arr).all():
            raise ValueError("Welford.add_all requires finite values")
        # Sequential updates keep the chunk digest identical to a one-path stream.
        for value in arr.tolist():
            self.add(float(value))

    def state(self) -> WelfordState:
        return (self.n, self.mean, self.m2)

    @property
    def variance(self) -> float | None:
        """Unbiased sample variance. ``None`` until two observations exist."""
        if self.n < 2:
            return None
        return self.m2 / (self.n - 1)


def merge_welford(left: WelfordState, right: WelfordState) -> WelfordState:
    """Chan merge. Fold left-to-right in chunk-id order for a stable result."""
    n_left, mean_left, m2_left = left
    n_right, mean_right, m2_right = right
    if n_left < 0 or n_right < 0:
        raise ValueError("Welford counts must be non-negative")
    if n_left == 0:
        return right
    if n_right == 0:
        return left
    n_total = n_left + n_right
    delta = mean_right - mean_left
    mean = mean_left + delta * (n_right / n_total)
    m2 = m2_left + m2_right + delta * delta * n_left * n_right / n_total
    return (n_total, float(mean), float(m2))


def welford_variance(state: WelfordState) -> float | None:
    n, _, m2 = state
    if n < 2:
        return None
    return float(m2 / (n - 1))


def compress_centroids(
    means: FloatArray,
    weights: FloatArray,
    compression: float,
) -> tuple[FloatArray, FloatArray]:
    """Merge sorted weighted points with the t-digest size bound.

    A centroid near quantile ``q`` is allowed weight about
    ``4 n q (1-q) / compression`` (Dunning's scale), and at least 1.
    Total weight is preserved.
    """
    if compression < 10.0:
        raise ValueError("compression must be >= 10")
    mean_arr = np.asarray(means, dtype=np.float64).ravel()
    weight_arr = np.asarray(weights, dtype=np.float64).ravel()
    if mean_arr.shape != weight_arr.shape:
        raise ValueError("means and weights must have the same shape")
    if mean_arr.size == 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty
    if not np.isfinite(mean_arr).all() or not np.isfinite(weight_arr).all():
        raise ValueError("centroids must be finite")
    if np.any(weight_arr < 0.0):
        raise ValueError("centroid weights must be non-negative")
    order = np.argsort(mean_arr, kind="mergesort")
    mean_arr = mean_arr[order]
    weight_arr = weight_arr[order]
    total = float(weight_arr.sum())
    if total <= 0.0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty
    out_means: list[float] = []
    out_weights: list[float] = []
    acc_sum = 0.0
    acc_weight = 0.0
    consumed = 0.0
    for mean, weight in zip(mean_arr.tolist(), weight_arr.tolist(), strict=True):
        if weight == 0.0:
            continue
        q = (consumed + 0.5 * weight) / total
        q = min(1.0, max(0.0, q))
        limit = max(1.0, 4.0 * total * q * (1.0 - q) / compression)
        if acc_weight > 0.0 and acc_weight + weight > limit:
            out_means.append(acc_sum / acc_weight)
            out_weights.append(acc_weight)
            acc_sum = mean * weight
            acc_weight = weight
        else:
            acc_sum += mean * weight
            acc_weight += weight
        consumed += weight
    if acc_weight > 0.0:
        out_means.append(acc_sum / acc_weight)
        out_weights.append(acc_weight)
    return (
        np.asarray(out_means, dtype=np.float64),
        np.asarray(out_weights, dtype=np.float64),
    )


class TDigest:
    """Merging centroid digest. Quantiles are the centroid step function.

    The step-function query is deliberate: interpolation would be another
    estimator, and the report labels digest quantiles as approximate either way.
    """

    def __init__(
        self,
        compression: float = 100.0,
        means: FloatArray | None = None,
        weights: FloatArray | None = None,
    ) -> None:
        if compression < 10.0:
            raise ValueError("compression must be >= 10")
        self.compression = float(compression)
        if means is None and weights is None:
            self.means = np.zeros(0, dtype=np.float64)
            self.weights = np.zeros(0, dtype=np.float64)
        elif means is None or weights is None:
            raise ValueError("means and weights must be supplied together")
        else:
            self.means = np.asarray(means, dtype=np.float64).ravel()
            self.weights = np.asarray(weights, dtype=np.float64).ravel()
            if self.means.shape != self.weights.shape:
                raise ValueError("means and weights must have the same shape")

    def add_all(self, values: FloatArray) -> None:
        arr = np.asarray(values, dtype=np.float64).ravel()
        if arr.size == 0:
            return
        if not np.isfinite(arr).all():
            raise ValueError("t-digest values must be finite")
        ones = np.ones(arr.size, dtype=np.float64)
        means = np.concatenate([self.means, arr])
        weights = np.concatenate([self.weights, ones])
        self.means, self.weights = compress_centroids(means, weights, self.compression)

    def merge(self, other: TDigest) -> TDigest:
        if self.compression != other.compression:
            raise ValueError("t-digest compression does not match")
        means = np.concatenate([self.means, other.means])
        weights = np.concatenate([self.weights, other.weights])
        merged_means, merged_weights = compress_centroids(means, weights, self.compression)
        return TDigest(self.compression, merged_means, merged_weights)

    @property
    def total_weight(self) -> float:
        if self.weights.size == 0:
            return 0.0
        return float(self.weights.sum())

    def quantile(self, probability: float) -> float:
        p = float(probability)
        if not math.isfinite(p) or not 0.0 <= p <= 1.0:
            raise ValueError("probability must be in [0, 1]")
        if self.weights.size == 0:
            raise ValueError("t-digest is empty")
        cumulative = np.cumsum(self.weights)
        target = p * float(cumulative[-1])
        index = int(np.searchsorted(cumulative, target, side="left"))
        index = min(index, int(self.means.size) - 1)
        return float(self.means[index])

    def expected_shortfall(self, alpha: float) -> float:
        """Spectral ES of the discrete centroid distribution. Approximate."""
        a = float(alpha)
        if not math.isfinite(a) or not 0.0 < a < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if self.weights.size == 0:
            raise ValueError("t-digest is empty")
        total = float(self.weights.sum())
        hi = np.cumsum(self.weights) / total
        lo = hi - self.weights / total
        overlap = np.clip(np.minimum(hi, 1.0) - np.maximum(lo, a), 0.0, None)
        mass = float(overlap.sum())
        if mass <= 0.0:
            return float(self.means[-1])
        return float(np.dot(self.means, overlap) / mass)


class P2Quantile:
    """Jain-Chlamtac P² single-stream quantile estimator.

    Five markers track the requested probability. The estimate is undefined
    until five observations have been seen.
    """

    def __init__(self, probability: float) -> None:
        p = float(probability)
        if not math.isfinite(p) or not 0.0 < p < 1.0:
            raise ValueError("probability must be in (0, 1)")
        self.probability = p
        self.count = 0
        self._initial: list[float] = []
        self._height = [0.0, 0.0, 0.0, 0.0, 0.0]
        self._position = [0, 0, 0, 0, 0]
        self._desired = [0.0, 0.0, 0.0, 0.0, 0.0]
        self._increment = [0.0, p / 2.0, p, (1.0 + p) / 2.0, 1.0]

    def add(self, value: float) -> None:
        x = float(value)
        if not math.isfinite(x):
            raise ValueError("P2Quantile.add requires a finite value")
        if self.count < 5:
            self._initial.append(x)
            self.count += 1
            if self.count == 5:
                ordered = sorted(self._initial)
                self._height = ordered
                self._position = [1, 2, 3, 4, 5]
                self._desired = [1.0 + 4.0 * inc for inc in self._increment]
            return
        self.count += 1
        heights = self._height
        if x < heights[0]:
            heights[0] = x
            cell = 0
        elif x >= heights[4]:
            heights[4] = x
            cell = 3
        else:
            cell = 0
            for marker in range(1, 5):
                if x < heights[marker]:
                    cell = marker - 1
                    break
        for marker in range(cell + 1, 5):
            self._position[marker] += 1
        for marker in range(5):
            self._desired[marker] += self._increment[marker]
        for marker in range(1, 4):
            delta = self._desired[marker] - self._position[marker]
            if (delta >= 1.0 and self._position[marker + 1] - self._position[marker] > 1) or (
                delta <= -1.0 and self._position[marker - 1] - self._position[marker] < -1
            ):
                step = 1 if delta > 0.0 else -1
                proposed = self._parabolic(marker, step)
                if heights[marker - 1] < proposed < heights[marker + 1]:
                    heights[marker] = proposed
                else:
                    heights[marker] = self._linear(marker, step)
                self._position[marker] += step

    def add_all(self, values: FloatArray) -> None:
        arr = np.asarray(values, dtype=np.float64).ravel()
        for value in arr.tolist():
            self.add(float(value))

    def _parabolic(self, marker: int, step: int) -> float:
        height = self._height
        position = self._position
        left = position[marker] - position[marker - 1]
        right = position[marker + 1] - position[marker]
        span = position[marker + 1] - position[marker - 1]
        return height[marker] + (step / span) * (
            (left + step) * (height[marker + 1] - height[marker]) / right
            + (right - step) * (height[marker] - height[marker - 1]) / left
        )

    def _linear(self, marker: int, step: int) -> float:
        height = self._height
        position = self._position
        return height[marker] + step * (
            (height[marker + step] - height[marker]) / (position[marker + step] - position[marker])
        )

    @property
    def value(self) -> float:
        if self.count < 5:
            raise ValueError("P² quantile needs at least 5 observations")
        return float(self._height[2])
