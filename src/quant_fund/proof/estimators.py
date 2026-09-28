"""Reference estimators for the proven runner (WAVE2.md §4.3).

Two TINY deterministic numpy estimators. No randomness anywhere: same inputs
always produce byte-identical outputs and state dumps.

Estimator protocol (duck-typed, consumed by ``proof.runner.run_proven``):

- ``fit(X: np.ndarray, y: np.ndarray) -> None`` — ``X`` shape ``(n, k)``,
  ``y`` shape ``(n,)``, both float64; called once per window with the full
  expanding walk-forward training set;
- ``predict(x: np.ndarray) -> float`` — ``x`` shape ``(k,)``;
- ``state_vector() -> np.ndarray`` — fixed-layout float64 state;
- ``state_bytes() -> bytes`` — ``np.save`` (npy format, no timestamps, no
  pickle) of ``state_vector()`` into an in-memory buffer. npy bytes are a
  pure function of (dtype, shape, values), so identical state yields
  byte-identical dumps on a given numpy version. This is the documented
  determinism mechanism for ``DecisionTraceRow.estimator_state_sha256``.

Cold-start (declared, deterministic): an estimator that has never been fit
predicts exactly ``0.0``.
"""

from __future__ import annotations

import io

import numpy as np

__all__ = [
    "ALLOWLISTED_ESTIMATORS",
    "ESTIMATOR_REGISTRY",
    "EwmaSignal",
    "LinearRegressionNumpy",
]

#: The v1 estimator allowlist (WAVE2.md §4.3). Module-level frozenset so the
#: AST linter and future gates can see it statically. Anything outside this
#: set fails closed in the runner with ProofError("estimator_not_allowlisted").
ALLOWLISTED_ESTIMATORS: frozenset[str] = frozenset({"linear_regression_np", "ewma_signal"})


def _state_bytes_from_vector(vector: np.ndarray) -> bytes:
    """Canonical state dump: npy v1 bytes of a float64 C-contiguous vector."""
    array = np.ascontiguousarray(vector, dtype=np.float64)
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


class LinearRegressionNumpy:
    """Ridge regression, closed form via ``np.linalg.solve`` on (X'X + alpha*I).

    Pure numpy, no randomness, no iteration state. An intercept is fit by
    prepending a column of ones; the ridge penalty is applied to every
    coefficient including the intercept (documented v1 simplification).
    """

    def __init__(self, *, alpha: float = 1e-8) -> None:
        self.alpha = float(alpha)
        self._coef: np.ndarray | None = None  # shape (k + 1,), intercept first

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0] or X.shape[0] == 0:
            raise ValueError("fit requires X (n,k), y (n,) with n >= 1")
        design = np.column_stack([np.ones(X.shape[0]), X])
        gram = design.T @ design
        ridge = self.alpha * np.eye(gram.shape[0])
        self._coef = np.linalg.solve(gram + ridge, design.T @ y)

    def predict(self, x: np.ndarray) -> float:
        if self._coef is None:
            return 0.0  # declared cold-start: unfitted estimator predicts 0.0
        x = np.asarray(x, dtype=np.float64)
        return float(self._coef @ np.concatenate([[1.0], x]))

    def state_vector(self) -> np.ndarray:
        if self._coef is None:
            return np.array([0.0, self.alpha], dtype=np.float64)
        return np.concatenate([[1.0, self.alpha], self._coef])

    def state_bytes(self) -> bytes:
        return _state_bytes_from_vector(self.state_vector())


class EwmaSignal:
    """Pure-float EWMA over the signal series (the realized label series).

    ``fit`` folds the training labels into the EWMA in order with
    ``alpha = 2 / (span + 1)``; ``predict`` returns the current EWMA level
    (feature vector ignored — documented v1 reference behavior).
    """

    def __init__(self, *, span: float = 3.0) -> None:
        if not float(span) >= 1.0:
            raise ValueError("ewma span must be >= 1")
        self.span = float(span)
        self.alpha = 2.0 / (self.span + 1.0)
        self._value = 0.0
        self._count = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 1 or y.shape[0] == 0:
            raise ValueError("fit requires y (n,) with n >= 1")
        for value in y:
            self._value = self.alpha * float(value) + (1.0 - self.alpha) * self._value
        self._count += int(y.shape[0])

    def predict(self, x: np.ndarray) -> float:
        if self._count == 0:
            return 0.0  # declared cold-start
        return float(self._value)

    def state_vector(self) -> np.ndarray:
        return np.array([float(self._count), self.span, self._value], dtype=np.float64)

    def state_bytes(self) -> bytes:
        return _state_bytes_from_vector(self.state_vector())


#: Allowlist name -> estimator class. Membership in ALLOWLISTED_ESTIMATORS is
#: the gate; this registry is the constructor lookup.
ESTIMATOR_REGISTRY: dict[str, type] = {
    "linear_regression_np": LinearRegressionNumpy,
    "ewma_signal": EwmaSignal,
}
