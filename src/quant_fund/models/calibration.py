"""Isotonic / Platt calibration fitted on train/validation only."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from quant_fund.models.base import JoblibMixin, ModelMeta


class ProbabilityCalibrator(JoblibMixin):
    _METHODS = ("isotonic", "platt")

    def __init__(self, method: str = "isotonic") -> None:
        if method not in self._METHODS:
            raise ValueError(f"calibration method must be one of {self._METHODS}, got {method!r}")
        self.method = method
        self.iso = IsotonicRegression(out_of_bounds="clip")
        self.platt = LogisticRegression()
        self.fitted = False
        self.score_feature: str | None = None
        self.label: str | None = None
        self.horizon: str | None = None
        self.fit_start: str | None = None
        self.fit_end: str | None = None
        self.oos_start: str | None = None
        self.oos_end: str | None = None

    def fit(
        self, scores: NDArray[np.float64], labels: NDArray[np.float64]
    ) -> ProbabilityCalibrator:
        s = np.asarray(scores, dtype=float)
        y = np.asarray(labels, dtype=float)
        mask = np.isfinite(s) & np.isfinite(y)
        s, y = s[mask], y[mask]
        if s.size < 10:
            return self
        if self.method == "platt":
            self.platt.fit(s.reshape(-1, 1), (y > 0.5).astype(int))
        else:
            self.iso.fit(s, y)
        self.fitted = True
        return self

    def predict(self, scores: NDArray[np.float64]) -> NDArray[np.float64]:
        s = np.asarray(scores, dtype=float)
        if not self.fitted:
            raise RuntimeError(
                "ProbabilityCalibrator is not fitted: returning raw scores as "
                "'probabilities' would masquerade an uncalibrated identity as a "
                "calibrated forecast. Call fit() with >=10 finite rows first."
            )
        if self.method == "platt":
            return np.asarray(self.platt.predict_proba(s.reshape(-1, 1))[:, 1], dtype=np.float64)
        return np.asarray(self.iso.predict(s), dtype=np.float64)

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="calibration",
            name=self.method,
            version="v1",
            extra={
                "score_feature": self.score_feature,
                "label": self.label,
                "horizon": getattr(self, "horizon", None),
                "fit_start": self.fit_start,
                "fit_end": self.fit_end,
                "oos_start": self.oos_start,
                "oos_end": self.oos_end,
            },
        )
