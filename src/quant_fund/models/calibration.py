"""Isotonic / Platt calibration fitted on train/validation only."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from quant_fund.models.base import JoblibMixin, ModelMeta


class ProbabilityCalibrator(JoblibMixin):
    def __init__(self, method: str = "isotonic") -> None:
        self.method = method
        self.iso = IsotonicRegression(out_of_bounds="clip")
        self.platt = LogisticRegression()
        self.fitted = False

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
            return s
        if self.method == "platt":
            return self.platt.predict_proba(s.reshape(-1, 1))[:, 1]
        return self.iso.predict(s)

    def metadata(self) -> ModelMeta:
        return ModelMeta(family="calibration", name=self.method, version="v1")
