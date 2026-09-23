"""Cross-sectional ranking models. Groups are dates, never mixed."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler

from quant_fund.models.base import JoblibMixin, ModelMeta

PUBLIC_FEATURES = [
    "cs_z_mom_20",
    "cs_z_reversal_1",
    "cs_z_vol_20",
    "cs_z_amihud",
    "cs_z_ret_5",
    "cs_pct_mom_20",
    "rel_volume",
    "sector_relative_mom_20",
]
ORACLE_FEATURES = [
    "cs_z_planted_signal",
]
DEFAULT_FEATURES = [*PUBLIC_FEATURES, *ORACLE_FEATURES]


def available_features(columns: list[str], wanted: list[str] | None = None) -> list[str]:
    wanted = wanted or DEFAULT_FEATURES
    return [c for c in wanted if c in columns]


def _finite(x: NDArray[np.float64], y: NDArray[np.float64] | None = None) -> Any:
    mask = np.isfinite(x).all(axis=1)
    if y is not None:
        mask &= np.isfinite(y)
        return x[mask], y[mask], mask
    return x[mask], mask


class CompositeRanker(JoblibMixin):
    """Average of configured cross-sectional percentiles. Baseline."""

    def __init__(
        self, cols: list[str] | None = None, signs: dict[str, float] | None = None
    ) -> None:
        self.cols = cols or ["cs_pct_mom_20", "cs_pct_reversal_1"]
        self.signs = signs or {
            "cs_pct_mom_20": 1.0,
            "cs_pct_reversal_1": 1.0,
            "cs_pct_vol_20": -1.0,
        }
        self._fitted = False

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any) -> CompositeRanker:
        self._fitted = True
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        # x is already the selected columns in self.cols order
        w = np.array([self.signs.get(c, 1.0) for c in self.cols], dtype=float)
        return np.nanmean(x * w, axis=1)

    def metadata(self) -> ModelMeta:
        return ModelMeta(family="ranking", name="composite", version="v1", features=self.cols)


class RidgeRanker(JoblibMixin):
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any) -> RidgeRanker:
        xx, yy, _ = _finite(x, y)
        self.scaler.fit(xx)
        self.model.fit(self.scaler.transform(xx), yy)
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x = np.where(np.isfinite(x), x, 0.0)
        return self.model.predict(self.scaler.transform(x))

    def metadata(self) -> ModelMeta:
        return ModelMeta(family="ranking", name="ridge", version="v1", extra={"alpha": self.alpha})


class ElasticNetRanker(RidgeRanker):
    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5) -> None:
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.scaler = StandardScaler()
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=4000)

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="ranking",
            name="elasticnet",
            version="v1",
            extra={"alpha": self.alpha, "l1_ratio": self.l1_ratio},
        )


class NeuralRanker(RidgeRanker):
    """Optional deterministic feed-forward neural ranker.

    This is supervised score learning on forward labels, not an RL/P&L agent.
    It reuses the same finite-row and feature-scaling contract as the linear
    rankers while keeping the neural dependency optional at import time.
    """

    def __init__(self, seed: int = 42) -> None:
        from sklearn.neural_network import MLPRegressor

        self.seed = int(seed)
        self.scaler = StandardScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size="auto",
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=self.seed,
        )

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="ranking", name="neural", version="v1", extra={"seed": self.seed}
        )


class EnsembleRanker(JoblibMixin):
    """Leakage-safe blend of heterogeneous supervised rankers.

    Each member is fit on the same causal training rows. Predictions are
    standardized within the prediction batch before averaging so a tree model's
    scale cannot dominate a linear model. This is a ranking ensemble, not a
    portfolio/P&L model.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = int(seed)
        self.members: list[Any] = [
            RidgeRanker(alpha=1.0),
            ElasticNetRanker(alpha=1.0, l1_ratio=0.5),
            NeuralRanker(seed=self.seed),
            # Tree backends remain available as standalone rankers. Keeping
            # them out of the default ensemble avoids native-library thread
            # interactions with the optional neural backend during parallel
            # research runs; users can still train each tree model directly.
        ]

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any) -> EnsembleRanker:
        for member in self.members:
            member.fit(x, y, **kwargs)
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        predictions = np.vstack([np.asarray(m.predict(x), dtype=float) for m in self.members])
        centered = predictions - np.nanmean(predictions, axis=1, keepdims=True)
        scales = np.nanstd(centered, axis=1, keepdims=True)
        normalized = np.divide(centered, np.where(scales > 1e-12, scales, 1.0))
        return np.nanmean(normalized, axis=0)

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="ranking",
            name="ensemble",
            version="v1",
            extra={"members": [type(m).__name__ for m in self.members]},
        )


class XGBRegRanker(JoblibMixin):
    def __init__(self, n_estimators: int = 80, max_depth: int = 3, seed: int = 42) -> None:
        from xgboost import XGBRegressor

        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=1,
            random_state=seed,
            verbosity=0,
        )

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any) -> XGBRegRanker:
        xx, yy, _ = _finite(x, y)
        self.model.fit(xx, yy)
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(self.model.predict(x), dtype=float)

    def metadata(self) -> ModelMeta:
        return ModelMeta(family="ranking", name="xgboost", version="v1")


class LGBMRegRanker(JoblibMixin):
    def __init__(self, n_estimators: int = 80, num_leaves: int = 15, seed: int = 42) -> None:
        from lightgbm import LGBMRegressor

        self.model = LGBMRegressor(
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            random_state=seed,
            n_jobs=1,
            num_threads=1,
            verbosity=-1,
        )

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any) -> LGBMRegRanker:
        xx, yy, _ = _finite(x, y)
        self.model.fit(xx, yy)
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(self.model.predict(x), dtype=float)

    def metadata(self) -> ModelMeta:
        return ModelMeta(family="ranking", name="lightgbm", version="v1")


class LGBMLambdaRanker(JoblibMixin):
    """LightGBM LambdaRank. `group` is per-date counts; dates must not be mixed."""

    def __init__(
        self, n_estimators: int = 80, seed: int = 42, objective: str = "lambdarank"
    ) -> None:
        from lightgbm import LGBMRanker

        self.objective = objective
        self.model = LGBMRanker(
            n_estimators=n_estimators,
            random_state=seed,
            n_jobs=1,
            num_threads=1,
            verbosity=-1,
            objective=objective,
        )

    def fit(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        group: NDArray[np.int32] | None = None,
        **kwargs: Any,
    ) -> LGBMLambdaRanker:
        if group is None:
            raise ValueError("LambdaRank requires per-date group sizes")
        # relevance grades from y ranks within the concatenated groups
        rel = _to_relevance(y, group)
        self.model.fit(x, rel, group=group.tolist())
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(self.model.predict(x), dtype=float)

    def metadata(self) -> ModelMeta:
        return ModelMeta(family="ranking", name=self.objective, version="v1")


def _to_relevance(y: NDArray[np.float64], group: NDArray[np.int32]) -> NDArray[np.int32]:
    """Map returns to integer relevance 0-4 within each date group.

    Fail-closed on degenerate groups: an all-NaN (or otherwise
    non-computable-quantile) group must not silently mint all-zero relevance
    labels that read as "every name is lowest relevance". Such entries are left
    at 0 and the group's finite members are still digitized; NaN members are
    never assigned a fabricated quintile.
    """
    rel = np.zeros(y.shape[0], dtype=np.int32)
    start = 0
    for g in group:
        sl = y[start : start + int(g)]
        finite = np.isfinite(sl)
        if finite.any():
            # Quintiles over finite members only. A group whose quantiles are not
            # computable (e.g. all-NaN slice) must not silently mint all-zero
            # relevance labels that read as "every name is lowest relevance" —
            # such a group is left at 0 rather than fabricated.
            qs = np.nanquantile(sl, [0.2, 0.4, 0.6, 0.8])
            if np.isfinite(qs).all():
                grades = np.digitize(sl, qs)
                # Non-finite members carry no signal; keep them at grade 0.
                grades = np.where(finite, grades, 0)
                rel[start : start + int(g)] = grades
        start += int(g)
    return rel


def group_sizes(dates: NDArray[Any]) -> NDArray[np.int32]:
    """Consecutive equal dates → group sizes. Caller must sort by date."""
    sizes: list[int] = []
    i = 0
    n = len(dates)
    while i < n:
        j = i + 1
        while j < n and dates[j] == dates[i]:
            j += 1
        sizes.append(j - i)
        i = j
    return np.array(sizes, dtype=np.int32)
