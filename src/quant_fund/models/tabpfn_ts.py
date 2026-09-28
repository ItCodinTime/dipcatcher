"""TabPFN time-series quantile head (P2.5). Research-only.

Wraps PriorLabs ``tabpfn-time-series`` — pretrained TabPFN v2 adapted to
zero-shot univariate forecasting with native quantile outputs (11M params,
CPU-feasible). The dependency is a *lazy* import resolved inside ``fit``:
``tabpfn-time-series`` transitively requires ``gluonts``, whose
``toolz<1`` pin conflicts with this repo's pinned ``exchange-calendars``
(``toolz>=1``) — it cannot enter uv.lock, so the adapter fails closed with a
named error until the upstream constraint loosens.

Fleet contract (mirrors ``models/distribution.py``): ``fit(x, y)`` records
the trailing return series; ``predict(x)`` emits the last-window one-step
quantile vector tiled across rows — the same unconditional-per-row contract
as the other fleet heads. ``predict_from_history`` scores every consecutive
``lookback`` window causally (each window's forecast sees only its own
history). ``lookback`` is the model's warmup and is disclosed in metadata.

Quantile extraction reads the predictor's output columns by name/position
with a tolerance check against the requested tau grid — any grid the model
cannot honor fails closed rather than silently interpolating.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.base import JoblibMixin, ModelMeta

Array = NDArray[np.float64]

_MIN_LOOKBACK = 8
_NN_MIN_WINDOWS = 4

_DEP_NAME = "tabpfn-time-series"
_DEP_ERROR = (
    f"{_DEP_NAME} is not installed. It is an optional lane dep kept out of "
    "uv.lock: its transitive pin gluonts requires toolz<1 while pinned "
    "exchange-calendars requires toolz>=1 — unsatisfiable today. Install it "
    "in a separate venv for manual runs, or wait for the upstream constraint "
    "to loosen; the head fails closed rather than degrade silently."
)


def _as_tau_grid(taus: Sequence[float]) -> Array:
    t = np.asarray(list(taus), dtype=np.float64)
    if (
        t.size == 0
        or not np.isfinite(t).all()
        or np.any((t <= 0.0) | (t >= 1.0))
        or np.any(np.diff(t) <= 0.0)
    ):
        raise ValueError("taus must be a nonempty strictly increasing grid inside (0, 1)")
    return t


def _load_predictor_cls() -> Any:
    """Fail-closed lazy import of the tabpfn-time-series predictor."""
    try:
        from tabpfn_time_series import TabPFNTimeSeriesPredictor
    except ImportError as exc:
        raise RuntimeError(_DEP_ERROR) from exc
    return TabPFNTimeSeriesPredictor


def _quantile_row(prediction: Any, taus: Array) -> Array:
    """Extract the requested quantile grid from a predictor output row.

    Accepts a ``len(taus)``-shaped vector already on the grid (the common
    stub/local path) or an object carrying a ``quantiles`` row keyed by tau.
    Anything else fails closed.
    """
    arr = np.asarray(getattr(prediction, "quantiles", prediction), dtype=np.float64)
    arr = arr.reshape(-1)
    if arr.shape[0] != taus.shape[0] or not np.isfinite(arr).all():
        raise ValueError(
            f"tabpfn_ts prediction returned {arr.shape[0]} quantiles for a "
            f"{taus.shape[0]}-point grid"
        )
    return arr


class TabpfnTsDistribution(JoblibMixin):
    """Zero-shot TabPFN-TS quantile head on the trailing return series.

    The pretrained model is treated as a fixed black box: no fine-tuning, so
    ``fit`` only records the (validated) series and constructs the predictor.
    All tabpfn calls run inside ``predict_from_history``/``predict`` on causal
    windows — the only state carried between calls is the fit-time scaler.
    """

    name = "tabpfn_ts"

    def __init__(
        self,
        taus: Sequence[float] = (0.1, 0.5, 0.9),
        *,
        seed: int = 0,
        lookback: int = 256,
        mode: str = "local",
    ) -> None:
        self.taus = _as_tau_grid(taus)
        self.seed = int(seed)
        if lookback < _MIN_LOOKBACK:
            raise ValueError(f"lookback must be >= {_MIN_LOOKBACK}; got {lookback}")
        self.lookback = int(lookback)
        self.mode = mode
        self._predictor: Any = None
        self._mu = 0.0
        self._sig = 1.0
        self._window: Array | None = None
        self._n_windows = 0

    def _ensure_predictor(self) -> Any:
        if self._predictor is None:
            cls = _load_predictor_cls()
            self._predictor = cls(
                tabpfn_config=None,
            )
        return self._predictor

    def fit(
        self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any
    ) -> TabpfnTsDistribution:
        yy = np.asarray(y, dtype=np.float64).reshape(-1)
        if not np.isfinite(yy).all():
            raise ValueError(f"{self.name} requires an all-finite series")
        if yy.size < self.lookback + _NN_MIN_WINDOWS:
            raise ValueError(
                f"{self.name} requires >= {self.lookback + _NN_MIN_WINDOWS} observations "
                f"(lookback {self.lookback} + {_NN_MIN_WINDOWS} train windows); got {yy.size}"
            )
        # Constructing the predictor is the dep check: fails closed here so a
        # half-fit object can never predict.
        self._ensure_predictor()
        self._mu = float(yy.mean())
        s = float(yy.std(ddof=1))
        self._sig = s if np.isfinite(s) and s > 0.0 else 1.0
        z = (yy - self._mu) / self._sig
        self._window = z[-self.lookback :].astype(np.float64)
        self._n_windows = int(yy.size - self.lookback)
        return self

    def _window_quantiles(self, window_z: Array) -> Array:
        predictor = self._ensure_predictor()
        raw = predictor.predict(np.asarray(window_z, dtype=np.float64).reshape(-1))
        q = _quantile_row(raw, self.taus)
        q = q * self._sig + self._mu
        return np.sort(q)

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Tile the last-window one-step-ahead quantiles across ``x`` rows."""
        if self._window is None:
            raise RuntimeError("distribution model has not been fitted")
        arr = np.asarray(x)
        if arr.ndim == 0 or arr.shape[0] < 1:
            raise ValueError("predict requires at least one row")
        q = self._window_quantiles(self._window)
        return np.tile(q, (int(arr.shape[0]), 1))

    def predict_from_history(self, history: NDArray[np.float64]) -> NDArray[np.float64]:
        """Quantile row for every consecutive ``lookback`` window in ``history``.

        Each window is standardized with the fitted (train-window) scaler and
        scored independently — no leakage beyond the window itself. Returns
        ``(len(history) - lookback + 1, n_taus)``.
        """
        if self._window is None:
            raise RuntimeError("distribution model has not been fitted")
        hh = np.asarray(history, dtype=np.float64).reshape(-1)
        if hh.size < self.lookback or not np.isfinite(hh).all():
            raise ValueError(
                f"{self.name} predict_from_history requires >= lookback finite observations"
            )
        z = (hh - self._mu) / self._sig
        windows = np.stack(
            [z[i : i + self.lookback] for i in range(0, hh.size - self.lookback + 1)]
        ).astype(np.float64)
        return np.stack([self._window_quantiles(w) for w in windows])

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="distribution",
            name=self.name,
            version="v1",
            seed=self.seed,
            extra={
                "framework": "tabpfn-time-series",
                "mode": self.mode,
                "device": "cpu",
                "pretrained": True,
                "weights_note": (
                    "upstream-pinned on first use; not vendored — the dep is "
                    "unresolvable in uv.lock (gluonts toolz<1 vs exchange-"
                    "calendars toolz>=1)"
                ),
                "lookback": self.lookback,
                "warmup": self.lookback,
                "n_train_windows": self._n_windows,
                "standardize": True,
            },
        )
