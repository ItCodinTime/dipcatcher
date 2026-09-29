"""Moirai-2.0 quantile head (P2.1). Research-only.

Wraps Salesforce ``uni2ts`` — the Moirai-2.0 decoder-only transformer
pretrained for universal time-series forecasting, loading the
``Salesforce/moirai-2.0-R-small`` weights (#1 non-leaking on GIFT-Eval MASE
at the 2026-09-22 research scan). The dependency is a *lazy* import resolved
inside ``fit``: every published ``uni2ts`` (1.1.0–2.0.0) pins
``scipy>=1.11.3,<1.12.dev0`` and ``numpy~=1.26.0`` while this repo requires
``scipy>=1.14`` and ``numpy>=2.0``, and its ``gluonts~=0.14.3`` pin carries
the same ``toolz<1`` conflict as tabpfn-time-series — it cannot enter
uv.lock, so the adapter fails closed with a named error until the upstream
constraints loosen. Upstream ``pyproject.toml`` on main pins the same
ranges, so a git install does not help either.

Fleet contract (mirrors ``models/distribution.py``): ``fit(x, y)`` records
the trailing return series; ``predict(x)`` emits the last-window one-step
quantile vector tiled across rows — the same unconditional-per-row contract
as the other fleet heads. ``predict_from_history`` scores every consecutive
``lookback`` window causally (each window's forecast sees only its own
history). ``lookback`` is the model's warmup and is disclosed in metadata.

Quantile extraction runs over the forecast's ``samples`` (uni2ts emits
sample paths; the requested tau grid is taken with ``np.quantile``), and a
plain ``len(taus)`` vector already on the grid is accepted for the stub
path — any output the model cannot honor fails closed rather than silently
interpolating. The uni2ts call sequence below follows the documented
inference API (``MoiraiModule.from_pretrained`` → ``MoiraiForecast`` →
``create_predictor`` → ``predict`` over single-series GluonTS-style dicts);
it is unexercised in this env and unverifiable until the dep resolves.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.base import JoblibMixin, ModelMeta

Array = NDArray[np.float64]

_MIN_LOOKBACK = 8
_NN_MIN_WINDOWS = 4
_NUM_SAMPLES = 100
_PREDICT_BATCH = 32
_WEIGHTS_ID = "Salesforce/moirai-2.0-R-small"
# Deterministic GluonTS-style start timestamp for the one-series predict
# payload; only ordering within the window matters, so any fixed date works.
_START = "2000-01-01"

_DEP_NAME = "uni2ts"
_DEP_ERROR = (
    f"{_DEP_NAME} is not installed. It is an optional lane dep kept out of "
    "uv.lock: every published uni2ts (1.1.0-2.0.0) pins "
    "scipy>=1.11.3,<1.12.dev0 and numpy~=1.26.0 while this repo requires "
    "scipy>=1.14 and numpy>=2.0, and its gluonts~=0.14.3 pin requires "
    "toolz<1 against pinned exchange-calendars' toolz>=1 — unsatisfiable "
    "today (uv resolver: 'all versions of uni2ts depend on "
    "scipy>=1.11.3,<1.12.dev0 and your project depends on scipy>=1.14'). "
    "Install it in a separate venv for manual runs, or wait for the "
    "upstream constraints to loosen; the head fails closed rather than "
    "degrade silently."
)


def availability() -> bool:
    """Whether the optional ``uni2ts`` lane dep is importable right now."""
    if "uni2ts" in sys.modules or "uni2ts.model.moirai" in sys.modules:
        return True
    import importlib.util

    return importlib.util.find_spec("uni2ts") is not None


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


def _load_moirai() -> tuple[Any, Any]:
    """Fail-closed lazy import of the uni2ts Moirai forecast classes."""
    try:
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    except ImportError as exc:
        raise RuntimeError(_DEP_ERROR) from exc
    return MoiraiForecast, MoiraiModule


def _quantile_row(prediction: Any, taus: Array) -> Array:
    """Extract the requested quantile grid from a predictor output row.

    Accepts an object carrying ``samples`` (uni2ts emits sample paths — the
    grid is read off them with ``np.quantile``) or a ``len(taus)``-shaped
    vector already on the grid (the common stub/local path). Anything else
    fails closed.
    """
    samples = getattr(prediction, "samples", None)
    if samples is not None:
        s = np.asarray(samples, dtype=np.float64).reshape(-1)
        if s.size == 0 or not np.isfinite(s).all():
            raise ValueError("moirai2 prediction returned empty or non-finite samples")
        return np.quantile(s, taus)
    arr = np.asarray(prediction, dtype=np.float64)
    arr = arr.reshape(-1)
    if arr.shape[0] != taus.shape[0] or not np.isfinite(arr).all():
        raise ValueError(
            f"moirai2 prediction returned {arr.shape[0]} quantiles for a {taus.shape[0]}-point grid"
        )
    return arr


class Moirai2Distribution(JoblibMixin):
    """Zero-shot Moirai-2.0 quantile head on the trailing return series.

    The pretrained model is treated as a fixed black box: no fine-tuning, so
    ``fit`` only records the (validated) series and constructs the predictor.
    All uni2ts calls run inside ``predict_from_history``/``predict`` on causal
    windows — the only state carried between calls is the fit-time scaler.
    """

    name = "moirai2"

    def __init__(
        self,
        taus: Sequence[float] = (0.1, 0.5, 0.9),
        *,
        seed: int = 0,
        lookback: int = 256,
        mode: str = "local",
        weights_id: str = _WEIGHTS_ID,
    ) -> None:
        self.taus = _as_tau_grid(taus)
        self.seed = int(seed)
        if lookback < _MIN_LOOKBACK:
            raise ValueError(f"lookback must be >= {_MIN_LOOKBACK}; got {lookback}")
        self.lookback = int(lookback)
        self.mode = mode
        self.weights_id = weights_id
        self._predictor: Any = None
        self._mu = 0.0
        self._sig = 1.0
        self._window: Array | None = None
        self._n_windows = 0

    def _ensure_predictor(self) -> Any:
        if self._predictor is None:
            forecast_cls, module_cls = _load_moirai()
            module = module_cls.from_pretrained(self.weights_id)
            forecast = forecast_cls(
                module=module,
                prediction_length=1,
                context_length=self.lookback,
                patch_size="auto",
                num_samples=_NUM_SAMPLES,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
            self._predictor = forecast.create_predictor(batch_size=_PREDICT_BATCH)
        return self._predictor

    def fit(
        self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any
    ) -> Moirai2Distribution:
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
        dataset = [{"target": np.asarray(window_z, dtype=np.float32), "start": _START}]
        raw = next(iter(predictor.predict(dataset)), None)
        if raw is None:
            raise ValueError("moirai2 predictor returned no forecasts")
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
                "framework": "uni2ts",
                "weights_id": self.weights_id,
                "mode": self.mode,
                "device": "cpu",
                "pretrained": True,
                "weights_note": (
                    "upstream-pinned on first use; not vendored — the dep is "
                    "unresolvable in uv.lock (uni2ts scipy>=1.11.3,<1.12.dev0 "
                    "and numpy~=1.26.0 vs pinned scipy>=1.14 / numpy>=2.0)"
                ),
                "lookback": self.lookback,
                "warmup": self.lookback,
                "n_train_windows": self._n_windows,
                "standardize": True,
            },
        )
