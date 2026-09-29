"""TiRex-2 time-series quantile head (P2.2). Research-only.

Wraps NX-AI ``tirex-2`` — pretrained xLSTM-stack (sLSTM/mLSTM attention
blocks) zero-shot multivariate forecaster with a native 9-point quantile
output grid ``{0.1, …, 0.9}`` (~380M state-dict checkpoint, CPU-feasible
torch backend). The dependency is a *lazy* import resolved inside ``fit``:
``tirex-2`` is an optional lane dep of the ``nn`` extra — install with
``uv sync --extra nn`` — so the adapter fails closed with a named error
when the package is absent.

Dep-resolution evidence (2026-09-29): ``tirex-2>=0.3.0`` resolves cleanly
in uv.lock (``Resolved 222 packages``: adds einops 0.8.2, flashrnn 1.0.8,
mlstm-kernels 2.0.6, ninja, safetensors, tirex-2 0.3.0, triton-windows
marker, xlstm 2.0.6 — all universal or multiplatform wheels) and installs
on macOS arm64; ``import tirex2`` + ``load_model("NX-AI/TiRex-2",
device="cpu")`` verified end to end. Package-name footgun: ``tirex`` on
PyPI is an unrelated dimensionality-reduction package, and ``tirex-ts``
(TiRex-1 lineage, NXAI Community License) cannot load TiRex-2
checkpoints — its loader expects a lightning ``hyper_parameters`` dict and
dies with ``KeyError: 'hyper_parameters'`` on the TiRex-2 raw state-dict.
``tirex-2`` (Apache-2.0) is the only correct package name.

Checkpoints: default ``NX-AI/TiRex-2``. The plan prefers a *decontaminated*
checkpoint for the fev-bench/GIFT overlap question — NX-AI ships
``NX-AI/TiRex-2-gifteval-zs`` and ``NX-AI/TiRex-2-fevbench`` for exactly
that; pass either via ``model_id`` when scoring benchmark cells. Weights
are not vendored: the first ``fit`` downloads the checkpoint from
HuggingFace (network); nothing runs without them.

Fleet contract (mirrors ``models/distribution.py`` and the tabpfn_ts
adapter): ``fit(x, y)`` records the trailing return series;
``predict(x)`` emits the last-window one-step quantile vector tiled
across rows — the same unconditional-per-row contract as the other fleet
heads. ``predict_from_history`` scores every consecutive ``lookback``
window causally (each window's forecast sees only its own history,
batched into one ``forecast`` call). ``lookback`` is the context window
and is disclosed in metadata. Series are passed raw — TiRex-2 normalizes
internally (arcsinh scaler in ``model-config.yaml``).

Quantile extraction matches the requested tau grid against the model's
*native* grid read from ``model.quantiles`` within ``grid_tol`` — any
tau the checkpoint cannot honor natively fails closed rather than
silently interpolating or clamping (the default 7-point fleet grid
``{0.05,…,0.95}`` is therefore not honorably supported; score this head
on a subset of the native grid, e.g. ``--taus 0.1,0.5,0.9``).
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

_DEP_NAME = "tirex-2"
_DEP_ERROR = (
    f"{_DEP_NAME} is not installed. It is an optional lane dep of the `nn` "
    "extra — resolved into uv.lock with torch/xlstm/flashrnn; install with "
    "`uv sync --extra nn`. The head fails closed rather than degrade "
    "silently."
)

# Upstream decontaminated checkpoints for benchmark-overlap cells (P2.2).
DECONTAMINATED_MODEL_IDS = (
    "NX-AI/TiRex-2-gifteval-zs",
    "NX-AI/TiRex-2-fevbench",
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


def _load_model_fns() -> tuple[Any, Any]:
    """Fail-closed lazy import of the tirex-2 loader and input type."""
    try:
        from tirex2 import TimeseriesType, load_model
    except ImportError as exc:
        raise RuntimeError(_DEP_ERROR) from exc
    return load_model, TimeseriesType


def _native_grid(model: Any) -> Array:
    """Read the checkpoint's native quantile levels as a sorted float grid."""
    raw = np.asarray(model.quantiles, dtype=np.float64).reshape(-1)
    if raw.size == 0 or not np.isfinite(raw).all() or np.any(np.diff(raw) <= 0.0):
        raise ValueError("tirex-2 model exposes no usable native quantile grid")
    return raw


def _native_indices(taus: Array, native: Array, tol: float) -> NDArray[np.int64]:
    """Map each requested tau to its native-grid index, or fail closed.

    The checkpoint emits exactly ``len(native)`` quantile columns; honoring a
    tau that is not natively present would require interpolation or edge
    clamping, which the honesty contract disallows — so a miss raises
    rather than approximating.
    """
    idx = np.empty(taus.shape[0], dtype=np.int64)
    for i, t in enumerate(taus):
        d = np.abs(native - t)
        j = int(np.argmin(d))
        if d[j] > tol:
            raise ValueError(
                f"tirex2 cannot honor tau={t:g}: native grid is "
                f"{[round(float(v), 6) for v in native]} — score this head on "
                "native levels only"
            )
        idx[i] = j
    return idx


def _quantile_row(forecast_row: Any, idx: NDArray[np.int64]) -> Array:
    """Extract the native-grid quantile row from one forecast output.

    Accepts a ``[V_t, Q, H]`` array (the upstream ``output_type="numpy"``
    contract) or a ``[Q]``/``[Q, H]`` stub row; picks variate 0, the mapped
    native indices, horizon 0. Anything else fails closed.
    """
    arr = np.asarray(forecast_row, dtype=np.float64)
    row: Array
    if arr.ndim == 3:
        row = arr[0, :, 0]
    elif arr.ndim == 2:
        row = arr[:, 0]
    else:
        row = arr.reshape(-1)
    if int(idx.max()) >= row.shape[0] or not np.isfinite(row).all():
        raise ValueError(
            f"tirex2 forecast returned {row.shape[0]} quantiles for a "
            f"native grid of {int(idx.max()) + 1}+ levels"
        )
    return np.asarray(row[idx], dtype=np.float64)


class Tirex2Distribution(JoblibMixin):
    """Zero-shot TiRex-2 quantile head on the trailing return series.

    The pretrained model is treated as a fixed black box: no fine-tuning, so
    ``fit`` only records the (validated) series and loads the checkpoint.
    All tirex-2 calls run inside ``predict_from_history``/``predict`` on
    causal windows — the only state carried between calls is the recorded
    trailing window.
    """

    name = "tirex2"

    def __init__(
        self,
        taus: Sequence[float] = (0.1, 0.5, 0.9),
        *,
        seed: int = 0,
        lookback: int = 256,
        model_id: str = "NX-AI/TiRex-2",
        device: str = "cpu",
        grid_tol: float = 1e-6,
    ) -> None:
        self.taus = _as_tau_grid(taus)
        self.seed = int(seed)
        if lookback < _MIN_LOOKBACK:
            raise ValueError(f"lookback must be >= {_MIN_LOOKBACK}; got {lookback}")
        self.lookback = int(lookback)
        self.model_id = str(model_id)
        self.device = str(device)
        self.grid_tol = float(grid_tol)
        if not np.isfinite(self.grid_tol) or self.grid_tol < 0.0:
            raise ValueError("grid_tol must be a nonnegative finite tolerance")
        self._model: Any = None
        self._ts_type: Any = None
        self._idx: NDArray[np.int64] | None = None
        self._window: Array | None = None
        self._n_windows = 0

    def _ensure_model(self) -> tuple[Any, NDArray[np.int64]]:
        """Load (once) and return the model plus its native-grid index map."""
        if self._model is None or self._idx is None:
            load_model, ts_type = _load_model_fns()
            model = load_model(self.model_id, device=self.device)
            self._ts_type = ts_type
            native = _native_grid(model)
            self._idx = _native_indices(self.taus, native, self.grid_tol)
            self._model = model
        return self._model, self._idx

    def fit(
        self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any
    ) -> Tirex2Distribution:
        yy = np.asarray(y, dtype=np.float64).reshape(-1)
        if not np.isfinite(yy).all():
            raise ValueError(f"{self.name} requires an all-finite series")
        if yy.size < self.lookback + _NN_MIN_WINDOWS:
            raise ValueError(
                f"{self.name} requires >= {self.lookback + _NN_MIN_WINDOWS} observations "
                f"(lookback {self.lookback} + {_NN_MIN_WINDOWS} train windows); got {yy.size}"
            )
        # Loading the checkpoint is the dep + weights check: fails closed here
        # so a half-fit object can never predict.
        self._ensure_model()
        self._window = yy[-self.lookback :].astype(np.float64)
        self._n_windows = int(yy.size - self.lookback)
        return self

    def _window_quantiles(self, windows: Array) -> Array:
        """Quantile rows for a ``(n_windows, lookback)`` batch of raw windows."""
        model, idx = self._ensure_model()
        import torch  # local lane dep; installed alongside tirex-2

        ctx = torch.from_numpy(np.asarray(windows, dtype=np.float32))
        batch = [
            self._ts_type(
                target=ctx[i : i + 1],
                past_covariates=None,
                future_covariates=None,
            )
            for i in range(ctx.shape[0])
        ]
        out = model.forecast(batch, prediction_length=1, output_type="numpy")
        rows = np.stack([_quantile_row(o, idx) for o in out])
        return np.sort(rows, axis=1)

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Tile the last-window one-step-ahead quantiles across ``x`` rows."""
        if self._window is None:
            raise RuntimeError("distribution model has not been fitted")
        arr = np.asarray(x)
        if arr.ndim == 0 or arr.shape[0] < 1:
            raise ValueError("predict requires at least one row")
        q = self._window_quantiles(self._window[None, :])[0]
        return np.tile(q, (int(arr.shape[0]), 1))

    def predict_from_history(self, history: NDArray[np.float64]) -> NDArray[np.float64]:
        """Quantile row for every consecutive ``lookback`` window in ``history``.

        Each window is scored independently by the pretrained model — no
        leakage beyond the window itself. Returns
        ``(len(history) - lookback + 1, n_taus)``.
        """
        if self._window is None:
            raise RuntimeError("distribution model has not been fitted")
        hh = np.asarray(history, dtype=np.float64).reshape(-1)
        if hh.size < self.lookback or not np.isfinite(hh).all():
            raise ValueError(
                f"{self.name} predict_from_history requires >= lookback finite observations"
            )
        windows = np.stack(
            [hh[i : i + self.lookback] for i in range(0, hh.size - self.lookback + 1)]
        ).astype(np.float64)
        return self._window_quantiles(windows)

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="distribution",
            name=self.name,
            version="v1",
            seed=self.seed,
            extra={
                "framework": "tirex-2",
                "model_id": self.model_id,
                "decontaminated_model_ids": list(DECONTAMINATED_MODEL_IDS),
                "device": self.device,
                "pretrained": True,
                "weights_note": (
                    "HF hub checkpoint downloaded on first fit; not vendored. "
                    "tirex-2>=0.3.0 resolved into uv.lock on 2026-09-29 — "
                    "unlike tabpfn-time-series, the lane is installable."
                ),
                "lookback": self.lookback,
                "warmup": self.lookback,
                "n_train_windows": self._n_windows,
                "standardize": False,
                "quantile_mapping": "native",
                "grid_tol": self.grid_tol,
            },
        )
