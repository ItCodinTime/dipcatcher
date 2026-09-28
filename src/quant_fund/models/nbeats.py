"""Deterministic CPU N-BEATS / N-HiTS quantile heads (P2.7). Research-only.

Direct torch implementations of N-BEATS (Oreshkin et al. 2020, doubly-residual
stacks of MLP blocks with learned generic bases) and N-HiTS (Challu et al.
2023, multi-rate input pooling + coarse forecast interpolation). No Darts
dependency. Torch is the optional ``nn`` extra and is imported lazily inside
``fit``/``predict_from_history``, so this module imports without it.

Fleet contract (mirrors ``models/distribution.py``): ``fit(x, y)`` slides a
causal ``lookback`` window over the trailing return series and trains on the
pinball loss of the next-step quantile grid. ``predict(x)`` emits the
one-step-ahead quantile vector forecast from the last ``lookback``
observations of the fit series, tiled across rows — the same
unconditional-per-row contract the other fleet heads satisfy. ``lookback``
is the model's warmup and is disclosed in ``metadata().extra``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.base import JoblibMixin, ModelMeta

Array = NDArray[np.float64]

_NN_MIN_WINDOWS = 24
_MIN_LOOKBACK = 8


def _as_tau_grid(taus: Any) -> Array:
    t = np.asarray(list(taus), dtype=np.float64)
    if (
        t.size == 0
        or not np.isfinite(t).all()
        or np.any((t <= 0.0) | (t >= 1.0))
        or np.any(np.diff(t) <= 0.0)
    ):
        raise ValueError("taus must be a nonempty strictly increasing grid inside (0, 1)")
    return t


def _make_nbeats(
    in_dim: int, out_dim: int, *, hidden: int, depth: int, stacks: int, blocks: int
) -> Any:
    """Generic-basis N-BEATS: doubly-residual blocks summing to a forecast."""
    import torch

    class _Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[torch.nn.Module] = []
            d = in_dim
            for _ in range(depth):
                layers += [torch.nn.Linear(d, hidden), torch.nn.ReLU()]
                d = hidden
            self.mlp = torch.nn.Sequential(*layers)
            self.theta = torch.nn.Linear(hidden, in_dim + out_dim)
            self.basis_b = torch.nn.Linear(in_dim, in_dim, bias=False)
            self.basis_f = torch.nn.Linear(out_dim, out_dim, bias=False)

        def forward(self, x: Any) -> tuple[Any, Any]:
            t = self.theta(self.mlp(x))
            return self.basis_b(t[:, :in_dim]), self.basis_f(t[:, in_dim:])

    class _Net(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList(_Block() for _ in range(stacks * blocks))

        def forward(self, x: Any) -> Any:
            residual = x
            forecast = None
            for block in self.blocks:
                backcast, step = block(residual)
                residual = residual - backcast
                forecast = step if forecast is None else forecast + step
            return forecast

    return _Net()


def _make_nhits(
    in_dim: int, out_dim: int, *, hidden: int, depth: int, pools: tuple[int, ...]
) -> Any:
    """Minimal N-HiTS: per-stack avg-pooled input, coarse forecast upsampled."""
    import torch
    import torch.nn.functional as F

    class _Block(torch.nn.Module):
        def __init__(self, pool: int) -> None:
            super().__init__()
            self.pool = int(pool)
            coarse = max(1, out_dim // self.pool)
            self.coarse = coarse
            pooled_in = max(1, in_dim // self.pool)
            layers: list[torch.nn.Module] = []
            d = pooled_in
            for _ in range(depth):
                layers += [torch.nn.Linear(d, hidden), torch.nn.ReLU()]
                d = hidden
            self.mlp = torch.nn.Sequential(*layers)
            self.theta = torch.nn.Linear(hidden, in_dim + coarse)
            self.basis_b = torch.nn.Linear(in_dim, in_dim, bias=False)
            self.basis_f = torch.nn.Linear(coarse, coarse, bias=False)

        def forward(self, x: Any) -> tuple[Any, Any]:
            pooled = F.avg_pool1d(
                x[:, None, :], kernel_size=self.pool, stride=self.pool, ceil_mode=False
            )[:, 0]
            t = self.theta(self.mlp(pooled))
            backcast = self.basis_b(t[:, :in_dim])
            f_coarse = self.basis_f(t[:, in_dim:])
            if self.coarse == out_dim:
                forecast = f_coarse
            else:
                forecast = F.interpolate(
                    f_coarse[:, None, :], size=out_dim, mode="linear", align_corners=True
                )[:, 0]
            return backcast, forecast

    class _Net(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList(_Block(p) for p in pools)

        def forward(self, x: Any) -> Any:
            residual = x
            forecast = None
            for block in self.blocks:
                backcast, step = block(residual)
                residual = residual - backcast
                forecast = step if forecast is None else forecast + step
            return forecast

    return _Net()


class _QuantileSequenceBase(JoblibMixin):
    """Shared fit/predict for sliding-window next-step quantile nets."""

    name = "base"

    def __init__(
        self,
        taus: Any,
        *,
        lookback: int = 32,
        seed: int = 0,
        epochs: int = 150,
        hidden: int = 48,
        lr: float = 1e-3,
    ) -> None:
        self.taus = [float(t) for t in _as_tau_grid(taus)]
        if isinstance(lookback, bool) or int(lookback) < _MIN_LOOKBACK:
            raise ValueError(f"lookback must be an int >= {_MIN_LOOKBACK}")
        if isinstance(epochs, bool) or int(epochs) < 1:
            raise ValueError("epochs must be a positive int")
        self.lookback = int(lookback)
        self.seed = int(seed)
        self.epochs = int(epochs)
        self.hidden = int(hidden)
        self.lr = float(lr)
        self.mu_ = 0.0
        self.sig_ = 1.0
        self._net: Any = None
        self._window: Array | None = None
        self._n_windows = 0

    def _build(self) -> Any:
        raise NotImplementedError

    def fit(
        self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any
    ) -> _QuantileSequenceBase:
        import torch

        yy = np.asarray(y, dtype=np.float64).reshape(-1)
        if not np.isfinite(yy).all():
            raise ValueError(f"{self.name} requires an all-finite series")
        if yy.size < self.lookback + _NN_MIN_WINDOWS:
            raise ValueError(
                f"{self.name} requires >= {self.lookback + _NN_MIN_WINDOWS} observations "
                f"(lookback {self.lookback} + {_NN_MIN_WINDOWS} train windows); got {yy.size}"
            )
        torch.manual_seed(self.seed)
        torch.set_num_threads(1)
        self.mu_ = float(yy.mean())
        s = float(yy.std(ddof=1))
        self.sig_ = s if np.isfinite(s) and s > 0.0 else 1.0
        z = (yy - self.mu_) / self.sig_
        windows = np.stack(
            [z[i - self.lookback : i] for i in range(self.lookback, yy.size)]
        ).astype(np.float32)
        target = z[self.lookback :].astype(np.float32)
        net = self._build()
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        xb = torch.as_tensor(windows)
        tb = torch.as_tensor(target)
        taus_t = torch.as_tensor(np.asarray(self.taus, dtype=np.float32))
        net.train()
        for _ in range(self.epochs):
            opt.zero_grad(set_to_none=True)
            pred = net(xb)
            e = tb[:, None] - pred
            loss = torch.mean(torch.maximum(taus_t[None, :] * e, (taus_t[None, :] - 1.0) * e))
            loss.backward()
            opt.step()
        net.eval()
        self._net = net
        self._window = z[-self.lookback :].astype(np.float32)
        self._n_windows = int(windows.shape[0])
        return self

    def _quantiles(self, windows_z: Array) -> Array:
        import torch

        torch.set_num_threads(1)
        with torch.no_grad():
            out = self._net(torch.as_tensor(windows_z.astype(np.float32)))
        q = np.asarray(out.numpy(), dtype=np.float64) * self.sig_ + self.mu_
        return np.sort(q, axis=1)

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Tile the last-window one-step-ahead quantiles across ``x`` rows."""
        if self._net is None or self._window is None:
            raise RuntimeError("distribution model has not been fitted")
        arr = np.asarray(x)
        if arr.ndim == 0 or arr.shape[0] < 1:
            raise ValueError("predict requires at least one row")
        q = self._quantiles(self._window[None, :])[0]
        return np.tile(q, (int(arr.shape[0]), 1))

    def predict_from_history(self, history: NDArray[np.float64]) -> NDArray[np.float64]:
        """Quantile row for every consecutive ``lookback`` window in ``history``.

        The history is standardized with the fitted (train-window) scaler —
        no refit, no leakage beyond the window itself. Returns
        ``(len(history) - lookback, n_taus)``.
        """
        if self._net is None:
            raise RuntimeError("distribution model has not been fitted")
        hh = np.asarray(history, dtype=np.float64).reshape(-1)
        if hh.size < self.lookback or not np.isfinite(hh).all():
            raise ValueError(
                f"{self.name} predict_from_history requires >= lookback finite observations"
            )
        z = (hh - self.mu_) / self.sig_
        windows = np.stack(
            [z[i - self.lookback : i] for i in range(self.lookback, hh.size + 1)]
        ).astype(np.float64)
        return self._quantiles(windows)

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="distribution",
            name=self.name,
            version="v1",
            seed=self.seed,
            extra={
                "framework": "torch",
                "device": "cpu",
                "lookback": self.lookback,
                "warmup": self.lookback,
                "epochs": self.epochs,
                "hidden": self.hidden,
                "n_train_windows": self._n_windows,
                "standardize": True,
            },
        )


class NBeatsDistribution(_QuantileSequenceBase):
    """N-BEATS quantile head: 2 stacks x 2 generic blocks, pinball loss."""

    name = "nbeats"

    def _build(self) -> Any:
        return _make_nbeats(
            self.lookback,
            len(self.taus),
            hidden=self.hidden,
            depth=2,
            stacks=2,
            blocks=2,
        )


class NHiTsDistribution(_QuantileSequenceBase):
    """N-HiTS quantile head: multi-rate pooled stacks (pool 8/4/1, clipped to
    lookback)."""

    name = "nhits"

    def _build(self) -> Any:
        pools = tuple(p for p in (8, 4, 1) if p <= self.lookback)
        return _make_nhits(
            self.lookback,
            len(self.taus),
            hidden=self.hidden,
            depth=2,
            pools=pools,
        )
