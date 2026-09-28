"""PatchTST-style patch-transformer quantile head (P2.8). Research-only.

Direct torch implementation of the PatchTST reference line (Nie et al. 2023,
arXiv:2211.14730; the frontier item cites arXiv:2602.06909): a univariate
series is segmented into non-overlapping patches, each linearly embedded,
attended over by a TransformerEncoder, flattened, and mapped to the
quantile grid by a linear head. Channel-independence is degenerate here
(the fleet contract is one series), and global standardization plays the
RevIN role — instance stats come from the train window only. Torch is the
optional ``nn`` extra; it is imported lazily inside the network builder so
this module imports without it.

Fleet contract (mirrors ``models/distribution.py`` via
``_QuantileSequenceBase``): ``fit(x, y)`` slides a causal ``lookback``
window over the trailing series and trains on the pinball loss of the
next-step quantile grid; ``predict(x)`` emits the one-step-ahead quantile
vector from the last ``lookback`` observations, tiled across rows. When
``lookback`` is not divisible by ``patch_len`` the leading remainder is
dropped — patches are real data only, never padded.
"""

from __future__ import annotations

from typing import Any

from quant_fund.models.base import ModelMeta
from quant_fund.models.nbeats import _QuantileSequenceBase


def _make_patchtst(
    in_dim: int,
    out_dim: int,
    *,
    patch_len: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
) -> Any:
    """Channel-independent patch transformer: embed patches, attend, head."""
    import torch

    n_patches = in_dim // patch_len

    class _Net(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch_len = int(patch_len)
            self.n_patches = int(n_patches)
            self.embed = torch.nn.Linear(self.patch_len, d_model)
            self.pos = torch.nn.Parameter(torch.zeros(self.n_patches, d_model))
            layer = torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=2 * d_model,
                dropout=0.0,  # determinism: no stochastic regularization
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = torch.nn.TransformerEncoder(layer, num_layers=n_layers)
            self.head = torch.nn.Linear(self.n_patches * d_model, out_dim)

        def forward(self, x: Any) -> Any:
            # (B, in_dim) -> last n_patches*patch_len values -> (B, P, L)
            x = x[:, -self.n_patches * self.patch_len :]
            p = x.reshape(x.shape[0], self.n_patches, self.patch_len)
            h = self.embed(p) + self.pos
            h = self.encoder(h)
            return self.head(h.flatten(1))

    return _Net()


class PatchTSTDistribution(_QuantileSequenceBase):
    """PatchTST quantile head: patch_len-8 patching, 2-layer encoder."""

    name = "patchtst"

    def __init__(
        self,
        taus: Any,
        *,
        lookback: int = 32,
        seed: int = 0,
        epochs: int = 150,
        hidden: int = 64,
        lr: float = 1e-3,
        patch_len: int = 8,
        n_heads: int = 4,
        n_layers: int = 2,
    ) -> None:
        super().__init__(
            taus,
            lookback=lookback,
            seed=seed,
            epochs=epochs,
            hidden=hidden,
            lr=lr,
        )
        if isinstance(patch_len, bool) or int(patch_len) < 1:
            raise ValueError("patch_len must be a positive int")
        self.patch_len = int(patch_len)
        if self.lookback // self.patch_len < 2:
            raise ValueError(
                f"patch_len {self.patch_len} leaves < 2 patches over lookback "
                f"{self.lookback}; need lookback >= 2*patch_len"
            )
        if isinstance(n_heads, bool) or int(n_heads) < 1 or self.hidden % int(n_heads):
            raise ValueError("n_heads must be a positive int dividing hidden (d_model)")
        if isinstance(n_layers, bool) or int(n_layers) < 1:
            raise ValueError("n_layers must be a positive int")
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)

    def _build(self) -> Any:
        return _make_patchtst(
            self.lookback,
            len(self.taus),
            patch_len=self.patch_len,
            d_model=self.hidden,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
        )

    def metadata(self) -> ModelMeta:
        meta = super().metadata()
        meta.extra = {
            **dict(meta.extra or {}),
            "architecture": "patchtst",
            "patch_len": self.patch_len,
            "n_patches": self.lookback // self.patch_len,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "effective_lookback": (self.lookback // self.patch_len) * self.patch_len,
        }
        return meta
