"""Tiny causal decoder for hierarchical K-line tokens (research-scale).

This is the Kronos decoder-only contract in NumPy: hierarchical embeddings,
causal self-attention, then s1 logits and s2 logits conditioned on the sampled
s1 embedding. Weights are seeded, not the 24M–100M Hugging Face checkpoints.
Use ``decoder=hierarchical_markov`` for the default lookback-conditioned path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.robinhood_plus.autoregress import nucleus_sample
from quant_fund.models.robinhood_plus.tokenizer import HierarchicalBSQTokenizer, HierarchicalTokens

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _softmax(logits: Array, axis: int = -1) -> Array:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(exp.sum(axis=axis, keepdims=True), 1e-12, None)


def _silu(x: Array) -> Array:
    return x / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _rms_norm(x: Array, weight: Array, eps: float = 1e-5) -> Array:
    scale = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / scale) * weight


def _causal_attention(q: Array, k: Array, v: Array) -> Array:
    """q,k,v: (T, H, D) → (T, H, D)."""
    t, heads, dim = q.shape
    scale = 1.0 / np.sqrt(dim)
    scores = np.einsum("thd,shd->hts", q, k) * scale
    mask = np.triu(np.ones((t, t), dtype=bool), k=1)
    scores[:, mask] = -1e9
    weights = _softmax(scores, axis=-1)
    return np.einsum("hts,shd->thd", weights, v)


@dataclass
class TinyHierarchicalTransformer:
    """Seeded causal transformer with Kronos-style dual heads."""

    tokenizer: HierarchicalBSQTokenizer
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 1
    temperature: float = 1.0
    top_p: float = 0.9
    seed: int = 42

    def __post_init__(self) -> None:
        if self.d_model < 1 or self.n_heads < 1 or self.n_layers < 1:
            raise ValueError("transformer sizes must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not np.isfinite(self.top_p) or not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be finite and in (0, 1]")
        rng = np.random.default_rng(self.seed)
        d = self.d_model
        scale = d**-0.5
        self.emb_s1 = rng.normal(0.0, scale, (self.tokenizer.s1_vocab, d))
        self.emb_s2 = rng.normal(0.0, scale, (self.tokenizer.s2_vocab, d))
        self.fuse = rng.normal(0.0, scale, (2 * d, d))
        self.layers: list[dict[str, Array]] = []
        for _ in range(self.n_layers):
            self.layers.append(
                {
                    "wq": rng.normal(0.0, scale, (d, d)),
                    "wk": rng.normal(0.0, scale, (d, d)),
                    "wv": rng.normal(0.0, scale, (d, d)),
                    "wo": rng.normal(0.0, scale, (d, d)),
                    "w1": rng.normal(0.0, scale, (d, 2 * d)),
                    "w3": rng.normal(0.0, scale, (d, 2 * d)),
                    "w2": rng.normal(0.0, scale, (2 * d, d)),
                    "n1": np.ones(d),
                    "n2": np.ones(d),
                }
            )
        self.norm = np.ones(d)
        self.head_s1 = rng.normal(0.0, scale, (d, self.tokenizer.s1_vocab))
        self.head_s2 = rng.normal(0.0, scale, (d, self.tokenizer.s2_vocab))
        self.dep = rng.normal(0.0, scale, (2 * d, d))

    def _embed(self, s1: IntArray, s2: IntArray) -> Array:
        fused = np.concatenate([self.emb_s1[s1], self.emb_s2[s2]], axis=-1)
        return fused @ self.fuse

    def _forward(self, s1: IntArray, s2: IntArray) -> Array:
        x = self._embed(s1, s2)
        head_dim = self.d_model // self.n_heads
        for layer in self.layers:
            q = (x @ layer["wq"]).reshape(x.shape[0], self.n_heads, head_dim)
            k = (x @ layer["wk"]).reshape(x.shape[0], self.n_heads, head_dim)
            v = (x @ layer["wv"]).reshape(x.shape[0], self.n_heads, head_dim)
            attn = _causal_attention(q, k, v).reshape(x.shape[0], self.d_model)
            x = _rms_norm(x + attn @ layer["wo"], layer["n1"])
            ff = _silu(x @ layer["w1"]) * (x @ layer["w3"])
            x = _rms_norm(x + ff @ layer["w2"], layer["n2"])
        return _rms_norm(x, self.norm)

    def generate(
        self,
        tokens: HierarchicalTokens,
        pred_len: int,
        *,
        sample_count: int = 1,
        seed: int | None = None,
        max_context: int = 128,
    ) -> tuple[IntArray, IntArray]:
        if pred_len < 1 or sample_count < 1:
            raise ValueError("pred_len and sample_count must be positive")
        if max_context < 1:
            raise ValueError("max_context must be positive")
        hist_s1 = np.asarray(tokens.s1, dtype=np.int64).reshape(-1)
        hist_s2 = np.asarray(tokens.s2, dtype=np.int64).reshape(-1)
        rng = np.random.default_rng(self.seed if seed is None else int(seed))
        out_s1 = np.empty((sample_count, pred_len), dtype=np.int64)
        out_s2 = np.empty((sample_count, pred_len), dtype=np.int64)
        for sample in range(sample_count):
            s1 = hist_s1.copy()
            s2 = hist_s2.copy()
            for step in range(pred_len):
                window_s1 = s1[-max_context:]
                window_s2 = s2[-max_context:]
                context = self._forward(window_s1, window_s2)
                next_s1 = nucleus_sample(
                    context[-1] @ self.head_s1,
                    rng,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                cond = np.concatenate([context[-1], self.emb_s1[next_s1]], axis=-1) @ self.dep
                next_s2 = nucleus_sample(
                    cond @ self.head_s2,
                    rng,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                s1 = np.concatenate([s1, np.asarray([next_s1], dtype=np.int64)])
                s2 = np.concatenate([s2, np.asarray([next_s2], dtype=np.int64)])
                out_s1[sample, step] = next_s1
                out_s2[sample, step] = next_s2
        return out_s1, out_s2
