"""Hierarchical binary-spherical tokenizer for K-line sequences.

Implements the Kronos stage-1 contract (Shi et al., 2025): a 6-d OHLCV+amount
vector is projected, L2-normalized, and quantized into coarse (s1) and fine (s2)
bipolar bits. The research-lab encoder is a seeded orthonormal linear map, not
the pretrained Kronos transformer encoder. Decode uses the same bit layout as
Kronos ``BSQuantizer.bits_to_indices`` (LSB at bit 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.robinhood_plus.constants import (
    DEFAULT_CLIP,
    DEFAULT_S1_BITS,
    DEFAULT_S2_BITS,
    KLINE_FEATURE_NAMES,
    NORM_EPS,
)

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]


def orthonormal_encoder(n_in: int, n_out: int, seed: int) -> Array:
    """Return a (n_in, n_out) linear map with orthonormal rows or columns."""
    if n_in < 1 or n_out < 1:
        raise ValueError("encoder dimensions must be positive")
    rng = np.random.default_rng(int(seed))
    if n_out >= n_in:
        raw = rng.standard_normal((n_out, n_in))
        q, _ = np.linalg.qr(raw)
        return np.asarray(q[:, :n_in].T, dtype=np.float64)
    raw = rng.standard_normal((n_in, n_out))
    q, _ = np.linalg.qr(raw)
    return np.asarray(q[:, :n_out], dtype=np.float64)


def bits_to_indices(bits: Array) -> IntArray:
    """Pack bipolar / boolean bits into integer codes (LSB = axis -1 index 0)."""
    binary = (np.asarray(bits) >= 0).astype(np.int64)
    n_bits = binary.shape[-1]
    powers = np.left_shift(1, np.arange(n_bits, dtype=np.int64))
    return np.asarray((binary * powers).sum(axis=-1), dtype=np.int64)


def indices_to_bits(indices: IntArray, n_bits: int) -> Array:
    """Unpack integer codes into bipolar ±1 bits (LSB first)."""
    if n_bits < 1:
        raise ValueError("n_bits must be positive")
    idx = np.asarray(indices, dtype=np.int64)
    shifts = np.arange(n_bits, dtype=np.int64)
    binary = ((idx[..., None] >> shifts) & 1).astype(np.float64)
    return binary * 2.0 - 1.0


def repair_ohlc(frame: Array) -> Array:
    """Enforce high ≥ max(open, close) and low ≤ min(open, close); clip volume."""
    out = np.asarray(frame, dtype=np.float64).copy()
    if out.ndim != 2 or out.shape[1] < 4:
        raise ValueError("K-line frame must be (T, ≥4)")
    open_px = out[:, 0]
    high = out[:, 1]
    low = out[:, 2]
    close = out[:, 3]
    body_hi = np.maximum(open_px, close)
    body_lo = np.minimum(open_px, close)
    out[:, 1] = np.maximum(high, body_hi)
    out[:, 2] = np.minimum(low, body_lo)
    if out.shape[1] > 4:
        out[:, 4:] = np.maximum(out[:, 4:], 0.0)
    return out


@dataclass(frozen=True)
class HierarchicalTokens:
    s1: IntArray
    s2: IntArray
    mean: Array
    std: Array
    normalized: Array


class HierarchicalBSQTokenizer:
    """Fixed-linear hierarchical BSQ tokenizer (research backend)."""

    def __init__(
        self,
        *,
        s1_bits: int = DEFAULT_S1_BITS,
        s2_bits: int = DEFAULT_S2_BITS,
        clip: float = DEFAULT_CLIP,
        seed: int = 42,
    ) -> None:
        if s1_bits < 1 or s2_bits < 1:
            raise ValueError("s1_bits and s2_bits must be positive")
        if not np.isfinite(clip) or clip <= 0:
            raise ValueError("clip must be finite and positive")
        self.s1_bits = int(s1_bits)
        self.s2_bits = int(s2_bits)
        self.codebook_dim = self.s1_bits + self.s2_bits
        self.clip = float(clip)
        self.seed = int(seed)
        self.n_in = len(KLINE_FEATURE_NAMES)
        self.weight = orthonormal_encoder(self.n_in, self.codebook_dim, self.seed)
        self.q_scale = 1.0 / float(np.sqrt(self.codebook_dim))
        self.s1_vocab = 1 << self.s1_bits
        self.s2_vocab = 1 << self.s2_bits

    def normalize(self, kline: Array) -> tuple[Array, Array, Array]:
        x = np.asarray(kline, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.n_in:
            raise ValueError(f"K-line must be (T, {self.n_in})")
        if x.shape[0] < 1:
            raise ValueError("K-line must contain at least one bar")
        if not np.isfinite(x).all():
            raise ValueError("K-line contains non-finite values")
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        normed = (x - mean) / (std + NORM_EPS)
        normed = np.clip(normed, -self.clip, self.clip)
        return normed, mean, std

    def encode(self, kline: Array) -> HierarchicalTokens:
        normed, mean, std = self.normalize(kline)
        latent = normed @ self.weight
        denom = np.linalg.norm(latent, axis=-1, keepdims=True)
        latent = latent / np.clip(denom, NORM_EPS, None)
        s1 = bits_to_indices(latent[:, : self.s1_bits])
        s2 = bits_to_indices(latent[:, self.s1_bits :])
        return HierarchicalTokens(s1=s1, s2=s2, mean=mean, std=std, normalized=normed)

    def decode_normalized(self, s1: IntArray, s2: IntArray) -> Array:
        pre = indices_to_bits(s1, self.s1_bits)
        post = indices_to_bits(s2, self.s2_bits)
        bits = np.concatenate([pre, post], axis=-1)
        latent = bits * self.q_scale
        return latent @ self.weight.T

    def decode(self, s1: IntArray, s2: IntArray, mean: Array, std: Array) -> Array:
        normed = self.decode_normalized(s1, s2)
        frame = normed * (np.asarray(std, dtype=np.float64) + NORM_EPS) + np.asarray(
            mean, dtype=np.float64
        )
        return repair_ohlc(frame)
