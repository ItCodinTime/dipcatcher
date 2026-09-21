"""Small deterministic hierarchical bit tokenizer used by the NumPy research backend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def indices_to_bits(indices: np.ndarray, n_bits: int) -> np.ndarray:
    """Encode non-negative integer indices as a final-axis MSB-first bit array."""
    if n_bits < 1 or n_bits > 16:
        raise ValueError("n_bits must be in [1, 16]")
    arr = np.asarray(indices, dtype=np.int64)
    if np.any(arr < 0) or np.any(arr >= 2**n_bits):
        raise ValueError("indices are outside the requested bit width")
    shifts = np.arange(n_bits - 1, -1, -1, dtype=np.int64)
    return ((arr[..., None] >> shifts) & 1).astype(np.uint8)


def bits_to_indices(bits: np.ndarray) -> np.ndarray:
    """Decode an MSB-first final-axis bit array."""
    arr = np.asarray(bits)
    if arr.ndim < 1 or arr.shape[-1] < 1 or arr.shape[-1] > 16:
        raise ValueError("bits must have a final axis in [1, 16]")
    if np.any((arr != 0) & (arr != 1)):
        raise ValueError("bits must contain only zero and one")
    weights = 2 ** np.arange(arr.shape[-1] - 1, -1, -1, dtype=np.int64)
    return np.asarray(arr, dtype=np.int64) @ weights


def repair_ohlc(values: np.ndarray) -> np.ndarray:
    """Repair invalid OHLCV rows without introducing non-finite values."""
    out = np.asarray(values, dtype=float).copy()
    if out.ndim != 2 or out.shape[1] != 6:
        raise ValueError("K-line values must have shape (n, 6)")
    out[~np.isfinite(out)] = 0.0
    out[:, :4] = np.maximum(out[:, :4], 1e-12)
    out[:, 1] = np.maximum(out[:, 1], np.maximum(out[:, 0], out[:, 3]))
    out[:, 2] = np.minimum(out[:, 2], np.minimum(out[:, 0], out[:, 3]))
    out[:, 4:] = np.maximum(out[:, 4:], 0.0)
    return out


@dataclass(frozen=True)
class TokenBatch:
    s1: np.ndarray
    s2: np.ndarray
    mean: np.ndarray
    std: np.ndarray


class HierarchicalBSQTokenizer:
    """Feature-wise bounded scalar quantizer; seed retained for API compatibility."""

    def __init__(self, s1_bits: int = 5, s2_bits: int = 5, seed: int = 42) -> None:
        if not 1 <= s1_bits <= 8 or not 1 <= s2_bits <= 8:
            raise ValueError("s1_bits and s2_bits must be in [1, 8]")
        self.s1_bits, self.s2_bits, self.seed = int(s1_bits), int(s2_bits), int(seed)

    def encode(self, values: np.ndarray) -> TokenBatch:
        x = repair_ohlc(values)
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        std = np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)
        z = np.clip((x - mean) / std, -6.0, 6.0)
        # Two deterministic residual quantizers. Their combination is not used as
        # a lossy packed code; retaining both levels makes the decoder stable.
        s1 = np.rint((z + 6.0) / 12.0 * (2**self.s1_bits - 1)).astype(np.int64)
        s2 = np.rint((z + 6.0) / 12.0 * (2**self.s2_bits - 1)).astype(np.int64)
        return TokenBatch(
            indices_to_bits(s1, self.s1_bits), indices_to_bits(s2, self.s2_bits), mean, std
        )

    def decode(
        self, s1: np.ndarray, s2: np.ndarray, mean: np.ndarray, std: np.ndarray
    ) -> np.ndarray:
        a = bits_to_indices(s1) / max(2**self.s1_bits - 1, 1)
        b = bits_to_indices(s2) / max(2**self.s2_bits - 1, 1)
        z = ((a + b) * 0.5) * 12.0 - 6.0
        x = z * np.asarray(std, dtype=float) + np.asarray(mean, dtype=float)
        return repair_ohlc(x)
