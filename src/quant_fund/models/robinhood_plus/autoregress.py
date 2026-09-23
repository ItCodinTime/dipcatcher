"""Hierarchical autoregressive decoding on (s1, s2) K-line tokens.

Kronos samples coarse s1 then fine s2 conditioned on s1 (Shi et al., 2025).
The default research decoder is a Laplace-smoothed hierarchical Markov model
fit on the causal lookback — used when pretrained transformer weights are
absent. Optional ``transformer`` decoding lives in ``transformer.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.robinhood_plus.tokenizer import HierarchicalBSQTokenizer, HierarchicalTokens

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


def nucleus_sample(
    log_counts: FloatArray,
    rng: np.random.Generator,
    *,
    temperature: float,
    top_p: float,
) -> int:
    """Sample one token from Laplace counts with temperature and nucleus filter."""
    if log_counts.ndim != 1 or log_counts.size < 1:
        raise ValueError("log_counts must be a non-empty vector")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not np.isfinite(top_p) or not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be finite and in (0, 1]")
    logits = np.asarray(log_counts, dtype=np.float64) / float(temperature)
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0:
        probs = np.full(logits.size, 1.0 / logits.size)
    else:
        probs = probs / total
    if top_p < 1.0:
        order = np.argsort(probs)[::-1]
        cumulative = np.cumsum(probs[order])
        keep = cumulative <= top_p
        keep[0] = True
        mask = np.zeros_like(probs, dtype=bool)
        mask[order[keep]] = True
        probs = np.where(mask, probs, 0.0)
        denom = float(probs.sum())
        if denom <= 0:
            probs = np.full(logits.size, 1.0 / logits.size)
        else:
            probs = probs / denom
    return int(rng.choice(probs.size, p=probs))


@dataclass
class HierarchicalMarkovDecoder:
    """Order-1 hierarchical Markov language model on BSQ tokens."""

    tokenizer: HierarchicalBSQTokenizer
    temperature: float = 1.0
    top_p: float = 0.9
    seed: int = 42

    def __post_init__(self) -> None:
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not np.isfinite(self.top_p) or not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be finite and in (0, 1]")
        self.s1_vocab = self.tokenizer.s1_vocab
        self.s2_vocab = self.tokenizer.s2_vocab
        self._s1_from_ctx: dict[tuple[int, int], FloatArray] = {}
        self._s2_from_ctx: dict[tuple[int, int, int], FloatArray] = {}
        self._s1_unigram = np.ones(self.s1_vocab, dtype=np.float64)
        self._s2_unigram = np.ones(self.s2_vocab, dtype=np.float64)
        self._fitted = False

    def _s1_counts(self, prev_s1: int, prev_s2: int) -> FloatArray:
        key = (prev_s1, prev_s2)
        counts = self._s1_from_ctx.get(key)
        return self._s1_unigram if counts is None else counts

    def _s2_counts(self, prev_s1: int, prev_s2: int, cur_s1: int) -> FloatArray:
        key = (prev_s1, prev_s2, cur_s1)
        counts = self._s2_from_ctx.get(key)
        return self._s2_unigram if counts is None else counts

    def fit(self, tokens: HierarchicalTokens) -> HierarchicalMarkovDecoder:
        s1 = np.asarray(tokens.s1, dtype=np.int64).reshape(-1)
        s2 = np.asarray(tokens.s2, dtype=np.int64).reshape(-1)
        if s1.size != s2.size or s1.size < 1:
            raise ValueError("s1/s2 token streams must be aligned and non-empty")
        if np.any(s1 < 0) or np.any(s1 >= self.s1_vocab):
            raise ValueError("s1 token out of vocabulary")
        if np.any(s2 < 0) or np.any(s2 >= self.s2_vocab):
            raise ValueError("s2 token out of vocabulary")
        self._s1_from_ctx = {}
        self._s2_from_ctx = {}
        self._s1_unigram = np.ones(self.s1_vocab, dtype=np.float64)
        self._s2_unigram = np.ones(self.s2_vocab, dtype=np.float64)
        for token in s1:
            self._s1_unigram[int(token)] += 1.0
        for token in s2:
            self._s2_unigram[int(token)] += 1.0
        for i in range(1, s1.size):
            prev_s1 = int(s1[i - 1])
            prev_s2 = int(s2[i - 1])
            cur_s1 = int(s1[i])
            cur_s2 = int(s2[i])
            s1_key = (prev_s1, prev_s2)
            s1_counts = self._s1_from_ctx.get(s1_key)
            if s1_counts is None:
                s1_counts = np.ones(self.s1_vocab, dtype=np.float64)
                self._s1_from_ctx[s1_key] = s1_counts
            s1_counts[cur_s1] += 1.0
            s2_key = (prev_s1, prev_s2, cur_s1)
            s2_counts = self._s2_from_ctx.get(s2_key)
            if s2_counts is None:
                s2_counts = np.ones(self.s2_vocab, dtype=np.float64)
                self._s2_from_ctx[s2_key] = s2_counts
            s2_counts[cur_s2] += 1.0
        self._fitted = True
        return self

    def generate(
        self,
        tokens: HierarchicalTokens,
        pred_len: int,
        *,
        sample_count: int = 1,
        seed: int | None = None,
    ) -> tuple[IntArray, IntArray]:
        if not self._fitted:
            raise ValueError("decoder must be fit before generate")
        if pred_len < 1 or sample_count < 1:
            raise ValueError("pred_len and sample_count must be positive")
        s1 = np.asarray(tokens.s1, dtype=np.int64).reshape(-1)
        s2 = np.asarray(tokens.s2, dtype=np.int64).reshape(-1)
        rng = np.random.default_rng(self.seed if seed is None else int(seed))
        out_s1 = np.empty((sample_count, pred_len), dtype=np.int64)
        out_s2 = np.empty((sample_count, pred_len), dtype=np.int64)
        for sample in range(sample_count):
            cur_s1 = int(s1[-1])
            cur_s2 = int(s2[-1])
            for step in range(pred_len):
                next_s1 = nucleus_sample(
                    np.log(self._s1_counts(cur_s1, cur_s2)),
                    rng,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                next_s2 = nucleus_sample(
                    np.log(self._s2_counts(cur_s1, cur_s2, next_s1)),
                    rng,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                out_s1[sample, step] = next_s1
                out_s2[sample, step] = next_s2
                cur_s1, cur_s2 = next_s1, next_s2
        return out_s1, out_s2
