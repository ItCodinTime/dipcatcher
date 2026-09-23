"""Kronos-style K-line predictor: normalize, tokenize, sample, invert.

Public output is horizon returns / path quantiles for Dipcatcher fusion — not
a live BUY/SELL signal. The numpy backend implements the two-stage Kronos
framework. The optional torch backend can load official Kronos weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.robinhood_plus.autoregress import HierarchicalMarkovDecoder
from quant_fund.models.robinhood_plus.constants import (
    DEFAULT_CLIP,
    DEFAULT_LOOKBACK,
    DEFAULT_MAX_CONTEXT,
    DEFAULT_PRED_LEN,
    DEFAULT_S1_BITS,
    DEFAULT_S2_BITS,
    DEFAULT_SAMPLE_COUNT,
    ENGINE_NAME,
    ENGINE_VERSION,
    FAMILY,
    KLINE_FEATURE_NAMES,
    PRICE_SPACE,
    STATUS_NONFINITE,
    STATUS_OK,
)
from quant_fund.models.robinhood_plus.tokenizer import HierarchicalBSQTokenizer
from quant_fund.models.robinhood_plus.transformer import TinyHierarchicalTransformer

Array = NDArray[np.float64]
DecoderName = Literal["hierarchical_markov", "transformer"]


@dataclass(frozen=True)
class PathForecast:
    """Sampled K-line paths and derived horizon statistics."""

    paths: Array  # (sample_count, pred_len, 6)
    mean_path: Array  # (pred_len, 6)
    last_close: float
    expected_returns: dict[str, float]
    quantiles: dict[str, dict[float, float]]
    probability_positive: dict[str, float]
    rank_score: float
    confidence: float
    status: str
    diagnostics: dict[str, float | str] = field(default_factory=dict)


def _horizon_name(bars: int) -> str:
    return f"{int(bars)}d"


def _finite_return(end_close: float, last_close: float) -> float:
    if not np.isfinite(end_close) or not np.isfinite(last_close) or last_close <= 0:
        return 0.0
    value = float(end_close / last_close - 1.0)
    return value if np.isfinite(value) else 0.0


class RobinhoodPlusPredictor:
    """In-repo Kronos predictor (robinhood+)."""

    def __init__(
        self,
        *,
        s1_bits: int = DEFAULT_S1_BITS,
        s2_bits: int = DEFAULT_S2_BITS,
        clip: float = DEFAULT_CLIP,
        lookback: int = DEFAULT_LOOKBACK,
        pred_len: int = DEFAULT_PRED_LEN,
        sample_count: int = DEFAULT_SAMPLE_COUNT,
        temperature: float = 1.0,
        top_p: float = 0.9,
        max_context: int = DEFAULT_MAX_CONTEXT,
        seed: int = 42,
        decoder: DecoderName = "hierarchical_markov",
        quantile_levels: tuple[float, ...] = (0.05, 0.50, 0.95),
        horizons: tuple[int, ...] = (1, 5, 20),
    ) -> None:
        if lookback < 2:
            raise ValueError("lookback must be at least 2 bars")
        if pred_len < 1 or sample_count < 1:
            raise ValueError("pred_len and sample_count must be positive")
        if any(not np.isfinite(q) or not 0.0 < q < 1.0 for q in quantile_levels):
            raise ValueError("quantile_levels must be finite and in (0, 1)")
        if any(h < 1 for h in horizons):
            raise ValueError("horizons must be positive")
        self.tokenizer = HierarchicalBSQTokenizer(
            s1_bits=s1_bits, s2_bits=s2_bits, clip=clip, seed=seed
        )
        self.lookback = int(lookback)
        self.pred_len = int(pred_len)
        self.sample_count = int(sample_count)
        self.max_context = int(max_context)
        self.seed = int(seed)
        self.decoder_name = decoder
        self.quantile_levels = tuple(float(q) for q in quantile_levels)
        self.horizons = tuple(int(h) for h in horizons)
        if decoder == "transformer":
            self._decoder: HierarchicalMarkovDecoder | TinyHierarchicalTransformer = (
                TinyHierarchicalTransformer(
                    self.tokenizer,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                )
            )
        elif decoder == "hierarchical_markov":
            self._decoder = HierarchicalMarkovDecoder(
                self.tokenizer,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
            )
        else:
            raise ValueError(f"unknown robinhood+ decoder {decoder!r}")

    def predict_kline(self, kline: Array) -> PathForecast:
        x = np.asarray(kline, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(KLINE_FEATURE_NAMES):
            raise ValueError(f"K-line must be (T, {len(KLINE_FEATURE_NAMES)})")
        if x.shape[0] < 2:
            raise ValueError("K-line lookback must contain at least two bars")
        if not np.isfinite(x).all():
            raise ValueError("K-line contains non-finite values")
        window = x[-self.lookback :]
        tokens = self.tokenizer.encode(window)
        if isinstance(self._decoder, HierarchicalMarkovDecoder):
            self._decoder.fit(tokens)
            gen_s1, gen_s2 = self._decoder.generate(
                tokens, self.pred_len, sample_count=self.sample_count, seed=self.seed
            )
        else:
            gen_s1, gen_s2 = self._decoder.generate(
                tokens,
                self.pred_len,
                sample_count=self.sample_count,
                seed=self.seed,
                max_context=self.max_context,
            )
        paths = np.empty((self.sample_count, self.pred_len, len(KLINE_FEATURE_NAMES)))
        for i in range(self.sample_count):
            paths[i] = self.tokenizer.decode(gen_s1[i], gen_s2[i], tokens.mean, tokens.std)
        if not np.isfinite(paths).all():
            raise ValueError("robinhood+ decoded paths must be finite")
        last_close = float(window[-1, 3])
        return path_forecast_from_paths(
            paths,
            last_close,
            quantile_levels=self.quantile_levels,
            horizons=self.horizons,
            diagnostics={
                "engine": ENGINE_NAME,
                "engine_version": ENGINE_VERSION,
                "family": FAMILY,
                "backend": "numpy",
                "decoder": self.decoder_name,
                "price_space": PRICE_SPACE,
                "lookback": float(window.shape[0]),
                "pred_len": float(self.pred_len),
                "sample_count": float(self.sample_count),
                "s1_bits": float(self.tokenizer.s1_bits),
                "s2_bits": float(self.tokenizer.s2_bits),
                "last_close": last_close,
            },
        )


def path_forecast_from_paths(
    paths: Array,
    last_close: float,
    *,
    quantile_levels: tuple[float, ...] = (0.05, 0.50, 0.95),
    horizons: tuple[int, ...] = (1, 5, 20),
    diagnostics: dict[str, float | str] | None = None,
) -> PathForecast:
    """Horizon returns / quantiles from an ensemble of K-line paths."""
    x = np.asarray(paths, dtype=np.float64)
    if x.ndim != 3 or x.shape[0] < 1 or x.shape[1] < 1 or x.shape[2] < 4:
        return empty_forecast(STATUS_NONFINITE)
    if not np.isfinite(x).all() or not np.isfinite(last_close) or last_close <= 0:
        return empty_forecast(STATUS_NONFINITE)
    sample_count, pred_len, _ = x.shape
    mean_path = np.mean(x, axis=0)
    expected: dict[str, float] = {}
    quantiles: dict[str, dict[float, float]] = {}
    p_pos: dict[str, float] = {}
    for horizon in horizons:
        if horizon > pred_len:
            continue
        idx = horizon - 1
        rets = np.array(
            [_finite_return(float(x[i, idx, 3]), last_close) for i in range(sample_count)],
            dtype=np.float64,
        )
        name = _horizon_name(horizon)
        expected[name] = float(np.mean(rets))
        quantiles[name] = {level: float(np.quantile(rets, level)) for level in quantile_levels}
        p_pos[name] = float(np.mean(rets > 0.0))
    if not expected:
        return empty_forecast(STATUS_NONFINITE)
    if "5d" in expected:
        rank_score = expected["5d"]
        conf_idx = min(4, pred_len - 1)
    else:
        rank_score = next(iter(expected.values()))
        conf_idx = pred_len - 1
    conf_rets = np.array(
        [_finite_return(float(x[i, conf_idx, 3]), last_close) for i in range(sample_count)],
        dtype=np.float64,
    )
    dispersion = float(np.std(conf_rets))
    confidence = float(np.clip(1.0 / (1.0 + dispersion * 20.0), 0.2, 1.0))
    extra: dict[str, float | str] = dict(diagnostics or {})
    extra.setdefault("engine", ENGINE_NAME)
    extra.setdefault("engine_version", ENGINE_VERSION)
    extra.setdefault("family", FAMILY)
    extra.setdefault("price_space", PRICE_SPACE)
    extra["last_close"] = float(last_close)
    extra["path_dispersion"] = dispersion
    extra["pred_len"] = float(pred_len)
    extra["sample_count"] = float(sample_count)
    return PathForecast(
        paths=x,
        mean_path=mean_path,
        last_close=float(last_close),
        expected_returns=expected,
        quantiles=quantiles,
        probability_positive=p_pos,
        rank_score=float(rank_score),
        confidence=confidence,
        status=STATUS_OK,
        diagnostics=extra,
    )


def empty_forecast(status: str) -> PathForecast:
    """Structured miss so callers can fail closed without inventing a path."""
    if status == STATUS_OK:
        raise ValueError("empty_forecast cannot stamp status=ok")
    return PathForecast(
        paths=np.zeros((0, 0, len(KLINE_FEATURE_NAMES))),
        mean_path=np.zeros((0, len(KLINE_FEATURE_NAMES))),
        last_close=float("nan"),
        expected_returns={},
        quantiles={},
        probability_positive={},
        rank_score=0.0,
        confidence=0.2,
        status=status,
        diagnostics={"engine": ENGINE_NAME, "status": status, "backend": "numpy"},
    )
