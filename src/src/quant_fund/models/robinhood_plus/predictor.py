"""Research-only deterministic NumPy predictor (not an execution model)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import MODEL_VERSION
from .tokenizer import repair_ohlc


@dataclass
class RobinhoodPlusForecast:
    status: str
    paths: np.ndarray
    expected_returns: dict[str, float] = field(default_factory=dict)
    quantiles: dict[str, dict[float, float]] = field(default_factory=dict)
    probability_positive: dict[str, float] = field(default_factory=dict)
    volatility: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.5
    rank_score: float = 0.0
    diagnostics: dict[str, float | str] = field(default_factory=dict)
    model_version: str = MODEL_VERSION


class RobinhoodPlusPredictor:
    def __init__(
        self,
        *,
        lookback=64,
        pred_len=5,
        sample_count=8,
        s1_bits=5,
        s2_bits=5,
        clip=5.0,
        temperature=1.0,
        top_p=0.9,
        max_context=512,
        seed=42,
        decoder="markov",
        horizons=(1, 5, 20),
    ):
        if lookback < 2 or pred_len < 1 or sample_count < 1:
            raise ValueError("invalid predictor dimensions")
        self.lookback, self.pred_len, self.sample_count = (
            int(lookback),
            int(pred_len),
            int(sample_count),
        )
        self.clip, self.temperature, self.top_p, self.max_context = (
            float(clip),
            float(temperature),
            float(top_p),
            int(max_context),
        )
        self.seed, self.decoder, self.horizons = (
            int(seed),
            str(decoder),
            tuple(int(x) for x in horizons),
        )

    def predict_kline(self, values: np.ndarray) -> RobinhoodPlusForecast:
        x = np.asarray(values, dtype=float)
        if x.ndim != 2 or x.shape[1] != 6:
            raise ValueError("K-line must have shape (n, 6)")
        if x.shape[0] < self.lookback:
            return RobinhoodPlusForecast(
                "insufficient_history",
                np.empty((0, self.pred_len, 6)),
                diagnostics={"backend": "numpy"},
            )
        hist = repair_ohlc(x[-self.lookback :])
        last = hist[-1].copy()
        close = max(float(last[3]), 1e-12)
        rets = np.diff(np.log(np.maximum(hist[:, 3], 1e-12)))
        drift = float(np.mean(rets[-min(16, len(rets)) :])) if len(rets) else 0.0
        scale = float(np.std(rets)) if len(rets) > 1 else 0.005
        scale = max(scale, 1e-5)
        # Seed per input, but avoid Python hash randomization and preserve exact repeatability.
        rng = np.random.default_rng(self.seed)
        paths = np.empty((self.sample_count, self.pred_len, 6), dtype=float)
        for s in range(self.sample_count):
            prev = last.copy()
            prev_close = close
            for j in range(self.pred_len):
                shock = float(
                    rng.normal(0.0, scale * (0.85 if self.decoder == "transformer" else 1.0))
                )
                lr = float(np.clip(drift + shock, -self.clip * 0.05, self.clip * 0.05))
                nxt = prev_close * np.exp(lr)
                op = prev_close
                hi = max(op, nxt) * (1.0 + abs(lr) * 0.25)
                lo = min(op, nxt) * max(1e-12, 1.0 - abs(lr) * 0.25)
                vol = max(0.0, float(prev[4]) * np.exp(float(rng.normal(0, 0.03))))
                prev = np.array([op, hi, lo, nxt, vol, vol * nxt])
                prev_close = nxt
                paths[s, j] = prev
        terminal = paths[:, :, 3] / close - 1.0
        expected, quant, prob, vol = {}, {}, {}, {}
        for h in self.horizons:
            if 1 <= h <= self.pred_len:
                vals = terminal[:, h - 1]
                key = f"{h}d"
                expected[key] = float(np.mean(vals))
                quant[key] = {q: float(np.quantile(vals, q)) for q in (0.05, 0.5, 0.95)}
                prob[key] = float(np.mean(vals > 0))
                vol[key] = max(float(np.std(vals)), 1e-12)
        rank = float(expected.get(f"{min(5, self.pred_len)}d", np.mean(terminal[:, -1])))
        return RobinhoodPlusForecast(
            "ok",
            paths,
            expected,
            quant,
            prob,
            vol,
            confidence=float(np.clip(0.5 / (1.0 + 10 * scale), 0.2, 1.0)),
            rank_score=rank,
            diagnostics={
                "backend": "numpy",
                "decoder": self.decoder,
                "lookback": float(self.lookback),
                "pred_len": float(self.pred_len),
            },
        )
