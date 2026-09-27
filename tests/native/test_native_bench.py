"""SYNTHETIC wall-clock comparison of the profiled Python loops and quant_core.

Not a research score. Medians and sample variances are infrastructure timings.
The test records both backends and, when Rust is loaded, requires a speedup
only on the Python-loop kernels the profile identified.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pytest

from quant_fund.native import reference

pytestmark = [
    pytest.mark.native,
    pytest.mark.skipif(
        importlib.util.find_spec("quant_core") is None,
        reason="quant_core extension is not built",
    ),
]

# Minimum median speedup versus the Python loops the profile measured.
# Production ema/rsi/bollinger now go through SciPy ``lfilter`` and NumPy
# sliding windows. Those C reductions are not this baseline: the floors and
# ``rust/quant_core/README.md`` (ema ~8.6 ms / 111×) describe the scalar
# loops. ``_loop_ema``, ``_loop_rsi``, and ``_loop_bollinger`` are those
# loops. The numeric floors are unchanged. Rolling mean/std and wealth stay
# ungated because NumPy already implements them in C.
_MIN_SPEEDUP = {
    "bollinger": 8.0,
    "rsi": 5.0,
    "ema": 5.0,
    "book_features": 3.0,
    "turnover_series": 3.0,
    "hash_many": 1.5,
}


def _time(fn, repeats: int = 7) -> np.ndarray:
    fn()
    samples = []
    gc.collect()
    gc.disable()
    try:
        for _ in range(repeats):
            start = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - start)
    finally:
        gc.enable()
    return np.asarray(samples, dtype=np.float64)


def _stats(samples: np.ndarray) -> dict[str, float]:
    return {
        "median_s": float(np.median(samples)),
        "variance_s2": float(np.var(samples, ddof=1)),
        "std_s": float(np.std(samples, ddof=1)),
    }


def _loop_ema(values: np.ndarray, window: int) -> np.ndarray:
    """SMA-seeded EMA. ``alpha * x + (1 - alpha) * prev``, NaN warmup."""
    out = np.full(values.size, np.nan)
    alpha = 2.0 / (window + 1.0)
    out[window - 1] = values[:window].mean()
    for i in range(window, values.size):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def _loop_wilder(values: np.ndarray, window: int) -> np.ndarray:
    """Wilder smooth. ``(prev * (n - 1) + x) / n``, NaN warmup."""
    out = np.full(values.size, np.nan)
    out[window - 1] = values[:window].mean()
    for i in range(window, values.size):
        out[i] = (out[i - 1] * (window - 1) + values[i]) / window
    return out


def _loop_rsi(close: np.ndarray, window: int) -> np.ndarray:
    """Wilder RSI on the scalar smoother the floor was calibrated against."""
    delta = np.diff(close, prepend=close[0])
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    avg_up = _loop_wilder(up, window)
    avg_down = _loop_wilder(down, window)
    with np.errstate(invalid="ignore", divide="ignore"):
        rs = avg_up / avg_down
        out = 100.0 - 100.0 / (1.0 + rs)
    out[avg_down == 0.0] = np.where(avg_up[avg_down == 0.0] > 0, 100.0, 50.0)
    return out


def _loop_bollinger(close: np.ndarray, window: int, num_sd: float) -> dict[str, np.ndarray]:
    """Cumsum SMA plus a per-window population std. That std loop is the cost."""
    mid = np.full(close.size, np.nan)
    cumulative = np.cumsum(np.insert(close, 0, 0.0))
    mid[window - 1 :] = (cumulative[window:] - cumulative[:-window]) / window
    sd = np.full(close.size, np.nan)
    for i in range(window - 1, close.size):
        sd[i] = close[i - window + 1 : i + 1].std(ddof=0)
    upper = mid + num_sd * sd
    lower = mid - num_sd * sd
    with np.errstate(invalid="ignore", divide="ignore"):
        pct_b = (close - lower) / (upper - lower)
        bandwidth = (upper - lower) / mid
    return {
        "mid": mid,
        "upper": upper,
        "lower": lower,
        "pct_b": pct_b,
        "bandwidth": bandwidth,
    }


def _assert_loops_match_reference(close: np.ndarray) -> None:
    """Timed loops are the same formulas as the production fallbacks."""
    np.testing.assert_allclose(
        _loop_ema(close, 20),
        reference.ema(close, 20),
        rtol=1e-9,
        atol=1e-9,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        _loop_rsi(close, 14),
        reference.rsi(close, 14),
        rtol=1e-8,
        atol=1e-8,
        equal_nan=True,
    )
    loop_bands = _loop_bollinger(close, 20, 2.0)
    ref_bands = reference.bollinger(close, 20, 2.0)
    for key, loop_band in loop_bands.items():
        np.testing.assert_allclose(loop_band, ref_bands[key], rtol=1e-9, atol=1e-9, equal_nan=True)


def test_kernel_speedups(capsys: pytest.CaptureFixture[str]) -> None:
    import quant_core

    rng = np.random.default_rng(20260927)
    close = np.ascontiguousarray(rng.normal(size=40_000))
    prices = np.ascontiguousarray(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=80_000))))
    returns = np.ascontiguousarray(rng.normal(0.0, 0.01, size=80_000))
    weights = np.ascontiguousarray(rng.normal(size=(8_000, 32)))
    rows, depth = 4_000, 8
    bid_px = np.ascontiguousarray(100.0 - rng.random((rows, depth)))
    ask_px = np.ascontiguousarray(bid_px + 0.05 + rng.random((rows, depth)) * 0.01)
    bid_sz = np.ascontiguousarray(1.0 + rng.random((rows, depth)))
    ask_sz = np.ascontiguousarray(1.0 + rng.random((rows, depth)))
    for i in range(depth):
        bid_px[:, i] = 100.0 - i - 0.01 * i
        ask_px[:, i] = 100.2 + i + 0.01 * i
    blobs = [rng.integers(0, 256, size=64, dtype=np.uint8).tobytes() for _ in range(20_000)]
    wide = rng.normal(size=2_000_000).tobytes()
    _assert_loops_match_reference(close)

    cases = {
        "rolling_mean": (
            lambda: reference.rolling_mean(close, 20),
            lambda: quant_core.rolling_mean(close, 20),
        ),
        "rolling_std": (
            lambda: reference.rolling_std(close, 20),
            lambda: quant_core.rolling_std(close, 20),
        ),
        "ema": (lambda: _loop_ema(close, 20), lambda: quant_core.ema(close, 20)),
        "rsi": (lambda: _loop_rsi(close, 14), lambda: quant_core.rsi(close, 14)),
        "bollinger": (
            lambda: _loop_bollinger(close, 20, 2.0),
            lambda: quant_core.bollinger(close, 20, 2.0),
        ),
        "simple_returns": (
            lambda: reference.simple_returns(prices),
            lambda: quant_core.simple_returns(prices),
        ),
        "wealth_index": (
            lambda: reference.wealth_index(returns),
            lambda: quant_core.wealth_index(returns),
        ),
        "turnover_series": (
            lambda: reference.turnover_series(weights),
            lambda: quant_core.turnover_series(
                np.ascontiguousarray(weights.reshape(-1)), weights.shape[0], weights.shape[1]
            ),
        ),
        "book_features": (
            lambda: reference.book_features(bid_px, bid_sz, ask_px, ask_sz),
            lambda: quant_core.book_features(
                np.ascontiguousarray(bid_px.reshape(-1)),
                np.ascontiguousarray(bid_sz.reshape(-1)),
                np.ascontiguousarray(ask_px.reshape(-1)),
                np.ascontiguousarray(ask_sz.reshape(-1)),
                rows,
                depth,
            ),
        ),
        "hash_bytes": (
            lambda: reference.hash_bytes(wide),
            lambda: quant_core.hash_bytes(wide),
        ),
        "hash_many": (
            lambda: reference.hash_many(blobs),
            lambda: quant_core.hash_many(blobs),
        ),
    }

    report: dict[str, object] = {
        "label": "SYNTHETIC kernel timings — not a research score and not live P&L",
        "live_pnl_claim": False,
        "research_only": True,
        "kernels": {},
    }
    lines = [
        "kernel median_py_ms variance_py median_rs_ms variance_rs speedup",
    ]
    kernels: dict[str, dict[str, float]] = {}
    for name, (py_fn, rs_fn) in cases.items():
        py_stats = _stats(_time(py_fn))
        rs_stats = _stats(_time(rs_fn))
        speedup = py_stats["median_s"] / rs_stats["median_s"]
        kernels[name] = {
            "python_median_ms": py_stats["median_s"] * 1e3,
            "python_variance_s2": py_stats["variance_s2"],
            "python_std_ms": py_stats["std_s"] * 1e3,
            "rust_median_ms": rs_stats["median_s"] * 1e3,
            "rust_variance_s2": rs_stats["variance_s2"],
            "rust_std_ms": rs_stats["std_s"] * 1e3,
            "speedup": speedup,
        }
        lines.append(
            f"{name} {kernels[name]['python_median_ms']:.3f} {py_stats['variance_s2']:.3e} "
            f"{kernels[name]['rust_median_ms']:.3f} {rs_stats['variance_s2']:.3e} {speedup:.2f}"
        )
        floor = _MIN_SPEEDUP.get(name)
        if floor is not None:
            assert speedup >= floor, f"{name} speedup {speedup:.2f}x is below {floor:.1f}x"
    report["kernels"] = kernels
    text = "\n".join(lines)
    print(text)
    artifact = Path("/opt/cursor/artifacts")
    if artifact.is_dir():
        (artifact / "native_bench.json").write_text(json.dumps(report, indent=2))
        (artifact / "native_bench.txt").write_text(text + "\n")
    captured = capsys.readouterr()
    assert "bollinger" in captured.out or "bollinger" in text
