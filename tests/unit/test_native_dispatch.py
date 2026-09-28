"""Coverage: native/__init__ dispatch layer.

``quant_core`` is absent in this environment, so the python path is what
runs; the dispatch logic (flag parsing, array hygiene, validation) still
gets exercised.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

import quant_fund.native as native
from quant_fund.native import reference as ref


def test_flag_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUANT_FUND_NATIVE", "")
    assert native._flag() == "auto"
    monkeypatch.setenv("QUANT_FUND_NATIVE", " AUTO ")
    assert native._flag() == "auto"
    for raw in ("python", "py", "numpy", "PyThOn"):
        monkeypatch.setenv("QUANT_FUND_NATIVE", raw)
        assert native._flag() == "python"
    for raw in ("rust", "native", "RUST"):
        monkeypatch.setenv("QUANT_FUND_NATIVE", raw)
        assert native._flag() == "rust"
    monkeypatch.setenv("QUANT_FUND_NATIVE", "bogus")
    with pytest.raises(ValueError, match="QUANT_FUND_NATIVE"):
        native._flag()


def test_load_rust_fallback() -> None:
    assert native._load_rust("python") is None
    # quant_core is not installed here: auto degrades to None
    try:
        import quant_core  # noqa: F401

        has_rust = True
    except ImportError:
        has_rust = False
    if not has_rust:
        assert native._load_rust("auto") is None
        with pytest.raises(ImportError, match="QUANT_FUND_NATIVE=rust"):
            native._load_rust("rust")


def test_width_bounds() -> None:
    assert native._width(5) == 5
    assert native._width(-5) == -5  # in range; the kernel decides
    assert native._width(2**63) is None
    assert native._width(-(2**63) - 1) is None


def test_f64_hygiene() -> None:
    arr = native._f64([1, 2, 3], "x")
    assert arr.dtype == np.float64 and arr.flags["C_CONTIGUOUS"]
    arr2 = native._f64(np.ones((2, 4)), "x")
    assert arr2.shape == (2, 4)
    with pytest.raises(ValueError, match="1-d series or a 2-d"):
        native._f64(np.ones((2, 2, 2)), "x")


def test_rolling_kernels_match_reference() -> None:
    x = np.linspace(1.0, 5.0, 40) ** 2
    np.testing.assert_allclose(native.rolling_mean(x, 5), ref.rolling_mean(x, 5))
    np.testing.assert_allclose(native.rolling_std(x, 7), ref.rolling_std(x, 7))
    np.testing.assert_allclose(native.ema(x, 10), ref.ema(x, 10))
    np.testing.assert_allclose(native.rsi(x, 14), ref.rsi(x, 14))
    native_2d = native.rolling_mean(np.stack([x, x * 2]), 5)
    if native.BACKEND == "python":
        ref_2d = ref.rolling_mean(np.stack([x, x * 2]), 5)
        np.testing.assert_allclose(native_2d, ref_2d)


def test_bollinger_dict() -> None:
    x = np.sin(np.linspace(0, 6, 60)) + 5
    out = native.bollinger(x, window=10, num_sd=1.5)
    ref_out = ref.bollinger(x, 10, 1.5)
    assert set(out) == set(ref_out)
    for key in out:
        np.testing.assert_allclose(out[key], ref_out[key])


def test_returns_and_wealth() -> None:
    px = np.array([100.0, 101.0, 99.0, 102.0])
    np.testing.assert_allclose(native.simple_returns(px), ref.simple_returns(px))
    np.testing.assert_allclose(
        native.wealth_index(native.simple_returns(px)),
        ref.wealth_index(ref.simple_returns(px)),
    )
    panel = np.stack([px, px * 1.1])
    np.testing.assert_allclose(native.simple_returns(panel), ref.simple_returns(panel))


def test_turnover() -> None:
    w = np.array([0.6, 0.4])
    p = np.array([0.5, 0.5])
    assert native.turnover(w, p) == pytest.approx(ref.turnover(w, p))
    series = np.array([[0.5, 0.5], [0.7, 0.3], [0.7, 0.3]])
    np.testing.assert_allclose(native.turnover_series(series), ref.turnover_series(series))


def test_book_features() -> None:
    rng = np.random.default_rng(1)
    bid_px = 100.0 + rng.normal(0, 0.1, (2, 3))
    bid_sz = rng.uniform(100, 500, (2, 3))
    ask_px = bid_px + 0.05
    ask_sz = rng.uniform(100, 500, (2, 3))
    out = native.book_features(bid_px, bid_sz, ask_px, ask_sz)
    ref_out = ref.book_features(bid_px, bid_sz, ask_px, ask_sz)
    assert set(out) == set(ref_out)
    for key in out:
        np.testing.assert_allclose(out[key], ref_out[key])


def test_hash_parity() -> None:
    blob = b"dipcatcher"
    assert native.hash_bytes(blob) == hashlib.sha256(blob).hexdigest()
    assert native.hash_bytes(bytearray(blob)) == hashlib.sha256(blob).hexdigest()
    assert native.hash_bytes(memoryview(blob)) == hashlib.sha256(blob).hexdigest()
    chunks = [b"a", b"ab", b"abc"]
    assert native.hash_many(chunks) == [hashlib.sha256(c).hexdigest() for c in chunks]


def test_backend_reported() -> None:
    assert native.BACKEND in {"python", "rust"}
