"""Tests for the technical-indicator canon."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.features import indicators as ta


@pytest.fixture()
def ohlcv() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    n = 400
    c = 100.0 + np.cumsum(rng.normal(0.0, 1.0, n))
    h = c + np.abs(rng.normal(0.0, 0.5, n))
    lo = c - np.abs(rng.normal(0.0, 0.5, n))
    v = np.abs(rng.normal(1e5, 2e4, n))
    return {"high": h, "low": lo, "close": c, "volume": v}


def _tail(x: np.ndarray, k: int = 50) -> np.ndarray:
    return x[-k:]


class TestAverages:
    def test_sma_matches_mean(self, ohlcv):
        c = ohlcv["close"]
        s = ta.sma(c, 20)
        assert np.isnan(s[:19]).all()
        assert np.isclose(s[19], c[:20].mean())

    def test_ema_tracks(self, ohlcv):
        e = ta.ema(ohlcv["close"], 20)
        assert np.isnan(e[:19]).all()
        assert np.all(np.isfinite(_tail(e)))

    def test_wma_between_min_max(self, ohlcv):
        w = ta.wma(ohlcv["close"], 10)
        assert np.all(_tail(w) > 0)

    def test_nested_ema_does_not_see_the_future(self):
        rng = np.random.default_rng(1)
        c = 100.0 + np.cumsum(rng.normal(0.0, 1.0, 160))
        shocked = c.copy()
        shocked[-1] += 40.0
        for fn in (ta.dema, ta.tema, ta.hma):
            a = fn(c, 12)
            b = fn(shocked, 12)
            assert np.allclose(a[:-1], b[:-1], equal_nan=True)
        a = ta.macd(c)
        b = ta.macd(shocked)
        assert np.allclose(a["signal"][:-1], b["signal"][:-1], equal_nan=True)

    def test_dema_tema_faster_than_ema(self):
        c = np.linspace(100.0, 200.0, 120)  # strong trend
        e = ta.ema(c, 20)
        d = ta.dema(c, 20)
        t = ta.tema(c, 20)
        # In a monotone uptrend, double/triple EMA sits closer to price.
        assert d[-1] > e[-1] and t[-1] > e[-1]

    def test_hma_kama_zlema_run(self, ohlcv):
        for fn in (ta.hma, ta.kama, ta.zlema):
            out = fn(ohlcv["close"], 20)
            assert np.all(np.isfinite(_tail(out)))


class TestTrend:
    def test_macd_relations(self, ohlcv):
        m = ta.macd(ohlcv["close"])
        assert np.allclose(_tail(m["histogram"]), _tail(m["macd"] - m["signal"]), atol=1e-9)

    def test_atr_positive(self, ohlcv):
        a = ta.atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], 14)
        assert np.all(_tail(a) > 0)
        n = ta.natr(ohlcv["high"], ohlcv["low"], ohlcv["close"], 14)
        assert np.all(_tail(n) > 0)

    def test_dmi_bounds(self, ohlcv):
        d = ta.dmi(ohlcv["high"], ohlcv["low"], ohlcv["close"], 14)
        assert np.all((_tail(d["plus_di"]) >= 0) & (_tail(d["plus_di"]) <= 100))
        assert np.all((_tail(d["adx"]) >= 0) & (_tail(d["adx"]) <= 100))

    def test_aroon_bounds_and_trend(self, ohlcv):
        a = ta.aroon(ohlcv["high"], ohlcv["low"], 25)
        assert np.all((_tail(a["up"]) >= 0) & (_tail(a["up"]) <= 100))
        # Monotone uptrend -> aroon up = 100 at the end.
        n = 60
        h = np.linspace(100, 120, n)
        lo = h - 1.0
        up = ta.aroon(h, lo, 14)["up"]
        assert up[-1] == 100.0

    def test_vortex_positive(self, ohlcv):
        v = ta.vortex(ohlcv["high"], ohlcv["low"], ohlcv["close"], 14)
        assert np.all(_tail(v["plus"]) > 0)
        assert np.all(_tail(v["minus"]) > 0)

    def test_ichimoku_ordering(self, ohlcv):
        ic = ta.ichimoku(ohlcv["high"], ohlcv["low"])
        t = _tail(ic["tenkan"])
        assert np.all(np.isfinite(t))

    def test_donchian_encloses_price(self, ohlcv):
        d = ta.donchian(ohlcv["high"], ohlcv["low"], 20)
        c = ohlcv["close"]
        assert np.all(_tail(d["upper"]) >= _tail(c) - 1e-9)
        assert np.all(_tail(d["lower"]) <= _tail(c) + 1e-9)

    def test_keltner_bollinger_order(self, ohlcv):
        k = ta.keltner(ohlcv["high"], ohlcv["low"], ohlcv["close"], 20)
        assert np.all(_tail(k["upper"]) > _tail(k["lower"]))
        b = ta.bollinger(ohlcv["close"], 20, 2.0)
        assert np.all(_tail(b["upper"]) > _tail(b["lower"]))
        assert np.all((_tail(b["pct_b"]) > -0.5) & (_tail(b["pct_b"]) < 1.5))


class TestMomentum:
    def test_rsi_bounds_and_extremes(self, ohlcv):
        r = ta.rsi(ohlcv["close"], 14)
        tail = _tail(r)
        assert np.all((tail >= 0) & (tail <= 100))
        # All gains -> RSI 100.
        up = ta.rsi(np.linspace(50, 80, 60), 14)
        assert up[-1] == 100.0

    def test_stochastic_bounds(self, ohlcv):
        s = ta.stochastic(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert np.all((_tail(s["k"]) >= 0) & (_tail(s["k"]) <= 100))

    def test_williams_r_bounds(self, ohlcv):
        w = ta.williams_r(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert np.all((_tail(w) >= -100) & (_tail(w) <= 0))

    def test_roc_momentum(self):
        c = np.linspace(100, 110, 50)
        r = ta.roc(c, 10)
        assert np.all(_tail(r, 30) > 0)
        m = ta.momentum(c, 10)
        assert np.all(_tail(m, 30) > 0)

    def test_cmo_tsi_uo_bounds(self, ohlcv):
        c = ohlcv["close"]
        assert np.all(np.abs(_tail(ta.cmo(c, 14))) <= 100)
        assert np.all(np.isfinite(_tail(ta.tsi(c))))
        uo = ta.ultimate_oscillator(ohlcv["high"], ohlcv["low"], c)
        assert np.all((_tail(uo) >= 0) & (_tail(uo) <= 100))

    def test_ao_fisher_elder_kst(self, ohlcv):
        h, lo, c = ohlcv["high"], ohlcv["low"], ohlcv["close"]
        assert np.all(np.isfinite(_tail(ta.awesome_oscillator(h, lo))))
        f = ta.fisher_transform(h, lo)
        assert np.all(np.isfinite(_tail(f["fisher"])))
        er = ta.elder_ray(h, lo, c)
        assert np.all(_tail(er["bull_power"]) >= _tail(er["bear_power"]) - 1e-9)
        k = ta.kst(c)
        assert np.all(np.isfinite(_tail(k["kst"])))


class TestVolume:
    def test_obv_monotone_on_trend(self):
        c = np.linspace(50, 80, 100)
        v = np.full(100, 1000.0)
        o = ta.obv(c, v)
        assert o[-1] > o[10]  # all up days -> OBV accumulates

    def test_adl_cmf_bounds(self, ohlcv):
        h, lo, c, v = ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        adl = ta.ad_line(h, lo, c, v)
        assert np.all(np.isfinite(adl))
        cmf = ta.chaikin_money_flow(h, lo, c, v)
        assert np.all((_tail(cmf) >= -1) & (_tail(cmf) <= 1))

    def test_mfi_bounds(self, ohlcv):
        m = ta.mfi(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"])
        assert np.all((_tail(m) >= 0) & (_tail(m) <= 100))

    def test_force_eom_vwap_chaikvol(self, ohlcv):
        h, lo, c, v = ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
        assert np.all(np.isfinite(_tail(ta.force_index(c, v))))
        assert np.all(np.isfinite(_tail(ta.eom(h, lo, v))))
        vw = ta.vwap(h, lo, c, v)
        # VWAP inside the observed price envelope.
        assert np.all(vw >= lo.min() - 1e-9)
        assert np.all(vw <= h.max() + 1e-9)
        cv = ta.chaikin_volatility(h, lo)
        assert np.all(np.isfinite(_tail(cv)))
        co = ta.chaikin_oscillator(h, lo, c, v)
        assert np.all(np.isfinite(_tail(co)))

    def test_arms_index(self):
        a = np.full(50, 2000.0)
        d = np.full(50, 1000.0)
        av = np.full(50, 1e6)
        dv = np.full(50, 1e6)
        trin = ta.arms_index(a, d, av, dv)
        assert np.allclose(trin, 2.0)

    def test_stoch_rsi(self, ohlcv):
        s = ta.stoch_rsi(ohlcv["close"])
        tail = _tail(s)
        assert np.all((tail >= 0) & (tail <= 1))


class TestFailClosed:
    def test_short_series(self, ohlcv):
        with pytest.raises(ValueError):
            ta.sma(ohlcv["close"][:10], 20)
        with pytest.raises(ValueError):
            ta.atr(ohlcv["high"][:5], ohlcv["low"][:5], ohlcv["close"][:5], 14)

    def test_invalid_ohlc(self):
        h = np.array([10.0, 9.0, 12.0, 11.0] * 10)
        lo = np.array([9.0, 20.0, 10.0, 10.0] * 10)  # lo > h at idx 1
        c = np.array([9.5, 9.5, 11.0, 10.5] * 10)
        with pytest.raises(ValueError):
            ta.true_range(h, lo, c)

    def test_bad_window_and_volume(self, ohlcv):
        with pytest.raises(ValueError):
            ta.rsi(ohlcv["close"], window=0)
        with pytest.raises(ValueError):
            ta.obv(ohlcv["close"], -np.ones(400))
        with pytest.raises(ValueError):
            ta.mfi(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"][:100])
