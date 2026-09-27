"""Wrappee fingerprint / _WRAPPEE_CACHE correctness (hit, miss, invalidation).

Wave 5: train-primary adjacent reuse when cal slides but train window holds;
prove no stale reuse across label/alpha/config change.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from quant_fund.config.models import AppConfig
from quant_fund.pipeline.forecast import (
    _WRAPPEE_CACHE,
    clear_forecast_caches,
    clear_wrappee_cache,
    conformal_sets_asof,
    wrappee_cache_size,
    wrappee_cal_fingerprint,
)


def test_wrappee_fingerprint_stable_and_distinct() -> None:
    a = wrappee_cal_fingerprint(
        train_date_keys=("2019-01-01", "2019-01-02"),
        cal_date_keys=("2019-01-03",),
        alpha=0.10,
        n_tr=10,
        n_cal=5,
        y_tr_mean=0.01,
        scale_tr_mean=0.02,
        label="future_log_return_1",
    )
    b = wrappee_cal_fingerprint(
        train_date_keys=("2019-01-01", "2019-01-02"),
        cal_date_keys=("2019-01-03",),
        alpha=0.10,
        n_tr=10,
        n_cal=5,
        y_tr_mean=0.01,
        scale_tr_mean=0.02,
        label="future_log_return_1",
    )
    assert a == b
    # Alpha (interval config) change → miss
    c = wrappee_cal_fingerprint(
        train_date_keys=("2019-01-01", "2019-01-02"),
        cal_date_keys=("2019-01-03",),
        alpha=0.05,
        n_tr=10,
        n_cal=5,
        y_tr_mean=0.01,
        scale_tr_mean=0.02,
        label="future_log_return_1",
    )
    assert c != a
    # Label (distribution target config) change → miss
    d = wrappee_cal_fingerprint(
        train_date_keys=("2019-01-01", "2019-01-02"),
        cal_date_keys=("2019-01-03",),
        alpha=0.10,
        n_tr=10,
        n_cal=5,
        y_tr_mean=0.01,
        scale_tr_mean=0.02,
        label="future_log_return_5",
    )
    assert d != a
    # Cal window slide alone → STILL SAME under train-primary default (Wave 5)
    e = wrappee_cal_fingerprint(
        train_date_keys=("2019-01-01", "2019-01-02"),
        cal_date_keys=("2019-01-04",),
        alpha=0.10,
        n_tr=10,
        n_cal=5,
        y_tr_mean=0.01,
        scale_tr_mean=0.02,
        label="future_log_return_1",
    )
    assert e == a
    # Strict include_cal=True → cal slide misses
    e_strict = wrappee_cal_fingerprint(
        train_date_keys=("2019-01-01", "2019-01-02"),
        cal_date_keys=("2019-01-04",),
        alpha=0.10,
        n_tr=10,
        n_cal=5,
        y_tr_mean=0.01,
        scale_tr_mean=0.02,
        label="future_log_return_1",
        include_cal=True,
    )
    a_strict = wrappee_cal_fingerprint(
        train_date_keys=("2019-01-01", "2019-01-02"),
        cal_date_keys=("2019-01-03",),
        alpha=0.10,
        n_tr=10,
        n_cal=5,
        y_tr_mean=0.01,
        scale_tr_mean=0.02,
        label="future_log_return_1",
        include_cal=True,
    )
    assert e_strict != a_strict
    # Train window slide → miss
    f = wrappee_cal_fingerprint(
        train_date_keys=("2019-01-01", "2019-01-03"),
        cal_date_keys=("2019-01-04",),
        alpha=0.10,
        n_tr=10,
        n_cal=5,
        y_tr_mean=0.01,
        scale_tr_mean=0.02,
        label="future_log_return_1",
    )
    assert f != a


def test_clear_wrappee_cache_empties() -> None:
    clear_wrappee_cache()
    assert wrappee_cache_size() == 0
    fp = wrappee_cal_fingerprint(
        train_date_keys=("a",),
        cal_date_keys=("b",),
        alpha=0.1,
        n_tr=1,
        n_cal=1,
        y_tr_mean=0.0,
        scale_tr_mean=0.01,
        label="x",
    )
    _WRAPPEE_CACHE[fp] = object()
    assert wrappee_cache_size() == 1
    clear_wrappee_cache()
    assert wrappee_cache_size() == 0
    clear_forecast_caches()
    assert wrappee_cache_size() == 0


def _panel(n_days: int = 40, n_names: int = 6, seed: int = 7) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime(2019, 1, 2, 16, 0, 0)
    times = [start + timedelta(days=i) for i in range(n_days)]
    rows: list[dict[str, object]] = []
    for t in times:
        for j in range(n_names):
            vol = 0.01 + 0.02 * (j / max(n_names - 1, 1))
            ret = float(vol * rng.normal())
            rows.append(
                {
                    "event_time": t,
                    "security_id": f"S{j:02d}",
                    "ret_1": ret,
                    "vol_20": vol,
                    "mom_20": ret,
                    "future_log_return_1": float(vol * rng.normal()),
                    "future_log_return_5": float(vol * np.sqrt(5.0) * rng.normal()),
                    "future_max_drawdown_5": float(-abs(vol * (0.5 + abs(rng.normal())))),
                }
            )
    return pl.DataFrame(rows)


def test_wrappee_cache_hit_same_asof_and_invalidate_on_alpha() -> None:
    clear_forecast_caches()
    frame = _panel()
    asof = frame["event_time"].max()
    cfg = AppConfig()
    assert isinstance(asof, datetime)

    r1 = conformal_sets_asof(frame, asof, cfg, alpha=0.10)
    assert r1 is not None
    size_after_cold = wrappee_cache_size()
    assert size_after_cold >= 1

    # Same asof + alpha: wrappee cache must not grow (hit)
    r2 = conformal_sets_asof(frame, asof, cfg, alpha=0.10)
    assert r2 is not None
    assert wrappee_cache_size() == size_after_cold

    # Alpha config change: new fingerprint → cache grows (miss)
    r3 = conformal_sets_asof(frame, asof, cfg, alpha=0.05)
    assert r3 is not None
    assert wrappee_cache_size() >= size_after_cold + 1

    clear_wrappee_cache()
    assert wrappee_cache_size() == 0
    # Conformal cache alone would skip the wrappee path — clear all forecast caches
    # so the cold path refits and re-populates _WRAPPEE_CACHE.
    clear_forecast_caches()
    assert wrappee_cache_size() == 0
    r4 = conformal_sets_asof(frame, asof, cfg, alpha=0.10)
    assert r4 is not None
    assert wrappee_cache_size() >= 1


def test_wrappee_no_stale_reuse_on_label_change() -> None:
    """Label / distribution target change must miss — no stale wrappee reuse."""
    clear_forecast_caches()
    frame = _panel()
    asof = frame["event_time"].max()
    assert isinstance(asof, datetime)
    cfg = AppConfig()
    cfg.train.distribution_target = "future_log_return_1"
    r1 = conformal_sets_asof(frame, asof, cfg, alpha=0.10)
    assert r1 is not None
    size1 = wrappee_cache_size()
    # Force conformal miss by clearing conformal only — keep wrappee, then change label
    from quant_fund.pipeline import forecast as fmod

    fmod._CONFORMAL_CACHE.clear()
    cfg2 = AppConfig()
    cfg2.train.distribution_target = "future_log_return_5"
    r2 = conformal_sets_asof(frame, asof, cfg2, alpha=0.10)
    assert r2 is not None
    # New label → new fingerprint → cache grows (no stale hit)
    assert wrappee_cache_size() >= size1 + 1


def test_wrappee_adjacent_asof_train_primary_hit() -> None:
    """Adjacent asofs that share the train window should hit under train-primary key.

    Honest: not every adjacent pair shares train keys (70% cut may move). We scan
    a short asof sequence and require at least one adjacent hit on this panel, OR
    fall back to proving train-primary fingerprint equality (cal slide) which is
    the documented reuse condition.
    """
    clear_forecast_caches()
    frame = _panel(n_days=55, n_names=8, seed=11)
    times = [t for t in sorted(frame["event_time"].unique().to_list()) if isinstance(t, datetime)]
    asofs = times[-12:]
    cfg = AppConfig()
    from quant_fund.pipeline import forecast as fmod

    adjacent_hit = False
    populated = False
    for i, asof in enumerate(asofs):
        fmod._CONFORMAL_CACHE.clear()
        before = wrappee_cache_size()
        r = conformal_sets_asof(frame, asof, cfg, alpha=0.10)
        if r is None:
            continue
        populated = True
        after = wrappee_cache_size()
        if i > 0 and after == before and before >= 1:
            adjacent_hit = True
    assert populated
    # Fingerprint-level proof that cal slide alone does not invalidate (reuse valid)
    fp_a = wrappee_cal_fingerprint(
        train_date_keys=tuple(f"d{i}" for i in range(20)),
        cal_date_keys=tuple(f"c{i}" for i in range(5)),
        alpha=0.1,
        n_tr=100,
        n_cal=25,
        y_tr_mean=0.0,
        scale_tr_mean=0.02,
        label="future_log_return_1",
    )
    fp_b = wrappee_cal_fingerprint(
        train_date_keys=tuple(f"d{i}" for i in range(20)),
        cal_date_keys=tuple(f"c{i}" for i in range(6)),  # cal grew
        alpha=0.1,
        n_tr=100,
        n_cal=30,
        y_tr_mean=0.0,
        scale_tr_mean=0.02,
        label="future_log_return_1",
    )
    assert fp_a == fp_b
    assert fp_a == fp_b
    # On this 55d panel, train-primary reuse should produce ≥1 adjacent hit.
    assert adjacent_hit, "expected ≥1 adjacent train-primary wrappee cache hit"
