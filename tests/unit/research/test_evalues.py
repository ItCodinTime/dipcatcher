from pathlib import Path

import numpy as np

from quant_fund.metrics import evalues
from quant_fund.metrics.evalues import (
    bench_e_coverage,
    e_process,
    e_process_threshold,
    e_value_bernoulli,
)


def test_e_process_nonnegative_unit_start() -> None:
    unit = e_process(np.asarray([], dtype=float), 0.10)
    assert unit.shape == (1,)
    assert unit[0] == 1.0
    path = e_process(np.array([0.0, 1.0, 0.0, 0.0]), 0.10)
    assert np.all(path >= 0.0)
    assert np.all(np.isfinite(path))
    assert np.isclose(path[0], e_value_bernoulli(0.0, 0.10))


def test_one_step_is_unit_mean_under_null() -> None:
    e_miss = e_value_bernoulli(1.0, 0.10)
    e_hit = e_value_bernoulli(0.0, 0.10)
    assert e_miss > 1.0
    assert 0.0 < e_hit < 1.0
    assert np.isclose(0.10 * e_miss + 0.90 * e_hit, 1.0)


def test_null_bernoulli_rarely_crosses_twenty() -> None:
    rng = np.random.default_rng(42)
    misses = (rng.random(80) < 0.10).astype(float)
    path = e_process(misses, 0.10)
    out = e_process_threshold(path, level=0.05)
    assert out["threshold"] == 20.0
    assert out["reject"] is False
    assert out["first_cross"] is None
    assert float(np.max(path)) < 20.0


def test_miss_streak_raises_e() -> None:
    path = e_process(np.ones(12), 0.10)
    assert path[-1] > path[0]
    assert path[-1] > 1.0
    assert np.all(np.diff(path) > 0.0)


def test_threshold_rejects_after_cross() -> None:
    path = e_process(np.ones(16), 0.10)
    out = e_process_threshold(path, level=0.05)
    assert out["reject"] is True
    assert out["first_cross"] is not None
    assert int(out["first_cross"]) < path.size
    assert float(path[int(out["first_cross"])]) >= 20.0


def test_bench_e_coverage_keys_and_seeded_final() -> None:
    rng = np.random.default_rng(7)
    covered = (rng.random(32) >= 0.10).astype(float)
    bench = bench_e_coverage(covered, alpha=0.10)
    assert set(bench) == {"coverage", "e_final", "ever_cross", "n"}
    assert bench["n"] == 32
    assert bench["ever_cross"] is False
    assert np.isclose(float(bench["coverage"]), float(np.mean(covered)))
    assert np.isclose(float(bench["e_final"]), 0.26282509860274916)


def test_module_has_no_sharpe() -> None:
    text = Path(evalues.__file__).read_text(encoding="utf-8").lower()
    assert "sharpe" not in text
    assert not any("sharpe" in name.lower() for name in dir(evalues))


def test_bad_alpha_fail_closed() -> None:
    import pytest

    for alpha in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="alpha"):
            e_value_bernoulli(0.0, alpha)
        with pytest.raises(ValueError, match="alpha"):
            e_process(np.array([0.0, 1.0]), alpha)
        with pytest.raises(ValueError, match="alpha"):
            bench_e_coverage(np.array([1.0, 0.0]), alpha=alpha)


def test_bad_level_fail_closed() -> None:
    import pytest

    path = e_process(np.array([0.0, 1.0]), 0.10)
    for level in (0.0, 1.0, -0.05, 2.0):
        with pytest.raises(ValueError, match="level"):
            e_process_threshold(path, level=level)


def test_bench_empty_coverage_honest() -> None:
    bench = bench_e_coverage(np.asarray([], dtype=float), alpha=0.10)
    assert bench["n"] == 0
    assert bench["e_final"] == 1.0
    assert bench["ever_cross"] is False
    assert bench["coverage"] != bench["coverage"]  # NaN


def test_miss_out_of_range_clips_like_soft_coverage() -> None:
    """Out-of-range miss matches clipped [0,1] — documented soft clip, no API change."""
    alpha = 0.10
    assert e_value_bernoulli(-0.5, alpha) == e_value_bernoulli(0.0, alpha)
    assert e_value_bernoulli(1.5, alpha) == e_value_bernoulli(1.0, alpha)
    # Fractional miss (soft coverage) stays interior — intentional, not fail-closed.
    mid = e_value_bernoulli(0.25, alpha)
    assert e_value_bernoulli(0.0, alpha) < mid < e_value_bernoulli(1.0, alpha)
