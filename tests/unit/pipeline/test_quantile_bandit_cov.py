"""Coverage pack for quantile_bandit: date keying paths, cold-start draws,
tied-arm selection, posterior contraction, and group-size boundaries."""

from __future__ import annotations

from datetime import UTC, date, datetime

import numpy as np

from quant_fund.models.quantile_bandit import (
    QuantileThompson,
    _as_key,
    _ordered_groups,
)


def test_as_key_stringifies_each_date_kind() -> None:
    assert _as_key(np.datetime64("2024-03-05")) == "2024-03-05"
    assert _as_key(datetime(2024, 3, 5, tzinfo=UTC)) == "2024-03-05T00:00:00+00:00"
    assert _as_key(date(2024, 3, 5)) == "2024-03-05"
    assert _as_key("d7") == "d7"  # plain strings pass through
    assert _as_key(2.5) == "2.5"


def test_ordered_groups_np_datetime64_sorts_chronologically() -> None:
    dates = np.array(
        ["2024-01-03", "2024-01-01", "2024-01-02", "2024-01-01"],
        dtype="datetime64[D]",
    )
    keys, order = _ordered_groups(dates)
    assert keys[0] == "2024-01-03"
    assert order == ["2024-01-01", "2024-01-02", "2024-01-03"]


def test_run_panel_walks_np_datetime64_dates() -> None:
    rng = np.random.default_rng(21)
    n_dates, n_names, d = 6, 8, 3
    base = np.datetime64("2024-02-01", "D")
    dates = np.repeat(base + np.arange(n_dates), n_names)
    x = rng.normal(size=(n_dates * n_names, d))
    y = x[:, 0] + 0.05 * rng.normal(size=x.shape[0])
    bandit = QuantileThompson(n_quantiles=5, ridge=1.0, seed=21)
    trace = bandit.run_panel(x, y, dates, k=2)
    assert trace.dates == [str(base + np.timedelta64(i, "D")) for i in range(n_dates)]
    assert np.isfinite(trace.policy_reward).all()
    assert trace.policy_reward.size == n_dates
    assert len(bandit._xs) == n_dates * 2  # updated on each chosen arm


def test_run_panel_walks_isoformat_dates_chronologically() -> None:
    rng = np.random.default_rng(22)
    n_dates, n_names = 5, 6
    stamps = [datetime(2024, 3, day, tzinfo=UTC) for day in (3, 1, 5, 2, 4)]
    dates = np.repeat(np.asarray(stamps, dtype=object), n_names)
    x = rng.normal(size=(n_dates * n_names, 2))
    y = x[:, 0] + 0.05 * rng.normal(size=x.shape[0])
    trace = QuantileThompson(n_quantiles=3, ridge=1.0, seed=22).run_panel(x, y, dates, k=2)
    assert trace.dates == [
        datetime(2024, 3, day, tzinfo=UTC).isoformat() for day in (1, 2, 3, 4, 5)
    ]


def test_cold_start_draws_centered_reproducible_noise() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(3000, 4))
    fresh = QuantileThompson(n_quantiles=5, ridge=1.0, seed=4)
    scores = fresh.scores(x)
    assert scores.shape == (3000,)
    assert np.isfinite(scores).all()
    # beta starts at zero: scores are pure prior-draw noise, symmetric around 0
    assert abs(float(np.mean(scores))) < 0.15
    assert float(np.std(scores)) > 0.5
    twin = QuantileThompson(n_quantiles=5, ridge=1.0, seed=4)
    assert np.allclose(twin.scores(x), scores)  # same seed reproduces the draw


def test_tied_arms_select_last_k_indices() -> None:
    """Identical context rows tie on score; argsort keeps the last k."""
    x = np.tile(np.array([[1.0, -2.0]]), (6, 1))
    bandit = QuantileThompson(n_quantiles=5, ridge=1.0, seed=6)
    idx = bandit.select(x, k=2)
    assert idx.tolist() == [4, 5]


def test_tied_arms_panel_reward_is_last_k_mean() -> None:
    """Every arm in a group scores identically, so policy reward is the
    deterministic mean over the last k rows of each date group."""
    n_dates, n_names = 4, 6
    dates = np.repeat(np.arange(n_dates, dtype=float), n_names)
    row = np.array([1.0, 0.5])
    x = np.tile(row, (n_dates * n_names, 1))
    rng = np.random.default_rng(9)
    y = rng.normal(size=n_dates * n_names)
    bandit = QuantileThompson(n_quantiles=3, ridge=1.0, seed=9)
    trace = bandit.run_panel(x, y, dates, k=2)
    expected = [float(np.mean(y[i * n_names + 4 : i * n_names + 6])) for i in range(n_dates)]
    assert np.allclose(trace.policy_reward, expected)


def test_posterior_draw_dispersion_decays_with_updates() -> None:
    """Thompson exploration decays: per-row score spread across posterior
    draws shrinks once the bandit has seen consistent evidence."""
    rng = np.random.default_rng(5)
    n_obs, d = 200, 3
    w = np.array([1.5, -1.0, 0.5])
    x = rng.normal(size=(n_obs, d))
    y = x @ w + 0.01 * rng.normal(size=n_obs)
    probe = rng.normal(size=(60, d))

    def draw_spread(model: QuantileThompson) -> float:
        draws = np.stack([model.scores(probe) for _ in range(30)])
        return float(np.mean(np.std(draws, axis=0)))

    # n_quantiles=1 fixes the tau draw so dispersion isolates posterior noise
    pre = draw_spread(QuantileThompson(n_quantiles=1, ridge=1.0, seed=5))
    bandit = QuantileThompson(n_quantiles=1, ridge=1.0, seed=5)
    for xi, yi in zip(x, y, strict=True):
        bandit.update(xi, yi)
    post = draw_spread(bandit)
    assert post < 0.25 * pre
    # and the learned scores track the planted linear signal
    assert np.corrcoef(bandit.scores(probe), probe @ w)[0, 1] > 0.95


def test_group_size_boundary_is_inclusive() -> None:
    """A date group of exactly max(2k, 4) rows is kept; one row fewer skips."""
    rng = np.random.default_rng(31)
    dates = np.asarray([0.0] * 4 + [1.0] * 3 + [2.0] * 4)
    x = rng.normal(size=(11, 2))
    y = rng.normal(size=11)
    trace = QuantileThompson(n_quantiles=3, ridge=1.0, seed=31).run_panel(x, y, dates, k=2)
    assert trace.dates == ["0.0", "2.0"]

    # same boundary on the finite filter: 4 rows with only 3 finite -> skipped
    y2 = y.copy()
    y2[0] = np.nan  # date 0.0 now has 3 finite rows
    trace2 = QuantileThompson(n_quantiles=3, ridge=1.0, seed=31).run_panel(x, y2, dates, k=2)
    assert trace2.dates == ["2.0"]


def test_nonfinite_update_does_not_latch_feature_dim() -> None:
    """A rejected (non-finite) update returns before _ensure_dim, so the
    bandit stays uninitialized and accepts a different dim afterwards."""
    bandit = QuantileThompson(n_quantiles=3, ridge=1.0, seed=8)
    bandit.update(np.array([1.0, 2.0]), float("nan"))
    assert bandit._d is None
    assert not bandit._xs
    bandit.update(np.ones(3), 1.0)
    assert bandit._d == 3
    assert len(bandit._xs) == 1
