"""Sign-flip paper books: costs even, Sharpe odd, DD vs -150% incompatible."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from quant_fund.hedge_lab.mirror import (
    dollar_neutral_weights,
    negate_target_weights,
    nested_worst_univariate_scores,
    pair_book_and_mirror,
    random_date_scores,
)
from quant_fund.metrics.returns import max_drawdown, sharpe_ratio, wealth_index


def test_negate_target_weights_flips_sign_keeps_gross() -> None:
    w = pl.DataFrame(
        {
            "event_time": ["2020-01-02", "2020-01-02"],
            "security_id": ["A", "B"],
            "target_weight": [0.4, -0.4],
        }
    )
    anti = negate_target_weights(w)
    assert anti["target_weight"].to_list() == [-0.4, 0.4]
    assert float(np.sum(np.abs(anti["target_weight"].to_numpy()))) == pytest.approx(0.8)


def test_frictionless_mirror_sharpe_is_odd() -> None:
    rng = np.random.default_rng(7)
    n_dates, n_names = 80, 20
    signal = rng.normal(size=(n_dates, n_names))
    y = (0.2 * signal + 0.03 * rng.normal(size=signal.shape)).reshape(-1)
    scores = signal.reshape(-1)
    dates = np.repeat(np.arange(n_dates), n_names)
    ids = np.tile(np.arange(n_names), n_dates)
    pair = pair_book_and_mirror(scores, y, dates, ids, one_way_cost=0.0)
    assert pair["book"]["economic"]["sharpe"] > 1.0
    assert pair["mirror"]["economic"]["sharpe"] < -1.0
    assert abs(pair["sharpe_sum"]) < 0.2
    scaled = 3.0 * pair["book"]["returns"]
    assert sharpe_ratio(scaled)["sharpe"] == pytest.approx(
        pair["book"]["economic"]["sharpe"], rel=1e-10, abs=1e-10
    )


def test_costs_do_not_flip_with_the_signal() -> None:
    rng = np.random.default_rng(8)
    n_dates, n_names = 60, 16
    scores = rng.normal(size=n_dates * n_names)
    y = rng.normal(scale=0.02, size=n_dates * n_names)
    dates = np.repeat(np.arange(n_dates), n_names)
    ids = np.tile(np.arange(n_names), n_dates)
    free = pair_book_and_mirror(scores, y, dates, ids, one_way_cost=0.0)
    taxed = pair_book_and_mirror(scores, y, dates, ids, one_way_cost=0.002)
    assert taxed["sharpe_sum"] < free["sharpe_sum"] - 0.05
    assert taxed["book"]["economic"]["sharpe"] <= free["book"]["economic"]["sharpe"] + 1e-12
    assert taxed["mirror"]["economic"]["sharpe"] <= free["mirror"]["economic"]["sharpe"] + 1e-12


def test_five_percent_dd_cannot_be_minus_150_percent() -> None:
    r = np.full(120, -0.012)
    total = float(wealth_index(r)[-1] - 1.0)
    dd = float(max_drawdown(r))
    assert total < -0.75
    assert dd < -0.5
    assert not (dd > -0.05 and total < -1.5)


def test_dollar_neutral_gross_is_one() -> None:
    w = dollar_neutral_weights(np.arange(10, dtype=float), k_frac=0.2)
    assert float(np.sum(np.abs(w))) == pytest.approx(1.0)
    assert float(np.sum(w)) == pytest.approx(0.0)


def test_nested_worst_univariate_picks_train_loser() -> None:
    from datetime import datetime, timedelta

    from quant_fund.config import load_config

    rng = np.random.default_rng(9)
    n_dates, n_names = 40, 12
    x0 = rng.normal(size=(n_dates * n_names, 1))
    x1 = rng.normal(size=(n_dates * n_names, 1))
    x = np.concatenate([x0, x1], axis=1)
    y = -0.4 * x[:, 0] + 0.05 * rng.normal(size=n_dates * n_names)
    stamps = [datetime(2020, 1, 2) + timedelta(days=int(i)) for i in range(n_dates)]
    dates = np.repeat(np.array(stamps, dtype=object), n_names)
    cfg = load_config("configs/research.yaml")
    cfg.validation.train_bars = 12
    cfg.validation.val_bars = 4
    cfg.validation.test_bars = 4
    scores = nested_worst_univariate_scores(x, y, dates, cfg, horizon_bars=1)
    finite = np.isfinite(scores)
    assert int(finite.sum()) > 20
    # Column 0 is the train loser (negative IC). Selected scores track x0.
    assert float(np.corrcoef(scores[finite], x[finite, 0])[0, 1]) > 0.7


def test_random_scores_are_date_local() -> None:
    dates = np.repeat(np.arange(5), 4)
    a = random_date_scores(dates, seed=1)
    b = random_date_scores(dates, seed=1)
    assert a.tolist() == b.tolist()
    assert float(np.std(a)) > 0.0
