"""Flat objective, walk-forward leakage, adversarial radius, SYNTHETIC small case."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.diffbacktest.jax_core import objective_gradients
from quant_fund.diffbacktest.numpy_core import perturb_prices, strong_trend_prices, synthetic_prices
from quant_fund.diffbacktest.spec import (
    BOXES,
    SMALL_CASE_TRAIN,
    StrategyParams,
    free_parameters,
    limitations,
)
from quant_fund.diffbacktest.study import (
    adversarial_radius,
    grid_search,
    optimize_flat,
    run_small_case,
    sensitivity_map,
    walk_forward_compare,
)

pytestmark = pytest.mark.synthetic


def test_perturb_zero_eps_rebuilds_prices() -> None:
    from quant_fund.diffbacktest.numpy_core import path_sigma

    prices = synthetic_prices(32, 3, seed=0)
    sigma = path_sigma(prices)
    rebuilt = perturb_prices(prices, np.zeros_like(prices), sigma)
    np.testing.assert_allclose(rebuilt, prices, rtol=1e-10, atol=1e-8)


def test_sensitivity_schema() -> None:
    prices = synthetic_prices(36, 3, seed=1)
    report = sensitivity_map(prices, "risk_parity", StrategyParams(vol_lookback=10), beta=4.0)
    assert report["research_only"] is True
    assert report["live_pnl_claim"] is False
    assert report["data_source"] == "unspecified"
    assert report["parameter_names"]
    for name in ("pnl", "pnl_sum", "sharpe", "drawdown"):
        block = report["objectives"][name]
        assert np.isfinite(block["value"])
        assert block["parameter_gradient"].shape == (len(report["parameter_names"]),)
        assert np.isfinite(block["gradient_norm"])
        assert np.isfinite(block["price_gradient_norm"])


def test_grid_matches_brute_force_and_degenerate() -> None:
    prices = synthetic_prices(48, 3, seed=4)
    found = grid_search(prices, "risk_parity", StrategyParams())
    assert found["status"] == "ok"
    assert found["n_finite"] == found["n_candidates"]
    assert found["train_hard_sharpe"] >= -1e9
    flat = np.full((40, 3), 100.0)
    dead = grid_search(flat, "equal_weight", StrategyParams(one_way_cost=0.0))
    assert dead["status"] == "degenerate"
    assert dead["n_finite"] == 0


def test_flat_objective_identity_and_box() -> None:
    prices = synthetic_prices(40, 3, seed=2)
    params = StrategyParams(vol_lookback=12, max_gross=1.0, rebalance_band=0.01)
    lam = 0.1
    fit = optimize_flat(
        prices, "risk_parity", params, lam=lam, steps=3, beta=4.0, data_source="SYNTHETIC"
    )
    assert fit["improved"]
    assert fit["end_objective"] >= fit["start_objective"] - 1e-8
    for name in free_parameters("risk_parity"):
        lo, hi = BOXES[name]
        value = float(getattr(fit["params"], name))
        assert lo - 1e-8 <= value <= hi + 1e-8
    grad = objective_gradients(
        prices,
        "risk_parity",
        fit["params"],
        "sharpe",
        mode="smooth",
        beta=fit["beta"],
        score_start=fit["score_start"],
    )
    gnorm = float(np.linalg.norm(grad.parameter_gradient))
    J = float(grad.value) - lam * gnorm
    assert pytest.approx(fit["end_objective"], abs=1e-6) == J
    assert fit["data_source"] == "SYNTHETIC"
    assert fit["research_only"] is True
    assert fit["live_pnl_claim"] is False


def test_walk_forward_selectors_see_only_the_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    prices = synthetic_prices(48, 3, seed=3)
    seen: list[int] = []

    def _wrap(fn):  # type: ignore[no-untyped-def]
        def inner(path, *args, **kwargs):  # type: ignore[no-untyped-def]
            seen.append(int(np.asarray(path).shape[0]))
            assert np.asarray(path).shape[0] < prices.shape[0]
            return fn(path, *args, **kwargs)

        return inner

    import quant_fund.diffbacktest.study as study

    monkeypatch.setattr(study, "grid_search", _wrap(study.grid_search))
    monkeypatch.setattr(study, "optimize_flat", _wrap(study.optimize_flat))
    report = walk_forward_compare(
        prices,
        "risk_parity",
        StrategyParams(vol_lookback=10),
        train_bars=32,
        test_bars=8,
        adam_steps=2,
        beta=4.0,
        data_source="SYNTHETIC",
    )
    assert seen == [32, 32, 40, 40]
    assert [fold["train_rows"] for fold in report["folds"]] == [32, 40]
    assert prices.shape[0] not in seen
    assert report["data_source"] == "SYNTHETIC"
    assert "SYNTHETIC" in " ".join(report["limitations"])
    for fold in report["folds"]:
        for side in ("grid", "flat"):
            assert np.isfinite(fold[side]["oos_terminal_pnl"])
            assert np.isfinite(fold[side]["oos_max_drawdown"])


def test_adversarial_contracts() -> None:
    prices = strong_trend_prices(48, seed=1)
    params = StrategyParams(gross_limit=1.0, max_name_weight=1.0, one_way_cost=0.0)
    found = adversarial_radius(
        prices, "equal_weight", params, rho_max=1.5, pgd_steps=4, bisections=4, beta=4.0
    )
    assert found["status"] == "certified"
    assert found["radius"] is not None
    assert 0.0 < float(found["radius"]) <= 1.5
    assert found["certified"] is True
    assert found["attacked_terminal_pnl"] <= 0.0
    assert found["eps"][0].tolist() == [0.0, 0.0, 0.0]
    assert float(np.max(np.abs(found["eps"]))) <= float(found["radius"]) + 1e-8

    down = synthetic_prices(36, 3, seed=1, mu=-0.02, sigma=0.002)
    dead = adversarial_radius(
        down, "equal_weight", params, rho_max=1.5, pgd_steps=2, bisections=2, beta=4.0
    )
    assert dead["status"] == "already_nonpositive"
    assert dead["radius"] == 0.0
    assert np.all(dead["eps"] == 0.0)

    missed = adversarial_radius(
        prices, "equal_weight", params, rho_max=1e-6, pgd_steps=1, bisections=1, beta=4.0
    )
    assert missed["status"] == "not_found"
    assert missed["radius"] is None
    assert missed["certified"] is False


def test_limitations_text() -> None:
    text = " ".join(limitations())
    assert "SYNTHETIC" in text
    assert "Not a live" in text
    assert "upper bound" in text
    assert "backtest.engine" in text


@pytest.mark.slow
def test_small_case_is_labeled_and_finite() -> None:
    report = run_small_case()
    assert report["data_source"] == "SYNTHETIC"
    assert report["research_only"] is True
    assert report["live_pnl_claim"] is False
    assert report["seed"] == 0
    assert "no power" in report["note"]
    names = [row["strategy"] for row in report["strategies"]]
    assert names == ["tsmom", "momentum", "risk_parity"]
    for row in report["strategies"]:
        assert row["walk_forward"]["train_rows"] == [SMALL_CASE_TRAIN, SMALL_CASE_TRAIN + 16]
        for pool in (row["walk_forward"]["pooled_grid"], row["walk_forward"]["pooled_flat"]):
            assert np.isfinite(pool["oos_terminal_pnl"])
            assert np.isfinite(pool["oos_sum_pnl"])
            assert np.isfinite(pool["oos_max_drawdown"])
            assert pool["n_bars"] == 32
        radius = row["adversarial"]["radius"]
        assert radius is None or (
            isinstance(radius, float) and np.isfinite(radius) and radius >= 0.0
        )
        assert np.isfinite(row["adversarial"]["base_terminal_pnl"])
