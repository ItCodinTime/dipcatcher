"""Generated prices verify accounting/timing, never real-market profitability."""

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from quant_fund.research.net_replay import MarketPanel, ReplayConfig, Strategy, replay
from quant_fund.research.net_tournament import compare_returns, prepare_tournament, run_tournament
from quant_fund.research.real_benchmark import prepare_benchmark


def panel(n=80, names=4):
    dates = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n)]
    prices = np.full((n, names), 100.0)
    return MarketPanel(
        dates,
        [f"s{i}" for i in range(names)],
        prices.copy(),
        prices.copy(),
        np.full_like(prices, 1e6),
        np.ones_like(prices, dtype=bool),
    )


def run(p, strategy=None, cfg=None):
    return replay(
        p,
        strategy or Strategy("equal", "equal_weight"),
        cfg or ReplayConfig(),
        start=p.dates[20].date().isoformat(),
        end=p.dates[-1].date().isoformat(),
        history=20,
    )


def test_flat_market_nav_equals_capital_less_every_cost():
    cfg = ReplayConfig(commission_bps=10, half_spread_bps=20)
    p = panel()
    result = run(p, cfg=cfg)
    shares = np.zeros(4)
    cash = cfg.initial_nav
    for row in result["daily"]:
        cash -= row["borrow"] + row["financing"]
        for fill in result["fills"]:
            if fill["execution_session"] == row["date"]:
                cost = sum(fill[k] for k in ("commission", "spread", "impact"))
                cash -= fill["quantity"] * fill["price"] + cost
                shares[p.names.index(fill["security_id"])] += fill["quantity"]
        assert row["cash"] == pytest.approx(cash)
        assert row["nav"] == pytest.approx(cash + shares.sum() * 100)
    total_cost = sum(
        sum(row[k] for k in ("commission", "spread", "impact", "borrow", "financing"))
        for row in result["daily"]
    )
    assert result["daily"][-1]["nav"] == pytest.approx(cfg.initial_nav - total_cost)
    assert result["liquidation_complete"] is True
    assert result["total_return"] < 0


def test_future_close_and_volume_cannot_change_first_fill():
    p = panel()
    original = run(p)
    p.close[21] = 1000
    p.volume[21] = 1
    changed = run(p)
    date = original["fills"][0]["execution_session"]
    assert [f for f in original["fills"] if f["execution_session"] == date] == [
        f for f in changed["fills"] if f["execution_session"] == date
    ]


def test_orders_are_sized_at_signal_close_and_capped_with_known_adv():
    p = panel()
    p.volume[:] = 10
    cfg = ReplayConfig(participation_limit=0.1)
    result = run(p, cfg=cfg)
    assert result["fills"][0]["quantity"] == pytest.approx(1)
    p.volume[21:] = 1e9
    assert run(p, cfg=cfg)["fills"][0]["quantity"] == pytest.approx(1)
    p = panel()
    p.opening[21] = 101
    cfg = ReplayConfig(max_name_weight=0.3)
    assert run(p, cfg=cfg)["fills"][0]["quantity"] == pytest.approx(
        cfg.initial_nav * 0.95 * 0.99 / 4 / 100
    )


def test_borrow_charged_over_calendar_gap_on_prior_open_short_value():
    p = panel(n=23)
    p.close *= np.exp(np.arange(23)[:, None] * np.array([-0.003, -0.001, 0.001, 0.003]))
    p.opening = p.close.copy()
    p.dates[22] += timedelta(days=2)
    cfg = ReplayConfig(borrow_apr=0.365, commission_bps=0, half_spread_bps=0, impact_y=0)
    result = run(p, Strategy("ls", "momentum", long_short=True), cfg)
    first = [f for f in result["fills"] if f["execution_session"] == p.dates[21].isoformat()]
    short_value = -sum(f["quantity"] * f["price"] for f in first if f["quantity"] < 0)
    assert short_value > 0
    assert result["daily"][1]["borrow"] == pytest.approx(short_value * 0.365 * 3 / 365)


def test_financing_and_cash_interest_have_opposite_signs():
    p = panel(n=23, names=2)
    cfg = ReplayConfig(
        gross_limit=1.8,
        max_name_weight=1,
        funding_apr=0.365,
        commission_bps=0,
        half_spread_bps=0,
        impact_y=0,
    )
    leveraged = run(p, cfg=cfg)
    debt = -leveraged["daily"][0]["cash"]
    assert debt > 0
    assert leveraged["daily"][1]["financing"] == pytest.approx(debt / 1000)
    unlevered = run(p, cfg=replace(cfg, gross_limit=0.5, cash_apr=0.365))
    assert unlevered["daily"][1]["financing"] == pytest.approx(
        -unlevered["daily"][0]["cash"] / 1000
    )


def test_missing_held_asset_cannot_be_valued_at_zero():
    p = panel()
    p.opening[22, 0] = np.nan
    with pytest.raises(ValueError, match="held asset"):
        run(p)


def test_terminal_liquidation_respects_capacity():
    p = panel(n=40)
    p.volume[:] = 10
    result = run(p, cfg=ReplayConfig(participation_limit=0.1))
    assert result["liquidation_complete"] is False
    assert result["terminal_residual_gross"] > 0


@pytest.mark.parametrize(
    "change", [{"participation_limit": 0}, {"funding_apr": float("nan")}, {"gross_limit": 3}]
)
def test_invalid_execution_assumptions_fail(change):
    with pytest.raises(ValueError):
        run(panel(), cfg=replace(ReplayConfig(), **change))


@pytest.fixture
def tournament(tmp_path):
    rng = np.random.default_rng(88)
    p = panel(n=240)
    p.close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, p.close.shape), axis=0))
    p.opening[1:] = p.close[:-1] * np.exp(rng.normal(0, 0.004, p.opening[1:].shape))
    frame = pl.concat(
        [
            pl.DataFrame(
                {
                    "security_id": [name] * 240,
                    "event_time": p.dates,
                    "available_time": p.dates,
                    "ingested_time": p.dates,
                    "close": p.close[:, i],
                    "open": p.opening[:, i],
                    "volume": p.volume[:, i],
                    "source": ["test_contract"] * 240,
                }
            )
            for i, name in enumerate(p.names)
        ]
    )
    data = tmp_path / "bars.parquet"
    frame.write_parquet(data)
    spec = {
        "dataset_path": str(data),
        "dataset_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "source_url": "https://example.com/test",
        "usage_basis": "generated contract test",
        "price_column": "close",
        "price_adjustment": "no actions in fixture",
        "universe_description": "generated",
        "survivorship_bias": True,
        "availability_basis": "reconstructed",
        "holdout_previously_inspected": True,
        "train_start": "2020-01-01",
        "train_end": "2020-03-31",
        "validation_start": "2020-04-05",
        "validation_end": "2020-05-31",
        "test_start": "2020-06-05",
        "test_end": "2020-08-20",
        "min_train_rows": 100,
        "min_score_dates": 30,
    }
    source = tmp_path / "benchmark.json"
    source.write_text(json.dumps(spec))
    benchmark = tmp_path / "benchmark"
    prepare_benchmark(source, benchmark)
    execution = {
        "execution": {},
        "trials": [
            {"name": "mom", "family": "momentum"},
            {"name": "rev", "family": "reversal", "lookback": 1},
        ],
        "benchmark": {"name": "equal", "family": "equal_weight"},
        "open_column": "open",
        "volume_column": "volume",
        "price_basis": "raw_price_return",
        "n_boot": 99,
        "block_sessions": 5,
        "seed": 17,
    }
    spec_path = tmp_path / "slate.json"
    spec_path.write_text(json.dumps(execution))
    destination = tmp_path / "tournament"
    prepare_tournament(benchmark, spec_path, destination)
    return destination, data


def test_validation_selection_is_frozen_and_every_trial_retained(tournament):
    run_dir, _ = tournament
    with pytest.raises(FileNotFoundError):
        run_tournament(run_dir, "test")
    validation = run_tournament(run_dir, "validation")
    test = run_tournament(run_dir, "test")
    assert test["selected"] == validation["selected"]
    assert test["selected"] in {"mom", "rev"}
    assert test["promote"] is False
    for scenario in test["scenarios"].values():
        assert set(scenario["trials"]) == {"equal", "mom", "rev"}
        assert scenario["comparison"]["tested_candidates"] == ["mom", "rev"]
        assert scenario["comparison"]["status"] == "tested"
    with pytest.raises(FileExistsError):
        run_tournament(run_dir, "test")


def test_tournament_binds_a_portable_benchmark_run(tournament):
    from pathlib import Path

    run_dir, _ = tournament
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert not Path(manifest["benchmark_run"]).is_absolute()
    moved = run_dir.parent.parent / f"{run_dir.parent.name}_moved"
    run_dir.parent.rename(moved)
    result = run_tournament(moved / run_dir.name, "validation")
    assert set(result["scenarios"]) == {"configured", "double_impact"}


def test_failure_is_retained_and_blocks_comparison(tournament, monkeypatch):
    import quant_fund.research.net_tournament as module

    original = module.replay

    def fail_one(panel, trial, *args, **kwargs):
        if trial.name == "mom":
            raise ValueError("injected missing price")
        return original(panel, trial, *args, **kwargs)

    monkeypatch.setattr(module, "replay", fail_one)
    run_dir, _ = tournament
    result = run_tournament(run_dir, "validation")
    assert result["complete"] is False
    assert result["selected"] is None
    for scenario in result["scenarios"].values():
        assert scenario["trials"]["mom"]["status"] == "failed"
        assert scenario["comparison"]["status"] == "incomplete_trials"


def test_constant_candidate_does_not_silently_disappear():
    daily = [{"date": str(i), "net_return": 0.001} for i in range(50)]
    outcomes = {name: {"status": "completed", "daily": daily} for name in ("base", "one", "two")}
    result = compare_returns(
        outcomes, ["one", "two"], "base", n_boot=99, block=5, seed=1, min_dates=30
    )
    assert result["status"] == "insufficient_or_degenerate"
    assert result["degenerate_candidates"] == ["one", "two"]


def test_corrected_comparison_keeps_candidate_names_aligned():
    rng = np.random.default_rng(122)
    base = rng.normal(0, 0.01, 100)
    outcomes = {}
    for name, returns in (
        ("base", base),
        ("loser", base - 0.03 + rng.normal(0, 0.002, 100)),
        ("winner", base + 0.03 + rng.normal(0, 0.002, 100)),
    ):
        outcomes[name] = {
            "status": "completed",
            "daily": [
                {"date": str(i), "net_return": float(value)} for i, value in enumerate(returns)
            ],
        }
    result = compare_returns(
        outcomes, ["loser", "winner"], "base", n_boot=99, block=5, seed=1, min_dates=30
    )
    assert result["status"] == "tested"
    assert result["stepm_adjusted_p"]["winner"] < 0.05
    assert result["stepm_adjusted_p"]["loser"] >= 0.05


def test_incomplete_attempt_cannot_be_silently_retried(tournament, monkeypatch):
    import quant_fund.research.net_tournament as module

    def crash(*args, **kwargs):
        raise RuntimeError("simulated process interruption")

    monkeypatch.setattr(module, "replay", crash)
    run_dir, _ = tournament
    with pytest.raises(RuntimeError):
        run_tournament(run_dir, "validation")
    assert (run_dir / "validation.attempt.json").exists()
    assert not (run_dir / "validation.json").exists()
    with pytest.raises(FileExistsError):
        run_tournament(run_dir, "validation")


def test_tournament_rejects_changed_dataset_before_scoring(tournament):
    run_dir, data = tournament
    data.write_bytes(data.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="dataset hash"):
        run_tournament(run_dir, "validation")


def test_tournament_cli_help():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "quant_fund.research.net_tournament", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "prepare" in result.stdout and "run" in result.stdout


def test_cost_aware_tournament_freezes_config_and_compares_matched_controls(tournament):
    from quant_fund.research.net_tournament import _read_receipt, _spec

    run_dir, _ = tournament
    spec_path = run_dir.parent / "slate.json"
    spec = json.loads(spec_path.read_text())
    spec["trials"].append(
        {
            **spec["trials"][0],
            "name": "mom_cost",
            "allocation": {"risk_window": 60, "uncertainty_aversion": 0.0},
        }
    )
    spec_path.write_text(json.dumps(spec))
    target = run_dir.parent / "cost_tournament"
    prepared = prepare_tournament(run_dir.parent / "benchmark", spec_path, target)
    assert prepared["candidate_count"] == 3
    assert "cost_allocation.py" in prepared["code_sha256"]
    assert {"cvxpy", "clarabel", "scipy"} <= prepared["runtime"].keys()
    validation = run_tournament(target, "validation")
    assert validation["complete"]
    holdout = run_tournament(target, "test")
    assert holdout["selected"] == validation["selected"]
    assert not holdout["promote"]
    for scenario in holdout["scenarios"].values():
        assert scenario["trials"]["mom_cost"]["allocations"]
        ablation = scenario["allocation_ablations"][0]
        assert (ablation["candidate"], ablation["control"]) == ("mom_cost", "mom")
        assert ablation["status"] == "completed"
        assert scenario["comparison"]["tested_candidates"] == ["mom", "rev", "mom_cost"]
    receipt = _read_receipt(target / "test.json")
    assert receipt["receipt_sha256"] == holdout["receipt_sha256"]
    spec["trials"].pop(0)
    with pytest.raises(ValueError, match="identical rank control"):
        _spec(spec)


def test_cost_solver_failure_is_retained_in_full_slate(tournament, monkeypatch):
    import quant_fund.research.cost_allocation as module

    run_dir, _ = tournament
    spec_path = run_dir.parent / "slate.json"
    spec = json.loads(spec_path.read_text())
    spec["trials"].append({**spec["trials"][0], "name": "mom_cost", "allocation": {}})
    spec_path.write_text(json.dumps(spec))
    target = run_dir.parent / "failed_cost_tournament"
    prepare_tournament(run_dir.parent / "benchmark", spec_path, target)

    def fail(*args, **kwargs):
        raise module.cp.error.SolverError("injected solver failure")

    monkeypatch.setattr(module.cp.Problem, "solve", fail)
    report = run_tournament(target, "validation")
    assert not report["complete"]
    assert report["selected"] is None
    for scenario in report["scenarios"].values():
        assert scenario["trials"]["mom_cost"]["status"] == "failed"
        assert scenario["trials"]["mom_cost"]["solver_diagnostic"]["weights_accepted"] is False
        assert scenario["comparison"]["status"] == "incomplete_trials"
        assert scenario["allocation_ablations"][0]["status"] == "incomplete_pair"
