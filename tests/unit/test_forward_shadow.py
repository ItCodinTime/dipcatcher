"""Prospective clocks are controlled fixtures; no live performance is asserted."""

import copy
import json
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from quant_fund.research import forward_shadow as shadow
from quant_fund.research import shadow_journal as journal
from quant_fund.research.forward_evidence import evidence_plan, long_run_variance
from quant_fund.research.net_replay import MarketPanel, ReplayConfig, Strategy, replay


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    start = datetime(2030, 3, 1, 16, tzinfo=UTC)
    current = [start - timedelta(hours=1)]
    monkeypatch.setattr(shadow, "_now", lambda: current[0])
    rng = np.random.default_rng(41)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, (30, 4)), axis=0))
    dates = [start + timedelta(days=i - 20) for i in range(30)]
    bars = [
        {
            "event_time": t.isoformat(),
            "available_time": t.isoformat(),
            "close": prices[i].tolist(),
            "volume": [1e6] * 4,
        }
        for i, t in enumerate(dates)
    ]
    spec = {
        "strategy": {"name": "momentum", "family": "momentum"},
        "benchmark": {"name": "equal", "family": "equal_weight"},
        "execution": {},
        "assets": ["a", "b", "c", "d"],
        "source_url": "https://example.org/generated-fixture",
        "price_basis": "raw_price_return",
        "selection_basis": "generated contract test",
        "lead_seconds": 60,
        "sessions": [
            {
                "signal_time": dates[i].isoformat(),
                "execution_time": (dates[i + 1] - timedelta(hours=1)).isoformat(),
            }
            for i in range(20, 29)
        ],
        "calibration": {
            "end_time": dates[19].isoformat(),
            "source": "generated calibration",
            "net_differences": rng.normal(0, 0.001, 100).tolist(),
        },
        "evidence": {"effect_bps": 5.0, "lag": 3},
    }
    path = tmp_path / "forward.sqlite"
    shadow.freeze(spec, bars[:20], path)
    return path, spec, bars, current


def decision(experiment, i=0):
    _, _, bars, clock = experiment
    clock[0] = shadow._time(bars[20 + i]["event_time"]) + timedelta(seconds=1)
    return {"session": i, "bar": copy.deepcopy(bars[20 + i])}


def settlement(experiment, i=0):
    _, spec, bars, clock = experiment
    t = shadow._time(spec["sessions"][i]["execution_time"])
    clock[0] = t + timedelta(seconds=1)
    return {
        "session": i,
        "event_time": t.isoformat(),
        "available_time": t.isoformat(),
        "prices": bars[21 + i]["close"],
    }


def test_freeze_is_exclusive_and_schedule_is_future(experiment, tmp_path):
    path, spec, bars, clock = experiment
    with pytest.raises(FileExistsError):
        shadow.freeze(spec, bars[:20], path)
    clock[0] += timedelta(days=2)
    with pytest.raises(ValueError, match="future"):
        shadow.freeze(spec, bars[:20], tmp_path / "late.sqlite")


def test_full_forward_replay_parity_and_cash_reconstruction(experiment):
    path, spec, bars, _ = experiment
    for i in range(9):
        shadow.record(path, "decide", decision(experiment, i))
        shadow.record(path, "settle", settlement(experiment, i))
    result = shadow.report(path)
    assert result["schedule_complete"] and result["terminal_liquidation_complete"]
    assert result["reconciled"] and result["completed_sessions"] == 9
    assert not result["promote"] and not result["external_timestamp_verified"]
    assert result["inference"]["p_value"] is None
    closes = np.array([b["close"] for b in bars])
    p = MarketPanel(
        [shadow._time(b["event_time"]) for b in bars],
        spec["assets"],
        closes,
        closes.copy(),
        np.full_like(closes, 1e6),
        np.ones_like(closes, dtype=bool),
    )
    for raw in (spec["strategy"], spec["benchmark"]):
        expected = replay(
            p, Strategy(**raw), ReplayConfig(), start="2030-03-01", end="2030-03-10", history=20
        )
        book = result["state"]["books"][raw["name"]]
        assert book["cash"] == pytest.approx(expected["daily"][-1]["nav"], abs=1e-8)
        actual = [
            e["payload"]["outcomes"][raw["name"]]["daily"]["nav"]
            for e in result["events"]
            if e["kind"] == "settlement"
        ]
        np.testing.assert_allclose(actual, [r["nav"] for r in expected["daily"]], rtol=0, atol=1e-8)
        cash = 100000.0
        shares = np.zeros(4)
        for event in result["events"]:
            if event["kind"] != "settlement":
                continue
            out = event["payload"]["outcomes"][raw["name"]]
            daily = out["daily"]
            cash -= daily["borrow"] + daily["financing"]
            for fill in out["fills"]:
                cash -= fill["quantity"] * fill["price"] + sum(
                    fill[k] for k in ("commission", "spread", "impact")
                )
                shares[spec["assets"].index(fill["security_id"])] += fill["quantity"]
            assert cash == pytest.approx(out["state"]["cash"])
            np.testing.assert_allclose(shares, out["state"]["shares"], atol=1e-10)


def test_restart_and_duplicate_commands_are_exactly_once(experiment):
    path, _, _, clock = experiment
    request = decision(experiment)
    committed = shadow.record(path, "decide", request)
    clock[0] += timedelta(days=3)
    assert shadow.record(path, "decide", request) == committed
    data = settlement(experiment)
    settled = shadow.record(path, "settle", data)
    assert shadow.record(path, "settle", data) == settled
    assert shadow.report(path)["completed_sessions"] == 1
    altered = copy.deepcopy(data)
    altered["prices"][0] *= 2
    with pytest.raises(ValueError, match="conflicting retry"):
        shadow.record(path, "settle", altered)
    assert shadow.report(path)["completed_sessions"] == 1


def test_concurrent_duplicate_settlements_commit_once(experiment):
    path, *_ = experiment
    shadow.record(path, "decide", decision(experiment))
    data = settlement(experiment)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: shadow.record(path, "settle", data), range(2)))
    assert results[0] == results[1]
    assert shadow.report(path)["completed_sessions"] == 1


def test_crash_between_event_and_projection_rolls_back(experiment, monkeypatch):
    path, *_ = experiment
    original = journal._save_state
    before = shadow.report(path)["head_sha256"]

    def crash(*args, **kwargs):
        raise RuntimeError("simulated process failure before commit")

    monkeypatch.setattr(journal, "_save_state", crash)
    data = decision(experiment)
    with pytest.raises(RuntimeError):
        shadow.record(path, "decide", data)
    monkeypatch.setattr(journal, "_save_state", original)
    assert shadow.report(path)["head_sha256"] == before
    shadow.record(path, "decide", data)
    assert shadow.report(path)["state"]["pending"] is not None


def test_corrupt_projection_requires_explicit_repair(experiment):
    path, *_ = experiment
    shadow.record(path, "decide", decision(experiment))
    shadow.record(path, "settle", settlement(experiment))
    expected = shadow.report(path)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE projection SET state='{}'")
    with pytest.raises(ValueError, match="projection mismatch"):
        shadow.report(path)
    repaired = shadow.report(path, repair=True, expected_head=expected["head_sha256"])
    assert repaired == expected


def test_hash_corruption_cannot_be_repaired(experiment):
    path, *_ = experiment
    with sqlite3.connect(path) as db:
        db.execute("DROP TRIGGER no_event_update")
        body = json.loads(db.execute("SELECT body FROM events WHERE seq=1").fetchone()[0])
        body["payload"]["execution"]["initial_nav"] *= 2
        db.execute("UPDATE events SET body=? WHERE seq=1", (json.dumps(body),))
    with pytest.raises(ValueError, match="hash/sequence"):
        shadow.report(path, repair=True)


def test_retained_head_detects_truncated_journal(experiment):
    path, *_ = experiment
    shadow.record(path, "decide", decision(experiment))
    head = shadow.report(path)["head_sha256"]
    with sqlite3.connect(path) as db:
        db.execute("DROP TRIGGER no_event_delete")
        db.execute("DELETE FROM events WHERE seq=2")
    with pytest.raises(ValueError, match="external head"):
        shadow.report(path, repair=True, expected_head=head)


def test_settlement_cannot_precede_decision_or_outcome(experiment):
    path, _, _, clock = experiment
    data = settlement(experiment)
    with pytest.raises(ValueError, match="committed decision"):
        shadow.record(path, "settle", data)
    shadow.record(path, "miss", {"session": 0, "bar": experiment[2][20], "reason": "worker down"})
    # Recorded times never go backwards, even for a syntactically valid retry.
    clock[0] = shadow._time(data["event_time"]) - timedelta(seconds=1)
    with pytest.raises(ValueError, match="future-dated"):
        shadow.record(path, "settle", data)


def test_late_decision_records_failure_and_explicit_miss(experiment):
    path, _, _, clock = experiment
    data = decision(experiment)
    settlement(experiment)
    with pytest.raises(ValueError, match="window is closed"):
        shadow.record(path, "decide", data)
    missed = shadow.record(path, "miss", {**data, "reason": "missed feed deadline"})
    assert all(not p["orders"] for p in missed["payload"]["plans"].values())
    shadow.record(path, "settle", settlement(experiment))
    report = shadow.report(path)
    assert report["state"]["missed"] == 1 and report["state"]["failures"] == 1
    assert report["inference"]["p_value"] is None


def test_skipping_and_future_data_are_rejected(experiment):
    path, *_ = experiment
    data = decision(experiment)
    data["bar"]["available_time"] = (shadow._now() + timedelta(days=1)).isoformat()
    with pytest.raises(ValueError, match="future-dated"):
        shadow.record(path, "decide", data)
    with pytest.raises(ValueError, match="next frozen session"):
        shadow.record(path, "decide", decision(experiment, 1))
    assert shadow.report(path)["completed_sessions"] == 0


def test_missing_held_mark_blocks_entire_pair_then_can_recover(experiment):
    path, *_ = experiment
    shadow.record(path, "decide", decision(experiment))
    shadow.record(path, "settle", settlement(experiment))
    before = shadow.report(path)["state"]["books"]
    shadow.record(path, "decide", decision(experiment, 1))
    data = settlement(experiment, 1)
    broken = copy.deepcopy(data)
    broken["prices"] = [None] * 4
    with pytest.raises(ValueError, match="held asset"):
        shadow.record(path, "settle", broken)
    assert shadow.report(path)["state"]["books"] == before
    shadow.record(path, "settle", data)
    assert shadow.report(path)["completed_sessions"] == 2


def test_code_or_runtime_change_blocks_resume(experiment, monkeypatch):
    path, *_ = experiment
    monkeypatch.setattr(shadow, "_identity", lambda: {"changed": True})
    with pytest.raises(ValueError, match="code/runtime changed"):
        shadow.report(path, repair=True)


def test_evidence_plan_increases_with_positive_dependence():
    rng = np.random.default_rng(11)
    innovations = rng.normal(0, 0.01, 2000)
    correlated = np.zeros(2000)
    for i in range(1, 2000):
        correlated[i] = 0.8 * correlated[i - 1] + innovations[i]
    correlated *= innovations.std() / correlated.std()
    iid = evidence_plan(innovations.tolist(), effect_bps=5, lag=10)
    dependent = evidence_plan(correlated.tolist(), effect_bps=5, lag=10)
    assert dependent["required_sessions"] > 3 * iid["required_sessions"]
    smaller = evidence_plan(innovations.tolist(), effect_bps=2.5, lag=10)
    assert smaller["required_sessions"] == pytest.approx(4 * iid["required_sessions"], abs=4)


def test_hac_matches_direct_bartlett_calculation():
    x = np.random.default_rng(2).normal(size=100)
    y = x - x.mean()
    expected = np.dot(y, y) / 100 + 2 * sum(
        (1 - k / 4) * np.dot(y[k:], y[:-k]) / 100 for k in range(1, 4)
    )
    assert long_run_variance(x, 3) == pytest.approx(expected)


@pytest.mark.parametrize("change", [{"effect_bps": 0}, {"power": 1}, {"alpha": 0.8}, {"lag": -1}])
def test_bad_evidence_design_rejected(change):
    options = {"effect_bps": 5, "lag": 3, **change}
    with pytest.raises(ValueError):
        evidence_plan(np.arange(100).tolist(), **options)


def test_degenerate_calibration_cannot_shorten_evidence():
    with pytest.raises(ValueError, match="degenerate"):
        evidence_plan([0.0] * 100, effect_bps=1, lag=3)


def test_cli_exposes_no_clock_override():
    result = subprocess.run(
        [sys.executable, "-m", "quant_fund.research.forward_shadow", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "freeze" in result.stdout and "settle" in result.stdout
    assert "--now" not in result.stdout


def test_real_process_exit_before_commit_preserves_prior_head(experiment):
    path, *_ = experiment
    request = decision(experiment)
    before = shadow.report(path)["head_sha256"]
    script = """
import json, os, sys
from pathlib import Path
from quant_fund.research import forward_shadow as shadow, shadow_journal as journal
shadow._now = lambda: shadow._time(sys.argv[3])
def terminate(*args, **kwargs):
    os._exit(17)
journal._save_state = terminate
shadow.record(Path(sys.argv[1]), 'decide', json.loads(sys.argv[2]))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path), json.dumps(request), shadow._now().isoformat()],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 17, result.stderr
    assert shadow.report(path)["head_sha256"] == before
    shadow.record(path, "decide", request)


def test_cost_aware_strategy_records_allocation_and_actual_fills(experiment):
    path, spec, bars, clock = experiment
    spec = copy.deepcopy(spec)
    spec["strategy"]["allocation"] = {"risk_window": 20, "uncertainty_aversion": 0.0}
    newpath = path.with_name("cost.sqlite")
    shadow.freeze(spec, bars[:20], newpath)
    modified = (newpath, spec, bars, clock)
    result = shadow.record(newpath, "decide", decision(modified))
    allocation = result["payload"]["plans"]["momentum"]["allocation"]
    assert allocation["status"] == "optimal"
    shadow.record(newpath, "settle", settlement(modified))
    shadow.record(newpath, "decide", decision(modified, 1))
    assert shadow.report(newpath)["completed_sessions"] == 1


def test_terminal_capacity_residual_blocks_final_evidence(experiment):
    path, spec, bars, clock = experiment
    spec = copy.deepcopy(spec)
    spec["execution"]["participation_limit"] = 1e-6
    newpath = path.with_name("illiquid.sqlite")
    shadow.freeze(spec, bars[:20], newpath)
    modified = (newpath, spec, bars, clock)
    for i in range(9):
        shadow.record(newpath, "decide", decision(modified, i))
        shadow.record(newpath, "settle", settlement(modified, i))
    result = shadow.report(newpath)
    assert result["schedule_complete"]
    assert not result["terminal_liquidation_complete"]
    assert result["inference"]["p_value"] is None
    assert any(abs(q) > 0 for b in result["state"]["books"].values() for q in b["shares"])


def test_inference_only_after_frozen_schedule_and_power_requirement(experiment):
    path, spec, bars, clock = experiment
    spec = copy.deepcopy(spec)
    start = shadow._time(bars[0]["event_time"])
    prices = 100 * np.exp(np.cumsum(np.random.default_rng(10).normal(0, 0.01, (56, 4)), axis=0))
    newbars = [
        {
            "event_time": (start + timedelta(days=i)).isoformat(),
            "available_time": (start + timedelta(days=i)).isoformat(),
            "close": prices[i].tolist(),
            "volume": [1e6] * 4,
        }
        for i in range(56)
    ]
    spec["sessions"] = [
        {
            "signal_time": newbars[i]["event_time"],
            "execution_time": (
                shadow._time(newbars[i + 1]["event_time"]) - timedelta(hours=1)
            ).isoformat(),
        }
        for i in range(20, 55)
    ]
    spec["evidence"]["effect_bps"] = 100.0
    newpath = path.with_name("final.sqlite")
    frozen = shadow.freeze(spec, newbars[:20], newpath)
    assert frozen["evidence_plan"]["required_sessions"] == 30
    modified = (newpath, spec, newbars, clock)
    for i in range(35):
        shadow.record(newpath, "decide", decision(modified, i))
        shadow.record(newpath, "settle", settlement(modified, i))
        if i == 30:
            assert shadow.report(newpath)["inference"]["p_value"] is None
    result = shadow.report(newpath)
    assert result["inference"]["status"] == "fixed_horizon_approximation"
    assert 0 <= result["inference"]["p_value"] <= 1
    assert not result["promote"]


def test_deadline_is_checked_again_immediately_before_commit(experiment, monkeypatch):
    path, spec, _, clock = experiment
    original = journal._save_state
    deadline = shadow._time(spec["sessions"][0]["execution_time"]) - timedelta(seconds=60)

    def slow_write(*args, **kwargs):
        original(*args, **kwargs)
        clock[0] = deadline

    monkeypatch.setattr(journal, "_save_state", slow_write)
    with pytest.raises(ValueError, match="before commit"):
        shadow.record(path, "decide", decision(experiment))
    state = shadow.report(path)["state"]
    assert state["pending"] is None and state["cursor"] == 0
    assert state["failures"] == 1
