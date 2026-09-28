"""pretrade bench harness: helpers, report shape, CLI — tiny sample counts."""

from __future__ import annotations

import json

import pytest

from quant_fund.pretrade import bench


def test_pin_affinity_reports_or_unavailable() -> None:
    assert bench._pin_affinity() in {"0", "unavailable"}


def test_benchmark_config_is_wide_and_allowing() -> None:
    cfg = bench.benchmark_config()
    assert cfg.limits.max_order_notional > 1e10
    assert cfg.limits.max_gross_notional > 1e10
    assert cfg.limits.price_collar_bps == 50.0
    assert cfg.session.timezone == "America/New_York"
    assert cfg.session.open_minute < cfg.session.close_minute


def test_session_open_ns_fixed() -> None:
    assert bench._session_open_ns() == bench._session_open_ns()
    assert bench._session_open_ns() > 0


class TestPercentile:
    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="no samples"):
            bench._percentile([], 0.5)

    def test_rounding_indices(self) -> None:
        xs = list(range(10))
        assert bench._percentile(xs, 0.0) == 0
        assert bench._percentile(xs, 0.5) == xs[round(0.5 * 9)]
        assert bench._percentile(xs, 1.0) == 9
        # p below zero index / above last index clamps.
        assert bench._percentile(xs, 0.999) == 9


def test_summarize_reports_all_quantiles() -> None:
    out = bench._summarize([5, 1, 9, 3, 7])
    assert out["n"] == 5
    assert out["min_ns"] == 1
    assert out["max_ns"] == 9
    assert out["p50_ns"] <= out["p90_ns"] <= out["p99_ns"] <= out["p999_ns"]
    assert out["mean_ns"] == pytest.approx(5.0)


def test_prepare_builds_allowing_view() -> None:
    engine = bench.PretradeEngine(
        bench.benchmark_config(),
        hmac_key=bench._HMAC,
        initial_nav=1_000_000.0,
        initial_cash=1_000_000.0,
        ts_ns=bench._session_open_ns(),
    )
    view = bench._prepare(engine, bench._session_open_ns())
    assert engine.check(view, apply=False) == 0


def test_run_trial_rejects_denies(monkeypatch) -> None:
    engine = bench.PretradeEngine(
        bench.benchmark_config(),
        hmac_key=bench._HMAC,
        initial_nav=1_000_000.0,
        initial_cash=1_000_000.0,
        ts_ns=bench._session_open_ns(),
    )
    view = bench._prepare(engine, bench._session_open_ns())
    monkeypatch.setattr(bench.PretradeEngine, "check", lambda self, v, apply: 0x2)
    with pytest.raises(RuntimeError, match="expected an allow"):
        bench._run_trial(engine, view, 4, 1_000)


def test_run_benchmark_shape_and_gate() -> None:
    # The qty ladder repeats each trial, so phases must be at least
    # duplicate_window_ns/step_ns = 50 samples apart to stay an allow path.
    report = bench.run_benchmark(trials=2, samples=60, warmup=60, gate=True)
    assert report["schema"] == 1
    assert report["numba"] is False and report["rust"] is False
    assert report["gc_disabled_during_samples"] is True
    assert report["samples_per_trial"] == 60
    assert len(report["trials"]) == 2
    assert 0 <= report["median_trial_index"] < 2
    assert report["gate_requested"] is True
    assert isinstance(report["gate_pass"], bool)
    assert report["p50_ns"] > 0
    assert report["path"] == "allow"
    assert len(report["checks"]) >= 20
    assert report["affinity"] in {"0", "unavailable"}


def test_run_benchmark_validation() -> None:
    with pytest.raises(ValueError, match="positive"):
        bench.run_benchmark(trials=0)
    with pytest.raises(ValueError, match="positive"):
        bench.run_benchmark(samples=1)


def test_main_json_out_and_exit(tmp_path, capsys) -> None:
    out = tmp_path / "bench.json"
    code = bench.main(
        ["--trials", "1", "--samples", "20", "--warmup", "60", "--json-out", str(out)]
    )
    assert code == 0
    text = capsys.readouterr().out
    assert json.loads(text)["schema"] == 1
    written = json.loads(out.read_text())
    assert written["trials"][0]["n"] == 20


def test_main_gate_failure_returns_one(monkeypatch, capsys) -> None:
    monkeypatch.setattr(bench, "P50_LIMIT_NS", 0)
    monkeypatch.setattr(bench, "P99_LIMIT_NS", 0)
    code = bench.main(["--trials", "1", "--samples", "10", "--warmup", "60", "--gate"])
    assert code == 1
    err = capsys.readouterr().err
    assert "latency gate failed" in err


def test_main_no_gate_never_fails(capsys) -> None:
    code = bench.main(["--trials", "1", "--samples", "10", "--warmup", "60"])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["gate_requested"] is False
