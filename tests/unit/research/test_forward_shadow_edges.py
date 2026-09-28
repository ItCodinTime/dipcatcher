"""research/forward_shadow edge paths: freeze spec validation matrix,
_vector/_bar guards, record/rebuild journal edge cases, report inference
statuses, and the argparse CLI surface."""

from __future__ import annotations

import copy
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from quant_fund.research import forward_shadow as shadow

pytestmark = pytest.mark.synthetic


def _spec_and_bars() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    start = datetime(2030, 3, 1, 16, tzinfo=UTC)
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
    spec: dict[str, Any] = {
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
    return spec, bars


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    start = datetime(2030, 3, 1, 16, tzinfo=UTC)
    current = [start - timedelta(hours=1)]
    monkeypatch.setattr(shadow, "_now", lambda: current[0])
    spec, bars = _spec_and_bars()
    return spec, bars, current, tmp_path / "forward.sqlite"


def _freeze(spec: dict, bars: list, path: Path) -> dict:
    return shadow.freeze(copy.deepcopy(spec), copy.deepcopy(bars), path)


class TestFreezeSpecValidation:
    def test_unknown_or_missing_keys(self, env) -> None:
        spec, bars, _, path = env
        spec["extra"] = 1
        with pytest.raises(ValueError, match="missing or unknown"):
            _freeze(spec, bars[:20], path)

    def test_benchmark_rules(self, env) -> None:
        spec, bars, _, path = env
        spec["benchmark"] = {"name": "other", "family": "momentum"}
        with pytest.raises(ValueError, match="equal-weight"):
            _freeze(spec, bars[:20], path)
        spec2, _ = _spec_and_bars()
        spec2["benchmark"] = {"name": "momentum", "family": "equal_weight"}
        with pytest.raises(ValueError, match="equal-weight"):
            _freeze(spec2, bars[:20], path)

    def test_allocation_requires_funding(self, env) -> None:
        spec, bars, _, path = env
        spec["strategy"]["allocation"] = {"risk_window": 20}
        spec["execution"] = {"funding_apr": 0.0, "cash_apr": 0.01}
        with pytest.raises(ValueError, match="funding_apr"):
            _freeze(spec, bars[:20], path)

    @pytest.mark.parametrize(
        "bad",
        [
            ["a"],
            "notalist",
            ["a", "a", "b"],
            ["b", "a"],
            ["a", " ", "b"],
            ["a", 2, "b"],
            [f"s{i}" for i in range(501)],
        ],
    )
    def test_asset_rules(self, env, bad) -> None:
        spec, bars, _, path = env
        spec["assets"] = bad
        with pytest.raises(ValueError, match="unique sorted|identifiers"):
            _freeze(spec, bars[:20], path)

    def test_source_and_price_basis(self, env) -> None:
        spec, bars, _, path = env
        spec["source_url"] = "http://insecure.example"
        with pytest.raises(ValueError, match="price-return|source"):
            _freeze(spec, bars[:20], path)
        spec["source_url"] = "https://example.org/x"
        spec["selection_basis"] = "   "
        with pytest.raises(ValueError):
            _freeze(spec, bars[:20], path)
        spec["selection_basis"] = "ok"
        spec["price_basis"] = "total_return"
        with pytest.raises(ValueError):
            _freeze(spec, bars[:20], path)

    @pytest.mark.parametrize("lead", [0, 3601, 1.5, "60", True])
    def test_lead_seconds(self, env, lead) -> None:
        spec, bars, _, path = env
        spec["lead_seconds"] = lead
        with pytest.raises(ValueError, match="lead_seconds"):
            _freeze(spec, bars[:20], path)

    def test_session_rules(self, env) -> None:
        spec, bars, current, path = env
        spec["sessions"] = spec["sessions"][:1]
        with pytest.raises(ValueError, match="2..10000"):
            _freeze(spec, bars[:20], path)
        spec, _ = _spec_and_bars()
        spec["sessions"] = [{"signal_time": s["signal_time"]} for s in spec["sessions"]]
        with pytest.raises(ValueError, match="signal_time and execution_time"):
            _freeze(spec, bars[:20], path)
        spec, _ = _spec_and_bars()
        # Same-day execution cannot carry the overnight lead.
        sig = shadow._time(spec["sessions"][0]["signal_time"])
        spec["sessions"][0]["execution_time"] = (sig + timedelta(hours=2)).isoformat()
        with pytest.raises(ValueError, match="lead time"):
            _freeze(spec, bars[:20], path)

    def test_bootstrap_rules(self, env) -> None:
        spec, bars, _, path = env
        with pytest.raises(ValueError, match="sufficient"):
            _freeze(spec, bars[:5], path)
        with pytest.raises(ValueError, match="sufficient"):
            _freeze(spec, "notalist", path)  # type: ignore[arg-type]
        dup = copy.deepcopy(bars[:20])
        dup[1] = dict(dup[1], event_time=dup[0]["event_time"])
        with pytest.raises(ValueError, match="one close per UTC date"):
            _freeze(spec, dup, path)

    def test_calibration_rules(self, env) -> None:
        spec, bars, current, path = env
        spec["calibration"]["extra"] = 1
        with pytest.raises(ValueError, match="calibration requires"):
            _freeze(spec, bars[:20], path)
        spec, _ = _spec_and_bars()
        spec["calibration"]["end_time"] = (current[0] + timedelta(hours=2)).isoformat()
        with pytest.raises(ValueError, match="precede freeze"):
            _freeze(spec, bars[:20], path)
        spec, _ = _spec_and_bars()
        spec["calibration"]["source"] = ""
        with pytest.raises(ValueError):
            _freeze(spec, bars[:20], path)

    def test_manifest_fields(self, env) -> None:
        spec, bars, _, path = env
        out = _freeze(spec, bars[:20], path)
        assert out["classification"] == "local_forward_simulated"
        assert out["external_timestamp_verified"] is False
        assert "manifest_sha256" in out and "head_sha256" in out
        # The manifest (journal event 0) carries the honesty flags.
        rep = shadow.report(path)
        manifest = rep["events"][0]["payload"]
        assert manifest["live_pnl_claim"] is False


class TestVectorAndBar:
    def test_vector_rules(self) -> None:
        with pytest.raises(ValueError, match="asset order"):
            shadow._vector([1.0, 2.0], 3, positive=True, nullable=False)
        with pytest.raises(ValueError, match="price or volume"):
            shadow._vector([1.0, True], 2, positive=True, nullable=False)
        with pytest.raises(ValueError, match="price or volume"):
            shadow._vector([1.0, "x"], 2, positive=True, nullable=False)
        with pytest.raises(ValueError, match="price or volume"):
            shadow._vector([1.0, float("nan")], 2, positive=True, nullable=False)
        with pytest.raises(ValueError, match="price or volume"):
            shadow._vector([1.0, 0.0], 2, positive=True, nullable=False)
        with pytest.raises(ValueError, match="price or volume"):
            shadow._vector([1.0, -1.0], 2, positive=False, nullable=False)
        with pytest.raises(ValueError, match="price or volume"):
            shadow._vector([1.0, None], 2, positive=True, nullable=False)
        assert shadow._vector([1.0, None], 2, positive=True, nullable=True) == [1.0, None]

    def test_bar_rules(self) -> None:
        now = datetime(2030, 1, 1, tzinfo=UTC)
        with pytest.raises(ValueError, match="close and volume"):
            shadow._bar({"event_time": "x"}, 2, now)
        bar = {
            "event_time": "2030-01-02T00:00:00+00:00",
            "available_time": "2030-01-01T00:00:00+00:00",
            "close": [1.0] * 2,
            "volume": [1.0] * 2,
        }
        with pytest.raises(ValueError, match="inconsistent availability"):
            shadow._bar(bar, 2, now)
        future = dict(bar, available_time="2030-01-03T00:00:00+00:00")
        with pytest.raises(ValueError, match="future-dated"):
            shadow._bar(future, 2, now + timedelta(days=1))

    def test_naive_timestamp_rejected(self) -> None:
        with pytest.raises(ValueError, match="timezone"):
            shadow._time("2030-01-01T00:00:00")


def _settled(env) -> tuple[Path, dict, list, list]:
    """Freeze + drive session 0 through decide and settle."""
    spec, bars, current, path = env
    _freeze(spec, bars[:20], path)
    current[0] = shadow._time(bars[20]["event_time"]) + timedelta(seconds=1)
    shadow.record(path, "decide", {"session": 0, "bar": copy.deepcopy(bars[20])})
    t = shadow._time(spec["sessions"][0]["execution_time"])
    current[0] = t + timedelta(seconds=1)
    shadow.record(
        path,
        "settle",
        {
            "session": 0,
            "event_time": t.isoformat(),
            "available_time": t.isoformat(),
            "prices": bars[21]["close"],
        },
    )
    return path, spec, bars, current


class TestRecordEdges:
    def test_bad_operation_and_session_type(self, env) -> None:
        spec, bars, _, path = env
        _freeze(spec, bars[:20], path)
        with pytest.raises(ValueError, match="integer session"):
            shadow.record(path, "bogus", {"session": 0})
        with pytest.raises(ValueError, match="integer session"):
            shadow.record(path, "decide", {"session": "0"})

    def test_out_of_order_session_rejected(self, env) -> None:
        spec, bars, current, path = env
        _freeze(spec, bars[:20], path)
        current[0] = shadow._time(bars[22]["event_time"]) + timedelta(seconds=1)
        with pytest.raises(ValueError, match="next frozen session"):
            shadow.record(path, "decide", {"session": 2, "bar": copy.deepcopy(bars[22])})

    def test_settle_requires_pending(self, env) -> None:
        spec, bars, current, path = env
        _freeze(spec, bars[:20], path)
        t = shadow._time(spec["sessions"][0]["execution_time"])
        current[0] = t + timedelta(seconds=1)
        with pytest.raises(ValueError, match="committed decision"):
            shadow.record(
                path,
                "settle",
                {
                    "session": 0,
                    "event_time": t.isoformat(),
                    "available_time": t.isoformat(),
                    "prices": bars[21]["close"],
                },
            )

    def test_settle_requires_pending_and_fields(self, env) -> None:
        spec, bars, current, path = env
        _freeze(spec, bars[:20], path)
        current[0] = shadow._time(bars[20]["event_time"]) + timedelta(seconds=1)
        shadow.record(path, "decide", {"session": 0, "bar": copy.deepcopy(bars[20])})
        t = shadow._time(spec["sessions"][0]["execution_time"])
        current[0] = t + timedelta(seconds=1)
        with pytest.raises(ValueError, match="requires session"):
            shadow.record(path, "settle", {"session": 0})
        with pytest.raises(ValueError, match="belongs to another session"):
            shadow.record(
                path,
                "settle",
                {
                    "session": 0,
                    "event_time": (t + timedelta(hours=2)).isoformat(),
                    "available_time": t.isoformat(),
                    "prices": bars[21]["close"],
                },
            )

    def test_decide_bar_must_match_signal_time(self, env) -> None:
        spec, bars, current, path = env
        _freeze(spec, bars[:20], path)
        current[0] = shadow._time(bars[20]["event_time"]) + timedelta(seconds=1)
        bad = copy.deepcopy(bars[19])
        with pytest.raises(ValueError, match="scheduled signal"):
            shadow.record(path, "decide", {"session": 0, "bar": bad})

    def test_decide_outside_window_and_miss_reason(self, env) -> None:
        spec, bars, current, path = env
        _freeze(spec, bars[:20], path)
        # After the deadline but before execution: must record a miss.
        t_exec = shadow._time(spec["sessions"][0]["execution_time"])
        current[0] = t_exec - timedelta(seconds=30)
        with pytest.raises(ValueError, match="missed session"):
            shadow.record(path, "decide", {"session": 0, "bar": copy.deepcopy(bars[20])})
        # Miss before execution elapsed: rejected.
        with pytest.raises(ValueError, match="elapsed"):
            shadow.record(
                path, "miss", {"session": 0, "bar": copy.deepcopy(bars[20]), "reason": "net down"}
            )
        # Miss after execution with blank reason: rejected.
        current[0] = t_exec + timedelta(seconds=1)
        with pytest.raises(ValueError, match="reason"):
            shadow.record(
                path, "miss", {"session": 0, "bar": copy.deepcopy(bars[20]), "reason": " "}
            )

    def test_decide_idempotent_and_conflicting_retry(self, env) -> None:
        spec, bars, current, path = env
        _freeze(spec, bars[:20], path)
        current[0] = shadow._time(bars[20]["event_time"]) + timedelta(seconds=1)
        req = {"session": 0, "bar": copy.deepcopy(bars[20])}
        first = shadow.record(path, "decide", copy.deepcopy(req))
        second = shadow.record(path, "decide", copy.deepcopy(req))
        assert first["hash"] == second["hash"]
        mutated = copy.deepcopy(req)
        mutated["bar"] = copy.deepcopy(bars[19])
        with pytest.raises(ValueError, match="conflicting retry"):
            shadow.record(path, "decide", mutated)


class TestRebuildAndReport:
    def test_report_before_completion(self, env) -> None:
        spec, bars, current, path = env
        _freeze(spec, bars[:20], path)
        rep = shadow.report(path)
        assert rep["inference"]["status"] == "not_final_or_insufficient_or_missed"
        assert rep["live_pnl_claim"] is False and rep["promote"] is False
        assert rep["schedule_complete"] is False
        assert rep["mean_daily_net_difference"] is None

    def test_expected_head_mismatch(self, env) -> None:
        spec, bars, current, path = env
        out = _freeze(spec, bars[:20], path)
        with pytest.raises(ValueError):
            shadow.report(path, expected_head="0" * 64)
        rep = shadow.report(path, expected_head=out["head_sha256"])
        assert rep["head_sha256"] == out["head_sha256"]

    def test_rebuild_rejects_unknown_and_duplicates(self, env) -> None:
        spec, bars, current, path = env
        _freeze(spec, bars[:20], path)
        events, state = shadow.shadow_journal.reconcile(path, shadow.rebuild)
        manifest_event = events[0]
        with pytest.raises(ValueError, match="unknown shadow event"):
            shadow.rebuild(
                [
                    manifest_event,
                    {
                        "kind": "bogus",
                        "payload": {"session": 0},
                        "recorded_at": "2999-01-01T00:00:00+00:00",
                    },
                ]
            )
        with pytest.raises(ValueError, match="recording times moved backwards"):
            shadow.rebuild(
                [
                    manifest_event,
                    {
                        "kind": "decision",
                        "payload": {"session": 0, "bar": bars[20]},
                        "recorded_at": "1999-01-01T00:00:00+00:00",
                    },
                ]
            )


class TestCli:
    def test_cli_freeze_and_reconcile(self, env, monkeypatch, capsys, tmp_path) -> None:
        spec, bars, current, path = env
        spec_path = tmp_path / "spec.json"
        boot_path = tmp_path / "boot.json"
        spec_path.write_text(json.dumps(spec))
        boot_path.write_text(json.dumps(bars[:20]))
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "x",
                "freeze",
                "--spec",
                str(spec_path),
                "--bootstrap",
                str(boot_path),
                "--run",
                str(path),
            ],
        )
        shadow.main()
        out = json.loads(capsys.readouterr().out)
        assert out["classification"] == "local_forward_simulated"

    def test_cli_decide_settle_and_reconcile_output(
        self, env, monkeypatch, capsys, tmp_path
    ) -> None:
        spec, bars, current, path = env
        _freeze(spec, bars[:20], path)
        inp = tmp_path / "in.json"
        inp.write_text(json.dumps({"session": 0, "bar": bars[20]}))
        current[0] = shadow._time(bars[20]["event_time"]) + timedelta(seconds=1)
        monkeypatch.setattr(sys, "argv", ["x", "decide", "--run", str(path), "--input", str(inp)])
        shadow.main()
        ev = json.loads(capsys.readouterr().out)
        assert ev["kind"] == "decision"
        t = shadow._time(spec["sessions"][0]["execution_time"])
        current[0] = t + timedelta(seconds=1)
        inp.write_text(
            json.dumps(
                {
                    "session": 0,
                    "event_time": t.isoformat(),
                    "available_time": t.isoformat(),
                    "prices": bars[21]["close"],
                }
            )
        )
        monkeypatch.setattr(sys, "argv", ["x", "settle", "--run", str(path), "--input", str(inp)])
        shadow.main()
        assert json.loads(capsys.readouterr().out)["kind"] == "settlement"
        out_path = tmp_path / "report.json"
        monkeypatch.setattr(
            sys, "argv", ["x", "reconcile", "--run", str(path), "--output", str(out_path)]
        )
        shadow.main()
        rep = json.loads(capsys.readouterr().out)
        assert "events" not in rep and "state" not in rep
        assert rep["completed_sessions"] == 1
        assert json.loads(out_path.read_text())["completed_sessions"] == 1

    def test_cli_error_maps_to_system_exit(self, env, monkeypatch, tmp_path) -> None:
        spec, bars, current, path = env
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"session": "nope"}))
        monkeypatch.setattr(sys, "argv", ["x", "decide", "--run", str(path), "--input", str(bad)])
        with pytest.raises(SystemExit):
            shadow.main()
