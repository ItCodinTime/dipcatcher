"""Coverage: forward_shadow deterministic helpers (hashing, sealed IO,
timestamps, warmup cutoff, commitment, book configs, broker compaction)."""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from quant_fund.execution.simulated_broker import SimulatedBroker
from quant_fund.paper import forward_shadow as fs
from quant_fund.research.real_benchmark import _seal


def test_hash_and_packet_hash() -> None:
    assert fs._hash(b"abc") == __import__("hashlib").sha256(b"abc").hexdigest()
    a = fs._packet_hash({"b": 1, "a": 2})
    b = fs._packet_hash({"a": 2, "b": 1})
    assert a == b  # canonical key order
    assert len(a) == 64


class TestDt:
    def test_aware_ok(self) -> None:
        out = fs._dt("2026-09-18T20:00:00Z")
        assert out == datetime(2026, 9, 18, 20, 0, tzinfo=UTC)

    def test_offset_normalized_to_utc(self) -> None:
        out = fs._dt("2026-09-18T16:00:00-04:00")
        assert out == datetime(2026, 9, 18, 20, 0, tzinfo=UTC)

    def test_non_string_rejected(self) -> None:
        with pytest.raises(ValueError):
            fs._dt(12345)

    def test_naive_rejected(self) -> None:
        with pytest.raises(ValueError, match="UTC offset"):
            fs._dt("2026-09-18T20:00:00")


class TestLoadSealedWrite:
    def test_load_plain_and_gz(self, tmp_path: Path) -> None:
        plain = tmp_path / "a.json"
        plain.write_text('{"x": 1}')
        assert fs._load(plain) == {"x": 1}
        gz = tmp_path / "a.json.gz"
        gz.write_bytes(gzip.compress(b'{"y": 2}'))
        assert fs._load(gz) == {"y": 2}

    def test_load_non_object_rejected(self, tmp_path: Path) -> None:
        p = tmp_path / "l.json"
        p.write_text("[1,2]")
        with pytest.raises(ValueError, match="expected JSON object"):
            fs._load(p)

    def test_sealed_roundtrip_and_tamper(self, tmp_path: Path) -> None:
        content = {"alpha": 1, "nested": {"k": [1, 2]}}
        sealed = _seal(content)
        p = tmp_path / "r.json"
        p.write_text(json.dumps(sealed))
        out = fs._sealed(p)
        assert out["alpha"] == 1
        assert out["receipt_sha256"] == sealed["receipt_sha256"]
        # tamper -> mismatch
        sealed2 = dict(sealed)
        sealed2["alpha"] = 2
        p.write_text(json.dumps(sealed2))
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            fs._sealed(p)

    def test_write_new_is_exclusive(self, tmp_path: Path) -> None:
        p = tmp_path / "w.json"
        fs._write_new(p, {"a": 1})
        assert json.loads(p.read_text()) == {"a": 1}
        with pytest.raises(FileExistsError):
            fs._write_new(p, {"a": 2})

    def test_write_new_rejects_nan(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            fs._write_new(tmp_path / "n.json", {"x": float("nan")})


def test_warmup_cutoff() -> None:
    warmup = [
        {
            "event_time": "2026-09-17T20:00:00Z",
            "available_time": "2026-09-17T21:00:00Z",
            "ingested_time": "2026-09-18T01:00:00Z",
        },
        {
            "event_time": "2026-09-18T20:00:00Z",
            "available_time": "2026-09-18T20:30:00Z",
            "ingested_time": "2026-09-19T02:00:00Z",
        },
    ]
    assert fs._warmup_cutoff(warmup) == datetime(2026, 9, 19, 2, 0, tzinfo=UTC)


def test_commitment_deterministic() -> None:
    kwargs: dict[str, Any] = {
        "spec_sha256": "a" * 64,
        "benchmark_sha256": "b" * 64,
        "validation_sha256": "c" * 64,
        "code_sha256": {"m.py": "d" * 64},
        "warmup": [{"x": 1}],
        "power_sha256": "e" * 64,
        "git_revision": "f" * 40,
        "git_worktree_sha256": "0" * 64,
        "schedule_sha256": "1" * 64,
        "historical_index_sha256": "2" * 64,
    }
    assert fs._commitment(**kwargs) == fs._commitment(**kwargs)
    kwargs2 = {**kwargs, "spec_sha256": "9" * 64}
    assert fs._commitment(**kwargs) != fs._commitment(**kwargs2)


def _spec_dict() -> dict:
    return {
        "execution": {
            "initial_nav": 50_000.0,
            "gross_limit": 0.9,
            "max_name_weight": 0.2,
            "participation_limit": 0.01,
            "commission_bps": 1.0,
            "half_spread_bps": 5.0,
            "impact_y": 0.1,
            "borrow_apr": 0.03,
            "funding_apr": 0.06,
            "cash_apr": 0.0,
            "max_names": 50,
            "target_buffer": 0.01,
        }
    }


class TestBookConfig:
    def test_configured_books_baseline(self) -> None:
        cfg = fs._configured_books(_spec_dict(), "configured")
        assert cfg.costs.impact_y == pytest.approx(0.1)
        assert cfg.costs.borrow_bps_per_year == pytest.approx(0.03 * 1e4)
        assert cfg.risk_gate.max_order_notional == pytest.approx(100_000.0)

    def test_configured_books_double_impact(self) -> None:
        cfg = fs._configured_books(_spec_dict(), "double_impact")
        assert cfg.costs.impact_y == pytest.approx(0.2)

    def test_compact_restore_roundtrip(self) -> None:
        cfg = fs._configured_books(_spec_dict(), "configured")
        broker = SimulatedBroker(cfg, initial_cash=50_000.0, slot="momentum_20:configured")
        state = fs._compact(broker)
        assert "history" not in state
        manifest = {"tournament_spec": _spec_dict()}
        restored = fs._restore(manifest, state, "momentum_20:configured")
        assert restored.slot == "momentum_20:configured"
        assert restored.cash == pytest.approx(50_000.0)

    def test_restore_slot_mismatch(self) -> None:
        cfg = fs._configured_books(_spec_dict(), "configured")
        broker = SimulatedBroker(cfg, initial_cash=1_000.0, slot="momentum_20:configured")
        state = fs._compact(broker)
        manifest = {"tournament_spec": _spec_dict()}
        with pytest.raises(ValueError, match="slot"):
            fs._restore(manifest, state, "equal_weight:double_impact")

    def test_initial_builds_all_books(self) -> None:
        manifest = {
            "tournament_spec": _spec_dict(),
            "receipt_sha256": "ab" * 32,
            "freeze": {"recorded_at": "2026-09-19T00:00:00Z"},
            "warmup": [{"event_time": "2026-09-18T20:00:00Z"}],
        }
        state = fs._initial(manifest)
        assert state["cursor"] == 0
        assert state["phase"] == "close"
        assert len(state["books"]) == len(fs.BOOKS) * len(fs.SCENARIOS)
        assert "momentum_20:double_impact" in state["books"]
        assert state["paired_sessions"] == 0


def test_summary_and_event_path(tmp_path: Path) -> None:
    state = {"cursor": 3, "history": [{"a": 1}, {"a": 2}], "x": 9}
    out = fs._summary(state)
    assert "history" not in out
    assert len(out["history_sha256"]) == 64
    assert out["cursor"] == 3
    p = fs._event_path(tmp_path, 7)
    assert p.name == "00000007.json"
    assert p.parent.name == "events"
