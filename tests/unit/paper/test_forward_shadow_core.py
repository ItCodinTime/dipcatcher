"""paper/forward_shadow edge paths: timestamp/load/seal helpers, manifest
guard matrix, close/open packet validators, and a fabricated-run
decide/execute/verify round-trip against the pinned XNYS calendar."""

from __future__ import annotations

import gzip
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from quant_fund.paper import forward_shadow as fwd
from quant_fund.paper.xnys_calendar import materialize_xnys_schedule

pytestmark = pytest.mark.synthetic

CAL = materialize_xnys_schedule(fwd._CALENDAR_FIRST, fwd._CALENDAR_LAST)


def _session_after(day: date):
    return next(s for s in CAL.sessions if s.day > day)


class TestHelpers:
    def test_dt_rejects_non_string_and_naive(self) -> None:
        with pytest.raises(ValueError, match="ISO timestamp"):
            fwd._dt(20240101)
        with pytest.raises(ValueError, match="explicit UTC offset"):
            fwd._dt("2030-01-01T00:00:00")
        assert fwd._dt("2030-01-01T12:00:00+02:00") == datetime(2030, 1, 1, 10, tzinfo=UTC)
        assert fwd._dt("2030-01-01T00:00:00Z") == datetime(2030, 1, 1, tzinfo=UTC)

    def test_load_gzip_and_non_dict(self, tmp_path: Path) -> None:
        p = tmp_path / "x.json.gz"
        with gzip.open(p, "wt") as fh:
            json.dump({"a": 1}, fh)
        assert fwd._load(p) == {"a": 1}
        bad = tmp_path / "y.json"
        bad.write_text("[1, 2]")
        with pytest.raises(ValueError, match="JSON object"):
            fwd._load(bad)

    def test_sealed_mismatch_and_write_new_exclusive(self, tmp_path: Path) -> None:
        p = tmp_path / "m.json"
        p.write_text(json.dumps({"a": 1, "receipt_sha256": "f" * 64}))
        with pytest.raises(ValueError, match="receipt SHA-256 mismatch"):
            fwd._sealed(p)
        out = tmp_path / "n.json"
        fwd._write_new(out, {"b": 2})
        with pytest.raises(FileExistsError):
            fwd._write_new(out, {"b": 3})

    def test_hash_and_packet_hash_stable(self) -> None:
        packet = {"b": 1, "a": [1, 2]}
        assert fwd._packet_hash(packet) == fwd._packet_hash({"a": [1, 2], "b": 1})
        assert fwd._hash(b"x") == fwd._hash(b"x") != fwd._hash(b"y")

    def test_event_path_and_summary(self) -> None:
        assert fwd._event_path(Path("/r"), 7).name == "00000007.json"
        state = {"cursor": 0, "history": [{"a": 1}], "x": "y"}
        summary = fwd._summary(state)
        assert "history" not in summary and "history_sha256" in summary

    def test_warmup_cutoff(self) -> None:
        rows = [
            {
                "event_time": "2030-01-01T21:00:00+00:00",
                "available_time": "2030-01-01T21:00:01+00:00",
                "ingested_time": "2030-01-01T21:00:02+00:00",
            },
            {
                "event_time": "2030-01-02T21:00:00+00:00",
                "available_time": "2030-01-02T21:00:00+00:00",
                "ingested_time": "2030-01-02T21:00:00+00:00",
            },
        ]
        assert fwd._warmup_cutoff(rows) == datetime(2030, 1, 2, 21, tzinfo=UTC)


def _warmup_rows(n_sessions: int = 30) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for s in CAL.sessions[:n_sessions]:
        for sid in ("AAA", "BBB"):
            rows.append(
                {
                    "security_id": sid,
                    "event_time": s.close_utc.isoformat(),
                    "available_time": s.close_utc.isoformat(),
                    "ingested_time": (s.close_utc + timedelta(seconds=1)).isoformat(),
                    "close": 100.0 if sid == "AAA" else 50.0,
                    "volume": 1e6,
                }
            )
    return rows


def _manifest(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A manifest that satisfies every _check_manifest gate against the
    repository's real pinned artifacts; only the git identity is stubbed."""
    root = fwd._ROOT
    spec = json.loads((root / "configs/net_tournament.json").read_text())
    benchmark = fwd._sealed(root / "data/metadata/real_benchmark/us_wide_20260925/manifest.json")
    validation = fwd._sealed(
        root / "data/metadata/net_tournament/us_wide_20260925/validation.json.gz"
    )
    rev = ("0" * 40, "1" * 64)
    monkeypatch.setattr(fwd, "_checkout_state", lambda: rev)
    warmup = _warmup_rows()
    spec_sha = fwd._hash((root / "configs/net_tournament.json").read_bytes())
    power_sha = fwd._hash((root / "docs/FORWARD_SHADOW_POWER.md").read_bytes())
    code = fwd._source_hashes()
    freeze_dt = CAL.sessions[40].close_utc + timedelta(minutes=5)
    commitment = fwd._commitment(
        spec_sha256=spec_sha,
        benchmark_sha256=benchmark["receipt_sha256"],
        validation_sha256=validation["receipt_sha256"],
        code_sha256=code,
        warmup=warmup,
        power_sha256=power_sha,
        git_revision=rev[0],
        git_worktree_sha256=rev[1],
        schedule_sha256=CAL.schedule_sha256,
        historical_index_sha256=fwd._PUBLISHED_INDEX_SHA256,
    )
    return {
        "kind": "forward_shadow_manifest",
        "schema_version": 1,
        "created_at": (freeze_dt + timedelta(seconds=1)).isoformat(),
        "freeze": {
            "recorded_at": freeze_dt.isoformat(),
            "reference": "external-ref",
            "issuer": "external-issuer",
            "commitment_sha256": commitment,
        },
        "protocol_commitment_sha256": commitment,
        "strategy": "momentum_20",
        "benchmark": "equal_weight",
        "source": "yahoo",
        "expected_security_ids": ["AAA", "BBB"],
        "tournament_spec": spec,
        "spec_sha256": spec_sha,
        "benchmark_receipt_sha256": benchmark["receipt_sha256"],
        "validation_receipt_sha256": validation["receipt_sha256"],
        "power_plan_sha256": power_sha,
        "historical_index_receipt_sha256": fwd._PUBLISHED_INDEX_SHA256,
        "exchange_schedule": CAL.as_dict(),
        "min_paired_sessions": 1400,
        "warmup": warmup,
        "warmup_status": "historical_only_excluded_from_evidence",
        "dataset_sha256": benchmark["protocol"]["dataset_sha256"],
        "code_sha256": code,
        "git_revision": rev[0],
        "git_worktree_sha256": rev[1],
        "live_pnl_claim": False,
        "research_only": True,
        "external_attestation_verified": False,
        "limitations": [],
    }


def _make_run(tmp_path: Path, manifest: dict[str, Any]) -> Path:
    run = tmp_path / "run"
    (run / "events").mkdir(parents=True)
    fwd._write_new(run / "manifest.json", fwd._seal(manifest))
    return run


def _bar(sid: str, event: datetime, **kw: Any) -> dict[str, Any]:
    row = {
        "security_id": sid,
        "event_time": event.isoformat(),
        "available_time": event.isoformat(),
        "ingested_time": (event + timedelta(seconds=1)).isoformat(),
        "close": 100.0,
        "volume": 1e6,
        "open": 100.0,
    }
    row.update(kw)
    return row


def _close_packet(event: datetime, ids=("AAA", "BBB"), **kw: Any) -> dict[str, Any]:
    packet = {
        "kind": "forward_shadow_close",
        "source": "yahoo",
        "external_attestation": {
            "issuer": "external-issuer",
            "reference": "ref",
            "recorded_at": (event + timedelta(seconds=2)).isoformat(),
        },
        "bars": [_bar(sid, event) for sid in ids],
    }
    packet.update(kw)
    return packet


def _open_packet(event: datetime, ids=("AAA", "BBB"), **kw: Any) -> dict[str, Any]:
    packet = {
        "kind": "forward_shadow_open",
        "source": "yahoo",
        "external_attestation": {
            "issuer": "external-issuer",
            "reference": "ref2",
            "recorded_at": (event + timedelta(seconds=2)).isoformat(),
        },
        "bars": [_bar(sid, event, open=101.0) for sid in ids],
    }
    packet.update(kw)
    return packet


def _before() -> dict[str, Any]:
    return {"last_close": None, "last_open": None, "last_observed_at": None}


class TestCheckManifestGuards:
    def test_clean_manifest_passes(self, monkeypatch) -> None:
        fwd._check_manifest(_manifest(monkeypatch))

    def test_kind_and_honesty_gates(self, monkeypatch) -> None:
        m = _manifest(monkeypatch)
        m["kind"] = "other"
        with pytest.raises(ValueError, match="kind/schema"):
            fwd._check_manifest(m)
        m = _manifest(monkeypatch)
        m["live_pnl_claim"] = True
        with pytest.raises(ValueError, match="honesty"):
            fwd._check_manifest(m)
        m = _manifest(monkeypatch)
        m["research_only"] = False
        with pytest.raises(ValueError, match="honesty"):
            fwd._check_manifest(m)

    def test_strategy_and_horizon_gates(self, monkeypatch) -> None:
        m = _manifest(monkeypatch)
        m["strategy"] = "other"
        with pytest.raises(ValueError, match="comparator"):
            fwd._check_manifest(m)
        m = _manifest(monkeypatch)
        m["min_paired_sessions"] = 5
        with pytest.raises(ValueError, match="power plan"):
            fwd._check_manifest(m)

    def test_calendar_bounds_gate(self, monkeypatch) -> None:
        m = _manifest(monkeypatch)
        other = materialize_xnys_schedule(date(2027, 1, 1), fwd._CALENDAR_LAST)
        m["exchange_schedule"] = other.as_dict()
        with pytest.raises(ValueError, match="bounds"):
            fwd._check_manifest(m)

    def test_code_and_git_gates(self, monkeypatch) -> None:
        m = _manifest(monkeypatch)
        m["code_sha256"] = {**m["code_sha256"], "x": "0" * 64}
        with pytest.raises(ValueError, match="source code changed"):
            fwd._check_manifest(m)
        m = _manifest(monkeypatch)
        m["git_revision"] = "9" * 40
        with pytest.raises(ValueError, match="worktree"):
            fwd._check_manifest(m)

    def test_spec_and_power_gates(self, monkeypatch) -> None:
        m = _manifest(monkeypatch)
        m["spec_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="spec hash"):
            fwd._check_manifest(m)
        m = _manifest(monkeypatch)
        m["tournament_spec"] = {"other": 1}
        with pytest.raises(ValueError, match="spec content"):
            fwd._check_manifest(m)
        m = _manifest(monkeypatch)
        m["power_plan_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="power analysis"):
            fwd._check_manifest(m)

    def test_receipt_consistency_gate(self, monkeypatch) -> None:
        m = _manifest(monkeypatch)
        m["benchmark_receipt_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="receipts"):
            fwd._check_manifest(m)

    def test_warmup_and_freeze_gates(self, monkeypatch) -> None:
        m = _manifest(monkeypatch)
        m["warmup"] = []
        with pytest.raises(ValueError, match="warmup"):
            fwd._check_manifest(m)
        m = _manifest(monkeypatch)
        m["expected_security_ids"] = ["AAA", "ZZZ"]
        with pytest.raises(ValueError, match="universe"):
            fwd._check_manifest(m)
        m = _manifest(monkeypatch)
        m["freeze"]["reference"] = ""
        with pytest.raises(ValueError, match="freeze"):
            fwd._check_manifest(m)
        m = _manifest(monkeypatch)
        m["freeze"]["commitment_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="commitment"):
            fwd._check_manifest(m)


class TestClosePacket:
    def test_kind_and_shape(self, monkeypatch) -> None:
        m = _manifest(monkeypatch)
        before = _before()
        with pytest.raises(ValueError, match="forward_shadow_close"):
            fwd._close_packet({"kind": "x"}, m, before, datetime.now(UTC))
        with pytest.raises(ValueError, match="two names"):
            fwd._close_packet(
                {"kind": "forward_shadow_close", "bars": [{}]}, m, before, datetime.now(UTC)
            )

    def test_single_session_unique_names(self, monkeypatch) -> None:
        m = _manifest(monkeypatch)
        freeze = fwd._dt(m["freeze"]["recorded_at"])
        first = _session_after(freeze.date())
        packet = _close_packet(first.close_utc)
        packet["bars"] = [
            _bar("AAA", first.close_utc),
            _bar("BBB", first.close_utc + timedelta(seconds=1)),
        ]
        with pytest.raises(ValueError, match="one session"):
            fwd._close_packet(packet, m, _before(), first.close_utc + timedelta(seconds=5))

    def test_not_a_scheduled_close(self, monkeypatch) -> None:
        m = _manifest(monkeypatch)
        freeze = fwd._dt(m["freeze"]["recorded_at"])
        first = _session_after(freeze.date())
        with pytest.raises(ValueError, match="scheduled XNYS close"):
            fwd._close_packet(
                _close_packet(first.open_utc + timedelta(hours=1)),
                m,
                _before(),
                first.close_utc + timedelta(seconds=5),
            )

    def test_first_session_must_be_first(self, monkeypatch) -> None:
        m = _manifest(monkeypatch)
        freeze = fwd._dt(m["freeze"]["recorded_at"])
        second = _session_after(_session_after(freeze.date()).day)
        with pytest.raises(ValueError, match="first scheduled XNYS session"):
            fwd._close_packet(
                _close_packet(second.close_utc),
                m,
                _before(),
                second.close_utc + timedelta(seconds=5),
            )

    def test_universe_mismatch(self, monkeypatch) -> None:
        m = _manifest(monkeypatch)
        freeze = fwd._dt(m["freeze"]["recorded_at"])
        first = _session_after(freeze.date())
        packet = _close_packet(first.close_utc, ids=("AAA", "ZZZ"))
        with pytest.raises(ValueError, match="missing/extra"):
            fwd._close_packet(packet, m, _before(), first.close_utc + timedelta(seconds=5))

    def test_bad_missed_session_reason(self, monkeypatch) -> None:
        m = _manifest(monkeypatch)
        freeze = fwd._dt(m["freeze"]["recorded_at"])
        first = _session_after(freeze.date())
        packet = _close_packet(
            first.close_utc, missed_sessions=[{"date": "x", "reason": "no_feed"}]
        )
        with pytest.raises(ValueError, match="market_closed"):
            fwd._close_packet(packet, m, _before(), first.close_utc + timedelta(seconds=5))

    def test_bar_validations(self, monkeypatch) -> None:
        m = _manifest(monkeypatch)
        freeze = fwd._dt(m["freeze"]["recorded_at"])
        first = _session_after(freeze.date())
        packet = _close_packet(first.close_utc)
        packet["bars"][0]["close"] = -1.0
        with pytest.raises(ValueError, match="finite and positive"):
            fwd._close_packet(packet, m, _before(), first.close_utc + timedelta(seconds=5))
        packet = _close_packet(first.close_utc)
        packet["bars"][0]["available_time"] = (first.close_utc + timedelta(seconds=1)).isoformat()
        with pytest.raises(ValueError, match="decision cutoff"):
            fwd._close_packet(packet, m, _before(), first.close_utc + timedelta(seconds=5))
        packet = _close_packet(first.close_utc)
        packet["bars"][0].pop("volume")
        with pytest.raises(ValueError, match="causal timestamps"):
            fwd._close_packet(packet, m, _before(), first.close_utc + timedelta(seconds=5))

    def test_attestation_gates(self, monkeypatch) -> None:
        m = _manifest(monkeypatch)
        freeze = fwd._dt(m["freeze"]["recorded_at"])
        first = _session_after(freeze.date())
        packet = _close_packet(first.close_utc)
        packet["external_attestation"] = {"issuer": "i", "reference": "r"}
        with pytest.raises(ValueError, match="reference required"):
            fwd._close_packet(packet, m, _before(), first.close_utc + timedelta(seconds=5))
        packet = _close_packet(first.close_utc)
        packet["source"] = "other"
        with pytest.raises(ValueError, match="source"):
            fwd._close_packet(packet, m, _before(), first.close_utc + timedelta(seconds=5))
        packet = _close_packet(first.close_utc)
        packet["external_attestation"]["recorded_at"] = (
            first.close_utc - timedelta(seconds=1)
        ).isoformat()
        with pytest.raises(ValueError, match="bounds"):
            fwd._close_packet(packet, m, _before(), first.close_utc + timedelta(seconds=5))

    def test_valid_packet_returns_bars(self, monkeypatch) -> None:
        m = _manifest(monkeypatch)
        freeze = fwd._dt(m["freeze"]["recorded_at"])
        first = _session_after(freeze.date())
        bars, event = fwd._close_packet(
            _close_packet(first.close_utc),
            m,
            _before(),
            first.close_utc + timedelta(seconds=5),
        )
        assert len(bars) == 2 and event == first.close_utc

    def test_backfilled_close_rejected(self, monkeypatch) -> None:
        m = _manifest(monkeypatch)
        freeze = fwd._dt(m["freeze"]["recorded_at"])
        first = _session_after(freeze.date())
        before = _before()
        before["last_close"] = first.close_utc.isoformat()
        with pytest.raises(ValueError, match="backfilled"):
            fwd._close_packet(
                _close_packet(first.close_utc), m, before, first.close_utc + timedelta(seconds=5)
            )


class TestOpenPacket:
    def _open_before(self, monkeypatch) -> tuple[dict, dict, Any, Any]:
        m = _manifest(monkeypatch)
        freeze = fwd._dt(m["freeze"]["recorded_at"])
        close = _session_after(freeze.date())
        nxt = _session_after(close.day)
        before = {
            "last_close": close.close_utc.isoformat(),
            "last_open": None,
            "last_observed_at": close.close_utc.isoformat(),
        }
        return m, before, close, nxt

    def test_kind_shape_universe(self, monkeypatch) -> None:
        m, before, close, nxt = self._open_before(monkeypatch)
        with pytest.raises(ValueError, match="forward_shadow_open"):
            fwd._open_packet({"kind": "x"}, m, before, datetime.now(UTC))
        with pytest.raises(ValueError, match="bars"):
            fwd._open_packet(
                {"kind": "forward_shadow_open", "bars": []}, m, before, datetime.now(UTC)
            )
        packet = _open_packet(nxt.open_utc, ids=("AAA", "ZZZ"))
        with pytest.raises(ValueError, match="missing/extra"):
            fwd._open_packet(packet, m, before, nxt.open_utc + timedelta(seconds=5))

    def test_must_be_immediate_next_session(self, monkeypatch) -> None:
        m, before, close, nxt = self._open_before(monkeypatch)
        later = _session_after(nxt.day)
        packet = _open_packet(later.open_utc)
        with pytest.raises(ValueError, match="immediately next"):
            fwd._open_packet(packet, m, before, nxt.open_utc + timedelta(seconds=5))

    def test_monotonicity(self, monkeypatch) -> None:
        m, before, close, nxt = self._open_before(monkeypatch)
        before["last_observed_at"] = (nxt.open_utc + timedelta(seconds=10)).isoformat()
        with pytest.raises(ValueError, match="observed close"):
            fwd._open_packet(
                _open_packet(nxt.open_utc), m, before, nxt.open_utc + timedelta(seconds=5)
            )

    def test_bar_price_and_attestation(self, monkeypatch) -> None:
        m, before, close, nxt = self._open_before(monkeypatch)
        before["last_observed_at"] = (close.close_utc).isoformat()
        packet = _open_packet(nxt.open_utc)
        packet["bars"][0].pop("open")
        with pytest.raises(ValueError, match="open"):
            fwd._open_packet(packet, m, before, nxt.open_utc + timedelta(seconds=5))
        packet = _open_packet(nxt.open_utc)
        packet["bars"][0]["open"] = -1.0
        with pytest.raises(ValueError, match="finite and positive"):
            fwd._open_packet(packet, m, before, nxt.open_utc + timedelta(seconds=5))

    def test_valid_open_returns_prices(self, monkeypatch) -> None:
        m, before, close, nxt = self._open_before(monkeypatch)
        before["last_observed_at"] = close.close_utc.isoformat()
        prices, event = fwd._open_packet(
            _open_packet(nxt.open_utc), m, before, nxt.open_utc + timedelta(seconds=5)
        )
        assert prices == {"AAA": 101.0, "BBB": 101.0} and event == nxt.open_utc


def _write_packet(tmp_path: Path, packet: dict[str, Any]) -> Path:
    p = tmp_path / "packet.json"
    p.write_text(json.dumps(packet))
    return p


class TestDecideExecuteVerify:
    def test_full_round_trip_and_verify(self, tmp_path, monkeypatch) -> None:
        m = _manifest(monkeypatch)
        run = _make_run(tmp_path, m)
        freeze = fwd._dt(m["freeze"]["recorded_at"])
        close = _session_after(freeze.date())
        nxt = _session_after(close.day)

        close_packet = _write_packet(tmp_path, _close_packet(close.close_utc))
        receipt1 = fwd.decide(run, close_packet, now=close.close_utc + timedelta(seconds=5))
        assert receipt1["live_pnl_claim"] is False
        assert receipt1["research_only"] is True

        # Phase gate: another close before the open is rejected.
        with pytest.raises(ValueError, match="reconcile"):
            fwd.decide(run, close_packet, now=close.close_utc + timedelta(seconds=6))

        open_packet = _write_packet(tmp_path, _open_packet(nxt.open_utc))
        receipt2 = fwd.execute(run, open_packet, now=nxt.open_utc + timedelta(seconds=5))
        assert receipt2["live_pnl_claim"] is False

        out = fwd.verify(run)
        assert out["valid"] is True
        assert out["paired_sessions"] == 1
        assert out["live_pnl_claim"] is False and out["research_only"] is True
        assert out["external_attestation_verified"] is False

    def test_synthetic_packet_rejected(self, tmp_path, monkeypatch) -> None:
        m = _manifest(monkeypatch)
        run = _make_run(tmp_path, m)
        freeze = fwd._dt(m["freeze"]["recorded_at"])
        close = _session_after(freeze.date())
        packet = _close_packet(close.close_utc, source="synthetic-feed")
        p = _write_packet(tmp_path, packet)
        with pytest.raises(ValueError, match="synthetic"):
            fwd.decide(run, p, now=close.close_utc + timedelta(seconds=5))

    def test_decide_on_missing_run(self, tmp_path, monkeypatch) -> None:
        run = tmp_path / "nope"
        run.mkdir()
        (run / "manifest.json").write_text("{}")
        (run / "events").mkdir()
        out = fwd.verify(run)
        assert out["valid"] is False
