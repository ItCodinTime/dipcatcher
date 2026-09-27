"""The paired paper journal never turns historical input into forward evidence."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.execution.costs import total_cost
from quant_fund.paper import forward_shadow as fw
from quant_fund.research.real_benchmark import _seal
from quant_fund.utils.reproducibility import git_revision


def _time(day: int, hour: int = 20, minute: int = 0) -> datetime:
    return datetime(2026, 9, 1, hour, minute, tzinfo=UTC) + timedelta(days=day)


def _bars(day: int, *, open_price: bool = False) -> list[dict]:
    moment = _time(day, 13, 30) if open_price else _time(day, 20)
    return [
        {
            "security_id": sid,
            "event_time": moment.isoformat(),
            "available_time": moment.isoformat(),
            "ingested_time": moment.isoformat(),
            ("open" if open_price else "close"): price,
            **({} if open_price else {"volume": 1e6}),
        }
        for sid, price in (("A", 100.0 + day), ("B", 100.0 - day / 2))
    ]


def _packet(day: int, *, open_price: bool = False) -> dict:
    moment = _time(day, 13, 30) if open_price else _time(day, 20)
    return {
        "kind": "forward_shadow_open" if open_price else "forward_shadow_close",
        "source": "yahoo",
        "bars": _bars(day, open_price=open_price),
        "external_attestation": {
            "issuer": "example-provider",
            "reference": f"external:{day}",
            "recorded_at": (moment + timedelta(minutes=1)).isoformat(),
        },
    }


def _put(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data))
    return path


@pytest.fixture(scope="module")
def schedule() -> dict:
    return fw.materialize_xnys_schedule(fw._CALENDAR_FIRST, fw._CALENDAR_LAST).as_dict()


@pytest.fixture
def sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, schedule: dict
) -> tuple[Path, Path, Path]:
    run = tmp_path / "run"
    (run / "events").mkdir(parents=True)
    revision = git_revision()
    monkeypatch.setattr(fw, "_checkout_state", lambda: (revision, fw._hash(b"")))
    spec = json.loads(Path("configs/net_tournament.json").read_text())
    warmup = [row for day in range(3, 24) for row in _bars(day)]
    benchmark = fw._sealed(Path("data/metadata/real_benchmark/us_wide_20260925/manifest.json"))
    validation = fw._sealed(
        Path("data/metadata/net_tournament/us_wide_20260925/validation.json.gz")
    )
    source_sha = fw._source_hashes()
    spec_sha = fw._hash(Path("configs/net_tournament.json").read_bytes())
    power_sha = fw._hash(Path("docs/FORWARD_SHADOW_POWER.md").read_bytes())
    commitment = fw._commitment(
        spec_sha256=spec_sha,
        benchmark_sha256=benchmark["receipt_sha256"],
        validation_sha256=validation["receipt_sha256"],
        code_sha256=source_sha,
        warmup=warmup,
        power_sha256=power_sha,
        git_revision=revision,
        git_worktree_sha256=fw._hash(b""),
        schedule_sha256=schedule["schedule_sha256"],
        historical_index_sha256=fw._PUBLISHED_INDEX_SHA256,
    )
    manifest = _seal(
        {
            "kind": "forward_shadow_manifest",
            "schema_version": 1,
            "created_at": _time(24, 0).isoformat(),
            "freeze": {
                "recorded_at": _time(23, 23).isoformat(),
                "issuer": "test-witness",
                "reference": "external:test-protocol",
                "commitment_sha256": commitment,
            },
            "protocol_commitment_sha256": commitment,
            "source": "yahoo",
            "expected_security_ids": ["A", "B"],
            "tournament_spec": spec,
            "spec_sha256": spec_sha,
            "benchmark_receipt_sha256": benchmark["receipt_sha256"],
            "validation_receipt_sha256": validation["receipt_sha256"],
            "dataset_sha256": benchmark["protocol"]["dataset_sha256"],
            "power_plan_sha256": power_sha,
            "historical_index_receipt_sha256": fw._PUBLISHED_INDEX_SHA256,
            "exchange_schedule": schedule,
            "git_revision": revision,
            "git_worktree_sha256": fw._hash(b""),
            "strategy": "momentum_20",
            "benchmark": "equal_weight",
            "warmup": warmup,
            "min_paired_sessions": 1400,
            "code_sha256": source_sha,
            "research_only": True,
            "live_pnl_claim": False,
            "external_attestation_verified": False,
        }
    )
    _put(run / "manifest.json", manifest)
    return (
        run,
        _put(tmp_path / "close.json", _packet(24)),
        _put(tmp_path / "open.json", _packet(27, open_price=True)),
    )


def test_paired_next_open_costed_books_and_restart(sample: tuple[Path, Path, Path]) -> None:
    run, close, opening = sample
    first = fw.decide(run, close, now=_time(24, 22))
    assert first["intended_orders"]["momentum_20:configured"]
    assert "orders" not in first  # Decisions have no open price or filled order.
    assert first["before"]["paired_sessions"] == 0
    assert fw.verify(run)["state"] == "open"
    with pytest.raises(ValueError, match="next-open"):
        fw.decide(run, close, now=_time(24, 22))
    second = fw.execute(run, opening, now=_time(27, 15))
    assert fw.verify(run)["valid"] is True
    assert fw.verify(run)["paired_sessions"] == 1
    assert fw.verify(run)["deterministic_close_replay"] is True
    assert fw.verify(run)["independent_strategy_replay"] is False
    assert second["after"]["phase"] == "close"
    for scenario in fw.SCENARIOS:
        for name in fw.BOOKS:
            key = f"{name}:{scenario}"
            rows = second["orders"][key]
            assert len(rows) == len(first["intended_orders"][key])
            assert all(row["fill_price"] in (None, 127.0, 86.5) for row in rows)
            assert all(abs(row["fill_quantity"]) <= abs(row["requested_quantity"]) for row in rows)
            cash = second["after"]["books"][key]["cash"]
            before_cash = first["after"]["books"][key]["cash"]
            assert cash == pytest.approx(
                before_cash + sum(x["delta"] for x in second["cash_ledger"][key])
            )
            assert second["daily"][key]["nav"] == pytest.approx(
                cash
                + sum(
                    second["positions"][key][sid] * px
                    for sid, px in {"A": 127.0, "B": 86.5}.items()
                    if sid in second["positions"][key]
                )
            )
            for row in rows:
                if row["fill_quantity"]:
                    cfg = fw._configured_books(
                        json.loads(Path("configs/net_tournament.json").read_text()), scenario
                    )
                    expected = total_cost(
                        row["fill_quantity"],
                        row["fill_price"],
                        row["known_adv"],
                        row["known_volatility"],
                        cfg.costs,
                    )
                    assert row["commission"] == pytest.approx(expected["commission"])
                    assert row["spread"] == pytest.approx(expected["spread"])
                    assert row["impact"] == pytest.approx(expected["impact"])
                    assert abs(row["fill_quantity"] * row["fill_price"]) <= (
                        cfg.costs.participation_limit * row["known_adv"] + 1e-7
                    )
    assert (
        second["costs"]["momentum_20:double_impact"]["impact"]
        >= (second["costs"]["momentum_20:configured"]["impact"])
    )
    with pytest.raises(ValueError, match="prior close"):
        fw.execute(run, opening, now=_time(27, 15))
    # Fresh invocations reload the sealed journal and reconcile the prior books.
    _put(close, _packet(27))
    fw.decide(run, close, now=_time(27, 22))
    _put(opening, _packet(28, open_price=True))
    fw.execute(run, opening, now=_time(28, 15))
    assert fw.verify(run)["paired_sessions"] == 2


def test_equivalent_offset_close_remains_verifiable_after_next_open(
    sample: tuple[Path, Path, Path],
) -> None:
    run, close, opening = sample
    packet = _packet(24)
    packet["bars"][0]["event_time"] = "2026-09-25T20:00:00Z"
    packet["bars"][1]["event_time"] = "2026-09-25T16:00:00-04:00"
    _put(close, packet)
    fw.decide(run, close, now=_time(24, 22).astimezone(timezone(timedelta(hours=-4))))
    assert fw.verify(run)["valid"] is True
    fw.execute(run, opening, now=_time(27, 15))
    report = fw.verify(run)
    assert report["valid"] is True, report["errors"]
    assert report["paired_sessions"] == 1


def test_rejects_backfill_and_mutated_history(sample: tuple[Path, Path, Path]) -> None:
    run, close, _ = sample
    past = _packet(23)
    _put(close, past)
    with pytest.raises(ValueError, match="backfilled"):
        fw.decide(run, close, now=_time(24, 22))
    _put(close, _packet(24))
    fw.decide(run, close, now=_time(24, 22))
    event = run / "events" / "00000001.json"
    modified = json.loads(event.read_text())
    modified["intended_orders"]["equal_weight:configured"] = []
    _put(event, modified)
    assert fw.verify(run)["valid"] is False
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        fw.decide(run, close, now=_time(24, 22))


def test_source_attestation_and_temporal_order_are_required(
    sample: tuple[Path, Path, Path],
) -> None:
    run, close, _ = sample
    packet = json.loads(close.read_text())
    packet["external_attestation"]["recorded_at"] = _time(24, 23).isoformat()
    _put(close, packet)
    with pytest.raises(ValueError, match="outside event/observation"):
        fw.decide(run, close, now=_time(24, 22))
    packet["external_attestation"]["recorded_at"] = _time(24, 21).isoformat()
    packet["bars"][0]["available_time"] = _time(24, 21).isoformat()
    packet["bars"][0]["ingested_time"] = _time(24, 21).isoformat()
    _put(close, packet)
    with pytest.raises(ValueError, match="decision cutoff"):
        fw.decide(run, close, now=_time(24, 22))


def test_delayed_close_cannot_backfill_open(sample: tuple[Path, Path, Path]) -> None:
    run, close, opening = sample
    fw.decide(run, close, now=_time(27, 14))
    with pytest.raises(ValueError, match="open must follow the observed close"):
        fw.execute(run, opening, now=_time(27, 15))
    assert fw.verify(run)["paired_sessions"] == 0


def test_partial_universe_and_skipped_weekday_are_rejected(sample: tuple[Path, Path, Path]) -> None:
    run, close, _ = sample
    partial = _packet(24)
    partial["bars"].pop()
    _put(close, partial)
    with pytest.raises(ValueError, match="at least two names|missing/extra frozen-universe"):
        fw.decide(run, close, now=_time(24, 22))
    later = _packet(27)
    _put(close, later)
    with pytest.raises(ValueError, match="first scheduled XNYS session"):
        fw.decide(run, close, now=_time(27, 22))
    later["missed_sessions"] = [{"date": "2026-09-25", "reason": "no_feed"}]
    _put(close, later)
    with pytest.raises(ValueError, match="no_feed/downtime must interrupt"):
        fw.decide(run, close, now=_time(27, 22))


def test_open_must_be_a_separate_next_market_session(sample: tuple[Path, Path, Path]) -> None:
    run, close, opening = sample
    fw.decide(run, close, now=_time(24, 22))
    _put(opening, _packet(24, open_price=True))
    with pytest.raises(ValueError, match="immediately next"):
        fw.execute(run, opening, now=_time(27, 15))
    _put(opening, _packet(30, open_price=True))
    with pytest.raises(ValueError, match="immediately next"):
        fw.execute(run, opening, now=_time(30, 15))
    assert fw.verify(run)["paired_sessions"] == 0


def test_packet_instants_must_match_frozen_exchange_hours(sample: tuple[Path, Path, Path]) -> None:
    run, close, opening = sample
    shifted_close = _packet(24)
    for row in shifted_close["bars"]:
        for field in ("event_time", "available_time", "ingested_time"):
            row[field] = _time(24, 21).isoformat()
    shifted_close["external_attestation"]["recorded_at"] = _time(24, 21, 1).isoformat()
    _put(close, shifted_close)
    with pytest.raises(ValueError, match="scheduled XNYS close"):
        fw.decide(run, close, now=_time(24, 22))
    _put(close, _packet(24))
    fw.decide(run, close, now=_time(24, 22))
    shifted_open = _packet(27, open_price=True)
    for row in shifted_open["bars"]:
        for field in ("event_time", "available_time", "ingested_time"):
            row[field] = _time(27, 13).isoformat()
    shifted_open["external_attestation"]["recorded_at"] = _time(27, 13, 1).isoformat()
    _put(opening, shifted_open)
    with pytest.raises(ValueError, match="scheduled XNYS open"):
        fw.execute(run, opening, now=_time(27, 15))
    assert fw.verify(run)["paired_sessions"] == 0


def test_resealed_manifest_horizon_cannot_change(sample: tuple[Path, Path, Path]) -> None:
    run, _, _ = sample
    original = json.loads((run / "manifest.json").read_text())
    original.pop("receipt_sha256")
    original["min_paired_sessions"] = 2
    _put(run / "manifest.json", _seal(original))
    assert fw.verify(run)["valid"] is False
    assert "horizon" in fw.verify(run)["errors"][0]


def _recommit(manifest: dict) -> dict:
    manifest.pop("receipt_sha256", None)
    commitment = fw._commitment(
        spec_sha256=manifest["spec_sha256"],
        benchmark_sha256=manifest["benchmark_receipt_sha256"],
        validation_sha256=manifest["validation_receipt_sha256"],
        code_sha256=manifest["code_sha256"],
        warmup=manifest["warmup"],
        power_sha256=manifest["power_plan_sha256"],
        git_revision=manifest["git_revision"],
        git_worktree_sha256=manifest["git_worktree_sha256"],
        schedule_sha256=manifest["exchange_schedule"]["schedule_sha256"],
        historical_index_sha256=manifest["historical_index_receipt_sha256"],
    )
    manifest["protocol_commitment_sha256"] = commitment
    manifest["freeze"]["commitment_sha256"] = commitment
    return _seal(manifest)


def test_resealed_fake_git_revision_is_rejected(sample: tuple[Path, Path, Path]) -> None:
    run, _, _ = sample
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["git_revision"] = "f" * 40
    _put(run / "manifest.json", _recommit(manifest))
    report = fw.verify(run)
    assert report["valid"] is False
    assert "Git commit" in report["errors"][0]


def test_freeze_follows_all_warmup_ingestion(sample: tuple[Path, Path, Path]) -> None:
    run, _, _ = sample
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["warmup"][0]["ingested_time"] = _time(24, 0).isoformat()
    _put(run / "manifest.json", _recommit(manifest))
    report = fw.verify(run)
    assert report["valid"] is False
    assert "external freeze timestamp" in report["errors"][0]


def test_verify_from_outside_repository(
    sample: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run, _, _ = sample
    monkeypatch.chdir(tmp_path)
    assert fw.verify(run)["valid"] is True


def test_published_commitment_loads_archived_receipts_and_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bronze = fw._ROOT / "data/file_us_wide/bronze/bars.parquet"
    if not bronze.is_file():
        pytest.skip("published historical bronze snapshot is not present")
    monkeypatch.chdir(fw._ROOT)
    revision = git_revision()
    monkeypatch.setattr(fw, "_checkout_state", lambda: (revision, fw._hash(b"")))
    result = CliRunner().invoke(app, ["paper", "--forward-stage", "commitment"])
    assert result.exit_code == 0, result.output
    output = json.loads(result.stdout[result.stdout.index("{") :])
    assert output["git_revision"] == revision
    assert output["git_worktree_sha256"] == fw._hash(b"")
    assert len(output["protocol_commitment_sha256"]) == 64
    assert output["last_historical_warmup_session"] == "2026-09-18T20:00:00+00:00"
    assert output["research_only"] is True
    assert output["live_pnl_claim"] is False


def test_resealed_open_metrics_and_attestation_fail_verify(sample: tuple[Path, Path, Path]) -> None:
    run, close, opening = sample
    fw.decide(run, close, now=_time(24, 22))
    fw.execute(run, opening, now=_time(27, 15))
    event = run / "events" / "00000002.json"
    original = json.loads(event.read_text())
    original.pop("receipt_sha256")
    original["daily"]["equal_weight:configured"]["nav"] += 100
    _put(event, _seal(original))
    assert "NAV" in fw.verify(run)["errors"][0]
    original["daily"]["equal_weight:configured"]["nav"] -= 100
    original["packet"]["external_attestation"]["reference"] = ""
    original["packet_sha256"] = fw._packet_hash(original["packet"])
    _put(event, _seal(original))
    assert "external attestation" in fw.verify(run)["errors"][0]


@pytest.mark.parametrize(
    ("tamper", "error"),
    [
        ("weights", "weights or decisions"),
        ("orders", "intended orders"),
        ("cash", "broker or state"),
        ("shares", "broker or state"),
    ],
)
def test_resealed_close_decision_and_book_fail_replay(
    sample: tuple[Path, Path, Path], tamper: str, error: str
) -> None:
    run, close, _ = sample
    fw.decide(run, close, now=_time(24, 22))
    event = run / "events" / "00000001.json"
    forged = json.loads(event.read_text())
    forged.pop("receipt_sha256")
    key = "momentum_20:configured"
    if tamper == "weights":
        forged["decisions"][key]["target_weights"]["A"] += 0.01
    elif tamper == "orders":
        forged["intended_orders"][key][0]["requested_quantity"] += 1
        forged["after"]["pending"][key][0]["requested_quantity"] += 1
    elif tamper == "cash":
        forged["after"]["books"][key]["cash"] += 100
    else:
        forged["after"]["books"][key]["shares"]["A"] = 100.0
    _put(event, _seal(forged))
    report = fw.verify(run)
    assert report["valid"] is False
    assert error in report["errors"][0]


def test_missing_feed_interruption_is_permanent_and_visible(
    sample: tuple[Path, Path, Path],
) -> None:
    run, close, _ = sample
    stopped = fw.interrupt(
        run,
        "no_feed",
        now=_time(24, 10),
        attempted_stage="decide",
        attempted_packet=close,
        error="upstream feed unavailable",
    )
    assert stopped["packet"]["attempted_packet_sha256"] == fw._packet_hash(_packet(24))
    report = fw.verify(run)
    assert report["valid"] is True
    assert report["state"] == "blocked"
    assert report["paired_sessions"] == 0
    assert report["forward_evidence_accepted"] is False
    with pytest.raises(ValueError, match="next-open"):
        fw.decide(run, close, now=_time(24, 22))
    event = run / "events" / "00000001.json"
    tampered = json.loads(event.read_text())
    tampered.pop("receipt_sha256")
    tampered["after"]["books"]["equal_weight:configured"]["cash"] += 10
    _put(event, _seal(tampered))
    assert fw.verify(run)["valid"] is False
    assert "interruption event" in fw.verify(run)["errors"][0]


def test_paper_cli_interrupt_and_verify_research_dispatch(sample: tuple[Path, Path, Path]) -> None:
    run, _, _ = sample
    runner = CliRunner()
    stopped = runner.invoke(
        app,
        [
            "paper",
            "--forward-stage",
            "interrupt",
            "--forward-run",
            str(run),
            "--forward-reason",
            "no_feed",
        ],
    )
    assert stopped.exit_code == 0, stopped.output
    assert "LOCAL_LEDGER_VERIFY_ONLY" in stopped.stdout
    verified = runner.invoke(app, ["verify-research", str(run)])
    assert verified.exit_code == 0, verified.output
    report = json.loads(verified.stdout)
    assert report["valid"] is True
    assert report["state"] == "blocked"
    assert report["forward_evidence_accepted"] is False
