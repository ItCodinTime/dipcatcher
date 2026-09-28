"""Point-in-time reality trial: membership, grid lock, and append-only ledger."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from quant_fund.config.loader import load_config
from quant_fund.proofcore.contracts import sha256_hex_bytes
from quant_fund.research.reality_survivorship import (
    anchor_coverage,
    append_delisting_exits,
    append_ledger_lines,
    assert_prior_ledger,
    emit_weight_rows,
    formation_return,
    gate_dip_weights,
    grid_cells,
    load_membership,
    load_spec,
    long_short_weights,
    members_asof,
    spy_regime,
    with_eligibility,
    yahoo_symbol,
)
from quant_fund.research.reality_sweep import assert_cost_lock

_ROOT = Path(__file__).resolve().parents[2]
_SPEC = _ROOT / "research" / "reality" / "survivorship" / "preregistration.json"
_MEMBERSHIP = _ROOT / "research" / "reality" / "survivorship" / "membership.json"
_LEDGER = _ROOT / "research" / "reality" / "trials.jsonl"


def _spec() -> dict:
    return load_spec(_SPEC)


def test_grid_matches_the_frozen_survivorship_preregistration() -> None:
    spec = _spec()
    cells = grid_cells(spec)
    assert len(cells) == 13
    assert sum(cell.strategy == "xs_momentum" for cell in cells) == 3
    assert sum(cell.strategy == "st_reversal" for cell in cells) == 2
    assert sum(cell.strategy == "low_vol" for cell in cells) == 3
    assert sum(cell.strategy == "dip_regime" for cell in cells) == 4
    assert sum(cell.strategy == "equal_weight_pit" for cell in cells) == 1
    assert len({cell.trial_id(spec["study_id"]) for cell in cells}) == 13
    assert spec["expected_trials"]["ledger_total"] == 42
    assert spec["reality_filter"]["thresholds_unchanged"] is True
    assert spec["reality_filter"]["dsr_pass"] == 0.95


def test_cost_lock_accepts_the_new_spec() -> None:
    spec = _spec()
    config = load_config(_ROOT / "configs" / "backtest.yaml")
    assert_cost_lock(config, spec)


def test_membership_file_matches_the_pinned_hash() -> None:
    spec = _spec()
    assert sha256_hex_bytes(_MEMBERSHIP.read_bytes()) == spec["data"]["membership_sha256"]


def test_members_asof_undoes_only_later_events() -> None:
    data = {
        "current": ["AAA", "BBB"],
        "changes": [
            {"date": "2020-01-03", "added": "BBB", "removed": "CCC"},
            {"date": "2020-01-01", "added": "AAA", "removed": ""},
        ],
    }
    assert members_asof(data, "2020-01-02") == {"AAA", "CCC"}
    assert members_asof(data, "2020-01-03") == {"AAA", "BBB"}
    assert "CCC" not in members_asof(data, "2020-01-03")


def test_vendored_log_drops_sivb_on_its_removal_date() -> None:
    data = load_membership(_MEMBERSHIP)
    assert "SIVB" in members_asof(data, "2023-03-14")
    assert "SIVB" not in members_asof(data, "2023-03-15")
    assert len(members_asof(data, "2016-06-30")) == 509
    assert "TWTR" in members_asof(data, "2020-06-30")
    assert "TWTR" not in members_asof(data, "2023-01-03")


def test_yahoo_symbol_only_rewrites_the_share_class_dot() -> None:
    assert yahoo_symbol("BRK.B") == "BRK-B"
    assert yahoo_symbol("BF.B") == "BF-B"
    assert yahoo_symbol("META") == "META"


def test_prior_ledger_prefix_still_matches_the_frozen_hash() -> None:
    spec = _spec()
    lines = _LEDGER.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 29
    prefix = ("\n".join(lines[:29]) + "\n").encode()
    assert sha256_hex_bytes(prefix) == spec["prior_study"]["ledger_sha256"]
    if len(lines) == 29:
        assert_prior_ledger(_LEDGER, spec)


def test_append_ledger_keeps_the_prefix(tmp_path: Path) -> None:
    path = tmp_path / "trials.jsonl"
    original = b'{"trial_id": "aa"}\n{"trial_id": "bb"}\n'
    path.write_bytes(original)
    append_ledger_lines(path, original, ['{"trial_id": "cc"}'])
    updated = path.read_bytes()
    assert updated.startswith(original)
    assert updated.decode().splitlines() == [
        '{"trial_id": "aa"}',
        '{"trial_id": "bb"}',
        '{"trial_id": "cc"}',
    ]


def test_formation_return_ignores_the_decision_close() -> None:
    start = datetime(2024, 1, 2, 21, tzinfo=UTC)
    closes = [float(index + 1) for index in range(12)]
    bars = pl.DataFrame(
        {
            "security_id": ["AAA"] * len(closes),
            "event_time": [start + timedelta(days=index) for index in range(len(closes))],
            "close": closes,
        }
    )
    formed = formation_return(bars, lookback=3, skip=1)
    before = formed["formation"].to_list()
    bumped = bars.with_columns(
        pl.when(pl.col("close") == closes[-1])
        .then(pl.lit(10_000.0))
        .otherwise(pl.col("close"))
        .alias("close")
    )
    after = formation_return(bumped, lookback=3, skip=1)["formation"].to_list()
    assert before == after
    # Row t uses close[t-1-skip] and close[t-1-skip-lookback], not close[t].
    hand = closes[9] / closes[6] - 1.0
    assert before[-1] == pytest.approx(hand)


def test_long_short_is_flat_when_a_leg_is_smaller_than_five() -> None:
    start = datetime(2024, 1, 2, 21, tzinfo=UTC)
    names = [f"N{index}" for index in range(8)]
    frame = pl.DataFrame(
        {
            "event_time": [start] * len(names),
            "security_id": names,
            "formation": [float(index) for index in range(len(names))],
        }
    )
    weights = long_short_weights(
        frame,
        signal="formation",
        quantile=0.2,
        higher_is_long=True,
        max_name=0.05,
        gross_scale=1.0,
    )
    assert weights.is_empty()


def test_spy_regime_is_on_only_when_the_shifted_return_is_negative() -> None:
    start = datetime(2024, 1, 2, 21, tzinfo=UTC)
    rising = [100.0 + index for index in range(8)]
    falling = [100.0 - index for index in range(8)]
    up = pl.DataFrame(
        {
            "security_id": ["SPY"] * len(rising),
            "event_time": [start + timedelta(days=index) for index in range(len(rising))],
            "close": rising,
        }
    )
    down = up.with_columns(pl.Series("close", falling))
    assert spy_regime(up, lookback=3)["regime_on"].to_list()[-1] is False
    assert spy_regime(down, lookback=3)["regime_on"].to_list()[-1] is True


def test_dip_gate_drops_weights_when_the_regime_is_off() -> None:
    stamp = datetime(2024, 1, 2, 21, tzinfo=UTC)
    weights = pl.DataFrame(
        {
            "event_time": [stamp, stamp],
            "security_id": ["AAA", "BBB"],
            "target_weight": [0.02, -0.02],
        }
    )
    off = pl.DataFrame({"event_time": [stamp], "regime_on": [False]})
    on = pl.DataFrame({"event_time": [stamp], "regime_on": [True]})
    assert gate_dip_weights(weights, off).is_empty()
    kept = gate_dip_weights(weights, on)
    assert kept["security_id"].to_list() == ["AAA"]
    assert float(kept["target_weight"][0]) == pytest.approx(0.02)


def test_emit_flattens_a_name_that_leaves_the_eligible_set() -> None:
    start = datetime(2024, 1, 2, 21, tzinfo=UTC)
    calendar = [start, start + timedelta(days=1)]
    raw = {start: {"AAA": 0.1, "BBB": 0.1}}
    eligible = {start: {"AAA", "BBB"}, calendar[1]: {"BBB"}}
    emitted = emit_weight_rows(raw, calendar, every=2, eligible=eligible)
    second = emitted.filter(pl.col("event_time") == calendar[1])
    assert second["security_id"].to_list() == ["BBB"]
    assert float(second["target_weight"][0]) == pytest.approx(0.1)


def test_last_real_session_is_not_eligible_and_exit_bar_is_flagged() -> None:
    start = datetime(2024, 1, 2, 21, tzinfo=UTC)
    rows = []
    for name, n_days in (("AAA", 4), ("BBB", 2)):
        for index in range(n_days):
            rows.append(
                {
                    "security_id": name,
                    "event_time": start + timedelta(days=index),
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.0,
                    "close_total_return": 10.0,
                    "volume": 1000.0,
                    "adv": 10_000.0,
                    "vol_20": 0.02,
                    "source": "yahoo",
                }
            )
    real = pl.DataFrame(rows)
    membership = {
        "current": ["AAA", "BBB"],
        "changes": [{"date": "2020-01-01", "added": "AAA", "removed": ""}],
    }
    # BBB was never added and is not in current after the only change is undone
    # for dates after 2020. Both names are in current, and the 2020 change is
    # already in effect on 2024 dates, so both are members.
    marked = with_eligibility(real, membership)
    last_bbb = marked.filter((pl.col("security_id") == "BBB") & pl.col("is_last_real"))
    assert last_bbb.height == 1
    assert bool(last_bbb["eligible"][0]) is False
    earlier = marked.filter((pl.col("security_id") == "BBB") & ~pl.col("is_last_real"))
    assert earlier.height == 1
    assert bool(earlier["eligible"][0]) is True
    panel, n_exit = append_delisting_exits(real)
    assert n_exit == 1
    exit_rows = panel.filter(pl.col("source") == "last_close_delisting_exit")
    assert exit_rows["security_id"].to_list() == ["BBB"]
    assert exit_rows["event_time"].to_list() == [start + timedelta(days=2)]


def test_coverage_uses_members_with_a_real_bar() -> None:
    start = datetime(2016, 1, 4, 21, tzinfo=UTC)
    real = pl.DataFrame(
        {
            "security_id": ["AAA"],
            "event_time": [start],
            "source": ["yahoo"],
        }
    )
    membership = {"current": ["AAA", "BBB"], "changes": []}
    report = anchor_coverage(real, membership, ["2016-01-04"])
    assert report[0]["session"] == date(2016, 1, 4).isoformat()
    assert report[0]["members"] == 2
    assert report[0]["with_bar"] == 1
    assert report[0]["coverage"] == pytest.approx(0.5)
    assert report[0]["missing"] == ["BBB"]


def test_preregistration_json_round_trips() -> None:
    payload = json.loads(_SPEC.read_text(encoding="utf-8"))
    assert payload["status"] == "frozen_before_results"
    assert payload["prior_study"]["n_trials"] == 29
