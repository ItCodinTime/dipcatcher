"""Structured property tests for the fuzz campaign targets (Hypothesis).

Every parser/reader under test has a declared failure channel; these
properties assert nothing escapes it. The bounded random-corpus campaign in
``fuzz/`` exercises the same surface byte-wise; here we drive structured
inputs.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from quant_fund.cli._app import _collect_param_value
from quant_fund.config.loader import load_config
from quant_fund.data.adapters.stooq import parse_stooq_csv
from quant_fund.data.adapters.yahoo_eod import parse_yahoo_chart
from quant_fund.data.sources.base import SourceError, parse_time
from quant_fund.data.sources.normalize import csv_rows, normalize_ohlcv
from quant_fund.metrics.analytics import validate_analytics_export
from quant_fund.paper.ledger import (
    validate_ledger_schema,
    validate_promotion_dry_run_receipt,
)

json_scalar = st.one_of(
    st.none(),
    st.booleans(),
    st.floats(allow_nan=True, allow_infinity=True),
    st.integers(),
    st.text(max_size=64),
)
json_value = st.recursive(
    json_scalar,
    lambda children: st.one_of(
        st.lists(children, max_size=8),
        st.dictionaries(st.text(max_size=32), children, max_size=8),
    ),
    max_leaves=24,
)


@given(st.text())
@settings(max_examples=400, suppress_health_check=list(HealthCheck), deadline=None)
def test_stooq_csv_never_raises(text: str) -> None:
    # Tolerant tape parser: any text in, a (possibly empty) frame out.
    out = parse_stooq_csv(text, security_id="S", stooq_symbol="spy.us")
    assert out is not None


@given(json_value)
@settings(max_examples=400, suppress_health_check=list(HealthCheck), deadline=None)
def test_yahoo_chart_never_raises(payload: object) -> None:
    out = parse_yahoo_chart(payload, security_id="S", yahoo_symbol="SPY")
    assert out is not None


@given(st.text())
@settings(max_examples=300, suppress_health_check=list(HealthCheck), deadline=None)
def test_csv_rows_raises_only_source_error(text: str) -> None:
    try:
        csv_rows(text)
    except SourceError:
        return


@given(json_value)
@settings(max_examples=300, suppress_health_check=list(HealthCheck), deadline=None)
def test_parse_time_raises_only_source_error(value: object) -> None:
    assume(not isinstance(value, dict | list))
    try:
        parse_time(value)
    except SourceError:
        return


@given(st.lists(st.dictionaries(st.text(max_size=16), json_value, max_size=8), max_size=6))
@settings(max_examples=300, suppress_health_check=list(HealthCheck), deadline=None)
def test_normalize_ohlcv_raises_only_source_error(rows: list[dict]) -> None:
    try:
        normalize_ohlcv(rows, source="fuzz")
    except SourceError:
        return


@given(json_value)
@settings(max_examples=300, suppress_health_check=list(HealthCheck), deadline=None)
def test_promotion_receipt_validator_never_raises(value: object) -> None:
    errors = validate_promotion_dry_run_receipt(value)
    assert isinstance(errors, list)
    assert all(isinstance(e, str) for e in errors)


@given(json_value)
@settings(max_examples=300, suppress_health_check=list(HealthCheck), deadline=None)
def test_analytics_export_validator_never_raises(value: object) -> None:
    blob = value if isinstance(value, dict) else {"wrapped": value}
    report = validate_analytics_export(blob)
    assert isinstance(report["ok"], bool)
    assert isinstance(report["errors"], list)


@given(st.text(max_size=200))
@settings(max_examples=300, suppress_health_check=list(HealthCheck), deadline=None)
def test_collect_param_value_returns_scalar(text: str) -> None:
    out = _collect_param_value(text)
    assert isinstance(out, int | float | str)


@given(st.text())
@settings(max_examples=300, suppress_health_check=list(HealthCheck), deadline=None)
def test_load_config_fails_closed(tmp_path: Path, text: str) -> None:
    import pydantic
    import yaml

    target = tmp_path / "cfg.yaml"
    target.write_text(text, errors="replace")
    try:
        load_config(target)
    except (ValueError, yaml.YAMLError, pydantic.ValidationError, OSError):
        # Declared rejection channels: ValueError (contract checks), YAMLError
        # (malformed YAML), ValidationError (schema), OSError (missing files).
        return


def _write_ledger_artifact(run_dir: Path, name: str, blob: object) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / name).write_text(json.dumps(blob, default=str))


@given(json_value)
@settings(max_examples=200, suppress_health_check=list(HealthCheck), deadline=None)
def test_ledger_validator_never_raises_on_json(tmp_path: Path, value: object) -> None:
    run_dir = tmp_path / "run-1"
    for name in ("meta.json", "broker_state.json", "promotion_dry_run.json"):
        _write_ledger_artifact(run_dir, name, value)
    report = validate_ledger_schema(run_dir)
    assert isinstance(report["ok"], bool)


@given(st.datetimes(timezones=st.none() | st.just(UTC)))
@settings(max_examples=50)
def test_parse_time_accepts_iso_roundtrip(ts: datetime) -> None:
    out = parse_time(ts.isoformat())
    assert out.tzinfo is not None
    if ts.tzinfo is not None:
        assert out == ts.astimezone(UTC)
