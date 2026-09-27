"""Minimized crash regressions from the k3 fuzz campaign (fuzz/).

Each test pins one input that previously escaped a parser/reader's declared
failure channel. See docs/FUZZING.md for reproduction via the harnesses.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from quant_fund.cli._app import _collect_param_value
from quant_fund.config.loader import load_config
from quant_fund.data.adapters.stooq import parse_stooq_csv
from quant_fund.data.adapters.vendor_http import (
    VendorDataError,
    VendorHttpAdapter,
    VendorHttpConfig,
)
from quant_fund.data.adapters.yahoo_eod import parse_yahoo_chart
from quant_fund.data.sources.base import SourceError, parse_time, pit_frame
from quant_fund.data.sources.normalize import csv_rows, normalize_ohlcv
from quant_fund.paper.ledger import (
    latest_run_id,
    validate_ledger_schema,
    validate_promotion_dry_run_receipt,
)


class TestCsvNewlineCrash:
    """_csv.Error escaped ``for row in DictReader`` on a bare CR in an
    unquoted field (fuzz/stooq_csv, fuzz/normalize)."""

    def test_stooq_bare_cr_in_field_is_tolerated(self) -> None:
        # Previously: _csv.Error "new-line character seen in unquoted field".
        out = parse_stooq_csv("a,b\nc\rd,e\n", security_id="S", stooq_symbol="spy.us")
        assert isinstance(out, pl.DataFrame)

    def test_stooq_valid_crlf_rows_still_parse(self) -> None:
        out = parse_stooq_csv(
            "Date,Open,High,Low,Close,Volume\r\n"
            "2024-01-02,1.0,2.0,0.5,1.5,900\r\n",
            security_id="S",
            stooq_symbol="spy.us",
        )
        assert out.height == 1

    def test_csv_rows_bare_cr_is_tolerated(self) -> None:
        assert csv_rows("a,b\n1\r2,3\n") == [{"a": "1", "b": None}, {"a": "2", "b": "3"}]

    def test_csv_rows_oversized_field_fails_closed(self) -> None:
        # Beyond csv.field_size_limit — SourceError is the documented channel.
        with pytest.raises(SourceError, match="malformed"):
            csv_rows("a,b\n" + "x" * 200_000 + ",1\n")


class TestYahooChartCrash:
    """Non-conforming ``chart.result[0]`` members escaped the row-level try
    as KeyError/AttributeError/ValueError/OverflowError."""

    @pytest.mark.parametrize(
        "payload",
        [
            # quote a dict, not a list -> dict[0] KeyError
            {"chart": {"result": [{"timestamp": [1704153600],
                                   "indicators": {"quote": {"open": [1.0]}}}]}},
            # quote list of non-dicts -> AttributeError on .get
            {"chart": {"result": [{"timestamp": [1704153600],
                                   "indicators": {"quote": [7]}}]}},
            # non-numeric timestamp -> ValueError outside the try
            {"chart": {"result": [{"timestamp": ["x"],
                                   "indicators": {"quote": [{"open": [1], "high": [2],
                                                             "low": [1], "close": [1.5],
                                                             "volume": [5]}]}}]}},
            # absurd timestamp -> OverflowError from fromtimestamp
            {"chart": {"result": [{"timestamp": [1e20],
                                   "indicators": {"quote": [{"open": [1], "high": [2],
                                                             "low": [1], "close": [1.5],
                                                             "volume": [5]}]}}]}},
            # open series a dict -> opens[i] KeyError (uncaught)
            {"chart": {"result": [{"timestamp": [1704153600],
                                   "indicators": {"quote": [{"open": {"a": 1}, "high": [2],
                                                             "low": [1], "close": [1.5],
                                                             "volume": [5]}]}}]}},
            # indicators a list -> .get AttributeError
            {"chart": {"result": [{"timestamp": [1704153600], "indicators": [1, 2]}]}},
        ],
    )
    def test_malformed_payload_returns_frame_not_crash(self, payload: dict) -> None:
        out = parse_yahoo_chart(payload, security_id="S", yahoo_symbol="SPY")
        assert isinstance(out, pl.DataFrame)

    def test_valid_payload_still_parses(self) -> None:
        payload = {"chart": {"result": [{"timestamp": [1704153600],
                                         "indicators": {"quote": [{
                                             "open": [100.5], "high": [101.0],
                                             "low": [99.5], "close": [100.9],
                                             "volume": [1234567]}]}}]}}
        assert parse_yahoo_chart(payload, security_id="S", yahoo_symbol="SPY").height == 1


class TestConfigInheritCrash:
    """``inherit:`` with a non-string YAML value raised TypeError at
    ``path.parent / inherit`` (fuzz/config_yaml)."""

    @pytest.mark.parametrize("value", ["[a,b]", "5", "{k: v}", "1.5"])
    def test_non_string_inherit_raises_valueerror(self, tmp_path: Path, value: str) -> None:
        target = tmp_path / "c.yaml"
        target.write_text(f"inherit: {value}\n")
        with pytest.raises(ValueError, match="inherit"):
            load_config(target)


class TestParseTimeCrash:
    """``parse_time`` leaked OverflowError/OSError/ValueError instead of
    SourceError on out-of-range or wrong-typed input (fuzz/normalize)."""

    @pytest.mark.parametrize("value", [1e300, -1e300, "garbage", float("nan"), float("inf")])
    def test_unparseable_raises_source_error(self, value: object) -> None:
        with pytest.raises(SourceError):
            parse_time(value)

    def test_pit_frame_wraps_bad_timestamp(self) -> None:
        with pytest.raises(SourceError):
            pit_frame([{"event_time": 1e300, "available_time": None, "v": 1}], source="s")

    def test_normalize_ohlcv_wraps_bad_timestamp(self) -> None:
        row = {"security_id": "S", "event_time": 1e300, "open": 1, "high": 2,
               "low": 1, "close": 1.5, "volume": 5}
        with pytest.raises(SourceError):
            normalize_ohlcv([row], source="fuzz")


class TestLedgerValidatorCrash:
    """``validate_ledger_schema`` claims to report, never raise. Fuzz found
    UnicodeDecodeError on artifact reads, TypeError/AttributeError on
    non-dict promotion receipts, polars PanicException on corrupt parquet,
    and ValueError from ``latest_run_id`` on hostile run_id."""

    def _run_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "metadata" / "paper" / "run-1"
        d.mkdir(parents=True)
        return d

    @pytest.mark.parametrize("artifact", ["meta.json", "broker_state.json",
                                          "promotion_dry_run.json", "analytics_export.json"])
    def test_undecodable_artifact_is_reported(self, tmp_path: Path, artifact: str) -> None:
        d = self._run_dir(tmp_path)
        (d / artifact).write_bytes(b"\xff\xfe invalid utf8 \x00")
        report = validate_ledger_schema(d)
        assert report["ok"] is False
        assert any("invalid_json" in e for e in report["errors"])

    @pytest.mark.parametrize("payload", ['"run_id"', "5", "null", "true"])
    def test_non_dict_promotion_receipt_is_reported(
        self, tmp_path: Path, payload: str
    ) -> None:
        d = self._run_dir(tmp_path)
        (d / "promotion_dry_run.json").write_text(payload)
        report = validate_ledger_schema(d)
        assert report["ok"] is False  # not a dict -> schema errors, no crash

    def test_corrupt_equity_parquet_is_reported(self, tmp_path: Path) -> None:
        d = self._run_dir(tmp_path)
        (d / "equity.parquet").write_bytes(b"PAR1" + b"\xff" * 200 + b"PAR1")
        report = validate_ledger_schema(d)
        assert report["ok"] is False
        assert any("equity_unreadable" in e for e in report["errors"])

    def test_hostile_latest_run_id_returns_none(self, tmp_path: Path) -> None:
        paper = tmp_path / "metadata" / "paper"
        paper.mkdir(parents=True)
        (paper / "latest_run.json").write_text(json.dumps({"run_id": "../escape"}))
        assert latest_run_id(tmp_path) is None

    def test_promotion_receipt_validator_non_dict(self) -> None:
        for bad in (5, "x", None, [1], 1.5):
            assert validate_promotion_dry_run_receipt(bad) == [
                "promotion_receipt_not_object"
            ]


class TestCliParamValue:
    def test_collect_param_value_coerces_or_stringifies(self) -> None:
        assert _collect_param_value("42") == 42
        assert _collect_param_value("1e999") == float("inf")
        assert _collect_param_value(" x ") == "x"
        assert _collect_param_value("\x00\x01") == "\x00\x01"


class TestVendorHttpFailsClosed:
    """Vendor payload parser must only ever raise VendorDataError."""

    def _adapter(self, body: bytes) -> VendorHttpAdapter:
        return VendorHttpAdapter(
            VendorHttpConfig(
                vendor="fuzz",
                base_url="https://vendor.invalid",
                license_acknowledged=True,
                environ={"VENDOR_HTTP_API_KEY": "k"},
            ),
            transport=lambda _u, _h: body,
        )

    @pytest.mark.parametrize(
        "body",
        [
            b"",
            b"null",
            b"[1,2]",
            b'{"bars": {}}',
            b'{"bars": [{"ts": 5}]}',
            b'{"bars": [{"security_id": "S", "ts": "x", "release_ts": "x", "ingest_ts": "x"}]}',
        ],
    )
    def test_bars_payloads_raise_vendor_data_error(self, body: bytes) -> None:
        with pytest.raises(VendorDataError):
            self._adapter(body).get_bars()
