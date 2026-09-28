"""Research dividend adjustment. SYNTHETIC bars, not market evidence."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quant_fund.backtest.engine import run_backtest
from quant_fund.config.loader import load_config
from quant_fund.data.adapters.stooq import session_close
from quant_fund.data.adapters.yahoo_eod import fetch_yahoo_chart
from quant_fund.data.corporate_actions import adjust_prices
from quant_fund.research.reality_sweep import prepare_bars
from quant_fund.research.total_return import (
    YAHOO_CHART_EVENTS,
    apply_research_total_return,
    parse_yahoo_corporate_actions,
)
from quant_fund.schemas.errors import PointInTimeError

_T0 = datetime(2024, 6, 3, 20, tzinfo=UTC)
_T1 = datetime(2024, 6, 4, 20, tzinfo=UTC)
_T2 = datetime(2024, 6, 5, 20, tzinfo=UTC)


def _bars(closes: list[float], *, opens: list[float] | None = None) -> pl.DataFrame:
    n = len(closes)
    times = [_T0 + timedelta(days=i) for i in range(n)]
    px_open = closes if opens is None else opens
    return pl.DataFrame(
        {
            "security_id": ["A"] * n,
            "event_time": times,
            "open": px_open,
            "high": [max(o, c) for o, c in zip(px_open, closes, strict=True)],
            "low": [min(o, c) for o, c in zip(px_open, closes, strict=True)],
            "close": closes,
            "volume": [1_000_000.0] * n,
            "source": ["synthetic"] * n,
        }
    )


def _dividend(when: datetime, amount: float, *, kind: str = "cash_dividend") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "security_id": ["A"],
            "event_time": [when],
            "action_type": [kind],
            "amount": [amount],
        }
    )


def _research_config(tmp_path):
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.costs.frictionless = True
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_net = 1.0
    cfg.risk_gate.max_gross = 1.0
    cfg.risk_gate.max_order_notional = 1e12
    return cfg


def _flat_book(panel: pl.DataFrame, tmp_path) -> float:
    weights = pl.DataFrame(
        {
            "event_time": [panel["event_time"][0]],
            "security_id": ["A"],
            "target_weight": [1.0],
        }
    )
    result = run_backtest(panel, weights, _research_config(tmp_path), initial_nav=100_000.0)
    return float(result.metrics["total_return"])


def test_parse_yahoo_events_aligns_to_session_close() -> None:
    payload = {
        "chart": {
            "result": [
                {
                    "timestamp": [1_704_153_600, 1_704_240_000],
                    "indicators": {
                        "quote": [
                            {
                                "open": [100.0, 100.0],
                                "high": [101.0, 101.0],
                                "low": [99.0, 99.0],
                                "close": [100.0, 100.0],
                                "volume": [1_000_000, 1_000_000],
                            }
                        ]
                    },
                    "events": {
                        "dividends": {
                            "1704240000": {"amount": 0.25, "date": 1_704_240_000},
                            "other": {"amount": 0.75, "date": 1_704_240_000},
                        },
                        "splits": {
                            "1704240000": {
                                "date": 1_704_240_000,
                                "numerator": 4,
                                "denominator": 1,
                                "splitRatio": "4:1",
                            }
                        },
                    },
                }
            ]
        }
    }
    actions = parse_yahoo_corporate_actions(payload, security_id="AAPL", yahoo_symbol="AAPL")
    assert actions.height == 2
    dividend = actions.filter(pl.col("action_type") == "cash_dividend")
    assert dividend["amount"].to_list() == [1.0]
    assert dividend["source"].to_list() == ["yahoo"]
    assert dividend["revision_id"].to_list() == ["YAHOO_VENDOR_ADJ"]
    assert dividend["available_time"].to_list() == dividend["event_time"].to_list()
    split = actions.filter(pl.col("action_type") == "split")
    assert split["factor"].to_list() == [4.0]
    from quant_fund.data.adapters.yahoo_eod import parse_yahoo_chart

    bars = parse_yahoo_chart(payload, security_id="AAPL", yahoo_symbol="AAPL")
    assert dividend["event_time"].to_list() == [bars["event_time"].to_list()[1]]


def test_parse_yahoo_uk_session_and_list_payload() -> None:
    stamp = 1_704_240_000
    payload = {
        "chart": {
            "result": [
                {
                    "events": {
                        "dividends": [{"amount": "1.5", "date": stamp}],
                    }
                }
            ]
        }
    }
    actions = parse_yahoo_corporate_actions(payload, security_id="BP", yahoo_symbol="BP.L")
    day = datetime.fromtimestamp(stamp, tz=UTC).date()
    assert actions["event_time"].to_list() == [session_close(day, ".uk")]
    assert actions["amount"].to_list() == [1.5]


def test_parse_yahoo_rejects_bad_events() -> None:
    base = {"chart": {"result": [{"events": {"dividends": {"1": {"amount": "nope", "date": 10}}}}]}}
    with pytest.raises(PointInTimeError, match="dividend amount"):
        parse_yahoo_corporate_actions(base, security_id="A", yahoo_symbol="A")
    zero = {
        "chart": {
            "result": [
                {"events": {"splits": {"1": {"numerator": 0, "denominator": 1, "date": 10}}}}
            ]
        }
    }
    with pytest.raises(PointInTimeError, match="numerator"):
        parse_yahoo_corporate_actions(zero, security_id="A", yahoo_symbol="A")
    dup = {
        "chart": {
            "result": [
                {
                    "events": {
                        "splits": [
                            {"numerator": 2, "denominator": 1, "date": 1_704_240_000},
                            {"numerator": 2, "denominator": 1, "date": 1_704_240_000},
                        ]
                    }
                }
            ]
        }
    }
    with pytest.raises(PointInTimeError, match="duplicate yahoo split"):
        parse_yahoo_corporate_actions(dup, security_id="A", yahoo_symbol="A")
    with pytest.raises(PointInTimeError, match="non-blank"):
        parse_yahoo_corporate_actions({"chart": {}}, security_id=" ", yahoo_symbol="A")
    empty = parse_yahoo_corporate_actions({}, security_id="A", yahoo_symbol="A")
    assert empty.height == 0
    assert "action_type" in empty.columns


def test_yahoo_events_query_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def capture(*_args: object, **kwargs: object) -> tuple[int, bytes]:
        url = kwargs.get("url", _args[1] if len(_args) > 1 else "")
        seen.append(str(url))
        return 200, b'{"chart": {"result": []}}'

    monkeypatch.setattr("quant_fund.data.adapters.yahoo_eod.pooled_request", capture)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 6, 1, tzinfo=UTC)
    fetch_yahoo_chart("SPY", start=start, end=end, retries=0)
    assert "events=" not in seen[0]
    fetch_yahoo_chart("SPY", start=start, end=end, retries=0, events=YAHOO_CHART_EVENTS)
    assert seen[1].endswith("&events=div%2Csplit")
    with pytest.raises(ValueError, match="events"):
        fetch_yahoo_chart("SPY", start=start, end=end, retries=0, events="div split")


def test_flat_price_dividend_lifts_total_return_close() -> None:
    bars = _bars([100.0, 100.0, 100.0])
    out = apply_research_total_return(bars, _dividend(_T2, 5.0)).sort("event_time")
    assert out["close_quote"].to_list() == [100.0, 100.0, 100.0]
    assert out["close_total_return"].to_list() == pytest.approx([100.0, 100.0, 105.0])
    assert out["close"].to_list() == pytest.approx([100.0, 100.0, 105.0])
    assert out["open"].to_list() == pytest.approx([100.0, 100.0, 105.0])
    assert out["return_basis"].to_list() == ["total_return", "total_return", "total_return"]
    assert out["dividend"].to_list() == pytest.approx([0.0, 0.0, 5.0])


def test_special_and_cash_dividends_sum() -> None:
    actions = pl.concat([_dividend(_T1, 2.0), _dividend(_T1, 3.0, kind="special_dividend")])
    out = apply_research_total_return(_bars([100.0, 100.0]), actions).sort("event_time")
    assert out["close_total_return"].to_list() == pytest.approx([100.0, 105.0])


def test_split_is_not_reapplied_when_prices_are_already_adjusted() -> None:
    bars = _bars([100.0, 110.0])
    actions = pl.concat(
        [
            _dividend(_T1, 5.0),
            pl.DataFrame(
                {
                    "security_id": ["A"],
                    "event_time": [_T1],
                    "action_type": ["split"],
                    "factor": [2.0],
                }
            ),
        ],
        how="diagonal_relaxed",
    )
    out = apply_research_total_return(bars, actions).sort("event_time")
    assert out["close_quote"].to_list() == [100.0, 110.0]
    assert out["close_total_return"].to_list() == pytest.approx([100.0, 115.0])


def test_raw_prints_follow_adjust_prices() -> None:
    bars = _bars([200.0, 102.0], opens=[200.0, 100.0])
    actions = pl.DataFrame(
        {
            "security_id": ["A", "A"],
            "event_time": [_T1, _T1],
            "action_type": ["split", "cash_dividend"],
            "factor": [2.0, None],
            "amount": [None, 1.0],
        }
    )
    out = apply_research_total_return(bars, actions, prices_already_split_adjusted=False).sort(
        "event_time"
    )
    expected = adjust_prices(bars, actions).sort("event_time")
    assert out["close_total_return"].to_list() == pytest.approx(
        expected["close_total_return"].to_list()
    )
    assert out["close"].to_list() == pytest.approx(out["close_total_return"].to_list())
    assert out["close_quote"].to_list() == [200.0, 102.0]


def test_dividend_outside_sample_is_dropped_and_gap_fails() -> None:
    bars = _bars([100.0, 100.0, 100.0])
    before = apply_research_total_return(bars, _dividend(_T0 - timedelta(days=3), 5.0))
    assert before["close"].to_list() == [100.0, 100.0, 100.0]
    first = apply_research_total_return(bars, _dividend(_T0, 5.0))
    assert first["close"].to_list() == [100.0, 100.0, 100.0]
    after = apply_research_total_return(bars, _dividend(_T2 + timedelta(days=3), 5.0))
    assert after["close"].to_list() == [100.0, 100.0, 100.0]
    gap = _bars([100.0, 100.0, 100.0]).filter(pl.col("event_time") != _T1)
    with pytest.raises(PointInTimeError, match="does not match a research bar"):
        apply_research_total_return(gap, _dividend(_T1, 1.0))


def test_late_available_time_and_unknown_id_fail() -> None:
    bars = _bars([100.0, 100.0])
    late = _dividend(_T1, 1.0).with_columns(
        (pl.col("event_time") + timedelta(days=1)).alias("available_time")
    )
    with pytest.raises(PointInTimeError, match="available_time"):
        apply_research_total_return(bars, late)
    other = _dividend(_T1, 1.0).with_columns(pl.lit("ZZ").alias("security_id"))
    with pytest.raises(PointInTimeError, match="absent"):
        apply_research_total_return(bars, other)
    with pytest.raises(PointInTimeError, match="already total-return"):
        apply_research_total_return(
            apply_research_total_return(bars, _dividend(_T1, 1.0)),
            _dividend(_T1, 1.0),
        )


def test_prepare_bars_default_schema_is_unchanged() -> None:
    bars = _bars([100.0, 110.0])
    plain = prepare_bars(bars)
    assert plain.columns == [
        "security_id",
        "event_time",
        "open",
        "high",
        "low",
        "close",
        "close_total_return",
        "volume",
        "adv",
        "vol_20",
        "source",
    ]
    assert plain["close_total_return"].to_list() == plain["close"].to_list()
    adjusted = prepare_bars(bars, _dividend(_T1, 5.0))
    assert "close_quote" in adjusted.columns
    assert adjusted["return_basis"][0] == "total_return"
    assert adjusted.sort("event_time")["close_total_return"].to_list() == pytest.approx(
        [100.0, 115.0]
    )


def test_research_backtest_credits_dividend_only_to_names_already_held(tmp_path) -> None:
    """SYNTHETIC three-session book. Quote path omits the cash dividend."""
    quote = _bars([100.0, 100.0, 100.0])
    held = prepare_bars(quote, _dividend(_T2, 5.0))
    quote_mark = prepare_bars(quote)
    held_return = _flat_book(held, tmp_path)
    quote_return = _flat_book(quote_mark, tmp_path)
    assert held_return == pytest.approx(0.05)
    assert quote_return == pytest.approx(0.0)

    entered_on_ex = prepare_bars(quote, _dividend(_T1, 5.0))
    assert _flat_book(entered_on_ex, tmp_path) == pytest.approx(0.0)

    dropped = _bars([100.0, 100.0, 95.0])
    adjusted_drop = prepare_bars(dropped, _dividend(_T2, 5.0))
    quote_drop = prepare_bars(dropped)
    assert _flat_book(adjusted_drop, tmp_path) == pytest.approx(0.0)
    assert _flat_book(quote_drop, tmp_path) == pytest.approx(-0.05)
