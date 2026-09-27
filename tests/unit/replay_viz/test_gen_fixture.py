"""Unit tests for the replay-viz SYNTHETIC session fixture generator.

The generator lives outside ``src`` (replay/scripts/gen_fixture.py) since it
drives a TypeScript/canvas viewer; it is loaded here by file path. Everything
it emits is labeled SYNTHETIC — correctness fixtures, not market evidence.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
GEN_PATH = REPO_ROOT / "replay" / "scripts" / "gen_fixture.py"
FIXTURE_PATH = REPO_ROOT / "replay" / "public" / "fixtures" / "session.synthetic.json"


def _load_gen() -> Any:
    spec = importlib.util.spec_from_file_location("gen_fixture", GEN_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen() -> Any:
    return _load_gen()


def test_session_structure_and_labels(gen: Any) -> None:
    session = gen.gen_session(n_symbols=3, n_bars=60, seed=3)
    assert session["format"] == "dipcatcher.replay.session"
    assert session["source"] == "synthetic"  # SYNTHETIC label
    assert len(session["symbols"]) == 3
    sym = session["symbols"][0]
    # mirror quant_fund.schemas.market.Bar / security master fields
    for field in ("security_id", "symbol", "name", "exchange", "currency", "sector"):
        assert field in sym
    bars = session["bars"][sym["symbol"]]
    for field in ("event_time", "open", "high", "low", "close", "volume"):
        assert field in bars
    book = session["books"][sym["symbol"]]
    for field in ("event_time", "depth", "bid_price", "bid_size", "ask_price", "ask_size"):
        assert field in book
    assert session["markers"], "expected strategy decision markers"


def test_generated_session_validates_clean(gen: Any) -> None:
    session = gen.gen_session(n_symbols=4, n_bars=120, seed=11)
    assert gen.validate_session(session) == []


def test_books_are_uncrossed_and_sorted(gen: Any) -> None:
    session = gen.gen_session(n_symbols=2, n_bars=50, seed=5)
    for sym in (s["symbol"] for s in session["symbols"]):
        book = session["books"][sym]
        for bp, ap in zip(book["bid_price"], book["ask_price"], strict=True):
            assert bp == sorted(bp, reverse=True)
            assert ap == sorted(ap)
            assert bp[0] < ap[0], "book must not be crossed/locked"


def test_deterministic_for_seed(gen: Any) -> None:
    a = gen.gen_session(n_symbols=2, n_bars=30, seed=42)
    b = gen.gen_session(n_symbols=2, n_bars=30, seed=42)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    c = gen.gen_session(n_symbols=2, n_bars=30, seed=43)
    assert json.dumps(a, sort_keys=True) != json.dumps(c, sort_keys=True)


def test_committed_fixture_present_small_and_valid(gen: Any) -> None:
    assert FIXTURE_PATH.exists(), "run replay/scripts/gen_fixture.py"
    size = FIXTURE_PATH.stat().st_size
    assert size < 2_000_000, f"fixture {size} bytes exceeds 2MB cap"
    session = json.loads(FIXTURE_PATH.read_text())
    assert gen.validate_session(session) == []
    assert len(session["symbols"]) == 8
    assert session["bar_interval_seconds"] == 60
