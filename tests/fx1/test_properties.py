"""Property-based tests: invariants that must hold for ALL inputs, not just
the examples an author thought of. An auditor's favorite genre."""

import contextlib

from hypothesis import given, settings
from hypothesis import strategies as st

from fx1.bench.dip import detect_dip_events
from fx1.eval.masking import mask_text
from fx1.honesty import Fx1HonestyError, validate_fx1_output
from fx1.reward import score_response

_text = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    max_size=200,
)


@given(text=_text)
@settings(max_examples=50)
def test_honesty_validator_never_crashes(text: str):
    with contextlib.suppress(Fx1HonestyError):  # raising is fine; crashing is not
        validate_fx1_output(text)


@given(text=_text)
@settings(max_examples=50)
def test_masking_is_idempotent(text: str):
    once = mask_text(text)
    assert mask_text(once) == once


@given(text=_text)
@settings(max_examples=50)
def test_reward_total_is_bounded(text: str):
    result = score_response(text)
    assert -10.0 <= result.total <= 10.0


_prices = st.lists(
    st.floats(min_value=1.0, max_value=10_000.0, allow_nan=False),
    min_size=2, max_size=100,
)


@given(closes=_prices)
@settings(max_examples=50)
def test_dip_events_are_causal_and_bounded(closes: list[float]):
    # Unique, monotonically increasing dates: date strings repeat under
    # `i % 28`, which makes dates.index() ambiguous for len(closes) > 28.
    dates = [
        f"2026-{i // 28 + 1:02d}-{i % 28 + 1:02d}" for i in range(len(closes))
    ]
    events = detect_dip_events(closes, dates, "T", threshold=0.10,
                               horizons_bars={"1m": 5})
    for event in events:
        assert 0.10 <= event.depth < 1.0
        # trough must not precede its peak
        assert dates.index(event.trough_date) >= dates.index(event.peak_date)
