"""Wave 10: paper multi-challenger L1 divergence identity properties."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_fund.paper.loop import _weight_l1_divergence


def test_identical_weights_l1_zero() -> None:
    w = {"A": 0.4, "B": -0.2, "C": 0.8}
    assert _weight_l1_divergence(w, dict(w)) == 0.0
    assert _weight_l1_divergence({}, {}) == 0.0


def test_opposite_signs_larger_l1_than_same_sign() -> None:
    champ = {"A": 0.5, "B": -0.5}
    same = {"A": 0.5, "B": -0.5}
    opposite = {"A": -0.5, "B": 0.5}
    assert _weight_l1_divergence(champ, same) == 0.0
    assert _weight_l1_divergence(champ, opposite) == pytest.approx(2.0)
    assert _weight_l1_divergence(champ, opposite) > _weight_l1_divergence(champ, same)


def test_missing_keys_treated_as_zero() -> None:
    assert _weight_l1_divergence({"A": 0.3}, {"B": 0.4}) == pytest.approx(0.7)
    assert _weight_l1_divergence({"A": 0.3}, {}) == pytest.approx(0.3)


@given(
    st.dictionaries(
        st.sampled_from(["A", "B", "C", "D"]),
        st.floats(-2.0, 2.0, allow_nan=False, allow_infinity=False),
        max_size=4,
    )
)
@settings(max_examples=40)
def test_l1_identity_and_nonnegativity(weights: dict[str, float]) -> None:
    assert _weight_l1_divergence(weights, dict(weights)) == 0.0
    flipped = {k: -v for k, v in weights.items()}
    l1_flip = _weight_l1_divergence(weights, flipped)
    assert l1_flip >= 0.0
    # Opposite signs: L1 = 2 * sum(|w|) when domains match.
    assert l1_flip == pytest.approx(2.0 * sum(abs(v) for v in weights.values()))
