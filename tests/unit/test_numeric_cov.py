"""Coverage for utils/numeric.py — require_finite guard and clip_positive floor."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.utils.numeric import clip_positive, require_finite


def test_require_finite_returns_input_object_when_all_finite() -> None:
    values = np.array([1.0, -2.5, 0.0])
    assert require_finite(values) is values


def test_require_finite_empty_and_single_element_pass() -> None:
    empty = np.array([], dtype=np.float64)
    assert require_finite(empty) is empty
    single = np.array([3.5])
    assert require_finite(single) is single


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_require_finite_raises_on_any_nonfinite(bad: float) -> None:
    with pytest.raises(ValueError, match="array contains NaN or inf"):
        require_finite(np.array([1.0, bad, 2.0]))


def test_require_finite_error_uses_supplied_name() -> None:
    with pytest.raises(ValueError, match="mid_prices contains NaN or inf"):
        require_finite(np.array([np.nan]), name="mid_prices")


def test_require_finite_duck_types_scalar_and_list() -> None:
    assert require_finite(1.5) == 1.5
    values = [1.0, 2.0]
    assert require_finite(values) is values
    with pytest.raises(ValueError, match="contains NaN or inf"):
        require_finite([1.0, np.nan])


@pytest.mark.parametrize("value", [0.0, -3.0, 1e-13])
def test_clip_positive_scalar_below_floor_collapses(value: float) -> None:
    result = clip_positive(value)
    assert isinstance(result, float)
    assert result == 1e-12


def test_clip_positive_scalar_at_or_above_floor_passes_through() -> None:
    assert clip_positive(1e-12) == 1e-12
    result = clip_positive(2.5)
    assert isinstance(result, float)
    assert result == 2.5


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_clip_positive_nonfinite_scalar_collapses_to_floor(value: float) -> None:
    assert clip_positive(value, floor=0.25) == 0.25


def test_clip_positive_int_and_numpy_scalar_take_scalar_path() -> None:
    assert clip_positive(3) == 3.0
    result = clip_positive(np.float64(-1.0), floor=0.5)
    assert isinstance(result, float)
    assert result == 0.5


def test_clip_positive_array_clips_and_replaces_nonfinite() -> None:
    values = np.array([np.nan, np.inf, -np.inf, -1.0, 0.0, 3.0])
    original = values.copy()
    np.testing.assert_array_equal(clip_positive(values, floor=0.5), [0.5, 0.5, 0.5, 0.5, 0.5, 3.0])
    np.testing.assert_array_equal(values, original)


def test_clip_positive_array_float64_and_shape_preserved() -> None:
    result = clip_positive(np.full((2, 3), -1.0))
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert result.shape == (2, 3)
    np.testing.assert_array_equal(result, np.full((2, 3), 1e-12))


def test_clip_positive_zero_d_and_empty_arrays_use_array_path() -> None:
    zero_d = clip_positive(np.asarray(-2.0), floor=0.5)
    assert isinstance(zero_d, np.ndarray)
    assert zero_d.shape == ()
    assert float(zero_d) == 0.5

    empty = clip_positive(np.array([]))
    assert isinstance(empty, np.ndarray)
    assert empty.dtype == np.float64
    assert empty.shape == (0,)


def test_clip_positive_list_like_input_uses_array_path() -> None:
    result = clip_positive([-1.0, 4.0], floor=0.5)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, [0.5, 4.0])
