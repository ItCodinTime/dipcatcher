"""Attribution edge paths: input validation and the optional-SHAP boundary."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from quant_fund.research.explainability.attribution import (
    permutation_attribution,
    shap_attribution,
)
from quant_fund.research.explainability.scoring import ProperScoreSpec

pytestmark = pytest.mark.synthetic

Array = NDArray[np.float64]


def _linear(x: Array) -> Array:
    return x[:, 0] * 2.0 + x[:, 1]


def _xy(n: int = 40, k: int = 3, seed: int = 0) -> tuple[Array, Array]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, k))
    return x, _linear(x) + rng.normal(scale=0.01, size=n)


def test_x_must_be_2d() -> None:
    x, y = _xy()
    with pytest.raises(ValueError, match="2-D"):
        permutation_attribution(_linear, x.ravel(), y)


def test_x_must_have_features() -> None:
    _, y = _xy()
    with pytest.raises(ValueError, match="at least one feature"):
        permutation_attribution(_linear, np.zeros((10, 0)), y[:10])


def test_xy_length_must_match() -> None:
    x, y = _xy()
    with pytest.raises(ValueError, match="mismatch"):
        permutation_attribution(_linear, x, y[:5])


def test_xy_must_be_finite() -> None:
    x, y = _xy()
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        permutation_attribution(_linear, x, y)


def test_max_rows_must_be_positive() -> None:
    x, y = _xy()
    with pytest.raises(ValueError, match="max_rows"):
        permutation_attribution(_linear, x, y, max_rows=0)


def test_max_rows_subsamples_and_warns() -> None:
    x, y = _xy(n=60)
    out = permutation_attribution(_linear, x, y, max_rows=10, n_repeats=2, seed=3)
    assert any("subsampled 10 of 60" in w for w in out.warnings)
    assert out.attributions


def test_non_finite_deltas_fail_closed() -> None:
    x, y = _xy()

    class _LaxScore(ProperScoreSpec):
        def __call__(self, _y: Array, pred: Array) -> float:
            return float(np.nanmean(pred))

    calls: list[int] = [0]

    def explodes(arr: Array) -> Array:
        calls[0] += 1
        if calls[0] > 1:  # baseline call is finite; permuted calls are not
            return np.full(arr.shape[0], np.nan)
        return _linear(arr)

    with pytest.raises(ValueError, match="deltas must be finite"):
        permutation_attribution(
            explodes,
            x,
            y,
            n_repeats=1,
            scoring=_LaxScore(name="lax", fn=lambda _a, _b: 0.0),
        )


def test_shap_unavailable_raises_import_error() -> None:
    pytest.importorskip("quant_fund.research.explainability.attribution")
    try:
        import shap  # noqa: F401
    except ImportError:
        x, y = _xy()
        with pytest.raises(ImportError, match="shap"):
            shap_attribution(_linear, x, y)
    else:
        pytest.skip("shap installed; ImportError path unreachable")


def test_shap_max_rows_must_be_positive_when_shap_installed() -> None:
    pytest.importorskip("shap")
    x, y = _xy()
    with pytest.raises(ValueError, match="max_rows"):
        shap_attribution(_linear, x, y, max_rows=0)
