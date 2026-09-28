"""Gradient-backend registration and adapter validation edges."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.robustness.gradients import (
    CallableBackend,
    FiniteDifferenceBackend,
    get_gradient_backend,
    register_gradient_backend,
    registered_backend_names,
    unregister_gradient_backend,
)


class _Nameless:
    name = ""

    def value_and_grad(self, path):  # pragma: no cover - never reached
        return 0.0, path


def test_register_requires_name() -> None:
    with pytest.raises(ValueError, match="non-empty name"):
        register_gradient_backend(_Nameless())


def test_unregister_missing_is_silent() -> None:
    unregister_gradient_backend("definitely_not_registered")
    assert "definitely_not_registered" not in registered_backend_names()


def test_get_unknown_lists_registered() -> None:
    with pytest.raises(KeyError, match="unknown gradient backend"):
        get_gradient_backend("nope_xyz")


class TestFiniteDifference:
    def test_validates_args(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            FiniteDifferenceBackend(3)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="finite and positive"):
            FiniteDifferenceBackend(lambda p: 0.0, step=0.0)
        with pytest.raises(ValueError, match="finite and positive"):
            FiniteDifferenceBackend(lambda p: 0.0, step=np.inf)

    def test_rejects_empty_or_nonfinite_path(self) -> None:
        backend = FiniteDifferenceBackend(lambda p: float(np.sum(p)))
        with pytest.raises(ValueError, match="non-empty finite"):
            backend.value_and_grad(np.array([]))
        with pytest.raises(ValueError, match="non-empty finite"):
            backend.value_and_grad(np.array([np.nan]))

    def test_central_difference_matches_linear_gradient(self) -> None:
        weights = np.array([1.0, -2.0, 3.0])

        def margin(path: np.ndarray) -> float:
            return float(weights @ path + 0.5)

        backend = FiniteDifferenceBackend(margin, step=1e-5)
        value, grad = backend.value_and_grad(np.array([0.1, 0.2, -0.3]))
        assert value == pytest.approx(float(weights @ np.array([0.1, 0.2, -0.3]) + 0.5))
        assert grad == pytest.approx(weights, rel=1e-4, abs=1e-4)


class TestCallableBackend:
    def test_validates_args(self) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            CallableBackend("", lambda p: (0.0, p))
        with pytest.raises(TypeError, match="callable"):
            CallableBackend("x", 4)  # type: ignore[arg-type]

    def test_gradient_alignment_enforced(self) -> None:
        backend = CallableBackend("bad", lambda p: (0.0, np.ones(p.size + 1)))
        with pytest.raises(ValueError, match="aligned"):
            backend.value_and_grad(np.ones(3))

    def test_nonfinite_gradient_rejected(self) -> None:
        backend = CallableBackend("bad", lambda p: (0.0, np.full(p.size, np.nan)))
        with pytest.raises(ValueError, match="finite"):
            backend.value_and_grad(np.ones(3))

    def test_nonfinite_margin_rejected(self) -> None:
        backend = CallableBackend("bad", lambda p: (np.inf, np.ones(p.size)))
        with pytest.raises(ValueError, match="finite"):
            backend.value_and_grad(np.ones(3))

    def test_passthrough(self) -> None:
        backend = CallableBackend("ok", lambda p: (float(np.sum(p)), np.ones(p.size)))
        margin, grad = backend.value_and_grad(np.array([1.0, 2.0]))
        assert margin == pytest.approx(3.0)
        assert grad.tolist() == [1.0, 1.0]
