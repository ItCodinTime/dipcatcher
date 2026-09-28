"""Catalog honesty helpers catch ImportError only on lazy microstructure imports.

Three sites previously used ``except Exception`` around
``from quant_fund.microstructure... import ...``. The callees are closed:
import machinery raises ``ImportError`` (including ``ModuleNotFoundError``)
when the module or name is missing. A non-import failure inside those
modules is a real bug and must propagate.
"""

from __future__ import annotations

import builtins
from collections.abc import Callable

import pytest

from quant_fund.research.catalog.candle import (
    candle_feature_cols_ic_completeness_honesty_errors,
    candle_feature_cols_ic_implies_mean_honesty_errors,
)
from quant_fund.research.catalog.receipt import (
    northset_metrics_required_keys_finite_when_present_honesty_errors,
)

_CANDLE_BLOB = {"family": "candle_order_book", "n_scored": 1, "ic_ofi": 0.1}
_RECEIPT_BLOB = {"spread": 0.01}


def _patch_import(
    monkeypatch: pytest.MonkeyPatch,
    *,
    module: str,
    exc: BaseException,
) -> None:
    real_import = builtins.__import__

    def gated(
        name: str,
        globals: dict[str, object] | None = None,  # noqa: A002 - import protocol
        locals: dict[str, object] | None = None,  # noqa: A002
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == module or name.endswith("." + module):
            raise exc
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", gated)


def _import_error_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
    *,
    module: str,
    call: Callable[[object], list[str]],
    blob: dict[str, object],
) -> None:
    _patch_import(monkeypatch, module=module, exc=ImportError("simulated missing"))
    assert call(blob) == []


def _non_import_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
    *,
    module: str,
    call: Callable[[object], list[str]],
    blob: dict[str, object],
) -> None:
    _patch_import(monkeypatch, module=module, exc=RuntimeError("not an ImportError"))
    with pytest.raises(RuntimeError, match="not an ImportError"):
        call(blob)


def test_candle_ic_implies_mean_import_error_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _import_error_returns_empty(
        monkeypatch,
        module="quant_fund.microstructure.bench",
        call=candle_feature_cols_ic_implies_mean_honesty_errors,
        blob=_CANDLE_BLOB,
    )


def test_candle_ic_implies_mean_non_import_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _non_import_error_propagates(
        monkeypatch,
        module="quant_fund.microstructure.bench",
        call=candle_feature_cols_ic_implies_mean_honesty_errors,
        blob=_CANDLE_BLOB,
    )


def test_candle_ic_completeness_import_error_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _import_error_returns_empty(
        monkeypatch,
        module="quant_fund.microstructure.bench",
        call=candle_feature_cols_ic_completeness_honesty_errors,
        blob=_CANDLE_BLOB,
    )


def test_candle_ic_completeness_non_import_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _non_import_error_propagates(
        monkeypatch,
        module="quant_fund.microstructure.bench",
        call=candle_feature_cols_ic_completeness_honesty_errors,
        blob=_CANDLE_BLOB,
    )


def test_northset_metrics_keys_import_error_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _import_error_returns_empty(
        monkeypatch,
        module="quant_fund.microstructure.book_metrics",
        call=northset_metrics_required_keys_finite_when_present_honesty_errors,
        blob=_RECEIPT_BLOB,
    )


def test_northset_metrics_keys_non_import_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _non_import_error_propagates(
        monkeypatch,
        module="quant_fund.microstructure.book_metrics",
        call=northset_metrics_required_keys_finite_when_present_honesty_errors,
        blob=_RECEIPT_BLOB,
    )
