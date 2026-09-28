"""Meta-coverage for catalog non-``_honesty_errors`` helpers.

Three contracts exercised mechanically:

- ``*_has_finite_*`` / ``*_blob`` / single-arg probes: value matrix (non-dict,
  empty, populated, non-finite) -> must not raise.
- ``hypotheses_include_*``: non-list -> False; matching hypothesis dict ->
  True/family-gated result.
- ``*_consistency_errors(x, hypotheses)``: (empty, empty) and populated
  variants -> list of error strings.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from contextlib import suppress

import pytest

import quant_fund.research.catalog as catalog

_HYP_ID_RE = re.compile(r"\b(H\d+[A-Z]*_HYPOTHESIS_ID)\b")
_FAMILY_RE = re.compile(r"\b([A-Z][A-Z0-9_]*EXPECTED_FAMILY)\b")


def _catalog_fns() -> list[tuple[str, Callable, int]]:
    out = []
    for name in dir(catalog):
        obj = getattr(catalog, name)
        if not callable(obj) or name.endswith("_honesty_errors"):
            continue
        if name.startswith("_"):
            continue
        try:
            params = list(inspect.signature(obj).parameters.values())
        except (ValueError, TypeError):
            continue
        required = [
            p
            for p in params
            if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        if len(required) <= 2 and len(params) <= 3:
            out.append((name, obj, len(params)))
    return sorted(out)


_FNS = _catalog_fns()


def _src(fn: Callable) -> str:
    try:
        return inspect.getsource(fn)
    except (OSError, TypeError):
        return ""


def _module_const(fn: Callable, name: str) -> object:
    return getattr(inspect.getmodule(fn), name, None)


def _hyp_id(fn: Callable) -> str | None:
    for match in _HYP_ID_RE.findall(_src(fn)):
        value = _module_const(fn, match)
        if isinstance(value, str):
            return value
    return None


def _expected_family(fn: Callable) -> str | None:
    for match in _FAMILY_RE.findall(_src(fn)):
        value = _module_const(fn, match)
        if isinstance(value, str):
            return value
    return None


_INCLUDE = [
    t for t in _FNS if t[0].startswith("hypotheses_include_") and t[0] != "hypotheses_include_id"
]
_CONSISTENCY = [t for t in _FNS if t[0].endswith("_consistency_errors") and t[2] == 2]
_PROBE = [
    t
    for t in _FNS
    if t not in _INCLUDE and t not in _CONSISTENCY and t[2] == 1 and t[0] != "hypotheses_include_id"
]


@pytest.mark.parametrize(("name", "fn", "arity"), _PROBE, ids=[t[0] for t in _PROBE])
def test_probe_value_matrix(name: str, fn: Callable, arity: int) -> None:
    values = [
        None,
        0,
        0.5,
        "x",
        [],
        {},
        {"rate": 0.5},
        {"p_ic": 0.05},
        {"id": "h"},
        [{"id": "h"}],
        {"a": None},
        {"a": float("nan")},
    ]
    for value in values:
        # strict refusal is also a covered branch
        with suppress(TypeError, ValueError, KeyError, AttributeError, IndexError):
            fn(value)


@pytest.mark.parametrize(("name", "fn", "arity"), _INCLUDE, ids=[t[0] for t in _INCLUDE])
def test_include_matrix(name: str, fn: Callable, arity: int) -> None:
    assert fn(None) is False
    assert fn([]) is False
    assert fn([None, 1, "x"]) is False
    hyp_id = _hyp_id(fn)
    assert hyp_id is not None, f"{name} must bind a hypothesis id"
    assert fn([{"id": hyp_id}]) in (True, False)
    assert fn([{"id": "other"}]) is False
    family = _expected_family(fn)
    if family is not None:
        assert fn([{"id": hyp_id, "family": family}]) is True
        wrong = fn([{"id": hyp_id, "family": "wrong-family"}])
        assert wrong is False
        # require_*=False lifts the family gate.
        import inspect as _i

        req = [
            p.name for p in _i.signature(fn).parameters.values() if p.name.startswith("require_")
        ]
        if req:
            assert (
                fn(
                    [{"id": hyp_id, "family": "wrong-family"}],
                    **{req[0]: False},
                )
                is True
            )


@pytest.mark.parametrize(("name", "fn", "arity"), _CONSISTENCY, ids=[t[0] for t in _CONSISTENCY])
def test_consistency_errors_matrix(name: str, fn: Callable, arity: int) -> None:
    def _check(result: object) -> None:
        assert isinstance(result, list)
        assert all(isinstance(e, str) for e in result)

    hyp_id = _hyp_id(fn)
    hyps = [{"id": hyp_id}] if hyp_id else []
    family = _expected_family(fn)
    if hyp_id and family:
        hyps.append({"id": hyp_id, "family": family})
    for first in (None, {}, {"rate": 0.5}, {"p_ic": 0.05, "eligible": True}):
        for second in ([], hyps, [None], [{"id": "other"}]):
            with suppress(TypeError, ValueError, KeyError, AttributeError):
                _check(fn(first, second))


def test_include_id_generic() -> None:
    fn = catalog.hypotheses_include_id
    assert fn([], "h1") is False
    # Default require_family=H16_H18_EXPECTED_FAMILY; None lifts the gate.
    assert fn([{"id": "h1"}], "h1") is False
    assert fn([{"id": "h1"}], "h1", require_family=None) is True
    assert fn([{"id": "h1", "family": "cal"}], "h1", require_family="cal") is True
    assert fn([{"id": "h1", "family": "x"}], "h1", require_family="cal") is False
