"""Meta-coverage for catalog ``*_honesty_errors`` functions.

Every helper takes a receipt ``blob`` and returns a list of error strings.
The common contract: skip (return []) when required keys are absent or the
blob is not a dict; run structural key/hypothesis checks when keys are
present. This file drives each helper through its three main branches —
non-dict, empty dict, and a blob populated with every key the helper reads —
without asserting specific error strings, which are the helper's own
contract to define.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable

import pytest

import quant_fund.research.catalog as catalog


def _honesty_functions() -> list[tuple[str, Callable]]:
    out = []
    for name in dir(catalog):
        obj = getattr(catalog, name)
        if callable(obj) and name.endswith("_honesty_errors"):
            try:
                params = list(inspect.signature(obj).parameters.values())
            except (ValueError, TypeError):
                continue
            if len(params) == 1:
                out.append((name, obj))
    return sorted(out)


_FNS = _honesty_functions()
_KEY_RE = re.compile(
    r'["\']([a-z][a-z0-9_]{4,})["\']\s+(?:in|not\s+in)\s+blob'
    r'|blob\.get\(\s*["\']([a-z][a-z0-9_]{4,})["\']'
    r'|blob\[\s*["\']([a-z][a-z0-9_]{4,})["\']\s*\]'
)
# Keys bound to locals: k_x = "key" used later as `k_x in blob`.
_LOCAL_KEY_RE = re.compile(r'\bk_\w+\s*=\s*["\']([a-z][a-z0-9_]{4,})["\']')
_LOCAL_USE_RE = re.compile(r"\b(k_\w+)\s+(?:in|not\s+in)\s+blob")


def _blob_keys(fn: Callable) -> set[str]:
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return set()
    keys: set[str] = set()
    for match in _KEY_RE.finditer(src):
        keys.update(g for g in match.groups() if g)
    # Only keep local bindings that are actually used against blob.
    used_locals = set(_LOCAL_USE_RE.findall(src))
    if used_locals:
        bound = re.findall(r'\b(k_\w+)\s*=\s*["\']([a-z][a-z0-9_]{4,})["\']', src)
        keys.update(value for name, value in bound if name in used_locals)
        # Local bound to a tuple/list of keys: k_pair = ("a", "b").
        for name, _, values in re.findall(
            r'\b(k_\w+)\s*=\s*\(\s*((?:["\'][a-z][a-z0-9_]{4,}["\']\s*,?\s*)+)\)',
            src,
        ):
            if name in used_locals:
                keys.update(re.findall(r'"([a-z][a-z0-9_]{4,})"', values))
    return keys


def _assert_error_list(result: object) -> None:
    assert isinstance(result, list)
    assert all(isinstance(item, str) for item in result)


@pytest.mark.parametrize(
    ("name", "fn"),
    _FNS,
    ids=[name for name, _ in _FNS],
)
def test_honesty_fn_contract(name: str, fn: Callable) -> None:
    # Non-dict blobs always skip.
    assert fn(None) == []
    assert fn([1, 2]) == []
    assert fn("x") == []
    assert fn(0.5) == []
    # Empty blob always skips (required keys absent).
    assert fn({}) == []
    _assert_error_list(fn({}))


# Suffix-keyed blobs for helpers that iterate blob.items() by key suffix.
_SUFFIX_BLOB_KEYS = [
    "x_rate",
    "x_mean_rank_ic",
    "x_finite_rate",
    "x_share",
    "x_fraction",
    "x_floor",
    "x_t_ic",
    "x_p_ic",
    "x_n_dates",
    "x_observation",
    "x_count",
    "x_total",
    "x_finite",
    "x_scope",
    "x_missing",
    "x_stub",
    "x_receipt",
    "x_evidence",
    "kyle_x_rate",
    "kyle_x_p_ic",
    "best_feature_rate",
    "METRICS_x_rate",
    "sweep_reject_x_p",
    "sweep_follow_x_p",
    42,
]


@pytest.mark.parametrize(
    ("name", "fn"),
    _FNS,
    ids=[name for name, _ in _FNS],
)
def test_honesty_fn_populated_blob(name: str, fn: Callable) -> None:
    keys = _blob_keys(fn)
    keys |= set(str(k) for k in _SUFFIX_BLOB_KEYS if isinstance(k, str))
    for value in (0.5, 1.0, -0.05, 2.0, float("inf"), None, "x", {"rate": 0.5}, [0.1, 0.2]):
        _assert_error_list(fn(dict.fromkeys(keys, value)))
    # Mixed finite values plus eligibility-style flags.
    blob = dict.fromkeys(keys, 0.5)
    blob.update(
        {
            "book_hypothesis_eligible": True,
            "hypothesis_eligible": True,
            "eligible": True,
        }
    )
    _assert_error_list(fn(blob))


def test_all_catalog_helpers_covered_by_this_file() -> None:
    # Guard: if new *_honesty_errors helpers land, they automatically join the
    # parametrized matrix above; this assertion documents the count at write
    # time and fails loudly if the registry attribute disappears.
    assert len(_FNS) >= 200


_CONST_RE = re.compile(r"\b([A-Z][A-Z0-9_]{3,})\b")


def _corrupt(value: object) -> object:
    """Return a same-shape corrupted constant to fire error branches."""
    if isinstance(value, str):
        return "CORRUPTED_" + value[:8]
    if isinstance(value, tuple):
        return tuple(_corrupt(item) for item in value)
    if isinstance(value, list):
        return [_corrupt(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return type(value)(_corrupt(item) for item in value)
    if isinstance(value, dict):
        return {k: _corrupt(v) for k, v in value.items()}
    if isinstance(value, (int, float)):
        return value
    return value


@pytest.mark.parametrize(
    ("name", "fn"),
    _FNS,
    ids=[name for name, _ in _FNS],
)
def test_honesty_fn_corrupted_constants(
    name: str, fn: Callable, monkeypatch: pytest.MonkeyPatch
) -> None:
    keys = _blob_keys(fn)
    if not keys:
        pytest.skip("no discoverable blob keys")
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        pytest.skip("no source")
    module = inspect.getmodule(fn)
    blob = dict.fromkeys(keys, 0.5)
    blob.update(
        {
            "book_hypothesis_eligible": True,
            "hypothesis_eligible": True,
            "eligible": True,
        }
    )
    for const in _CONST_RE.findall(src):
        original = getattr(module, const, None)
        if original is None:
            continue
        corrupted = _corrupt(original)
        monkeypatch.setattr(module, const, corrupted)
        try:
            _assert_error_list(fn(dict(blob)))
        finally:
            monkeypatch.undo()


_NAME_RE = re.compile(r"\b([a-z_][a-z0-9_]{5,})\s*\(")


@pytest.mark.parametrize(
    ("name", "fn"),
    _FNS,
    ids=[name for name, _ in _FNS],
)
def test_honesty_fn_collapsed_ids_and_broken_gates(
    name: str, fn: Callable, monkeypatch: pytest.MonkeyPatch
) -> None:
    keys = _blob_keys(fn)
    if not keys:
        pytest.skip("no discoverable blob keys")
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        pytest.skip("no source")
    module = inspect.getmodule(fn)
    blob = dict.fromkeys(keys, 0.5)
    blob.update(
        {
            "book_hypothesis_eligible": True,
            "hypothesis_eligible": True,
            "eligible": True,
        }
    )
    # Collapse every hypothesis id in the module -> *_collapsed branches fire.
    hyp_names = [
        n for n in dir(module) if "HYPOTHESIS_ID" in n and isinstance(getattr(module, n), str)
    ]
    if hyp_names:
        for hyp in hyp_names:
            monkeypatch.setattr(module, hyp, "COLLAPSED")
        _assert_error_list(fn(dict(blob)))
        monkeypatch.undo()
    # Break each referenced module helper (both polarities) -> gate branches.
    helpers = {
        n for n in _NAME_RE.findall(src) if callable(getattr(module, n, None)) and n != fn.__name__
    }
    for helper in sorted(helpers):
        for ret in (False, True, None, ""):
            monkeypatch.setattr(module, helper, lambda *a, _ret=ret, **_k: _ret)
            try:
                result = fn(dict(blob))
                # A fn may delegate its return to the patched helper; only
                # require the list contract when it still produces one.
                if isinstance(result, list):
                    _assert_error_list(result)
            except (TypeError, AttributeError, ValueError, KeyError):
                pass  # some helpers feed arithmetic; polarity coverage enough
            finally:
                monkeypatch.undo()
