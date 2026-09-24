"""Hypothesis invariants for utils/{hashing,reproducibility,seeds}."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import polars as pl
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from quant_fund.utils.hashing import (
    canonical_frame_fingerprint,
    canonical_json_bytes,
    fingerprint,
    hash_bytes,
    hash_file,
)
from quant_fund.utils.reproducibility import git_revision, git_worktree_sha256
from quant_fund.utils.seeds import set_global_seed

# ---------------- hashing ----------------


@given(data=st.binary())
@settings(max_examples=60, deadline=None)
def test_hash_bytes_deterministic_and_hex(data: bytes) -> None:
    a, b = hash_bytes(data), hash_bytes(data)
    assert a == b
    assert len(a) == 64 and all(c in "0123456789abcdef" for c in a)


@given(a=st.binary(), b=st.binary())
@settings(max_examples=60, deadline=None)
def test_hash_bytes_domain_separation(a: bytes, b: bytes) -> None:
    if a != b:
        assert hash_bytes(a) != hash_bytes(b)


@given(data=st.binary())
@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_hash_file_matches_content_hash(tmp_path: Path, data: bytes) -> None:
    p = tmp_path / "blob.bin"
    p.write_bytes(data)
    assert hash_file(p) == hash_bytes(data)


_json_scalars = st.recursive(
    st.one_of(
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.text(),
        st.booleans(),
        st.none(),
        st.binary(),
    ),
    lambda children: st.one_of(
        st.lists(children, max_size=5),
        st.dictionaries(st.text(), children, max_size=5),
    ),
    max_leaves=8,
)


@given(v=_json_scalars)
@settings(max_examples=80, deadline=None)
def test_canonical_json_deterministic(v) -> None:
    assert canonical_json_bytes(v) == canonical_json_bytes(v)


@given(
    pairs=st.lists(st.tuples(st.text(), st.integers()), min_size=1, max_size=8),
    perm=st.data(),
)
@settings(max_examples=80, deadline=None)
def test_canonical_json_key_order_invariance(pairs, perm) -> None:
    d1 = dict(pairs)
    keys = list(d1.keys())
    perm.draw(st.permutations(keys))
    order = perm.draw(st.permutations(keys))
    d2 = {k: d1[k] for k in order}
    assert canonical_json_bytes(d1) == canonical_json_bytes(d2)


@given(v=_json_scalars)
@settings(max_examples=60, deadline=None)
def test_canonical_json_is_valid_json(v) -> None:
    json.loads(canonical_json_bytes(v).decode("utf-8"))


def test_canonical_json_nonfinite_becomes_null() -> None:
    assert canonical_json_bytes(float("nan")) == canonical_json_bytes(None)
    assert canonical_json_bytes(float("inf")) == canonical_json_bytes(None)
    assert canonical_json_bytes({"x": float("-inf")}) == canonical_json_bytes({"x": None})


@given(
    sids=st.lists(st.text(min_size=1, max_size=6), min_size=1, max_size=6, unique=True),
    n=st.integers(min_value=1, max_value=8),
    perm=st.data(),
)
@settings(max_examples=60, deadline=None)
def test_frame_fingerprint_row_col_order_invariant(sids, n, perm) -> None:
    data = {s: [float(i) for i in range(n)] for s in sids}
    df = pl.DataFrame(data)
    h1 = canonical_frame_fingerprint(df)
    # permute columns
    df2 = df.select(perm.draw(st.permutations(sids)))
    # permute rows
    row_idx = perm.draw(st.permutations(list(range(n))))
    df3 = df[row_idx]
    assert canonical_frame_fingerprint(df2) == h1
    assert canonical_frame_fingerprint(df3) == h1


def test_frame_fingerprint_counts_duplicate_rows() -> None:
    df = pl.DataFrame({"a": [1.0, 1.0], "b": [2.0, 2.0]})
    df1 = pl.DataFrame({"a": [1.0], "b": [2.0]})
    assert canonical_frame_fingerprint(df) != canonical_frame_fingerprint(df1)


@given(
    rc=st.integers(min_value=0),
    t0=st.text(max_size=20),
    t1=st.text(max_size=20),
    cols=st.lists(st.text(min_size=1, max_size=8), max_size=8),
)
@settings(max_examples=60, deadline=None)
def test_fingerprint_deterministic(rc, t0, t1, cols) -> None:
    kw = dict(
        row_count=rc,
        min_timestamp=t0,
        max_timestamp=t1,
        columns=cols,
        feature_version="f",
        universe_version="u",
        label_version="l",
    )
    assert fingerprint(**kw) == fingerprint(**kw)


def test_fingerprint_domain_separation() -> None:
    base = dict(
        row_count=10,
        min_timestamp="a",
        max_timestamp="b",
        columns=["x"],
        feature_version="f",
        universe_version="u",
        label_version="l",
    )
    h0 = fingerprint(**base)
    for key, val in [
        ("row_count", 11),
        ("min_timestamp", "z"),
        ("max_timestamp", "z"),
        ("columns", ["x", "y"]),
        ("feature_version", "g"),
        ("universe_version", "v"),
        ("label_version", "m"),
    ]:
        kw = dict(base)
        kw[key] = val
        assert fingerprint(**kw) != h0, f"{key} change must alter fingerprint"


def test_fingerprint_column_order_is_significant() -> None:
    kw = dict(
        row_count=1,
        min_timestamp="a",
        max_timestamp="b",
        feature_version="f",
        universe_version="u",
        label_version="l",
    )
    assert fingerprint(columns=["a", "b"], **kw) != fingerprint(columns=["b", "a"], **kw)


# ---------------- reproducibility ----------------


def test_worktree_sha_is_hex_or_unknown() -> None:
    h = git_worktree_sha256()
    assert h == "UNKNOWN" or (len(h) == 64 and all(c in "0123456789abcdef" for c in h))


def test_git_revision_is_hex_or_unknown() -> None:
    r = git_revision()
    assert r == "UNKNOWN" or (len(r) >= 7 and all(c in "0123456789abcdef" for c in r))


# ---------------- seeds ----------------


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=40, deadline=None)
def test_global_seed_reproducible_python_and_numpy(seed: int) -> None:
    set_global_seed(seed)
    a = random.random()
    na = np.random.random(4)
    set_global_seed(seed)
    assert random.random() == a
    np.testing.assert_array_equal(np.random.random(4), na)


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=40, deadline=None)
def test_global_seed_sets_hashseed_env(seed: int) -> None:
    set_global_seed(seed)
    assert os.environ["PYTHONHASHSEED"] == str(seed)


def test_different_seeds_give_different_streams() -> None:
    set_global_seed(1)
    a = np.random.random(8)
    set_global_seed(2)
    b = np.random.random(8)
    assert not np.allclose(a, b)
