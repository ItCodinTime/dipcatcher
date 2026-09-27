"""Coverage for the systematic estimator-identity prover (SYNTHETIC).

Honest bundles must pass every registered identity; planted-violation
bundles must make the corresponding identity FAIL — the prover detects
contract breaks rather than fabricating passes.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.northset import identities as northset_identities
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent
from quant_fund.research.identity_sweep import (
    IDENTITY_REGISTRY,
    IDENTITY_SWEEP_SCHEMA_VERSION,
    SyntheticBundle,
    catalog_identity_families,
    enumerate_catalog_identity_pairs,
    format_identity_table,
    make_synthetic_bundle,
    registered_families,
    run_identity_sweep,
    write_identity_receipt,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes


def _sweep(n_trials: int = 2, seed: int = 7, **kwargs: object) -> dict:
    return run_identity_sweep(n_trials=n_trials, seed=seed, **kwargs)  # type: ignore[arg-type]


def _verdicts(receipt: dict) -> dict[str, str]:
    return {row["name"]: row["verdict"] for row in receipt["identities"]}


def _corrupt_session_volume(seed: int) -> SyntheticBundle:
    """Planted violation: session volumes drift 2x from the daily total."""
    bundle = make_synthetic_bundle(seed)
    session = bundle.session.with_columns(pl.col("volume") * 2.0)
    return replace(bundle, session=session)


def _corrupt_daily_high(seed: int) -> SyntheticBundle:
    """Planted violation: daily high pinned below the close (OHLC break)."""
    bundle = make_synthetic_bundle(seed)
    bars = bundle.bars.with_columns(
        pl.min_horizontal(pl.col("low"), pl.col("open"), pl.col("close")).alias("high")
        - pl.lit(0.5)
    )
    return replace(bundle, bars=bars)


def _corrupt_book_lock(seed: int) -> SyntheticBundle:
    """Planted violation: locked book (best_bid == best_ask)."""
    bundle = make_synthetic_bundle(seed)
    book = bundle.book.with_columns(pl.col("best_ask").alias("best_bid"))
    return replace(bundle, book=book)


def _corrupt_session_chain(seed: int) -> SyntheticBundle:
    """Planted violation: break close_i == open_{i+1} on every second candle."""
    bundle = make_synthetic_bundle(seed)
    session = bundle.session.with_columns(
        pl.when(pl.col("session_index") == 1)
        .then(pl.col("open") + 1.5)
        .otherwise(pl.col("open"))
        .alias("open")
    )
    return replace(bundle, session=session)


def test_honest_generator_passes_all_identities() -> None:
    receipt = _sweep(n_trials=3, seed=11)
    assert receipt["all_passed"] is True
    assert receipt["n_failed"] == 0
    assert receipt["n_identities"] == len(IDENTITY_REGISTRY)
    assert receipt["n_passed"] == len(IDENTITY_REGISTRY)
    for row in receipt["identities"]:
        assert row["verdict"] == "pass", row["name"]
        assert math.isfinite(row["residual_max"])
        assert row["residual_max"] <= row["tolerance"]
        assert row["trials"] == 3


@pytest.mark.parametrize(
    ("factory", "expected_failure"),
    [
        (_corrupt_session_volume, "session_volume_conservation"),
        (_corrupt_daily_high, "ohlc_identity"),
        (_corrupt_book_lock, "book_uncrossed"),
        (_corrupt_session_chain, "session_chain"),
    ],
)
def test_planted_violation_fails_corresponding_identity(
    factory: object, expected_failure: str
) -> None:
    receipt = run_identity_sweep(
        n_trials=2,
        seed=5,
        bundle_factory=factory,  # type: ignore[arg-type]
    )
    verdicts = _verdicts(receipt)
    assert receipt["all_passed"] is False
    assert receipt["n_failed"] >= 1
    assert verdicts[expected_failure] == "fail"
    # Discrimination: unrelated identities still pass — not a blanket fail.
    assert sum(v == "pass" for v in verdicts.values()) >= 1


def test_determinism_under_seed() -> None:
    first = _sweep(n_trials=2, seed=13)
    second = _sweep(n_trials=2, seed=13)
    a = [(r["name"], r["residual_max"], r["verdict"]) for r in first["identities"]]
    b = [(r["name"], r["residual_max"], r["verdict"]) for r in second["identities"]]
    assert a == b


def test_receipt_schema_and_hash() -> None:
    receipt = _sweep(n_trials=1, seed=3)
    assert receipt["kind"] == "identity_sweep"
    assert receipt["schema_version"] == IDENTITY_SWEEP_SCHEMA_VERSION
    assert receipt["synthetic"] is True
    assert receipt["claim"] == "research_only"
    assert isinstance(receipt["git_revision"], str)
    for row in receipt["identities"]:
        assert {"name", "family", "statement", "tolerance", "residual_max", "verdict"} <= set(row)
        assert row["verdict"] in {"pass", "fail"}
    # receipt_sha256 covers the canonical payload minus itself.
    digest = receipt.pop("receipt_sha256")
    assert hash_bytes(canonical_json_bytes(receipt)) == digest
    # Honesty contract: no Sharpe/P&L/NAV keys anywhere in the payload.
    assert family_blob_forbidden_metrics_absent(receipt)


def test_write_identity_receipt_atomic_json(tmp_path: Path) -> None:
    receipt = _sweep(n_trials=1, seed=4)
    out = write_identity_receipt(tmp_path / "nested" / "receipt.json", receipt)
    payload = json.loads(out.read_text())
    assert payload["kind"] == "identity_sweep"
    assert payload["receipt_sha256"] == receipt["receipt_sha256"]
    assert len(payload["identities"]) == len(IDENTITY_REGISTRY)
    # No temp files left behind.
    assert list((tmp_path / "nested").glob("*.tmp")) == []


def test_registry_completeness_vs_catalog_guards() -> None:
    """Every guard-named rate family in the catalog must be swept."""
    guard_pairs = {
        p["family"] for p in enumerate_catalog_identity_pairs() if p["kind"] == "guard_impl"
    }
    missing = guard_pairs - registered_families()
    assert not missing, f"registry misses catalog-guarded families: {sorted(missing)}"
    # Every public *_rate prover in northset.identities is registered.
    rate_stems = {
        name[: -len("_rate")]
        for name, obj in vars(northset_identities).items()
        if name.endswith("_rate") and callable(obj)
    }
    missing_rates = rate_stems - registered_families()
    assert not missing_rates, f"registry misses northset rate fns: {sorted(missing_rates)}"


def test_enumerate_pairs_finds_known_families() -> None:
    pairs = enumerate_catalog_identity_pairs()
    assert pairs, "expected catalog introspection to find identity pairs"
    kinds = {p["kind"] for p in pairs}
    assert "never_equate" in kinds
    assert "guard_impl" in kinds
    families = catalog_identity_families()
    for known in (
        "ohlc_identity",
        "book_uncrossed",
        "session_reconstructs_daily",
        "session_volume_conservation",
        "session_chain",
        "gap_finite",
    ):
        assert known in families


def test_sweep_rejects_degenerate_args() -> None:
    with pytest.raises(ValueError):
        run_identity_sweep(n_trials=0)
    with pytest.raises(ValueError):
        run_identity_sweep(registry=[])
    with pytest.raises(ValueError):
        make_synthetic_bundle(1, n_days=5)


def test_format_identity_table() -> None:
    receipt = _sweep(n_trials=1, seed=2)
    table = format_identity_table(receipt)
    assert "identity" in table and "verdict" in table
    assert f"{receipt['n_passed']}/{receipt['n_identities']}" in table


def test_cli_verify_identities_exit_codes(tmp_path: Path) -> None:
    out = tmp_path / "identity_sweep.json"
    ok = CliRunner().invoke(
        app, ["verify-identities", "--out", str(out), "--trials", "1", "--seed", "9"]
    )
    assert ok.exit_code == 0, ok.output
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["all_passed"] is True
    assert "session_volume_conservation" in ok.output
