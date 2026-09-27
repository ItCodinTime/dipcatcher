"""In-memory fake implementing the §4.3 PitVault.asof API for W2 tests.

W1 lands in parallel (DESIGN.md §10); W2 codes against the §4.3 signatures
and tests against this fake. The fake applies the vault's bitemporal filter
(latest known_at <= t per key) and records every read into the attached
recorder AFTER computing content_sha256 on the returned bytes — exactly the
recorder contract of §5.1.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from quant_fund.proof.recorder import InMemoryRecorder, make_read_record
from quant_fund.proofcore.contracts import sha256_hex_bytes

T0 = datetime(2024, 1, 1, tzinfo=UTC)


@dataclass
class FakePitFrame:
    """§4.3 PitFrame surface."""

    frame: pl.DataFrame
    dataset: str
    asof: datetime
    max_known_at: datetime
    rows: int
    content_sha256: str


class FakeVault:
    """In-memory §4.3 PitVault: asof(name, t) is the only read path."""

    def __init__(self, recorder: InMemoryRecorder, datasets: dict[str, pl.DataFrame]) -> None:
        self.recorder = recorder
        self._datasets = dict(datasets)

    def asof(self, name: str, t: datetime, **_kwargs: object) -> FakePitFrame:
        if name not in self._datasets:
            from quant_fund.proofcore.contracts import VaultError

            raise VaultError(f"unknown dataset: {name}")
        frame = self._datasets[name]
        if "known_at" in frame.columns:
            frame = frame.filter(pl.col("known_at") <= t)
            if frame.height == 0:
                from quant_fund.proofcore.contracts import VaultUnavailableError

                raise VaultUnavailableError(f"{name}: no version with known_at <= {t}")
        buffer = io.BytesIO()
        frame.write_parquet(buffer)
        payload = buffer.getvalue()
        max_known = frame["known_at"].max() if "known_at" in frame.columns and frame.height else t
        pit_frame = FakePitFrame(
            frame=frame,
            dataset=name,
            asof=t,
            max_known_at=max_known,
            rows=frame.height,
            content_sha256=sha256_hex_bytes(payload),
        )
        self.recorder.record(
            make_read_record(
                name,
                t,
                params={},
                rows=pit_frame.rows,
                content_sha256=pit_frame.content_sha256,
            )
        )
        return pit_frame


def synthetic_bars(sids: list[str], n_days: int, seed: int = 7) -> pl.DataFrame:
    """Deterministic daily bars panel in the engine's expected schema."""
    rng = np.random.default_rng(seed)
    rows = []
    for k, sid in enumerate(sids):
        px = 100.0 * (1 + 0.05 * k)
        for i in range(n_days):
            drift = 0.0004 * ((i + 3 * k) % 7 - 3)
            noise = float(rng.normal(0, 0.02))
            close = px * (1 + drift + noise)
            rows.append(
                {
                    "security_id": sid,
                    "event_time": T0 + timedelta(days=i),
                    "open": px * (1 + noise / 2),
                    "close": close,
                    "close_total_return": close,
                    "volume": 1_000_000.0 + 10_000 * i,
                    "adv": close * (1_000_000.0 + 10_000 * i),
                    "vol_20": 0.02 + 0.001 * k,
                    "source": "file",
                    "known_at": T0 + timedelta(days=i, hours=16),
                }
            )
            px = close
    return pl.DataFrame(rows).with_columns(
        pl.col("event_time").cast(pl.Datetime("us", "UTC")),
        pl.col("known_at").cast(pl.Datetime("us", "UTC")),
    )


def synthetic_weights(sids: list[str], n_days: int, seed: int = 11) -> pl.DataFrame:
    """Deterministic rebalance grid (event_time, security_id, target_weight)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_days):
        for sid in sids:
            rows.append(
                {
                    "event_time": T0 + timedelta(days=i),
                    "security_id": sid,
                    "target_weight": float(rng.uniform(-0.2, 0.8)),
                    "known_at": T0 + timedelta(days=i, hours=16),
                }
            )
    return pl.DataFrame(rows).with_columns(
        pl.col("event_time").cast(pl.Datetime("us", "UTC")),
        pl.col("known_at").cast(pl.Datetime("us", "UTC")),
    )


def make_fake_vault(
    recorder: InMemoryRecorder, sids: list[str] | None = None, n_days: int = 30
) -> FakeVault:
    """Fake vault preloaded with the runner's two datasets (runner constants)."""
    from quant_fund.proof.runner import BARS_DATASET, WEIGHTS_DATASET

    sids = sids or ["AAA", "BBB", "CCC"]
    return FakeVault(
        recorder,
        {
            BARS_DATASET: synthetic_bars(sids, n_days),
            WEIGHTS_DATASET: synthetic_weights(sids, n_days),
        },
    )


def mint_proven_run(tmp_path, *, seed: int = 42, signer=None, n_days: int = 30):
    """One full proven run against the fake vault; returns (bundle, bundle_dir).

    Engine-coupled: importing the backtest engine pulls the repo SCC (mlflow
    et al.). Callers should ``pytest.importorskip("mlflow")`` first.
    """
    from quant_fund.config.models import AppConfig
    from quant_fund.proof.runner import run_backtest_proven

    recorder = InMemoryRecorder()
    vault = make_fake_vault(recorder, n_days=n_days)
    bundle_dir = tmp_path / "proofs"
    bundle = run_backtest_proven(
        AppConfig(),
        seed=seed,
        pit_root=tmp_path / "pit",
        bundle_dir=bundle_dir,
        signer=signer,
        vault=vault,
        recorder=recorder,
    )
    return bundle, bundle_dir


def synthetic_trade_log(n_fills: int = 20) -> pl.DataFrame:
    """Trade log in the runner's schema (fills + nav marks), deterministic."""
    rng = np.random.default_rng(3)
    rows = []
    nav = 1_000_000.0
    for i in range(1, n_fills + 1):
        nav *= 1 + float(rng.normal(0.0005, 0.01))
        rows.append(
            {
                "fill_time": T0 + timedelta(days=i),
                "signal_time": T0 + timedelta(days=i - 1),
                "security_id": "AAA" if i % 2 else "BBB",
                "quantity": 10.0 * ((-1) ** i),
                "price": 100.0 + i,
                "fee": 0.10,
                "spread_cost": 0.05,
                "impact_cost": 0.02,
                "decision_price": 99.0 + i,
                "nav": nav,
            }
        )
    return pl.DataFrame(rows).with_columns(
        pl.col("fill_time").cast(pl.Datetime("us", "UTC")),
        pl.col("signal_time").cast(pl.Datetime("us", "UTC")),
    )


def synthetic_signal_log(n_rows: int = 20) -> pl.DataFrame:
    """Signal log in the §5.3 schema (decision_time, security_id, score, weight)."""
    rng = np.random.default_rng(5)
    return pl.DataFrame(
        {
            "decision_time": [T0 + timedelta(days=i) for i in range(n_rows)],
            "security_id": ["AAA" if i % 2 else "BBB" for i in range(n_rows)],
            "score": [float(rng.normal()) for _ in range(n_rows)],
            "weight": [float(rng.uniform(-0.2, 0.8)) for _ in range(n_rows)],
        }
    ).with_columns(pl.col("decision_time").cast(pl.Datetime("us", "UTC")))


def mint_synthetic_bundle(tmp_path, *, seed: int = 42, signer=None):
    """Engine-free bundle mint via build_bundle (no backtest-engine import).

    Records two fake vault reads so the data manifest is non-trivial.
    Returns (bundle, bundle_dir).
    """
    from quant_fund.proof.bundle import build_bundle

    recorder = InMemoryRecorder()
    recorder.record(
        make_read_record("silver/bars", T0, rows=90, content_sha256=sha256_hex_bytes(b"bars"))
    )
    recorder.record(
        make_read_record("gold/weights", T0, rows=90, content_sha256=sha256_hex_bytes(b"weights"))
    )
    bundle_dir = tmp_path / "proofs"
    bundle = build_bundle(
        run_kind="backtest",
        data_manifest=recorder.manifest_summary(),
        config_dump={"runtime": {"mode": "research"}, "seed_tag": "synthetic"},
        seed=seed,
        signal_log=synthetic_signal_log(),
        trade_log=synthetic_trade_log(),
        engine_metrics={"total_return": 0.01, "label": "SYNTHETIC"},
        bundle_dir=bundle_dir,
        signer=signer,
    )
    return bundle, bundle_dir
