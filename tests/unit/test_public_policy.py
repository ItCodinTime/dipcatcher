"""Public-feature policy benches: planted is never a fit column."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl

from quant_fund.config import load_config
from quant_fund.models.ranking import PUBLIC_FEATURES
from quant_fund.research.benches import (
    bench_alpha,
    bench_drawdown,
    bench_quantile_bandit,
    bench_rl,
)

_ORACLE_KEYS = ("oracle_definition", "oracle_column")


def _public_frame(n_dates: int = 40, n_names: int = 10) -> pl.DataFrame:
    rng = np.random.default_rng(7)
    signal = np.zeros(n_names)
    rows: list[dict[str, object]] = []
    for t in range(n_dates):
        signal = 0.70 * signal + rng.normal(size=n_names)
        vol = 0.02 + 0.005 * rng.random(n_names)
        day = np.datetime64("2020-01-01") + np.timedelta64(t, "D")
        for i in range(n_names):
            noise = rng.normal()
            mom = float(signal[i] + 0.4 * noise)
            y = float(0.04 * signal[i] * vol[i] + 0.002 * rng.normal())
            rows.append(
                {
                    "event_time": day,
                    "security_id": f"N{i}",
                    "future_idio_return_1": y,
                    "cs_z_mom_20": mom,
                    "cs_z_reversal_1": float(-0.2 * signal[i] + rng.normal()),
                    "cs_z_vol_20": float(vol[i]),
                    "cs_z_amihud": float(rng.normal()),
                    "cs_z_ret_5": float(0.6 * signal[i] + rng.normal()),
                    "cs_pct_mom_20": mom,
                    "rel_volume": float(1.0 + 0.1 * rng.normal()),
                    "sector_relative_mom_20": float(0.8 * signal[i] + 0.3 * rng.normal()),
                    "cs_z_planted_signal": float(signal[i]),
                    "planted_signal": float(signal[i]),
                    "future_tail_event_5": float(y < -0.002),
                    "future_max_drawdown_5": float(min(y, 0.0)),
                }
            )
    return pl.DataFrame(rows)


def _assert_public_features(feats: list[object]) -> None:
    names = [str(f) for f in feats]
    assert names
    assert not any("planted" in n.lower() for n in names)
    assert all(n in PUBLIC_FEATURES for n in names)


def test_alpha_drawdown_rl_qt_features_exclude_planted() -> None:
    frame = _public_frame()
    cfg = load_config("configs/research.yaml")
    label = "future_idio_return_1"
    blobs = [
        bench_alpha(frame, cfg, label),
        bench_drawdown(frame, cfg),
        bench_rl(frame, label),
        bench_quantile_bandit(frame, label),
    ]
    for blob in blobs:
        assert blob
        _assert_public_features(list(blob.get("features", [])))


def test_h6_h14_statements_mention_ridge_or_static_public_policy() -> None:
    src = Path(__file__).resolve().parents[2] / "src" / "quant_fund" / "research" / "agent.py"
    text = src.read_text(encoding="utf-8")
    for hid in ("H6_linucb_vs_uniform", "H14_quantile_thompson"):
        idx = text.index(hid)
        block = text[idx : idx + 900].lower()
        assert "ridge" in block or "static public" in block, block


def test_bandits_share_n_dates_and_oracle_definition_keys() -> None:
    frame = _public_frame()
    label = "future_idio_return_1"
    rl = bench_rl(frame, label)
    qt = bench_quantile_bandit(frame, label)
    assert rl and qt
    assert int(rl["n_dates"]) == int(qt["n_dates"])
    for key in _ORACLE_KEYS:
        assert key in rl and key in qt
        assert rl[key] == qt[key]
    assert rl["oracle_definition"] in {"planted", "y_greedy"}
    assert rl["oracle_column"] == "cs_z_planted_signal"
    assert rl["leak_label"] == "SYNTHETIC"
    assert qt["leak_label"] == "SYNTHETIC"
    assert math.isfinite(float(rl["leak_ic_mom_vs_residual"]))
    assert math.isfinite(float(rl["leak_ic_mom_vs_planted"]))
    assert math.isfinite(float(rl["mean_advantage_vs_ridge"]))
    assert math.isfinite(float(qt["mean_advantage_vs_uniform"]))
    assert len(rl["reward_gap_series"]) == int(rl["n_dates"])
    assert len(qt["reward_gap_series"]) == int(qt["n_dates"])
