"""Wave 40: research bench_* return blobs must omit forbidden headline metrics.

SYNTHETIC tiny panel only — research regression, not live P&L
(live_pnl_claim=false). Skip network.
"""

from __future__ import annotations

from typing import Any

import pytest

from quant_fund.config import load_config
from quant_fund.pipeline.dataset import build_gold, panel
from quant_fund.research.benches import (
    bench_alpha,
    bench_conformal,
    bench_cpcv_audit,
    bench_cv_plus,
    bench_distribution,
    bench_drawdown,
    bench_evalues,
    bench_liquidity,
    bench_ranking,
    bench_regime,
    bench_tail,
    bench_volatility,
)
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent


def _assert_bench_blob_clean(blob: object, name: str) -> None:
    """Fail closed: no sharpe/sortino/calmar/pnl/nav key tokens in return mapping."""
    assert family_blob_forbidden_metrics_absent(blob) is True, name
    if isinstance(blob, dict):
        assert blob.get("live_pnl_claim") is not True
    elif isinstance(blob, list):
        for i, item in enumerate(blob):
            assert family_blob_forbidden_metrics_absent(item) is True, f"{name}[{i}]"


@pytest.fixture(scope="module")
def tiny_synth_panel(tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, Any, str]:
    root = tmp_path_factory.mktemp("bench_forbidden")
    cfg = load_config("configs/research.yaml")
    cfg.data.root = root
    cfg.data.synthetic_n_assets = 8
    cfg.data.synthetic_n_days = 60
    build_gold(cfg)
    df = panel(cfg)
    label = cfg.train.ranking_target
    if label not in df.columns:
        cands = [c for c in df.columns if str(c).startswith("future_return")]
        label = cands[0] if cands else "future_return_1"
    return cfg, df, str(label)


def test_bench_sample_forbidden_metrics_absent(tiny_synth_panel: tuple[Any, Any, str]) -> None:
    """Representative bench_* returns are clean of forbidden research-headline keys."""
    cfg, df, label = tiny_synth_panel

    samples: list[tuple[str, object]] = [
        ("liquidity", bench_liquidity(df)),
        ("volatility", bench_volatility(df, cfg)),
        ("alpha", bench_alpha(df, cfg, label)),
        ("distribution", bench_distribution(df, cfg)),
        ("regime", bench_regime(df, cfg)),
        ("tail", bench_tail(df, cfg)),
        ("drawdown", bench_drawdown(df, cfg)),
        ("ranking", bench_ranking(df, cfg, label)),
        ("conformal", bench_conformal(df, cfg)),
        ("evalues", bench_evalues(df, cfg)),
        ("cv_plus", bench_cv_plus()),
        ("cpcv", bench_cpcv_audit()),
    ]

    # At least half of the frame-dependent benches should produce non-empty blobs
    # on SYNTHETIC gold (guards against silent skip-all regressions).
    nonempty = 0
    for name, blob in samples:
        _assert_bench_blob_clean(blob, name)
        if (isinstance(blob, dict) and blob) or (isinstance(blob, list) and blob):
            nonempty += 1
    assert nonempty >= 6, f"too many empty bench blobs on SYNTHETIC: {nonempty}"


def test_bench_ranking_rows_individually_clean(tiny_synth_panel: tuple[Any, Any, str]) -> None:
    cfg, df, label = tiny_synth_panel
    rows = bench_ranking(df, cfg, label)
    assert isinstance(rows, list)
    assert rows  # ranking should recover at least oracle_raw on SYNTHETIC
    for row in rows:
        _assert_bench_blob_clean(row, str(row.get("name", "row")))
        # Explicit token check on top-level keys (defense in depth)
        for key in row:
            parts = str(key).lower().replace("-", "_").split("_")
            assert not any(t in {"sharpe", "sortino", "calmar", "pnl", "nav"} for t in parts if t)


def test_poisoned_blob_caught_by_helper() -> None:
    """Sanity: the helper under test still fails closed on planted poisons."""
    assert family_blob_forbidden_metrics_absent({"mean_ic": 0.1}) is True
    assert family_blob_forbidden_metrics_absent({"sharpe": 1.0}) is False
    assert family_blob_forbidden_metrics_absent({"diag": {"calmar_ratio": 0.2}}) is False
