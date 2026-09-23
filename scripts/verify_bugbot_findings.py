"""Verify the 4 Bugbot findings are FIXED (post-fix regression harness).

Run: uv run python scripts/verify_bugbot_findings.py
Each check asserts the corrected contract and prints OK (fixed) or BROKEN.
Exit code 0 only when all four behave correctly.
"""

from __future__ import annotations

import numpy as np


def check_1_fz0_properness() -> bool:
    """F1 fix: canonical FZ0 must minimize at e=ES (proper), not (1-alpha)*ES."""
    from quant_fund.metrics.scoring import fissler_ziegel_loss, mean_fissler_ziegel

    alpha = 0.95
    # Two-point tail: L=1 w.p. 0.05, else 0. True VaR=0, true ES=1.
    rng = np.random.default_rng(0)
    losses = (rng.random(400_000) < 0.05).astype(float)
    var_true = 0.0
    es_true = 1.0
    grid = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    scores = [
        float(
            mean_fissler_ziegel(
                losses, np.full(losses.size, var_true), np.full(losses.size, e), alpha
            )
        )
        for e in grid
    ]
    argmin = grid[int(np.argmin(scores))]
    # Cross-check with the raw elementwise API too
    single = float(
        np.mean(
            fissler_ziegel_loss(
                losses, np.full(losses.size, var_true), np.full(losses.size, es_true), alpha
            )
        )
    )
    print(
        f"  e-grid scores: {dict(zip(map(str, grid), [round(s, 4) for s in scores], strict=True))}"
    )
    print(
        f"  argmin over e-grid = {argmin} | true ES=1.0 | (1-alpha)*ES=0.05 | score@true-ES={single:.4f}"
    )
    reproduced = abs(argmin - es_true) < 1e-12
    return bool(reproduced)


def check_2_pairwise_dm_truncation() -> bool:
    """F2 fix: mismatched lengths must raise ValueError, never truncate."""
    from quant_fund.metrics.inference import pairwise_diebold_mariano

    rng = np.random.default_rng(1)
    la = rng.normal(0.5, 1.0, 200)  # different means -> DM should be significant
    lb = rng.normal(0.0, 1.0, 150)
    raised = False
    try:
        pairwise_diebold_mariano({"a": la, "b": lb})
    except ValueError as exc:
        raised = True
        print(f"  mismatched lengths raise ValueError: {exc}")
    if not raised:
        print("  mismatched lengths did NOT raise")
    return raised


def check_3_kill_switch_accounting() -> bool:
    """F3 fix: every attempted order blocked by the halt is counted."""
    import polars as pl

    from quant_fund.backtest.engine import run_backtest
    from quant_fund.config import load_config

    cfg = load_config("configs/research.yaml")
    # Force kill switch to HALT so every order attempt is blocked
    cfg = cfg.model_copy(deep=True)
    object.__setattr__(
        cfg, "kill_switch", cfg.kill_switch.model_copy(update={"state": "HALT_NEW_ORDERS"})
    )

    dates = pl.datetime_range(pl.date(2024, 1, 1), pl.date(2024, 1, 10), "1d", eager=True).alias(
        "event_time"
    )
    bars = pl.DataFrame(
        {
            "event_time": dates,
            "security_id": ["AAA"] * 10,
            "open": [10.0] * 10,
            "high": [11.0] * 10,
            "low": [9.0] * 10,
            "close": [10.5] * 10,
            "close_total_return": [10.5] * 10,
            "volume": [1000] * 10,
            "source": ["synthetic"] * 10,
        }
    )
    weights = pl.DataFrame(
        {
            "event_time": dates,
            "security_id": ["AAA"] * 10,
            "target_weight": [0.5] * 10,
        }
    )
    result = run_backtest(bars, weights, cfg)
    m = result.metrics
    print(
        f"  kill_switch_halts={m['kill_switch_halts']} risk_gate_rejects={m['risk_gate_rejects']} "
        f"n_fills={result.fills.height} total_return={m['total_return']}"
    )
    # 10 decision dates, next-open fills → 9 execution turns × 1 name = 9 attempts
    counted = (
        int(m["kill_switch_halts"]) == 9
        and int(m["risk_gate_rejects"]) == 0
        and result.fills.height == 0
    )
    return counted


def check_4_ready_error_masking() -> bool:
    """F4 fix: internal bugs in doctor() must propagate, not become 503 invalid-config.

    Genuine config errors (bad YAML / schema) still fail closed as 503.
    """
    from fastapi.testclient import TestClient

    from quant_fund.api.app import app

    client = TestClient(app, raise_server_exceptions=False)
    # Sanity: an invalid config still fails closed as 503 configuration-invalid.
    bad = client.get("/ready", params={"config_path": "configs/nonexistent_check.yaml"})
    config_fail_closed = bad.status_code in (400, 404, 503)
    # Now poison doctor() with an internal bug: it must PROPAGATE (500), not mask.
    # NB: quant_fund.api re-exports `app`, shadowing the submodule attribute —
    # `import quant_fund.api.app as m` would bind the FastAPI instance, so use sys.modules.
    import sys

    app_mod = sys.modules["quant_fund.api.app"]

    class Boom:
        def __call__(self, *_a, **_k):
            raise AttributeError("simulated code bug inside doctor()")

    original = app_mod.doctor
    app_mod.doctor = Boom()
    try:
        resp2 = client.get("/ready", params={"config_path": "configs/research.yaml"})
        propagated = resp2.status_code == 500
    finally:
        app_mod.doctor = original
    print(
        f"  bad-config -> {bad.status_code} (fail-closed ok); "
        f"doctor() AttributeError -> {resp2.status_code} (propagated)"
    )
    return bool(config_fail_closed and propagated)


CHECKS = [
    ("F1 FZ0 proper (minimizes at true ES)", check_1_fz0_properness),
    ("F2 pairwise DM raises on length mismatch", check_2_pairwise_dm_truncation),
    ("F3 kill-switch counts every blocked order", check_3_kill_switch_accounting),
    ("F4 /ready propagates internal bugs (config still fail-closed)", check_4_ready_error_masking),
]


def main() -> int:
    results: list[tuple[str, bool, str]] = []
    for name, fn in CHECKS:
        try:
            reproduced = bool(fn())
            results.append((name, reproduced, ""))
        except Exception as exc:  # noqa: BLE001
            results.append((name, False, f"{type(exc).__name__}: {exc}"))
    print()
    for name, ok, err in results:
        status = "OK (fixed)" if ok else "BROKEN"
        suffix = f"  [ERROR {err}]" if err else ""
        print(f"{status:16s} {name}{suffix}")
    broken = [r for r in results if not r[1]]
    return 0 if not broken else 1


if __name__ == "__main__":
    raise SystemExit(main())
