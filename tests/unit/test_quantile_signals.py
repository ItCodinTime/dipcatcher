"""quantile_signals: causal forecasters + policy map + sim-live driver."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quant_fund.config.models import AppConfig
from quant_fund.paper.quantile_signals import (
    DEFAULT_TAUS,
    FORECASTERS,
    QuantilePolicy,
    compute_quantile_panel,
    evt_quantiles,
    ewma_emp_quantiles,
    quantile_moments,
    quantile_panels_to_weights,
    weights_from_quantiles,
)

TAUS = np.asarray(DEFAULT_TAUS)
T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _prices(n: int, seed: int = 7, drift: float = 0.0005) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rets = drift + 0.02 * rng.standard_normal(n)
    return 100.0 * np.exp(np.cumsum(np.concatenate([[0.0], rets])))


# --------------------------------------------------------------------------
# Forecaster contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("spec", sorted(FORECASTERS))
def test_forecaster_emits_monotone_finite_row(spec: str) -> None:
    closes = _prices(1200, seed=11)
    rets = np.diff(closes) / closes[:-1]
    fn, need = FORECASTERS[spec]
    q = fn(rets[-750:], TAUS)
    assert q.shape == (TAUS.size,)
    assert np.isfinite(q).all()
    assert np.all(np.diff(q) >= -1e-12)


@pytest.mark.parametrize("spec", sorted(FORECASTERS))
def test_forecaster_fails_closed_on_short_history(spec: str) -> None:
    closes = _prices(120)
    rets = np.diff(closes) / closes[:-1]
    fn, need = FORECASTERS[spec]
    if rets.size >= need:
        pytest.skip("window long enough for spec")
    with pytest.raises(ValueError):
        fn(rets[: max(need - 5, 2)], TAUS)


def test_quantile_panel_causality() -> None:
    """Corrupting the price tail must not change earlier panel rows."""
    closes = _prices(900, seed=3)
    corrupted = closes.copy()
    corrupted[-200:] = closes[-200:] * 1.5 + 3.0
    p1, _ = compute_quantile_panel(closes, "empirical", TAUS, window=300)
    p2, _ = compute_quantile_panel(corrupted, "empirical", TAUS, window=300)
    # Row i uses closes <= i: rows through index n-201 must be identical.
    np.testing.assert_array_equal(p1[: 900 - 201], p2[: 900 - 201])
    assert np.isnan(p1[-1]).all()  # last bar has no next bar


def test_quantile_panel_determinism() -> None:
    closes = _prices(700, seed=5)
    p1, s1 = compute_quantile_panel(closes, "evt", TAUS, window=250)
    p2, s2 = compute_quantile_panel(closes, "evt", TAUS, window=250)
    np.testing.assert_array_equal(p1, p2)
    assert s1 == s2


def test_quantile_panel_warmup_is_nan() -> None:
    closes = _prices(300, seed=9)
    panel, stats = compute_quantile_panel(closes, "empirical", TAUS, window=100)
    assert stats["warmup"] >= 59  # needs >=60 returns
    assert np.isnan(panel[:60]).all()
    assert stats["emitted"] > 0


# --------------------------------------------------------------------------
# Policy map
# --------------------------------------------------------------------------


def _q(mu: float, disp: float, taus: np.ndarray = TAUS) -> np.ndarray:
    from scipy import stats as st

    return mu + disp * st.norm.ppf(taus)


def test_policy_long_flat_never_shorts() -> None:
    pol = QuantilePolicy(mode="long_flat", kappa=1.0, name_cap=0.5, cost_gate=0.0)
    out = weights_from_quantiles(
        {"A": _q(-0.05, 0.02), "B": _q(0.05, 0.02)}, TAUS, pol
    )
    assert "A" not in out or out["A"] == 0.0
    assert out.get("B", 0.0) > 0.0
    assert all(w >= 0.0 for w in out.values())


def test_policy_symmetric_allows_short() -> None:
    pol = QuantilePolicy(mode="symmetric", kappa=1.0, name_cap=0.5, cost_gate=0.0)
    out = weights_from_quantiles({"A": _q(-0.05, 0.02)}, TAUS, pol)
    assert out["A"] < 0.0


def test_policy_cost_gate_blocks_small_mu() -> None:
    pol = QuantilePolicy(mode="long_flat", kappa=1.0, cost_gate=0.01)
    out = weights_from_quantiles({"A": _q(0.005, 0.02)}, TAUS, pol)
    assert out.get("A", 0.0) == 0.0


def test_policy_caps_and_gross_norm() -> None:
    pol = QuantilePolicy(mode="symmetric", kappa=10.0, name_cap=0.25, gross_target=0.5)
    out = weights_from_quantiles(
        {"A": _q(0.9, 0.01), "B": _q(0.8, 0.01), "C": _q(0.7, 0.01)}, TAUS, pol
    )
    assert all(abs(w) <= 0.25 + 1e-12 for w in out.values())
    assert sum(abs(w) for w in out.values()) <= 0.5 + 1e-9


def test_policy_deadband_carries_prior_target() -> None:
    """A tiny move must re-emit the prior target, not omit (omit = flatten)."""
    pol = QuantilePolicy(mode="long_flat", kappa=1.0, name_cap=0.5, deadband=0.05)
    q_strong = {"A": _q(0.02, 0.05)}
    first = weights_from_quantiles(q_strong, TAUS, pol)
    prev = first["A"]
    assert prev > 0
    # Slightly smaller mu -> raw target ~4% lower -> inside deadband -> hold.
    second = weights_from_quantiles({"A": _q(0.0196, 0.05)}, TAUS, pol, first)
    assert second["A"] == pytest.approx(prev)


def test_policy_zero_target_emits_exit() -> None:
    pol = QuantilePolicy(mode="long_flat", kappa=1.0, name_cap=0.5, deadband=0.005)
    first = weights_from_quantiles({"A": _q(0.02, 0.05)}, TAUS, pol)
    second = weights_from_quantiles({"A": _q(-0.5, 0.05)}, TAUS, pol, first)
    assert second["A"] == 0.0


def test_policy_skips_nan_row() -> None:
    pol = QuantilePolicy()
    nan_row = np.full(TAUS.size, np.nan)
    out = weights_from_quantiles({"A": nan_row, "B": _q(0.05, 0.02)}, TAUS, pol)
    assert "A" not in out
    assert out["B"] > 0.0


def test_quantile_moments_band() -> None:
    pol = QuantilePolicy()
    mu, disp = quantile_moments(_q(0.01, 0.03), TAUS, pol)
    assert mu == pytest.approx(0.01, abs=1e-9)
    assert disp == pytest.approx(0.03, rel=1e-6)


def test_policy_edge_gate_normalizes_by_disp() -> None:
    """gate_on='edge' thresholds |mu/disp|: same mu, wide disp must not trade."""
    pol = QuantilePolicy(mode="symmetric", kappa=1.0, cost_gate=0.5, gate_on="edge")
    narrow = weights_from_quantiles({"A": _q(0.02, 0.02)}, TAUS, pol)  # edge=1.0
    wide = weights_from_quantiles({"A": _q(0.02, 0.20)}, TAUS, pol)  # edge=0.1
    assert narrow["A"] > 0.0
    assert wide.get("A", 0.0) == 0.0


def test_policy_edge_gate_mu_gate_equivalence_boundary() -> None:
    """At disp=1 the edge gate equals the mu gate."""
    pol = QuantilePolicy(mode="symmetric", kappa=1.0, cost_gate=0.5, gate_on="edge")
    out = weights_from_quantiles({"A": _q(0.6, 1.0)}, TAUS, pol)
    assert out["A"] > 0.0
    out2 = weights_from_quantiles({"A": _q(0.4, 1.0)}, TAUS, pol)
    assert out2.get("A", 0.0) == 0.0


def test_policy_gate_on_validation() -> None:
    with pytest.raises(ValueError, match="gate_on"):
        QuantilePolicy(gate_on="bogus")


def test_policy_risk_sizing_scales_inverse_disp() -> None:
    """sizing='risk': w = kappa*edge/disp — quiet regimes size up, loud size down."""
    pol = QuantilePolicy(
        mode="symmetric", kappa=0.015, cost_gate=0.0, sizing="risk",
        name_cap=10.0, gross_target=10.0,
    )
    quiet = weights_from_quantiles({"A": _q(0.01, 0.01)}, TAUS, pol)  # edge=1, disp=0.01
    loud = weights_from_quantiles({"A": _q(0.01, 0.10)}, TAUS, pol)   # edge=0.1, disp=0.10
    assert quiet["A"] == pytest.approx(0.015 * 1.0 / 0.01)
    assert loud["A"] == pytest.approx(0.015 * 0.1 / 0.10)
    assert quiet["A"] > loud["A"]


def test_policy_sizing_validation() -> None:
    with pytest.raises(ValueError, match="sizing"):
        QuantilePolicy(sizing="bogus")


def test_policy_persist_bars_requires_consecutive_passes() -> None:
    """persist_bars=2: first qualifying bar does not enter; second does; a
    failing bar resets the streak."""
    pol = QuantilePolicy(
        mode="long_flat", kappa=1.0, name_cap=0.5, cost_gate=0.5,
        gate_on="edge", deadband=0.0, persist_bars=2,
    )
    streaks: dict[str, int] = {}
    first = weights_from_quantiles({"A": _q(0.02, 0.02)}, TAUS, pol, {}, streaks)
    assert first.get("A", 0.0) == 0.0  # streak=1 < 2
    second = weights_from_quantiles({"A": _q(0.02, 0.02)}, TAUS, pol, first, streaks)
    assert second["A"] > 0.0  # streak=2 -> confirmed
    # Held name keeps trading while passing (prior != 0 confirms instantly)
    third = weights_from_quantiles({"A": _q(0.02, 0.02)}, TAUS, pol, second, streaks)
    assert third["A"] > 0.0
    # Failing bar resets the streak -> next pass is streak=1 again
    flat = weights_from_quantiles({"A": _q(0.001, 0.02)}, TAUS, pol, third, streaks)
    assert flat.get("A", 0.0) == 0.0
    re1 = weights_from_quantiles({"A": _q(0.02, 0.02)}, TAUS, pol, flat, streaks)
    assert re1.get("A", 0.0) == 0.0  # re-entry blocked until streak rebuilt


def test_policy_book_vol_target_scales_both_ways() -> None:
    """bvt as a target: quiet book scales up (bounded by gross), loud book down."""
    pol = QuantilePolicy(
        mode="symmetric", kappa=0.001, cost_gate=0.0, sizing="risk",
        name_cap=10.0, gross_target=10.0, book_vol_target=0.05,
    )
    quiet = weights_from_quantiles({"A": _q(0.5, 0.01)}, TAUS, pol)  # raw book_vol tiny
    loud = weights_from_quantiles({"A": _q(0.5, 1.0)}, TAUS, pol)
    # book_vol = |w|*disp -> scale to 0.05 in both directions
    assert quiet["A"] * 0.01 == pytest.approx(0.05, rel=1e-6)
    assert loud["A"] * 1.0 == pytest.approx(0.05, rel=1e-6)


def test_mkt_disp_cut_flattens_book() -> None:
    """Median cross-asset disp above the cut emits explicit zeros (flat), and
    clears persistence streaks so re-entry must re-confirm."""
    times = np.array([T0 + timedelta(hours=4 * i) for i in range(4)])
    storm = np.vstack([_q(0.02, 0.02)] * 2 + [_q(0.02, 0.20), _q(0.02, 0.02)])
    storm = np.vstack([_q(0.02, 0.02)] * 2 + [_q(0.02, 0.20), _q(0.02, 0.02)])
    pol = QuantilePolicy(
        mode="long_flat", kappa=1.0, name_cap=0.5, cost_gate=0.0,
        deadband=0.0, mkt_disp_cut=0.10,
    )
    w = quantile_panels_to_weights(
        {"A": storm, "B": storm}, {"A": times, "B": times}, pol, TAUS
    )
    day2 = w.filter(pl.col("event_time") == times[2])
    # Median disp on day2 = 0.20 > cut -> explicit flat for held names
    assert day2.height == 2
    assert (day2["target_weight"] == 0.0).all()
    day3 = w.filter(pl.col("event_time") == times[3])
    assert day3.height == 2 and (day3["target_weight"] > 0.0).all()
    # sanity: without the cut, day2 trades
    pol2 = QuantilePolicy(
        mode="long_flat", kappa=1.0, name_cap=0.5, cost_gate=0.0, deadband=0.0
    )
    w2 = quantile_panels_to_weights(
        {"A": storm, "B": storm}, {"A": times, "B": times}, pol2, TAUS
    )
    assert (w2.filter(pl.col("event_time") == times[2])["target_weight"] > 0.0).all()


# --------------------------------------------------------------------------
# Panel -> weights
# --------------------------------------------------------------------------


def _bars_panel(sids: list[str], n: int, seed: int = 13) -> pl.DataFrame:
    rows = []
    for k, sid in enumerate(sids):
        closes = _prices(n, seed=seed + k, drift=0.0008)
        for i in range(n):
            c = float(closes[i])
            rows.append(
                {
                    "security_id": sid,
                    "event_time": T0 + timedelta(hours=4 * i),
                    "open": c * 0.999,
                    "high": c * 1.003,
                    "low": c * 0.997,
                    "close": c,
                    "volume": 1e6,
                    "source": "binance",
                }
            )
    return pl.DataFrame(rows)


def test_weight_panel_schema_and_hold_semantics() -> None:
    closes = _prices(400, seed=17)
    times = np.array([T0 + timedelta(hours=4 * i) for i in range(400)])
    panel, _ = compute_quantile_panel(closes, "empirical", TAUS, window=150)
    pol = QuantilePolicy(mode="long_flat", kappa=1.0, cost_gate=0.0, deadband=0.0)
    w = quantile_panels_to_weights({"A": panel}, {"A": times}, pol, TAUS)
    assert {"event_time", "security_id", "target_weight"} <= set(w.columns)
    dup = w.group_by(["event_time", "security_id"]).len().filter(pl.col("len") > 1)
    assert dup.height == 0
    assert (w["target_weight"].abs() <= 0.25 + 1e-9).all()


def test_weight_panel_disjoint_calendars() -> None:
    closes_a = _prices(300, seed=21)
    closes_b = _prices(300, seed=22)
    times_a = np.array([T0 + timedelta(hours=4 * i) for i in range(300)])
    times_b = np.array([T0 + timedelta(hours=2) + timedelta(hours=4 * i) for i in range(300)])
    pa, _ = compute_quantile_panel(closes_a, "empirical", TAUS, window=120)
    pb, _ = compute_quantile_panel(closes_b, "empirical", TAUS, window=120)
    pol = QuantilePolicy(mode="long_flat", kappa=1.0, cost_gate=0.0)
    w = quantile_panels_to_weights(
        {"A": pa, "B": pb}, {"A": times_a, "B": times_b}, pol, TAUS
    )
    assert w.height > 0
    dup = w.group_by(["event_time", "security_id"]).len().filter(pl.col("len") > 1)
    assert dup.height == 0


# --------------------------------------------------------------------------
# End-to-end: synthetic bars -> run_sim_live -> receipt
# --------------------------------------------------------------------------


def test_sim_live_end_to_end_synthetic(tmp_path: Path) -> None:
    from quant_fund.paper.sim_live import StrategySlot, run_sim_live

    bars_root = tmp_path / "bars"
    bars_root.mkdir()
    for k, sid in enumerate(["AAA", "BBB"]):
        _bars_panel([sid], 260, seed=31 + k).write_parquet(bars_root / f"{sid.lower()}_1d.parquet")
    cfg = AppConfig.model_validate(
        {
            "data": {"root": str(tmp_path / "data"), "source": "synthetic"},
            "paper": {"ledger_subdir": "sim_live_test", "enable_shadow": False},
            "risk_gate": {"max_name": 0.5, "max_gross": 2.0, "max_net": 1.0},
        }
    )
    pol = QuantilePolicy(mode="long_flat", kappa=1.0, cost_gate=0.0, deadband=0.0)
    res = run_sim_live(
        bars_root=bars_root,
        symbols=["AAA", "BBB"],
        interval="1d",
        config=cfg,
        champion=StrategySlot(name="empirical_long_flat", spec="empirical", policy=pol),
        challengers=[
            StrategySlot(name="ewma_emp_long_flat", spec="ewma_emp", policy=pol)
        ],
        window=120,
        out_dir=tmp_path / "out",
        run_id="test-sim-live",
    )
    receipt = res.receipt
    assert receipt["live_pnl_claim"] is False
    assert receipt["simulated_only"] is True
    assert receipt["strategy"]["champion"]["spec"] == "empirical"
    assert set(receipt["bars"]["sha256"]) == {"AAA", "BBB"}
    assert receipt["loop_metrics"]["n_steps_this_run"] > 0
    assert res.receipt_path.is_file()
    # Quantile cache written and is deterministic on re-read.
    cache_files = list((tmp_path / "out" / "qpanel_cache").glob("*.npz"))
    assert len(cache_files) == 4  # 2 sids x 2 specs
    # Bench pass: both slots get book stats on identical fill semantics.
    assert set(receipt["book_stats"]) == {"empirical_long_flat", "ewma_emp_long_flat"}


def test_sim_live_bench_only_skips_paper_loop(tmp_path: Path) -> None:
    from quant_fund.paper.sim_live import StrategySlot, run_sim_live

    bars_root = tmp_path / "bars"
    bars_root.mkdir()
    _bars_panel(["AAA"], 200, seed=41).write_parquet(bars_root / "aaa_1d.parquet")
    cfg = AppConfig.model_validate(
        {
            "data": {"root": str(tmp_path / "data"), "source": "synthetic"},
            "paper": {"ledger_subdir": "sim_live_test", "enable_shadow": False},
        }
    )
    pol = QuantilePolicy(mode="long_flat", kappa=1.0, cost_gate=0.0, deadband=0.0)
    res = run_sim_live(
        bars_root=bars_root,
        symbols=["AAA"],
        interval="1d",
        config=cfg,
        champion=StrategySlot(name="emp_lf", spec="empirical", policy=pol),
        window=120,
        out_dir=tmp_path / "out",
        run_id="bench-only-test",
        bench_only=True,
    )
    assert res.loop is None
    receipt = res.receipt
    assert receipt["kind"] == "sim_live_bench_receipt"
    assert receipt["bench_only"] is True
    assert receipt["live_pnl_claim"] is False
    assert receipt["loop_metrics"]["status"] == "bench_only_no_paper_loop"
    assert receipt["book_stats"]["emp_lf"]["status"] == "ok"


def test_sim_live_resume_continues_without_duplicate_fills(tmp_path: Path) -> None:
    """A mid-calendar stop must resume from broker_state, not restart."""
    from quant_fund.paper.sim_live import StrategySlot, run_sim_live

    bars_root = tmp_path / "bars"
    bars_root.mkdir()
    _bars_panel(["AAA"], 260, seed=43).write_parquet(bars_root / "aaa_1d.parquet")
    cfg = AppConfig.model_validate(
        {
            "data": {"root": str(tmp_path / "data"), "source": "synthetic"},
            "paper": {"ledger_subdir": "sim_live_test", "enable_shadow": False},
        }
    )
    pol = QuantilePolicy(mode="long_flat", kappa=1.0, cost_gate=0.0, deadband=0.0)
    common = dict(
        bars_root=bars_root,
        symbols=["AAA"],
        interval="1d",
        config=cfg,
        champion=StrategySlot(name="emp_lf", spec="empirical", policy=pol),
        window=120,
        out_dir=tmp_path / "out",
        run_id="resume-e2e",
        bench=False,
    )
    first = run_sim_live(**common, max_steps=80, prefer_latest=False)
    m1 = first.receipt["loop_metrics"]
    assert m1["n_steps_this_run"] == 80
    nav1 = first.receipt["champion_equity_stats"]["nav_end"]
    fills1 = int(m1["n_fills"] or 0)

    second = run_sim_live(**common, max_steps=80, prefer_latest=False, resume=True)
    m2 = second.receipt["loop_metrics"]
    assert m2["resumed"] is True
    assert m2["n_steps_this_run"] == 80
    assert m2["n_steps"] == 160
    # The durable equity file is append-safe across resume: 160 rows total,
    # NAV continuous at the seam (row 79 = end of the first run's state).
    eq = pl.read_parquet(Path(second.receipt["paths"]["equity"]))
    assert eq.height == 160
    assert eq["nav"][79] == pytest.approx(nav1, rel=1e-9)
    # Fills are cumulative across the resume — no duplicated orders.
    orders_path = Path(second.receipt["paths"].get("orders") or "nonexistent")
    if orders_path.is_file():
        orders = pl.read_parquet(orders_path)
        assert orders.height == fills1 + int(m2["n_fills"] or 0)


def test_vincent_blend_averages_member_panels(tmp_path: Path) -> None:
    """vincent(a+b) emits the element-wise mean of member quantile rows."""
    from quant_fund.paper.sim_live import StrategySlot, run_sim_live

    bars_root = tmp_path / "bars"
    bars_root.mkdir()
    _bars_panel(["AAA"], 200, seed=47).write_parquet(bars_root / "aaa_1d.parquet")
    cfg = AppConfig.model_validate(
        {
            "data": {"root": str(tmp_path / "data"), "source": "synthetic"},
            "paper": {"ledger_subdir": "sim_live_test", "enable_shadow": False},
        }
    )
    pol = QuantilePolicy(mode="long_flat", kappa=1.0, cost_gate=0.0, deadband=0.0)
    res = run_sim_live(
        bars_root=bars_root,
        symbols=["AAA"],
        interval="1d",
        config=cfg,
        champion=StrategySlot(name="vin", spec="vincent(empirical+ewma_emp)", policy=pol),
        window=120,
        out_dir=tmp_path / "out",
        run_id="vincent-test",
        bench_only=True,
    )
    receipt = res.receipt
    assert receipt["strategy"]["champion"]["spec"] == "vincent(empirical+ewma_emp)"
    # Both member panels got cached under their own specs.
    cache = tmp_path / "out" / "qpanel_cache"
    names = [p.name for p in cache.glob("*.npz")]
    assert any(n.startswith("qpanel_AAA_empirical_") for n in names)
    assert any(n.startswith("qpanel_AAA_ewma_emp_") for n in names)
    assert receipt["book_stats"]["vin"]["status"] == "ok"


def test_eval_tail_clips_bars_not_panels(tmp_path: Path) -> None:
    """--eval-tail: panels from full history, book sees only the tail window."""
    from quant_fund.paper.sim_live import StrategySlot, run_sim_live

    bars_root = tmp_path / "bars"
    bars_root.mkdir()
    _bars_panel(["AAA"], 200, seed=53).write_parquet(bars_root / "aaa_1d.parquet")
    cfg = AppConfig.model_validate(
        {
            "data": {"root": str(tmp_path / "data"), "source": "synthetic"},
            "paper": {"ledger_subdir": "sim_live_test", "enable_shadow": False},
        }
    )
    pol = QuantilePolicy(mode="long_flat", kappa=1.0, cost_gate=0.0, deadband=0.0)
    res = run_sim_live(
        bars_root=bars_root,
        symbols=["AAA"],
        interval="1d",
        config=cfg,
        champion=StrategySlot(name="emp_lf", spec="empirical", policy=pol),
        window=120,
        out_dir=tmp_path / "out",
        run_id="eval-tail-test",
        bench_only=True,
        eval_tail_bars=60,
    )
    receipt = res.receipt
    assert receipt["bars"]["eval_tail_bars"] == 60
    st = receipt["book_stats"]["emp_lf"]
    assert st["status"] == "ok"
    assert st["n_marks"] <= 60


def test_evt_and_ewma_emp_quantiles_match_lane_math() -> None:
    closes = _prices(900, seed=29)
    rets = np.diff(closes) / closes[:-1]
    qe = evt_quantiles(rets[-750:], TAUS)
    qw = ewma_emp_quantiles(rets[-750:], TAUS)
    assert np.isfinite(qe).all() and np.isfinite(qw).all()
    # Both should bracket the plain empirical 5%/95% in the same ballpark.
    qe_emp = np.quantile(rets[-750:], [0.05, 0.95])
    assert qe[0] < qe_emp[1] and qe[-1] > qe_emp[0]
