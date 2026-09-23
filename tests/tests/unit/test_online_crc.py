import numpy as np
import pytest

from quant_fund.models.online_crc import OnlineCRC, bench_online_crc


def test_miss_streak_expands_bound_and_raises_lambda() -> None:
    oc = OnlineCRC(alpha=0.05, gamma=0.05, B=1.0)
    oc.initialize(np.array([0.01, 0.02]), np.array([1.0, 1.0]))
    start_lambda = oc.lambda_t
    first = float(np.ravel(oc.update(10.0, 0.01))[0])
    for _ in range(7):
        oc.update(10.0, 0.01)
    later = float(np.ravel(oc.update(10.0, 0.01))[0])
    assert oc.lambda_t > start_lambda
    assert later > first


def test_long_calibrated_path_mean_risk_near_alpha() -> None:
    row = bench_online_crc(alpha=0.05, gamma=0.05, seed=12, n_cal=200, n_test=800)
    assert abs(row["mean_risk"] - row["nominal"]) <= 0.04


def test_monotone_in_alpha() -> None:
    rng = np.random.default_rng(3)
    n_cal, n_test = 150, 400
    losses = rng.exponential(scale=0.04, size=n_cal + n_test)
    base = np.full_like(losses, 0.01)
    dates = np.arange(n_cal + n_test)
    loose = OnlineCRC(alpha=0.20, gamma=0.05)
    tight = OnlineCRC(alpha=0.05, gamma=0.05)
    loose.initialize(losses[:n_cal], base[:n_cal])
    tight.initialize(losses[:n_cal], base[:n_cal])
    p_lo = loose.run(losses[n_cal:], base[n_cal:], dates[n_cal:])
    p_hi = tight.run(losses[n_cal:], base[n_cal:], dates[n_cal:])
    assert tight.lambda_t >= loose.lambda_t
    assert float(np.nanmean(p_hi.bounds)) >= float(np.nanmean(p_lo.bounds)) - 1e-12
    assert float(np.nanmean(p_hi.hit)) <= float(np.nanmean(p_lo.hit)) + 1e-12


def test_run_updates_once_per_date_never_stacks_names() -> None:
    losses = np.array([1.0, 1.0, 1.0])
    base = np.zeros(3)
    dates = np.array(["2020-01-02", "2020-01-02", "2020-01-02"])
    batched = OnlineCRC(alpha=0.05, gamma=0.05)
    batched.run(losses, base, dates)
    stacked = OnlineCRC(alpha=0.05, gamma=0.05)
    for loss, bound in zip(losses, base, strict=True):
        stacked.update(loss, bound)
    assert batched.lambda_t == pytest.approx(0.05 * (1.0 - 0.05))
    assert stacked.lambda_t > batched.lambda_t


def test_bench_online_crc_keys_no_sharpe() -> None:
    row = bench_online_crc(seed=12)
    assert {"mean_risk", "nominal", "n"} <= set(row)
    assert row.get("dgp") == "fixture"
    assert "sharpe" not in {k.lower() for k in row}
    assert row["nominal"] == 0.05
    assert row["n"] == 800.0
    assert np.isfinite(row["mean_risk"])


def test_online_crc_length_mismatch_raises() -> None:
    oc = OnlineCRC(alpha=0.05, gamma=0.05)
    # size-1 base broadcasts; true length mismatch must raise
    with pytest.raises(ValueError):
        oc.update(np.array([1.0, 2.0]), np.array([0.1, 0.2, 0.3]))
    with pytest.raises(ValueError):
        oc.run(np.array([1.0, 2.0]), np.array([0.1, 0.1]), dates=[0])


def test_online_crc_rejects_nonpositive_gamma() -> None:
    with pytest.raises(ValueError):
        OnlineCRC(alpha=0.05, gamma=-0.1)
    with pytest.raises(ValueError):
        OnlineCRC(alpha=0.05, gamma=float("nan"))
