"""Operational scaled wrappee: Student-t coverage, CRC uses vol, bench keys."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from quant_fund.config.models import AppConfig
from quant_fund.models.crc import ConformalRiskControl
from quant_fund.models.distribution import ScaledStudentTDistribution
from quant_fund.research.benches import bench_conformal, bench_crc


def _hetero_panel(
    n_days: int = 48,
    n_names: int = 8,
    seed: int = 3,
    *,
    fat: bool = False,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime(2019, 1, 2, 16, 0, 0)
    times = [start + timedelta(days=i) for i in range(n_days)]
    rows: list[dict[str, object]] = []
    for t in times:
        for j in range(n_names):
            vol = 0.008 + 0.06 * (j / max(n_names - 1, 1))
            shock = float(rng.standard_t(5)) if fat else float(rng.normal())
            ret = float(vol * shock)
            rows.append(
                {
                    "event_time": t,
                    "security_id": f"S{j:02d}",
                    "ret_1": ret,
                    "vol_20": vol,
                    "mom_20": ret,
                    "future_log_return_1": float(
                        vol * (rng.standard_t(5) if fat else rng.normal())
                    ),
                    "future_log_return_5": float(vol * np.sqrt(5.0) * rng.normal()),
                    "future_max_drawdown_5": float(-abs(vol * (0.5 + abs(rng.normal())))),
                }
            )
    return pl.DataFrame(rows)


def test_scaled_student_t_covers_heteroskedastic_90pct() -> None:
    rng = np.random.default_rng(4)
    n = 800
    scale = rng.choice(np.array([0.5, 2.0]), size=n)
    y = rng.normal(0.0, scale)
    tr, te = slice(0, 400), slice(400, None)
    m = ScaledStudentTDistribution([0.05, 0.95]).fit(y[tr], scale[tr])
    q = m.predict(scale[te])
    cov = float(np.mean((y[te] >= q[:, 0]) & (y[te] <= q[:, 1])))
    assert cov >= 0.85
    assert 3.0 <= m.nu <= 30.0


def test_crc_with_scale_high_vol_bounds_wider() -> None:
    frame = _hetero_panel()
    out = bench_crc(frame, AppConfig())
    assert out
    high = out.get("high_vol_mean_bound")
    low = out.get("low_vol_mean_bound")
    assert high is not None and low is not None
    assert float(high) > float(low)
    y = np.array([r["future_log_return_1"] for r in frame.iter_rows(named=True)], dtype=float)
    scale = np.array([r["vol_20"] for r in frame.iter_rows(named=True)], dtype=float)
    n = y.size
    tr, cal, te = slice(0, n // 2), slice(n // 2, int(0.7 * n)), slice(int(0.7 * n), n)
    wrap = ScaledStudentTDistribution([0.05, 0.95]).fit(y[tr], scale[tr])
    q_cal = wrap.predict(scale[cal])
    q_te = wrap.predict(scale[te])
    crc = ConformalRiskControl(0.05).calibrate(-y[cal], -q_cal[:, 0])
    pred = crc.predict_bound(-q_te[:, 0])
    high_m = scale[te] >= float(np.median(scale[te]))
    assert float(np.mean(pred[high_m])) > float(np.mean(pred[~high_m]))


def test_bench_conformal_has_raw_scaled_cqr_raw() -> None:
    frame = _hetero_panel()
    out = bench_conformal(frame, AppConfig())
    assert "gaussian_raw" in out
    assert "scaled" in out
    assert "cqr_raw" in out
    assert out["gaussian_raw"].get("coverage") is not None
    assert out["scaled"].get("coverage") is not None
    assert out["cqr_raw"].get("coverage") is not None
    assert out["cqr_raw"].get("qhat") is not None
    assert "aci_raw" in out
    raw_cov = float(out["gaussian_raw"]["coverage"])
    cqr_raw_cov = float(out["cqr_raw"]["coverage"])
    assert cqr_raw_cov + 1e-9 >= raw_cov
    assert out.get("wrappee") in {"scaled_gaussian", "scaled_student_t"}
