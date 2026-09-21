import numpy as np
import polars as pl

from quant_fund.config.models import AppConfig, HorizonConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.data.corporate_actions import adjust_prices
from quant_fund.features.engine import build_features
from quant_fund.labels.engine import build_labels
from quant_fund.metrics.scoring import pearson_ic


def test_features_not_use_future_returns() -> None:
    p = SyntheticMarketProvider(n_assets=6, n_days=90, seed=4)
    bars = adjust_prices(p.get_bars(), p.get_corporate_actions())
    master = p.get_security_master()
    bars = bars.join(
        master.select(["security_id", "sector", "industry", "exchange"]),
        on="security_id",
        how="left",
    )
    cfg = AppConfig()
    feats = build_features(bars, cfg)
    assert "max_source_available_time" in feats.columns
    # available_time should never exceed event_time for bars
    bad = feats.filter(pl.col("max_source_available_time") > pl.col("decision_time"))
    assert bad.height == 0


def test_public_feature_zoo_is_pit_and_finite() -> None:
    """Wave 147: every public column exists, is finite where present, and has no oracle."""
    from quant_fund.features.engine import CROSS_SECTIONAL_COLUMNS
    from quant_fund.models.ranking import (
        NEUTRAL_FILL_FEATURES,
        ORACLE_COLUMNS,
        PUBLIC_FEATURES,
        PUBLIC_FEATURES_CORE,
        PUBLIC_FEATURES_LONG,
    )

    p = SyntheticMarketProvider(n_assets=12, n_days=140, seed=11)
    bars = adjust_prices(p.get_bars(), p.get_corporate_actions())
    master = p.get_security_master()
    bars = bars.join(
        master.select(["security_id", "sector", "industry", "exchange"]),
        on="security_id",
        how="left",
    )
    feats = build_features(bars, AppConfig())
    assert len(PUBLIC_FEATURES) >= 25
    assert set(PUBLIC_FEATURES) == set(PUBLIC_FEATURES_CORE) | set(PUBLIC_FEATURES_LONG)
    assert set(NEUTRAL_FILL_FEATURES) == set(PUBLIC_FEATURES_LONG)
    assert not set(PUBLIC_FEATURES) & set(ORACLE_COLUMNS)
    assert "planted_signal" in CROSS_SECTIONAL_COLUMNS
    missing = [c for c in PUBLIC_FEATURES if c not in feats.columns]
    assert missing == []
    # Core columns need <= 60 sessions: the tail of a 140-day panel is populated.
    tail = feats.sort("event_time").filter(
        pl.col("event_time") >= feats["event_time"].unique().sort()[-20]
    )
    for col in PUBLIC_FEATURES_CORE:
        values = tail[col].drop_nulls().to_numpy().astype(float)
        assert values.size > 0, col
        assert np.isfinite(values).all(), col
    # Long-lookback columns are honestly null on a short panel, never NaN.
    for col in PUBLIC_FEATURES_LONG:
        values = feats[col].drop_nulls().to_numpy().astype(float)
        assert np.isfinite(values).all(), col
    # Market-relative columns exist and use only trailing bars.
    assert {"beta_60", "idio_vol_60", "idio_mom_20", "mom_skip_5_20"} <= set(feats.columns)
    assert feats.filter(pl.col("max_source_available_time") > pl.col("decision_time")).height == 0


def test_design_frame_neutral_fills_long_lookback_only() -> None:
    from quant_fund.pipeline.dataset import design_frame

    frame = pl.DataFrame(
        {
            "event_time": [1, 2, 3, 4],
            "security_id": ["A", "B", "A", "B"],
            "future_idio_return_5": [0.1, None, 0.2, 0.3],
            "cs_z_mom_20": [0.5, 0.4, None, 0.1],
            "cs_z_mom_12_1": [None, 0.2, 0.3, None],
        }
    )
    out = design_frame(frame, "future_idio_return_5", ["cs_z_mom_20", "cs_z_mom_12_1"])
    # Row 2 has a null label, row 3 has a null core feature: both dropped.
    assert out["security_id"].to_list() == ["A", "B"]
    assert out["cs_z_mom_12_1"].to_list() == [0.0, 0.0]


def test_future_max_drawdown_is_path_drawdown_not_origin_to_minimum() -> None:
    from datetime import datetime

    times = pl.datetime_range(
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 4),
        interval="1d",
        eager=True,
    )
    bars = pl.DataFrame(
        {
            "security_id": ["A"] * 4 + ["SEC_MKT"] * 4,
            "event_time": list(times) * 2,
            "close_total_return": [100.0, 110.0, 99.0, 120.0] * 2,
        }
    )
    cfg = AppConfig(horizons=HorizonConfig(bars=[3], names=["3d"]))

    labels = build_labels(bars, cfg).filter(pl.col("security_id") == "A").sort("event_time")

    # 100 -> 110 -> 99 is a 10% peak-to-trough drawdown, not 1% from origin.
    assert np.isclose(labels["future_max_drawdown_3"][0], 99.0 / 110.0 - 1.0)


def test_forward_realized_volatility_is_defined_for_one_bar() -> None:
    from datetime import datetime

    times = pl.datetime_range(
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 3),
        interval="1d",
        eager=True,
    )
    bars = pl.DataFrame(
        {
            "security_id": ["A"] * 3 + ["SEC_MKT"] * 3,
            "event_time": list(times) * 2,
            "close_total_return": [100.0, 110.0, 121.0] * 2,
        }
    )
    cfg = AppConfig(horizons=HorizonConfig(bars=[1], names=["1d"]))

    labels = build_labels(bars, cfg).filter(pl.col("security_id") == "A").sort("event_time")

    assert np.isclose(labels["future_realized_vol_1"][0], np.log(1.1))
    assert np.isclose(labels["future_realized_var_1"][0], np.log(1.1) ** 2)


def test_benchmark_forward_is_aligned_on_benchmark_calendar() -> None:
    from datetime import datetime

    times = pl.datetime_range(
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 4),
        interval="1d",
        eager=True,
    )
    bars = pl.DataFrame(
        {
            "security_id": ["A"] * 3 + ["SEC_MKT"] * 4,
            "event_time": [times[0], times[2], times[3], *times],
            "close_total_return": [100.0, 99.0, 120.0, 100.0, 110.0, 121.0, 133.1],
        }
    )
    cfg = AppConfig(horizons=HorizonConfig(bars=[1], names=["1d"]))

    labels = build_labels(bars, cfg).filter(pl.col("security_id") == "A").sort("event_time")

    # A has no row on day 2.  The benchmark forward from day 1 must still be
    # 110/100 - 1, rather than shifting over A's sparse joined rows to day 3.
    assert np.isclose(
        labels["future_excess_return_1"][0],
        (99.0 / 100.0 - 1.0) - (110.0 / 100.0 - 1.0),
    )


def test_labels_are_forward() -> None:
    p = SyntheticMarketProvider(n_assets=5, n_days=80, seed=5)
    bars = adjust_prices(p.get_bars(), p.get_corporate_actions())
    labs = build_labels(bars, AppConfig())
    # last 5 rows per id should be null for 5-day label
    sid = labs["security_id"][1]
    sub = labs.filter(pl.col("security_id") == sid).sort("event_time")
    assert (
        sub["future_return_5"][-1] is None
        or (isinstance(sub["future_return_5"][-1], float) and np.isnan(sub["future_return_5"][-1]))
        or sub["future_return_5"][-1] is None
    )


def test_planted_signal_has_positive_ic() -> None:
    """SYNTHETIC: planted_signal at t-1 is the labeled oracle for residual return at t."""
    p = SyntheticMarketProvider(n_assets=10, n_days=200, seed=7)
    bars = p.get_bars()
    # alpha_t = oracle_beta * planted_signal_{t-1}; market row is excluded from the residual plant.
    df = (
        bars.sort(["security_id", "event_time"])
        .with_columns(
            (pl.col("close") / pl.col("close").shift(1).over("security_id") - 1).alias("ret"),
            pl.col("planted_signal").shift(1).over("security_id").alias("sig_lag"),
        )
        .drop_nulls()
    )
    ic = pearson_ic(df["sig_lag"].to_numpy(), df["ret"].to_numpy())
    # qualitative recovery, labeled synthetic (pooled, includes the market row)
    assert ic > 0.05
