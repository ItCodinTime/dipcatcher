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
