"""Ingest provider output into the bronze/silver lake."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.parquet import ParquetMarketProvider
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.data.corporate_actions import adjust_prices
from quant_fund.data.lake import Lake


def make_provider(config: AppConfig) -> SyntheticMarketProvider | ParquetMarketProvider:
    if config.data.source == "synthetic":
        return SyntheticMarketProvider(
            n_assets=config.data.synthetic_n_assets,
            n_days=config.data.synthetic_n_days,
            seed=config.data.synthetic_seed,
            oracle_beta=config.data.synthetic_oracle_beta,
            oracle_phi=config.data.synthetic_oracle_phi,
        )
    root = config.data.parquet_path or (Path(config.data.root) / "raw")
    return ParquetMarketProvider(Path(root))


def ingest(config: AppConfig) -> dict[str, Path]:
    lake = Lake(Path(config.data.root))
    provider = make_provider(config)
    bars = provider.get_bars()
    actions = provider.get_corporate_actions()
    master = provider.get_security_master()
    paths = {
        "bars": lake.write_parquet(bars, "bronze/bars.parquet"),
        "actions": lake.write_parquet(actions, "bronze/corporate_actions.parquet"),
        "master": lake.write_parquet(master, "bronze/security_master.parquet"),
    }
    silver = (
        adjust_prices(bars, actions)
        if not actions.is_empty()
        else adjust_prices(bars, pl.DataFrame())
    )
    if not master.is_empty() and "sector" in master.columns:
        silver = silver.join(
            master.select(["security_id", "sector", "industry", "exchange"]),
            on="security_id",
            how="left",
        )
    paths["silver"] = lake.write_parquet(silver, "silver/bars.parquet")
    return paths
