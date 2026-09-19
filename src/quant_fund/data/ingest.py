"""Ingest provider output into the bronze/silver lake."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.parquet import ParquetMarketProvider
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.data.corporate_actions import adjust_prices, apply_listing_actions
from quant_fund.data.lake import Lake
from quant_fund.data.security_master import attach_master_attributes
from quant_fund.data.universe import build_membership_panel


def make_provider(config: AppConfig) -> SyntheticMarketProvider | ParquetMarketProvider:
    """Route config.data.source to a market provider (fail-closed).

    Allowed: ``synthetic`` → SyntheticMarketProvider;
    ``file`` / ``parquet`` → ParquetMarketProvider.
    Unknown sources raise ValueError (defense in depth beyond DataConfig).
    """
    source = str(config.data.source).strip().lower()
    if source == "synthetic":
        return SyntheticMarketProvider(
            n_assets=config.data.synthetic_n_assets,
            n_days=config.data.synthetic_n_days,
            seed=config.data.synthetic_seed,
            oracle_beta=config.data.synthetic_oracle_beta,
            oracle_phi=config.data.synthetic_oracle_phi,
        )
    if source in {"file", "parquet"}:
        root = config.data.parquet_path or (Path(config.data.root) / "raw")
        return ParquetMarketProvider(Path(root))
    raise ValueError(
        f"unknown data source {config.data.source!r}; expected one of: synthetic, file, parquet"
    )


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
    silver = apply_listing_actions(
        silver, actions, include_delisted=config.universe.include_delisted
    )
    if not master.is_empty() and "sector" in master.columns:
        silver = attach_master_attributes(silver, master)
    timestamps = (
        silver.get_column("event_time").unique().sort().to_list() if not silver.is_empty() else []
    )
    universe = (
        build_membership_panel(silver, master, timestamps, config.universe, actions=actions)
        if timestamps
        else pl.DataFrame()
    )
    paths["silver"] = lake.write_parquet(silver, "silver/bars.parquet")
    paths["universe"] = lake.write_parquet(universe, "silver/universe.parquet")
    frames = {
        "bars": bars,
        "actions": actions,
        "master": master,
        "silver": silver,
        "universe": universe,
    }
    manifest = {
        "schema_version": 1,
        "source": str(config.data.source),
        "artifacts": {
            name: {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "rows": frame.height,
                "columns": sorted(frame.columns),
            }
            for name, path in paths.items()
            if name in frames
            for frame in [frames[name]]
        },
    }
    manifest_path = Path(config.data.root) / "metadata" / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    paths["manifest"] = manifest_path
    return paths
