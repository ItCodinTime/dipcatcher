# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Hugging Face OHLCV-1m
#
# Data label: network when `HF_OHLCV_1M_ALLOW_DOWNLOAD=1`; otherwise the checked-in fixture.
#
# Not investment advice. No live-trading claim.
# `mito0o852/OHLCV-1m` declares no license. This repository does not vendor that
# dataset and does not grant redistribution rights. Adjustment is undeclared
# (`HF_OHLCV_1M_UNDECLARED_ADJ`). The offline fixture is a few synthetic vendor-schema
# rows for the adapter, not a slice of the Hub corpus. A network run downloads one
# monthly file (the adapter's granule) into a temporary cache and does not write it
# back into the repository.

# %%
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import polars as pl

from quant_fund.data.adapters.hf_ohlcv_1m import (
    REVISION_ID,
    SOURCE_NAME,
    HfOhlcv1mProvider,
    minute_gap_report,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples" / "fixtures" / "hf_ohlcv_1m" / "ohlcv_sample.parquet"
FIXTURE_REVISION = "example-fixture"


def _network_enabled() -> bool:
    return os.environ.get("HF_OHLCV_1M_ALLOW_DOWNLOAD", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _refuse_download(url: str, dest: Path) -> None:
    raise RuntimeError(f"download is disabled for the fixture path: {url} -> {dest}")


def _print_common(bars: pl.DataFrame, *, label: str) -> None:
    quality = minute_gap_report(bars)
    sessions = sorted(str(value) for value in bars["session"].unique().to_list())
    missing = int(quality["n_rth_missing"].sum()) if quality.height else 0
    print(f"data_label={label}")
    print("claim=research_only")
    print("not_investment_advice=true")
    print("no_live_trading_claim=true")
    print(f"source={SOURCE_NAME}")
    print(f"revision_id={REVISION_ID}")
    print("license=undeclared")
    print("adjustment=undeclared")
    print("vendored_dataset=false")
    print("redistribution=refused")
    print(f"n_bars={bars.height}")
    print(f"symbols={','.join(sorted(str(value) for value in bars['symbol'].unique().to_list()))}")
    print(f"sessions={','.join(sessions)}")
    print(f"n_rth_missing={missing}")
    if set(bars["source"].to_list()) != {SOURCE_NAME}:
        raise SystemExit("bars were not stamped hf_ohlcv_1m")
    if set(bars["revision_id"].to_list()) != {REVISION_ID}:
        raise SystemExit("bars were not stamped with the undeclared-adjustment revision")


def _run_fixture() -> None:
    if not FIXTURE.is_file():
        raise SystemExit(f"checked-in fixture is absent: {FIXTURE}")
    with tempfile.TemporaryDirectory(prefix="dipcatcher-hf-fixture-") as raw_cache:
        cache = Path(raw_cache)
        destination = cache / FIXTURE_REVISION / "ohlcv_1992-01.parquet"
        destination.parent.mkdir(parents=True)
        shutil.copyfile(FIXTURE, destination)
        provider = HfOhlcv1mProvider(
            cache,
            symbols="AAPL,MSFT",
            start="1992-01-02",
            end="1992-01-02",
            revision=FIXTURE_REVISION,
            allow_download=False,
            fetcher=_refuse_download,
        )
        bars = provider.get_bars()
        _print_common(bars, label="fixture")
        print("allow_download=false")
        print(f"fixture_bytes={FIXTURE.stat().st_size}")


def _run_network() -> None:
    with tempfile.TemporaryDirectory(prefix="dipcatcher-hf-network-") as raw_cache:
        provider = HfOhlcv1mProvider(
            Path(raw_cache),
            symbols="AAPL",
            start="1992-01-02",
            end="1992-01-02",
            allow_download=True,
            max_months=1,
        )
        bars = provider.get_bars()
        _print_common(bars, label="network")
        print("allow_download=true")
        print("download_cached_in_repo=false")
        print("download_granule=one_monthly_parquet")


def main() -> None:
    if _network_enabled():
        _run_network()
    else:
        _run_fixture()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"example_failed={type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
