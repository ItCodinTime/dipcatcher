"""Predeclared basis-reversion pair candidate on Binance spot and perpetual bars.

Long spot/short perp at the following open when the completed-close perp
premium exceeds its own 30-day history by one standard deviation. Exit when
the z-score drops below 0.5 or the premium disappears. Sparse target events
avoid daily rebalance churn. Research replay only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from quant_fund.backtest.carry_engine import run_carry_backtest  # noqa: E402
from quant_fund.config.loader import load_config  # noqa: E402


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def basis_pair_events(perp: pl.DataFrame, spot: pl.DataFrame) -> pl.DataFrame:
    """Build causal enter/exit requests, including explicit zero exits."""
    keys = ["security_id", "event_time"]
    panel = (
        perp.select(keys + ["close", "available_time"])
        .rename({"close": "perp_close", "available_time": "perp_known"})
        .join(
            spot.select(keys + ["close", "available_time"]).rename(
                {"close": "spot_close", "available_time": "spot_known"}
            ),
            on=keys,
            how="inner",
        )
        .sort(keys)
        .with_columns((pl.col("perp_close") / pl.col("spot_close") - 1.0).alias("basis"))
        .with_columns(
            pl.col("basis").rolling_mean(30, min_samples=30).over("security_id").alias("mu"),
            pl.col("basis").rolling_std(30, min_samples=30).over("security_id").alias("sd"),
        )
        .with_columns(((pl.col("basis") - pl.col("mu")) / pl.col("sd")).alias("z"))
    )
    dates = sorted(panel["event_time"].unique().to_list())
    next_time = dict(zip(dates, dates[1:], strict=False))
    by_date: dict[datetime, list[dict]] = {}
    for row in panel.iter_rows(named=True):
        by_date.setdefault(row["event_time"], []).append(row)
    active: dict[str, float] = {}
    out: list[dict] = []
    for date in dates[:-1]:
        visible = {}
        for row in by_date.get(date, []):
            if row["perp_known"] < next_time[date] and row["spot_known"] < next_time[date]:
                visible[str(row["security_id"])] = row
        for sid in sorted(active):
            row = visible.get(sid)
            z = float(row["z"]) if row is not None and row["z"] is not None else np.nan
            basis = float(row["basis"]) if row is not None else np.nan
            if not np.isfinite(z) or z < 0.5 or basis <= 0:
                out.append({"event_time": date, "security_id": sid, "target_weight": 0.0})
                active.pop(sid)
            else:
                drift = float(row["spot_close"]) / active[sid]
                if drift > 1.5 or drift < 1.0 / 1.5:
                    out.append({"event_time": date, "security_id": sid, "target_weight": 0.05})
                    active[sid] = float(row["spot_close"])
        candidates = sorted(
            (
                (sid, row)
                for sid, row in visible.items()
                if sid not in active
                and row["z"] is not None
                and np.isfinite(float(row["z"]))
                and float(row["z"]) >= 1.0
                and float(row["basis"]) > 0
            ),
            key=lambda item: float(item[1]["z"]),
            reverse=True,
        )
        for sid, row in candidates:
            if len(active) >= 10:
                break
            active[sid] = float(row["spot_close"])
            out.append({"event_time": date, "security_id": sid, "target_weight": 0.05})
    if not out:
        return pl.DataFrame(
            schema={
                "event_time": pl.Datetime(time_zone="UTC"),
                "security_id": pl.String,
                "target_weight": pl.Float64,
            }
        )
    return pl.DataFrame(out).sort(keys)


def _summary(metrics: dict) -> dict:
    out = {
        k: metrics.get(k)
        for k in (
            "total_return",
            "cagr",
            "sharpe",
            "max_drawdown",
            "n",
            "periods_per_year",
            "funding_net",
            "mean_turnover",
            "liquidation_count",
            "risk_gate_rejects",
            "commission",
            "spread",
            "impact",
            "ruined",
        )
    }
    attribution = metrics.get("pnl_attribution", {})
    out["pnl_attribution_totals"] = attribution.get("totals")
    out["pnl_attribution_conservation_error"] = attribution.get("conservation_error")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=_REPO / "configs/research.yaml")
    args = parser.parse_args()
    sources = args.data_root / "raw" / "sources"
    manifest_path = args.data_root / "collection_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    symbols = sorted(
        {p.name.split("_")[0] for p in sources.glob("*_1d.perp.parquet")}
        & {p.name.split("_")[0] for p in sources.glob("*_1d.spot.parquet")}
        & {p.name.split(".")[0] for p in sources.glob("*.funding.parquet")}
    )
    if len(symbols) < 8:
        raise ValueError("need at least eight paired assets with funding")
    files = [
        sources / f"{sid}_1d.{venue}.parquet" for sid in symbols for venue in ("perp", "spot")
    ] + [sources / f"{sid}.funding.parquet" for sid in symbols]
    hashes = {p.name: _sha256(p) for p in files}
    for name, digest in hashes.items():
        if manifest["files"][name]["sha256"] != digest:
            raise ValueError(f"manifest hash mismatch: {name}")
    perp = pl.concat([pl.read_parquet(p) for p in files if ".perp." in p.name])
    spot = pl.concat([pl.read_parquet(p) for p in files if ".spot." in p.name])
    funding = pl.concat([pl.read_parquet(p) for p in files if ".funding." in p.name])
    if set(perp["source"].unique().to_list()) != {"binance_usdtm_perp"}:
        raise ValueError("unexpected perp source")
    if set(spot["source"].unique().to_list()) != {"binance_public_data"}:
        raise ValueError("unexpected spot source")
    if set(funding["source"].unique().to_list()) != {"binance_funding_rate"}:
        raise ValueError("unexpected funding source")
    weights = basis_pair_events(perp, spot)
    dates = sorted(perp["event_time"].unique().to_list())
    cut = dates[int(len(dates) * 0.8)]
    cfg = load_config(args.config)
    if cfg.costs.frictionless:
        raise ValueError("frictionless cost model is inadmissible")
    cfg.risk_gate.max_gross = 1.0
    cfg.risk_gate.max_net = 1.0
    cfg.risk_gate.max_name = 0.05
    development = run_carry_backtest(
        perp.filter(pl.col("event_time") < cut),
        spot.filter(pl.col("event_time") < cut),
        funding.filter(pl.col("event_time") < cut),
        weights.filter(pl.col("event_time") < cut),
        cfg,
    )
    dev = _summary(development.metrics)
    eligible = bool(
        np.isfinite(dev["sharpe"])
        and dev["sharpe"] > 0
        and dev["cagr"] > 0
        and dev["max_drawdown"] > -0.05
    )
    receipt = {
        "schema": "basis_pair_candidate.v1",
        "created_at": datetime.now(tz=UTC).isoformat(),
        "evidence_level": "historical_development_only",
        "research_only": True,
        "live_pnl_claim": False,
        "source": "binance_spot_perp_funding",
        "survivorship": "currently listed Binance symbols; historical delistings absent",
        "n_assets": len(symbols),
        "development_cut": str(cut),
        "n_target_events": weights.height,
        "development": dev,
        "development_eligible": eligible,
        "input_hashes": hashes,
        "config_sha256": _sha256(args.config),
        "script_sha256": _sha256(Path(__file__)),
        "metrics_sha256": _sha256(_REPO / "src/quant_fund/metrics/returns.py"),
        "engine_sha256": _sha256(_REPO / "src/quant_fund/backtest/carry_engine.py"),
    }
    if eligible:
        full = run_carry_backtest(perp, spot, funding, weights, cfg)
        eq = full.equity.sort("event_time").with_columns(pl.col("nav").pct_change().alias("ret"))
        tail = eq.filter(pl.col("event_time") >= cut)["ret"].drop_nulls().to_numpy()
        from quant_fund.metrics.returns import cagr, max_drawdown, sharpe_ratio

        ppy = float(full.metrics["periods_per_year"])
        receipt["historical_tail_exploratory"] = {
            "net_return": float(np.prod(1 + tail) - 1),
            "cagr": cagr(tail, periods_per_year=ppy),
            "sharpe": sharpe_ratio(tail, periods_per_year=ppy)["sharpe"],
            "max_drawdown": max_drawdown(tail),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, default=str))
    print(
        json.dumps(
            {
                k: receipt.get(k)
                for k in (
                    "n_assets",
                    "n_target_events",
                    "development",
                    "development_eligible",
                    "historical_tail_exploratory",
                )
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
