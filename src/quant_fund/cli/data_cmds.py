"""Ingest, collect, and gold-build commands.

Split out of the original module. Import the parent path; it re-exports these names.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from .app import app
from .support import _cfg, _collect_param_value


@app.command()
def doctor(config: Path = typer.Option(Path("configs/research.yaml"))) -> None:
    from quant_fund.pipeline.doctor import doctor as run_doctor

    info = run_doctor(str(config))
    for k, v in info.items():
        typer.echo(f"{k}: {v}")
    # Research-only Kyle/OFI dump path (CLI hint; not a live gate).
    typer.echo(
        "kyle_ofi: dipcatcher kyle-ofi --dump-lambda-series PATH "
        "(research_only λ date-series parquet; no Sharpe/pnl)"
    )
    typer.echo(
        "verify-research: runs northset_session_means_honesty_errors "
        "(session soft-verify dispatcher; research_only, no Sharpe)"
    )
    typer.echo(
        "ls: dipcatcher ls hunt|book|race|confirm "
        "(frozen Lightspeed engines; research_only, no live broker)"
    )
    cfg = _cfg(config)
    ns = cfg.northset
    typer.echo(
        f"northset.shape_floors: depth_shape_finite_floor={ns.depth_shape_finite_floor} "
        f"concentration_top_finite_floor={ns.concentration_top_finite_floor} "
        f"queue_priority_finite_floor={ns.queue_priority_finite_floor} "
        f"side_notional_finite_floor={ns.side_notional_finite_floor} "
        f"tob_size_share_finite_floor={ns.tob_size_share_finite_floor} "
        f"(shape_columns_ensured is a receipt stamp when synth ensure runs)"
    )
    required = [
        info.get(f"dir_{part}") == "ok" for part in ("raw", "bronze", "silver", "gold", "metadata")
    ]
    healthy = (
        all(required)
        and info.get("data_manifest") == "ok"
        and info.get("research_receipt") == "ok"
        and info.get("core_imports") == "ok"
        and info.get("live") != "REJECTED"
    )
    if not healthy:
        raise typer.Exit(code=1)


def get_logger(**binds: Any) -> Any:
    """Resolve the structured logger on first call.

    Kept as a module attribute so tests can patch it, without importing
    structlog (and the utils barrel) when the CLI is only showing help.
    """
    from quant_fund.utils.logging import get_logger as _get_logger

    return _get_logger(**binds)


@app.command()
def ingest(config: Path = typer.Option(Path("configs/research.yaml"))) -> None:
    from quant_fund.data.ingest import ingest as do_ingest

    cfg = _cfg(config)
    paths = do_ingest(cfg)
    log = get_logger(cmd="ingest", source=cfg.data.source)
    log.info("ingested", paths={k: str(v) for k, v in paths.items()})
    if cfg.data.source == "synthetic":
        typer.echo("SYNTHETIC ingest complete")
    for k, v in paths.items():
        typer.echo(f"{k}: {v}")


@app.command()
def collect(
    config: Path = typer.Option(Path("configs/research.yaml")),
    source: str = typer.Option(..., "--source", help="Registered public source name"),
    param: list[str] = typer.Option(
        [], "--param", help="Adapter fetch kwarg as key=value (repeatable)"
    ),
    filename: str | None = typer.Option(
        None, help="Output filename below raw/sources/ (default <source>.parquet)"
    ),
) -> None:
    """Explicit opt-in public-source collection (may use the network).

    Writes the normalized PIT frame to ``<data.root>/raw/sources/<source>.parquet``
    plus a JSON receipt with sha256, row counts, and provenance. This is
    collection only — the offline ingest pipeline is unchanged.
    """
    from quant_fund.data.collector import collect_source

    cfg = _cfg(config)
    fetch_kwargs: dict[str, object] = {}
    for item in param:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise typer.BadParameter(f"--param must be key=value, got {item!r}")
        fetch_kwargs[key.strip()] = _collect_param_value(value)
    if "path" not in fetch_kwargs and cfg.data.source_path is not None:
        fetch_kwargs["path"] = cfg.data.source_path
    try:
        result = collect_source(source, cfg.data.root, fetch_kwargs=fetch_kwargs, filename=filename)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"source={source} rows={result.frame.height}")
    typer.echo(f"data: {result.data}")
    typer.echo(f"receipt: {result.receipt}")


@app.command("build-features")
def build_features_cmd(config: Path = typer.Option(Path("configs/research.yaml"))) -> None:
    from quant_fund.pipeline.dataset import build_gold

    cfg = _cfg(config)
    feats, _ = build_gold(cfg)
    typer.echo(f"features rows={feats.height} cols={len(feats.columns)}")


@app.command("build-labels")
def build_labels_cmd(config: Path = typer.Option(Path("configs/research.yaml"))) -> None:
    from quant_fund.pipeline.dataset import build_gold

    cfg = _cfg(config)
    _, labs = build_gold(cfg)
    typer.echo(f"labels rows={labs.height}")


@app.command("membership-coverage")
def membership_coverage_cmd(
    membership: Path = typer.Option(
        ...,
        "--membership",
        help="Point-in-time index membership JSON (current + newest-first changes)",
    ),
    bars: list[Path] = typer.Option(
        [],
        "--bars",
        help="Parquet bar panel (repeatable). Unioned before coverage.",
    ),
    anchor: list[str] = typer.Option(
        [],
        "--anchor",
        help="Coverage anchor YYYY-MM-DD (repeatable). Default: 2016-01-04,2020-01-02,2023-01-03",
    ),
    out: Path | None = typer.Option(None, "--out", help="Write JSON coverage report"),
    label: str = typer.Option(
        "bars",
        "--label",
        help="Short label for the source set (echoed in the report)",
    ),
) -> None:
    """Report point-in-time index-membership price coverage.

    For each anchor date, finds the first panel session on or after that day
    and prints the fraction of index members with at least one real bar.
    Research diagnostic only — not a live-trading claim.
    """
    import json

    import polars as pl

    from quant_fund.data.index_membership import (
        coverage_report_dict,
        load_membership,
        membership_price_coverage,
        merge_bar_panels,
    )
    from quant_fund.proofcore.contracts import sha256_hex_bytes

    if not membership.is_file():
        raise typer.BadParameter(f"membership file not found: {membership}")
    if not bars:
        raise typer.BadParameter("pass at least one --bars parquet")
    anchors = anchor or ["2016-01-04", "2020-01-02", "2023-01-03"]
    frames: list[pl.DataFrame] = []
    for path in bars:
        if not path.is_file():
            raise typer.BadParameter(f"bars parquet not found: {path}")
        frames.append(pl.read_parquet(path))
    panel = merge_bar_panels(frames)
    member_payload = load_membership(membership)
    member_hash = sha256_hex_bytes(membership.read_bytes())
    rows = membership_price_coverage(panel, member_payload, anchors)
    report = coverage_report_dict(
        rows,
        sources=[label],
        membership_path=str(membership),
        membership_sha256=member_hash,
        extra={"n_bar_rows": panel.height, "bar_paths": [str(path) for path in bars]},
    )
    typer.echo(
        f"membership_coverage label={label} members_file={membership.name} "
        f"sha256={member_hash[:12]}… n_bars={panel.height}"
    )
    typer.echo("| anchor | session | members | with_bar | coverage | n_missing |")
    typer.echo("|---|---|---:|---:|---:|---:|")
    for row in rows:
        session = row["session"] or "—"
        typer.echo(
            f"| {row['anchor']} | {session} | {row['members']} | {row['with_bar']} | "
            f"{float(row['coverage']):.6f} | {row['n_missing']} |"
        )
    typer.echo(f"mean_coverage={float(report['mean_coverage']):.6f}")
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        typer.echo(f"wrote {out}")


__all__ = [
    "build_features_cmd",
    "build_labels_cmd",
    "collect",
    "doctor",
    "ingest",
    "membership_coverage_cmd",
]
