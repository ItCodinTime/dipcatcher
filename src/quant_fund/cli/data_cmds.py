"""Ingest, collect, and gold-build commands.

Split out of the original module. Import the parent path; it re-exports these names.
"""

from __future__ import annotations

from pathlib import Path

import typer

from quant_fund.pipeline.doctor import doctor as run_doctor
from quant_fund.utils.logging import get_logger

from .app import app
from .support import _cfg, _collect_param_value


@app.command()
def doctor(config: Path = typer.Option(Path("configs/research.yaml"))) -> None:
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


__all__ = [
    "build_features_cmd",
    "build_labels_cmd",
    "collect",
    "doctor",
    "ingest",
]
